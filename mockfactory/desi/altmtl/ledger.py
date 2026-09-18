"""
Initialization of the alternative MTL ledgers.

One realization lives in ``<altmtl_dir>/Univ{realization:03d}``, and holds the healpix merged
target list ledgers that the loop updates as it replays the survey. Ledgers are built once
from the mock target catalog, then copied into each realization with a fresh draw of
``SUBPRIORITY``: two realizations differ only in which target wins a fiber when several
compete for it, which is exactly the source of randomness the bitweights average over.
"""

import os
import shutil
import logging

import numpy as np

from . import utils
from .compat import patch_write_mtl
from .tiletracker import make_tile_tracker


logger = logging.getLogger('altmtl.ledger')


#: The healpix nside at which merged target list ledgers are written.
MTL_NSIDE = 32


def get_ledger_dir(base_dir, survey='main', obscon='dark'):
    """Return the directory holding the healpix ledgers, under ``base_dir``."""
    from desitarget import io
    from desitarget.mtl import get_mtl_ledger_format
    return io.find_target_files(base_dir, flavor='mtl', resolve=True, survey=survey.lower(),
                                obscon=obscon.lower(), ender=get_mtl_ledger_format())


def get_healpixels(ledger_dir, obscon='dark'):
    """Return the sorted healpixels for which a ledger exists in ``ledger_dir``."""
    import glob
    fns = glob.glob(os.path.join(ledger_dir, 'mtl-{}-hp-*.ecsv'.format(obscon.lower())))
    return np.sort([int(os.path.basename(fn).split('hp-')[-1].split('.ecsv')[0]) for fn in fns])


def make_initial_ledgers(targets_fn, output_dir, survey='main', obscon='dark', numproc=1, overwrite=False):
    """
    Turn a mock target catalog into merged target list ledgers.

    This is the expensive step of the whole pipeline: it reads the full target catalog and
    writes one ledger per healpix. It is done once, outside of any realization.

    Parameters
    ----------
    targets_fn : str
        Path of the mock target catalog, in the format expected by the real survey merged
        target list, as produced by :mod:`mockfactory.desi.altmtl.targets`.

    output_dir : str
        Directory the initial ledgers are written to, under ``<survey>/<obscon>``.

    survey : str, default='main'
        Survey the ledgers belong to.

    obscon : str, default='dark'
        Observing conditions, 'dark' or 'bright'.

    numproc : int, default=1
        Number of processes to build the ledgers with.

    overwrite : bool, default=False
        If ``False`` and ledgers already exist in ``output_dir``, keep them.

    Returns
    -------
    healpixels : array
        Healpixels for which a ledger was written.
    """
    from desitarget import mtl

    ledger_dir = get_ledger_dir(output_dir, survey=survey, obscon=obscon)
    if os.path.isdir(ledger_dir) and len(get_healpixels(ledger_dir, obscon=obscon)):
        if not overwrite:
            healpixels = get_healpixels(ledger_dir, obscon=obscon)
            logger.info('Initial ledgers already in {}, keeping {:d} healpixels.'.format(ledger_dir, healpixels.size))
            return healpixels
        shutil.rmtree(ledger_dir)

    utils.mkdir(output_dir)
    logger.info('Building initial ledgers from {} with {:d} process(es).'.format(targets_fn, numproc))
    with patch_write_mtl():
        mtl.make_ledger(targets_fn, output_dir, obscon=obscon.upper(), numproc=numproc)

    healpixels = get_healpixels(ledger_dir, obscon=obscon)
    if not healpixels.size:
        raise ValueError('no ledger written to {}; is {} a valid target catalog?'.format(ledger_dir, targets_fn))
    logger.info('Wrote {:d} healpix ledgers to {}.'.format(healpixels.size, ledger_dir))

    fn = os.path.join(output_dir, 'hpxlist_{}.txt'.format(obscon.lower()))
    with open(fn, 'w') as file:
        file.write(','.join(map(str, healpixels)))
    logger.info('Wrote healpix list {}.'.format(fn))
    return healpixels


def _shuffle_subpriority(fn, seed):
    """Draw fresh subpriorities in the ledger ``fn``, in place."""
    from astropy.table import Table

    ledger = Table.read(fn, format='ascii.ecsv')
    rng = np.random.RandomState(seed=seed)
    # A freshly built ledger holds only the initial entry of each target, so every row is
    # redrawn. Were it to hold later entries too, only the initial timestamp should be touched.
    ledger['SUBPRIORITY'] = rng.uniform(size=len(ledger))
    ledger.write(fn, format='ascii.ecsv', overwrite=True)
    return len(ledger)


def initialize_realization(initial_dir, altmtl_dir, realization=0, survey='main', obscon='dark',
                           seed=None, shuffle_subpriority=True, start_date=None, end_date=None,
                           tiles_specstatus_fn=None, overwrite=False, **kwargs):
    """
    Set up one alternative realization: ledgers, tile tracker and specstatus.

    Parameters
    ----------
    initial_dir : str
        Directory holding the initial ledgers, as built by :func:`make_initial_ledgers`.

    altmtl_dir : str
        Directory of the realization, e.g. ``.../altmtl0/Univ000``.

    realization : int, default=0
        Index of the realization. It enters the subpriority seed, so that realizations of the
        same mock differ, while a given realization is reproducible.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions, 'dark' or 'bright'.

    seed : int, default=None
        Base seed for the subpriority draw. The seed of one ledger is
        ``seed + healpix + realization``, so that ledgers can be built independently.

    shuffle_subpriority : bool, default=True
        Whether to draw fresh subpriorities. Pass ``False`` to keep those of the mock, which
        only makes sense for a single realization: bitweights over realizations that share
        their subpriorities would all be identical.

    start_date : int, str, default=None
        Night before which actions are flagged as already done.

    end_date : int, str, default=None
        Night after which actions are dropped. Required.

    tiles_specstatus_fn : str, default=None
        Path of the tiles-specstatus file. Defaults to the surveyops copy.

    overwrite : bool, default=False
        Whether to rebuild ledgers and tile tracker that already exist.

    kwargs : dict
        Other arguments for :func:`mockfactory.desi.altmtl.tiletracker.make_tile_tracker`.

    Returns
    -------
    altmtl_dir : str
        Directory of the realization.
    """
    from astropy.table import Table

    if seed is None: seed = 314159
    if tiles_specstatus_fn is None: tiles_specstatus_fn = utils.TILES_SPECSTATUS_FN
    if 'trunk' in altmtl_dir.lower() or 'ops' in altmtl_dir.lower():
        raise ValueError('refusing to write alternative ledgers to {}: the path looks like the real '
                         'surveyops ledgers'.format(altmtl_dir))

    initial_ledger_dir = get_ledger_dir(initial_dir, survey=survey, obscon=obscon)
    healpixels = get_healpixels(initial_ledger_dir, obscon=obscon)
    if not healpixels.size:
        raise ValueError('no initial ledger found in {}'.format(initial_ledger_dir))

    ledger_dir = get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon)
    if os.path.isdir(ledger_dir) and len(get_healpixels(ledger_dir, obscon=obscon)) and not overwrite:
        logger.info('Ledgers already in {}, not rebuilding.'.format(ledger_dir))
    else:
        utils.mkdir(ledger_dir)
        nrows = 0
        for healpix in healpixels:
            basename = 'mtl-{}-hp-{:d}.ecsv'.format(obscon.lower(), healpix)
            fn = os.path.join(ledger_dir, basename)
            shutil.copyfile(os.path.join(initial_ledger_dir, basename), fn)
            if shuffle_subpriority:
                nrows += _shuffle_subpriority(fn, seed + int(healpix) + realization)
        logger.info('Realization {:d}: copied {:d} ledgers to {}{}.'.format(
            realization, healpixels.size, ledger_dir,
            ', reshuffling {:d} subpriorities'.format(nrows) if shuffle_subpriority else ''))

    # The loop does not read this copy, but downstream tools expect to find the observing
    # history next to the ledgers it produced.
    ztile_fn = os.path.join(altmtl_dir, os.path.basename(tiles_specstatus_fn))
    if not os.path.isfile(ztile_fn) or overwrite:
        specstatus = Table.read(tiles_specstatus_fn)
        lastnight = specstatus['LASTNIGHT'].astype(int)
        mask = np.ones(len(specstatus), dtype='?')
        if start_date is not None: mask &= lastnight >= utils.iso_to_night(start_date)
        if end_date is not None: mask &= lastnight <= utils.iso_to_night(end_date)
        specstatus[mask].write(ztile_fn, format='ascii.ecsv', overwrite=True)
        logger.info('Wrote {} ({:d} tiles).'.format(ztile_fn, mask.sum()))

    make_tile_tracker(altmtl_dir, survey=survey, obscon=obscon, start_date=start_date, end_date=end_date,
                      tiles_specstatus_fn=tiles_specstatus_fn, overwrite=overwrite,
                      meta={'seed': seed, 'realization': realization,
                            'shuffleSubpriorities': bool(shuffle_subpriority)}, **kwargs)
    return altmtl_dir

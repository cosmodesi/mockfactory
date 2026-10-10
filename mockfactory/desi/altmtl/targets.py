"""
Target catalogs for the alternative MTL loop.

The loop starts from a catalog in the format the real survey's merged target list is built
from: positions, target bits, and the initial priority and number of observations those bits
imply. This module turns mockfactory catalogs, one per tracer, into that format.

Priorities and numbers of observations are not written down here: they are read off the target
mask through :func:`desitarget.targets.initial_priority_numobs`, so a mock stays consistent
with whatever the survey currently declares.
"""

from pathlib import Path
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.targets')


#: Group a target catalog is written under, and the name its FITS predecessor used.
TARGETS_GROUP = 'TARGETS'

#: Columns a target catalog must carry for the ledgers to be built from it, for the main survey.
#: An SV catalog carries the same, with the four bitmask columns prefixed; see
#: :func:`get_target_columns`.
TARGET_COLUMNS = ('RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET',
                  'SUBPRIORITY', 'OBSCONDITIONS', 'PRIORITY_INIT', 'PRIORITY', 'NUMOBS_INIT',
                  'NUMOBS_MORE', 'ZWARN')

#: Bitmask columns, named as in the main survey.
_BIT_COLUMNS = ('DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET')


def get_bit_columns(survey='main'):
    """
    Return the names of the ``DESI_TARGET``, ``BGS_TARGET``, ``MWS_TARGET`` and ``SCND_TARGET``
    columns in ``survey``.

    desitarget tells the survey a catalog belongs to from these names alone, 'SV3_DESI_TARGET'
    meaning SV3, and then applies that survey's masks and priorities. So the names are all it
    takes for the ledgers to follow the SV3 rules.
    """
    survey = survey.lower()
    if survey == 'main':
        return _BIT_COLUMNS
    if not survey.startswith('sv'):
        raise ValueError('survey must be main or svX, got {}'.format(survey))
    return tuple('{}_{}'.format(survey.upper(), name) for name in _BIT_COLUMNS)


def get_target_columns(survey='main'):
    """Return the columns a target catalog of ``survey`` must carry, as :attr:`TARGET_COLUMNS`."""
    rename = dict(zip(_BIT_COLUMNS, get_bit_columns(survey)))
    return tuple(rename.get(name, name) for name in TARGET_COLUMNS)


def get_survey(columns):
    """Return the survey a target catalog with these ``columns`` belongs to, as desitarget does."""
    for name in columns:
        if name.endswith('_DESI_TARGET') and name.startswith('SV'):
            return name.split('_')[0].lower()
    return 'main'


def get_masks(survey='main'):
    """Return the ``desi_mask`` and ``bgs_mask`` of ``survey``."""
    survey = survey.lower()
    if survey == 'main':
        from desitarget.targetmask import desi_mask, bgs_mask
        return desi_mask, bgs_mask
    from importlib import import_module
    # The yaml keys are prefixed, sv3_desi_mask, but the module exports them unprefixed.
    module = import_module('desitarget.{0}.{0}_targetmask'.format(survey))
    return module.desi_mask, module.bgs_mask


def get_target_bits(tracer, survey='main'):
    """
    Return the ``DESI_TARGET``, ``BGS_TARGET`` and ``MWS_TARGET`` bits of ``tracer``.

    A bright galaxy carries the ``BGS_ANY`` bit in ``DESI_TARGET`` as well as its own bit in
    ``BGS_TARGET``, which is what sets its priority.

    Parameters
    ----------
    tracer : str, list
        Tracer name, e.g. 'LRG', 'QSO', 'ELG_LOP', 'BGS_BRIGHT', or several of them, whose bits
        are then combined, as they are for a target selected by more than one cut.

    survey : str, default='main'
        Survey whose masks the bits are taken from, 'main' or 'sv3'. SV3 has its own, with
        the same names at different bits, and an 'ELG_HIP' the main survey lacks.

    Returns
    -------
    desi_target, bgs_target, mws_target : int
        Target bits.
    """
    desi_mask, bgs_mask = get_masks(survey)

    if isinstance(tracer, str): tracer = [tracer]
    desi_target = bgs_target = mws_target = 0
    for name in tracer:
        name = name.upper()
        # Which mask a name belongs to is asked of desitarget rather than listed here, so
        # that a name like BGS_FAINT_HIP is not looked up in the main mask and raised on.
        # The two masks share no name, so the test is unambiguous.
        if name in bgs_mask.names():
            bgs_target |= bgs_mask[name]
            desi_target |= desi_mask['BGS_ANY']
        else:
            desi_target |= desi_mask[name]
    return desi_target, bgs_target, mws_target


def get_priority_numobs(desi_target, bgs_target=0, mws_target=0, obscon='DARK', survey='main'):
    """
    Return the initial priority and number of observations implied by a set of target bits.

    Returns
    -------
    priority_init, numobs_init : int
        Initial priority and number of observations.
    """
    from desitarget.targets import initial_priority_numobs

    names = get_bit_columns(survey)[:3]
    targets = np.zeros(1, dtype=[(name, 'i8') for name in names])
    for name, value in zip(names, (desi_target, bgs_target, mws_target)):
        targets[name] = value
    priority, numobs = initial_priority_numobs(targets, obscon=obscon.upper())
    return int(priority[0]), int(numobs[0])


def make_targets(catalogs, obscon='dark', seed=None, z=None, columns=(), survey='main', mpicomm=None):
    """
    Turn mockfactory catalogs into one target catalog the ledgers can be built from.

    Parameters
    ----------
    catalogs : dict
        Catalogs to merge, as ``{tracer: catalog}``. Each catalog must carry 'RA' and 'DEC',
        and a redshift column. A tracer key may name several cuts, as ``'ELG|ELG_LOP'``.

    columns : tuple, default=()
        Extra columns to carry over from the input catalogs, beyond the survey's own. The
        imaging quantities the clustering vetoes read -- 'MASKBITS', 'NOBS_G', 'NOBS_R',
        'NOBS_Z' -- have to come through here, since nothing downstream can recover them.

    obscon : str, default='dark'
        Observing conditions, 'dark' or 'bright'. It selects which priorities apply.

    seed : int, default=None
        Seed for the subpriorities, drawn uniformly. Every target needs one, and ties in
        subpriority would be broken arbitrarily by fiberassign.

    z : str, default=None
        Name of the redshift column, written out as 'RSDZ'. Defaults to 'Z' when present.

    survey : str, default='main'
        Survey the targets are selected for, 'main' or 'sv3'. It sets the masks the bits are
        taken from, and the names of the bitmask columns, e.g. 'SV3_DESI_TARGET', which is how
        the ledgers come to follow that survey's priorities.

    mpicomm : MPI communicator, default=None
        Communicator the catalogs are scattered over. Defaults to that of the first catalog.

    Returns
    -------
    targets : Catalog
        Merged target catalog, carrying :func:`get_target_columns` plus 'RSDZ'.
    """
    import mpytools as mpy
    from desitarget.targetmask import obsconditions

    if not catalogs:
        raise ValueError('no catalog to merge')
    if mpicomm is None: mpicomm = list(catalogs.values())[0].mpicomm

    obscondition = obsconditions.mask(obscon.upper())
    merged, offset = [], 0
    for tracer, catalog in catalogs.items():
        names = tracer.split('|')
        desi_target, bgs_target, mws_target = get_target_bits(names, survey=survey)
        priority_init, numobs_init = get_priority_numobs(desi_target, bgs_target, mws_target, obscon=obscon,
                                                         survey=survey)

        zcol = z
        if zcol is None:
            zcol = 'Z' if 'Z' in catalog.columns() else None
        if zcol is None:
            raise ValueError('catalog {} carries no redshift column; pass z=...'.format(tracer))

        # A plain catalog, not the input class: the output carries survey columns, not the box
        # or cutsky geometry the input was built with.
        target = mpy.Catalog(data={}, mpicomm=catalog.mpicomm)
        target['RA'] = np.asarray(catalog['RA'], dtype='f8')
        target['DEC'] = np.asarray(catalog['DEC'], dtype='f8')
        target['RSDZ'] = np.asarray(catalog[zcol], dtype='f8')
        size = target.size
        desi_name, bgs_name, mws_name, scnd_name = get_bit_columns(survey)
        for name, value, dtype in [(desi_name, desi_target, 'i8'), (bgs_name, bgs_target, 'i8'),
                                   (mws_name, mws_target, 'i8'), (scnd_name, 0, 'i8'),
                                   ('PRIORITY_INIT', priority_init, 'i8'), ('PRIORITY', priority_init, 'i8'),
                                   ('NUMOBS_INIT', numobs_init, 'i8'), ('NUMOBS_MORE', numobs_init, 'i8'),
                                   ('OBSCONDITIONS', obscondition, 'i8'), ('ZWARN', 0, 'i8')]:
            target[name] = np.full(size, value, dtype=dtype)
        for name in columns:
            target[name] = catalog[name]
        # Identifiers have to be unique over the whole catalog, so each tracer is offset by the
        # total size of those before it.
        target['TARGETID'] = offset + target.cindex()
        # csize is an allreduce on every access, not a cached number, so every rank takes it
        # and only rank 0 logs it; asking for it inside the branch hangs rank 0 against ranks
        # that have moved on. Same reason as in write_targets below.
        csize = target.csize
        offset += csize
        merged.append(target)
        if mpicomm.rank == 0:
            logger.info('Tracer {}: {:d} targets, desi_target {:d}, priority {:d}, numobs {:d}.'.format(
                tracer, csize, desi_target, priority_init, numobs_init))

    targets = mpy.Catalog.concatenate(merged) if len(merged) > 1 else merged[0]
    rng = mpy.random.MPIRandomState(size=targets.size, seed=seed, mpicomm=mpicomm)
    targets['SUBPRIORITY'] = rng.uniform()
    csize = targets.csize
    if mpicomm.rank == 0:
        logger.info('Merged {:d} targets over {:d} tracer(s).'.format(csize, len(catalogs)))
    return targets


def read_targets(targets_fn, columns=None):
    """
    Read a target catalog written by :func:`write_targets`, as an :class:`astropy.table.Table`.

    Both formats are read, so a catalog produced before the move to HDF5 still works; only
    writing is fixed to one.
    """
    from ..tables import as_table
    if Path(targets_fn).suffix not in ('.h5', '.hdf5'):
        import fitsio
        return as_table(fitsio.read(targets_fn, columns=columns))
    import h5py
    from astropy.table import Table
    with h5py.File(targets_fn, 'r') as file:
        group = file[TARGETS_GROUP]
        names = list(group) if columns is None else list(columns)
        # Viewed as dtype.str, not dtype: h5py hangs metadata such as
        # {'h5py_encoding': 'ascii'} off a string dtype, it rides along through every array
        # built from it, and fitsio cannot hash it when the catalog is eventually written back
        # out.
        arrays = {}
        for name in names:
            array = group[name][:]
            arrays[name] = array.view(np.dtype(array.dtype.str))
    return Table(arrays, copy=False)


def write_targets(targets, output_fn, obscon='dark', mpicomm=None):
    """
    Write a target catalog to ``output_fn``, as the single file the ledgers are built from.

    :func:`desitarget.mtl.make_ledger` reads one file, so the catalog is collected into one
    rather than written per rank. It also identifies a target file by two header keywords,
    which are stamped on afterwards: the table extension must be named 'TARGETS', and
    'OBSCON' must agree with the observing conditions the ledgers are built for.

    Parameters
    ----------
    targets : Catalog
        Target catalog, as built by :func:`make_targets`.

    output_fn : str
        Path to write to.

    obscon : str, default='dark'
        Observing conditions, recorded in the header.

    mpicomm : MPI communicator, default=None
        Communicator the catalog is scattered over.

    Returns
    -------
    output_fn : str
        Path that was written.
    """
    import fitsio

    if mpicomm is None: mpicomm = targets.mpicomm
    # The survey is read off the column names, as desitarget does, rather than passed: a
    # catalog whose names and header disagreed would be built under one survey's rules and
    # assigned under the other's.
    survey = get_survey(targets.columns())
    missing = [name for name in get_target_columns(survey) if name not in targets.columns()]
    if missing:
        raise ValueError('target catalog is missing {}'.format(missing))
    if mpicomm.rank == 0:
        utils.mkdir(Path(output_fn).parent)
    mpicomm.Barrier()
    # HDF5, not FITS: mpytools writes it with every rank putting its own slice into the file,
    # where the FITS writer gathers the whole catalog onto one rank and writes it serially --
    # 24 MB/s for a four gigabyte catalog, whatever the number of ranks.
    targets.write(output_fn, filetype='hdf5', group=TARGETS_GROUP,
                  header={'OBSCON': obscon.upper(), 'SURVEY': survey})
    # csize is a collective, so every rank takes it and only rank 0 logs it; asking for it
    # inside the branch deadlocks rank 0 against the others' barrier below
    csize = targets.csize
    if mpicomm.rank == 0:
        logger.info('Wrote {:d} targets to {}.'.format(csize, output_fn))
    mpicomm.Barrier()
    return output_fn

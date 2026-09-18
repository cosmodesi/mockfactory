"""
Target catalogs for the alternative MTL loop.

The loop starts from a catalog in the format the real survey's merged target list is built
from: positions, target bits, and the initial priority and number of observations those bits
imply. This module turns mockfactory catalogs, one per tracer, into that format.

Priorities and numbers of observations are not written down here: they are read off the target
mask through :func:`desitarget.targets.initial_priority_numobs`, so a mock stays consistent
with whatever the survey currently declares.
"""

import os
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.targets')


#: Columns a target catalog must carry for the ledgers to be built from it.
TARGET_COLUMNS = ('RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET',
                  'SUBPRIORITY', 'OBSCONDITIONS', 'PRIORITY_INIT', 'PRIORITY', 'NUMOBS_INIT',
                  'NUMOBS_MORE', 'ZWARN')

#: Tracers whose bits live in ``BGS_TARGET`` rather than ``DESI_TARGET``.
BGS_TRACERS = ('BGS_BRIGHT', 'BGS_FAINT')


def get_target_bits(tracer):
    """
    Return the ``DESI_TARGET``, ``BGS_TARGET`` and ``MWS_TARGET`` bits of ``tracer``.

    A bright galaxy carries the ``BGS_ANY`` bit in ``DESI_TARGET`` as well as its own bit in
    ``BGS_TARGET``, which is what sets its priority.

    Parameters
    ----------
    tracer : str, list
        Tracer name, e.g. 'LRG', 'QSO', 'ELG_LOP', 'BGS_BRIGHT', or several of them, whose bits
        are then combined, as they are for a target selected by more than one cut.

    Returns
    -------
    desi_target, bgs_target, mws_target : int
        Target bits.
    """
    from desitarget.targetmask import desi_mask, bgs_mask

    if isinstance(tracer, str): tracer = [tracer]
    desi_target = bgs_target = mws_target = 0
    for name in tracer:
        name = name.upper()
        if name in BGS_TRACERS:
            bgs_target |= bgs_mask[name]
            desi_target |= desi_mask['BGS_ANY']
        else:
            desi_target |= desi_mask[name]
    return desi_target, bgs_target, mws_target


def get_priority_numobs(desi_target, bgs_target=0, mws_target=0, obscon='DARK'):
    """
    Return the initial priority and number of observations implied by a set of target bits.

    Returns
    -------
    priority_init, numobs_init : int
        Initial priority and number of observations.
    """
    from desitarget.targets import initial_priority_numobs

    targets = np.zeros(1, dtype=[('DESI_TARGET', 'i8'), ('BGS_TARGET', 'i8'), ('MWS_TARGET', 'i8')])
    targets['DESI_TARGET'], targets['BGS_TARGET'], targets['MWS_TARGET'] = desi_target, bgs_target, mws_target
    priority, numobs = initial_priority_numobs(targets, obscon=obscon.upper())
    return int(priority[0]), int(numobs[0])


def make_targets(catalogs, obscon='dark', seed=None, z=None, mpicomm=None):
    """
    Turn mockfactory catalogs into one target catalog the ledgers can be built from.

    Parameters
    ----------
    catalogs : dict
        Catalogs to merge, as ``{tracer: catalog}``. Each catalog must carry 'RA' and 'DEC',
        and a redshift column. A tracer key may name several cuts, as ``'ELG|ELG_LOP'``.

    obscon : str, default='dark'
        Observing conditions, 'dark' or 'bright'. It selects which priorities apply.

    seed : int, default=None
        Seed for the subpriorities, drawn uniformly. Every target needs one, and ties in
        subpriority would be broken arbitrarily by fiberassign.

    z : str, default=None
        Name of the redshift column, written out as 'RSDZ'. Defaults to 'Z' when present.

    mpicomm : MPI communicator, default=None
        Communicator the catalogs are scattered over. Defaults to that of the first catalog.

    Returns
    -------
    targets : Catalog
        Merged target catalog, carrying :attr:`TARGET_COLUMNS` plus 'RSDZ'.
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
        desi_target, bgs_target, mws_target = get_target_bits(names)
        priority_init, numobs_init = get_priority_numobs(desi_target, bgs_target, mws_target, obscon=obscon)

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
        for name, value, dtype in [('DESI_TARGET', desi_target, 'i8'), ('BGS_TARGET', bgs_target, 'i8'),
                                   ('MWS_TARGET', mws_target, 'i8'), ('SCND_TARGET', 0, 'i8'),
                                   ('PRIORITY_INIT', priority_init, 'i8'), ('PRIORITY', priority_init, 'i8'),
                                   ('NUMOBS_INIT', numobs_init, 'i8'), ('NUMOBS_MORE', numobs_init, 'i8'),
                                   ('OBSCONDITIONS', obscondition, 'i8'), ('ZWARN', 0, 'i8')]:
            target[name] = np.full(size, value, dtype=dtype)
        # Identifiers have to be unique over the whole catalog, so each tracer is offset by the
        # total size of those before it.
        target['TARGETID'] = offset + target.cindex()
        offset += target.csize
        merged.append(target)
        if mpicomm.rank == 0:
            logger.info('Tracer {}: {:d} targets, desi_target {:d}, priority {:d}, numobs {:d}.'.format(
                tracer, target.csize, desi_target, priority_init, numobs_init))

    targets = mpy.Catalog.concatenate(merged) if len(merged) > 1 else merged[0]
    rng = mpy.random.MPIRandomState(size=targets.size, seed=seed, mpicomm=mpicomm)
    targets['SUBPRIORITY'] = rng.uniform()
    if mpicomm.rank == 0:
        logger.info('Merged {:d} targets over {:d} tracer(s).'.format(targets.csize, len(catalogs)))
    return targets


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
    missing = [name for name in TARGET_COLUMNS if name not in targets.columns()]
    if missing:
        raise ValueError('target catalog is missing {}'.format(missing))
    if mpicomm.rank == 0:
        utils.mkdir(os.path.dirname(output_fn))
    mpicomm.Barrier()
    targets.write(output_fn, filetype='fits')
    if mpicomm.rank == 0:
        with fitsio.FITS(output_fn, 'rw') as fits:
            fits[1].write_key('EXTNAME', 'TARGETS')
            fits[1].write_key('OBSCON', obscon.upper())
        logger.info('Wrote {:d} targets to {}.'.format(targets.csize, output_fn))
    mpicomm.Barrier()
    return output_fn

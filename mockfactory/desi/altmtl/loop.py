"""
The alternative MTL loop.

Actions are carried out in the order the real survey did them. An ``fa`` action assigns a tile
against the ledgers as they stand, and records which alternative target landed on each fiber.
The ``update`` action that follows takes the real redshifts of that tile, relabels them with
the alternative target that shares their fiber, and folds them into the alternative ledgers:
the alternative survey then carries on from a state that differs from the real one only
through which target won each fiber.

Realizations are independent, so they are spread over the ranks of an MPI communicator.
"""

import os
import time
import logging
from multiprocessing import get_context

import numpy as np

from . import utils
from .assignment import do_fiber_assignment
from .ledger import get_ledger_dir
from .tiletracker import SUPPORTED_ACTIONTYPES, get_actions, mark_actions_done, read_tile_tracker


logger = logging.getLogger('altmtl.loop')


def _to_big_endian(table):
    """
    Return ``table`` with every field big-endian.

    The ledger update compares against the on-disk data model, which is big-endian as fits
    files are, while a table just read through astropy may not be.
    """
    from astropy.table import Table

    array = table.as_array()
    for name in array.dtype.names:
        byteorder = array.dtype[name].byteorder
        if byteorder == '<' or (byteorder == '=' and np.little_endian):
            return Table(array.byteswap().view(array.dtype.newbyteorder('>')))
    return table.copy()


def update_ledgers(altmtl_dir, action, fiber_map, survey='main', obscon='dark', zcat_dir=None,
                   numobs_from_ledger=True):
    """
    Carry out one ``update`` action: fold the real observations of a tile into the ledgers.

    Parameters
    ----------
    altmtl_dir : str
        Directory of the realization.

    action : astropy.table.Row
        Action to carry out, giving the tile and its archive date.

    fiber_map : FiberMap
        Correspondence between the real and the alternative assignment of this tile.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    zcat_dir : str, default=None
        Directory holding the real redshift catalogs. Defaults to the daily reduction.

    numobs_from_ledger : bool, default=True
        Whether to take the number of observations so far from the ledger rather than from the
        redshift catalog.

    Returns
    -------
    nz : int
        Number of redshifts folded in.
    """
    from astropy.table import Table
    from desitarget.mtl import make_zcat, update_ledger

    if zcat_dir is None: zcat_dir = utils.ZCAT_DIR

    zcat = _to_big_endian(Table(make_zcat(zcat_dir, [action], obscon.upper(), survey)))
    # The real observation of a fiber becomes an observation of whichever target the
    # alternative assignment put on that same fiber.
    alt_zcat = zcat.copy()
    alt_zcat['TARGETID'] = fiber_map.real_to_alt(zcat['TARGETID'])

    update_ledger(get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon), alt_zcat,
                  obscon=obscon.upper(), numobs_from_ledger=numobs_from_ledger, tabform='ascii.ecsv')
    return len(alt_zcat)


def reprocess_ledgers(altmtl_dir, action, fiber_map, survey='main', obscon='dark', zcat_dir=None):
    """
    Carry out one ``reproc`` action: refold a tile the spectroscopic pipeline reprocessed.

    A tile whose reduction changed cannot simply be folded in again: every observation of the
    targets it holds has to be replayed from their unobserved state, in tile order, because a
    changed redshift can alter how many observations a target went on to request. That replay
    is :func:`desitarget.mtl.reprocess_ledger`, the same code the real survey runs; only the
    redshift catalog handed to it is relabelled through ``fiber_map``.

    Parameters
    ----------
    altmtl_dir : str
        Directory of the realization.

    action : astropy.table.Row
        Action to carry out.

    fiber_map : FiberMap
        Correspondence between the real and the alternative assignment of this tile.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    zcat_dir : str, default=None
        Directory holding the real redshift catalogs.

    Returns
    -------
    timestamps : dict
        Timestamp at which each reprocessed tile was refolded.
    """
    from astropy.table import Table
    from desitarget.mtl import make_zcat, reprocess_ledger

    if zcat_dir is None: zcat_dir = utils.ZCAT_DIR

    # Reprocessing revisits tiles that overlap the one being reprocessed, so a target may
    # legitimately appear more than once here.
    zcat = _to_big_endian(Table(make_zcat(zcat_dir, [action], obscon.upper(), survey, allow_overlaps=True)))
    alt_zcat = zcat.copy()
    alt_zcat['TARGETID'] = fiber_map.real_to_alt(zcat['TARGETID'])

    return reprocess_ledger(get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon), alt_zcat,
                            obscon=obscon.upper())


#: Set in the parent before forking, so that the workers of a fiber assignment batch inherit
#: it rather than being handed it through a pickle: a loaded focal plane does not pickle.
_batch_options = {}


def _assign_one(tileid):
    """Carry out the fiber assignment of one tile, in a worker of a batch."""
    do_fiber_assignment(_batch_options['altmtl_dir'], tileid, **_batch_options['kwargs'])
    return tileid


def group_actions(actions):
    """
    Split ``actions`` into the runs that can be carried out together.

    Fiber assignment actions sharing a timestamp are one run: the real survey assigned those
    tiles in a single pass, from one state of the ledgers, and none of them writes to the
    ledgers, so they are independent of each other. Everything else is its own run, because an
    update rewrites the ledgers that the next assignment reads.

    Parameters
    ----------
    actions : astropy.table.Table
        Actions, in the order they must be carried out.

    Yields
    ------
    run : list
        Actions that may be carried out together.
    """
    run = []
    for action in actions:
        if run and action['ACTIONTYPE'] == 'fa' == run[0]['ACTIONTYPE'] \
                and action['ACTIONTIME'] == run[0]['ACTIONTIME']:
            run.append(action)
            continue
        if run: yield run
        run = [action]
    if run: yield run


def warm_hardware(tileids, fiberassign_dir=None):
    """
    Load the focal plane state of every distinct run date among ``tileids``.

    Loading one costs seconds, and fiberassign reloads it per tile from a cache held inside
    desimodel. Warming that cache in the parent before forking means a batch pays once per run
    date rather than once per tile per worker.

    Returns
    -------
    rundates : list
        The distinct run dates that were loaded.
    """
    import fitsio
    from fiberassign.hardware import load_hardware, get_default_exclusion_margins

    rundates = []
    for tileid in tileids:
        header = fitsio.read_header(utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir))
        rundate = str(header['RUNDATE'])
        if rundate not in rundates: rundates.append(rundate)
    margins = get_default_exclusion_margins()
    for rundate in rundates:
        load_hardware(rundate=rundate, add_margins=margins)
    return rundates


def run_realization(altmtl_dir, survey='main', obscon='dark', zcat_dir=None, numobs_from_ledger=True,
                    overwrite=False, fiberassign_dir=None, fiberassign_input_dir=None, nactions=None,
                    numproc=1):
    """
    Replay the survey for one realization, carrying out every action not yet done.

    Parameters
    ----------
    altmtl_dir : str
        Directory of the realization, e.g. ``.../altmtl0/Univ000``.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    zcat_dir : str, default=None
        Directory holding the real redshift catalogs.

    numobs_from_ledger : bool, default=True
        Whether to take the number of observations so far from the ledger.

    overwrite : bool, default=False
        Whether to redo assignments that already exist.

    fiberassign_dir : str, default=None
        Directory of the real fiberassign files.

    fiberassign_input_dir : str, default=None
        Directory of the real per-tile assignment inputs.

    nactions : int, default=None
        Stop after this many actions. Useful to try the loop out on a few tiles.

    numproc : int, default=1
        Number of processes to assign a batch of tiles with. Assignments sharing a timestamp
        are independent, so they run together; updates stay sequential, as each rewrites the
        ledgers the next assignment reads.

    Returns
    -------
    nactions : int
        Number of actions carried out.
    """
    if 'trunk' in altmtl_dir.lower() or 'ops' in altmtl_dir.lower():
        raise ValueError('refusing to update ledgers in {}: the path looks like the real '
                         'surveyops ledgers'.format(altmtl_dir))

    actions = get_actions(read_tile_tracker(altmtl_dir, survey=survey, obscon=obscon), done=False)
    if nactions is not None:
        actions = actions[:nactions]
    unsupported = set(np.unique(actions['ACTIONTYPE'])) - set(SUPPORTED_ACTIONTYPES)
    if unsupported:
        raise NotImplementedError(
            'action list holds {} action(s), which the loop cannot carry out; replaying past them '
            'would diverge from the real survey'.format(sorted(unsupported)))

    logger.info('{}: carrying out {:d} action(s) with {:d} process(es).'.format(
        altmtl_dir, len(actions), numproc))
    if numproc > 1 and os.environ.get('OMP_NUM_THREADS') != '1':
        # fiberassign is threaded, so each worker would otherwise spawn as many threads as the
        # node has cores. Leaving it unset turns an 8x speed-up into a slowdown.
        logger.warning('OMP_NUM_THREADS is {}, not 1; with {:d} processes fiberassign will '
                       'oversubscribe the node and run slower than sequentially.'.format(
                           os.environ.get('OMP_NUM_THREADS', 'unset'), numproc))
    kwargs = dict(survey=survey, obscon=obscon, overwrite=overwrite, fiberassign_dir=fiberassign_dir,
                  fiberassign_input_dir=fiberassign_input_dir)
    idone = 0
    for run in group_actions(actions):
        if len(run) > 1 and numproc > 1:
            # A batch of assignments, all reading the same state of the ledgers.
            tileids = [int(action['TILEID']) for action in run]
            start = time.time()
            rundates = warm_hardware(tileids, fiberassign_dir=fiberassign_dir)
            _batch_options.update(altmtl_dir=altmtl_dir, kwargs=kwargs)
            # Forked workers inherit the focal planes just loaded; spawned ones would not.
            with get_context('fork').Pool(processes=min(numproc, len(tileids))) as pool:
                pool.map(_assign_one, tileids)
            mark_actions_done(altmtl_dir, run, survey=survey, obscon=obscon)
            idone += len(run)
            logger.info('Actions {:d}/{:d} done: fa on {:d} tiles ({:d} run date(s)) in {:.1f} s.'.format(
                idone, len(actions), len(run), len(rundates), time.time() - start))
            continue

        for action in run:
            tileid = int(action['TILEID'])
            if action['ACTIONTYPE'] == 'fa':
                do_fiber_assignment(altmtl_dir, tileid, **kwargs)
            else:
                # The map was written by the fa action of this tile, and rebuilt if missing.
                fiber_map = do_fiber_assignment(altmtl_dir, tileid, **dict(kwargs, overwrite=False))
                if action['ACTIONTYPE'] == 'update':
                    nz = update_ledgers(altmtl_dir, action, fiber_map, survey=survey, obscon=obscon,
                                        zcat_dir=zcat_dir, numobs_from_ledger=numobs_from_ledger)
                    logger.debug('Tile {:d}: folded in {:d} redshift(s).'.format(tileid, nz))
                else:
                    timestamps = reprocess_ledgers(altmtl_dir, action, fiber_map, survey=survey,
                                                   obscon=obscon, zcat_dir=zcat_dir)
                    logger.debug('Tile {:d}: reprocessed, {:d} tile(s) refolded.'.format(
                        tileid, len(timestamps)))
            mark_actions_done(altmtl_dir, [action], survey=survey, obscon=obscon)
            idone += 1
            logger.info('Action {:d}/{:d} done: {} on tile {:d} at {}.'.format(
                idone, len(actions), action['ACTIONTYPE'], tileid, action['ACTIONTIME']))
    return len(actions)


def run_altmtl(base_dir, realizations=1, survey='main', obscon='dark', mpicomm=None, **kwargs):
    """
    Replay the survey for several realizations, one rank per realization.

    Parameters
    ----------
    base_dir : str
        Directory holding the realizations, as ``Univ000``, ``Univ001``, and so on. It may hold
        a ``{:d}``-style field, in which case it is formatted with the realization index: that
        is how a set of mocks, each with its own single realization, is laid out.

    realizations : int, list, default=1
        Number of realizations, or the list of realization indices to run.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    mpicomm : MPI communicator, default=None
        Communicator to spread realizations over. Defaults to ``COMM_WORLD``.

    kwargs : dict
        Other arguments for :func:`run_realization`.

    Returns
    -------
    nactions : int
        Total number of actions carried out, over all realizations.
    """
    from mpi4py import MPI

    if mpicomm is None: mpicomm = MPI.COMM_WORLD
    if np.ndim(realizations) == 0: realizations = list(range(realizations))

    nactions = 0
    for realization in realizations[mpicomm.rank::mpicomm.size]:
        if '{' in base_dir:
            altmtl_dir = base_dir.format(realization)
        else:
            altmtl_dir = utils.get_universe_dir(base_dir, realization=realization)
        start = MPI.Wtime()
        nactions += run_realization(altmtl_dir, survey=survey, obscon=obscon, **kwargs)
        logger.info('Realization {:d} replayed in elapsed time {:.2f} s.'.format(
            realization, MPI.Wtime() - start))

    return mpicomm.allreduce(nactions)

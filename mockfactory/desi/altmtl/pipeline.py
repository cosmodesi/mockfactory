"""
Replaying whole mocks, and several of them at once.

A realization spends almost all of its time inside fiberassign, about eleven seconds a tile on
one core, which is work none of this can avoid. What it can avoid is leaving cores idle during
the stretches that do not parallelise: the assignment groups holding a single tile, the
updates, and the reprocessing. Those are of order a third of a realization, and replaying
several mocks side by side overlaps one mock's serial stretch with another's parallel one.

So the throughput of a node is set by how the cores are split between mocks and the assignment
batches within a mock, and the useful knob is that split rather than either number alone.
"""

import os
import time
import logging
from multiprocessing import get_context

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.pipeline')


def run_mock(targets_fn, altmtl_dir, end_date, survey='main', obscon='dark', realization=0,
             numproc=1, shuffle_subpriority=False, seed=None, state=None, write_ledgers=False,
             **kwargs):
    """
    Replay the survey for one mock, from its target catalog to its assignments.

    Parameters
    ----------
    targets_fn : str
        Path of the mock target catalog.

    altmtl_dir : str
        Directory of the realization to write, e.g. ``.../altmtl0/Univ000``.

    end_date : int, str
        Night the replay stops at.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    realization : int, default=0
        Index of the realization, which enters the subpriority seed.

    numproc : int, default=1
        Processes to assign a batch of tiles with.

    shuffle_subpriority : bool, default=False
        Whether to draw fresh subpriorities. Off by default, which reproduces a mock's own
        assignment; several realizations of one mock need it on.

    seed : int, default=None
        Base seed for the subpriority draw.

    state : LedgerState, default=None
        State to replay against. Built from ``targets_fn`` when not given.

    write_ledgers : bool, default=False
        Whether to write the final state out as healpix ledgers, in the survey's format, under
        ``altmtl_dir``. The replay itself never needs them, and it keeps the state in memory,
        so without this the merged target list is gone once the replay returns.

    kwargs : dict
        Other arguments for :func:`mockfactory.desi.altmtl.loop.run_realization`.

    Returns
    -------
    result : dict
        What was replayed, and how long each part took.
    """
    from .ledger import initialize_realization
    from .loop import run_realization
    from .state import LedgerState

    start = time.time()
    initialize_realization(None, altmtl_dir, realization=realization, survey=survey, obscon=obscon,
                           seed=seed, shuffle_subpriority=shuffle_subpriority, end_date=end_date,
                           ledgers=False, numproc=numproc)
    t_setup = time.time() - start

    if state is None:
        start = time.time()
        state = LedgerState.from_targets(targets_fn, obscon=obscon, survey=survey)
        if shuffle_subpriority:
            rng = np.random.RandomState(seed=(314159 if seed is None else seed) + realization)
            state.current['SUBPRIORITY'] = rng.uniform(size=len(state))
        t_state = time.time() - start
    else:
        t_state = 0.

    start = time.time()
    nactions = run_realization(altmtl_dir, survey=survey, obscon=obscon, numproc=numproc,
                               state=state, **kwargs)
    t_replay = time.time() - start

    logger.info('{}: {:d} actions in {:.0f} s (setup {:.0f} s, state {:.0f} s).'.format(
        altmtl_dir, nactions, t_replay, t_setup, t_state))
    result = {'altmtl_dir': altmtl_dir, 'targets_fn': targets_fn, 'nactions': nactions,
              'seconds': t_replay, 'setup_seconds': t_setup, 'state_seconds': t_state,
              'ntargets': len(state)}
    if write_ledgers:
        start = time.time()
        result['ledger_dir'] = state.write_ledgers(altmtl_dir, survey=survey, obscon=obscon)
        result['ledger_seconds'] = time.time() - start
        logger.info('{}: ledgers written in {:.0f} s.'.format(altmtl_dir, result['ledger_seconds']))
    return result


_mock_options = {}


def _run_one(mock):
    """Replay one mock, in a worker."""
    targets_fn, altmtl_dir, realization, options = mock
    try:
        return run_mock(targets_fn, altmtl_dir, realization=realization,
                        **dict(_mock_options, **options))
    except Exception as exc:
        # One mock failing should not take the others down with it.
        logger.exception('{} failed: {}'.format(altmtl_dir, exc))
        return {'altmtl_dir': altmtl_dir, 'targets_fn': targets_fn, 'error': repr(exc)}


def _run_many(mocks, nummocks):
    """
    Replay ``mocks``, ``nummocks`` at a time, in processes that may fork pools of their own.

    Returns their results in the order the mocks were given.
    """
    ctx = get_context('fork')
    queue = ctx.Queue()

    def _target(index, mock):
        queue.put((index, _run_one(mock)))

    results = [None] * len(mocks)
    pending = list(enumerate(mocks))
    running, done = [], 0
    while done < len(mocks):
        while pending and len(running) < nummocks:
            index, mock = pending.pop(0)
            process = ctx.Process(target=_target, args=(index, mock), daemon=False)
            process.start()
            running.append(process)
        index, result = queue.get()
        results[index] = result
        done += 1
        running = [process for process in running if process.is_alive()]
    for process in running:
        process.join()
    return results


def run_mocks(mocks, end_date, survey='main', obscon='dark', numproc=1, nummocks=1, **kwargs):
    """
    Replay several mocks at once on one node.

    Parameters
    ----------
    mocks : list
        The mocks to replay, each ``(targets_fn, altmtl_dir)``,
        ``(targets_fn, altmtl_dir, realization)`` or ``(targets_fn, altmtl_dir, realization,
        options)``, where ``options`` is a dict of :func:`run_mock` arguments for that mock alone,
        taking precedence over ``kwargs``: e.g. ``{'zfix': 'qso3.txt'}``, since each mock has
        its own quasars.

    end_date : int, str
        Night the replays stop at.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    numproc : int, default=1
        Processes each mock assigns a batch of tiles with.

    nummocks : int, default=1
        How many mocks to replay at the same time. ``numproc * nummocks`` should not exceed the
        cores of the node: the two levels of processes compete for the same ones.

    kwargs : dict
        Other arguments for :func:`run_mock`.

    Returns
    -------
    results : list
        One entry per mock, in the order they were given.
    """
    defaults = (None, None, 0, {})
    mocks = [tuple(mock) + defaults[len(mock):] for mock in mocks]
    ncores = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    if numproc * nummocks > ncores:
        logger.warning('{:d} mocks x {:d} processes is {:d}, against {:d} cores; they will '
                       'contend.'.format(nummocks, numproc, numproc * nummocks, ncores))
    if os.environ.get('OMP_NUM_THREADS') != '1':
        logger.warning('OMP_NUM_THREADS is {}, not 1; fiberassign will spawn a thread per core '
                       'in every worker.'.format(os.environ.get('OMP_NUM_THREADS', 'unset')))

    _mock_options.update(end_date=end_date, survey=survey, obscon=obscon, numproc=numproc, **kwargs)
    logger.info('Replaying {:d} mock(s), {:d} at a time, {:d} process(es) each.'.format(
        len(mocks), nummocks, numproc))

    start = time.time()
    if nummocks > 1 and len(mocks) > 1:
        # Processes rather than a pool, because a pool's workers are daemonic and a mock has to
        # fork a pool of its own to assign its tiles; a daemonic process may not have children.
        # Forked, so that a worker inherits whatever the parent has already loaded; each mock
        # then builds its own state, which is the bulk of the memory.
        results = _run_many(mocks, nummocks)
    else:
        results = [_run_one(mock) for mock in mocks]

    elapsed = time.time() - start
    nfailed = sum('error' in result for result in results)
    logger.info('Replayed {:d} mock(s) in {:.2f} h, {:.2f} h each, {:d} failed.'.format(
        len(mocks), elapsed / 3600, elapsed / 3600 / max(len(mocks), 1), nfailed))
    return results

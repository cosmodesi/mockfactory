"""
Alternative merged target list (altMTL) fiber assignment for mocks.

The real survey is replayed against a mock catalog: the same tiles are assigned on the same
dates, with the same sky, focal plane state and hour angles, but with the mock's targets. Which
target wins a fiber then differs, and running several realizations of that gives, per target,
the probability of having been observed.

A typical run is:

.. code-block:: python

    from mockfactory.desi.altmtl import make_initial_ledgers, initialize_realization, run_altmtl

    # Once per mock: turn the target catalog into healpix ledgers.
    make_initial_ledgers(targets_fn, initial_dir, numproc=32)

    # Once per realization: copy the ledgers and build the action list.
    for realization in range(nrealizations):
        initialize_realization(initial_dir, get_universe_dir(altmtl_dir, realization),
                               realization=realization, end_date=20240418)

    # Replay the survey, one rank per realization.
    run_altmtl(altmtl_dir, realizations=nrealizations)

The ``fa``, ``update`` and ``reproc`` actions are carried out. An action list needing vetoes,
ledger additions or the Lyman-alpha numobs increase raises rather than silently skipping them;
those appear only for end dates from 2025 on.

:func:`compute_potential_assignments` gives, separately, which targets each fiber could have
reached, and :func:`write_bitweights` turns a set of realizations into the probability that
each target was observed.
"""

from .utils import get_universe_dir
from .compat import patch_write_mtl, numpy_converts_size_one_arrays
from .tiletracker import make_tile_tracker, read_tile_tracker, get_actions, get_tile_tracker_fn
from .ledger import make_initial_ledgers, initialize_realization, get_ledger_dir, get_healpixels
from .assignment import FiberMap, do_fiber_assignment, make_fiber_map
from .loop import (run_altmtl, run_realization, update_ledgers, reprocess_ledgers, group_actions,
                   update_batch, read_zcats)
from .state import LedgerState
from .reprocess import reprocess_state
from .pota import compute_potential_assignments
from .bitweights import compute_bitweights, write_bitweights, pack_bitweights, unpack_bitweights
from .targets import make_targets, write_targets, get_target_bits, get_priority_numobs

# altMTL fiber assignment for mocks: findings

Notes on `mockfactory.desi.altmtl`, a self-contained implementation of the alternative merged
target list pipeline of `desihub/LSS` `scripts/mock_tools/runHoli.txt`, steps 6 to 9. Written
for whoever runs or maintains DESI mock fiber assignment.

Status as of 2026-09-19. Branch `adematti-dev`, commits `56ba226..0068d91`.

## What it does

The real survey is replayed against a mock catalog: the same tiles are assigned on the same
dates, with the same sky, focal plane state, hour angles and run dates, but with the mock's
targets. Which target wins a fiber then differs, and running several realizations of that
gives, per target, the probability of having been observed.

It keeps `desitarget`, `fiberassign` and `desimodel`, which mockfactory already depended on,
and does not import `LSS`: the tile tracker, the real-to-alternative fiber map, the action
loop, the potential assignments and the bitweights are all carried out here. `fba_run` is
called in process through `parse_assign` / `run_assign_full` rather than through a generated
shell script, which needs fiberassign 4.0 or above.

## Validation

Against the reference run at
`DESI_ROOT/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl0`:

- the tile tracker reproduces its action list exactly, all 13610 actions
  (6671 `fa`, 6671 `update`, 268 `reproc`);
- `make_fiber_map` reproduces its fiber maps exactly on 12 tiles, about 50000 mappings;
- `pack_bitweights` is bit-identical to the LSS implementation for 1, 8, 64, 65, 128 and 130
  realizations;
- reprocessing reproduces `desitarget.mtl.reprocess_ledger` on every column that carries
  state: `NUMOBS`, `NUMOBS_MORE`, `PRIORITY`, `TARGET_STATE`, `Z`, `ZWARN`, `ZTILEID`.

### Complete DA2 replay

All 13610 actions against the production `forFA0.fits`, 42253356 targets, no ledgers on disk:

    13610 actions in 1.48 h (1.56 h including the state build), numproc=32
    6671 assignments written, none missing

Assignments against the official run: 87 identical, 6584 differing by a **median of 6
locations in 5020**, maximum 40. That is 99.88% of fibers agreeing per tile.

The residual is the software stack, not the implementation. The reference headers pin
fiberassign 5.7.2.dev3588 and desimodel 0.19.1.dev743, against 5.9.0.dev3738 and 0.20.1.dev850
here. `FIBERSTATUS` is identical on all 5020 locations, so it is not broken fibers, and the
differing locations include sky and standard targets swapping with science ones, which is what
a change in how fiberassign allocates them looks like. Over survey time the difference rises
from a median of 3 in the first eighth to 8 by the third and then plateaus:

    eighth       1     2     3     4     5     6     7     8
    median       3     5     7     6     7     7     8     8
    90th pct     7    10    13    13    14    14    14    14

A fixed per-tile perturbation compounding to a steady state looks like this; drift in the
replay would keep climbing, and a fault in reprocessing would have stepped at the first
reprocessed tiles in December 2021, in the second eighth. Neither happens. Figure:
`/pscratch/sd/a/adematti/claude/altmtl/diff_vs_time.png`.

This is an inference, not a demonstration. Settling it needs a run under fiberassign 5.7.2 and
desimodel 0.19.1, which are not installed here; `compat.supported` and
`assignment.accepts_fafns_for_stucksky` exist to make that possible.

## Two bugs worth reporting upstream

### `desitarget.io.write_mtl` fails under numpy 2.4

    dr = np.unique(release // 1000)     # array([0]) for any single-release catalog
    try:
        drint = int(dr)
    except TypeError:
        raise TypeError("Multiple data releases in MTL ({})".format(dr))

The `except` was meant for genuinely multiple releases, where `int()` on a multi-element array
has always raised. `int()` on a one-element array was only deprecated in numpy 1.25 and raises
from numpy 2.4, so every catalog now trips it and reports "Multiple data releases in MTL
([0])", which is the opposite of the truth. Mock and real target identifiers both decode to
release 0. The code is identical in desitarget 2.5.0.dev, 4.6.0.dev and 5.4.1, so no released
version fixes it. `make_ledger` simply fails under cosmodesi (numpy 2.4.3); it works under
`desi_environment` (numpy 2.3.5) with a deprecation warning.

Fix: test the length instead of catching an exception whose meaning changed --
`elif len(dr) > 1: raise ...` and `else: drint = int(dr[0])`.
Worked around in `compat.patch_write_mtl`, a no-op below numpy 2.4.

### Reprocessing writes the real survey's position onto the mock target

Relabelling a redshift catalog for an alternative universe rewrites `TARGETID` but leaves
`RA` and `DEC` as the real survey's; LSS `makeAlternateZCat` rewrites only the identifier. An
`update` never writes positions, so it is harmless there, and a run of 11.6 million targets
through `fa` and `update` came back with not one position altered. `reprocess_ledger` does
write them: both loops that copy the catalog back run over every column with no exclusion for
`RA` and `DEC`, and the rows added back for bad observations come straight from the catalog.

Measured on tile 1006: 440 of 3623 observed targets displaced, by up to 0.039 degrees, which
is the fiber patrol radius. Two crossed a healpix boundary, were appended to a second ledger,
and left the ledgers holding one target twice at two positions, 45617 rows for 45615 targets.

This is live in production. runHoli copies a frozen tracker holding 268 reprocessing actions
out of 13610, so every realization hits it: order 100000 displaced positions per realization,
all on observed targets, which is the clustering sample.

Fixed here by substituting the positions along with the identifiers, after which none is
altered. The ledger path has no state to look them up in and warns instead.

## Performance

Starting point, the sequential loop with healpix ledgers: 9.27 h per realization over the 6671
DA2 tiles. Now 1.48 h, about 6x, measured end to end.

Three things got it there, each verified to leave assignments and final state bit-identical.

**Batch the passes.** The real survey assigned and updated tiles in passes, so actions come in
groups sharing a timestamp. Tiles in an assignment group read one state and none of them
writes, so they run in a forked pool: 5.00 to 0.60 s per tile on one measured batch. Updates
merge into a single `make_mtl` call, 4.21 to 0.57 s over 28 tiles, after checking that no
target was observed on two tiles of the batch.

**Drop the ecsv ledgers.** They are the survey's persistence format, not something a mock
needs. `LedgerState` holds the merged target list in memory: reading one tile costs 0.12 s
against 1.5 s, building the state 95 s against about 37 min for `make_ledger` at this size,
and there is no 2 GB ledger copy per realization.

**Replay reprocessing in memory.** 26.4 s per reprocessed tile through ledgers, 1.27 s against
the state.

`OMP_NUM_THREADS=1` is mandatory with a pool: fiberassign is threaded, so each worker
otherwise spawns a thread per core and a batch runs slower than sequentially. LSS sets it at
the top of its own driver.

### Where the 1.48 h goes

Measured from the log of the complete run:

    fa batches     207 groups, 6574 tiles   2752 s   51.6%
    fa singletons   97 actions               302 s    5.6%   median 3.0 s
    update batches 155 groups, 6666 tiles    831 s   15.6%
    reprocessing   268 actions              1454 s   27.2%   median 5.42 s

## Handing fiberassign the targets rather than a file

`fba_run` takes its science targets as a file, so the loop wrote one per tile and fiberassign
read it back. Measured over six tiles, of the 4.7 s an `fa` action takes, 0.9 s is building and
writing that file and about 1.8 s more is inside fiberassign reading it and turning it into its
own objects; fiberassign's own timers put the geometry and the assignment at some 1.5 s and the
output at 0.3 s. Over half of an assignment was serialisation.

`assignment.targets_in_memory` removes it. It replaces
:func:`fiberassign.targets.load_target_file` for the one path that names the tile's target
file, handing :func:`fiberassign.targets.load_target_table` the array instead; the sky and
secondary files still go through the real function, and everything after loading, which is the
assignment itself, is fiberassign's own code untouched. That is the narrowest possible
substitution: the alternative, which `mockfactory.desi.fiber_assignment` takes, is to
reimplement fiberassign's driver and pin it to an upstream commit.

Turned on with `load_targets='memory'`, through `do_fiber_assignment`, `run_realization` or
`run_mock`. On eight tiles, twice:

    file      4.31 and 4.02 s per tile
    memory    2.48 and 2.48 s per tile      1.6 to 1.7x

and the three extensions of every `fba-*.fits` agree column for column between the two. Since
assignments are about 57% of a realization, that is roughly 0.90 h to 0.70 h overall. The
parallel efficiency of the assignment pool was 25%, and per tile file traffic was one of the
suspects, so a batched run may gain more than the per tile figure says; that is not measured.

It stays off by default. Eight tiles agreeing is not a full survey agreeing, and this is the
one place in the module that does not simply call what `fba_run` calls. The gate for making it
the default is a complete replay run both ways with identical assignments.

## The healpix index was being rebuilt in every worker

`LedgerState` builds its healpix index on demand, inside `rows_in_healpixels`. The only caller
that needs it is `targets_in_tiles`, and that runs **only in the forked workers that assign
tiles**: an update is keyed on the target identifier, and in the parent `rows_in_healpixels` is
reached only by reprocessing, 268 actions of 13610. So every worker of every assignment batch
argsorted all 42 million rows, used the index for its own handful of tiles, and discarded it on
exit. Cold it costs 5.28 s, warm 0.063 s, a factor 84, and thirty two of them argsorting at
once is worse than the sum of its parts.

`LedgerState.build_index()`, called in the loop before the pool forks, builds it once and lets
the workers inherit it. The same 128 tiles, the same harness, only that changed:

    workers   cold tiles/s   warm tiles/s   speed-up   efficiency cold -> warm
       8         2.242          2.659         1.19x        83% -> 100%
      16         3.151          4.954         1.57x        58% ->  93%
      32         3.321          8.329         2.51x        31% ->  78%
      64         2.346          8.768         3.74x        11% ->  41%

It costs 5.1 s once. Use 32 or 64 workers; sixty four had been the worst setting and is now the
best.

**Over a full replay it is worth 1.01 times**: 2899 s against 2928 s, `numproc=32`, identical
in every other respect. The projection from these tiles, that a realization would fall to near
half an hour, does not hold, and the reason is in the paragraph above. Reprocessing reaches
`rows_in_healpixels` in the parent; the first one falls at action 2451 of 13610, and only 1262
of the 6671 assignments come before it. For four fifths of the run the parent already held the
index, warmed by reprocessing rather than on purpose. The scan sees the full effect because it
never reprocesses.

The fix is kept: one line, five seconds, and it changes no assignment, checked on 200 tiles
drawn at random from the two replays. It is worth having for a replay with few or no
reprocessing actions, which is where the table above applies. What it is not is a speed-up of
production.

This is the third projection from a micro-benchmark of this pipeline that a full replay did not
confirm, after the in-memory targets and the worker count. The rule that follows is to measure
end to end before quoting a number.

Three things that had looked like separate findings were all this. The pool appearing to
saturate at four workers. Sixty four being slower than thirty two in absolute terms. And
handing fiberassign its targets in memory, worth 1.6 to 1.7 times on one tile, being worth
exactly nothing over a full replay: five seconds of index rebuilding swamps the 1.8 s of file
parsing it removes. A fourth, a scan that seemed to show the size of the state mattering, was
the same thing seen from the other side: building a cut-down state happened to call
`targets_in_tiles` in the parent first, which warmed the index. Shrinking the state twelve fold
is worth nothing up to thirty two workers and 15% at sixty four.

## Room for speed improvement

**The healpix index is invalidated on every reprocessing, about 1070 s.** `absorb_rows` sets
`_pixel_index = None`, so the next lookup rebuilds it with an `argsort` over all 42 million
rows. Measured: the same reprocessed tiles took 1.27 s before that index existed and 5.4 s
after, flat across the run, so it is the rebuild and not history accumulating. Since
reprocessed rows now keep the mock's positions, their healpix cannot change, so the index
never needs invalidating there. Invalidate only when a pixel actually changed, or update the
index for the few thousand rows touched. Nearly free, worth 20% of the run.

**Group consecutive assignments, not just simultaneous ones, about 900 s.** Grouping requires
an identical timestamp, which leaves 97 singleton tiles each taking a whole round alone.
Assignments do not write state, so any run of consecutive assignment actions is independent,
whatever their timestamps. Relaxing the rule gives 148 groups instead of 304, 283 rounds of 32
instead of 424, and 4 singletons instead of 97. Worth about 17% of the run.

**Parallel efficiency of the assignment pool is 25%.** A round of up to 32 tiles takes a median
7.03 s where a lone tile takes 3.0 s. The suspects are Lustre traffic, since every worker
writes a target file and an assignment, and memory bandwidth. Writing the per-tile target file
to `/dev/shm` rather than scratch is the obvious thing to try; LSS has a `usetmp` flag for
exactly this. Unmeasured, and the ceiling if it were perfect is another 25% of the run.

Together the first two would bring a realization to roughly 0.95 h, around 10x the sequential
loop rather than 6x.

## Not yet exercised

- `pota`, the potential assignments of step 9: written, never run. `check_pota.py` compares
  against the `FAVAIL` extension fiberassign writes, which is the natural reference.
- Bitweights over several realizations, and the MPI fan-out over realizations in `run_altmtl`.
- The bright program.
- `veto`, `lya1b` and `addnew` actions raise rather than being skipped silently. They appear
  only for end dates from 2025 on.

# altmtl

Alternative merged target list fiber assignment for mocks.

The real survey is replayed against a mock catalog: the same tiles on the same dates, with the
same sky, focal plane state, hour angles and run dates, but the mock's targets. Which target
wins a fiber then differs, and running several realizations gives, per target, the probability
of having been observed. That probability is what a clustering measurement needs in order to
correct for the targets the survey could not reach.

This is a self-contained implementation of steps 6 to 9 of `desihub/LSS`
`scripts/mock_tools/runHoli.txt`. It keeps `desitarget`, `fiberassign` and `desimodel`, and
does not import `LSS`.

## Input files

Two kinds go in: the mock's own catalog, and the real survey's record of what it did.

From the mock, one fits file holding every target of every tracer. `targets.make_targets`
builds it from mockfactory catalogs and `targets.write_targets` writes it;
`write_targets` refuses a catalog missing any of the required columns, which are
`targets.TARGET_COLUMNS`:

| column | dtype | what it is |
| --- | --- | --- |
| `TARGETID` | i8 | unique identifier. Assignment, ledgers, potential assignments and bitweights all key on it, so it has to be unique across the tracers merged into the file |
| `RA`, `DEC` | f8 | position in degrees. Also what the state is indexed by healpix on |
| `DESI_TARGET` | i8 | main target bitmask, from `desitarget.targetmask.desi_mask`. A bright galaxy carries `BGS_ANY` here as well as its own bit in `BGS_TARGET` |
| `BGS_TARGET` | i8 | bright galaxy bitmask, from `bgs_mask`; 0 for a dark tracer. It is this bit, not `DESI_TARGET`, that sets a bright galaxy's priority |
| `MWS_TARGET` | i8 | milky way survey bitmask; 0 for a mock of the extragalactic tracers |
| `SCND_TARGET` | i8 | secondary target bitmask; 0, secondaries come from the survey's own files |
| `SUBPRIORITY` | f8 | uniform in [0, 1). The tie break between targets of equal priority, and so the one column that differs between realizations of the same mock. Every target needs one: ties would otherwise be broken arbitrarily by fiberassign |
| `OBSCONDITIONS` | i8 | bitmask of the conditions the target may be observed under, from `desitarget.targetmask.obsconditions`. Must agree with the program being replayed |
| `PRIORITY_INIT` | i8 | priority before any observation, implied by the target bits |
| `PRIORITY` | i8 | current priority; equal to `PRIORITY_INIT` in a fresh catalog, and what the loop lowers as a target is observed |
| `NUMOBS_INIT` | i8 | observations the target asks for, implied by the target bits |
| `NUMOBS_MORE` | i8 | observations still wanted; equal to `NUMOBS_INIT` in a fresh catalog, and what the loop counts down |
| `ZWARN` | i8 | redshift warning bitmask; 0 in a fresh catalog |
| `RSDZ` | f8 | the redshift the target is placed at, redshift space distortions included. Carried through to the potential assignments, where the clustering stage reads it as the observed redshift |

`PRIORITY_INIT` and `NUMOBS_INIT` are not written down anywhere here: `get_priority_numobs`
reads them off the target mask through `desitarget.targets.initial_priority_numobs`, so a mock
stays consistent with whatever the survey currently declares. `PRIORITY` and `NUMOBS_MORE`
start equal to them.

Two header keywords identify the file, and `write_targets` stamps them: the table extension
must be named `TARGETS`, and `OBSCON` must be `DARK` or `BRIGHT`, matching the ledgers being
built.

Anything else in the file is carried along rather than required: `compute_potential_assignments`
propagates every column of the target catalog into `pota-<PROGRAM>.fits`. That is how the
survey's own `forFA` files come to hold `TRUEZ` and the imaging columns `MASKBITS`, `NOBS_G`,
`NOBS_R`, `NOBS_Z`, `R_MAG_ABS`, `R_MAG_APP`, `G_R_REST`, `G_R_OBS`: altmtl never looks at
them, and the clustering stage downstream needs them for its vetoes and its absolute magnitude
cut. Include them if that stage will run.

From the survey, read only, everything under `DESI_ROOT` (`/dvs_ro/cfs/cdirs/desi`, or
`DESI_ROOT_READONLY`). `utils` holds every path, and nothing here writes to any of them:

| file | what it is read for |
| --- | --- |
| `target/fiberassign/tiles/trunk/<ts3>/fiberassign-<ts>.fits.gz` | per tile: the mtl timestamp that orders the action list, and from the header the run date, field rotation and hour angle. Also the real fiber map, to compare against |
| `survey/fiberassign/<survey>/<ts3>/` | per tile: the sky, secondary, gfa and too target files handed to fiberassign unchanged |
| `survey/ops/surveyops/trunk/ops/tiles-specstatus.ecsv` | which tiles were observed, and when |
| `survey/ops/surveyops/trunk/mtl/mtl-done-tiles.ecsv` | the `update` actions and their archive dates |
| `survey/ops/surveyops/trunk/mtl/mtl-done-vetoes.ecsv` | the `veto` actions |
| `spectro/redux/daily/` | the real redshift catalogs, folded into the alternative ledgers |

`<ts>` is the tile id zero padded to six digits and `<ts3>` its first three, which is how the
survey shards these directories.

## What happens, step by step

### 1. The target state

`LedgerState.from_targets` turns the target catalog into the merged target list the survey
would have kept: one row per target, carrying the columns that change as it is observed, such
as `PRIORITY`, `NUMOBS_MORE` and `TIMESTAMP`. It is sorted by `TARGETID` so a row can be found
by binary search, and the healpix of each row is kept beside it, because a tile asks for a
handful of pixels at a time.

`initialize_realization` prepares one realization. With `shuffle_subpriority`, it draws fresh
subpriorities from a seed that includes the realization index: subpriority is the tie break
between targets of equal priority, so this is the one thing that differs between realizations
of the same mock, and the whole reason different targets win fibers.

### 2. The action list

`make_tile_tracker` reads `tiles-specstatus` and `mtl-done-tiles` and produces the table of
actions, sorted by time, that `tiletracker` documents: `fa` to assign a tile, `update` to fold
its observations back in, `reproc` for a tile the spectroscopic pipeline later reprocessed.
Replaying that order is the whole point; assigning a tile before the tiles that informed it
would give a different answer.

### 3. The loop

`run_realization` walks the action list in order.

An **`fa`** action runs fiberassign for one tile against the state as it stands. The survey's
own inputs are reused as they are, so the focal plane, the sky and the pointing are identical
to the real assignment and only the science targets differ. The result is written as
`fba-<ts>.fits` and reduced to a `FiberMap`: the alternative target now on fiber `f`, against
the real target the survey had on fiber `f`.

An **`update`** action reads the real redshifts of that tile, relabels each one with the
alternative target that shares its fiber, and folds them into the state. That is the hinge of
the method: the alternative survey never invents an observation, it reuses a real one and only
changes which target it belonged to. The state then advances by the ordinary rules, a target
that has had its observations dropping in priority and stopping being requested.

A **`reproc`** action replays every observation of the affected targets from their unobserved
state, which is why the state keeps its full history rather than only the latest row.

`veto`, `lya1b` and `addnew` raise rather than being skipped, so an action list that needs them
fails loudly instead of quietly diverging from the real survey. They appear only for end dates
from 2025 on.

A realization lands in `<altmtl_dir>/Univ<nnn>/fa/<SURVEY>/<rundate>/fba-<ts>.fits`, one
directory per fiberassign run date, plus the tile tracker recording what was done.

### 4. Potential assignments

`compute_potential_assignments` asks a different question, and does not depend on the ledgers
at all: for every tile, which targets each fiber *could* have reached, and which of those it
could only have reached by colliding with a neighbouring positioner. This is the denominator
the assignment is measured against, and one file serves every realization.

### 5. Bitweights

`write_bitweights` stacks the realizations: one bit per target per realization saying whether
it was observed, packed by `pack_bitweights`. The mean over realizations is the probability of
observation, and its inverse is the weight that corrects a clustering measurement for the
targets fiber assignment could not reach.

## Running

One mock, everything in memory:

```python
from mockfactory.desi.altmtl import run_mock

run_mock(targets_fn, altmtl_dir, end_date=20240418, obscon='dark', numproc=32)
```

Several mocks on one node, which is the efficient way to fill it:

```python
from mockfactory.desi.altmtl import run_mocks

mocks = [(f'forFA{i}.fits', f'altmtl{i}/Univ000') for i in range(25)]
run_mocks(mocks, end_date=20240418, obscon='bright', numproc=6, nummocks=10)
```

`nummocks` is how many run at the same time, not how many there are: that is `len(mocks)`.
Twenty five mocks go through ten at a time, six workers each, so sixty workers are busy. Keep
`numproc * nummocks` at or below the cores of the node.

Then the potential assignments, and the bitweights over the realizations:

```python
from mockfactory.desi.altmtl import compute_potential_assignments, write_bitweights

compute_potential_assignments(targets_fn, pota_fn, tiles_fn, program='DARK', numproc=32)
write_bitweights(altmtl_dir, output_dir, realizations=128)
```

`run_realization` and `do_fiber_assignment` are there for a single realization or a single
tile. The ledger path, `make_initial_ledgers` and `initialize_realization(ledgers=True)`, still
works and writes ecsv ledgers the way the survey does; it is much slower and only worth it to
compare against `desitarget` directly.

## Things that will bite you

- **Set `OMP_NUM_THREADS=1` before running with `numproc > 1`.** fiberassign is threaded, so
  each of the `numproc` workers otherwise spawns a thread per core: the node is oversubscribed
  and the batch runs slower than a plain serial loop. `numproc=1` is the default and needs
  nothing; `run_realization` warns when `numproc > 1` and the variable is not 1.
- **Assignment batches are small**, a median of 29 tiles for dark and 24 for bright, and the
  pool is sized by the batch. So more than about 32 workers per mock is wasted: 64 workers sit
  at 54% occupancy. Ten mocks at 6 workers each reach nearly twice the node throughput of one
  mock at 32.
- **`veto`, `lya1b` and `addnew` actions raise** rather than being skipped. They appear only
  for end dates from 2025 on.
- Compute nodes only, inside an interactive allocation.

## SV3

`survey='sv3'` replays the third survey validation programme (April to June 2021: 239 dark and
214 bright tiles, on rosettes passed over up to a dozen times) against a mock.

```python
from mockfactory.desi.altmtl import make_targets, write_targets, run_mock

targets = make_targets({'LRG': lrg, 'ELG_LOP|ELG_HIP': elg_hip, 'ELG_LOP': elg, 'QSO': qso},
                       obscon='dark', survey='sv3', seed=42)
write_targets(targets, targets_fn, obscon='dark')
run_mock(targets_fn, altmtl_dir, end_date=20210701, survey='sv3', obscon='dark', numproc=32)
```

What differs from the main survey, and why:

- **Target bits.** The catalog carries `SV3_DESI_TARGET`, `SV3_BGS_TARGET`, `SV3_MWS_TARGET` and
  `SV3_SCND_TARGET` in place of the main columns (`targets.get_target_columns('sv3')`), with bits
  from `desitarget.sv3.sv3_targetmask`. desitarget reads the survey off these names and applies
  the SV3 priorities: 103xxx in dark, `NUMOBS_INIT` 9 for LRG, ELG and BGS, a good redshift
  retiring a target to priority 2. `LedgerState.from_targets` refuses a catalog whose columns
  are not those of the survey it is asked to replay.
- **Fiberassign.** No SV3 tile can be assigned by the current fiberassign: its SV3 masks have
  no gaia standard bit, and the early run dates carry no timezone. Each SV3 tile is reassigned
  in a subprocess, by `fba_run` of the release that assigned it (2.2.0 to 4.0.0), from the
  desiconda 20230111-2.1.0 tree, with the options its `FAARGS` header records, and with
  `SKYBRICKS_DIR` set, without which stuck positioners never land on sky (some 950 fibers of a
  tile then differ): v2 for the 2.4 tiles, which differ by 8% with v3, and v3 from 2.5 on, with
  v2 instead of which a stuck positioner here and there misses its sky and its slitblock bumps a
  filler target to sky (`assignment.get_legacy_skybricks_dir`). The night of
  2021-04-10T21:28:37 is reproduced only with the focal plane of 20:00
  (`assignment.LEGACY_RUNDATES`). That tree's
  modulefile asks for a cray-mpich Perlmutter no longer has, so `assignment.get_legacy_environ`
  sets it up by hand. The label `2.2.0.dev2811` hides two codes: 2.2.0 reproduces it up to
  2021-04-13, 2.3.0 from 2021-04-14 (`assignment.LEGACY_DEV2811_SWITCH`). Targets reach the
  subprocess through a file, whatever `load_targets` says. Each tile pays for starting
  python and loading its focal plane: a pass of 16 tiles takes 30 s on 32 workers, the whole
  dark replay 19 minutes on one node.
- **Subpriorities.** SV3 drew them afresh on every tile: a target shared by tiles 1 and 2 has
  uncorrelated values on them (correlation -0.008 over 27535 targets), neither its ledger's.
  `tile_subpriority` does the same, from a seed and the tile id; `run_mock` turns it on for SV.
  One subpriority per target instead has the same targets lose every tie on every pass.
- **Order of the replay.** Many SV3 tiles were designed days before the `MTLTIME` stamped on
  them: tile 315 says 2021-04-22T18:55, but the latest ledger row in its target file is from
  2021-04-19, so it never saw the observations of tile 314 folded in on 2021-04-22T17:09. An SV
  `fa` is placed after the latest ledger row in the tile's own target file instead.
- **Paths and redshifts.** Per-tile inputs live under `survey/fiberassign/SV3/<night>/`, looked
  up by tile. Redshifts are read by desitarget's SV path, the `zbest` files of
  `daily/tiles/cumulative/<tile>/<ZDATE>`, so the tile tracker carries `ZDATE`. SV3 has no
  reprocessing and no veto actions.

Validation, replaying the real SV3 data: the real targets in their initial state (first row of
each in the surveyops SV3 ledgers) and `tile_subpriority='real'`, which takes each tile's own
subpriorities from its real target file, so that the replay should put the real target on every
fiber of every tile.

| | fibers holding a real target | differ | agreement |
| --- | --- | --- | --- |
| dark, 239 tiles | 1 028 319 | 0 | 1 |
| bright, 214 tiles | 911 914 | 0 | 1 |

The replay is exact. Dark agreement along the way, in the order the fixes were found: 61% with
one subpriority per target (first 80 actions), 97.4% with the tiles' own, 99.92% with `2.2.0.dev2811` split by
date, unchanged by the replay order (which took bright from 98.7% to 99.983%), 1 fiber left
with the 2021-04-10 run date, and none with skybricks v3 from fiberassign 2.5 on.

Compared with desihub/LSS (`LSS.SV3.altmtltools`, `LSS.SV3.fatools`), which replays SV3 too:
- the same release choice, including the 2.2.0 / 2.3.0 split of `2.2.0.dev2811` around
  2021-04-13; and two fixes taken from there: the run date moved from 2021-04-10T21:28:37 to
  20:00, without which the tiles of that night differ by some 15 fibers each, and through the
  observations they move 800 fibers over the dark replay, and skybricks v2 for the 2.4 tiles
  only;
- the assignment options are those of `FAARGS` here, set by release there;
- LSS orders an `fa` by `MTLTIME` and patches tile 315 alone, in its reproduction mode, with
  the priorities of the real target file; here every SV tile is ordered by the ledger state its
  target file was made from, which also applies to mocks;
- LSS shuffles one subpriority per target and realization; SV3 drew them per tile, which
  `tile_subpriority` reproduces, and which `run_mock` does by default for SV.

Not done yet: potential assignments (`compute_potential_assignments` still loads the current
fiberassign's focal plane, which an SV3 run date breaks), and bitweights over SV3 realizations.

## What makes it fast

Against the survey's own altMTL -- one tile at a time, the MTL persisted as ecsv ledgers -- on a
mock at the real dark-time density. Both levers leave assignments and final state bit for bit
identical:

| change | gain |
| --- | --- |
| batch the passes: tiles sharing an `ACTIONTIME` read one state and none writes, so they run in a forked pool | 5.00 -> 0.60 s a tile, 8.3x |
| hold the MTL in memory (`LedgerState`) rather than rewriting ecsv ledgers | `read_targets_in_tiles` 1.5 -> 0.12 s a tile, and no 2 GB ledger copy a realization |
| **one dark realization, end to end** | **9.3 h -> 0.6 h, 15x** |

Set `OMP_NUM_THREADS=1` whenever `numproc > 1`, or each worker spawns a thread per core and the
8.3x becomes a slowdown; `run_realization` warns. The survey's scripts set it too, so it is not
a gain over them, and no production wall clock was measured here, so the 15x is against the
approach rather than against a timed run.

Cost is set by occupancy, not by per-tile speed: removing 40% of the per-tile work changes the
total by nothing. Two optimisations are real per tile and vanish end to end --
`targets_in_memory` (1.6-1.7x a tile, 1.00x overall) and warming `LedgerState._pixel_index`
(84x cold, 1.01x overall). Quote them only as per-tile numbers. What does pay is filling the
node: ten bright mocks at `numproc=6` run 9.5 tiles a second against 5.96 for one at
`numproc=32`.

## Validation

**Verdict: this implementation is taken to be correct.** Every difference from production's own
replays has been traced to its cause, and each one is outside this code -- one in the
fiberassign build, one in `LSS`. Given the same software and the same inputs the replay is bit
for bit identical to production, and where it is not, it is the side following the rules
`desitarget` publishes. Nothing unexplained is left in the comparison.

That is a conclusion drawn from two productions, DA2 bright and DA2 dark, over 13 realizations.
It rests on the assignments and the merged target list state; end-to-end clustering statistics
are a separate question, and a third production could still turn something up.

Against `DESI_ROOT/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl0`: the
action list reproduces exactly, all 13610 actions; the fiber maps reproduce exactly;
`pack_bitweights` is bit-identical to the `LSS` implementation; reprocessing reproduces
`desitarget.mtl.reprocess_ledger` on every column carrying state.

Assignments, against production's own replays:

| | bright, `AbacusSummit_v4_1` | dark, `AbacusHF_DR2v2`, 12 mocks |
| --- | --- | --- |
| locations compared | | 338 675 655 |
| agreement | 0.99907 | **0.9999929** |
| differing per tile | median 4 of 5020 | median 0, worst 6 of 4230 |

Both residuals are understood and neither is ours. The bright one is the fiberassign build,
below. The dark one is `LSS`: `mockaltmtltools.py:1287` calls `update_ledger` with
`#, targets = targets` commented out, so ledger rows are found by healpix from the zcat's
`RA`/`DEC`, which `makeAlternateZCat` leaves as the real survey's. A mock target sitting across
a pixel boundary from the real target it replaced is never found and its observation is dropped,
after which it keeps `UNOBS` priority and wins fibers it should lose -- 0.0018% of observations,
against the 0.0007% of locations that differ. `LedgerState` is indexed by `TARGETID` and never
looks a target up by position, so the replay is the side that is right.

Under the reference's own software -- fiberassign 5.7.2, desimodel 0.19.1, desimeter 0.7.1 -- a
replay reproduces the official assignments exactly, every fiber of every tile. Under the
fiberassign installed here, 5.9.0, a complete replay agrees on 99.88% of fibers per tile, and
that residual is one line of fiberassign: the inner keepout radius in `Hardware::position_xy_bad`
was an `::abs` whose overload depended on the compiler. 5.8.0 made the choice explicit as
`--fba_use_fabs` and takes it from the rundate, which is what the real survey got; the DA2
references have 2021 rundates but were built under desiconda 20240425-2.2.0, a gcc 13 build, so
they carry the other behaviour. Passing `--fba_use_fabs 1` makes the replay bit-exact against
them again, all 1526080 locations, which is how the table above is measured. It is not what
`run_fiberassign` does: a mock should follow the data, not the reference. python, numpy,
desimodel and desimeter change nothing either way.

`../../../desi/altmtl_findings.md` has the evidence, the two upstream bugs this turned up, and the
performance history.

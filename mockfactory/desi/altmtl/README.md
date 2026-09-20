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

- **`OMP_NUM_THREADS=1` is mandatory.** fiberassign is threaded, so each worker otherwise
  spawns a thread per core and a batch runs slower than a serial loop.
- **Assignment batches are small**, a median of 29 tiles for dark and 24 for bright, and the
  pool is sized by the batch. So more than about 32 workers per mock is wasted: 64 workers sit
  at 54% occupancy. Ten mocks at 6 workers each reach nearly twice the node throughput of one
  mock at 32.
- **`veto`, `lya1b` and `addnew` actions raise** rather than being skipped. They appear only
  for end dates from 2025 on.
- Compute nodes only, inside an interactive allocation.

## Validation

Against `DESI_ROOT/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl0`: the
action list reproduces exactly, all 13610 actions; the fiber maps reproduce exactly;
`pack_bitweights` is bit-identical to the `LSS` implementation; reprocessing reproduces
`desitarget.mtl.reprocess_ledger` on every column carrying state. A complete replay agrees with
the official assignments on 99.88% of fibers per tile, the residual being a fiberassign version
difference.

`../altmtl_findings.md` has the evidence, the two upstream bugs this turned up, and the
performance history.

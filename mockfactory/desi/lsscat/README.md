# lsscat

Large scale structure catalogs from an alternative merged target list run.

`mockfactory.desi.altmtl` replays the survey against a mock and says which targets got a fiber.
This turns that into the catalogs a clustering measurement reads: one row per target, weighted
for the ones that lost their fiber to a neighbour, cut by the imaging and observing condition
vetoes, and paired with randoms carrying the same selection and the same redshift distribution.

It is a self-contained implementation of the mock arm of `desihub/LSS`, and does not import
`LSS`. Nothing is written between stages and nothing is read back: what the survey pipeline
keeps on disk in between, about 450 GB per realization for one tracer of the dark program, is
held in memory and passed along.

## The stages

Each is a function taking and returning arrays, and each can be used on its own:

| stage | what it makes |
| --- | --- |
| `combine_data` | one row per target per fiber that could have reached it |
| `count_tiles` | how many tiles cover each target, and which |
| `make_full_data`, `make_full_randoms` | one row per target, with what became of it |
| `apply_veto_data`, `apply_veto_randoms` | the same, cut to usable imaging and sky |
| `make_clustering_data`, `make_clustering_randoms` | positions, redshifts and weights |
| `compute_nz`, `add_nz_weights` | the number density and the Feldman-Kaiser-Peacock weight |

## Running

`run_tracer` chains them:

```python
from mockfactory.desi.lsscat import (read_assignments, combine_data, read_good_tilelocid,
                                     read_hpmaps, run_tracer)

assignments = read_assignments(altmtl_dir, tileids, numproc=32)
data = combine_data(potential, assignments, targets=targets,
                    good_tilelocid=read_good_tilelocid(spec_fn))

run_tracer(data, randoms, assignments, 'LRG', targets=targets, hpmaps=read_hpmaps(hpmap_dir, 'LRG'),
           good_tilelocid=good, numproc=18, output_dir=output_dir)
```

`randoms` may be arrays, paths, or a function of the index; the last two keep the parent from
holding eighteen catalogs of eight gigabytes at once.

A sample can be cut further with `data_selection`, any function of the vetoed full catalog returning
a boolean array. The bright galaxies are defined by a magnitude rather than a targeting bit, so
they need one, and a name to be written under:

```python
run_tracer(..., 'BGS_BRIGHT', data_selection=lambda catalog: catalog['R_MAG_ABS'] < -21.5,
           name='BGS_BRIGHT-21.5')

# the second generation samples, whose threshold is a polynomial in redshift
run_tracer(..., 'BGS_ANY', data_selection=absmag_selection(get_bgs_absmag_cut()), name='BGS_ANY-02')
```

It cuts the data. The randoms take their redshift and weight from the selected data, so the cut
reaches them through the redshift distribution rather than directly.

## What makes it fast

The random catalogs are nineteen twentieths of the work at the production eighteen, so that is
where the levers are. Each leaves every catalog and both densities unchanged:

| change | gain |
| --- | --- |
| build the randoms in a forked pool, one worker a random | 1.89x on two workers, 1.70x over the tracer |
| write in processes, not threads | six catalogs in 8.6 s against 40 s, threaded or serial |
| count tiles with an integer lexsort, not `unique` on a structured array | minutes to 85 s over 92 million rows |
| pass tables without copying their columns | `run_tracer` 611 -> 321 s, 38 -> 32 GB peak |

Writing is byte order conversion, not disk: a fits file stores its numbers the other way round
from the machine, and that conversion holds the interpreter lock, so threads buy nothing. The
writes are batched to the end and handed to a pool forked once the catalogs exist, which
inherits them instead of being sent gigabytes. Forking rather than spawning is why every pool is
cheap: a worker inherits the data catalog, the good locations and the maps, and reads its own
random from a path.

**Build every random before finishing any.** Building one costs minutes, finishing it adds `NX`
and `WEIGHT_FKP` in seconds. Finishing needs the number density and the completeness per number
of tiles, measured on the data and on the *first* random -- so the tempting order is build
random 0, measure, build the rest in parallel. That leaves the expensive half of random 0
outside the pool and caps the speed-up at `N / (1 + ceil((N - 1) / numproc))`, which for two
randoms is exactly 1. Build all `N` in the pool, then measure, then finish. The survey pipeline
measures on its first random too, so only the ordering differs.

At eighteen randoms and `numproc=18` that should put a tracer near a quarter of an hour against
a serial 2.1 h. Extrapolated from two measured randoms, not measured; the finishing step still
runs in the parent and would then cost about as much as the building.

## What bounds it

**Memory, not cores.** One bright mock with four randoms peaks at about 37 GB, in the random
phase rather than in the joins before it, so five a node is what fits and six is not: at six the
page cache has nowhere to go and the kernel starts taking workers. Measure the peak, not the
wall clock, and measure it where the peak is -- sizing this from `combine_data` gives 25 GB and
loses mocks.

That peak is set by `numproc_randoms`, which is how many random catalogs are in flight at once
and defaults to `numproc`. Each worker holds one catalog and everything built from it, so
lowering it trades the random stage's wall clock for its memory, one at a time being the
cheapest; the writes stay parallel under `numproc`. With `keep=False` and an `output_dir` each
worker writes its own catalog and hands nothing back, so nothing accumulates in the parent.

The costs in `../README.md` have no production counterpart: no timed run of the survey's own
chain was measured, so none of them is a speed-up against it.

## Things that will bite you

- **Completeness is measured before the healpix map veto**, not after. The other order puts
  `COMP_TILE` and `FRAC_TLOBS_TILES` wrong on most rows.
- **Volumes use the tabulated fiducial cosmology**, not a Boltzmann solve of the same
  parameters; the difference is a relative 1e-5 on every n(z) bin.
- **A random on a set of tiles the data does not have** gets `missing=1.` by default, which is
  what the survey's clustering catalogs carry. It is 2.35% of randoms and 3.44% of the random
  weight, and `'ntile'` is the defensible alternative.

## Validation

Every stage reproduces the DA2 reference exactly, each run on the reference's own input: the
full randoms, the vetoes, the clustering data and randoms including the resampled redshifts and
weights, the galactic cap split, and the densities. The full data catalog reproduces exactly on
every target whose kept row is unambiguous.

`../lsscat_findings.md` has the evidence, four defects this turned up in the reference
pipeline, and the costs.

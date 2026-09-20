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

For the bright galaxies, which are defined by a magnitude rather than a targeting bit, pass the
cut and the name it implies:

```python
run_tracer(..., 'BGS_BRIGHT', absmag_max=-21.5, name='BGS_BRIGHT-21.5')
```

## Things that will bite you

- **The random catalogs are nineteen twentieths of the work.** They are independent, so
  `numproc` builds them in parallel; use one worker per random.
- **Writing is not limited by the disk.** A fits file holds its numbers the other way round
  from the machine, so writing one is mostly byte order conversion under the global interpreter
  lock: threads buy nothing, processes buy a lot. The writes are batched and handed to a pool.
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

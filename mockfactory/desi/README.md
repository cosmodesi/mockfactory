# desi

A mock through the DESI survey, from a light cone to clustering catalogs.

Four stages, each a package of its own with its own README. They are separate because they fail
separately and because the middle two cost hours, so a run is normally restarted somewhere in
the middle rather than from the top.

```
light cone  ->  base  ->  target catalog  ->  altmtl  ->  assignments  ->  lsscat  ->  clustering
                                          \                            /
                                           \-->  altmtl.pota  -------->
```

## The stages

**`base`** turns a cubic box or a light cone into something that looks like DESI: the footprint
it has to fall in (`is_in_desi_footprint`), the imaging quantities of the bricks it lands on
(`get_brick_pixel_quantities`), the redshift error the spectrograph would have made, and a
single pass of fiber assignment. The target catalog it leads to is what everything downstream
starts from: one row per target, carrying `targets.TARGET_COLUMNS` plus the mock truth.

**`altmtl`** replays the real survey against that catalog: the same tiles on the same dates,
with the same sky, focal plane state, hour angles and run dates, but the mock's targets. Which
target wins a fiber then differs, and that is what a clustering measurement has to correct for.
One realization lands as `fba-<tileid>.fits` and `famap-<tileid>.npy` per tile, the tile id zero
padded to six digits. Those are fiberassign's own FITS, not the HDF5 the catalogs use.

**`altmtl.pota`** asks a different question and does not depend on the ledgers at all: for every
tile, which targets each fiber *could* have reached, and which of those only by colliding with
a neighbour. This is the denominator the assignment is measured against, and one file serves a
mock however many realizations it has.

Bitweights are a separate thing and a mock does not normally want them. They stack `nbits`
alternative realizations, 128 by default, into a probability that a target was reached, which
`lsscat` turns into a weight with `completeness='bitweights'`. That means running the replay
128 times rather than once. The default is `completeness='fracz'`, one realization -- `Univ000`
-- with the weight taken from the fraction observed at a fiber location, and everything in this
README is measured that way.

**`lsscat`** turns the two into the catalogs a clustering measurement reads: one row per target,
weighted for the ones that lost their fiber, cut by the imaging vetoes, and paired with randoms
carrying the same selection and the same redshift distribution.

## Tables

`base` and the building of target catalogs run over MPI and pass `mpytools.Catalog`. `altmtl`
and `lsscat` run in one process, with forked pools, and pass `astropy.table.Table`, the type
desitarget and fiberassign already take; `read_targets` is where the two meet. `tables.py`
holds what both single-process packages share: `as_table`, which takes a table, an `mpytools`
catalog or a structured array and copies none of their columns, `set_column`, and the left join
the catalogs are built with. Two things stay structured arrays on purpose: `LedgerState`'s rows,
which are desitarget's merged target list data model and go straight to `make_mtl`, and
`FiberMap`. And `pota` keeps astropy's own `join` rather than that one: astropy orders the
locations of a target by an unstable sort, every potential assignment file so far holds that
order, and the catalogs built from them depend on it through the rows their random draws pick.

A table is only as cheap as the arrays it replaced if it is used with care: add a column with
`set_column`, which does not copy it; drop or keep columns with `remove_columns` or
`keep_columns` on `table.copy(copy_data=False)`, since `table[names]` copies; and read
`table[name].value` in a heavy expression, arithmetic on a `Column` being about twice as slow.
Measured on `run_tracer` for one bright mock with two randoms, against the structured arrays:
321 s instead of 611, and 32 GB at the peak instead of 38.

## Where the bytes go

All per mock, one bright mock of about 17.4 million targets.

| stage | reads | writes | files out |
| --- | --- | --- | --- |
| target catalogs | 6.6 GB | 14.2 GB | 3 |
| `altmtl` | targets, and per tile the survey's fiberassign, sky and secondary files | 17 GB | 5171 |
| `pota` | targets, and per tile the fiberassign header | 8.1 GB | 1 |
| `lsscat` | 58 GB of randoms, the potential assignments, every `fba` file, the spectroscopic table | 24 GB | 17 |

Target catalogs are written three times, 7.3 GB unmasked, 4.0 GB masked and 2.9 GB cut to the
tiles. The `altmtl` output is 5171 `fba` files of about 3 MB each, and as many `famap` files of
about 66 KB. `lsscat` reads one legacy survey
random file per random at 14.5 GB, four of them here against production's eighteen, and its
output is quoted at those four.

## What it costs, for the bright program

Twenty five `BGS_ANY-02` mocks of about 17.4 million targets each, 5171 tiles a mock, on 2
Perlmutter CPU nodes throughout. Wall clock, not core hours; the last column is the third
divided by 25, so the four stages compare directly.

| stage | layout | 25 mocks | per mock |
| --- | --- | --- | --- |
| target catalogs | 25 then 250 ranks | 11 min | 26 s |
| `altmtl` | 13 and 12 mocks, `numproc=9` | 1 h 40 | 4.0 min |
| `pota` | 8 streams of 3 mocks, `numproc=30` | 50 min | 2.0 min |
| `lsscat` | five a node, `numproc=4` | 1 h 25 | 3.4 min |
| **total** | | **4 h 6** | **9.8 min** |

`altmtl` and `pota` each cover 129 275 tiles, `lsscat` uses four randoms. Two of the figures
were measured on a smaller run and scaled: `pota` at 8 min for 3 mocks on one stream, `lsscat`
at 27 min 51 for five on one node. One-off and shared by every run afterwards, converting the
DR9 bricks with `scripts/convert_bricks_to_h5.py` costs 24 min on 2 nodes for 347 206 bricks
and 138 GB.

Inside `lsscat`, over one mock of the five:

| step | wall clock | share |
| --- | --- | --- |
| the four randoms' imaging and tile counts | 418 s | 25% |
| full data catalog, joins and vetoes | 472 s | 28% |
| the first random, alone | 176 s | 11% |
| randoms 1 to 3 together, with the writes | 469 s | 28% |
| everything else | 128 s | 8% |
| **total** | **1663 s** | |

The first step depends on the randoms and not on the mock, and is computed once per process, so
a step handling several mocks pays it once where separate steps each pay it in full.

To scale from: `altmtl` runs at about 22 tiles a second on two nodes, `pota` at about 32 a
second a stream.

## What `altmtl` costs, for the dark program

The `AbacusHF_DR2v2` mocks: 41.7 million targets each, 6671 tiles and 13 610 actions a mock,
replayed to 20240418 with each mock's own quasar redshifts (`zfix`). One Perlmutter CPU node.

| layout | wall clock | per mock | peak memory |
| --- | --- | --- | --- |
| 1 mock, `numproc=32` | 52 min | 52 min | 166 GB as Slurm counts it |
| 12 mocks, `numproc=9` | 1 h 51 | 9.3 min | 435 GB of 503 |

Both are measured end to end. The single mock takes 82 s to build its state, then 51 min of
replay. The twelve take 6680 s, of which 301 s pass before the first tile while their states
build, so the replay itself is 1 h 46, a shade over twice the single mock's. That is 12.5 tiles
a second for the node against 2.2 for the mock alone, and 11 for bright at 12 or 13 a node.
Run it with `scripts/run_altmtl.py --imocks 0-11 --numproc 9 --nummocks 12`.

Memory is what sets twelve as the ceiling, and the peak is the **end** of the replay, not the
start. The twelve states build to 269 GB in the first ten minutes, and from there the node
climbs steadily -- 310 GB at a quarter of the way, 365 at two thirds, 405, and 435 GB with 68 GB
to spare in the last few minutes. So the headroom to watch is at the finish, and a thirteenth
mock would have nothing to do with whether the states fit. Slurm's figure for the single mock
sums the resident pages of every forked worker, which share the state, so it overstates what
one mock takes.

`srun` drops the tail of the forwarded stdout when a long step shuts down --
`eio_handle_mainloop: Abandoning IO 60 secs after job shutdown initiated` -- so the per-mock
summaries `run_altmtl.py` logs at the end can be missing from a redirected log while the run
itself completed. Count the `fba-*.fits` files before reading anything into a short log.

# LSS catalogs from an altMTL mock: findings

Notes on `mockfactory.desi.lsscat`, a self-contained implementation of the catalog arm of
`desihub/LSS` for mocks: the stages that turn an alternative merged target list run into the
clustering catalogs a measurement reads. Written for whoever runs or maintains DESI mock
catalog production.

Status as of 2026-09-19. Branch `adematti-dev`.

## What it does

`mockfactory.desi.altmtl` replays the survey against a mock and says which target won each
fiber. This takes it the rest of the way: the potential assignments and the fiber assignments
become a catalog with one row per target, weighted for the targets that lost their fiber to a
neighbour, cut by the imaging and observing condition vetoes, and paired with randoms carrying
the same selection and the same redshift distribution.

It does not import `LSS`. The stages follow the survey pipeline's and each is a function
taking and returning arrays: `combine_data`, `make_full_data`, `apply_veto_data`,
`make_clustering_data`, `make_clustering_randoms`, `compute_nz`, `add_nz_weights`.

## Relation to desi-clustering

Three of the weights already exist in `desi-clustering/clustering_statistics/tools.py`, in a
more general form than LSS uses, and the implementations here follow it rather than the survey
pipeline:

- `compute_iip_weight` is `_compute_iip_weight`, one over the observed fraction with the counts
  offset by one. LSS hardcodes `129/(1+128*PROB_OBS)`; here the number of realizations is an
  argument, since an altMTL run is free to use more or fewer than 128.
- `get_photsys` and `get_galactic_cap` are the north/south and galactic cap parts of
  `select_region`, which supersedes LSS `splitGC` and the scattered inline cuts.
- `compute_completeness_per_ntile` is `_compute_binned_weight` and `get_binned_weight`: the
  mean weight over the objects covered by the same number of tiles.

`_compute_missing_power` has no counterpart here yet. It is the natural thing to offer
alongside `completeness='bitweights'`, being the weighting that pairs with PIP.

## Validation

Against `DESI_ROOT/survey/catalogs/DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl0/mock0`,
luminous red galaxies, random 0. Each stage is run on the reference's own input, so that the
stages are judged separately.

| stage | result |
| --- | --- |
| `_full_noveto.dat.fits` | exact on every column, for every unambiguous target |
| `_full_noveto.ran` to `_full_HPmapcut.ran` | exact, 25412976 rows, every column |
| `_full_noveto.dat` to `_full_HPmapcut.dat` | exact on every column, 3 rows of 4351500 differing |
| `_full_HPmapcut.dat` to `_clustering.dat` | exact on every column |
| `_full_HPmapcut.ran` to `_clustering.ran` | exact, including the resampled redshift and weights |
| galactic cap split | exact, 1320216 north and 773342 south |
| `_nz.txt` | exact: density, counts and volume all at zero relative difference |

The clustering randoms matching bit for bit is worth stating plainly: the redshift and weights
a random is given are drawn at random from the data, and reproducing them means the draw runs
in the same order against the same generator.

Three things had to be found by experiment rather than read off the source.

**Completeness is measured before the healpix map veto, not after.** `apply_veto` computes
`COMP_TILE` and `FRAC_TLOBS_TILES` and returns, and the driver applies the map veto to what it
returned. Doing it the other way round puts both columns wrong on 62% of rows. It is also the
better order: the maps remove whole patches of sky, and a set of tiles straddling the edge of
one would otherwise be judged on whichever part of it survived.

**The volume must use the tabulated fiducial cosmology, not a Boltzmann solve of the same
parameters.** That alone was a relative 8e-6 on every n(z) bin.

**`subfrac`, the draw that brings a mock down to the density the real survey found, is taken
with an unseeded `np.random.random`.** The DA2 reference kept 0.976 of the luminous red
galaxies, and no rerun of it can produce the same catalog. Seeded here.

## Cost

Measured on one Perlmutter cpu node, one process, `OMP_NUM_THREADS=1`, luminous red galaxies,
one random catalog:

    read inputs                                36 s
    full data        (tile counts 55 s)        76 s
    vetoes                                     13 s
    clustering data                             2 s
      data arm                                 91 s
    full randoms     (tile counts, imaging)   251 s
    random vetoes                              74 s
    clustering randoms                         56 s
      building one random                     381 s
    n(z), FKP, galactic cap split              17 s
    finishing and writing one random           60 s

Both the building and the writing run in a forked pool. Two randoms, written out, against the
same run at `numproc=1`:

    data arm                     89 s ->  90 s    serial by nature
    building two randoms        693 s -> 366 s    1.89x on two workers
    n(z)                         12 s ->   9 s
    finishing and writing       126 s ->  76 s    1.66x
                               ------    ------
                              919.5 s   541.0 s   1.70x

Every one of the eleven catalogs written agrees between the two, and so do both densities.

Writing is worth its own note, because it is not what it looks like. A fits file holds its
numbers the other way round from the machine, so writing one is mostly byte order conversion,
and that runs under the global interpreter lock. Measured on a compute node, six threads write
six catalogs in 40.1 s against 40.0 s for one thread, exactly nothing; six processes take 8.6 s.
At 0.11 GB/s the disk is nowhere near the limit. So the writes are batched to the end of the
run and handed to a pool forked once the catalogs exist, which inherits them rather than being
sent two and a half gigabytes apiece.

Extrapolating to the production eighteen randoms at `numproc=18`, building falls from about
1.7 h to some 6 minutes and writing from 19 to 5, which would put a tracer near a quarter of an
hour against a serial 2.1 h. That step is an extrapolation from two measured randoms. What it
leaves as the next thing worth attention is the finishing itself, which still runs in the
parent and would then cost about as much as the building.

Counting the tiles over a target is the largest item on both arms. It is one of the two Python
loops mentioned below, and vectorizing it means sorting a hundred million rows: done over a
structured array, which is what a `unique` on two columns gives, numpy falls back to a generic
element comparison and it costs minutes. The integer lexsort it was replaced with brought the
ninety two million row pass to about eighty five seconds.

The random catalogs are nineteen twentieths of the work at the production eighteen, and they
are independent, so `run_tracer` takes `numproc` and builds them in a forked pool. Each draw is
seeded by the index of its catalog, so the result does not depend on how many run at once or in
what order they come back.

They are not quite independent to the end. The number density and the completeness per number
of tiles are measured on the data and on the first random catalog, and every catalog needs them
for its `NX` and `WEIGHT_FKP`. Measuring them first and building the rest afterwards is the
obvious reading, and it is the wrong one: it leaves the first catalog outside the pool and caps
the speed-up at `N / (1 + ceil((N - 1) / numproc))`, which for two catalogs is exactly one. So
all of them are built first, in parallel, and then finished, which takes seconds against the
minutes building one costs. Measuring on the first random catalog is what the survey pipeline
does too, reading `_0_clustering.ran`, so this is its ordering rather than a new assumption.

The pool is forked rather than spawned, so a worker inherits the data catalog, the good
locations and the maps instead of having them pickled, and it reads its own random catalog when
given a path rather than an array. At eighteen catalogs of eight gigabytes each, holding them
all in the parent is not an option.

### What is not written

Nothing between stages. The only reads are the survey's own inputs, the per tile assignments of
the altMTL run, the observing condition maps, the imaging columns of the parent randoms and,
when a path is given rather than an array, a random catalog itself; the only writes are the
clustering catalogs and the densities. Measured on the reference tree, for the dark program and
one realization, that is what the survey pipeline writes and reads back and this does not:

    combined tables, assignments, tile counts, collisions      23.6 GB
    rancomb, eighteen randoms                                 149.0 GB
    full catalogs before and after the vetoes, LRG alone       275.1 GB
                                                        ---------------
    one tracer                                                447.7 GB

The first two are shared between tracers and the third is not, so the three dark tracers cost
about a terabyte a realization. The clustering catalogs both keep, 85.4 GB for the luminous red
galaxies, are the only thing written here.

## Three defects worth reporting upstream

### The kept row among tied candidates comes from an unstable sort

`mkfulldat` keeps one row per target by building a `sort` column, calling `Table.sort('sort')`
and then `unique(keys=['TARGETID'], keep='last')`. astropy's single key sort is numpy
quicksort, which is unstable, so which row survives among those tied at the top of the ranking
is not determined by the inputs.

About a quarter of targets have such a tie, 24% for the DA2 luminous red galaxies. It is not
cosmetic: the kept row fixes the fiber location the target is charged to, hence
`FRACZ_TILELOCID` and so `WEIGHT_COMP`. Measured against the reference, `FRACZ_TILELOCID`
differs on 2.9% of targets with an rms of 0.085, and the mean completeness weight moves by
**+0.16%**. `COMP_TILE` then differs on 25% of rows purely by inheritance, being a mean over
everything sharing a set of tiles.

The ranking itself is sound: the reference's kept row has the maximum `sort` 99.9997% of the
time. Only the tie is loose. `last_of_each` takes a `tie` argument and settles it on the
smallest `TILELOCID`, which does not depend on how the combined table was assembled.

Figure: `/pscratch/sd/a/adematti/claude/lsscat/full_LRG.png`.

### A random on an unmatched set of tiles gets two different answers

`FRAC_TLOBS_TILES` is looked up per `TILES` string. A few per cent of randoms carry a set of
tiles that no data target of the tracer has, because an object drops from its set the tiles
whose fiber location was unusable and which those are depends on the location. LSS resolves
this twice and differently: the full random path leaves 0, and `add_tlobs_ran_array` catches
the lookup failure and fills 1. The two files then disagree on the same rows.

On DA2 luminous red galaxies, random 0: 596533 of 25412976 rows, **2.35%**, spread over one to
seven tiles, carrying **3.44% of the total random weight**. Zero throws them away, one grants
them full completeness, and the random weight is what normalises the selection function.

`add_frac_tlobs` takes `missing=`: a number, or `'ntile'` for the mean data completeness at the
same `NTILE`, which is the defensible middle.

### Two Python loops over tens of millions of rows

`count_tiles_input` builds each target's list of tiles in a loop over the distinct targets, 29
million of them for a dark time survey. `mkfulldat` builds `FRACZ_TILELOCID` in a loop over the
distinct fiber locations and then reads it back in a second loop over the rows. Both are
vectorized here, the first by cutting the groups out of a sorted array and building the names
in as many passes as the largest number of tiles over a target.

### The reference catalogs sit at positions that belong to no available target file

`DA2/mocks/SecondGenMocks/AbacusSummit_v4_1/altmtl0` is internally inconsistent about where its
targets are. `forFA0.fits` and `altmtl0/fba0/datcomb_darkwdup.fits` agree exactly on right
ascension and declination. `altmtl0/mock0/datcomb_dark_tarspecwdup_zdone.fits` and every catalog
under `altmtl0/mock0/LSScats/` agree exactly with each other and with neither, the median
separation being about 100 degrees, and none of `forFA0.fits` through `forFA24.fits` matches
them.

The rows themselves line up: `(TARGETID, TILEID, LOCATION)` matches one to one between the two,
and all 104994096 rows of the reference are found in the potential assignments. Only the
positions differ. Checked by direct lookup of individual identifiers rather than by a join,
because `forFA` orders its identifiers spatially and a handful of scattered ones legitimately
land in one patch.

Timestamps put `forFA0.fits` at 2024-07-12, `altmtl0/mock0/datcomb` at 2024-07-23, the
clustering catalogs at 2024-07-26 and `altmtl0/fba0` at 2024-10-13. The directory regenerated
last is the one that agrees with the oldest file.

The consequence is that the clustering of those published catalogs is not the clustering of the
mock whose assignment they encode. It is also why the step from potential assignments to the
combined table is the one stage here that could not be checked against the reference: there is
no consistent pair to check it with.

## Unresolved

The join from the potential assignments to the combined table, for the reason just given. Its
mechanics are exercised: every one of the reference's 104994096 rows is found and carries the
right assignment, and the row set differs only by the good hardware cut, whose exact definition
needs the survey's bad fiber and bad petal night tables.

Three rows of 4351500 differ in the map veto, 7e-7. The healpix pixels agree, so it is not the
angle conversion; one of the three is a target the reference kept at `PSFSIZE_G` 2.4355 against
a cut of 2.4, which should have removed it. The likeliest explanation is that the map files
were revised after the reference was produced, `jura-v1` carrying only one version now.

## The bright program

The bright arm is set up but not yet run. What was checked, against the real survey inputs
rather than by reading the source:

- the priorities come out right: `BGS_BRIGHT` is priority 2100 with two observations,
  `BGS_FAINT` 2000, both carrying `BGS_ANY` in `DESI_TARGET` and their own bit in `BGS_TARGET`;
- a bright tile tracker builds for the DA2 end date, 10442 actions over 5171 tiles, 5171 `fa`,
  5171 `update` and 100 `reproc`, and **nothing outside those three**, which is the gate,
  because the loop carries out no other kind;
- the per-tile inputs exist for the bright tiles, footprint, sky and secondary, with no targets
  of opportunity;
- so do the shared inputs the catalog stage needs: the bright spectroscopic table, the
  `BGS_BRIGHT` observing condition maps and the bright tile list;
- and there is a reference to validate against, laid out exactly as the dark one:
  `DA2/mocks/SecondGenMocks/AbacusSummitBGS_v2` with `forFA{n}.fits`, `mock{n}/pota-BRIGHT.fits`
  and `altmtl{n}/Univ000`.

Building the bright tile tracker takes about eleven minutes, against seconds for the dark one.
That is worth a look before a production run.

The parameters resolve: redshift range 0.1 to 0.5, maximum priority 2100, mask bit 11, and the
Feldman-Kaiser-Peacock amplitude 7000 in bins of 0.01. Two things had to be added. A bright
galaxy sample is defined by an absolute magnitude rather than by a targeting bit, so
`make_clustering_data` takes a `data_selection`, handed the vetoed full catalog and returning a
boolean array, and `run_tracer` takes a `name` so the catalogs can be written under the variant
the cut defines, such as `BGS_BRIGHT-21.5`. `absmag_selection(get_bgs_absmag_cut())` is that
sample's cut on `R_MAG_ABS`; any other column of the full catalog can be cut on the same way. The photometry the cut needs, `R_MAG_ABS` and the colours beside it, is now
carried through to the clustering catalog.

## Not yet exercised

- Tracers other than the luminous red galaxies, and the bright program.
- The completeness from bitweights, `completeness='bitweights'`, which is what an altMTL run
  over many realizations is for and which the reference catalogs do not use: they are built
  with the default, one over the fraction observed at the fiber location.
- The imaging systematic weights, which are fitted rather than derived and are left at one.

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

**The implementation is taken to be correct.** Two productions, DA2 bright and DA2 dark, over 13
realizations: every difference from production's own replays has been traced to a cause outside
this code, one in the fiberassign build and one in `LSS`. Given the same software and inputs the
replay is bit for bit identical to production, and where it differs it is the side that follows
the rules `desitarget` publishes. Nothing in the comparison is unexplained. The sections below
are how that was established, including the readings that turned out to be wrong; the claim
covers assignments and merged target list state, not end-to-end clustering statistics.

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
    reprocs      0   129     0     0     0     0     4   135
    assignments 310  1314  1136   470   669   707  1076   989

A fixed per-tile perturbation compounding to a steady state looks like this, and drift in the
replay would keep climbing instead of plateauing.

The last two rows are the reprocessing actions and the assignments falling in each eighth of
survey time, and they answer the obvious competing explanation: the position bug below, which
is live in the reference run and fixed here, so it is a second uncontrolled difference between
the two. Three things say it is not what this residual is.

- The first eighth holds no reprocessing at all and already disagrees by a median of 3. The
  first `reproc` is action 2452 of 13610, 18.0% through, with 1262 of the 6671 assignments
  before it, on 2021-12-14. It lands in the second eighth under every reading of the word:
  eighths of assignments, of all actions, or of calendar time. So there is a floor the bug
  cannot have caused.
- The third eighth rises from 5 to 7 with no reprocessing in it. A rise of that size is
  produced by whatever causes the floor, unaided.
- The eighth eighth holds 135 reprocs, half of all of them, and the median does not move.

The reprocessing is not spread through the run but clustered in two bursts, which is what makes
the profile informative rather than merely consistent. Figure:
`/pscratch/sd/a/adematti/claude/altmtl/diff_vs_time.png`.

**Resolved, down to the line of C++.** The reference's own stack is installed:
`desiconda/20240425-2.2.0` carries fiberassign 5.7.2, 5.8.0 and 5.9.0 side by side against
desimodel 0.19.1, desimeter 0.7.1, python 3.10.14 and numpy 1.22.4, the last two matching the
reference headers exactly. Replaying the first eighth of survey time, to 2021-09-25, is 304
assignments with no reprocessing, so the position bug below cannot contribute. Against the
official assignments:

    arm                                             median diff   max   identical   agreement
    reference stack, fiberassign 5.7.2                0 of 5020    0       304       1.0000
    reference stack, fiberassign 5.8.0                3 of 5020   10        16       0.9994
    reference stack, fiberassign 5.9.0                3 of 5020   10        16       0.9994
    current stack,   fiberassign 5.9.0.dev3738        3 of 5020   10        16       0.9994
    fiberassign 5.9.0, stuck sky computed here        3 of 5020   10        16       0.9994
    fiberassign 5.9.0, FIBERASSIGN_USE_FABS=1         0 of 5020    0       304       1.0000

The cause is one line of `src/hardware.cpp`, in `Hardware::position_xy_bad`, the inner keepout
test. 5.7.2 reads

    double r_min = ::abs(loc_theta_arm.at(loc) - loc_phi_arm.at(loc));

and which overload that `::abs` picked was never decided by fiberassign at all. PR #470, which
fixed it, says so outright: under gcc 11.2 (desi 23.10) it took the integer overload and
`::abs(-0.1718)` came out 0, while under gcc 13 (desi 24.4) it resolved to the double. The same
source, built twice, assigns differently.

5.8.0 made the choice explicit behind `FIBERASSIGN_USE_FABS` -- unset or 1 gives `::fabs`, 0
gives `::abs((int)r_min)`, labelled in the source as "the old 'buggy' behavior" -- and then
`parse_assign` picks the value from the **rundate** through `get_fba_use_fabs`, with
`data/cutoff-dates.yaml` mapping everything before 2025-05-12 to 0.

The rundate is the wrong variable. What decides the behaviour is the toolchain that built the
fiberassign which produced the assignments being reproduced, and that has no relation to when
the tile was observed. The DA2 mock references make the gap concrete: their tiles have 2021
rundates, so the table asks for the integer path, but the header of
`altmtl0/Univ000/fa/MAIN/20210514/fba-001000.fits` records `DEPVER21 =
.../desiconda/20240425-2.2.0/...` and the file was written on 2024-07-17. They were assigned by
a **gcc 13** build and carry the double behaviour.

The run agrees. Under 5.9.0 with the flag forced to 1 the replay reproduces the official
assignments **exactly**, all 304 tiles, all 1526080 locations, bit-identical to the 5.7.2 arm;
with the rundate default of 0 it is 987 locations away. Our 5.7.2 arm comes from the same
desiconda 20240425-2.2.0, which is why it needs no flag.

It is not a small difference. In the 2021 focalplane `|LENGTH_R1 - LENGTH_R2|` averages
0.084 mm and only one positioner of 5016 exceeds 1 mm, so the integer cast sends the inner
keepout radius to 0 for essentially every positioner.

**We do not pass `--fba_use_fabs`, and that is the deliberate choice.** The rundate default is
what the *real survey* got: those tiles were assigned in 2021 by an old fiberassign under an old
desiconda, so the integer path is the behaviour the data carries, and a mock exists to mimic the
data. The SecondGen references ran under gcc 13 and so got the numerically correct treatment of
`r_min` -- correct, but not the behaviour applied to the data they are a mock of. Reproducing
them bit for bit would mean inheriting that.

So the 0.06% here is a real difference between the references and a faithful replay, and it sits
on their side. Bit-exactness against `altmtl0` is available on demand -- `--fba_use_fabs 1`, or
simply running under desiconda 20240425-2.2.0 as the sections above do -- and is what the
validation table measures. It is a check that the implementation agrees with production given
the same inputs, not the configuration to generate mocks with.

Note that setting the environment variable does not do it: `run_assign_init` overwrites
`FIBERASSIGN_USE_FABS` from the rundate, so the option has to go through `parse_assign`.

Two other candidates were tested and are not involved.

- Not the coordinate transform. desimeter 0.7.1 and 0.9.0.dev1218 place the 4201 targets of
  tile 1000 at identical focal plane positions, to 0.00000 mm against a 6 mm patrol radius, and
  the `use_hardcoded_polmis_rotmat` argument 5.9.0 passes does not exist in desimeter 0.7.1.
- Not our own `--fafns_for_stucksky`. That option exists only from 5.8.0, so the 5.7.2 arm
  works the stuck-sky positioners out itself and every other arm reads them off the real
  survey's assignment -- a second variable moving with the version. Forcing it off leaves the
  disagreement exactly where it was, 0.9994.

Nothing else in 5.7.2..5.8.0 can reach a 2021 rundate: `obsthetacorr` (the hexapod correction,
PR #466) defaults to False, `obsdate` still resolves to the historic 2022-07-01 below its 2025
cutoff, `fieldrot_corr` is False, `stuck_on_sky_from_fafns` is purely additive, and
`freeze_iers` lands only in `bin/sv1-summary.py`. The one ungated change, `LGE` joining
`default_main_sciencemask()`, touches nothing here: of the 42253356 targets in `forFA0.fits`,
none carries that bit.

The measurement covers the first eighth only. Whether the later eighths, where the reprocessing
bursts fall, are also fully explained by this is untested; the 99.88% quoted above is for the
whole replay and mixes it with the position bug.

Run by `claude_bgs_altmtl_new/run_fa_version_test.sh` and `compare_fa_versions.py`.

### A second production, and it settles the question

`AbacusHighFidelity/altmtl0` is an independent reference: different targets, written 2025-11-27
against SecondGen's 2024-07-17, and it exercises the redshift fix SecondGen has no equivalent of
-- 3988255 quasars taking their own redshift on every update, from `qso.txt`. Its header names
the same fiberassign, `5.7.2.dev3588`, so the same arm should reproduce it. It does not:

    arm                          tiles   median diff   max   identical   agreement
    fiberassign 5.7.2             304       3 of 5020    9        17      0.999363
    fiberassign 5.9.0             304       0 of 5020    0       304      1.000000

Exactly backwards from SecondGen, and with the same signature. The header records the version
string and the `DESIMODEL` path; it does not record the compiler, and the compiler is what
decides this. Two builds of plain 5.7.2 are installed here, and disassembling
`Hardware::position_xy_bad` in each shows the whole story in one instruction:

    desiconda 20230111-2.1.0    cvttsd2si    truncate the double to an int -- the integer ::abs
    desiconda 20240425-2.2.0    andpd        clear the sign bit -- ::fabs

Same source, same version, two instructions. So the two productions disagree with each other:
SecondGen carries the double behaviour, AbacusHF the integer one, and **no single configuration
reproduces both**. Which is the right one to follow is not a matter of taste -- the integer path
is what the real survey's own assignments have, so AbacusHF is the one a mock should match, and
it is what the rundate default gives. `run_fiberassign` passes nothing, and reproduces
AbacusHF bit for bit under a current fiberassign, all 304 tiles of the first eighth.

That run also puts the redshift fix through its first test against a reference: bit-exact with
the substitution live.

Run by `claude_bgs_altmtl_new/run_abacushf_validation.sh` and `compare_abacushf.py`.

### Twelve mocks, the whole survey: 0.0007% of locations, and it is not the keepout radius

The sections above stop at the first eighth, where the residual is exactly zero once the build
matches. Twelve `AbacusHF_DR2v2` mocks replayed end to end -- 6671 tiles and all 13610 actions
each, on one node, under the current stack -- compared against their own references, which keep
no per-tile files, only the assigned `(TILEID, LOCATION, TARGETID)` of
`fba<i>/datcomb_darkassignwdup.h5`:

    mock              0    1    2    3    4    5    6    7    8    9   10   11
    locations       173  217  215  188  239  209  193  200  196  222  135  221
    tiles identical 6533 6516 6529 6552 6508 6515 6529 6517 6523 6528 6562 6512

Mean 201 locations of 28222946 assigned, 0.00071%, agreement 0.9999929, with 97.8% of tiles
identical and a median of 0 differing per tile. Mock 0 reproduces an earlier independent run of
the same comparison to the location, so this is stable, not scatter.

Small, but not zero, and not the `r_min` difference: the first eighth is clean under this
stack, and these differences fall in all eight eighths at much the same rate rather than
tracking the reprocessing bursts. Two features point elsewhere. The per-tile maximum is 3 to 6
locations, an order below the 10 the keepout radius produces. And 18 to 39 of the targets
involved in each mock are survey targets absent from that mock's `forFA`, which a mock replay
should not be able to assign at all. That is the thread to pull, and it is unexplained.

Run by `mockfactory/desi/scripts/run_altmtl.py` and
`claude_abacushf_altmtl/compare_assignments.py`, driven by `run_twelve.sh` there.

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
`update` never writes positions, and a run of 11.6 million targets through `fa` and `update`
came back with not one position altered. `reprocess_ledger` does write them: both loops that
copy the catalog back run over every column with no exclusion for `RA` and `DEC`, and the rows
added back for bad observations come straight from the catalog.

**That "harmless in an update" reading was wrong, and the section below has the correction**: an
update does not write positions but it *reads by* them, to decide which healpix ledgers to open,
so the same displacement makes updates go missing.

Measured on tile 1006: 440 of 3623 observed targets displaced, by up to 0.039 degrees, which
is the fiber patrol radius. Two crossed a healpix boundary, were appended to a second ledger,
and left the ledgers holding one target twice at two positions, 45617 rows for 45615 targets.

This is live in production. runHoli copies a frozen tracker holding 268 reprocessing actions
out of 13610, so every realization hits it: order 100000 displaced positions per realization,
all on observed targets, which is the clustering sample.

Fixed here by substituting the positions along with the identifiers, after which none is
altered. The ledger path has no state to look them up in and warns instead.

### The same displacement silently drops updates, and that is the residual

`update_alt_ledger` calls

    update_ledger(althpdirname, altZCat, obscon=obscon.upper(),
                  numobs_from_ledger=numobs_from_ledger, tabform='ascii.ecsv')#, targets = targets)

at `py/LSS/main/mockaltmtltools.py:1287`, with `targets` commented out -- four lines after a
`raise ValueError('If processing mocks, you MUST specify a target file')`, so it is in hand. The
equivalent calls on the real path, `py/LSS/SV3/altmtltools.py:1674,1676,1680`, all pass
`targets=targets`. That argument is what decides the branch in `desitarget.mtl.update_ledger`:

    if targets is None:
        nside = _get_mtl_nside()                      # 32
        theta, phi = np.radians(90-zcat["DEC"]), np.radians(zcat["RA"])
        pixnum = list(set(hp.ang2pix(nside, theta, phi, nest=True)))
        targets = io.read_mtl_in_hp(hpdirname, nside, pixnum, ...)

The ledger rows are found from the positions in the zcat, and those are the real survey's. A
mock target is therefore looked for where the real target it replaced sits, up to a patrol
radius away. When that lands in a healpix the tile's real positions never touch, `read_mtl_in_hp`
does not return it, `match` finds nothing, `make_mtl(trimtozcat=True)` returns nothing for it,
and **the observation is silently lost**: the target keeps `UNOBS` priority for the rest of the
survey and wins fibers it should have lost.

The per-tile pixel set is what makes this rare rather than common. 1.27% of assigned fibers have
their mock and real positions in different nside-32 pixels, but `pixnum` is the union over the
whole tile, so almost all are rescued by another fiber's real position falling in the same pixel.
Only **0.0018%** land in a pixel the tile touches nowhere else -- 3 of 169189 assigned fibers
over 40 tiles. Against the 0.0007% of locations that actually differ, that is the right order:
not every lost update is ever contested again.

Verified on the case dissected in the README. Target 220453062 took location 416 of tile 1846,
where the real survey's target sits 0.0065 degrees away -- pixel 3075 for the mock, 3081 for the
real. Its observation was archived on 2021-06-23 and never reached the ledger, and the
reference's own `datcomb_darkassignwdup.h5` records it at `PRIORITY 3100` on tile 11196 four
months later, still unobserved, where it took a fiber from an unobserved target of the same
class. We demote it, as `desitarget`'s priority table says to, and assign that fiber elsewhere.

Fix upstream: either uncomment `targets = targets`, or substitute the mock positions into
`altZCat` as this implementation does for reprocessing. Nothing to fix here -- `LedgerState` is
indexed by `TARGETID` and never consults a position to find a target.

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

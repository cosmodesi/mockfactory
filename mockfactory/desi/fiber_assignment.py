"""
Script to apply fiber assignment from pre-loaded catalog.

This example can be run with `srun -n 5 python fiber_assignment.py` (will take typically 1 minutes for 1pass and minutes for 2pass),
but one will typically import:

```
from mockfactory.fiber_assignment import apply_fiber_assignment
```

For an example, see desi/from_box_to_desi_cutsky script.
"""

import os
import logging

import fitsio
import numpy as np
import pandas as pd

from mpi4py import MPI


logger = logging.getLogger('F.A.')


def _build_tiles(release_tile_path='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits',
                 surveyops_tile_path='/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-main.ecsv',
                 program='dark', npasses=7):
    """ Load tiles properties from surveyops dir selecting only tiles in the desired release/program with enough pass."""
    from desitarget.targetmask import obsconditions

    # Load tiles from surveyops directory
    tiles = pd.read_csv(surveyops_tile_path, header=18, sep=' ')
    # Load tiles observed in the considered data release
    tile_observed = fitsio.FITS(release_tile_path)[1]['TILEID'][:]

    # keep only tiles observed in the correct program with pass < npasses
    # PASS from 0 to npasses - 1!
    tiles = tiles[np.isin(tiles['TILEID'], tile_observed) & (tiles['PROGRAM'] == program.upper()) & (tiles['PASS'] < npasses)]

    # add obsconditions (need it in fiberassign.tiles)
    tiles["OBSCONDITIONS"] = obsconditions.mask(program.upper())

    return tiles


def _run_assign_init(args, tiles, targets, plate_radec=True, use_sky_targets=True):
    """
    Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py#L281

    Instead of reading files, use preloaded targets and tiles.
    """
    from fiberassign.hardware import load_hardware

    def convert_tiles_to_fiberassign(args, tiles):
        """ Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/tiles.py.
            Do not read the tiles, but take it as an array..."""
        import warnings
        from desimodel.focalplane.fieldrot import field_rotation_angle
        import astropy.time
        from fiberassign._internal import Tiles

        # astropy ERFA doesn't like the future
        warnings.filterwarnings('ignore', message=r'ERFA function \"[a-z0-9_]+\" yielded [0-9]+ of \"dubious year')

        if args.obsdate is not None:
            # obstime is given, use that for all tiles
            obsdate = astropy.time.Time(args.obsdate)
            obsmjd = [obsdate.mjd, ] * tiles.shape[0]
            obsdatestr = [obsdate.isot, ] * tiles.shape[0]
        elif "OBSDATE" in tiles.names:
            # We have the obsdate for every tile in the file.
            obsdate = [astropy.time.Time(x) for x in tiles["OBSDATE"]]
            obsmjd = [x.mjd for x in obsdate]
            obsdatestr = [x.isot for x in obsdate]
        else:
            # default to middle of the survey
            obsdate = astropy.time.Time('2022-07-01')
            obsmjd = [obsdate.mjd, ] * tiles.shape[0]
            obsdatestr = [obsdate.isot, ] * tiles.shape[0]

        # Eventually, call a function from desimodel to query the field
        # rotation and hour angle for every tile time.
        if args.fieldrot is None:
            theta_obs = list()
            for tra, tdec, mjd in zip(tiles["RA"], tiles["DEC"], obsmjd):
                th = field_rotation_angle(tra, tdec, mjd)
                theta_obs.append(th)
            theta_obs = np.array(theta_obs)
        else:
            # support scalar or array args.fieldrot inputs
            theta_obs = np.zeros(tiles.shape[0], dtype=np.float64)
            theta_obs[:] = args.fieldrot

        # default to zero Hour Angle; may be refined later
        ha_obs = np.zeros(tiles.shape[0], dtype=np.float64)
        if args.ha is not None:
            ha_obs[:] = args.ha

        return Tiles(tiles["TILEID"].values, tiles["RA"].values, tiles["DEC"].values, tiles["OBSCONDITIONS"].values, obsdatestr, theta_obs, ha_obs)

    def convert_targets_to_fiberassign(args, targets, tiles, program, use_sky_targets=True):
        """ Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py#L281.
            Do not read the tiles, but take it as an array..."""
        from fiberassign.targets import Targets, create_tagalong, load_target_table
        from fiberassign.fba_launch_io import get_desitarget_paths
        from desitarget.io import read_targets_in_tiles

        # Create empty target list
        tgs = Targets()
        # Create structure for carrying along auxiliary target data not needed by C++.
        tagalong = create_tagalong(plate_radec=plate_radec)

        # Add input targets to fiberassign Class objects
        load_target_table(tgs, tagalong, targets, typecol=args.mask_column,
                          sciencemask=args.sciencemask, stdmask=args.stdmask, skymask=args.skymask,
                          safemask=args.safemask, excludemask=args.excludemask, gaia_stdmask=args.gaia_stdmask,
                          rundate=args.rundate)

        if use_sky_targets:
            # Now load the sky target files.  These are main-survey files that we will
            # force to be treated as the survey type of the other target files.
            mydirs = get_desitarget_paths('1.1.1', 'main', program, dr='dr9')
            skydirs = [mydirs["sky"]]
            if os.path.isdir(mydirs["skysupp"]):
                skydirs.append(mydirs["skysupp"])
            columns = ["RA", "DEC", "TARGETID", "DESI_TARGET", "BGS_TARGET", "MWS_TARGET", "SUBPRIORITY", "OBSCONDITIONS", "PRIORITY_INIT", "NUMOBS_INIT"]
            sky_targets = np.concatenate([read_targets_in_tiles(skydir, tiles=tiles, columns=columns, quick=True) for skydir in skydirs])
            # Add sky targets to fiberassign Class objects
            load_target_table(tgs, tagalong, sky_targets, survey=tgs.survey(), typecol=args.mask_column,
                              sciencemask=args.sciencemask, stdmask=args.stdmask, skymask=args.skymask,
                              safemask=args.safemask, excludemask=args.excludemask, gaia_stdmask=args.gaia_stdmask,
                              rundate=args.rundate)

        return tgs, tagalong

    # Read hardware properties
    hw = load_hardware(rundate=args.rundate, add_margins=args.margins)

    # convert target to fiberassign.Targets Class
    tgs, tagalong = convert_targets_to_fiberassign(args, targets, tiles, tiles['PROGRAM'].values[0], use_sky_targets=use_sky_targets)

    # convert tiles to fiberassign.Tiles Class
    tiles = convert_tiles_to_fiberassign(args, tiles)

    return (hw, tiles, tgs, tagalong)


def _run_assign_full(args, hw, tiles, tgs, tagalong):
    """
    Run fiber assignment over all tiles simultaneously.

    Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py
    """
    from fiberassign.utils import GlobalTimers
    from fiberassign.stucksky import stuck_on_sky
    from fiberassign.assign import Assignment, run
    from fiberassign.targets import TargetsAvailable, LocationsAvailable, targets_in_tiles

    gt = GlobalTimers.get()
    gt.start("run_assign_full calculation")

    # Find targets within tiles, and project their RA,Dec positions
    # into focal-plane coordinates.
    gt.start("Compute targets locations in tile")
    tile_targetids, tile_x, tile_y, tile_xy_cs5 = targets_in_tiles(hw, tgs, tiles, tagalong)
    gt.stop("Compute targets locations in tile")

    # Compute the targets available to each fiber for each tile.
    gt.start("Compute Targets Available")
    tgsavail = TargetsAvailable(hw, tiles, tile_targetids, tile_x, tile_y)
    gt.stop("Compute Targets Available")

    # Free the target locations
    del tile_targetids, tile_x, tile_y

    # Compute the fibers on all tiles available for each target and sky
    gt.start("Compute Locations Available")
    favail = LocationsAvailable(tgsavail)
    gt.stop("Compute Locations Available")

    # Find stuck positioners and compute whether they will land on acceptable
    # sky locations for each tile.
    gt.start("Compute Stuck locations on good sky")
    stucksky = stuck_on_sky(hw, tiles, args.lookup_sky_source, rundate=args.rundate)
    if stucksky is None:
        # (the pybind code doesn't like None when a dict is expected...)
        stucksky = {}
    gt.stop("Compute Stuck locations on good sky")

    # Create assignment object
    gt.start("Construct Assignment")
    asgn = Assignment(tgs, tgsavail, favail, stucksky)
    gt.stop("Construct Assignment")

    run(asgn, args.standards_per_petal, args.sky_per_petal, args.sky_per_slitblock,
        redistribute=not args.no_redistribute, use_zero_obsremain=not args.no_zero_obsremain)
    gt.stop("run_assign_full calculation")

    return asgn


def _extract_info_assignment(asgn, verbose=False):
    """ Extract tragets assigned and available (usefull for randoms) from Assignment Class of fiberassign
        Copy and Adapt from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/assign.py """
    # Target properties
    tgs = asgn.targets()
    # collect loc fibers
    fibers = dict(asgn.hardware().loc_fiber)

    # Loop over each tile
    tg_assign, tg_avail = [], []
    for t in asgn.tiles_assigned():
        tdata = asgn.tile_location_target(t)
        avail = asgn.targets_avail().tile_data(t)

        if len(tdata) > 0:
            # Only Collect science targets (ie) FA_TYPE & 2**0 != 0
            # Collect assign targets
            tg_assign_tmp = np.concatenate([np.array([[tdata[x], tgs.get(tdata[x]).type, fibers[x]]]) for x in tdata.keys() if (tgs.get(tdata[x]).type & 2**0) != 0])
            tg_assign.append(tg_assign_tmp)
            # Collect available targets
            # take care, there are overlaps between fibers
            # here, I do not collect fiber number for available target (modify np.unique ect...)
            tg_avail_tmp = [id for id in np.unique(np.concatenate([np.array(avail[x], dtype=np.int64) for x in avail.keys()])) if tgs.get(id).type & 2**0 != 0]
            tg_avail.append(tg_avail_tmp)

            if verbose: logger.info(f'Tile: {t}, Assign: {len(tg_assign_tmp)}, Avail: {len(tg_avail_tmp)}')

    tg_assign, tg_avail = np.concatenate(tg_assign), np.concatenate(tg_avail)

    tg_assign = {'TARGETID': tg_assign[:, 0], 'FA_TYPE': tg_assign[:, 1], 'FIBER': tg_assign[:, 2]}
    tg_avail = {'TARGETID': tg_avail}

    return tg_assign, tg_avail


def _apply_mtl_one_pass(targets, tg_assign, tg_available):
    """
    Proxi of true MTL --> OK FOR THE MOMENT BUT SHOULD BE UPDATE IN THE FUTUR. (reobservation of QSO z>2.1 has same priority than initial observation of QSO ?)

    Note we apply fiber assignment pass by pass. Only one observation per targets can be done in one pass.

    Use Available for randoms. Available = can be observed with at least one fiber but not chosen by the F.A. process.
    """

    from desitarget.geomask import match

    idx, idx2 = match(targets['TARGETID'], tg_assign['TARGETID'])
    targets["NUMOBS_MORE"][idx] -= 1
    targets["NUMOBS"][idx] += 1
    targets["FIBER"][idx] = tg_assign["FIBER"][idx2]

    idx, _ = match(targets['TARGETID'], tg_available['TARGETID'])
    targets['AVAILABLE'][idx] = True

    return targets


def run_fiber_assignment_one_pass(tiles, targets, opts_for_fa, plate_radec=True, use_sky_targets=True):
    """ From tiles and targets run step by step the fiber assignment process for one pass """
    from fiberassign.scripts.assign import parse_assign

    # load param for firber assignment
    ag = parse_assign(opts_for_fa)

    # Convert data to fiberassign class
    hw, tiles, tgs, tagalong = _run_assign_init(ag, tiles, targets, plate_radec=plate_radec, use_sky_targets=use_sky_targets)

    # run assignment
    asgn = _run_assign_full(ag, hw, tiles, tgs, tagalong)

    # from assignment collect which targets is selected and available (useful for randoms !)
    # on vuet plus d'info ? le numéros de la fibre par exemple ?
    tg_assign, tg_available = _extract_info_assignment(asgn)

    # update targets with 'the observation'
    targets = _apply_mtl_one_pass(targets, tg_assign, tg_available)

    return targets


def apply_fiber_assignment(targets, tiles, npass, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=True):
    """
    Apply fiber assignment with MPI parrallelisation on the number of tiles per pass.

    targets is expected to be scattered on all MPI processes. tiles should have the info on the rank.

    Based on Anand Raichoor's code:
    https://github.com/echaussidon/LSS/blob/main/scripts/mock_tools/fa_multipass.py

    Parameters
    ----------
    targets : array
        Array containing at least: columns_for_fa
        and 'TILES' (list of potential tileid for each target). Expected to be scattered on all the process.

    tiles : array
        Array containing surveyops info. Can be build with _build_tiles()

    npass : int
        Number of passes during the fiber assignment.

    opts_for_fa : list
        List of str containing the option for fiberassign.scripts.assign.parse_assign

    columns_for_fa : array
        Name of columns that will be exchange with MPI.
        For the moment should at least contains: ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE', 'NUMOBS_INIT', 'NUMOBS', 'AVAILABLE', 'FIBER']

    mpicomm : MPI communicator, default=None
        The current MPI communicator.

    use_sky_targets : bool
        If False, do not include sky targets. Useful for debug since sky targets are not read and it speed up the process.
    """

    import mpytools as mpy
    import mpsort

    def _dict_to_array(data):
        """
        Return dict as numpy array.

        Parameters
        ----------
        data : dict
            Data dictionary of name: array.

        Returns
        -------
        array : array
        """
        array = [(name, data[name]) for name in data]
        array = np.empty(array[0][1].shape[0], dtype=[(name, col.dtype, col.shape[1:]) for name, col in array])
        for name in data: array[name] = data[name]
        return array

    if mpicomm.rank == 0: logger.info(f'Start Fiber Assignment with {npass} (use sky targets? {use_sky_targets})')

    for pass_id in range(npass):
        tiles_in_pass = tiles[tiles['PASS'] == pass_id]
        if mpicomm.rank == 0: logger.info(f'Start pass {pass_id} with {tiles_in_pass.shape[0]} potential tiles')

        # could be time consuming...
        targets_tile_in_pass = np.isin(targets['TILES'], tiles_in_pass)
        sel_targets_in_pass = targets_tile_in_pass.sum(axis=1) > 0
        targets_tileid_in_pass = np.array(targets['TILES'][targets_tile_in_pass], dtype='i8')

        # Create unique identification as index column
        cumsize = np.cumsum([0] + mpicomm.allgather(targets_tileid_in_pass.size))[mpicomm.rank]
        index = cumsize + np.arange(targets_tileid_in_pass.size)
        targets_in_pass = {name: targets[name][sel_targets_in_pass] for name in columns_for_fa}
        targets_in_pass.update({'TILES': targets_tileid_in_pass, 'index': index})
        targets_in_pass = _dict_to_array(targets_in_pass)

        # Let's group particles by tileid, with ~ similar number of tileid on each rank
        # Caution: this may produce memory unbalance between different processes
        # hence potential memory error, which may be avoided using some criterion to rebalance load at the cost of less efficiency
        unique_tileid, counts_tileid = np.unique(targets_tileid_in_pass, return_counts=True)

        # Proceed rank-by-rank to save memory
        for irank in range(1, mpicomm.size):
            unique_tileid_irank = mpy.sendrecv(unique_tileid, source=irank, dest=0, tag=0, mpicomm=mpicomm)
            counts_tileid_irank = mpy.sendrecv(counts_tileid, source=irank, dest=0, tag=0, mpicomm=mpicomm)
            if mpicomm.rank == 0:
                unique_tileid, counts_tileid = np.concatenate([unique_tileid, unique_tileid_irank]), np.concatenate([counts_tileid, counts_tileid_irank])
                unique_tileid, inverse = np.unique(unique_tileid, return_inverse=True)
                counts_tileid = np.bincount(inverse, weights=counts_tileid).astype(int)

        # Compute the number particles that each rank must contain after sorting
        nbr_particles = None
        if mpicomm.rank == 0:
            nbr_tiles = [(irank * unique_tileid.size // mpicomm.size, (irank + 1) * unique_tileid.size // mpicomm.size) for irank in range(mpicomm.size)]
            nbr_particles = [np.sum(counts_tileid[nbr_tile_low:nbr_tile_high], dtype='i8') for nbr_tile_low, nbr_tile_high in nbr_tiles]
            nbr_tiles = np.diff(nbr_tiles, axis=-1)
            logger.info(f'Number of tiles to process per rank = {np.min(nbr_tiles)} - {np.max(nbr_tiles)} (min - max).')

        # Send the number particles that each rank must contain after sorting
        nbr_particles = mpicomm.scatter(nbr_particles, root=0)
        assert mpicomm.allreduce(nbr_particles) == mpicomm.allreduce(targets_in_pass.size), 'float in bincount messes up total particle counts'

        # Sort data to have same number of bricks in each rank
        targets_in_pass_tmp = np.empty_like(targets_in_pass, shape=nbr_particles)
        mpsort.sort(targets_in_pass, orderby='TILES', out=targets_in_pass_tmp)

        # Which tiles are treated on the current process
        sel_tiles_in_process = np.isin(tiles_in_pass['TILEID'], targets_in_pass_tmp['TILES'])
        # run F.A. only on these tiles
        if sel_tiles_in_process.sum() != 0:
            targets_in_pass_tmp = run_fiber_assignment_one_pass(tiles_in_pass[sel_tiles_in_process], targets_in_pass_tmp, opts_for_fa, use_sky_targets=use_sky_targets)

        # Put the new data in the intial order
        targets_in_pass = np.empty_like(targets_in_pass_tmp, shape=targets_tileid_in_pass.size)
        mpsort.sort(targets_in_pass_tmp, orderby='index', out=targets_in_pass)
        # Check if we find the correct initial order
        assert np.all(targets_in_pass['index'] == index)

        # Update the targets before starting a new pass
        for col in ['NUMOBS_MORE', 'NUMOBS', 'FIBER', 'AVAILABLE']:
            targets[col][sel_targets_in_pass] = targets_in_pass[col]


if __name__ == '__main__':

    from mockfactory import RandomCutskyCatalog, setup_logging
    import desimodel.footprint

    os.environ['DESI_LOGLEVEL'] = 'ERROR'

    setup_logging()

    mpicomm = MPI.COMM_WORLD
    if mpicomm.rank == 0: logger.info('Run simple example to illustrate how to run fiber assignment.')

    # program info:
    release = 'Y1'
    program = 'dark'
    npasses = 2
    use_sky_targets = True  # to debug: Use false to speed up the process

    # Collect tiles from surveyops directory on which the fiber assignment will be applied
    tiles = _build_tiles(release_tile_path=f'/global/cfs/cdirs/desi/survey/catalogs/{release}/LSS/tiles-{program.upper()}.fits', program=program, npasses=npasses)

    # Get info from origin fiberassign file and setup options for F.A.
    ts = str(tiles['TILEID'][0]).zfill(6)
    fht = fitsio.read_header(f'/global/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{ts[:3]}/fiberassign-{ts}.fits.gz')
    rundate = fht['RUNDATE']
    # see fiberassign.scripts.assign.parse_assign (Can modify margins, number of sky fibers for each petal ect...)
    opts_for_fa = ["--target", " ", "--rundate", rundate, "--mask_column", "DESI_TARGET"]

    # Generate example cutsky catalog, scattered on all processes (use a high completeness region):
    cutsky = RandomCutskyCatalog(rarange=(176.5, 182.1), decrange=(-6, 5), nbar=260, seed=44, mpicomm=mpicomm)
    # save inital ra, dec for plotting purpose (see end of this script)
    ra_ini, dec_ini = cutsky.cget('RA', mpiroot=0), cutsky.cget('DEC', mpiroot=0)

    # To apply F.A., we need to add some information as DESI_TARGET controlling the priority, number of observation per targets ect...
    # In order to speed the process, fiber assignment on each pass will be parrallelized on the number of tiles. Need also to include list of potential tiles for each targets.
    # This part should be avoid if the catalog is empty on the process (not check here)

    # Note: here for this small example, we emulate the F.A. for QSO targets. Since they have the highest priority we do not need to add other targets to mimic the real F.A.
    # To emulate the F.A. for ELG, we will want to add other targets (QSO / LRG) with correct DESI_TARGET column with random postions (it should be enought if no cross-correlation)
    # with the correct density (including the fluctuation from imaging systematics)
    # For this tiny example, we do not use reobservation for QSO with z>2.1 (could be easly done with we have the redshift column)

    # add all the tile_id for each target. One target could be in several tiles with different value of 'PASS'
    tile_id = [tiles['TILEID'].values[np.array(idx, dtype='int64')].tolist() for idx in desimodel.footprint.find_tiles_over_point(tiles, cutsky['RA'], cutsky['DEC'])]
    nbr_max_tiles = max(map(len, tile_id))
    cutsky['TILES'] = np.array([id + [np.nan] * (nbr_max_tiles - len(id)) for id in tile_id])

    # Remove targets without potential observation to mimic the desi footprint (Just to limit the cutsky to real desi cutsky)
    # This step is not mandatory since the F.A. will not consider targets in any tiles
    sel = np.nansum(cutsky['TILES'], axis=1) > 0
    cutsky = cutsky[sel]
    nbr_targets = cutsky.csize
    if mpicomm.rank == 0: logger.info(f'Keep only objects which is in a tile. Working with {nbr_targets} targets')

    # Add requiered columns for F.A.
    cutsky['DESI_TARGET'] = 2 * np.ones(cutsky.size, dtype='i8')
    cutsky['SUBPRIORITY'] = np.random.uniform(0, 1, cutsky.size)
    cutsky['OBSCONDITIONS'] = 3 * np.ones(cutsky.size, dtype='i8')
    cutsky['NUMOBS_MORE'] = np.ones(cutsky.size, dtype='i8')
    cutsky['NUMOBS_INIT'] = np.ones(cutsky.size, dtype='i8')
    cutsky['NUMOBS'] = np.zeros(cutsky.size, dtype='i8')
    cutsky['AVAILABLE'] = np.zeros(cutsky.size, dtype='?')
    cutsky['FIBER'] = np.nan * np.zeros(cutsky.size, dtype='i8')
    # take care with MPI ! TARGETID has to be unique !
    cumsize = np.cumsum([0] + mpicomm.allgather(cutsky.size))[mpicomm.rank]
    cutsky['TARGETID'] = cumsize + np.arange(cutsky.size)

    # columns needed to run the F.A. and collect the info (They will be exchange between processes during the F.A.)
    columns_for_fa = ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE', 'NUMOBS_INIT', 'NUMOBS', 'AVAILABLE', 'FIBER']

    # Let's do the F.A.:
    if mpicomm.rank == 0: logger.warning("There is NO reproductibility for the moment np.random.seed() at fix number of process does not work")
    apply_fiber_assignment(cutsky, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=use_sky_targets)

    # Summarize and plot:
    ra, dec = cutsky.cget('RA', mpiroot=0), cutsky.cget('DEC', mpiroot=0)
    numobs, available = cutsky.cget('NUMOBS', mpiroot=0), cutsky.cget('AVAILABLE', mpiroot=0)

    if mpicomm.rank == 0:
        import matplotlib.pyplot as plt

        observed = (numobs >= 1)
        logger.info(f"Nbr of targets observed: {observed.sum()} -- Nbr of targets available: {available.sum()} -- Nbr of targets: {ra.size}")

        tiles = tiles[tiles['PASS'] < npasses]
        tile_id = np.unique(np.concatenate([tiles['TILEID'].values[np.array(idx, dtype='int64')] for idx in desimodel.footprint.find_tiles_over_point(tiles, ra, dec)]))

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        ax = axs[0]
        for id in tile_id:
            tile = tiles[tiles['TILEID'] == id]
            c = plt.Circle((tile['RA'].values[0], tile['DEC'].values[0]), np.sqrt(8 / 3.14), color='lightgrey', alpha=1)
            ax.add_patch(c)

        ax.scatter(ra_ini, dec_ini, c='red', s=0.5, label='random')
        ax.scatter(ra, dec, c='green', s=0.5, label='in desi footprint')
        ax.legend()
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')

        ax = axs[1]
        for id in tile_id:
            tile = tiles[tiles['TILEID'] == id]
            c = plt.Circle((tile['RA'].values[0], tile['DEC'].values[0]), np.sqrt(8 / 3.14), color='lightgrey', alpha=1)
            ax.add_patch(c)

        ax.scatter(ra, dec, c='red', s=0.5, label='targets')
        ax.scatter(ra[available], dec[available], c='orange', s=0.5, label=f'available: {available.sum() / ra.size:2.2%}')
        ax.scatter(ra[observed], dec[observed], c='green', s=0.5, label=f'observed: {observed.sum() / ra.size:2.2%}')

        ax.legend()
        ax.set_xlabel('R.A. [deg]')

        plt.suptitle(f'Fiber assignment for {release} release - {program} program - {npasses} passes', y=0.96)

        plt.tight_layout()
        plt.savefig(f'fiberasignment_{npasses}npasses.png')
        plt.close()
        logger.info(f'Plot save in fiberasignment_{npasses}npasses.png')

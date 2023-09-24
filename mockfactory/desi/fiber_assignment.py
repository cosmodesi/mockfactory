"""
Script to apply fiber assignment from pre-loaded catalogs.

This example can be run with `srun -n 5 python fiber_assignment.py` (will take typically 1 minute for 1 pass),
but one will typically import:
```
from mockfactory.fiber_assignment import apply_fiber_assignment
```
For an example, see desi/apply_fiber_assignment_example.py script.
"""

import os
import logging

import fitsio
import numpy as np
import pandas as pd

from mpi4py import MPI


logger = logging.getLogger('F.A.')


def build_tiles_for_fa(release_tile_path='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits',
                       surveyops_tile_path='/global/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-main.ecsv',
                       program='dark', npasses=7):
    """Load tile properties from surveyops dir selecting only tiles in the desired release/program with enough passes."""
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


def _write_all_skytargets(dirname, columns=["RA", "DEC", "TARGETID", "DESI_TARGET", "SUBPRIORITY", "OBSCONDITIONS"], program='dark', nfiles=10, mpicomm=None):
    """It is a real nightmare to work with all the fits file in hpdirname_list more than 800 for each directory. Rewrite it in a smaller number of files."""
    from glob import glob
    import mpytools as mpy
    from fiberassign.fba_launch_io import get_desitarget_paths

    # Now load the sky target files.  These are main-survey files that we will
    # force to be treated as the survey type of the other target files.
    mydirs = get_desitarget_paths('1.1.1', 'main', program, dr='dr9')
    skydirs = [mydirs["sky"]]
    if os.path.isdir(mydirs["skysupp"]):
        skydirs.append(mydirs["skysupp"])

    for hpdirname in skydirs:
        fns = []
        if mpicomm.rank == 0:
            fns = glob(os.path.join(hpdirname, "*fits"))
        fns = list(mpy.bcast(fns, mpiroot=0))

        # Create mpytools.Catalog
        targets = mpy.Catalog.read(fns, mpicomm=mpicomm)

        # first save it in nfiles fits files.
        # Create filenames to write targets in less files.
        basename = '-'.join(os.path.basename(fns[0]).split('-')[:-2])
        fns_to_save = [os.path.join(dirname, f'{basename}-{i}.fits') for i in range(nfiles)]

        start = MPI.Wtime()
        targets[columns].write(fns_to_save, filetype='fits')
        mpicomm.Barrier()
        if mpicomm.rank == 0: logger.info(f'Write {basename} in {nfiles} done in {MPI.Wtime() - start:2.2f} s.')

        # Save it on one bigfile (more efficient ?)
        start = MPI.Wtime()
        targets[columns].write(os.path.join(dirname, f'{basename}'), filetype='bigfile')
        mpicomm.Barrier()
        if mpicomm.rank == 0: logger.info(f'Write {basename} in {nfiles} done in {MPI.Wtime() - start:2.2f} s.')


def read_sky_targets(dirname='/global/cfs/cdirs/desi/users/edmondc/desi_targets/sky_targets', filetype='fits', tiles=None, mpicomm=None):
    """
    To avoid to deal with /global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/ and its 800 fits files in each directory,
    we first run _write_all_skytargets in order to reduce the number of files to 20.
    This function loads sky targets in the corresponding tiles. It is useful to run before the F.A.
    in order to avoid to spend too much time in reading the sky target fits files during the F.A.

    Remark: dirname is about 34GB. No problem if loaded on several nodes / if you use small number of tiles!
    """
    from glob import glob
    import mpytools as mpy
    from desimodel.footprint import is_point_in_desi

    t_start = MPI.Wtime()
    if mpicomm.rank == 0: logger.info(f'Start read sky targets for {tiles.shape[0]} tiles')

    if filetype == 'fits':
        fns = []
        if mpicomm.rank == 0:
            fns = glob(os.path.join(dirname, "*.fits"))
        fns = list(mpy.bcast(fns, mpiroot=0))
    elif filetype == 'bigfile':
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        fns = []
        if mpicomm.rank == 0:
            fns = [name for name in glob(os.path.join(dirname, '*')) if len(name.split('.')) == 1]
        fns = list(mpy.bcast(fns, mpiroot=0))
    else:
        if mpicomm.rank == 0: logger.error(f'filetype={filetype} is not expected')

    # Note Catalog.read() reads only header (almost free), nothing is loaded it this time!
    sky_targets = mpy.Catalog.read(fns, filetype=filetype, mpicomm=mpicomm)

    # Temporary: there is a strange memory issue if you called sky_targets[mask]['column'] without having read the column before applying mask...
    # For safety: read all the column (load it in memory)
    start = MPI.Wtime()
    for name in sky_targets.columns():
        _ = sky_targets[name]
    mpicomm.Barrier()  # wait all the processes before continuing to avoid MPI waiting failure... (strange but it is like that)
    if mpicomm.rank == 0: logger.info(f'Pre-loaded {len(sky_targets.columns())} columns of all the sky targets done in {MPI.Wtime() - start:3.2} s.')

    # Keep only targets which are in the desired tiles:
    sky_targets = sky_targets[is_point_in_desi(tiles, sky_targets['RA'], sky_targets['DEC'])]

    csize = sky_targets.csize
    if mpicomm.rank == 0: logger.info(f'Loaded {csize} sky targets done in {MPI.Wtime() - t_start:2.2f} s.')

    return sky_targets


def _run_assign_init(args, tiles, targets, plate_radec=True, use_sky_targets=True, sky_targets=None):
    """
    Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py#L281

    Instead of reading files, use preloaded targets and tiles.
    """
    from fiberassign.hardware import load_hardware

    def convert_tiles_to_fiberassign(args, tiles):
        """
        Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/tiles.py.
        Do not read the tiles, but take it as an array...
        """
        import warnings
        from desimodel.focalplane.fieldrot import field_rotation_angle
        import astropy.time
        from fiberassign._internal import Tiles

        # astropy ERFA doesn't like the future
        warnings.filterwarnings('ignore', message=r'ERFA function \"[a-z0-9_]+\" yielded [0-9]+ of \"dubious year')

        if args.obsdate is not None:
            # obstime is given, use that for all tiles
            obsdate = astropy.time.Time(args.obsdate)
            #obsmjd = [obsdate.mjd, ] * tiles.shape[0]
            #obsdatestr = [obsdate.isot, ] * tiles.shape[0]
            obsmjd = [obsdate.mjd, ] * len(tiles)
            obsdatestr = [obsdate.isot, ] * len(tiles)
        elif "OBSDATE" in tiles.names:
            # We have the obsdate for every tile in the file.
            obsdate = [astropy.time.Time(x) for x in tiles["OBSDATE"]]
            obsmjd = [x.mjd for x in obsdate]
            obsdatestr = [x.isot for x in obsdate]
        else:
            # default to middle of the survey
            obsdate = astropy.time.Time('2022-07-01')
            #obsmjd = [obsdate.mjd, ] * tiles.shape[0]
            #obsdatestr = [obsdate.isot, ] * tiles.shape[0]
            obsmjd = [obsdate.mjd, ] * len(tiles)
            obsdatestr = [obsdate.isot, ] * len(tiles)

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
            #theta_obs = np.zeros(tiles.shape[0], dtype=np.float64)
            theta_obs = np.zeros(len(tiles), dtype=np.float64)
            theta_obs[:] = args.fieldrot

        # default to zero Hour Angle; may be refined later
        #ha_obs = np.zeros(tiles.shape[0], dtype=np.float64)
        ha_obs = np.zeros(len(tiles), dtype=np.float64)
        if args.ha is not None:
            ha_obs[:] = args.ha

        #return Tiles(tiles["TILEID"].values, tiles["RA"].values, tiles["DEC"].values, tiles["OBSCONDITIONS"].values, obsdatestr, theta_obs, ha_obs)
        return Tiles(tiles["TILEID"], tiles["RA"], tiles["DEC"], tiles["OBSCONDITIONS"], obsdatestr, theta_obs, ha_obs)


    def convert_targets_to_fiberassign(args, targets, tiles, program, use_sky_targets=True, sky_targets=None):
        """
        Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py#L281.
        Do not read the tiles, but take it as an array...
        """
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
            # If sky_targets is not already loaded, read the minimal one. This very time consumming, you can avoid multiple reading (especially if you apply F.A. on several mocks)
            if sky_targets is None:
                # Now load the sky target files. These are main-survey files that we will
                # force to be treated as the survey type of the other target files.
                mydirs = get_desitarget_paths('1.1.1', 'main', program, dr='dr9')
                skydirs = [mydirs["sky"]]
                if os.path.isdir(mydirs["skysupp"]):
                    skydirs.append(mydirs["skysupp"])
                columns = ["RA", "DEC", "TARGETID", "DESI_TARGET", "SUBPRIORITY", "OBSCONDITIONS"]
                sky_targets = np.concatenate([read_targets_in_tiles(skydir, tiles=tiles, columns=columns, quick=True) for skydir in skydirs])
            # Add sky targets to fiberassign Class objects
            load_target_table(tgs, tagalong, sky_targets, survey=tgs.survey(), typecol=args.mask_column,
                              sciencemask=args.sciencemask, stdmask=args.stdmask, skymask=args.skymask,
                              safemask=args.safemask, excludemask=args.excludemask, gaia_stdmask=args.gaia_stdmask,
                              rundate=args.rundate)

        return tgs, tagalong

    # Read hardware properties
    hw = load_hardware(rundate=args.rundate, add_margins=args.margins)

    # Convert target to fiberassign.Targets Class
    #tgs, tagalong = convert_targets_to_fiberassign(args, targets, tiles, tiles['PROGRAM'].values[0], use_sky_targets=use_sky_targets)
    tgs, tagalong = convert_targets_to_fiberassign(args, targets, tiles, tiles['PROGRAM'][0], use_sky_targets=use_sky_targets)

    # Convert tiles to fiberassign.Tiles Class
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
    """
    Extract tragets assigned and available (useful for randoms) from :class:`Assignment` of fiberassign.
    Since we work pass by pass, can concatenate the tiles without any problem (targets appear only once by pass).
    Copied and adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/assign.py
    """
    # Target properties
    tgs = asgn.targets()
    # collect loc fibers
    fibers = dict(asgn.hardware().loc_fiber)

    # Loop over each tile
    tg_assign, tg_avail = [], []
    for t in asgn.tiles_assigned():
        tdata = asgn.tile_location_target(t)
        avail = asgn.targets_avail().tile_data(t)

        # check if there is at least one science observed target
        if np.sum([tgs.get(tdata[x]).type & 2**0 != 0 for x in tdata.keys()]) > 0:
            # Only Collect science targets (ie) FA_TYPE & 2**0 != 0
            # Collect assign targets
            tg_assign_tmp = np.concatenate([np.array([[tdata[x], tgs.get(tdata[x]).type, fibers[x]]]) for x in tdata.keys() if (tgs.get(tdata[x]).type & 2**0) != 0])
            tg_assign.append(tg_assign_tmp)

            # Collect available targets and one fiber if available
            # take care, there are overlaps between fibers BUT NOT between tiles (since we work pass by pass)
            # Choose one fiber: take the first one with fiber != -1 in the list if several fibers are available for the same target)
            # Take care, location can exist wihtout fiber (fiber broken ? ect...). First step is to remove location without fiber !"

            # Available targets are targets which can be reach by fiber assigned for science case and fiber != -1 (working?)
            loc_fiber_ok = np.array([loc for loc in avail.keys() if (fibers[loc] in tg_assign_tmp[:, 2])])

            tg_avail_tmp = []
            for x in loc_fiber_ok:
                for av in avail[x]:
                    if (tgs.get(av).type & 2**0) != 0:
                        tg_avail_tmp.append([av, fibers[x]])

            # Keep for each available target only one fiber (for the completeness weight)
            _, idx = np.unique(np.array(tg_avail_tmp)[:, 0], return_index=True)
            tg_avail_tmp = np.array(tg_avail_tmp)[idx, :]
            tg_avail.append(tg_avail_tmp)

            if verbose: logger.info(f'Tile: {t}, Assign: {tg_assign_tmp.shape}, Avail: {tg_avail_tmp.shape}, Ratio: {np.isin(tg_avail_tmp[:, 1], tg_assign_tmp[:, 2]).sum() / tg_avail_tmp[:, 1].size}')

    if tg_assign == []:
        tg_assign = {'TARGETID': np.array([]), 'FA_TYPE': np.array([]), 'FIBER': np.array([])}
        tg_avail = {'TARGETID': np.array([]), 'FIBER': np.array([])}
    else:
        tg_assign, tg_avail = np.concatenate(tg_assign), np.concatenate(tg_avail)

        tg_assign = {'TARGETID': tg_assign[:, 0], 'FA_TYPE': tg_assign[:, 1], 'FIBER': tg_assign[:, 2]}
        tg_avail = {'TARGETID': tg_avail[:, 0], 'FIBER': tg_avail[:, 1]}

    return tg_assign, tg_avail


def _apply_mtl_one_pass(targets, tg_assign, tg_available):
    """
    Proxy of true MTL --> OK FOR THE MOMENT BUT SHOULD BE UPDATEd IN THE FUTURE.
    (reobservation of QSO z>2.1 has same priority than initial observation of QSO?)

    Note we apply fiber assignment pass by pass. Only one observation per target can be done in one pass.

    Use AVAILABLE for randoms. Available = can be observed with at least one fiber but not chosen by the F.A. process.
    """
    from desitarget.geomask import match

    idx, idx2 = match(targets['TARGETID'], tg_available['TARGETID'])
    targets['AVAILABLE'][idx] = True
    targets['FIBER'][idx] = np.array(tg_available["FIBER"][idx2], dtype='i8')

    idx, idx2 = match(targets['TARGETID'], tg_assign['TARGETID'])
    targets["NUMOBS_MORE"][idx] -= 1
    targets["NUMOBS"][idx] += 1
    targets["FIBER"][idx] = np.array(tg_assign["FIBER"][idx2], dtype='i8')  # rewrite with the correct assign fiber if several are available.
    targets["OBS_PASS"][idx] = True

    return targets


def _run_fiber_assignment_one_pass(tiles, targets, opts_for_fa, plate_radec=True, use_sky_targets=True, sky_targets=None):
    """
    From tiles and targets run step by step the fiber assignment process for one pass.

    Note: to work with fiberassign package (ie) for _run_assign_init function,
    targets should be a dtype numpy array and not a mpytools.Catalog. Convert it with Catalog.to_array().
    """
    from fiberassign.scripts.assign import parse_assign

    # load param for firber assignment
    ag = parse_assign(opts_for_fa)

    # Convert data to fiberassign class
    # targets should be a dtype numpy array here

    from astropy.table import Table
    tiles_new_format = Table(tiles.to_records())

    hw, tiles, tgs, tagalong = _run_assign_init(ag, tiles_new_format, targets.to_array(), plate_radec=plate_radec, use_sky_targets=use_sky_targets, sky_targets=sky_targets)

    # run assignment
    asgn = _run_assign_full(ag, hw, tiles, tgs, tagalong)

    # from assignment collect which targets is selected and available (useful for randoms !)
    tg_assign, tg_available = _extract_info_assignment(asgn)

    # update targets with 'the observation'
    targets = _apply_mtl_one_pass(targets, tg_assign, tg_available)


def apply_fiber_assignment(targets, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=True, sky_targets=None):
    """
    Apply fiber assignment with MPI parrallelisation on the number of tiles per pass.
    Targets are expected to be scattered on all MPI processes. Tiles should be load on each rank.

    Based on Anand Raichoor's code:
    https://github.com/desihub/LSS/blob/main/scripts/mock_tools/fa_multipass.py

    Parameters
    ----------
    targets : array
        Array containing at least: ``columns_for_fa``.

    tiles : array
        Array containing surveyops info. Can be build with ``_build_tiles()``.

    npasses : int
        Number of passes during the fiber assignment.

    opts_for_fa : list
        List of strings containing the option for :func:`fiberassign.scripts.assign.parse_assign`.

    columns_for_fa : array
        Name of columns that will be exchanged with MPI.
        For the moment should at least contains: ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE', 'NUMOBS_INIT']

    mpicomm : MPI communicator
        The current MPI communicator.

    use_sky_targets : bool, default=True
        If ``False``, do not include sky targets. Useful for debugging since sky targets are not read which speeds up the process.

    sky_targets : None or mpytool.Catalog
        If :class:Catalog, should contain all the sky targets available in each tile.
    """
    import mpytools as mpy
    import desimodel.footprint

    start = MPI.Wtime()

    csize = mpy.gather(targets.size, mpiroot=0)
    if mpicomm.rank == 0: logger.info(f'Start fiber assignment for {csize.sum()} objects and {npasses} pass(es): use sky targets? {use_sky_targets} -- use pre-loaded sky targets? {sky_targets is not None}')

    # Add columns to collect fiber assign output !
    targets['NUMOBS', 'AVAILABLE', 'FIBER', 'OBS_PASS'] = [np.zeros(targets.size, dtype='i8'), np.zeros(targets.size, dtype='?'),
                                                           -1 * np.ones((targets.size, npasses), dtype='i8'),  # To compute the completeness weight, need to collect at least (if avialable) one fiber per pass!
                                                           np.zeros((targets.size, npasses), dtype='?')]  # Take care to the size --> with mpytools should be (size, X) and not (X, size)

    for pass_id in range(npasses):
        tiles_in_pass = tiles[tiles['PASS'] == pass_id]
        if mpicomm.rank == 0: logger.info(f'    * Pass {pass_id} with {tiles_in_pass.shape[0]} potential tiles')

        # Since we consider only one pass at each time find_tiles_over_point can return only at most one tile for each target
        tile_id = np.array([tiles_in_pass['TILEID'].values[idx[0]] if len(idx) != 0 else -1
                            for idx in desimodel.footprint.find_tiles_over_point(tiles_in_pass, targets['RA'], targets['DEC'])])
        # keep only targets in the correct pass and with potential observation
        sel_targets_in_pass = (tile_id >= 0) & (targets["NUMOBS_MORE"] > 0)
        # create subcatalog
        targets_in_pass = targets[columns_for_fa + ['NUMOBS', 'AVAILABLE']][sel_targets_in_pass]
        targets_in_pass['TILEID', 'OBS_PASS', 'FIBER', 'index'] = [tile_id[sel_targets_in_pass], np.zeros(sel_targets_in_pass.sum(), dtype='?'),
                                                                   -1 * np.ones(sel_targets_in_pass.sum(), dtype='i8'),
                                                                   targets_in_pass.cindex()]
        # Copy unique identification to perform sanity check at the end
        index = targets_in_pass['index']

        # Sort data to have same number of tileid in each rank
        t_start = MPI.Wtime()
        targets_in_pass = targets_in_pass.csort('TILEID', size='orderby_counts')
        mpicomm.Barrier()
        if mpicomm.rank == 0: logger.info(f'        ** Particle exchange between all the processes took: {MPI.Wtime() - t_start:2.2f} s.')
        nbr_tiles = mpy.gather(np.unique(targets_in_pass['TILEID']).size, mpiroot=0)
        if mpicomm.rank == 0: logger.info(f'        ** Number of tiles to process per rank = {np.min(nbr_tiles)} - {np.max(nbr_tiles)} (min - max).')

        # sort sky_targets as a function of the TILEID to match the TILEID distribution of targets on all the ranks.
        sky_targets_in_pass = None
        if sky_targets is not None:
            t_start = MPI.Wtime()
            # sky_targets is supposed to be loaded on all the existing tiles_in_pass
            # but we want to match the tiles from targets ! sky_targets could be loaded also on tiles where they are no targets.
            # should scatter this array on all the process.
            target_tile_in_pass = np.unique(mpy.gather(np.unique(targets_in_pass['TILEID']), mpiroot=None))

            tile_id = np.array([target_tile_in_pass[idx[0]] if len(idx) != 0 else -1
                                for idx in desimodel.footprint.find_tiles_over_point(tiles_in_pass[np.isin(tiles_in_pass['TILEID'], target_tile_in_pass)], sky_targets['RA'], sky_targets['DEC'])])
            # keep only sky_targets in the correct pass and with potential observation
            sel_sky_targets_in_pass = (tile_id >= 0)
            # create subcatalog
            sky_targets_in_pass = sky_targets[sel_sky_targets_in_pass]
            sky_targets_in_pass['TILEID'] = np.array(tile_id[sel_sky_targets_in_pass], dtype='i8')

            # sort skay targets in order to have exactly the same tileid across all the ranks
            sky_targets_in_pass = sky_targets_in_pass.csort('TILEID', size='orderby_counts')
            mpicomm.Barrier()
            if mpicomm.rank == 0: logger.info(f'        ** Sky targets extraction and exchange between all the processes took: {MPI.Wtime() - t_start:2.2f} s.')

        # Which tiles are treated on the current process
        t_start = MPI.Wtime()
        sel_tiles_in_process = np.isin(tiles_in_pass['TILEID'], targets_in_pass['TILEID'])
        # run F.A. only on these tiles
        if sel_tiles_in_process.sum() != 0:
            _run_fiber_assignment_one_pass(tiles_in_pass[sel_tiles_in_process], targets_in_pass, opts_for_fa, use_sky_targets=use_sky_targets, sky_targets=sky_targets_in_pass)
        mpicomm.Barrier()
        if mpicomm.rank == 0: logger.info(f'        ** Apply F.A. Pass {pass_id} took: {MPI.Wtime() - t_start:2.2f} s.')

        # Put the new data in the intial order
        t_start = MPI.Wtime()
        targets_in_pass = targets_in_pass.csort('index', size=sel_targets_in_pass.sum())
        # Check if we find the correct initial order
        assert np.all(targets_in_pass['index'] == index)
        mpicomm.Barrier()
        if mpicomm.rank == 0: logger.info(f'        ** Particle exchange between all the processes took: {MPI.Wtime() - t_start:2.2f} s.')

        # Update the targets before starting a new pass (do it one by one)
        for col in ['NUMOBS_MORE', 'NUMOBS', 'AVAILABLE']:
            targets[col][sel_targets_in_pass] = targets_in_pass[col]
        targets['OBS_PASS'][sel_targets_in_pass, pass_id] = targets_in_pass['OBS_PASS']
        targets['FIBER'][sel_targets_in_pass, pass_id] = targets_in_pass['FIBER']

    mpicomm.Barrier()
    if mpicomm.rank == 0: logger.info(f'Apply fiber assign performed in elapsed time {MPI.Wtime() - start:2.2f} s.')


def _compute_completeness_weight_one_pass(tiles, targets):
    """
    Compute the completness weight on tiles for only one pass. When a target available unobserved is used to increase the completeness weight,
    it is set as NOT_USED_FOR_COMP_WEIGHT = False and not used in the next passes.

    Parameters
    ----------
    targets : array
        Array containing at least: FIBER', 'OBS_PASS' of shape (targets.size) of the current pass.

    tiles : array
        Array containing surveyops info of tiles from the current pass and for tiles treated in the the current process.
    """
    from desitarget.geomask import match

    # Loop over tiles
    for i in range(tiles.shape[0]):
        sel_targets_in_tile = targets['TILEID'] == tiles.iloc[i]['TILEID']
        # Extract only targets in this tile, needed for easier mask
        targets_in_tile = targets[sel_targets_in_tile]

        sel_obs = targets_in_tile['OBS_PASS']
        fiber_assign = targets_in_tile[sel_obs]['FIBER']

        # Want to know the unobserved targets which are available and not already used in the completeness weight
        # warning: do not use targets without fiber (not available in the current pass (ie) tile)
        sel_for_comp = targets_in_tile['NOT_USED_FOR_COMP_WEIGHT'] & (targets_in_tile["FIBER"] != -1)
        fiber_comp = targets_in_tile[sel_for_comp]['FIBER']
        fiber_id, counts = np.unique(fiber_comp, return_counts=True)

        # Find matched indices of fiber_id to targets_in_tile[sel_observed_targets]
        idx, idx2 = match(fiber_assign, fiber_id)

        # Need to do it in two steps (classic numpy memory attribution)
        # take care if one target is oberved several times (ie) in different passes, one need to add the comp_weight from each pass!
        comp_weight_tmp = targets_in_tile["COMP_WEIGHT"][sel_obs]
        comp_weight_tmp[idx] += counts[idx2]
        targets_in_tile["COMP_WEIGHT"][sel_obs] = comp_weight_tmp
        # Do not re-used these unobserved-available targets
        targets_in_tile['NOT_USED_FOR_COMP_WEIGHT'][sel_for_comp] = False

        # Update targets
        targets[sel_targets_in_tile] = targets_in_tile


def compute_completeness_weight(targets, tiles, npasses, mpicomm):
    """
    Compute the completeness weight associed to the fiber assignement.
    Targets should have been passed throught apply_fiber_assignment and contain all the assigned and available targets.
    Targets should have the NUMOBS, AVAILABLE and FIBER columns.
    The completeness weight is defined as the number of targets that "wanted" a particular fiber.
    Need to remove targets which are not observed in the first pass but in the next one.
    Targets are expected to be scattered on all MPI processes. Tiles should be load on each rank.

    Parameters
    ----------
    targets : array
        Array containing at least: 'RA', 'DEC', 'NUMOBS', 'AVAILABLE' of shape targets.size and 'FIBER', 'OBS_PASS' of shape (targets.size, npasses)

    tiles : array
        Array containing surveyops info. Can be build with :func:``_build_tiles``

    npasses : int
        Number of passes during the fiber assignment.

    mpicomm : MPI communicator
        The current MPI communicator.
    """
    import desimodel.footprint

    start = MPI.Wtime()

    if mpicomm.rank == 0: logger.info('Start completeness weight computation')

    # We will use only available targets which are not observed !
    not_used_for_comp_weight = targets['AVAILABLE'] & (targets['NUMOBS'] == 0)
    nbr_targets_for_comp_weight = mpicomm.gather(not_used_for_comp_weight.sum(), root=0)
    if mpicomm.rank == 0: logger.info(f'Starting completeness weight with {np.sum(nbr_targets_for_comp_weight)} unobserved but available targets')
    # Create Comp weight column to store it
    targets['COMP_WEIGHT'] = np.ones(targets.size)
    targets['COMP_WEIGHT'][targets['NUMOBS'] == 0] = np.nan

    for pass_id in range(npasses):
        tiles_in_pass = tiles[tiles['PASS'] == pass_id]
        if mpicomm.rank == 0: logger.debug(f'Pass {pass_id} with {tiles_in_pass.shape[0]} potential tiles')

        # Since we consider only one pass at each time find_tiles_over_point can return only at least one tile for each target
        tile_id = np.array([tiles_in_pass['TILEID'].values[idx[0]] if len(idx) != 0 else -1
                            for idx in desimodel.footprint.find_tiles_over_point(tiles_in_pass, targets['RA'], targets['DEC'])])
        # keep only targets in the correct pass
        sel_targets_in_pass = (tile_id >= 0)
        # create subcatalog
        targets_in_pass = targets['AVAILABLE', 'COMP_WEIGHT'][sel_targets_in_pass]
        targets_in_pass['TILEID', 'OBS_PASS', 'FIBER', 'NOT_USED_FOR_COMP_WEIGHT', 'index'] = [tile_id[sel_targets_in_pass], targets['OBS_PASS'][sel_targets_in_pass, pass_id],
                                                                                               targets['FIBER'][sel_targets_in_pass, pass_id],
                                                                                               not_used_for_comp_weight[sel_targets_in_pass],
                                                                                               targets_in_pass.cindex()]
        # Copy unique identification to perform sanity check at the end
        index = targets_in_pass['index']
        # Sort data to have same number of tileid in each rank
        targets_in_pass = targets_in_pass.csort('TILEID', size='orderby_counts')

        # Which tiles are treated on the current process
        sel_tiles_in_process = np.isin(tiles_in_pass['TILEID'], targets_in_pass['TILEID'])
        if sel_tiles_in_process.sum() != 0:
            _compute_completeness_weight_one_pass(tiles_in_pass[sel_tiles_in_process], targets_in_pass)

        # Put the new data in the intial order
        targets_in_pass = targets_in_pass.csort('index', size=sel_targets_in_pass.sum())
        # Check if we find the correct initial order
        assert np.all(targets_in_pass['index'] == index)

        # Update the targets before starting a new pass
        targets['COMP_WEIGHT'][sel_targets_in_pass] = targets_in_pass['COMP_WEIGHT']
        not_used_for_comp_weight[sel_targets_in_pass] = targets_in_pass['NOT_USED_FOR_COMP_WEIGHT']

        nbr_targets_for_comp_weight = mpicomm.gather(not_used_for_comp_weight.sum(), root=0)
        if mpicomm.rank == 0: logger.info(f'   * After pass: {pass_id} it remains {np.sum(nbr_targets_for_comp_weight)} targets available unobserved to compute completeness weight')

    mpicomm.Barrier()
    if mpicomm.rank == 0: logger.info(f'Completeness weight computed in elapsed time {MPI.Wtime() - start:2.2f} s.')


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
    npasses = 1
    use_sky_targets = False  # to debug: Use false to speed up the process
    preload_sky_targets = False  # very useful if F.A. is applied on several mocks.

    # Collect tiles from surveyops directory on which the fiber assignment will be applied
    tiles = build_tiles_for_fa(release_tile_path=f'/global/cfs/cdirs/desi/survey/catalogs/{release}/LSS/tiles-{program.upper()}.fits', program=program, npasses=npasses)

    sky_targets = None
    if use_sky_targets and preload_sky_targets:
        # tiles is not restricted here, we will load sky_targets for all the Y1 footprint
        sky_targets = read_sky_targets(dirname='/global/cfs/cdirs/desi/users/edmondc/desi_targets/sky_targets/', tiles=tiles, program=program, mpicomm=mpicomm)

    # Get info from origin fiberassign file and setup options for F.A.
    ts = str(tiles['TILEID'][0]).zfill(6)
    fht = fitsio.read_header(f'/global/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{ts[:3]}/fiberassign-{ts}.fits.gz')
    rundate = fht['RUNDATE']
    # see fiberassign.scripts.assign.parse_assign (Can modify margins, number of sky fibers for each petal etc.)
    opts_for_fa = ["--target", " ", "--rundate", rundate, "--mask_column", "DESI_TARGET"]

    # Generate example cutsky catalog, scattered on all processes (use a high completeness region):
    cutsky = RandomCutskyCatalog(rarange=(176.5, 182.1), decrange=(-6, 5), nbar=260, seed=44, mpicomm=mpicomm)
    # Save inital ra, dec for plotting purpose (see end of this script)
    ra_ini, dec_ini = cutsky.cget('RA', mpiroot=0), cutsky.cget('DEC', mpiroot=0)

    # To apply F.A., we need to add some information as DESI_TARGET controlling the priority, number of observation per targets etc.
    # In order to speed the process, fiber assignment on each pass will be parrallelized on the number of tiles. Need also to include list of potential tiles for each target.
    # This part should be avoided if the catalog is empty on the process (not checked here).

    # Note: here for this small example, we emulate the F.A. for QSO targets. Since they have the highest priority we do not need to add other targets to mimic the real F.A.
    # To emulate the F.A. for ELG, we will want to add other targets (QSO / LRG) with correct DESI_TARGET column with random postions (it should be enough if no cross-correlation)
    # with the correct density (including the fluctuation from imaging systematics)
    # For this tiny example, we do not use reobservation for QSO with z>2.1 (could be easily done with the redshift column).

    # Remove targets without potential observation to mimic the desi footprint (just to limit the cutsky to real desi cutsky).
    # Just to not consider targets outside the footprint --> not mandatory!!
    sel = np.array([(tiles['TILEID'].values[np.array(idx, dtype='int64')].size > 0) for idx in desimodel.footprint.find_tiles_over_point(tiles, cutsky['RA'], cutsky['DEC'])])
    cutsky = cutsky[sel]
    nbr_targets = cutsky.csize
    if mpicomm.rank == 0: logger.info(f'Keep only objects which is in a tile. Working with {nbr_targets} targets')

    # Add required columns for F.A.
    cutsky['DESI_TARGET'] = 2**2 * np.ones(cutsky.size, dtype='i8')
    # Warning: the reproducibility i.e. the choice of target when multiple targets are available is done via SUBPRIORITY. Need random generator invariant under MPI scaling!
    cutsky['SUBPRIORITY'] = cutsky.rng(seed=123).uniform(low=0, high=1, dtype='f8')
    cutsky['OBSCONDITIONS'] = 3 * np.ones(cutsky.size, dtype='i8')
    cutsky['NUMOBS_MORE'] = np.ones(cutsky.size, dtype='i8')
    # Take care with MPI! TARGETID has to be unique!
    cumsize = np.cumsum([0] + mpicomm.allgather(cutsky.size))[mpicomm.rank]
    cutsky['TARGETID'] = cumsize + np.arange(cutsky.size)

    # Columns needed to run the F.A. and collect the info (they will be exchange between processes during the F.A.)
    columns_for_fa = ['RA', 'DEC', 'TARGETID', 'DESI_TARGET', 'SUBPRIORITY', 'OBSCONDITIONS', 'NUMOBS_MORE']

    # Let's do the F.A.:
    apply_fiber_assignment(cutsky, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=use_sky_targets, sky_targets=sky_targets)
    # Compute the completeness weight: if multi-tracer, apply completeness weight once for each tracer independently
    compute_completeness_weight(cutsky, tiles, npasses, mpicomm)

    # Summarize and plot
    ra, dec = cutsky.cget('RA', mpiroot=0), cutsky.cget('DEC', mpiroot=0)
    numobs, available = cutsky.cget('NUMOBS', mpiroot=0), cutsky.cget('AVAILABLE', mpiroot=0)
    obs_pass, comp_weight = cutsky.cget('OBS_PASS', mpiroot=0), cutsky.cget('COMP_WEIGHT', mpiroot=0)

    if mpicomm.rank == 0:
        import matplotlib.pyplot as plt

        logger.info(f"Nbr of targets observed: {(numobs >= 1).sum()} -- per pass: {obs_pass.sum(axis=0)} -- Nbr of targets available: {available.sum()} -- Nbr of targets: {ra.size}")
        logger.info(f"In percentage: Observed: {(numobs >= 1).sum()/ra.size:2.2%} -- Available: {available.sum()/ra.size:2.2%}")
        values, counts = np.unique(comp_weight, return_counts=True)
        logger.info(f'Sanity check for completeness weight: {available.sum() - (numobs >= 1).sum()} avialable unobserved targets and {np.nansum([(val - 1) * count for val, count in zip(values, counts)])} from completeness counts')
        logger.info(f'Completeness counts: {values} -- {counts}')

        tiles = tiles[tiles['PASS'] < npasses]
        tile_id = np.unique(np.concatenate([tiles['TILEID'].values[np.array(idx, dtype='int64')] for idx in desimodel.footprint.find_tiles_over_point(tiles, ra, dec)]))

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        ax = axs[0]
        for id in tile_id:
            tile = tiles[tiles['TILEID'] == id]
            c = plt.Circle((tile['RA'].values[0], tile['DEC'].values[0]), np.sqrt(8 / 3.14), color='lightgrey', alpha=1)
            ax.add_patch(c)

        ax.scatter(ra_ini, dec_ini, c='red', s=0.3, label='random')
        ax.scatter(ra, dec, c='green', s=0.3, label='in desi footprint')
        ax.legend()
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylabel('Dec. [deg]')

        ax = axs[1]
        for id in tile_id:
            tile = tiles[tiles['TILEID'] == id]
            c = plt.Circle((tile['RA'].values[0], tile['DEC'].values[0]), np.sqrt(8 / 3.14), color='lightgrey', alpha=1)
            ax.add_patch(c)

        ax.scatter(ra, dec, c='red', s=0.3, label='targets')
        ax.scatter(ra[available], dec[available], c='orange', s=0.3, label=f'available: {available.sum() / ra.size:2.2%}')

        from matplotlib.axes._axes import _log as matplotlib_axes_logger
        matplotlib_axes_logger.setLevel('ERROR')
        colors = plt.cm.BuGn(np.linspace(0.6, 1, npasses))
        for i in range(npasses):
            if i == (npasses - 1):
                # ax.scatter(ra[obs_pass[:, i]], dec[obs_pass[:, i]], c=colors[i], s=0.3, label=f'Pass {i}: {obs_pass[:, i].sum() / ra.size:2.2%}')
                ax.scatter(ra[obs_pass[:, i]], dec[obs_pass[:, i]], c=colors[i], s=0.3, label=f'Pass {i}: {(numobs >= 1).sum() / ra.size:2.2%}')
            else:
                ax.scatter(ra[obs_pass[:, i]], dec[obs_pass[:, i]], c=colors[i], s=0.3, label=f'Pass {i}')
        ax.legend()
        ax.set_xlabel('R.A. [deg]')

        plt.suptitle(f'Fiber assignment for {release} release - {program} program - {npasses} passes', y=0.96)

        plt.tight_layout()
        plt.savefig(f'fiberasignment_{npasses}npasses.png')
        plt.close()
        logger.info(f'Plot save in fiberasignment_{npasses}npasses.png')

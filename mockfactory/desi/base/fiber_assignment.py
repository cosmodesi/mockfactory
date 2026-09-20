"""
Script to apply fiber assignment from pre-loaded catalogs.

This example can be run with `srun -n 5 python fiber_assignment.py` (will take typically 1 minute for 1 pass),
but one will typically import:
```
from mockfactory.desi.base import apply_fiber_assignment
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


#if os.getenv('DESI_TARGET') is None:
logger.debug('We are using DESI_TARGET = /dvs_ro/cfs/projectdirs/desi/target/')
os.environ['DESI_TARGET'] = '/dvs_ro/cfs/projectdirs/desi/target/'
logger.debug('We are using DESIMODEL = /dvs_ro/common/software/desi/perlmutter/desiconda/current/code/desimodel/main')
os.environ["DESIMODEL"] = '/dvs_ro/common/software/desi/perlmutter/desiconda/current/code/desimodel/main'


def build_tiles_for_fa(release_tile_path='/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/tiles-DARK.fits',
                       surveyops_tile_path='/dvs_ro/cfs/cdirs/desi/survey/ops/surveyops/trunk/ops/tiles-main.ecsv',
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


def read_sky_targets(tiles):
    """
    To avoid to deal with /global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/ and its 800 fits files in each directory, we read directly the sky targets saved for each tile during the main fiberassign for the real data. 
    """

    fn = '/dvs_ro/cfs/cdirs/desi/survey/fiberassign/main/{}/{}-sky.fits'
    fns = [fn.format(f"{tileid:06d}"[:3], f"{tileid:06d}") for tileid in tiles['TILEID']]

    from mpytools import Catalog
    # mpicomm=MPI.COMM_SELF -> read files only on this local process.
    sky_targets = Catalog.read(fns, filetype='fits', mpicomm=MPI.COMM_SELF)
    sky_targets = sky_targets.to_array()

    return sky_targets


def fafns_for_tiles(tiles):
    """ To speed up stuck_on_sky. See: https://github.com/desihub/fiberassign/pull/471 """
    fn = "/dvs_ro/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{}/fiberassign-{}.fits.gz"
    fns = [fn.format(f"{tileid:06d}"[:3], f"{tileid:06d}") for tileid in tiles['TILEID']]
    return ','.join(fns)


def _run_assign_init(args, tile, targets, plate_radec=True, use_sky_targets=True):
    """
    Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py#L281

    Instead of reading files, use preloaded targets and tiles.
    """
    from fiberassign.hardware import load_hardware

    def convert_tiles_to_fiberassign(args, tile):
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
            obsmjd = [obsdate.mjd, ] * len(tile)
            obsdatestr = [obsdate.isot, ] * len(tile)
        elif "OBSDATE" in tile.names:
            # We have the obsdate for every tile in the file.
            obsdate = [astropy.time.Time(x) for x in tile["OBSDATE"]]
            obsmjd = [x.mjd for x in obsdate]
            obsdatestr = [x.isot for x in obsdate]
        else:
            # default to middle of the survey
            obsdate = astropy.time.Time('2022-07-01')
            #obsmjd = [obsdate.mjd, ] * tiles.shape[0]
            #obsdatestr = [obsdate.isot, ] * tiles.shape[0]
            obsmjd = [obsdate.mjd, ] * len(tile)
            obsdatestr = [obsdate.isot, ] * len(tile)

        # Eventually, call a function from desimodel to query the field
        # rotation and hour angle for every tile time.
        if args.fieldrot is None:
            theta_obs = list()
            for tra, tdec, mjd in zip(tile["RA"], tile["DEC"], obsmjd):
                th = field_rotation_angle(tra, tdec, mjd)
                theta_obs.append(th)
            theta_obs = np.array(theta_obs)
        else:
            theta_obs = np.zeros(len(tile), dtype=np.float64)
            theta_obs[:] = args.fieldrot

        # default to zero Hour Angle; may be refined later
        ha_obs = np.zeros(len(tile), dtype=np.float64)
        if args.ha is not None:
            ha_obs[:] = args.ha

        return Tiles(tile["TILEID"], tile["RA"], tile["DEC"], tile["OBSCONDITIONS"], obsdatestr, theta_obs, ha_obs)

    def convert_targets_to_fiberassign(args, targets, tile, program, use_sky_targets=True):
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
            # Add read sky targets
            sky_targets = read_sky_targets(tile)

            # Add sky targets to fiberassign Class objects
            load_target_table(tgs, tagalong, sky_targets, survey=tgs.survey(), typecol=args.mask_column,
                                sciencemask=args.sciencemask, stdmask=args.stdmask, skymask=args.skymask,
                                safemask=args.safemask, excludemask=args.excludemask, gaia_stdmask=args.gaia_stdmask,
                                rundate=args.rundate)

        return tgs, tagalong

    # Read hardware properties
    t_start = MPI.Wtime()
    fafn = "/dvs_ro/cfs/cdirs/desi/target/fiberassign/tiles/trunk/{}/fiberassign-{}.fits.gz"
    rundate = fitsio.read_header(fafn.format(f"{tile['TILEID'][0]:06d}"[:3], f"{tile['TILEID'][0]:06d}"), 0)['RUNDATE']
    hw = load_hardware(rundate=rundate)
    logger.debug(f'                **** load_hardware took: {MPI.Wtime() - t_start:2.2f} s.')

    # Convert target to fiberassign.Targets Class
    t_start = MPI.Wtime()
    tgs, tagalong = convert_targets_to_fiberassign(args, targets, tile, tile['PROGRAM'][0], use_sky_targets=use_sky_targets)
    logger.debug(f'                **** convert_targets_to_fiberassign took: {MPI.Wtime() - t_start:2.2f} s.')

    # Convert tiles to fiberassign.Tiles Class
    t_start = MPI.Wtime()
    tile = convert_tiles_to_fiberassign(args, tile)
    logger.debug(f'                **** convert_tiles_to_fiberassign took: {MPI.Wtime() - t_start:2.2f} s.')

    return (hw, tile, tgs, tagalong)


def _run_assign_full(args, hw, tiles, tgs, tagalong):
    """
    Run fiber assignment over all tiles simultaneously.

    Adapted from https://github.com/desihub/fiberassign/blob/8e6e8264bf80fde07162de5e3f5343c621d65e3e/py/fiberassign/scripts/assign.py
    """
    from fiberassign.stucksky import stuck_on_sky #, stuck_on_sky_from_fafns
    from fiberassign.assign import Assignment, run
    from fiberassign.targets import TargetsAvailable, LocationsAvailable, targets_in_tiles

    ## TO BE REMOVED ONCE updated in fiberassign code.
    def stuck_on_sky_from_fafns(fafns):
        '''
        Retrieve the information if STUCK positioners land on good SKY locations from a list of fiberassign-TILEID.fits.gz files.

        Args:
            fafns: comma-separated list of full paths to fiberassign-TILEID.fits.gz files.

        Returns a nested dict:
            stuck_sky[tileid][loc] = bool_good_sky
        '''

        from fiberassign.hardware import FIBER_STATE_STUCK, FIBER_STATE_BROKEN
        from fiberassign.targets import TARGET_TYPE_SKY

        stuck_sky = dict()

        for fafn in fafns.split(","):

            hdr = fitsio.read_header(fafn, 0)
            tile_id = hdr["TILEID"]

            stuck_sky[tile_id] = dict()
            # FIBERASSIGN
            d = fitsio.read(fafn, "FIBERASSIGN", columns=["LOCATION", "FIBER", "FIBERSTATUS", "OBJTYPE"])
            sel = (d["FIBERSTATUS"] & (FIBER_STATE_STUCK | FIBER_STATE_BROKEN)) == FIBER_STATE_STUCK
            for loc, good in zip(d["LOCATION"][sel], d["OBJTYPE"][sel] == "SKY"):
                stuck_sky[tile_id][loc] = good
            # ETC
            d = fitsio.read(fafn, "SKY_MONITOR", columns=["LOCATION", "FA_TYPE"])
            for loc, good in zip(d["LOCATION"], (d["FA_TYPE"] & TARGET_TYPE_SKY) > 0):
                stuck_sky[tile_id][loc] = good
            # re-order by increasing locations, to reproduce stuck_on_sky()
            stuck_sky[tile_id] = dict(sorted(stuck_sky[tile_id].items()))

        return stuck_sky

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    # Find targets within tiles, and project their RA,Dec positions
    # into focal-plane coordinates.
    t_start = MPI.Wtime()
    tile_targetids, tile_x, tile_y, tile_xy_cs5 = targets_in_tiles(hw, tgs, tiles, tagalong)
    logger.debug(f'                **** targets_in_tiles took: {MPI.Wtime() - t_start:2.2f} s.')

    # Compute the targets available to each fiber for each tile.
    t_start = MPI.Wtime()
    tgsavail = TargetsAvailable(hw, tiles, tile_targetids, tile_x, tile_y)
    logger.debug(f'                **** TargetsAvailable took: {MPI.Wtime() - t_start:2.2f} s.')

    # Free the target locations
    del tile_targetids, tile_x, tile_y

    # Compute the fibers on all tiles available for each target and sky
    t_start = MPI.Wtime()
    favail = LocationsAvailable(tgsavail)
    logger.debug(f'                **** LocationsAvailable took: {MPI.Wtime() - t_start:2.2f} s.')

    # Find stuck positioners and compute whether they will land on acceptable
    # sky locations for each tile.
    t_start = MPI.Wtime()
    if args.fafns_for_stucksky is not None:
        stucksky = stuck_on_sky_from_fafns(args.fafns_for_stucksky)
    else:
        stucksky = stuck_on_sky(hw, tiles, args.lookup_sky_source, rundate=getattr(args, 'rundate', None))
    logger.debug(f'                **** stuck_on_sky took: {MPI.Wtime() - t_start:2.2f} s.')

    # Create assignment object
    t_start = MPI.Wtime()
    asgn = Assignment(tgs, tgsavail, favail, stucksky)
    logger.debug(f'                **** Assignment took: {MPI.Wtime() - t_start:2.2f} s.')

    t_start = MPI.Wtime()
    run(asgn, args.standards_per_petal, args.sky_per_petal, args.sky_per_slitblock,
        redistribute=not args.no_redistribute, use_zero_obsremain=not args.no_zero_obsremain)
    logger.debug(f'                **** run took: {MPI.Wtime() - t_start:2.2f} s.')

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


def _apply_mtl_one_tile(targets, tg_assign, tg_available, tileid):
    """
    Proxy of true MTL. 

    Note: we apply fiber assignment pass by pass (and in each pass we apply it tile per tile to have the correct hardware). In this configuration, only one observation per target can be done in one pass.

    Use AVAILABLE for randoms. Available = can be observed with at least one fiber but not chosen by the F.A. process.
    """
    from desitarget.geomask import match
    from desitarget.targetmask import zwarn_mask
    from pandas import read_csv
    from mpytools import Catalog

    # first load which fibers are correclty worked in the real life:
    t_start = MPI.Wtime()
    # collect LASNIGHT value corresponding to the tileid
    tile_info = read_csv('/dvs_ro/cfs/cdirs/desi/spectro/redux/loa/tiles-loa.csv')
    lastnight = tile_info['LASTNIGHT'][tile_info['TILEID'] == tileid].values[0]  # not super user-friendly.. 
    # read zmtl files (Note: we want to work with the daily version since this is what the MTL works with)
    # warning, these data directory are shit; files are missing when something goes wrong ... 
    zmtl_fn = [f'/dvs_ro/cfs/cdirs/desi/spectro/redux/daily/tiles/cumulative/{tileid}/{lastnight}/zmtl-{petal}-{tileid}-thru{lastnight}.fits' for petal in range(10)]
    zwarn = []
    for i in range(10):
        if os.path.isfile(zmtl_fn[i]):
            zwarn.append(Catalog.read(zmtl_fn[i], filetype='fits', mpicomm=MPI.COMM_SELF)['ZWARN']) # read files only on this local process.
        else:
            zwarn.append(zwarn_mask.mask("BAD_PETALQA")*np.ones(500, dtype='int'))
    zwarn = np.concatenate(zwarn)
    # reject fibers without data + with bad petal + with bad spectrograph quality
    sel_good_fibers = ~((zwarn & zwarn_mask["NODATA"] != 0) | (zwarn & zwarn_mask.mask("BAD_SPECQA|BAD_PETALQA") != 0))
    # keep only good fibers 
    sel = np.in1d(tg_assign["FIBER"], np.arange(0, 5000, 1)[sel_good_fibers])
    tg_assign = {name: tg_assign[name][sel] for name in tg_assign}
    sel = np.in1d(tg_available["FIBER"], np.arange(0, 5000, 1)[sel_good_fibers])
    tg_available = {name: tg_available[name][sel] for name in tg_available}
    logger.debug(f'                **** keep only tg_available in good fibers took: {MPI.Wtime() - t_start:2.2f} s.')

    # update the targets list for available target:
    idx, idx2 = match(targets['TARGETID'], tg_available['TARGETID'])
    targets['AVAILABLE'][idx] = True
    targets['FIBER'][idx] = np.array(tg_available["FIBER"][idx2], dtype='i8')
    # update the target list for targed target:
    idx, idx2 = match(targets['TARGETID'], tg_assign['TARGETID'])
    targets["NUMOBS_MORE"][idx] -= 1
    targets["NUMOBS"][idx] += 1
    targets["FIBER"][idx] = np.array(tg_assign["FIBER"][idx2], dtype='i8')  # rewrite with the correct assign fiber if several are available.
    targets["OBS_PASS"][idx] = True


def _run_fiber_assignment_one_tile(tile, targets, opts_for_fa, plate_radec=True, use_sky_targets=True):
    """
    From tiles and targets run step by step the fiber assignment process for one tile. We need to apply it tile per tile to load
    the correct hardware status for each tile...
    Note: to work with fiberassign package (ie) for _run_assign_init function,
    targets should be a dtype numpy array and not a mpytools.Catalog. Convert it with Catalog.to_array().
    """
    from fiberassign.scripts.assign import parse_assign
    mpicomm = MPI.COMM_WORLD

    # load param for firber assignment
    ag = parse_assign(opts_for_fa)

    # Add args.fafns_for_stucksky to speed up stuck_on_sky step (see: https://github.com/desihub/fiberassign/pull/471)
    ag.fafns_for_stucksky = fafns_for_tiles(tile)

    # Convert data to fiberassign class (targets should be a dtype numpy array here)
    from astropy.table import Table
    tile_new_format = Table(tile.to_records())
    t_start = MPI.Wtime()
    hw, tile, tgs, tagalong = _run_assign_init(ag, tile_new_format, targets.to_array(), plate_radec=plate_radec, use_sky_targets=use_sky_targets)
    logger.debug(f'            *** _run_assign_init took: {MPI.Wtime() - t_start:2.2f} s.')

    # run assignment
    t_start = MPI.Wtime()
    asgn = _run_assign_full(ag, hw, tile, tgs, tagalong)
    logger.debug(f'            *** _run_assign_full took: {MPI.Wtime() - t_start:2.2f} s.')
    
    # from assignment collect which targets is selected and available (useful for randoms !)
    t_start = MPI.Wtime()
    tg_assign, tg_available = _extract_info_assignment(asgn)
    logger.debug(f'            *** _extract_info_assignment took: {MPI.Wtime() - t_start:2.2f} s.')
    
    # update targets with 'the observation'
    t_start = MPI.Wtime()
    _apply_mtl_one_tile(targets, tg_assign, tg_available, tile_new_format['TILEID'][0])
    logger.debug(f'            *** _apply_mtl_one_pass took: {MPI.Wtime() - t_start:2.2f} s.')


def _run_fiber_assignment_one_pass(tiles, targets, opts_for_fa, plate_radec=True, use_sky_targets=True):
    """
    From tiles and targets run the fiber assignment process for one pass, need to apply it tile per tile to load the correct hardware status.
    """
    from fiberassign.scripts.assign import parse_assign

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD

    # load param for firber assignment
    ag = parse_assign(opts_for_fa)

    for idx_tile in range(tiles['TILEID'].size):
        # take care with pd.DataFrame
        tile = tiles[idx_tile:idx_tile+1]

        t_start = MPI.Wtime()
        _run_fiber_assignment_one_tile(tile, targets, opts_for_fa, plate_radec=plate_radec, use_sky_targets=use_sky_targets)
        logger.debug(f'            *** 1 tile took: {MPI.Wtime() - t_start:2.2f} s.')


def apply_fiber_assignment(targets, tiles, npasses, opts_for_fa, columns_for_fa, mpicomm, use_sky_targets=True):
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

    csize = mpy.gather(targets.size, mpiroot=None)
    logger.info(f'Start fiber assignment for {csize.sum()} objects and {npasses} pass(es): use sky targets? {use_sky_targets}')

    # Add columns to collect fiber assign output !
    targets['NUMOBS', 'AVAILABLE', 'FIBER', 'OBS_PASS'] = [np.zeros(targets.size, dtype='i8'), np.zeros(targets.size, dtype='?'),
                                                           -1 * np.ones((targets.size, npasses), dtype='i8'),  # To compute the completeness weight, need to collect at least (if avialable) one fiber per pass!
                                                           np.zeros((targets.size, npasses), dtype='?')]  # Take care to the size --> with mpytools should be (size, X) and not (X, size)

    for pass_id in range(npasses):
        tiles_in_pass = tiles[tiles['PASS'] == pass_id]
        logger.info(f'    * Pass {pass_id} with {tiles_in_pass.shape[0]} potential tiles')
        t_start = MPI.Wtime()
        # Since we consider only one pass at each time find_tiles_over_point can return only at most one tile for each target
        tile_id = np.array([tiles_in_pass['TILEID'].values[idx[0]] if len(idx) != 0 else -1
                            for idx in desimodel.footprint.find_tiles_over_point(tiles_in_pass, targets['RA'], targets['DEC'])])
        # keep only targets in the correct pass and with potential observation
        sel_targets_in_pass = (tile_id >= 0) & (targets["NUMOBS_MORE"] >= 1)
        # create subcatalog
        targets_in_pass = targets[columns_for_fa + ['NUMOBS', 'AVAILABLE']][sel_targets_in_pass]
        targets_in_pass['TILEID', 'OBS_PASS', 'FIBER', 'index'] = [tile_id[sel_targets_in_pass], np.zeros(sel_targets_in_pass.sum(), dtype='?'),
                                                                   -1 * np.ones(sel_targets_in_pass.sum(), dtype='i8'),
                                                                   targets_in_pass.cindex()]
        # Copy unique identification to perform sanity check at the end
        index = targets_in_pass['index']
        mpicomm.Barrier()
        logger.debug(f'        ** Build targets_in_pass took: {MPI.Wtime() - t_start:2.2f} s.')

        # Sort data to have same number of tileid in each rank
        t_start = MPI.Wtime()
        targets_in_pass = targets_in_pass.csort('TILEID', size='orderby_counts')
        mpicomm.Barrier()
        logger.debug(f'        ** Particle exchange between all the processes took: {MPI.Wtime() - t_start:2.2f} s.')
        nbr_tiles = mpy.gather(np.unique(targets_in_pass['TILEID']).size, mpiroot=None)
        logger.debug(f'        ** Number of tiles to process per rank = {np.min(nbr_tiles)} - {np.max(nbr_tiles)} (min - max).')

        # Which tiles are treated on the current process
        t_start = MPI.Wtime()
        sel_tiles_in_process = np.isin(tiles_in_pass['TILEID'], targets_in_pass['TILEID'])
        # run F.A. only on these tiles
        if sel_tiles_in_process.sum() != 0:
            _run_fiber_assignment_one_pass(tiles_in_pass[sel_tiles_in_process], targets_in_pass, opts_for_fa, use_sky_targets=use_sky_targets)
        mpicomm.Barrier()
        logger.debug(f'        ** Apply F.A. Pass {pass_id} took: {MPI.Wtime() - t_start:2.2f} s.')

        # Put the new data in the intial order
        t_start = MPI.Wtime()
        targets_in_pass = targets_in_pass.csort('index', size=sel_targets_in_pass.sum())
        # Check if we find the correct initial order
        assert np.all(targets_in_pass['index'] == index)
        mpicomm.Barrier()
        logger.debug(f'        ** Particle exchange between all the processes took: {MPI.Wtime() - t_start:2.2f} s.')

        # Update the targets before starting a new pass (do it one by one)
        for col in ['NUMOBS_MORE', 'NUMOBS', 'AVAILABLE']:
            targets[col][sel_targets_in_pass] = targets_in_pass[col]
        targets['OBS_PASS'][sel_targets_in_pass, pass_id] = targets_in_pass['OBS_PASS']
        targets['FIBER'][sel_targets_in_pass, pass_id] = targets_in_pass['FIBER']

    mpicomm.Barrier()
    logger.info(f'Apply fiber assign performed in elapsed time {MPI.Wtime() - start:2.2f} s.')


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

    # We will use only available targets which are not observed !
    not_used_for_comp_weight = targets['AVAILABLE'] & (targets['NUMOBS'] == 0)
    nbr_targets_for_comp_weight = mpicomm.gather(not_used_for_comp_weight.sum(), root=0)
    logger.info(f'Starting completeness weight with {np.sum(nbr_targets_for_comp_weight)} unobserved but available targets')
    # Create Comp weight column to store it
    targets['COMP_WEIGHT'] = np.ones(targets.size)
    targets['COMP_WEIGHT'][targets['NUMOBS'] == 0] = np.nan

    for pass_id in range(npasses):
        tiles_in_pass = tiles[tiles['PASS'] == pass_id]
        logger.debug(f'Pass {pass_id} with {tiles_in_pass.shape[0]} potential tiles')

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
        logger.debug(f'   * After pass: {pass_id} it remains {np.sum(nbr_targets_for_comp_weight)} targets available unobserved to compute completeness weight')

    mpicomm.Barrier()
    logger.info(f'Completeness weight computed in elapsed time {MPI.Wtime() - start:2.2f} s.')


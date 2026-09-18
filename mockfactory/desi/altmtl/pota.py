"""
Potential assignments.

For every tile, which targets each fiber *could* have reached, whether or not it did. This is
the denominator the assignment is measured against, and it does not depend on the alternative
ledgers at all: it is a property of the focal plane, the tile pointing and the target catalog.

Collisions are recorded too: a target a fiber can reach, but only by colliding with a
neighbouring positioner, is not really available.
"""

import os
import bisect
import logging
from datetime import datetime, timezone

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.pota')


#: The focal plane radius, with the margin the real survey uses when selecting targets per tile.
TILE_RADIUS_MARGIN = 1.1


def _parse_datetime(date):
    """Parse a fiberassign run date, assuming utc when it carries no timezone."""
    try:
        return datetime.strptime(date, '%Y-%m-%dT%H:%M:%S%z')
    except ValueError:
        return datetime.strptime(date, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc)


class _HardwareCache(object):
    """
    Focal plane state, keyed by the time range it is valid over.

    Loading the hardware takes seconds, and the thousands of tiles of a survey share a few
    hundred distinct run dates, so this is worth caching.
    """
    def __init__(self, margins=None):
        if margins is None:
            from fiberassign.hardware import get_default_exclusion_margins
            margins = get_default_exclusion_margins()
        self.margins = margins
        self.ranges = []

    def get(self, rundate):
        from fiberassign.hardware import load_hardware

        date = _parse_datetime(rundate)
        for start, stop, hardware in self.ranges:
            if start <= date and (stop is None or stop > date):
                return hardware
        hardware, start, stop = load_hardware(rundate=rundate, add_margins=self.margins, get_time_range=True)
        self.ranges.append((start, stop, hardware))
        return hardware


def get_targets_in_tile(targets, tile, tile_radius=None):
    """
    Return the targets of ``targets`` within one tile.

    ``targets`` must be sorted by declination: the search is narrowed by declination first,
    with a bisection, which is what makes this affordable over a survey's worth of tiles.
    """
    import desimodel.focalplane
    import desimodel.footprint

    if tile_radius is None:
        tile_radius = desimodel.focalplane.get_tile_radius_deg() * TILE_RADIUS_MARGIN
    dec = targets['DEC']
    start = bisect.bisect_left(dec, tile['DEC'] - tile_radius)
    stop = bisect.bisect_left(dec, tile['DEC'] + tile_radius, lo=start)
    index = desimodel.footprint.find_points_radec(tile['RA'], tile['DEC'],
                                                  targets['RA'][start:stop + 1],
                                                  targets['DEC'][start:stop + 1])
    return targets[start + np.array(index, dtype='i8')]


def _add_assignment_columns(targets):
    """Add the columns fiberassign needs but a mock target catalog may not carry."""
    from astropy.table import Table

    targets = Table(targets)
    defaults = {'DESI_TARGET': 2, 'NUMOBS_INIT': 0, 'NUMOBS_MORE': 1, 'PRIORITY': 3400}
    for name, value in defaults.items():
        if name not in targets.colnames:
            targets[name] = np.full(len(targets), value, dtype='i8')
    if 'SUBPRIORITY' not in targets.colnames:
        targets['SUBPRIORITY'] = np.random.uniform(size=len(targets))
    # Forced rather than defaulted: the value has to match what the tile is loaded with below.
    targets['OBSCONDITIONS'] = np.full(len(targets), 516, dtype='i8')
    return targets


def compute_potential_assignments_one_tile(tile, targets, columns, output_dir, hardware_cache,
                                           header, tile_temp_dir, collisions=True):
    """
    Return the potential assignments of one tile, as a structured array.

    Parameters
    ----------
    tile : astropy.table.Row
        Tile to process.

    targets : array
        Target catalog, sorted by declination.

    columns : list
        Target columns to carry through to the output.

    output_dir : str
        Directory the per-tile target file is written to.

    hardware_cache : _HardwareCache
        Cache of focal plane states.

    header : dict
        Header of the real fiberassign file of this tile.

    tile_temp_dir : str
        Directory the single-tile footprint file is written to.

    collisions : bool, default=True
        Whether to flag target-fiber pairs that can only be reached through a collision.

    Returns
    -------
    potential : astropy.table.Table
        One row per reachable target-location pair.
    """
    import fitsio
    from astropy.table import Table, join
    from fiberassign.tiles import load_tiles
    from fiberassign.targets import Targets, TargetsAvailable, LocationsAvailable, create_tagalong, \
        load_target_file, targets_in_tiles
    from fiberassign.assign import Assignment

    tileid = int(tile['TILEID'])
    utils.mkdir(output_dir)
    utils.mkdir(tile_temp_dir)

    targets_fn = os.path.join(output_dir, 'tilenofa-{:d}.fits'.format(tileid))
    _add_assignment_columns(get_targets_in_tile(targets, tile)).write(targets_fn, format='fits', overwrite=True)

    footprint_fn = os.path.join(tile_temp_dir, '{:d}-tiles.fits'.format(tileid))
    if not os.path.isfile(footprint_fn):
        footprint = Table(tile)
        footprint['OBSCONDITIONS'] = 516
        footprint['IN_DESI'] = 1
        footprint['MTLTIME'] = header['MTLTIME']
        footprint['FA_RUN'] = header['FA_RUN']
        footprint['PROGRAM'] = header['FAPRGRM'].upper() if 'FAPRGRM' in header else 'DARK'
        footprint.write(footprint_fn, overwrite=True)

    hardware = hardware_cache.get(header['RUNDATE'])
    tiles = load_tiles(tiles_file=footprint_fn, obsha=header['FA_HA'], obstheta=header['FIELDROT'],
                       select=[tileid])

    tgs = Targets()
    tagalong = create_tagalong(plate_radec=True)
    load_target_file(tgs, tagalong, targets_fn)
    target_table = fitsio.read(targets_fn, columns=columns)

    targetid, x, y, xy_cs5 = targets_in_tiles(hardware, tgs, tiles, tagalong)
    targets_available = TargetsAvailable(hardware, tiles, targetid, x, y)
    locations_available = LocationsAvailable(targets_available)
    # No stuck-sky determination: a potential assignment is about reach, not about what a
    # positioner happened to be parked on.
    assignment = Assignment(tgs, targets_available, locations_available, {})

    available = assignment.targets_avail().tile_data(tileid)
    navailable = sum(len(available[location]) for location in available)
    fiber_of_location = dict(hardware.loc_fiber)

    potential = Table()
    potential['LOCATION'] = np.zeros(navailable, dtype='i8')
    potential['FIBER'] = np.zeros(navailable, dtype='i8')
    potential['TARGETID'] = np.zeros(navailable, dtype='i8')
    offset = 0
    # Sorted by location, then by target, matching the real survey's available-targets table.
    for location in sorted(available):
        tids = sorted(available[location])
        potential['LOCATION'][offset:offset + len(tids)] = location
        potential['FIBER'][offset:offset + len(tids)] = fiber_of_location[location]
        potential['TARGETID'][offset:offset + len(tids)] = tids
        offset += len(tids)

    potential = join(potential, Table(target_table), keys=['TARGETID'], join_type='left')
    if collisions:
        collided = assignment.check_avail_collisions(tileid)
        keys = np.array(list(collided.keys())).transpose()
        locids = keys[1] * 10000 + keys[0]
        potential['COLLISION'] = np.isin(potential['TARGETID'] * 10000 + potential['LOCATION'], locids)
    potential['TILEID'] = tileid
    return potential


def compute_potential_assignments(targets_fn, output_fn, tiles_fn, program='DARK', numproc=1,
                                  collisions=True, output_dir=None, tile_temp_dir=None,
                                  fiberassign_dir=None):
    """
    Compute the potential assignments of a whole survey, and write them to ``output_fn``.

    Parameters
    ----------
    targets_fn : str
        Path of the mock target catalog.

    output_fn : str
        Path of the potential assignment file to write.

    tiles_fn : str
        Path of the tile file of the release being reproduced, e.g. ``tiles-DARK.fits``.

    program : str, default='DARK'
        Program, 'DARK' or 'BRIGHT'.

    numproc : int, default=1
        Number of processes to run tiles with.

    collisions : bool, default=True
        Whether to flag target-fiber pairs reachable only through a collision.

    output_dir : str, default=None
        Directory for the per-tile target files. Defaults to next to ``output_fn``.

    tile_temp_dir : str, default=None
        Directory for the single-tile footprint files. Defaults to ``output_dir``.

    fiberassign_dir : str, default=None
        Directory of the real fiberassign files.

    Returns
    -------
    output_fn : str
        Path of the file that was written.
    """
    import fitsio
    from astropy.table import Table

    if output_dir is None: output_dir = os.path.join(os.path.dirname(output_fn), 'tartiles')
    if tile_temp_dir is None: tile_temp_dir = output_dir

    targets = fitsio.read(targets_fn)
    columns = list(targets.dtype.names)
    # The per-tile selection bisects on declination, so the catalog has to be sorted by it.
    if not np.all(targets['DEC'][:-1] <= targets['DEC'][1:]):
        logger.info('Sorting {:d} targets by declination.'.format(targets.size))
        targets = targets[np.argsort(targets['DEC'])]

    tiles = Table.read(tiles_fn)
    logger.info('Computing potential assignments for {:d} {} tiles.'.format(len(tiles), program))

    headers = _read_headers([int(tileid) for tileid in tiles['TILEID']], fiberassign_dir=fiberassign_dir,
                            numproc=numproc)
    hardware_cache = _HardwareCache()
    # Loading the hardware is not thread safe and the run dates repeat, so warm the cache first.
    for rundate in sorted({header['RUNDATE'] for header in headers}):
        hardware_cache.get(rundate)
    logger.info('Loaded {:d} distinct focal plane state(s).'.format(len(hardware_cache.ranges)))

    utils.mkdir(os.path.dirname(output_fn))
    tmp_fn = output_fn + '.tmp'
    fits = fitsio.FITS(tmp_fn, 'rw', clobber=True)
    ntotal = ncollision = 0
    for itile, (tile, header) in enumerate(zip(tiles, headers)):
        potential = np.array(compute_potential_assignments_one_tile(
            tile, targets, columns, output_dir, hardware_cache, header, tile_temp_dir,
            collisions=collisions))
        ntotal += potential.size
        if collisions: ncollision += int(potential['COLLISION'].sum())
        if itile == 0:
            fits.write(potential, extname='LSS')
        else:
            fits[-1].append(potential)
        if (itile + 1) % 200 == 0:
            logger.info('Processed {:d}/{:d} tiles.'.format(itile + 1, len(tiles)))
    fits.close()
    os.rename(tmp_fn, output_fn)
    logger.info('Wrote {:d} potential assignments ({:d} collisions) to {}.'.format(
        ntotal, ncollision, output_fn))
    return output_fn


def _read_header(tileid_and_dir):
    import fitsio
    tileid, fiberassign_dir = tileid_and_dir
    header = fitsio.read_header(utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir))
    return {name: header[name] for name in ['RUNDATE', 'MTLTIME', 'FA_RUN', 'FA_HA', 'FIELDROT']}


def _read_headers(tileids, fiberassign_dir=None, numproc=1):
    args = [(tileid, fiberassign_dir) for tileid in tileids]
    if numproc > 1:
        from multiprocessing import Pool
        with Pool(processes=numproc) as pool:
            return pool.map(_read_header, args)
    return [_read_header(arg) for arg in args]

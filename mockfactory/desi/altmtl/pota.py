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
import time
from datetime import datetime, timezone

import numpy as np

from .targets import read_targets

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


def embed_sphere(ra, dec):
    """Unit vectors of sky coordinates, which is what a tree over the sky has to be built on."""
    phi = np.radians(np.asarray(ra, dtype='f8'))
    theta = np.radians(90. - np.asarray(dec, dtype='f8'))
    return np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=-1)


def build_target_tree(targets):
    """
    A tree over every target, to be queried once per tile.

    Built in the parent and inherited by the workers at fork, so it is paid for once a mock
    rather than once a tile. The point of a tree is many queries against one build;
    ``desimodel.footprint.find_points_radec`` does the opposite, building one over the
    candidates of a declination slice to ask about the single point of a tile centre.
    """
    from scipy.spatial import cKDTree
    return cKDTree(embed_sphere(targets['RA'], targets['DEC']))


def get_targets_in_tile(targets, tile, tile_radius=None, tree=None):
    """
    Return the targets of ``targets`` within one tile.

    ``targets`` must be sorted by declination: the search is narrowed by declination first,
    with a bisection, which is what makes this affordable over a survey's worth of tiles.
    """
    import desimodel.focalplane

    # Two radii, as before: the declination slice is padded by the margin so that it safely
    # holds everything the tile reaches, and the angular cut below is at the true tile radius.
    cut_radius = desimodel.focalplane.get_tile_radius_deg()
    if tile_radius is None:
        tile_radius = cut_radius * TILE_RADIUS_MARGIN
    threshold = 2. * np.sin(np.radians(cut_radius) / 2.)
    if tree is not None:
        # One query against the tree of the whole catalog: no declination slice to stream, and
        # workers=1 because the query is a single point and the pool is already full
        index = tree.query_ball_point(embed_sphere(tile['RA'], tile['DEC']), threshold,
                                      workers=1)
        return targets[np.sort(np.asarray(index, dtype='i8'))]

    dec = targets['DEC']
    start = bisect.bisect_left(dec, tile['DEC'] - tile_radius)
    stop = bisect.bisect_left(dec, tile['DEC'] + tile_radius, lo=start)
    # The angular cut, written out rather than asked of desimodel.footprint.find_points_radec:
    # that builds a tree over every candidate the declination slice holds, about a million of
    # them here, and then queries it at the one point of the tile centre. A tree is the wrong
    # shape for one query, and the same cut is a few operations on the slice.
    ra = np.radians(np.asarray(targets['RA'][start:stop + 1], dtype='f8'))
    sin_dec = np.sin(np.radians(np.asarray(dec[start:stop + 1], dtype='f8')))
    cos_dec = np.sqrt(1. - sin_dec**2)
    tile_ra, tile_dec = np.radians(tile['RA']), np.radians(tile['DEC'])
    # Compared on the chord, as the tree does, which is better conditioned near the edge than
    # the cosine of a small separation
    chord2 = 2. - 2. * (sin_dec * np.sin(tile_dec)
                        + cos_dec * np.cos(tile_dec) * np.cos(ra - tile_ra))
    index = np.flatnonzero(chord2 <= threshold**2)
    return targets[start + index]


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


#: What a worker needs, set before the pool is made so each inherits it at fork. A nested
#: function would be simpler and cannot be pickled.
_context = {}


def _one_tile(args):
    """One tile, taking everything but the tile itself from the inherited context."""
    tile, header = args
    return np.array(compute_potential_assignments_one_tile(
        tile, _context['targets'], _context['columns'], _context['hardware_cache'], header,
        _context['tile_temp_dir'], collisions=_context['collisions'], tree=_context['tree']))


def compute_potential_assignments_one_tile(tile, targets, columns, hardware_cache,
                                           header, tile_temp_dir, collisions=True,
                                           survey='main', tree=None):
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
        load_target_table, targets_in_tiles
    from fiberassign.assign import Assignment

    tileid = int(tile['TILEID'])
    utils.mkdir(tile_temp_dir)

    tile_targets = _add_assignment_columns(get_targets_in_tile(targets, tile, tree=tree))

    footprint_fn = os.path.join(tile_temp_dir, '{:d}-tiles.fits'.format(tileid))
    if not os.path.isfile(footprint_fn):
        footprint = Table(tile)
        footprint['OBSCONDITIONS'] = 516
        footprint['IN_DESI'] = 1
        footprint['MTLTIME'] = header['MTLTIME']
        footprint['FA_RUN'] = header['FA_RUN']
        footprint['PROGRAM'] = header['FAPRGRM'].upper() if 'FAPRGRM' in header else 'DARK'
        # Through a temporary name and a rename, which is atomic: the guard above is a
        # check and not a lock, so with a pool of workers a reader can otherwise reach a file
        # another one is halfway through writing, and load_tiles rejects it as corrupt.
        # The format is named rather than inferred: the temporary name does not end in .fits,
        # and astropy infers from the extension.
        tmp_footprint_fn = '{}.{:d}.tmp'.format(footprint_fn, os.getpid())
        footprint.write(tmp_footprint_fn, format='fits', overwrite=True)
        os.replace(tmp_footprint_fn, footprint_fn)

    hardware = hardware_cache.get(header['RUNDATE'])
    tiles = load_tiles(tiles_file=footprint_fn, obsha=header['FA_HA'], obstheta=header['FIELDROT'],
                       select=[tileid])

    tgs = Targets()
    tagalong = create_tagalong(plate_radec=True)
    # Handed to fiberassign as an array rather than written out and read back. The file cost
    # three passes over the same rows -- one to write it, one inside fiberassign, one here --
    # for something this function has already built in memory.
    load_target_table(tgs, tagalong, tile_targets, survey=survey)
    # As a plain structured array: writing it to FITS and reading it back used to do this
    # implicitly, and the join below carries whatever it is given into the output.
    target_table = tile_targets[columns].as_array()

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

    # astropy's join, not tables.join_left: it returns the rows sorted by target, with the
    # locations of one target in the order of an unstable sort, and every potential assignment
    # file written so far holds that order. The catalogs built from them depend on it, through
    # the row order their random draws follow.
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

    targets = read_targets(targets_fn)
    columns = list(targets.colnames)
    # The per-tile selection bisects on declination, so the catalog has to be sorted by it.
    if not np.all(targets['DEC'][:-1] <= targets['DEC'][1:]):
        logger.info('Sorting {:d} targets by declination.'.format(len(targets)))
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

    start = time.time()
    tree = build_target_tree(targets)
    logger.info('Built a tree over {:d} targets in {:.0f} s.'.format(len(targets), time.time() - start))
    _context.update(targets=targets, columns=columns, hardware_cache=hardware_cache,
                    tile_temp_dir=tile_temp_dir, collisions=collisions, tree=tree)

    # The tiles are independent, and the focal plane cache is warmed above so every worker
    # inherits it rather than loading the hardware itself, which is not thread safe. The
    # results are consumed in tile order, so the output is the same as a serial loop would
    # give; ProcessPoolExecutor rather than a pool, because a pool quietly replaces a worker
    # the kernel kills and then waits forever for the result it was carrying.
    work = list(zip(tiles, headers))
    if numproc > 1:
        from concurrent.futures import ProcessPoolExecutor
        pool = ProcessPoolExecutor(max_workers=numproc)
        results = pool.map(_one_tile, work, chunksize=8)
    else:
        pool, results = None, map(_one_tile, work)

    for itile, potential in enumerate(results):
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
    if pool is not None: pool.shutdown()
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

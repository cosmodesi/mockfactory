"""
Fiber assignment against the alternative ledgers.

For each tile, the real survey's own inputs are reused as they are: the footprint, the sky,
secondary, gfa and too target files, and, from the fiberassign header, the run date, field
rotation and hour angle. Only the science targets differ, coming from the alternative ledgers
rather than the real ones. The focal plane state and the sky are therefore identical to the
real assignment, and the two can be compared fiber by fiber.

That comparison is the :class:`FiberMap`: the target that the alternative survey put on fiber
``f`` stands in for the target the real survey put on fiber ``f``. It is what lets the real
redshifts be folded into the alternative ledgers.
"""

import os
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.assignment')


#: Fiberassign versions from 4.0 on run under the standard desi environment; earlier tiles
#: were assigned by versions that have to be loaded explicitly, which this module cannot do.
MIN_FIBERASSIGN_VERSION = 4.0


class FiberMap(object):
    """
    Correspondence between the real and the alternative assignment of one tile.

    Attributes
    ----------
    real_targetid : array
        Target observed by the real survey, one entry per fiber.

    alt_targetid : array
        Target the alternative survey put on that same fiber.
    """
    def __init__(self, real_targetid, alt_targetid):
        self.real_targetid = np.asarray(real_targetid)
        self.alt_targetid = np.asarray(alt_targetid)

    def __len__(self):
        return len(self.real_targetid)

    def real_to_alt(self, targetid):
        """
        Return the alternative counterpart of each real target in ``targetid``.

        Raises
        ------
        ValueError
            If a target was observed by the real survey but sits on no fiber of this tile in
            the alternative assignment. Dropping it would quietly lose an observation.
        """
        return self._map(targetid, self.real_targetid, self.alt_targetid, 'real')

    def alt_to_real(self, targetid):
        """Return the real counterpart of each alternative target in ``targetid``."""
        return self._map(targetid, self.alt_targetid, self.real_targetid, 'alternative')

    @staticmethod
    def _map(targetid, source, dest, name):
        targetid = np.asarray(targetid)
        argsort = np.argsort(source)
        index = np.searchsorted(source, targetid, sorter=argsort)
        index = np.clip(index, 0, source.size - 1)
        index = argsort[index]
        mask = source[index] != targetid
        if mask.any():
            raise ValueError('{:d} of {:d} target(s) have no {} counterpart on this tile, '
                             'e.g. {}'.format(mask.sum(), targetid.size, name, targetid[mask][:5]))
        return dest[index]

    def write(self, fn):
        """Save to ``fn``, as a structured array."""
        utils.mkdir(os.path.dirname(fn))
        array = np.empty(len(self), dtype=[('REAL_TARGETID', 'i8'), ('ALT_TARGETID', 'i8')])
        array['REAL_TARGETID'], array['ALT_TARGETID'] = self.real_targetid, self.alt_targetid
        np.save(fn, array)
        return fn

    @classmethod
    def read(cls, fn):
        """Load from ``fn``."""
        array = np.load(fn)
        return cls(array['REAL_TARGETID'], array['ALT_TARGETID'])


def get_fiber_map_fn(fa_dir, tileid):
    """Return the path of the fiber map of ``tileid``."""
    return os.path.join(fa_dir, 'famap-{}.npy'.format(utils.tile_string(tileid)))


def get_alt_fiberassign_fn(fa_dir, tileid):
    """Return the path of the alternative fiberassign file of ``tileid``."""
    return os.path.join(fa_dir, 'fba-{}.fits'.format(utils.tile_string(tileid)))


def get_fa_dir(altmtl_dir, fadate, survey='main'):
    """Return the directory holding the alternative assignment of a given assignment date."""
    return os.path.join(altmtl_dir, 'fa', survey.upper(), fadate)


def _read_assignment(fn, extnames):
    """Read the first of ``extnames`` present in ``fn``, else the first binary table."""
    import fitsio
    with fitsio.FITS(fn) as fits:
        available = [hdu.get_extname() for hdu in fits]
        for extname in extnames:
            if extname in available:
                return fits[extname].read()
        return fits[1].read()


def read_real_assignment(tileid, fiberassign_dir=None):
    """Return the real survey assignment of ``tileid``, with its FIBER and TARGETID columns."""
    return _read_assignment(utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir),
                            ['FIBERASSIGN', 'FASSIGN'])


def read_alt_assignment(fa_dir, tileid):
    """Return the alternative assignment of ``tileid``."""
    return _read_assignment(get_alt_fiberassign_fn(fa_dir, tileid), ['FASSIGN', 'FIBERASSIGN'])


def make_fiber_map(real_assignment, alt_assignment):
    """
    Build the :class:`FiberMap` pairing the two assignments of a tile fiber by fiber.

    Fibers holding no real target are dropped: they carry nothing to fold into the ledgers.
    """
    real_fiber, alt_fiber = real_assignment['FIBER'], alt_assignment['FIBER']
    real_targetid, alt_targetid = real_assignment['TARGETID'], alt_assignment['TARGETID']

    for name, fiber in [('real', real_fiber), ('alternative', alt_fiber)]:
        if np.unique(fiber).size != fiber.size:
            raise ValueError('{} assignment has repeated fibers; the map would be ambiguous'.format(name))

    # A fiber is only useful if both surveys put a target on it.
    mask_real = real_targetid >= 0
    real_fiber, real_targetid = real_fiber[mask_real], real_targetid[mask_real]
    argsort = np.argsort(alt_fiber)
    index = np.searchsorted(alt_fiber, real_fiber, sorter=argsort)
    index = argsort[np.clip(index, 0, alt_fiber.size - 1)]
    mask = alt_fiber[index] == real_fiber
    if not mask.all():
        logger.warning('{:d} real fiber(s) are absent from the alternative assignment.'.format((~mask).sum()))
    return FiberMap(real_targetid[mask], alt_targetid[index[mask]])


def write_alt_targets(tileid, ledger_dir, output_fn, footprint_fn, isodate=None):
    """
    Write the science target file the alternative assignment of ``tileid`` runs on.

    Targets are read from the alternative ledgers over the tile footprint, in the state they
    are in at this point of the replay. For a mock no inflation, plate column or proper motion
    correction is applied, following the real survey's mock path.

    Parameters
    ----------
    tileid : int
        Tile to read targets for.

    ledger_dir : str
        Directory of the alternative healpix ledgers.

    output_fn : str
        Path of the target file to write.

    footprint_fn : str
        Path of the real survey tile file giving the tile footprint.

    isodate : str, default=None
        Read the ledgers as they were at this timestamp. ``None`` reads their latest state.

    Returns
    -------
    ntargets : int
        Number of targets written.
    """
    import fitsio
    from astropy.table import Table
    from desitarget import io

    tiles = fitsio.read(footprint_fn)
    targets = io.read_targets_in_tiles(ledger_dir, tiles, quick=False, mtl=True, unique=True,
                                       isodate=isodate, tabform='ascii.ecsv')
    if not len(targets):
        raise ValueError('no target read from {} over tile {:d}'.format(ledger_dir, tileid))
    utils.mkdir(os.path.dirname(output_fn))
    Table(targets).write(output_fn, format='fits', overwrite=True)
    return len(targets)


def run_fiber_assignment(tileid, targets_fn, output_dir, header, footprint_fn, sky_fn,
                         scnd_fn=None, too_fn=None, overwrite=False, fiberassign_dir=None):
    """
    Run fiberassign for one tile, in process.

    This is what ``fba_run`` does, with the arguments the real survey used for this tile.

    Parameters
    ----------
    tileid : int
        Tile to assign.

    targets_fn : str
        Science target file, as written by :func:`write_alt_targets`.

    output_dir : str
        Directory the assignment is written to, as ``fba-<tileid>.fits``.

    header : dict
        Header of the real fiberassign file of this tile, giving the run date, field rotation,
        hour angle and fiberassign version.

    footprint_fn : str
        Real survey tile file.

    sky_fn : str
        Real survey sky target file.

    scnd_fn : str, default=None
        Real survey secondary target file, if any.

    too_fn : str, default=None
        Real survey targets-of-opportunity file, if any.

    overwrite : bool, default=False
        Whether to redo an assignment that already exists.

    fiberassign_dir : str, default=None
        Directory of the real fiberassign files, used to speed up the stuck-sky determination.

    Returns
    -------
    fn : str
        Path of the assignment that was written.
    """
    from fiberassign.scripts.assign import parse_assign, run_assign_full

    fn = get_alt_fiberassign_fn(output_dir, tileid)
    if os.path.isfile(fn) and not overwrite:
        logger.info('Assignment {} already exists, not redoing it.'.format(fn))
        return fn
    # Fiberassign leaves a temporary file behind when a run is interrupted, and then refuses
    # to start again.
    if os.path.exists(fn + '.tmp'):
        os.remove(fn + '.tmp')

    version = float(str(header['FA_VER'])[:3])
    if version < MIN_FIBERASSIGN_VERSION:
        raise NotImplementedError(
            'tile {:d} was assigned by fiberassign {}, which has to be loaded explicitly; only '
            '{} and above run under the standard environment'.format(tileid, header['FA_VER'],
                                                                     MIN_FIBERASSIGN_VERSION))

    rundate = header['RUNDATE']
    utils.mkdir(output_dir)
    optlist = ['--targets', targets_fn]
    if scnd_fn is not None: optlist.append(scnd_fn)
    if too_fn is not None: optlist.append(too_fn)
    optlist += ['--sky', sky_fn,
                '--footprint', footprint_fn,
                '--rundate', rundate,
                '--fieldrot', np.format_float_positional(header['FIELDROT']),
                '--dir', output_dir,
                '--sky_per_petal', '40',
                '--standards_per_petal', '10',
                '--sky_per_slitblock', '1',
                '--ha', str(header['FA_HA']),
                '--margin-gfa', '0.4', '--margin-petal', '0.4', '--margin-pos', '0.05',
                '--fafns_for_stucksky', utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir)]
    if overwrite: optlist.append('--overwrite')

    logger.debug('Running fiberassign for tile {:d} at rundate {}.'.format(tileid, rundate))
    run_assign_full(parse_assign(optlist=optlist))
    if not os.path.isfile(fn):
        raise ValueError('fiberassign did not write {}'.format(fn))
    return fn


def do_fiber_assignment(altmtl_dir, tileid, survey='main', obscon='dark', overwrite=False,
                        fiberassign_dir=None, fiberassign_input_dir=None):
    """
    Carry out one ``fa`` action: assign a tile and record the fiber map.

    Parameters
    ----------
    altmtl_dir : str
        Directory of the realization.

    tileid : int
        Tile to assign.

    survey : str, default='main'
        Survey to replay.

    obscon : str, default='dark'
        Observing conditions.

    overwrite : bool, default=False
        Whether to redo an assignment and a fiber map that already exist.

    fiberassign_dir : str, default=None
        Directory of the real fiberassign files.

    fiberassign_input_dir : str, default=None
        Directory of the real per-tile assignment inputs.

    Returns
    -------
    fiber_map : FiberMap
        Correspondence between the real and the alternative assignment of this tile.
    """
    import fitsio
    from .ledger import get_ledger_dir

    ts = utils.tile_string(tileid)
    header = fitsio.read_header(utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir))
    # The directory is named after the date the real survey assigned the tile, so that tiles
    # assigned the same night sit together, as they do in the real survey products.
    fadate = ''.join(str(header['RUNDATE']).split('T')[0].split('-'))
    fa_dir = get_fa_dir(altmtl_dir, fadate, survey=survey)

    fiber_map_fn = get_fiber_map_fn(fa_dir, tileid)
    if os.path.isfile(fiber_map_fn) and not overwrite:
        logger.info('Fiber map {} already exists, reusing it.'.format(fiber_map_fn))
        return FiberMap.read(fiber_map_fn)

    input_dir = utils.get_fiberassign_input_dir(tileid, survey=survey,
                                                fiberassign_input_dir=fiberassign_input_dir)
    footprint_fn = os.path.join(input_dir, '{}-tiles.fits'.format(ts))
    sky_fn = os.path.join(input_dir, '{}-sky.fits'.format(ts))
    for name, fn in [('footprint', footprint_fn), ('sky', sky_fn)]:
        if not os.path.isfile(fn):
            raise ValueError('{} file {} of tile {:d} not found'.format(name, fn, tileid))
    # Secondary targets and targets of opportunity exist only for some tiles.
    scnd_fn = os.path.join(input_dir, '{}-scnd.fits'.format(ts))
    if not os.path.isfile(scnd_fn): scnd_fn = None
    too_fn = os.path.join(input_dir, '{}-too.fits'.format(ts))
    if not os.path.isfile(too_fn): too_fn = None

    targets_fn = os.path.join(fa_dir, '{}-targ.fits'.format(ts))
    ntargets = write_alt_targets(tileid, get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon),
                                 targets_fn, footprint_fn)
    logger.debug('Tile {:d}: {:d} alternative targets in footprint.'.format(tileid, ntargets))

    run_fiber_assignment(tileid, targets_fn, fa_dir, header, footprint_fn, sky_fn,
                         scnd_fn=scnd_fn, too_fn=too_fn, overwrite=overwrite,
                         fiberassign_dir=fiberassign_dir)

    fiber_map = make_fiber_map(read_real_assignment(tileid, fiberassign_dir=fiberassign_dir),
                               read_alt_assignment(fa_dir, tileid))
    fiber_map.write(fiber_map_fn)
    logger.info('Tile {:d}: mapped {:d} fibers, {:d} of them to a different target.'.format(
        tileid, len(fiber_map), int((fiber_map.real_targetid != fiber_map.alt_targetid).sum())))
    return fiber_map

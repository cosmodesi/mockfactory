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
import contextlib
import hashlib
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.assignment')


#: Fiberassign versions from 4.0 on run under the standard desi environment; earlier tiles
#: were assigned by versions that have to be loaded explicitly, which this module cannot do.
MIN_FIBERASSIGN_VERSION = 4.0

_accepts_fafns_for_stucksky = None


def accepts_fafns_for_stucksky():
    """
    Whether the installed fiberassign takes ``--fafns_for_stucksky``.

    It reads the real assignment of a tile to find the positioners parked on sky, rather than
    working them out again, and was added after the versions that assigned the first data
    releases. Passing it to one of those is a hard argparse failure, so it is probed once.
    """
    global _accepts_fafns_for_stucksky
    if _accepts_fafns_for_stucksky is None:
        import inspect
        from fiberassign.scripts import assign
        try:
            _accepts_fafns_for_stucksky = 'fafns_for_stucksky' in inspect.getsource(assign.parse_assign)
        except (OSError, TypeError):
            _accepts_fafns_for_stucksky = False
        logger.info('fiberassign {} --fafns_for_stucksky.'.format(
            'takes' if _accepts_fafns_for_stucksky else 'does not take'))
    return _accepts_fafns_for_stucksky


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


def read_real_assignment(tileid, fiberassign_dir=None, header=False):
    """
    Return the real survey assignment of ``tileid``, with its FIBER and TARGETID columns.

    With ``header``, the header comes back too. The file is gzipped, so reading it for the
    header and again for the table decompresses it twice, once per tile.
    """
    import fitsio

    fn = utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir)
    if not header:
        return _read_assignment(fn, ['FIBERASSIGN', 'FASSIGN'])
    with fitsio.FITS(fn) as fits:
        available = [hdu.get_extname() for hdu in fits]
        data = None
        for extname in ['FIBERASSIGN', 'FASSIGN']:
            if extname in available:
                data = fits[extname].read()
                break
        if data is None:
            data = fits[1].read()
        # The run date, field rotation and hour angle are in the primary header, not the
        # table's own.
        return data, fits[0].read_header()


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


def read_alt_targets(tileid, ledger_dir, footprint_fn, isodate=None, state=None):
    """
    Return the science targets the alternative assignment of ``tileid`` runs on.

    Targets are read from the alternative ledgers over the tile footprint, in the state they
    are in at this point of the replay. For a mock no inflation, plate column or proper motion
    correction is applied, following the real survey's mock path.

    Parameters
    ----------
    tileid : int
        Tile to read targets for.

    ledger_dir : str
        Directory of the alternative healpix ledgers.

    footprint_fn : str
        Path of the real survey tile file giving the tile footprint.

    isodate : str, default=None
        Read the ledgers as they were at this timestamp. ``None`` reads their latest state.

    state : LedgerState, default=None
        State to read the targets from, instead of the ledgers in ``ledger_dir``.

    Returns
    -------
    targets : array
        The targets over the tile.
    """
    import fitsio

    tiles = fitsio.read(footprint_fn)
    if state is not None:
        targets = state.targets_in_tiles(tiles)
    else:
        from desitarget import io
        from .compat import supported
        targets = io.read_targets_in_tiles(ledger_dir, tiles, quick=False, mtl=True, unique=True,
                                           isodate=isodate,
                                           **supported(io.read_targets_in_tiles, tabform='ascii.ecsv'))
    if not len(targets):
        raise ValueError('no target over tile {:d}, read from {}; does the mock cover this '
                         'tile?'.format(tileid, 'the state in memory' if state is not None else ledger_dir))
    return targets


@contextlib.contextmanager
def targets_in_memory(targets, targets_fn, survey='main'):
    """
    Make fiberassign take the science targets of one tile from an array rather than a file.

    Handing fiberassign a file costs more than the assignment it then does: of the 4.7 s one
    tile takes, 0.9 s is building and writing the target file and about 1.8 s more is inside
    fiberassign reading it back and turning it into its own objects, against some 1.5 s of
    actual geometry and assignment.

    Only the reading is replaced. :func:`fiberassign.targets.load_target_file` opens the file,
    takes the survey from its header and hands the rows to
    :func:`fiberassign.targets.load_target_table`; this substitutes the array at that last
    step and leaves every other call, the sky and secondary files included, to go through the
    real function. Everything after loading, which is to say the assignment itself, is
    fiberassign's own code and is not touched.

    Parameters
    ----------
    targets : array
        Science targets of the tile, from :func:`read_alt_targets`.

    targets_fn : str
        The path fiberassign was told to read them from. It need not exist: it is only the
        name the substitution is keyed on, so that the sky and secondary files still load
        normally.

    survey : str, default='main'
        Survey the targets belong to, which a file would carry as ``FA_SURV`` in its header
        and which decides how the sky targets are loaded after them.
    """
    from fiberassign.scripts import assign
    from fiberassign.targets import load_target_table

    real = assign.load_target_file

    def load_target_file(tgs, tagalong, tfile, **kwargs):
        if tfile != targets_fn:
            return real(tgs, tagalong, tfile, **kwargs)
        kwargs.pop('rowbuffer', None)
        if kwargs.get('survey', None) is None:
            kwargs['survey'] = survey
        load_target_table(tgs, tagalong, targets, **kwargs)
        return kwargs['survey']

    assign.load_target_file = load_target_file
    try:
        yield
    finally:
        assign.load_target_file = real


def write_alt_targets(tileid, ledger_dir, output_fn, footprint_fn, isodate=None, state=None):
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

    state : LedgerState, default=None
        State to read the targets from, instead of the ledgers in ``ledger_dir``.

    Returns
    -------
    ntargets : int
        Number of targets written.
    """
    from astropy.table import Table

    targets = read_alt_targets(tileid, ledger_dir, footprint_fn, isodate=isodate, state=state)
    utils.mkdir(os.path.dirname(output_fn))
    Table(targets).write(output_fn, format='fits', overwrite=True)
    return len(targets)


def run_fiber_assignment(tileid, targets_fn, output_dir, header, footprint_fn, sky_fn,
                         scnd_fn=None, too_fn=None, overwrite=False, fiberassign_dir=None,
                         targets=None, survey='main'):
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

    targets : array, default=None
        Science targets, to hand fiberassign directly instead of having it read ``targets_fn``,
        which then does not have to exist. See :func:`targets_in_memory`.

    survey : str, default='main'
        Survey the targets belong to, used only with ``targets``.

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
                '--margin-gfa', '0.4', '--margin-petal', '0.4', '--margin-pos', '0.05']
    if accepts_fafns_for_stucksky():
        # Reuse the real survey's own stuck-sky determination rather than redoing it.
        optlist += ['--fafns_for_stucksky',
                    utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir)]
    if overwrite: optlist.append('--overwrite')

    logger.debug('Running fiberassign for tile {:d} at rundate {}.'.format(tileid, rundate))
    args = parse_assign(optlist=optlist)
    if targets is None:
        run_assign_full(args)
    else:
        with targets_in_memory(targets, targets_fn, survey=survey):
            run_assign_full(args)
    if not os.path.isfile(fn):
        raise ValueError('fiberassign did not write {}'.format(fn))
    return fn


def do_fiber_assignment(altmtl_dir, tileid, survey='main', obscon='dark', overwrite=False,
                        fiberassign_dir=None, fiberassign_input_dir=None, state=None,
                        tmp_dir=None, load_targets='file'):
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

    state : LedgerState, default=None
        State to read the targets from, instead of the healpix ledgers.

    tmp_dir : str, default=None
        Where to put the target file handed to fiberassign, which is read once and thrown
        away. Defaults to memory, through :func:`mockfactory.desi.altmtl.utils.get_tmp_dir`.
        Unused when ``load_targets`` is ``'memory'``.

    load_targets : str, default='file'
        How the science targets reach fiberassign. ``'file'`` writes them out and lets
        fiberassign read them, which is what ``fba_run`` does. ``'memory'`` hands it the array,
        which is faster and leaves the assignment itself untouched; see
        :func:`targets_in_memory`.

    Returns
    -------
    fiber_map : FiberMap
        Correspondence between the real and the alternative assignment of this tile.
    """
    from .ledger import get_ledger_dir

    ts = utils.tile_string(tileid)
    real_assignment, header = read_real_assignment(tileid, fiberassign_dir=fiberassign_dir,
                                                   header=True)
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

    # The target file is handed to fiberassign and never read again, so it goes to memory
    # rather than to the file system the assignments are written to.
    targets_dir = utils.get_tmp_dir(tmp_dir)
    if targets_dir is None:
        targets_dir = fa_dir
    else:
        # A directory of its own per realization. The file is named after the tile, and the
        # memory it goes to is shared by everything on the node, so two mocks replayed at once
        # reach the same tile and one overwrites or half-reads the other's targets.
        token = hashlib.md5(os.path.abspath(altmtl_dir).encode()).hexdigest()[:10]
        targets_dir = os.path.join(targets_dir, 'altmtl-{}'.format(token))
        utils.mkdir(targets_dir)
    targets_fn = os.path.join(targets_dir, '{}-targ.fits'.format(ts))
    ledger_dir = None if state is not None else get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon)
    in_memory = load_targets == 'memory'
    try:
        if in_memory:
            array = read_alt_targets(tileid, ledger_dir, footprint_fn, state=state)
            ntargets = len(array)
        else:
            array = None
            ntargets = write_alt_targets(tileid, ledger_dir, targets_fn, footprint_fn, state=state)
        logger.debug('Tile {:d}: {:d} alternative targets in footprint.'.format(tileid, ntargets))

        run_fiber_assignment(tileid, targets_fn, fa_dir, header, footprint_fn, sky_fn,
                             scnd_fn=scnd_fn, too_fn=too_fn, overwrite=overwrite,
                             fiberassign_dir=fiberassign_dir, targets=array, survey=survey)
    finally:
        if not in_memory and targets_dir is not fa_dir and os.path.isfile(targets_fn):
            os.remove(targets_fn)

    fiber_map = make_fiber_map(real_assignment, read_alt_assignment(fa_dir, tileid))
    fiber_map.write(fiber_map_fn)
    logger.info('Tile {:d}: mapped {:d} fibers, {:d} of them to a different target.'.format(
        tileid, len(fiber_map), int((fiber_map.real_targetid != fiber_map.alt_targetid).sum())))
    return fiber_map

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
from pathlib import Path
import contextlib
import hashlib
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.assignment')


#: Fiberassign versions from 4.0 on run under the standard desi environment; earlier tiles
#: were assigned by versions that have to be loaded explicitly, see :func:`run_legacy_fiber_assignment`.
MIN_FIBERASSIGN_VERSION = 4.0

#: The desiconda tree SV tiles are reassigned under. It installs every fiberassign release SV3
#: was assigned with, 2.2.0 to 4.0.0; the current fiberassign cannot load SV3 targets at all,
#: its SV3 masks having no gaia standard bit. Its modulefile asks for a cray-mpich Perlmutter
#: no longer has, so the environment is set up by hand, see :func:`get_legacy_environ`.
LEGACY_DESICONDA = os.environ.get('ALTMTL_LEGACY_DESICONDA',
                                  '/global/common/software/desi/perlmutter/desiconda/20230111-2.1.0')

#: Releases of the packages fiberassign imports, within :attr:`LEGACY_DESICONDA`.
LEGACY_DEPENDENCIES = {'desitarget': '2.6.0', 'desimodel': '0.18.0', 'desiutil': '3.2.6', 'desimeter': '0.7.0'}

#: Sky bricks the SV3 assignments looked stuck positioners up in, by fiberassign release. From
#: 2.4 on, a stuck positioner landing on blank sky becomes a sky fiber; without them a tile
#: loses some 550 sky and 400 science fibers. The 2.4 tiles used v2, and differ by 8% with v3;
#: later ones v3, and with v2 a stuck positioner here and there misses its sky, which costs a
#: slitblock its quota and bumps a filler target to sky (tiles 400, 579, 593). desihub/LSS
#: makes the same choice (LSS.SV3.fatools.get_fba_fromnewmtl).
LEGACY_SKYBRICKS = {'2.4': 'v2'}
LEGACY_SKYBRICKS_DEFAULT = 'v3'


def get_legacy_skybricks_dir(release):
    """
    Return the sky bricks directory fiberassign ``release`` looked stuck positioners up in.

    The release is matched on its major and minor numbers against :data:`LEGACY_SKYBRICKS`, and falls back on
    :data:`LEGACY_SKYBRICKS_DEFAULT`; the directory is ``target/skybricks/<version>`` under ``DESI_ROOT``.

    Parameters
    ----------
    release : str
        Installed fiberassign release, e.g. '2.4.0', as :func:`get_legacy_release` returns it.

    Returns
    -------
    skybricks_dir : Path
        Sky bricks directory, to be set as ``SKYBRICKS_DIR``.
    """
    version = LEGACY_SKYBRICKS.get('.'.join(str(release).split('.')[:2]), LEGACY_SKYBRICKS_DEFAULT)
    return Path(utils.DESI_ROOT) / 'target' / 'skybricks' / version

#: Assignment options carried over from a tile's ``FAARGS`` header; the others there built
#: its target file, which the replay provides itself.
LEGACY_FAARGS = ('--sky_per_petal', '--standards_per_petal', '--sky_per_slitblock', '--ha',
                 '--margin_gfa', '--margin_petal', '--margin_pos')

_accepts_fafns_for_stucksky = None


def is_legacy(header):
    """
    Whether the tile of this fiberassign header has to be reassigned in a legacy environment, that is whether its
    ``FA_SURV`` (``'main'`` if absent) is anything other than the main survey.

    Parameters
    ----------
    header : dict, fitsio.FITSHDR
        Primary header of the real fiberassign file of the tile.

    Returns
    -------
    legacy : bool
        ``True`` for an SV tile, to be run by :func:`run_legacy_fiber_assignment`.
    """
    return str(header.get('FA_SURV', 'main')).strip().lower() != 'main'


#: '2.2.0.dev2811' labels every SV3 tile assigned up to 2021-04-22, but the code behind the
#: label changed: tiles run up to 2021-04-13T23:27 are reproduced by 2.2.0 and not by 2.3.0,
#: tiles run from 2021-04-14T22:12 by 2.3.0 and not by 2.2.0, the wrong one missing up to 45%
#: of a tile's fibers. The switch is put in between.
LEGACY_DEV2811_SWITCH = '2021-04-14T12:00:00'

#: Run dates the real assignment did not use as recorded. The tiles of 2021-04-10T21:28:37
#: (4, 30, 58, 364, ...) differ by some 15 fibers each under any release and desimodel data,
#: and are reproduced exactly with the focal plane of 20:00, as desihub/LSS
#: (LSS.SV3.fatools.get_fba_fromnewmtl) does.
LEGACY_RUNDATES = {'2021-04-10T21:28:37': '2021-04-10T20:00:00'}


def get_legacy_release(version, rundate=None):
    """
    Return the installed fiberassign release standing in for ``version``, at run date ``rundate``.

    Some SV3 tiles were assigned by development builds, '2.2.0.dev2811' and '2.3.0.dev2838',
    which are not installed. '2.3.0.dev2838' runs under 2.3.0; '2.2.0.dev2811' under 2.2.0 or
    2.3.0 depending on the run date, see :attr:`LEGACY_DEV2811_SWITCH`. Any other version runs
    under its own release, with a ``.dev`` suffix stripped.

    Parameters
    ----------
    version : str
        Fiberassign version the tile was assigned by, the ``FA_VER`` header card.

    rundate : str, default=None
        Run date of the tile, the ``RUNDATE`` header card; required for '2.2.0.dev2811'.

    Returns
    -------
    release : str
        Release installed under :attr:`LEGACY_DESICONDA`.

    Raises
    ------
    ValueError
        If ``rundate`` is needed and not given, or if the release is not installed.
    """
    version = str(version)
    if version == '2.2.0.dev2811':
        if rundate is None:
            raise ValueError('fiberassign 2.2.0.dev2811 stands for two releases; pass the run date')
        release = '2.2.0' if str(rundate)[:19] < LEGACY_DEV2811_SWITCH else '2.3.0'
    else:
        release = version.split('.dev')[0]
    if not (Path(LEGACY_DESICONDA) / 'code' / 'fiberassign' / release).is_dir():
        raise ValueError('fiberassign {} (for {}) is not installed under {}'.format(
            release, version, LEGACY_DESICONDA))
    return release


def get_legacy_environ(release):
    """
    Return the environment running fiberassign ``release`` from :attr:`LEGACY_DESICONDA`,
    an installed release as :func:`get_legacy_release` returns it.

    This is what loading desimodules 23.1 and swapping in that fiberassign would set, minus MPI.
    The user site is switched off, so that nothing installed for the current python leaks in.
    The current environment is copied without its ``PYTHON*``, ``CONDA*`` and ``DESI*`` variables;
    ``PATH``, ``LD_LIBRARY_PATH``, ``PYTHONPATH`` (fiberassign and :data:`LEGACY_DEPENDENCIES`),
    ``DESIMODEL``, ``DESI_ROOT`` and ``SKYBRICKS_DIR`` are set to the legacy tree.

    Parameters
    ----------
    release : str
        Installed fiberassign release.

    Returns
    -------
    environ : dict
        Environment variables, to hand to :func:`subprocess.call`.
    """
    root = Path(LEGACY_DESICONDA)
    code, site = root / 'code', Path('lib') / 'python3.10' / 'site-packages'
    packages = dict(LEGACY_DEPENDENCIES, fiberassign=release)
    environ = {name: value for name, value in os.environ.items()
               if not name.startswith(('PYTHON', 'CONDA', 'DESI'))}
    environ.update(
        PATH=os.pathsep.join([str(code / 'fiberassign' / packages['fiberassign'] / 'bin'),
                              str(root / 'conda' / 'bin'), str(root / 'aux' / 'bin'),
                              os.environ.get('PATH', '')]),
        LD_LIBRARY_PATH=os.pathsep.join([str(root / 'aux' / 'lib'), os.environ.get('LD_LIBRARY_PATH', '')]),
        PYTHONPATH=os.pathsep.join([str(code / name / release / site) for name, release in packages.items()]
                                   + [str(root / 'conda' / site)]),
        PYTHONNOUSERSITE='1',
        DESIMODEL=str(code / 'desimodel' / packages['desimodel']),
        DESI_ROOT=str(utils.DESI_ROOT),
        SKYBRICKS_DIR=str(get_legacy_skybricks_dir(release)))
    return environ


def get_legacy_options(header):
    """
    Return the assignment options the ``FAARGS`` header of a tile records, as fba_run takes them.

    Only the options of :data:`LEGACY_FAARGS` are kept, each with the value that follows it; the margins are
    respelt with dashes (``--margin_gfa`` becomes ``--margin-gfa``).

    Parameters
    ----------
    header : dict, fitsio.FITSHDR
        Primary header of the real fiberassign file of the tile, with ``FAARGS``.

    Returns
    -------
    options : list
        Option names and values, as strings.
    """
    import shlex

    words = shlex.split(str(header['FAARGS']))
    options = []
    for iword, word in enumerate(words[:-1]):
        if word in LEGACY_FAARGS:
            # FAARGS spells the margins as fba_launch took them, fba_run with dashes.
            options += [word.replace('_', '-') if word.startswith('--margin') else word, words[iword + 1]]
    return options


def accepts_fafns_for_stucksky():
    """
    Whether the installed fiberassign takes ``--fafns_for_stucksky``.

    It reads the real assignment of a tile to find the positioners parked on sky, rather than
    working them out again, and was added after the versions that assigned the first data
    releases. Passing it to one of those is a hard argparse failure, so it is probed once,
    by looking for it in the source of ``fiberassign.scripts.assign.parse_assign``, and the
    answer is kept for the process.

    Returns
    -------
    accepts : bool
        Whether ``--fafns_for_stucksky`` can be passed.
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
        """
        Initialize the map from the two target lists, paired entry by entry.

        Parameters
        ----------
        real_targetid : array
            Target observed by the real survey, one entry per fiber.

        alt_targetid : array
            Target the alternative survey put on that same fiber, same length.
        """
        self.real_targetid = np.asarray(real_targetid)
        self.alt_targetid = np.asarray(alt_targetid)

    def __len__(self):
        """Return the number of fibers in the map."""
        return len(self.real_targetid)

    def real_to_alt(self, targetid):
        """
        Return the alternative counterpart of each real target in ``targetid``: the target the
        alternative survey put on the fiber the real survey put it on.

        Parameters
        ----------
        targetid : array
            Real survey target identifiers.

        Returns
        -------
        alt_targetid : array
            Alternative target identifiers, same shape as ``targetid``.

        Raises
        ------
        ValueError
            If a target was observed by the real survey but sits on no fiber of this tile in
            the alternative assignment. Dropping it would quietly lose an observation.
        """
        return self._map(targetid, self.real_targetid, self.alt_targetid, 'real')

    def alt_to_real(self, targetid):
        """
        Return the real counterpart of each alternative target in ``targetid``: the target the
        real survey put on the fiber the alternative survey put it on.

        Parameters
        ----------
        targetid : array
            Alternative target identifiers.

        Returns
        -------
        real_targetid : array
            Real survey target identifiers, same shape as ``targetid``.

        Raises
        ------
        ValueError
            If a target sits on no fiber of this tile in the alternative assignment.
        """
        return self._map(targetid, self.alt_targetid, self.real_targetid, 'alternative')

    @staticmethod
    def _map(targetid, source, dest, name):
        """Look each of ``targetid`` up in ``source`` by binary search and return the matching entries of ``dest``;
        raise a ValueError naming the ``name`` side if any is missing."""
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
        """
        Save to ``fn``, as a structured array with ``REAL_TARGETID`` and ``ALT_TARGETID`` columns (numpy ``.npy``),
        creating the parent directory if needed.

        Parameters
        ----------
        fn : str, Path
            Output file name.

        Returns
        -------
        fn : str, Path
            The file name, as given.
        """
        utils.mkdir(Path(fn).parent)
        array = np.empty(len(self), dtype=[('REAL_TARGETID', 'i8'), ('ALT_TARGETID', 'i8')])
        array['REAL_TARGETID'], array['ALT_TARGETID'] = self.real_targetid, self.alt_targetid
        np.save(fn, array)
        return fn

    @classmethod
    def read(cls, fn):
        """
        Load from ``fn``, as written by :meth:`write`.

        Parameters
        ----------
        fn : str, Path
            File name.

        Returns
        -------
        fiber_map : FiberMap
        """
        array = np.load(fn)
        return cls(array['REAL_TARGETID'], array['ALT_TARGETID'])


def get_fiber_map_fn(fa_dir, tileid):
    """
    Return the path of the fiber map of ``tileid``, ``famap-<tileid>.npy`` in ``fa_dir``.

    Parameters
    ----------
    fa_dir : str, Path
        Directory of the alternative assignment, see :func:`get_fa_dir`.

    tileid : int
        Tile identifier.

    Returns
    -------
    fn : Path
        Path of the fiber map.
    """
    return Path(fa_dir) / 'famap-{}.npy'.format(utils.tile_string(tileid))


def get_alt_fiberassign_fn(fa_dir, tileid):
    """
    Return the path of the alternative fiberassign file of ``tileid``, ``fba-<tileid>.fits`` in ``fa_dir``.

    Parameters
    ----------
    fa_dir : str, Path
        Directory of the alternative assignment, see :func:`get_fa_dir`.

    tileid : int
        Tile identifier.

    Returns
    -------
    fn : Path
        Path of the alternative fiberassign file.
    """
    return Path(fa_dir) / 'fba-{}.fits'.format(utils.tile_string(tileid))


def get_fa_dir(altmtl_dir, fadate, survey='main'):
    """
    Return the directory holding the alternative assignment of a given assignment date,
    ``<altmtl_dir>/fa/<SURVEY>/<fadate>``.

    Parameters
    ----------
    altmtl_dir : str, Path
        Directory of the realization.

    fadate : str
        Date the real survey assigned the tile, as ``YYYYMMDD``.

    survey : str, default='main'
        Survey, upper-cased in the path.

    Returns
    -------
    fa_dir : Path
        Directory of the alternative assignment.
    """
    return Path(altmtl_dir) / 'fa' / survey.upper() / fadate


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

    The table is the ``FIBERASSIGN`` extension, else ``FASSIGN``, else the first binary table.
    With ``header``, the header comes back too. The file is gzipped, so reading it for the
    header and again for the table decompresses it twice, once per tile.

    Parameters
    ----------
    tileid : int
        Tile identifier.

    fiberassign_dir : str, default=None
        Directory of the real fiberassign files; see :func:`mockfactory.desi.altmtl.utils.get_fiberassign_fn`.

    header : bool, default=False
        Whether to also return the primary header, which holds the run date, field rotation and hour angle.

    Returns
    -------
    assignment : array
        Real assignment of the tile.

    header : fitsio.FITSHDR
        Primary header of the fiberassign file, only if ``header`` is ``True``.
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
    """
    Return the alternative assignment of ``tileid``, the ``FASSIGN`` extension of its
    alternative fiberassign file (else ``FIBERASSIGN``, else the first binary table).

    Parameters
    ----------
    fa_dir : str, Path
        Directory of the alternative assignment, see :func:`get_fa_dir`.

    tileid : int
        Tile identifier.

    Returns
    -------
    assignment : array
        Alternative assignment of the tile, with its FIBER and TARGETID columns.
    """
    return _read_assignment(get_alt_fiberassign_fn(fa_dir, tileid), ['FASSIGN', 'FIBERASSIGN'])


def make_fiber_map(real_assignment, alt_assignment):
    """
    Build the :class:`FiberMap` pairing the two assignments of a tile fiber by fiber.

    Fibers holding no real target are dropped: they carry nothing to fold into the ledgers.
    Real fibers absent from the alternative assignment are dropped too, with a warning.

    Parameters
    ----------
    real_assignment : array
        Real assignment of the tile, with FIBER and TARGETID columns, see :func:`read_real_assignment`.

    alt_assignment : array
        Alternative assignment of the tile, with FIBER and TARGETID columns, see :func:`read_alt_assignment`.

    Returns
    -------
    fiber_map : FiberMap
        Real and alternative target of each fiber holding a real target.

    Raises
    ------
    ValueError
        If either assignment has a fiber more than once.
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


def set_tile_subpriority(targets, tileid, tile_subpriority, footprint_fn):
    """
    Give the targets of one tile the subpriorities this tile breaks its ties with.

    SV3 drew them afresh for every tile: a target shared by two overlapping tiles has
    uncorrelated subpriorities on them (correlation -0.008 over 27535 targets of tiles 1 and 2),
    and neither is the one its ledger holds. Keeping one per target instead would have the same
    targets lose every tie on every pass of a rosette.

    Parameters
    ----------
    targets : array
        Targets of the tile; modified in place.

    tileid : int
        Tile they are assigned on.

    tile_subpriority : int, str
        An integer draws them uniformly, from a generator seeded with it and the tile. 'real'
        takes those of the real tile target file, next to ``footprint_fn``, which only makes
        sense when the targets are the real ones, to check the replay against the data.

    footprint_fn : str
        Real survey tile file.

    Returns
    -------
    targets : array
        The same targets, their ``SUBPRIORITY`` column replaced.

    Raises
    ------
    ValueError
        If ``tile_subpriority`` is a string other than 'real', or if, with 'real', a target is
        missing from the real tile target file.
    """
    import fitsio

    if isinstance(tile_subpriority, str):
        if tile_subpriority != 'real':
            raise ValueError("tile_subpriority must be an integer seed or 'real', got {}".format(tile_subpriority))
        real = fitsio.read(Path(footprint_fn).parent / '{}-targ.fits'.format(utils.tile_string(tileid)),
                           columns=['TARGETID', 'SUBPRIORITY'])
        real = real[np.argsort(real['TARGETID'])]
        index = np.clip(np.searchsorted(real['TARGETID'], targets['TARGETID']), 0, len(real) - 1)
        found = real['TARGETID'][index] == targets['TARGETID']
        if not found.all():
            raise ValueError('{:d} target(s) of tile {:d} are not in its real target file'.format(
                int((~found).sum()), tileid))
        targets['SUBPRIORITY'] = real['SUBPRIORITY'][index]
        return targets
    # Seeded by the tile as well, so that the draw does not depend on the order or the process
    # the tiles are assigned in; ordered by target, so not on how the state is laid out either.
    rng = np.random.default_rng([int(tile_subpriority), int(tileid)])
    order = np.argsort(targets['TARGETID'])
    subpriority = np.empty(len(targets), dtype='f8')
    subpriority[order] = rng.uniform(size=len(targets))
    targets['SUBPRIORITY'] = subpriority
    return targets


def read_alt_targets(tileid, ledger_dir, footprint_fn, isodate=None, state=None, tile_subpriority=None):
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

    tile_subpriority : int, str, default=None
        Subpriorities to break this tile's ties with, see :func:`set_tile_subpriority`.
        ``None`` keeps those of the state, as the main survey does.

    Returns
    -------
    targets : array
        The targets over the tile; a copy when ``tile_subpriority`` is given, so that the state is never written through.

    Raises
    ------
    ValueError
        If no target falls over the tile.
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
    if tile_subpriority is not None:
        # Copied, so that the state is never written through.
        targets = set_tile_subpriority(np.array(targets), tileid, tile_subpriority, footprint_fn)
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

    Returns
    -------
    context : context manager
        Within its ``with`` block ``fiberassign.scripts.assign.load_target_file`` is replaced;
        the real function is put back on exit, even on an exception.
    """
    from fiberassign.scripts import assign
    from fiberassign.targets import load_target_table

    real = assign.load_target_file

    def load_target_file(tgs, tagalong, tfile, **kwargs):
        """Load ``targets`` through :func:`fiberassign.targets.load_target_table` when ``tfile`` is ``targets_fn``,
        else defer to the real loader; return the survey, as the real loader does."""
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


def write_alt_targets(tileid, ledger_dir, output_fn, footprint_fn, isodate=None, state=None,
                      tile_subpriority=None):
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

    tile_subpriority : int, str, default=None
        Subpriorities to break this tile's ties with, see :func:`set_tile_subpriority`.
        ``None`` keeps those of the state.

    Returns
    -------
    ntargets : int
        Number of targets written.
    """
    from astropy.table import Table

    targets = read_alt_targets(tileid, ledger_dir, footprint_fn, isodate=isodate, state=state,
                               tile_subpriority=tile_subpriority)
    utils.mkdir(Path(output_fn).parent)
    Table(targets).write(output_fn, format='fits', overwrite=True)
    return len(targets)


def run_legacy_fiber_assignment(tileid, targets_fn, output_dir, header, footprint_fn, sky_fn,
                                scnd_fn=None, too_fn=None):
    """
    Run fiberassign for one SV tile, under the release that assigned it, in a separate process.

    The run date, field rotation and the assignment options of the tile's ``FAARGS`` are passed
    as the real run took them, and the environment is :func:`get_legacy_environ`. Reassigning a
    real SV3 tile from its own target file this way reproduces the official assignment on
    every fiber, over one tile per release and program, 2.2.0.dev2811 to 4.0.0 (15 tiles).

    The fiberassign log goes to ``fba-<tileid>.log`` in ``output_dir``, and is removed when the run succeeds.

    Parameters
    ----------
    tileid : int
        Tile to assign.

    targets_fn : str
        Science target file, as written by :func:`write_alt_targets`; it has to exist.

    output_dir : str
        Directory the assignment is written to, as ``fba-<tileid>.fits``.

    header : dict
        Header of the real fiberassign file of this tile, giving ``FA_VER``, ``RUNDATE``, ``FIELDROT`` and ``FAARGS``.

    footprint_fn : str
        Real survey tile file.

    sky_fn : str
        Real survey sky target file.

    scnd_fn : str, default=None
        Real survey secondary target file, if any.

    too_fn : str, default=None
        Real survey targets-of-opportunity file, if any.

    Returns
    -------
    fn : str
        Path of the assignment that was written.

    Raises
    ------
    ValueError
        If fiberassign exits with a nonzero status or writes no assignment; the log is then kept.
    """
    import subprocess

    release = get_legacy_release(header['FA_VER'], rundate=header['RUNDATE'])
    environ = get_legacy_environ(release)
    fba_run = Path(LEGACY_DESICONDA) / 'code' / 'fiberassign' / release / 'bin' / 'fba_run'
    targets = [targets_fn] + [fn for fn in [scnd_fn, too_fn] if fn is not None]
    # The run date as the header has it: early SV3 dates carry no timezone, which the current
    # fiberassign rejects and the releases that wrote them expect.
    rundate = LEGACY_RUNDATES.get(str(header['RUNDATE']), str(header['RUNDATE']))
    optlist = ['--targets'] + targets + ['--sky', sky_fn, '--footprint', footprint_fn,
               '--rundate', rundate,
               '--fieldrot', np.format_float_positional(header['FIELDROT']),
               '--dir', output_dir, '--overwrite'] + get_legacy_options(header)
    utils.mkdir(output_dir)
    fn = get_alt_fiberassign_fn(output_dir, tileid)
    log_fn = Path(output_dir) / 'fba-{}.log'.format(utils.tile_string(tileid))
    command = [str(Path(LEGACY_DESICONDA) / 'conda' / 'bin' / 'python'), str(fba_run)] + [str(option) for option in optlist]
    logger.debug('Running fiberassign {} for tile {:d} at rundate {}.'.format(release, tileid, header['RUNDATE']))
    with open(log_fn, 'w') as log:
        status = subprocess.call(command, env=environ, stdout=log, stderr=subprocess.STDOUT)
    if status or not Path(fn).is_file():
        raise ValueError('fiberassign {} failed on tile {:d} (exit status {:d}); see {}'.format(
            release, tileid, status, log_fn))
    # Kept only when something went wrong: a replay writes one per tile per realization.
    log_fn.unlink()
    return fn


#: Focal plane states built by :func:`cached_load_hardware`, the most recent last, at most HARDWARE_CACHE_SIZE of them.
_hardware_cache = {}
HARDWARE_CACHE_SIZE = 8


def cached_load_hardware(focalplane=None, rundate=None, add_margins={}, get_time_range=False):
    """
    :func:`fiberassign.hardware.load_hardware`, the result kept for the next call with the same run date and margins. fiberassign
    builds the focal plane state anew for every tile it assigns: 0.3 s a tile, a tenth of an assignment, desimodel replaying the
    positioner state table row by row; the state of a run date is the same for every tile (14 MB each).
    At most :data:`HARDWARE_CACHE_SIZE` states are kept, the least recently used dropped first. A call with ``focalplane``
    or ``get_time_range`` is passed straight through, uncached.

    Parameters
    ----------
    focalplane : tuple, default=None
        Focal plane to build the state from, as :func:`fiberassign.hardware.load_hardware` takes it; bypasses the cache.

    rundate : str, default=None
        Run date the focal plane state is taken at.

    add_margins : dict, default={}
        Exclusion margins added to the positioners, name: value.

    get_time_range : bool, default=False
        Whether to also return the time range the state is valid over; bypasses the cache.

    Returns
    -------
    hardware : fiberassign.hardware.Hardware
        Focal plane state, shared between calls with the same key (and the time range with ``get_time_range``).
    """
    from fiberassign.hardware import load_hardware
    if focalplane is not None or get_time_range:
        return load_hardware(focalplane=focalplane, rundate=rundate, add_margins=add_margins, get_time_range=get_time_range)
    key = (str(rundate), tuple(sorted((str(name), float(value)) for name, value in dict(add_margins or {}).items())))
    if key in _hardware_cache:
        _hardware_cache[key] = _hardware_cache.pop(key)  # most recent last
    else:
        _hardware_cache[key] = load_hardware(rundate=rundate, add_margins=add_margins)
        while len(_hardware_cache) > HARDWARE_CACHE_SIZE: _hardware_cache.pop(next(iter(_hardware_cache)))
    return _hardware_cache[key]


def use_cached_hardware():
    """Have fiberassign's run_assign_full take its focal plane state from :func:`cached_load_hardware`."""
    import fiberassign.scripts.assign as assign
    if assign.load_hardware is not cached_load_hardware: assign.load_hardware = cached_load_hardware


def run_fiber_assignment(tileid, targets_fn, output_dir, header, footprint_fn, sky_fn,
                         scnd_fn=None, too_fn=None, overwrite=False, fiberassign_dir=None,
                         targets=None, survey='main'):
    """
    Run fiberassign for one tile, in process.

    This is what ``fba_run`` does, with the arguments the real survey used for this tile.
    An existing assignment is kept unless ``overwrite``; a leftover ``.tmp`` file from an interrupted
    run is removed first. SV tiles are handed to :func:`run_legacy_fiber_assignment`. The focal plane
    state comes from :func:`cached_load_hardware`.

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

    Raises
    ------
    ValueError
        If ``targets`` is given for a tile that is reassigned in a legacy environment, or if fiberassign writes no assignment.

    NotImplementedError
        If a main survey tile was assigned by a fiberassign older than :data:`MIN_FIBERASSIGN_VERSION`.
    """
    from fiberassign.scripts.assign import parse_assign, run_assign_full

    fn = get_alt_fiberassign_fn(output_dir, tileid)
    if Path(fn).is_file() and not overwrite:
        logger.info('Assignment {} already exists, not redoing it.'.format(fn))
        return fn
    # Fiberassign leaves a temporary file behind when a run is interrupted, and then refuses
    # to start again.
    tmp_fn = fn.parent / (fn.name + '.tmp')
    if tmp_fn.exists():
        tmp_fn.unlink()

    if is_legacy(header):
        if targets is not None:
            raise ValueError('tile {:d} is reassigned in a separate process, which reads its '
                             'targets from {}; pass load_targets=\'file\''.format(tileid, targets_fn))
        return run_legacy_fiber_assignment(tileid, targets_fn, output_dir, header, footprint_fn,
                                           sky_fn, scnd_fn=scnd_fn, too_fn=too_fn)

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
    # Nothing is passed for --fba_use_fabs, deliberately. The inner keepout radius of
    # Hardware::position_xy_bad is |theta_arm - phi_arm|, and fiberassign wrote ::abs of it;
    # which overload that picked depended on the compiler, gcc 11.2 taking the integer one and
    # gcc 13 the double. fiberassign 5.8.0 made the choice explicit and selects it from the
    # rundate, which is what the real survey got, so leaving it alone is what a mock wants.
    # It does mean the replay no longer matches the DA2 mock references bit for bit: those were
    # assigned under desiconda 20240425-2.2.0, a gcc 13 build, so they carry the double
    # behaviour their 2021 rundates did not. That gap is theirs, not ours.
    if overwrite: optlist.append('--overwrite')

    logger.debug('Running fiberassign for tile {:d} at rundate {}.'.format(tileid, rundate))
    # argparse indexes every entry as a string, so a Path has to be spelt out.
    optlist = [str(option) for option in optlist]
    args = parse_assign(optlist=optlist)
    use_cached_hardware()
    if targets is None:
        run_assign_full(args)
    else:
        with targets_in_memory(targets, targets_fn, survey=survey):
            run_assign_full(args)
    if not Path(fn).is_file():
        raise ValueError('fiberassign did not write {}'.format(fn))
    return fn


def do_fiber_assignment(altmtl_dir, tileid, survey='main', obscon='dark', overwrite=False,
                        fiberassign_dir=None, fiberassign_input_dir=None, state=None,
                        tmp_dir=None, load_targets='file', tile_subpriority=None):
    """
    Carry out one ``fa`` action: assign a tile and record the fiber map.

    The assignment goes to :func:`get_fa_dir` of the date the real survey assigned the tile, with its fiber
    map next to it; an existing fiber map is read back rather than redone, unless ``overwrite``. The
    target file handed to fiberassign is removed once the assignment is done.

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
        :func:`targets_in_memory`. SV tiles always go through a file.

    tile_subpriority : int, str, default=None
        Subpriorities to break this tile's ties with, see :func:`set_tile_subpriority`.

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
    if Path(fiber_map_fn).is_file() and not overwrite:
        logger.info('Fiber map {} already exists, reusing it.'.format(fiber_map_fn))
        return FiberMap.read(fiber_map_fn)

    input_dir = utils.get_fiberassign_input_dir(tileid, survey=survey,
                                                fiberassign_input_dir=fiberassign_input_dir)
    footprint_fn = Path(input_dir) / '{}-tiles.fits'.format(ts)
    sky_fn = Path(input_dir) / '{}-sky.fits'.format(ts)
    for name, fn in [('footprint', footprint_fn), ('sky', sky_fn)]:
        if not Path(fn).is_file():
            raise ValueError('{} file {} of tile {:d} not found'.format(name, fn, tileid))
    # Secondary targets and targets of opportunity exist only for some tiles.
    scnd_fn = Path(input_dir) / '{}-scnd.fits'.format(ts)
    if not Path(scnd_fn).is_file(): scnd_fn = None
    too_fn = Path(input_dir) / '{}-too.fits'.format(ts)
    if not Path(too_fn).is_file(): too_fn = None

    # The target file is handed to fiberassign and never read again, so it goes to memory
    # rather than to the file system the assignments are written to.
    targets_dir = utils.get_tmp_dir(tmp_dir)
    if targets_dir is None:
        targets_dir = fa_dir
    else:
        # A directory of its own per realization. The file is named after the tile, and the
        # memory it goes to is shared by everything on the node, so two mocks replayed at once
        # reach the same tile and one overwrites or half-reads the other's targets.
        token = hashlib.md5(str(Path(altmtl_dir).resolve()).encode()).hexdigest()[:10]
        targets_dir = Path(targets_dir) / 'altmtl-{}'.format(token)
        utils.mkdir(targets_dir)
    targets_fn = Path(targets_dir) / '{}-targ.fits'.format(ts)
    ledger_dir = None if state is not None else get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon)
    # A legacy tile is assigned in another interpreter, which can only be handed a file.
    in_memory = load_targets == 'memory' and not is_legacy(header)
    try:
        if in_memory:
            array = read_alt_targets(tileid, ledger_dir, footprint_fn, state=state,
                                     tile_subpriority=tile_subpriority)
            ntargets = len(array)
        else:
            array = None
            ntargets = write_alt_targets(tileid, ledger_dir, targets_fn, footprint_fn, state=state,
                                         tile_subpriority=tile_subpriority)
        logger.debug('Tile {:d}: {:d} alternative targets in footprint.'.format(tileid, ntargets))

        run_fiber_assignment(tileid, targets_fn, fa_dir, header, footprint_fn, sky_fn,
                             scnd_fn=scnd_fn, too_fn=too_fn, overwrite=overwrite,
                             fiberassign_dir=fiberassign_dir, targets=array, survey=survey)
    finally:
        if not in_memory and targets_dir is not fa_dir and Path(targets_fn).is_file():
            os.remove(targets_fn)

    fiber_map = make_fiber_map(real_assignment, read_alt_assignment(fa_dir, tileid))
    fiber_map.write(fiber_map_fn)
    logger.info('Tile {:d}: mapped {:d} fibers, {:d} of them to a different target.'.format(
        tileid, len(fiber_map), int((fiber_map.real_targetid != fiber_map.alt_targetid).sum())))
    return fiber_map

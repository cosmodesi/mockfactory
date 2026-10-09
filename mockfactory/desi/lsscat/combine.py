"""
Combination of the per-tile fiber assignment products into the survey-wide tables.

Three things come out of the alternative assignment of a mock, per tile: which targets each
fiber could have reached, which target each fiber was given, and the priority the assignment
saw. Put end to end over the tiles of the survey, they are what every later stage reads, and
a target appears once per tile that could have reached it.

The survey pipeline writes these out, 173 GB per realization between the data and the eighteen
random catalogs, and reads them back at each stage. Nothing here writes: the tables are
returned, and :mod:`mockfactory.desi.lsscat.pipeline` hands them straight to the next stage.
"""

import functools
import logging
from pathlib import Path

import numpy as np

from ..altmtl.assignment import get_alt_fiberassign_fn, get_fa_dir
from ..altmtl.tiletracker import read_tile_tracker
from astropy.table import Table, vstack

from .utils import NULL, as_table, encode_keys, join_left, set_column


logger = logging.getLogger('lsscat.combine')


def read_assignments(altmtl_dir, tileids, fadates=None, survey='main', obscon='dark', numproc=1):
    """
    Return, for the tiles of the survey, the target each fiber was given.

    Columns are ``TARGETID``, ``LOCATION``, ``TILEID``, and the ``PRIORITY`` and
    ``SUBPRIORITY`` the assignment saw, which is what tells a later stage whether a target lost
    its fiber to a higher priority one or was simply not reached.

    Parameters
    ----------
    altmtl_dir : str
        Directory of the alternative realization, holding ``Univ000/fa``.
    tileids : array
        Tiles to combine.
    fadates : array, default=None
        Assignment date of each tile, naming the subdirectory it was written to. Read from the
        tile tracker if not given.
    obscon : str, default='dark'
        Observing conditions, which names the tile tracker the dates are read from.
    numproc : int, default=1
        Number of processes to read with.
    """
    if fadates is None:
        # Located rather than computed: the directory is named after the date fiberassign ran
        # for that tile, which is not the date of the action that asked for it.
        found = _find_assignments(altmtl_dir, survey=survey)
        missing = [tileid for tileid in tileids if int(tileid) not in found]
        if missing:
            raise ValueError('no assignment written for tiles {}{}'.format(
                missing[:10], ' and {:d} more'.format(len(missing) - 10) if len(missing) > 10 else ''))
        args = [(found[int(tileid)],) for tileid in tileids]
    else:
        args = [(get_alt_fiberassign_fn(get_fa_dir(altmtl_dir, fadate, survey=survey), tileid),)
                for tileid, fadate in zip(tileids, fadates)]
    arrays = _map(_read_assignment_one_tile, args, numproc=numproc)
    toret = vstack([array for array in arrays if len(array)], join_type='exact',
                   metadata_conflicts='silent')
    logger.info('combined {:d} assignments over {:d} tiles'.format(len(toret), len(tileids)))
    return toret


def _find_assignments(altmtl_dir, survey='main'):
    """Return, per tile, the assignment written for it, wherever its date put it."""
    import glob
    import re
    toret = {}
    for fn in (Path(altmtl_dir) / 'fa' / survey.upper()).glob('*/fba-*.fits'):
        match = re.search(r'fba-(\d+)\.fits$', Path(fn).name)
        if match:
            toret[int(match.group(1))] = fn
    logger.info('found {:d} assignments under {}'.format(len(toret), altmtl_dir))
    return toret


def _read_assignment_one_tile(fn):
    """Return the assignment of one tile, with its priorities joined on."""
    import fitsio
    import re
    tileid = int(re.search(r'fba-(\d+)\.fits$', Path(fn).name).group(1))
    with fitsio.FITS(fn) as fits:
        assigned = fits['FASSIGN'].read(columns=['TARGETID', 'LOCATION'])
        targets = fits['FTARGETS'].read(columns=['TARGETID', 'PRIORITY', 'SUBPRIORITY'])
    # A location with no target carries a negative identifier; sky and standard targets are
    # dropped by the join, since they are not in the target file.
    toret = as_table(assigned[assigned['TARGETID'] >= 0])
    set_column(toret, 'TILEID', tileid, dtype='i8')
    return join_left(toret, targets, 'TARGETID')


def _get_fadates(altmtl_dir, tileids, survey='main', obscon='dark'):
    """Return the assignment date of each tile, from the tile tracker of the realization."""
    actions = read_tile_tracker(altmtl_dir, survey=survey, obscon=obscon)
    actions = actions[actions['ACTIONTYPE'] == 'fa']
    index = np.searchsorted(actions['TILEID'], tileids, sorter=np.argsort(actions['TILEID']))
    order = np.argsort(actions['TILEID'])
    index = order[np.clip(index, 0, len(order) - 1)]
    if not np.all(actions['TILEID'][index] == tileids):
        missing = tileids[actions['TILEID'][index] != tileids]
        raise ValueError('no assignment action for tiles {}'.format(missing[:10]))
    return [str(date).split('T')[0].replace('-', '') for date in actions['ACTIONTIME'][index]]


#: Default spectroscopic signal-to-noise floor, and the column it applies to, per program.
TSNR2_MIN = {'dark': ('TSNR2_ELG', 80.), 'bright': ('TSNR2_BGS', 1000.)}

#: Bits of ``COADD_FIBERSTATUS`` that mark a fiber whose spectrum cannot be used. The survey
#: rejects on these two for the second data assembly; rejecting on every bit that exists
#: removes far more and leaves the catalogs short.
BAD_FIBERSTATUS = (13, 14)


#: Lists the LSS catalogs drew up after the fact, of fibers and of petal nights whose redshifts
#: they do not trust; the spectra exist, and the merged target list used them at the time. They
#: are release specific, hence paths rather than code.
BAD_FIBER_FN = {'dark': ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/bad_nz_fibers_ks_test.txt',
                         '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/elg_bad_nz_spike_fibers_1.498_1.499.txt'],
                'bright': ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/bad_nz_fibers_ks_test.txt']}

BAD_PETAL_NIGHT_FN = {'dark': '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/lrg_bad_per_petal-night.txt',
                      'bright': '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/bgs_bright_bad_per_petal-night.txt'}


def read_bad_petal_nights(fn):
    """Return the (night, petal) pairs whose spectra the LSS catalogs reject."""
    toret = []
    with open(fn) as file:
        for line in file:
            fields = line.split()
            if fields:
                toret += [(int(fields[0]), int(petal)) for petal in fields[1:]]
    return toret


def _on_bad_petal_night(spec, bad_petal_nights, program='dark'):
    """
    Return whether each spectrum of ``spec`` (with ``FIBER``, ``LASTNIGHT``) was observed on a bad
    petal night: ``bad_petal_nights`` as ``(night, petal)`` pairs, the path of a list, or ``True``
    for :data:`BAD_PETAL_NIGHT_FN` of ``program``.
    """
    if bad_petal_nights is True:
        bad_petal_nights = BAD_PETAL_NIGHT_FN[program]
    if isinstance(bad_petal_nights, str):
        bad_petal_nights = read_bad_petal_nights(bad_petal_nights)
    # A petal is five hundred consecutive fibers, so a night and a petal name a block.
    bad = np.zeros(len(spec), dtype='?')
    for night, petal in bad_petal_nights:
        bad |= ((spec['LASTNIGHT'] == night) & (spec['FIBER'] >= 500 * petal)
                & (spec['FIBER'] < 500 * (petal + 1)))
    return bad


BAD_FIBER_TIME_FN = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/unique_badfibers_time-dependent.txt'


def read_bad_fibers_time_dependent(fn):
    """
    Return the fibers the LSS catalogs reject for part of the survey, as
    ``(fiber, [(first night, last night), ...])``.

    A line names a fiber and then the nights it was bad over, as half open intervals; a night
    left without a partner closes the list and means "from then on". A fiber on its own was bad
    throughout.
    """
    toret = []
    with open(fn) as file:
        for line in file:
            fields = [int(field) for field in line.split()]
            if not fields:
                continue
            fiber, nights = fields[0], fields[1:]
            if not nights:
                toret.append((fiber, None))
                continue
            spans = [(nights[i], nights[i + 1] if i + 1 < len(nights) else None)
                     for i in range(0, len(nights), 2)]
            toret.append((fiber, spans))
    return toret


def read_good_tilelocid(spec_fn, program='dark', tsnr2_min=None, fiberstatus_bits=BAD_FIBERSTATUS,
                        bad_fibers=None, bad_petal_nights=None, bad_fibers_time=None):
    """
    Return the fiber locations the real survey got a usable spectrum from.

    A mock is assigned on the real survey's focal plane and inherits its broken fibers and bad
    petals: a location that gave no usable spectrum in the data would have given none for a
    mock target either, so it counts neither towards a target's tile coverage nor towards the
    completeness of its neighbours.

    Parameters
    ----------
    spec_fn : str
        Combined spectroscopic table of the real survey, ``datcomb_{program}_spec_zdone.fits``.
    program : str, default='dark'
        Observing program, setting which signal-to-noise column is tested.
    tsnr2_min : float, default=None
        Signal-to-noise floor. Defaults to the survey's value for the program.
    fiberstatus_bits : tuple
        Bits of ``COADD_FIBERSTATUS`` that disqualify a location.
    bad_fibers : array, str, list, default=None
        Fibers found after the fact to have a poor redshift success rate, as values or as the
        paths of the LSS lists. ``True`` uses :data:`BAD_FIBER_FN` for the program.
    bad_petal_nights : list, str, default=None
        Nights and petals whose spectra the LSS catalogs reject, as ``(night, petal)`` pairs or
        as the path of their list. ``True`` uses :data:`BAD_PETAL_NIGHT_FN`.
    bad_fibers_time : list, str, default=None
        Fibers rejected over part of the survey only; see
        :func:`read_bad_fibers_time_dependent`. ``True`` uses :data:`BAD_FIBER_TIME_FN`.
    """
    import fitsio
    from desitarget.targetmask import zwarn_mask
    column, default = TSNR2_MIN[program]
    if tsnr2_min is None:
        tsnr2_min = default
    if bad_fibers is True:
        bad_fibers = BAD_FIBER_FN[program]
    if isinstance(bad_fibers, str):
        bad_fibers = [bad_fibers]
    if bad_fibers is not None and len(bad_fibers) and isinstance(bad_fibers[0], str):
        bad_fibers = np.concatenate([np.atleast_1d(np.loadtxt(fn)) for fn in bad_fibers])
    if bad_fibers_time is True:
        bad_fibers_time = BAD_FIBER_TIME_FN
    if isinstance(bad_fibers_time, str):
        bad_fibers_time = read_bad_fibers_time_dependent(bad_fibers_time)
    columns = ['TILEID', 'LOCATION', 'FIBER', 'ZWARN', 'ZWARN_MTL', 'COADD_FIBERSTATUS', column]
    if bad_petal_nights or bad_fibers_time:
        columns.append('LASTNIGHT')
    spec = fitsio.read(str(spec_fn), columns=columns)
    select = (spec['ZWARN'] != NULL) & (spec['ZWARN'] * 0 == 0)
    select &= (spec['ZWARN_MTL'] & zwarn_mask.mask('NODATA|BAD_SPECQA|BAD_PETALQA')) == 0
    select &= spec[column] >= tsnr2_min
    for bit in fiberstatus_bits:
        select &= (spec['COADD_FIBERSTATUS'] & 2**bit) == 0
    if bad_fibers is not None:
        select &= ~np.isin(spec['FIBER'], bad_fibers)
    if bad_fibers_time:
        bad = np.zeros(len(spec), dtype='?')
        for fiber, spans in bad_fibers_time:
            on_fiber = spec['FIBER'] == fiber
            if spans is None:
                bad |= on_fiber
                continue
            for first, last in spans:
                within = spec['LASTNIGHT'] >= first
                if last is not None:
                    within &= spec['LASTNIGHT'] < last
                bad |= on_fiber & within
        logger.info('{:d} spectra rejected by the time dependent fiber list'
                    .format(int(bad.sum())))
        select &= ~bad
    if bad_petal_nights:
        bad = _on_bad_petal_night(spec, bad_petal_nights, program=program)
        logger.info('{:d} spectra rejected by the petal night list'.format(int(bad.sum())))
        select &= ~bad
    logger.info('{:d} of {:d} locations gave a usable spectrum'.format(select.sum(), len(spec)))
    return np.unique(10000 * spec['TILEID'][select].astype('i8') + spec['LOCATION'][select])


#: The LSS catalogs' own products of the bad petal night lists: the locations, per program, and per random
#: catalog the randoms with a row there (ran_{i}_{program}_badpetalnight_TARGETID.txt).
BAD_PETAL_NIGHT_DIR = Path('/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1')
BAD_PETAL_NIGHT_TILELOCID_FN = {'dark': BAD_PETAL_NIGHT_DIR / 'dark_badpetalnight_TILELOCID.txt'}


def mask_bad_petal_night_targetid(rows, program='dark', spec_fn=None, bad_petal_nights=True):
    """
    Return the targets with a row at a fiber location the LSS catalogs reject for being observed
    on a bad petal night.

    The spectra were taken, and the merged target list read their redshifts at the time and
    marked the targets done; only the catalogs drop them, after the fact. Removing the targets
    returned here from the data, and the randoms of :func:`read_bad_petal_night_random_targetid`
    from the randoms, masks those locations at the object level: a target is dropped if any
    fiber that could reach it is there, whether or not it got that fiber. Data and randoms lose
    the same area, so the mask applies to the altmtl and to the complete catalogs alike; see
    ``mask_targetid`` and ``mask_random_targetid`` of
    :func:`~mockfactory.desi.lsscat.pipeline.run_tracer`.

    The rows must be the raw ones, the mock's potential assignments ``pota-{PROGRAM}``: the
    combined potential assignments of :func:`combine_data` have already lost the locations
    outside ``good_tilelocid``, and with them every location the catalogs reject.

    Parameters
    ----------
    rows : array, str, Path
        With ``TARGETID``, ``TILEID``, ``LOCATION``, or the path of a file holding them.
    program : str, default='dark'
        Observing program.
    spec_fn : str, Path, default=None
        Combined spectroscopic table of the real survey, ``datcomb_{program}_spec_zdone.fits``,
        to find the locations in, with ``bad_petal_nights``; read once per file and kept.
        Defaults to the LSS catalogs' list of locations, :data:`BAD_PETAL_NIGHT_TILELOCID_FN`,
        which only the dark program has.
    bad_petal_nights : str, bool, default=True
        With ``spec_fn``: path of the list of nights and petals; ``True`` uses
        :data:`BAD_PETAL_NIGHT_FN`.
    """
    import fitsio
    if isinstance(rows, (str, Path)):
        rows = fitsio.read(str(rows), columns=['TARGETID', 'TILEID', 'LOCATION'])
    if spec_fn is None:
        if program not in BAD_PETAL_NIGHT_TILELOCID_FN:
            raise ValueError('no list of bad petal night locations for program {}; give spec_fn'.format(program))
        tilelocid = np.loadtxt(BAD_PETAL_NIGHT_TILELOCID_FN[program], dtype='i8')
    else:
        if isinstance(bad_petal_nights, Path):
            bad_petal_nights = str(bad_petal_nights)
        tilelocid = _read_bad_petal_night_tilelocid(str(spec_fn), program, bad_petal_nights)
    tl = 10000 * np.asarray(rows['TILEID'], dtype='i8') + np.asarray(rows['LOCATION'])
    return np.unique(np.asarray(rows['TARGETID'])[np.isin(tl, tilelocid)])


def read_bad_petal_night_random_targetid(i, program='dark'):
    """
    Return the randoms of the survey's random catalog ``i`` with a row at a location the LSS
    catalogs reject for being observed on a bad petal night, as the LSS catalogs list them from
    ``rancomb_{i}{program}wdupspec_zdone``: the randoms' counterpart of
    :func:`mask_bad_petal_night_targetid`.
    """
    return np.loadtxt(BAD_PETAL_NIGHT_DIR / 'ran_{:d}_{}_badpetalnight_TARGETID.txt'.format(i, program), dtype='i8')


@functools.lru_cache
def _read_bad_petal_night_tilelocid(spec_fn, program, bad_petal_nights):
    """The locations of ``spec_fn`` observed on a bad petal night, ``10000 * TILEID + LOCATION``."""
    import fitsio
    spec = fitsio.read(spec_fn, columns=['TILEID', 'LOCATION', 'FIBER', 'LASTNIGHT'])
    bad = _on_bad_petal_night(spec, bad_petal_nights, program=program)
    toret = np.unique(10000 * spec['TILEID'][bad].astype('i8') + spec['LOCATION'][bad])
    logger.info('{:d} spectra, {:d} locations on bad petal nights'.format(int(bad.sum()), toret.size))
    return toret


#: Imaging randoms the catalogs are drawn from, carrying the legacy survey columns.
RANDOMS_DIR = '/dvs_ro/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve'

#: Per-tracer masks built after the fact, one file per random catalog.
RANDOM_MASK_DIR = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/main/LSS'


def read_random_imaging(rann, tracer=None, randoms_dir=None, mask_dir=None):
    """
    Return the imaging columns of one parent random catalog.

    The randoms laid over the tiles carry only positions and identifiers; what the imaging
    vetoes read comes from the catalog they were drawn from, matched by identifier. The
    luminous red galaxies have a mask of their own on top, built after the legacy survey bits
    and distributed per random catalog.

    Parameters
    ----------
    rann : int
        Index of the random catalog.
    tracer : str, default=None
        Target class, for the tracer specific mask. None reads only the legacy survey columns.
    randoms_dir : str, default=None
        Directory of the parent randoms. Defaults to :data:`RANDOMS_DIR`.
    mask_dir : str, default=None
        Directory of the tracer masks. Defaults to :data:`RANDOM_MASK_DIR`.
    """
    import fitsio
    randoms_dir = RANDOMS_DIR if randoms_dir is None else randoms_dir
    mask_dir = RANDOM_MASK_DIR if mask_dir is None else mask_dir
    toret = as_table(fitsio.read(Path(randoms_dir) / 'randoms-1-{:d}.fits'.format(rann),
                                   columns=['TARGETID', 'MASKBITS', 'PHOTSYS', 'NOBS_G', 'NOBS_R',
                                            'NOBS_Z']))
    if tracer is not None and tracer[:3] == 'LRG':
        mask = fitsio.read(Path(mask_dir) / 'randoms-1-{:d}lrgimask.fits'.format(rann))
        toret = join_left(toret, mask, 'TARGETID', columns=['lrg_mask'])
    logger.info('read imaging for random {:d}: {:d} rows'.format(rann, len(toret)))
    return toret


def join_on_assigned_location(potential, assignments, columns):
    """
    Add ``columns`` of ``assignments`` to the potential assignments they belong to.

    A row of ``potential`` is a target a fiber could have reached; it took that fiber only if
    the assignment at that location went to that target. Matching on the three of
    ``TARGETID``, ``LOCATION`` and ``TILEID`` says the same thing in a more expensive way: a
    location holds one target, so looking the location up and then asking whether the target
    it holds is this one gives the same answer from a single integer key. The composite key
    costs a dense relabelling of every column over both tables, tens of millions of rows each.

    Parameters
    ----------
    potential : array
        Potential assignments, carrying ``TARGETID`` and ``TILELOCID``.
    assignments : array
        Fibers given, carrying ``TARGETID``, ``LOCATION`` and ``TILEID``, each location once.
    columns : list
        Columns of ``assignments`` to add.
    """
    from .utils import last_of_each, match

    potential, assignments = as_table(potential), as_table(assignments)
    won = Table({name: assignments[name] for name in ['TARGETID'] + list(columns)}, copy=False)
    set_column(won, 'TILELOCID', 10000 * assignments['TILEID'].astype('i8')
               + assignments['LOCATION'], dtype='i8')
    # One row per location, in case the assignment table repeats one.
    won = won[last_of_each(won['TILELOCID'])]

    index = match(potential['TILELOCID'], won['TILELOCID'])
    found = index >= 0
    at = np.where(found, index, 0)
    # The location was this target's only if the target it holds is this one.
    taken = found & (won['TARGETID'][at] == potential['TARGETID'])

    toret = potential.copy(copy_data=False)
    for name in columns:
        column = won[name].value[at]
        if not taken.all():
            fill = np.nan if column.dtype.kind == 'f' \
                else NULL if column.dtype.kind in 'iu' else column.dtype.type()
            column = np.where(taken.reshape((-1,) + (1,) * (column.ndim - 1)), column,
                              fill).astype(won[name].dtype, copy=False)
        set_column(toret, name, column)
    logger.info('{:d} of {:d} potential assignments were taken'.format(int(taken.sum()), len(toret)))
    return toret


def combine_data(potential, assignments, targets=None, columns=('RSDZ', 'TRUEZ', 'ZWARN'),
                 collisions=True, good_tilelocid=None):
    """
    Return the potential assignments of the survey, with what actually happened at each of
    them, and the mock truth of each target.

    This is the table every later stage starts from: one row per target per fiber that could
    have reached it, carrying ``ZWARN`` where the fiber was in the end given to that target
    and :data:`~mockfactory.desi.lsscat.utils.NULL` where it was not.

    Parameters
    ----------
    potential : array
        Potential assignments, from
        :func:`~mockfactory.desi.altmtl.pota.compute_potential_assignments`.
    assignments : array
        Fibers given, from :func:`read_assignments`.
    targets : array, default=None
        Target catalog, for the mock truth columns. Taken from ``potential`` when it carries
        them already.
    columns : tuple
        Truth columns to carry over from ``targets``.
    collisions : bool, array, default=True
        The potential assignments a fiber collision made impossible, as a table of
        ``TARGETID``, ``LOCATION`` and ``TILEID`` to remove. ``True`` uses the ``COLLISION``
        column of ``potential`` instead, and ``False`` removes none.
    good_tilelocid : array, default=None
        Fiber locations the real survey got a usable spectrum from, from
        :func:`read_good_tilelocid`. Locations outside it are dropped, since no mock target
        could have been observed there either. Nothing is dropped when not given.
    """
    # Kept before the filter below rebinds it: these name what happened at the fiber, and the
    # potential assignments carry the target's own copy of every one of them.
    truth = tuple(columns)
    potential, assignments = as_table(potential), as_table(assignments)
    size = len(potential)
    if collisions is True:
        if 'COLLISION' in potential.colnames:
            potential = potential[potential['COLLISION'] == 0]
    elif collisions is not False and collisions is not None:
        collisions = as_table(collisions)
        keys = ['TARGETID', 'LOCATION', 'TILEID']
        code = encode_keys(*[np.concatenate([potential[key], collisions[key]]) for key in keys])
        potential = potential[~np.isin(code[:size], code[size:])]
    if len(potential) != size:
        logger.info('{:d} potential assignments left after removing collisions'
                    .format(len(potential)))

    columns = [column for column in columns if column not in assignments.colnames]
    if columns:
        if targets is None:
            missing = [column for column in columns if column not in potential.colnames]
            if missing:
                raise ValueError('truth columns {} need the target catalog'.format(missing))
            # The potential assignments repeat each target's truth unchanged on every row it
            # appears in, so one row per target is a target catalog. This is what the
            # docstring means by taking them from ``potential``; it has to happen here,
            # against the assignment table, and not by leaving the copies in place below.
            index = np.unique(potential['TARGETID'], return_index=True)[1]
            targets = potential[index]
        assignments = join_left(assignments, targets, 'TARGETID', columns=columns)

    # The potential assignments repeat the target's truth on every fiber that could have
    # reached it, under the same names the assignment table uses. Left in place they survive
    # the join below, which only adds names the table does not already have, and then every
    # potential assignment carries a redshift and a ZWARN as though its fiber had been given
    # to it -- so ZWARN != NULL everywhere, every target looks assigned, FRACZ_TILELOCID comes
    # out 1 and the completeness weight with it. Drop them, and let the join put them back
    # from the assignment side, NULL where the fiber went to another target.
    toret = potential.copy(copy_data=False)
    toret.remove_columns([name for name in truth if name in toret.colnames])
    set_column(toret, 'TILELOCID', 10000 * toret['TILEID'] + toret['LOCATION'], dtype='i8')
    if good_tilelocid is not None:
        keep = np.isin(toret['TILELOCID'], good_tilelocid)
        logger.info('{:d} of {:d} potential assignments are at a usable location'
                    .format(int(keep.sum()), len(keep)))
        toret = toret[keep]
    add = [name for name in assignments.colnames
           if name not in ('TARGETID', 'LOCATION', 'TILEID') and name not in toret.colnames]
    toret = join_on_assigned_location(toret, assignments, add)
    # The merged target list saw the same warning bits as the truth, since a mock has no
    # spectroscopic failures of its own; later stages read one or the other.
    if 'ZWARN' in toret.colnames and 'ZWARN_MTL' not in toret.colnames:
        set_column(toret, 'ZWARN_MTL', toret['ZWARN'].value.copy())
    logger.info('{:d} potential assignments, {:d} of them observed'
                .format(len(toret), int(np.sum(toret['ZWARN'] != NULL))))
    return toret


def combine_randoms(randoms, assignments, columns=('PRIORITY',)):
    """
    Return the survey's randoms with the priority the mock's assignment gave each location.

    The randoms are the real survey's, laid over the tiles it observed; what makes them a
    mock's randoms is the priority that ruled at each fiber location, since that is what says
    whether the tracer could have been put there. Everything else, the positions and the
    signal-to-noise of the real observation, is kept.

    Parameters
    ----------
    randoms : array
        Randoms with one row per tile reaching them, carrying ``LOCATION`` and ``TILEID``.
    assignments : array
        Fibers given, from :func:`read_assignments`.
    columns : tuple
        Columns to take from the assignment, replacing any the randoms already carry.
    """
    from .utils import last_of_each
    assignments = as_table(assignments)
    won = Table({name: assignments[name] for name in columns}, copy=False)
    set_column(won, 'TILELOCID', 10000 * assignments['TILEID'].astype('i8')
               + assignments['LOCATION'], dtype='i8')
    won = won[last_of_each(won['TILELOCID'])]

    # A copy, however shallow: columns are about to be set on it, and those are the caller's.
    toret = as_table(randoms).copy(copy_data=False)
    set_column(toret, 'TILELOCID', 10000 * toret['TILEID'].astype('i8') + toret['LOCATION'],
               dtype='i8')
    # A location the mock never reached keeps no priority, so nothing can be assigned there.
    toret.remove_columns([name for name in columns if name in toret.colnames])
    toret = join_left(toret, won, 'TILELOCID', columns=list(columns))
    logger.info('{:d} randoms re-priced from the mock assignment'.format(len(toret)))
    return toret


def read_dupran_randoms(fn, assignments, max_priority):
    """
    Return the full random catalog of one tracer from the survey's own vetoed randoms, as
    LSS ``mkCat_amtl.py`` builds the randoms of a mock.

    ``{tracer}_{i}_dupran_masked_HPmapcut`` holds one row per random and fiber location that
    could reach it, already cut on usable locations and on the imaging and map vetoes of the
    tracer. What is left to the mock is the priority: the rows at a location the mock gave to a
    target above the tracer's maximum priority are dropped, and each random keeps one row.

    It gives the same randoms as :func:`combine_randoms` followed by
    :func:`~mockfactory.desi.lsscat.full.make_full_randoms` and
    :func:`~mockfactory.desi.lsscat.veto.apply_veto_randoms`, with the same ``NTILE`` and
    ``TILES``, at a third of the cost, with two exceptions: it keeps the randoms whose only
    locations the mock left unassigned (some 20 in 25 million), and it carries the veto on
    imaging bits 1, 12 and 13 that the survey applies to every dark-time tracer.

    Parameters
    ----------
    fn : str, Path
        The survey's ``{tracer}_{i}_dupran_masked_HPmapcut`` catalog, ``.h5`` or ``.fits``.
    assignments : array
        Fibers given, from :func:`read_assignments`.
    max_priority : int
        Highest priority the tracer can be assigned at, from
        :func:`~mockfactory.desi.lsscat.full.get_max_priority`.
    """
    columns = ['TARGETID', 'RA', 'DEC', 'TILEID', 'LOCATION', 'NTILE', 'TILES', 'PHOTSYS']
    fn = str(fn)
    if fn.endswith('.h5'):
        import h5py
        import hdf5plugin  # noqa: F401  registers the codec
        with h5py.File(fn, 'r') as file:
            rows = {name: file['LSS'][name][...] for name in columns}
    else:
        import fitsio
        array = fitsio.read(fn, columns=columns)
        rows = {name: array[name] for name in columns}
    assignments = as_table(assignments)
    won = 10000 * np.asarray(assignments['TILEID'], dtype='i8') + np.asarray(assignments['LOCATION'])
    bad = won[np.asarray(assignments['PRIORITY']) > max_priority]
    keep = ~np.isin(10000 * rows['TILEID'].astype('i8') + rows['LOCATION'], bad)
    logger.info('{}: {:d} of {:d} rows at locations the mock gave a higher priority'
                .format(Path(fn).name, int((~keep).sum()), len(keep)))
    # NTILE and TILES belong to the random, not the row, so which row is kept does not matter.
    _, index = np.unique(rows['TARGETID'][keep], return_index=True)
    index = np.flatnonzero(keep)[index]
    toret = Table({'TARGETID': rows['TARGETID'][index], 'RA': rows['RA'][index],
                   'DEC': rows['DEC'][index], 'NTILE': rows['NTILE'][index].astype('i8'),
                   'TILES': tiles_code(rows['TILES'][index]),
                   'PHOTSYS': np.char.decode(rows['PHOTSYS'][index].astype('S1')).astype('U1')},
                  copy=False)
    logger.info('{}: {:d} randoms'.format(Path(fn).name, len(toret)))
    return toret


def tiles_code(names):
    """
    Return the code :func:`count_tiles` gives a set of tiles, from the name the survey pipeline
    gives it: the tile identifiers sorted and joined by ``-``.
    """
    names = np.asarray(names)
    unique, inverse = np.unique(names, return_inverse=True)
    prime = np.uint64(1099511628211)
    codes = np.empty(len(unique), dtype='u8')
    with np.errstate(over='ignore'):
        for i, name in enumerate(unique):
            name = name.decode() if isinstance(name, bytes) else str(name)
            code = np.uint64(14695981039346656037)
            for tile in sorted(int(tile) for tile in name.split('-')):
                code = (code ^ np.uint64(tile)) * prime
            codes[i] = code
    return codes[inverse].view('i8')


def count_tiles(array, tilelocids=False):
    """
    Return, per target, how many tiles could have reached it and which ones.

    ``NTILE`` is the number of distinct tiles and ``TILES`` stands for the set of them, so that
    a later stage can group targets by the overlap of tiles they sit under and measure how
    complete each such overlap is. The survey pipeline names the set by writing the tile
    identifiers out sorted and joined by ``-``; here the set is carried as a code instead, for
    the memory that name costs at the scale of a random catalog. See :func:`_group_code`.

    Parameters
    ----------
    array : array
        Potential assignments, with ``TARGETID``, ``TILEID`` and, for ``tilelocids``,
        ``TILELOCID``.
    tilelocids : bool, default=True
        Whether to also code the fiber locations, as ``TILELOCIDS``.
    """
    array = as_table(array)
    targetid, ntile, tiles = _group_code(array['TARGETID'], array['TILEID'])
    toret = Table({'TARGETID': targetid.astype(array['TARGETID'].dtype, copy=False),
                   'NTILE': ntile.astype('i8', copy=False), 'TILES': tiles}, copy=False)
    if tilelocids:
        set_column(toret, 'TILELOCIDS', _group_code(array['TARGETID'], array['TILELOCID'])[2])
    logger.info('counted tiles for {:d} targets, up to {:d} tiles each'
                .format(len(toret), int(ntile.max()) if len(ntile) else 0))
    return toret


def _group_code(targetid, value):
    """
    Return the distinct targets, how many distinct ``value`` each has, and a code standing for
    the sorted set of those values.

    The code is a hash of the set rather than a dense label, because the data and its randoms
    are grouped in separate calls and still have to agree: a later stage joins one to the other
    on this column, so the same set of tiles has to come out the same wherever it is met. A
    dense label would depend on which sets happened to be present in the call.

    Writing the set out as text agrees across calls too, and is what the survey pipeline does,
    but a name like ``1230-4560-7890`` takes a hundred and twenty bytes a row against eight,
    and at the thirty million rows of a random catalog that one column is four gigabytes.

    Two independent codes are accumulated in the same pass, which costs an exclusive or and a
    multiply on top of the masking that dominates the loop. Only the first is returned; the
    second is there to be checked against, so that a collision is raised rather than quietly
    merging two different overlaps of tiles and mis-weighting every target under them.
    """
    order = np.lexsort((value, targetid))
    sorted_targetid, sorted_value = np.asarray(targetid)[order], np.asarray(value)[order]
    distinct = np.empty(len(order), dtype='?')
    distinct[0] = True
    distinct[1:] = ((sorted_targetid[1:] != sorted_targetid[:-1])
                    | (sorted_value[1:] != sorted_value[:-1]))
    sorted_targetid, sorted_value = sorted_targetid[distinct], sorted_value[distinct]

    unique, start, counts = np.unique(sorted_targetid, return_index=True, return_counts=True)
    # Position of each value within its target's group, the values being already sorted.
    index = np.repeat(np.arange(len(unique)), counts)
    rank = np.arange(len(sorted_value)) - start[index]
    # Fowler-Noll-Vo over the values in sorted order, so that the code stands for the set and
    # not for the order the rows happened to arrive in.
    code = np.full(len(unique), np.uint64(14695981039346656037), dtype='u8')
    other = np.full(len(unique), np.uint64(14313749767032793493), dtype='u8')
    prime, other_prime = np.uint64(1099511628211), np.uint64(880355133930503)
    width = counts.max() if len(counts) else 0
    for i in range(width):
        select = rank == i
        at = index[select]
        seen = sorted_value[select].astype('u8')
        code[at] = (code[at] ^ seen) * prime
        other[at] = (other[at] ^ seen) * other_prime
    _check_code_collision(code, other)
    return unique, counts.astype('i8'), code.view('i8')


def _check_code_collision(code, other):
    """
    Raise if two distinct sets share a code, judged by a second, independent code.

    Distinct sets that agree on both codes would have to collide in a hundred and twenty eight
    bits at once, so counting the codes and counting the pairs is as good as comparing the sets
    themselves, and costs one sort of the targets rather than the sets written out.
    """
    if not len(code):
        return
    order = np.lexsort((other, code))
    first, second = code[order], other[order]
    changed = first[1:] != first[:-1]
    ncode = 1 + int(np.count_nonzero(changed))
    npair = 1 + int(np.count_nonzero(changed | (second[1:] != second[:-1])))
    if npair != ncode:
        raise ValueError('{:d} distinct sets of tiles share {:d} codes: the 64 bit code has '
                         'collided, which would merge unrelated overlaps of tiles'
                         .format(npair, ncode))


def _group_names(targetid, value):
    """
    Return the distinct targets, how many distinct ``value`` each has, and those values sorted
    and joined by ``-``.
    """
    # Sorted on the two columns and cut to distinct pairs. Deliberately not a unique over a
    # structured array: numpy falls back to a generic element comparison for those, which at
    # the hundred million rows of a random catalog costs minutes rather than seconds.
    order = np.lexsort((value, targetid))
    sorted_targetid, sorted_value = np.asarray(targetid)[order], np.asarray(value)[order]
    distinct = np.empty(len(order), dtype='?')
    distinct[0] = True
    distinct[1:] = ((sorted_targetid[1:] != sorted_targetid[:-1])
                    | (sorted_value[1:] != sorted_value[:-1]))
    sorted_targetid, sorted_value = sorted_targetid[distinct], sorted_value[distinct]

    unique, start, counts = np.unique(sorted_targetid, return_index=True, return_counts=True)
    # Position of each value within its target's group, the values being already sorted.
    index = np.repeat(np.arange(len(unique)), counts)
    rank = np.arange(len(sorted_value)) - start[index]
    strings = np.char.mod('%d', sorted_value)
    width = counts.max() if len(counts) else 0
    toret = np.zeros(len(unique), dtype='U{:d}'.format(max(1, width * (strings.dtype.itemsize // 4 + 1))))
    for i in range(width):
        select = rank == i
        at = index[select]
        toret[at] = strings[select] if i == 0 else np.char.add(np.char.add(toret[at], '-'), strings[select])
    return unique, counts.astype('i8'), toret


def _map(func, args, numproc=1):
    """Apply ``func`` to each of ``args``, in a pool of ``numproc`` processes."""
    if numproc <= 1:
        return [func(*arg) for arg in args]
    import multiprocessing
    with multiprocessing.Pool(numproc) as pool:
        return pool.starmap(func, args)

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

import logging
import os

import numpy as np

from ..altmtl.assignment import get_alt_fiberassign_fn, get_fa_dir
from ..altmtl.tiletracker import read_tile_tracker
from .utils import NULL, append_fields, drop_fields, encode_keys, join_left


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
    toret = np.concatenate([array for array in arrays if len(array)])
    logger.info('combined {:d} assignments over {:d} tiles'.format(len(toret), len(tileids)))
    return toret


def _find_assignments(altmtl_dir, survey='main'):
    """Return, per tile, the assignment written for it, wherever its date put it."""
    import glob
    import re
    toret = {}
    pattern = os.path.join(altmtl_dir, 'fa', survey.upper(), '*', 'fba-*.fits')
    for fn in glob.glob(pattern):
        match = re.search(r'fba-(\d+)\.fits$', os.path.basename(fn))
        if match:
            toret[int(match.group(1))] = fn
    logger.info('found {:d} assignments under {}'.format(len(toret), altmtl_dir))
    return toret


def _read_assignment_one_tile(fn):
    """Return the assignment of one tile, with its priorities joined on."""
    import fitsio
    import re
    tileid = int(re.search(r'fba-(\d+)\.fits$', os.path.basename(fn)).group(1))
    with fitsio.FITS(fn) as fits:
        assigned = fits['FASSIGN'].read(columns=['TARGETID', 'LOCATION'])
        targets = fits['FTARGETS'].read(columns=['TARGETID', 'PRIORITY', 'SUBPRIORITY'])
    # A location with no target carries a negative identifier; sky and standard targets are
    # dropped by the join, since they are not in the target file.
    assigned = assigned[assigned['TARGETID'] >= 0]
    toret = append_fields(assigned, [('TILEID', 'i8')])
    toret['TILEID'] = tileid
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


#: Lists the survey drew up after the fact, of fibers and of petal nights whose redshifts
#: cannot be trusted. They are survey and release specific, hence paths rather than code.
BAD_FIBER_FN = {'dark': ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/bad_nz_fibers_ks_test.txt',
                         '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/elg_bad_nz_spike_fibers_1.498_1.499.txt'],
                'bright': ['/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/bad_nz_fibers_ks_test.txt']}

BAD_PETAL_NIGHT_FN = {'dark': '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/lrg_bad_per_petal-night.txt',
                      'bright': '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/bgs_bright_bad_per_petal-night.txt'}


def read_bad_petal_nights(fn):
    """Return the (night, petal) pairs whose spectra the survey rejected."""
    toret = []
    with open(fn) as file:
        for line in file:
            fields = line.split()
            if fields:
                toret += [(int(fields[0]), int(petal)) for petal in fields[1:]]
    return toret


#: Fibers that went bad for part of the survey rather than all of it.
BAD_FIBER_TIME_FN = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/DA2/LSS/loa-v1/unique_badfibers_time-dependent.txt'


def read_bad_fibers_time_dependent(fn):
    """
    Return the fibers the survey rejected for part of its duration, as
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
        paths of the survey's lists. ``True`` uses :data:`BAD_FIBER_FN` for the program.
    bad_petal_nights : list, str, default=None
        Nights and petals whose spectra the survey rejected, as ``(night, petal)`` pairs or as
        the path of the survey's list. ``True`` uses :data:`BAD_PETAL_NIGHT_FN`.
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
    if bad_petal_nights is True:
        bad_petal_nights = BAD_PETAL_NIGHT_FN[program]
    if isinstance(bad_petal_nights, str):
        bad_petal_nights = read_bad_petal_nights(bad_petal_nights)
    if bad_fibers_time is True:
        bad_fibers_time = BAD_FIBER_TIME_FN
    if isinstance(bad_fibers_time, str):
        bad_fibers_time = read_bad_fibers_time_dependent(bad_fibers_time)
    columns = ['TILEID', 'LOCATION', 'FIBER', 'ZWARN', 'ZWARN_MTL', 'COADD_FIBERSTATUS', column]
    if bad_petal_nights or bad_fibers_time:
        columns.append('LASTNIGHT')
    spec = fitsio.read(spec_fn, columns=columns)
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
        # A petal is five hundred consecutive fibers, so a night and a petal name a block.
        bad = np.zeros(len(spec), dtype='?')
        for night, petal in bad_petal_nights:
            bad |= ((spec['LASTNIGHT'] == night) & (spec['FIBER'] >= 500 * petal)
                    & (spec['FIBER'] < 500 * (petal + 1)))
        logger.info('{:d} spectra rejected by the petal night list'.format(int(bad.sum())))
        select &= ~bad
    logger.info('{:d} of {:d} locations gave a usable spectrum'.format(select.sum(), len(spec)))
    return np.unique(10000 * spec['TILEID'][select].astype('i8') + spec['LOCATION'][select])


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
    toret = fitsio.read(os.path.join(randoms_dir, 'randoms-1-{:d}.fits'.format(rann)),
                        columns=['TARGETID', 'MASKBITS', 'PHOTSYS', 'NOBS_G', 'NOBS_R', 'NOBS_Z'])
    if tracer is not None and tracer[:3] == 'LRG':
        mask = fitsio.read(os.path.join(mask_dir, 'randoms-1-{:d}lrgimask.fits'.format(rann)))
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

    won = np.empty(len(assignments), dtype=[('TILELOCID', 'i8'),
                                            ('TARGETID', assignments['TARGETID'].dtype)]
                   + [(name, assignments[name].dtype, assignments[name].shape[1:])
                      for name in columns])
    won['TILELOCID'] = 10000 * assignments['TILEID'].astype('i8') + assignments['LOCATION']
    won['TARGETID'] = assignments['TARGETID']
    for name in columns:
        won[name] = assignments[name]
    # One row per location, in case the assignment table repeats one.
    won = won[last_of_each(won['TILELOCID'])]

    index = match(potential['TILELOCID'], won['TILELOCID'])
    found = index >= 0
    at = np.where(found, index, 0)
    # The location was this target's only if the target it holds is this one.
    taken = found & (won['TARGETID'][at] == potential['TARGETID'])

    toret = append_fields(potential, [(name, won[name].dtype, won[name].shape[1:])
                                      for name in columns])
    for name in columns:
        column = won[name][at]
        if not taken.all():
            fill = np.nan if column.dtype.kind == 'f' \
                else NULL if column.dtype.kind in 'iu' else column.dtype.type()
            column = np.where(taken.reshape((-1,) + (1,) * (column.ndim - 1)), column, fill)
        toret[name] = column
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
    size = len(potential)
    if collisions is True:
        if 'COLLISION' in potential.dtype.names:
            potential = potential[potential['COLLISION'] == 0]
    elif collisions is not False and collisions is not None:
        keys = ['TARGETID', 'LOCATION', 'TILEID']
        code = encode_keys(*[np.concatenate([potential[key], collisions[key]]) for key in keys])
        potential = potential[~np.isin(code[:size], code[size:])]
    if len(potential) != size:
        logger.info('{:d} potential assignments left after removing collisions'
                    .format(len(potential)))

    columns = [column for column in columns if column not in assignments.dtype.names]
    if columns:
        if targets is None:
            raise ValueError('truth columns {} need the target catalog'.format(columns))
        assignments = join_left(assignments, targets, 'TARGETID', columns=columns)

    toret = append_fields(potential, [('TILELOCID', 'i8')])
    toret['TILELOCID'] = 10000 * toret['TILEID'] + toret['LOCATION']
    if good_tilelocid is not None:
        keep = np.isin(toret['TILELOCID'], good_tilelocid)
        logger.info('{:d} of {:d} potential assignments are at a usable location'
                    .format(int(keep.sum()), len(keep)))
        toret = toret[keep]
    add = [name for name in assignments.dtype.names
           if name not in ('TARGETID', 'LOCATION', 'TILEID') and name not in toret.dtype.names]
    toret = join_on_assigned_location(toret, assignments, add)
    # The merged target list saw the same warning bits as the truth, since a mock has no
    # spectroscopic failures of its own; later stages read one or the other.
    if 'ZWARN' in toret.dtype.names and 'ZWARN_MTL' not in toret.dtype.names:
        toret = append_fields(toret, [('ZWARN_MTL', toret['ZWARN'].dtype)])
        toret['ZWARN_MTL'] = toret['ZWARN']
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
    won = np.empty(len(assignments), dtype=[('TILELOCID', 'i8')]
                   + [(name, assignments[name].dtype) for name in columns])
    won['TILELOCID'] = 10000 * assignments['TILEID'].astype('i8') + assignments['LOCATION']
    for name in columns:
        won[name] = assignments[name]
    from .utils import last_of_each
    won = won[last_of_each(won['TILELOCID'])]

    toret = randoms
    if 'TILELOCID' not in toret.dtype.names:
        toret = append_fields(toret, [('TILELOCID', 'i8')])
    toret['TILELOCID'] = 10000 * toret['TILEID'].astype('i8') + toret['LOCATION']
    # A location the mock never reached keeps no priority, so nothing can be assigned there.
    toret = drop_fields(toret, [name for name in columns if name in toret.dtype.names])
    toret = join_left(toret, won, 'TILELOCID', columns=list(columns))
    logger.info('{:d} randoms re-priced from the mock assignment'.format(len(toret)))
    return toret


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
    targetid, ntile, tiles = _group_code(array['TARGETID'], array['TILEID'])
    dtype = [('TARGETID', array['TARGETID'].dtype), ('NTILE', 'i8'), ('TILES', tiles.dtype)]
    if tilelocids:
        _, _, tilelocids_ = _group_code(array['TARGETID'], array['TILELOCID'])
        dtype += [('TILELOCIDS', tilelocids_.dtype)]
    toret = np.empty(len(targetid), dtype=dtype)
    toret['TARGETID'], toret['NTILE'], toret['TILES'] = targetid, ntile, tiles
    if tilelocids:
        toret['TILELOCIDS'] = tilelocids_
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

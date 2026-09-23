"""
The "full" catalogs: one row per target, carrying what became of it.

The combined tables hold a target once per fiber that could have reached it. A clustering
catalog needs it once, at the location that tells the most about it: the fiber it was given if
it got one, else a location that was usable, else any location at all. Choosing that row, and
recording how often the targets sharing a location or a set of tiles were observed, is what
this stage does. Nothing is cut yet, hence "full"; the vetoes come after.
"""

import logging

import numpy as np

from astropy.table import Table

from .utils import NULL, as_table, group_fraction, get_photsys, join_left, last_of_each, set_column


logger = logging.getLogger('lsscat.full')


#: Priority above which a target was not reachable, per tracer. A target of lower priority
#: than the one a fiber was given never had a chance at it, and the survey pipeline counts the
#: location against completeness only for the tracers that could have won it.
MAX_PRIORITY = {'LRG': 3200, 'ELG': 3400, 'QSO': 3400, 'BGS': 2100}


def get_max_priority(tracer, notqso=False):
    """Return the priority above which ``tracer`` could not have been assigned."""
    if notqso:
        return 3200
    return MAX_PRIORITY.get(tracer[:3], 3400)


def select_tracer(array, tracer, notqso=False, column=None):
    """
    Return the rows of ``array`` targeting ``tracer``.

    Parameters
    ----------
    array : array
        Any catalog carrying the targeting bit columns.
    tracer : str
        Target class, e.g. ``'LRG'``, ``'ELG_LOP'``, ``'QSO'``, ``'BGS_BRIGHT'``.
    notqso : bool, default=False
        Whether to reject the targets that are also quasar targets, as the emission line
        galaxy samples do to keep the two from overlapping.
    column : str, default=None
        Targeting bit column. Defaults to ``'BGS_TARGET'`` for the bright galaxy samples and
        ``'DESI_TARGET'`` otherwise.
    """
    from desitarget import targetmask
    if column is None:
        # The bright galaxy classes live in their own column, except the one that covers them
        # all, which is a bit of the main one.
        column = 'BGS_TARGET' if tracer in targetmask.bgs_mask.names() else 'DESI_TARGET'
    mask = targetmask.bgs_mask if column == 'BGS_TARGET' else targetmask.desi_mask
    select = (array[column] & mask[tracer]) > 0
    if notqso:
        select &= (array['DESI_TARGET'] & targetmask.desi_mask['QSO']) == 0
    logger.info('{:d} of {:d} rows target {}{}'
                .format(select.sum(), len(array), tracer, ' (not QSO)' if notqso else ''))
    return select


def make_full_data(data, assignments, tracer, tiles=None, targets=None, notqso=False,
                   truez='RSDZ', good_tilelocid=None, completeness_tiles=True,
                   truth=('RSDZ', 'TRUEZ', 'ZWARN', 'ZWARN_MTL')):
    """
    Return the full data catalog of one tracer.

    Parameters
    ----------
    data : array
        Combined potential assignments, from
        :func:`~mockfactory.desi.lsscat.combine.combine_data`.
    assignments : array
        Fibers given, from :func:`~mockfactory.desi.lsscat.combine.read_assignments`; the
        priority of the target that won each location says whether this tracer stood a chance.
    tracer : str
        Target class.
    tiles : array, default=None
        Tile counts, from :func:`~mockfactory.desi.lsscat.combine.count_tiles`. Counted here
        from the rows at good hardware when not given, which is what the survey pipeline does
        and what makes the count agree with ``good_tilelocid``.
    targets : array, default=None
        Target catalog, joined on for the imaging columns the vetoes need.
    notqso : bool, default=False
        Whether to reject targets that are also quasar targets.
    truez : str, default='RSDZ'
        Column of the mock truth to use as the observed redshift.
    good_tilelocid : array, default=None
        Fiber locations the real survey got a usable spectrum from. A mock inherits the real
        survey's broken fibers and bad petals, so a location missing from this list is one no
        mock target could have been observed at either. All locations count as usable when not
        given, which is right only if the assignment was already run against a masked focal
        plane.
    completeness_tiles : bool, default=True
        Whether to compute ``COMP_TILE``, the fraction observed among the targets sharing a
        set of tiles.
    truth : tuple
        Columns that say what happened at the fiber rather than what the target is, and which
        ``targets`` must therefore not overwrite. They come from
        :func:`~mockfactory.desi.lsscat.combine.combine_data`, which leaves them
        :data:`~mockfactory.desi.lsscat.utils.NULL` wherever the fiber went to another target.
    """
    maxp = get_max_priority(tracer, notqso=notqso)
    data, assignments = as_table(data), as_table(assignments)
    toret = data[select_tracer(data, tracer, notqso=notqso)]

    # The priority the winning target had, at each location this tracer could have reached.
    won = Table({'TILELOCID': (10000 * assignments['TILEID'].value
                               + assignments['LOCATION'].value).astype('i8', copy=False),
                 'PRIORITY_ASSIGNED': assignments['PRIORITY'].value.astype('i8', copy=False)},
                copy=False)
    won = won[last_of_each(won['TILELOCID'])]
    toret = join_left(toret, won, 'TILELOCID', fill={'PRIORITY_ASSIGNED': NULL})

    # A location whose winner outranked this tracer was never available to it; one no target
    # reached at all still was.
    set_column(toret, 'GOODPRI', (toret['PRIORITY_ASSIGNED'] <= maxp)
               | (toret['PRIORITY_ASSIGNED'] == NULL), dtype='?')
    # A mock inherits the real survey's broken fibers and bad petals, which is the only way a
    # location can be unusable to it: it has no spectroscopic failures of its own.
    set_column(toret, 'GOODHARDLOC', True if good_tilelocid is None
               else np.isin(toret['TILELOCID'], good_tilelocid), dtype='?')
    set_column(toret, 'LOCATION_ASSIGNED', (toret['ZWARN'] != NULL) & (toret['ZWARN'] * 0 == 0),
               dtype='?')
    set_column(toret, 'TILELOCID_ASSIGNED', np.isin(
        toret['TILELOCID'], np.unique(toret['TILELOCID'][toret['LOCATION_ASSIGNED']])), dtype='?')
    logger.info('{:d} assigned, {:d} of them at a good priority'
                .format(int(toret['LOCATION_ASSIGNED'].sum()),
                        int((toret['LOCATION_ASSIGNED'] & toret['GOODPRI']).sum())))

    if tiles is None:
        from .combine import count_tiles
        # Counted before the cut to one row per target, and only over the locations the target
        # could actually have been observed at: a tile reaching it through a broken fiber does
        # not make it any more likely to have been seen.
        tiles = count_tiles(toret[toret['GOODHARDLOC']])

    # Keep one row per target: the one that says the most about it. A quasar is ranked by the
    # priority it was observed at too, so that a target kept for Lyman-alpha follow-up wins
    # over the same target observed once.
    usable = toret['GOODHARDLOC'] & toret['GOODPRI']
    if tracer.startswith('QSO'):
        value = np.where(toret['LOCATION_ASSIGNED'], toret['PRIORITY'], 0)
    else:
        value = 1
    sort = (toret['LOCATION_ASSIGNED'] * usable * value + toret['TILELOCID_ASSIGNED'] * usable
            + toret['GOODHARDLOC'] + toret['GOODPRI'])
    toret = toret[last_of_each(toret['TARGETID'], sort=sort, tie=toret['TILELOCID'])]
    logger.info('cut to {:d} unique targets, {:d} of them assigned'
                .format(len(toret), int(toret['LOCATION_ASSIGNED'].sum())))

    toret = join_left(toret, tiles, 'TARGETID', fill={'NTILE': 0, 'TILES': 0, 'TILELOCIDS': 0})

    if targets is not None:
        targets = as_table(targets)
        # Positions and targeting bits come back from the target file, which carries the
        # imaging columns the vetoes read.
        #
        # Never the truth columns, though. The target file holds each target's own redshift
        # and ZWARN, and ``data`` holds what happened at the fiber -- NULL where the fiber
        # went to someone else. Letting the join overwrite them puts ZWARN = 0 back on every
        # row, so every target reads as observed however the assignment actually went, and
        # since LOCATION_ASSIGNED is settled above it stays right while the redshift selection
        # downstream goes wrong. FRACZ_TILELOCID then reaches 1 / 0 for the targets whose
        # fiber location never gave anything.
        columns = [name for name in targets.colnames
                   if name != 'TARGETID' and name not in truth]
        replaced = [name for name in ('RA', 'DEC', 'DESI_TARGET', 'BGS_TARGET') if name in columns]
        toret.remove_columns(replaced)
        toret = join_left(toret, targets, 'TARGETID', columns=columns)

    if truez in toret.colnames and 'Z' not in toret.colnames:
        # The observed redshift is the mock's own, carried over by the assignment; a target
        # that never got a fiber has none, and keeps the nan the join left.
        set_column(toret, 'Z', toret[truez].value.copy())

    if completeness_tiles:
        set_column(toret, 'COMP_TILE', group_fraction(toret['TILES'], toret['LOCATION_ASSIGNED']),
                   dtype='f8')
        logger.info('{:d} targets sit where nothing was observed'
                    .format(int((toret['COMP_TILE'] == 0).sum())))
    else:
        set_column(toret, 'COMP_TILE', 1., dtype='f8')
    # Of the targets of this tracer sharing a fiber location, the fraction that got observed;
    # one over it upweights a target for the ones it kept from being reached.
    set_column(toret, 'FRACZ_TILELOCID', group_fraction(toret['TILELOCID'],
                                                        toret['LOCATION_ASSIGNED']), dtype='f8')
    set_column(toret, 'PHOTSYS', get_photsys(toret['RA'], toret['DEC']), dtype='U1')
    return toret


def make_full_randoms(randoms, tracer, notqso=False, good_tilelocid=None, imaging=None,
                      tiles=None):
    """
    Return the full random catalog of one tracer.

    The randoms are the real survey's, laid over the tiles it observed, re-priced with the
    priority the mock's assignment gave each location: what they carry is which locations this
    tracer could have been assigned at, which is the selection function the data is divided by.

    Parameters
    ----------
    randoms : array
        Randoms with one row per tile reaching them, carrying the mock's ``PRIORITY``.
    tracer : str
        Target class.
    notqso : bool, default=False
        Whether the sample rejects quasar targets, which lowers the priority it could reach.
    good_tilelocid : array, default=None
        Locations the real survey got a usable spectrum from. All locations count as usable
        when not given.
    imaging : array, default=None
        Imaging columns of the parent randoms, joined on ``TARGETID``.
    tiles : array, default=None
        Tile counts, from :func:`~mockfactory.desi.lsscat.combine.count_tiles`. Counted here
        when not given. They are worth passing when several mocks share a random catalog: the
        count is taken over the locations that gave a usable spectrum, which is a property of
        the real survey, so every mock built on the same randoms gets the same answer and only
        the first need pay for it.
    """
    maxp = get_max_priority(tracer, notqso=notqso)
    # A copy, however shallow: columns are about to be set on it, and those are the caller's.
    toret = as_table(randoms).copy(copy_data=False)
    set_column(toret, 'TILELOCID', 10000 * toret['TILEID'] + toret['LOCATION'], dtype='i8')
    set_column(toret, 'ZPOSSLOC', True, dtype='?')
    set_column(toret, 'GOODHARDLOC', True if good_tilelocid is None
               else np.isin(toret['TILELOCID'], good_tilelocid), dtype='?')
    set_column(toret, 'GOODPRI', toret['PRIORITY'] <= maxp, dtype='?')
    logger.info('{:d} of {:d} random locations are usable by {}'
                .format(int((toret['GOODHARDLOC'] & toret['GOODPRI']).sum()), len(toret), tracer))

    if tiles is None:
        from .combine import count_tiles
        tiles = count_tiles(toret[toret['GOODHARDLOC']])

    sort = (toret['GOODPRI'] * toret['GOODHARDLOC'] * toret['ZPOSSLOC']
            + toret['GOODPRI'] * toret['GOODHARDLOC'])
    toret = toret[last_of_each(toret['TARGETID'], sort=sort, tie=toret['TILELOCID'])]
    logger.info('cut to {:d} unique randoms'.format(len(toret)))

    # The fiber location has done its work by here: what the vetoes and the clustering stage
    # read is the position, the imaging, the priority and the tiles. Dropped before the joins
    # rather than at the end, so that the row cuts that follow do not carry them along; at the
    # thirty million rows of a random catalog each column is a quarter of a gigabyte.
    done = ['LOCATION', 'FIBER', 'TILEID', 'TILELOCID', 'ZPOSSLOC', 'GOODPRI']
    toret.remove_columns([name for name in done if name in toret.colnames])
    toret = join_left(toret, tiles, 'TARGETID', fill={'NTILE': 0, 'TILES': 0, 'TILELOCIDS': 0})
    if imaging is not None:
        imaging = as_table(imaging)
        columns = [name for name in imaging.colnames if name not in toret.colnames]
        toret = join_left(toret, imaging, 'TARGETID', columns=columns)
    if 'PHOTSYS' not in toret.colnames:
        set_column(toret, 'PHOTSYS', get_photsys(toret['RA'], toret['DEC']), dtype='U1')
    return toret

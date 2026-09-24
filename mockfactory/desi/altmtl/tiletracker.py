"""
Action list driving the alternative MTL loop.

The alternative MTL replays the real survey: the same tiles are fiber-assigned and the same
ledger updates are applied, in the order the real survey did them. That order is recorded in
a tile tracker, a table of actions sorted by time:

========== ==========================================================================
ACTIONTYPE meaning
========== ==========================================================================
fa         run fiber assignment for this tile against the current alternative ledgers
update     fold the real observations of this tile into the alternative ledgers
reproc     a later update of a tile that was reprocessed by the spectroscopic pipeline
veto       a bright1b veto, applied to no particular tile
lya1b      the 2025-07-21 Lyman-alpha quasar numobs increase
addnew     a dated addition of new healpix ledgers
========== ==========================================================================

Only ``fa``, ``update`` and ``reproc`` are carried out by :mod:`mockfactory.desi.altmtl.loop`; the others
are still listed, so that an action list which needs them fails loudly rather than silently
diverging from the real survey.
"""

from pathlib import Path
import logging
from datetime import datetime, timedelta

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.tiletracker')


#: Action types the loop knows how to carry out.
SUPPORTED_ACTIONTYPES = ('fa', 'update', 'reproc')

#: Timestamp at which the real survey raised the number of observations requested for
#: Lyman-alpha quasars. See https://github.com/desihub/desitarget/pull/845.
LYA1B_TIMESTAMP = '2025-07-21T23:36:04+00:00'


def _read_mtl_time(tileid_and_dir):
    """Return the mtl time recorded in the real fiberassign header of one tile."""
    import fitsio
    tileid, fiberassign_dir = tileid_and_dir
    return str(fitsio.read_header(utils.get_fiberassign_fn(tileid, fiberassign_dir=fiberassign_dir))['MTLTIME'])


def _read_mtl_times(tileids, fiberassign_dir=None, numproc=1):
    """Return the mtl time of each tile in ``tileids``, reading headers in parallel."""
    args = [(tileid, fiberassign_dir) for tileid in tileids]
    if numproc > 1:
        from multiprocessing import Pool
        with Pool(processes=numproc) as pool:
            return pool.map(_read_mtl_time, args)
    return [_read_mtl_time(arg) for arg in args]


def get_tile_tracker_fn(altmtl_dir, survey='main', obscon='dark'):
    """Return the path of the tile tracker of one alternative realization."""
    return Path(altmtl_dir) / '{}survey-{}obscon-TileTracker.ecsv'.format(survey, obscon.upper())


def read_tile_tracker(altmtl_dir, survey='main', obscon='dark'):
    """Read the tile tracker of one alternative realization, as an :class:`astropy.table.Table`."""
    from astropy.table import Table
    return Table.read(get_tile_tracker_fn(altmtl_dir, survey=survey, obscon=obscon), format='ascii.ecsv')


def get_actions(tile_tracker, done=False):
    """
    Return the actions of ``tile_tracker``, in the order they must be carried out.

    Parameters
    ----------
    tile_tracker : astropy.table.Table
        Tile tracker, as read by :func:`read_tile_tracker`.

    done : bool, default=False
        If ``False``, return only the actions that have not been carried out yet.
        If ``None``, return all actions.

    Returns
    -------
    actions : astropy.table.Table
        Actions, sorted by time.
    """
    actions = tile_tracker
    if done is not None:
        actions = actions[actions['DONEFLAG'] == done]
    actions = actions.copy()
    actions.sort(['ACTIONTIME', 'ACTIONTYPE', 'TILEID'])
    return actions


def mark_actions_done(altmtl_dir, actions, survey='main', obscon='dark'):
    """
    Flag ``actions`` as carried out in the tile tracker of ``altmtl_dir``.

    An action is identified by the triplet (tile, type, time): a tile that was reprocessed
    carries several actions, which differ only by their time.
    """
    fn = get_tile_tracker_fn(altmtl_dir, survey=survey, obscon=obscon)
    from astropy.table import Table
    tile_tracker = Table.read(fn, format='ascii.ecsv')
    for action in actions:
        mask = ((tile_tracker['TILEID'] == action['TILEID'])
                & (tile_tracker['ACTIONTYPE'] == action['ACTIONTYPE'])
                & (tile_tracker['ACTIONTIME'] == action['ACTIONTIME']))
        if not mask.any():
            raise ValueError('action {} on tile {} at {} not found in {}'.format(
                action['ACTIONTYPE'], action['TILEID'], action['ACTIONTIME'], fn))
        tile_tracker['DONEFLAG'][mask] = True
    tile_tracker.write(fn, format='ascii.ecsv', overwrite=True)
    return fn


def make_tile_tracker(altmtl_dir, survey='main', obscon='dark', start_date=None, end_date=None,
                      tiles_specstatus_fn=None, mtl_done_tiles_fn=None, mtl_done_vetoes_fn=None,
                      fiberassign_dir=None, ledgers_yaml_dir=None, lya1b=True, numproc=1,
                      overwrite=False, meta=None):
    """
    Build the action list replaying the real survey, and write it to ``altmtl_dir``.

    Parameters
    ----------
    altmtl_dir : str
        Directory of one alternative realization, e.g. ``.../altmtl0/Univ000``.

    survey : str, default='main'
        Survey to replay. Only 'main' is supported.

    obscon : str, default='dark'
        Observing conditions, 'dark' or 'bright'.

    start_date : int, str, default=None
        Night (yyyymmdd, or an ISO date) before which actions are flagged as already done.
        If ``None``, no action is flagged as done and the whole survey is replayed.

    end_date : int, str, default=None
        Night (yyyymmdd, or an ISO date) after which actions are dropped. Required: replaying
        up to a fixed date is what makes a mock correspond to a given data release.

    tiles_specstatus_fn : str, default=None
        Path of the tiles-specstatus file, listing which tiles were observed in which program.
        Defaults to the surveyops copy.

    mtl_done_tiles_fn : str, default=None
        Path of the mtl-done-tiles file, giving the times at which the real ledgers were
        updated. Defaults to the surveyops copy.

    mtl_done_vetoes_fn : str, default=None
        Path of the mtl-done-vetoes file. Defaults to the surveyops copy.

    fiberassign_dir : str, default=None
        Directory of the real survey per-tile fiberassign files, whose headers give the time
        at which each tile was assigned. Defaults to the trunk copy.

    ledgers_yaml_dir : str, default=None
        Directory holding ``{OBSCON}-ledgers.yaml``, listing the dates at which new healpix
        ledgers were added. If ``None``, no ledger addition action is emitted.

    lya1b : bool, default=True
        Whether to emit the Lyman-alpha quasar numobs increase action, when the action list
        extends past the date at which the real survey applied it.

    numproc : int, default=1
        Number of processes to read the fiberassign headers with. One gzipped header is read
        per observed tile, so this is what the build time is spent on.

    overwrite : bool, default=False
        Whether to overwrite an existing tile tracker.

    meta : dict, default=None
        Extra metadata to store in the tile tracker header.

    Returns
    -------
    fn : str
        Path of the tile tracker that was written.
    """
    from astropy.table import Table
    from desitarget.mtl import add_to_iso_date

    if survey.lower() != 'main':
        raise ValueError('only the main survey is supported, got {}'.format(survey))
    if end_date is None:
        raise ValueError('end_date is required: it is what ties the mock to a data release')

    fn = get_tile_tracker_fn(altmtl_dir, survey=survey, obscon=obscon)
    if Path(fn).is_file() and not overwrite:
        # Building the action list reads one header per observed tile, so it is worth skipping.
        logger.info('Tile tracker {} already exists, not rebuilding.'.format(fn))
        return fn

    if tiles_specstatus_fn is None: tiles_specstatus_fn = utils.TILES_SPECSTATUS_FN
    if mtl_done_tiles_fn is None: mtl_done_tiles_fn = utils.MTL_DONE_TILES_FN
    if mtl_done_vetoes_fn is None: mtl_done_vetoes_fn = utils.MTL_DONE_VETOES_FN

    start_night = utils.iso_to_night(start_date) if start_date is not None else 19990101
    end_night = utils.iso_to_night(end_date)
    start_dt = datetime.strptime(str(start_night), '%Y%m%d')
    end_dt = datetime.strptime(str(end_night), '%Y%m%d')
    start_str, end_str = start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')

    specstatus = Table.read(tiles_specstatus_fn)
    mask = (specstatus['SURVEY'] == survey.lower()) & (specstatus['FAPRGRM'] == obscon.lower())
    tileids = np.unique(specstatus['TILEID'][mask])
    logger.info('{:d} {} tiles observed in the {} survey.'.format(tileids.size, obscon.lower(), survey))

    done_tiles = Table.read(mtl_done_tiles_fn)
    done_tiles.sort(['TILEID', 'TIMESTAMP'])
    done_tileid = np.asarray(done_tiles['TILEID'])
    # A tile that was observed but never folded into the real ledgers carries no action.
    start = np.searchsorted(done_tileid, tileids, side='left')
    stop = np.searchsorted(done_tileid, tileids, side='right')
    mask = stop > start
    tileids, start, stop = tileids[mask], start[mask], stop[mask]
    logger.info('{:d} of them were folded into the real ledgers.'.format(tileids.size))

    # One header per tile, and they are gzipped: reading them serially is minutes of wall time.
    mtltimes = _read_mtl_times(tileids, fiberassign_dir=fiberassign_dir, numproc=numproc)

    tileid, actiontype, actiontime, doneflag, archivedate = [], [], [], [], []

    for tid, istart, istop, mtltime in zip(tileids, start, stop, mtltimes):
        # The fiber assignment of a tile happens at the mtl time recorded in its fiberassign
        # header; one second is added so that it sorts after the update that preceded it.
        fa_time = add_to_iso_date(mtltime, 1)
        fa_night = utils.iso_to_night(fa_time)
        if fa_night > end_night:
            continue

        updates = [update for update in done_tiles[istart:istop]
                   if utils.iso_to_night(update['TIMESTAMP']) <= end_night]
        if not updates:
            # The tile was assigned within the range, but only observed after it. Assigning it
            # would leave the ledgers untouched, and whether it shows up here at all would
            # depend on how much the survey operations files have grown since.
            continue

        tileid.append(tid)
        actiontype.append('fa')
        actiontime.append(fa_time)
        doneflag.append(fa_night < start_night)
        archivedate.append(fa_night)

        for iupdate, update in enumerate(updates):
            update_night = utils.iso_to_night(update['TIMESTAMP'])
            tileid.append(tid)
            # Only the first update folds in a fresh observation; any later one reprocesses it.
            actiontype.append('reproc' if iupdate else 'update')
            actiontime.append(update['TIMESTAMP'])
            doneflag.append(update_night < start_night)
            archivedate.append(update['ARCHIVEDATE'])

    # Vetoes are not attached to a tile, hence the -1 tile id.
    vetoes = Table.read(mtl_done_vetoes_fn)
    mask = (vetoes['TIMESTAMP'] >= start_str) & (vetoes['TIMESTAMP'] < end_str) & (vetoes['PROGRAM'] == obscon.upper())
    for veto in vetoes[mask]:
        # Two seconds, so that a veto sorts after the done-tiles entry it accompanies.
        tileid.append(-1)
        actiontype.append('veto')
        actiontime.append(datetime.fromisoformat(veto['TIMESTAMP']) + timedelta(seconds=2))
        actiontime[-1] = actiontime[-1].isoformat()
        doneflag.append(False)
        archivedate.append(-1)

    if lya1b and obscon.lower() == 'dark' and actiontime and max(actiontime) > LYA1B_TIMESTAMP:
        tileid.append(-1)
        actiontype.append('lya1b')
        actiontime.append(LYA1B_TIMESTAMP)
        doneflag.append(False)
        archivedate.append(utils.iso_to_night(LYA1B_TIMESTAMP))

    if ledgers_yaml_dir is not None:
        import yaml
        with open(Path(ledgers_yaml_dir) / '{}-ledgers.yaml'.format(obscon.upper())) as file:
            # The first entry is the initial set of ledgers; later ones are dated additions.
            for date in list(yaml.safe_load(file).keys())[1:]:
                if start_str < date <= end_str:
                    tileid.append(-1)
                    actiontype.append('addnew')
                    actiontime.append('{}T00:00:00+00:00'.format(date))
                    doneflag.append(False)
                    archivedate.append(int(str(date).replace('-', '')))

    if meta is None: meta = {}
    # str(), not the Path: the tracker is an ecsv, whose header goes through yaml, and yaml
    # has no representer for a PosixPath.
    meta = dict({'Name': 'AltMTLTileTracker', 'StartDate': start_night, 'EndDate': end_night,
                 'amtldir': str(altmtl_dir)}, **meta)
    tile_tracker = Table([tileid, actiontype, actiontime, doneflag, archivedate],
                         names=('TILEID', 'ACTIONTYPE', 'ACTIONTIME', 'DONEFLAG', 'ARCHIVEDATE'),
                         dtype=('<i8', '<U6', '<U25', 'bool', '<i8'), meta=meta)
    tile_tracker.sort(['ACTIONTIME', 'ACTIONTYPE', 'TILEID'])

    counts = {name: int((tile_tracker['ACTIONTYPE'] == name).sum()) for name in np.unique(tile_tracker['ACTIONTYPE'])}
    logger.info('Built {:d} actions: {}.'.format(len(tile_tracker), counts))
    unsupported = {name: count for name, count in counts.items() if name not in SUPPORTED_ACTIONTYPES}
    if unsupported:
        logger.warning('Action list contains types the loop cannot carry out: {}.'.format(unsupported))

    utils.mkdir(altmtl_dir)
    tile_tracker.write(fn, format='ascii.ecsv', overwrite=True)
    logger.info('Wrote tile tracker {}.'.format(fn))
    return fn

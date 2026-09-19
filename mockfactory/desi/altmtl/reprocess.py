"""
Reprocessing, against the state held in memory.

When the spectroscopic pipeline redoes the reduction of a tile, the redshifts already folded
in change. A merged target list cannot be patched at its endpoint, because its state is
path-dependent: the number of observations counts them, and priority and target state depend
on the order the observations arrived in. Every observation of the targets involved has to be
replayed from their unobserved state, with the corrected one substituted.

This is a port of :func:`desitarget.mtl.reprocess_ledger`, which does the same replay against
ledgers on disk. The decisions are unchanged and still come from
:func:`desitarget.mtl.make_mtl`; what goes is the round trip through ecsv, which dominated the
cost: reading the healpixels a tile touches, writing them back, and the per-row Python loops
in between.

Two departures, both deliberate. The tile and target hash is vectorised rather than built as a
list of formatted strings. And positions are left alone: desitarget copies whole rows of the
redshift catalog back into the ledger, which for an alternative universe would write the real
survey's position onto the mock target that replaced it.
"""

import logging

import numpy as np


logger = logging.getLogger('altmtl.reprocess')


#: Bits that mark an observation the merged target list must not count.
BAD_ZWARN = 'BAD_SPECQA|BAD_PETALQA|NODATA'


def last_of_each(*keys):
    """
    Return the indices of the last row of each distinct combination of ``keys``.

    They come back in their original order, which is what taking the last occurrence of each
    combination and then restoring the ordering amounts to.
    """
    size = len(keys[0])
    if size == 0:
        return np.zeros(0, dtype='i8')
    # Sort by the keys, then by position, so the last row of a group sorts last within it.
    order = np.lexsort((np.arange(size),) + tuple(reversed(keys)))
    is_last = np.empty(size, dtype='?')
    is_last[-1] = True
    if size > 1:
        changed = np.zeros(size - 1, dtype='?')
        for key in keys:
            sorted_key = np.asarray(key)[order]
            changed |= sorted_key[1:] != sorted_key[:-1]
        is_last[:-1] = changed
    return np.sort(order[is_last])


def reprocess_state(state, zcat, obscon='dark', survey='main'):
    """
    Replay the observations of the targets in ``zcat``, and fold the result into ``state``.

    Parameters
    ----------
    state : LedgerState
        State to reprocess within. It is changed in place.

    zcat : astropy.table.Table
        Redshift catalog of the reprocessed tile, with target identifiers already relabelled
        to this realization.

    obscon : str, default='dark'
        Observing conditions.

    survey : str, default='main'
        Survey being replayed.

    Returns
    -------
    timestamps : dict
        Timestamp at which each reprocessed tile was refolded, as
        :func:`desitarget.mtl.reprocess_ledger` returns.
    """
    from astropy.table import Table
    from desitarget.mtl import make_mtl, get_utc_date, add_to_iso_date
    from desitarget.targetmask import zwarn_mask
    from desitarget.geomask import match, match_to

    zcat = Table(zcat)
    reproctiles = set(np.unique(np.asarray(zcat['ZTILEID'])).tolist())

    # Every row the state holds for the healpixels this tile touches, oldest first: the replay
    # starts from the unobserved state, so the history is what it works on.
    healpixels = state.healpixels_of(zcat['RA'], zcat['DEC'])
    rows = state.rows_in_healpixels(healpixels)
    rows = rows[np.argsort(rows['TIMESTAMP'], kind='stable')]

    # Only the targets the catalog refers to matter.
    rows = rows[np.isin(rows['TARGETID'], np.asarray(zcat['TARGETID']))]
    if not len(rows):
        logger.warning('no target of this redshift catalog is held by the state.')
        return {}

    # The first row of a target is its unobserved state; the rows carrying a tile are its
    # observations.
    _, first = np.unique(rows['TARGETID'], return_index=True)
    unobs = rows[np.sort(first)]
    observed = rows[rows['ZTILEID'] != -1]
    missing = np.setdiff1d(observed['TARGETID'], unobs['TARGETID'])
    if missing.size:
        raise ValueError('{:d} target(s) have no unobserved state to replay from, e.g. {}'.format(
            missing.size, missing[:5]))
    if np.unique(unobs['TARGETID']).size != len(unobs):
        raise ValueError('some targets hold more than one unobserved state')

    # Tiles in the order they first observed any of these targets, which is the order the
    # replay has to follow.
    _, first = np.unique(observed['ZTILEID'], return_index=True)
    ordered_tiles = observed['ZTILEID'][np.sort(first)]

    # Every observation, old and reprocessed, as one redshift catalog. The new rows come last,
    # so that they win where a tile had been observed before.
    from_rows = np.zeros(len(observed), dtype=zcat.dtype)
    for name in zcat.dtype.names:
        from_rows[name] = observed[name]
    allzcat = np.concatenate([from_rows, np.asarray(zcat)])
    allzcat = Table(allzcat[last_of_each(allzcat['ZTILEID'], allzcat['TARGETID'])])
    logger.debug('reprocessing {:d} targets over {:d} tiles, {:d} observations.'.format(
        len(unobs), len(ordered_tiles), len(allzcat)))

    now = get_utc_date(survey=survey)
    timestamps = {tileid: add_to_iso_date(now, i) for i, tileid in enumerate(ordered_tiles)}

    mtl = Table(unobs)
    done, timedict = [], {}
    bad = zwarn_mask.mask(BAD_ZWARN)
    for tileid in ordered_tiles:
        timestamp = timestamps[tileid]
        mini = allzcat[allzcat['ZTILEID'] == tileid]
        if np.unique(np.asarray(mini['TARGETID'])).size != len(mini):
            raise ValueError('duplicate targets on tile {}'.format(tileid))

        # The number of observations comes from the running state, not from the catalog.
        mii, zii = match(mtl['TARGETID'], mini['TARGETID'])
        mini['NUMOBS'][zii] = mtl['NUMOBS'][mii] + 1
        mini = mini[zii]

        zmtl = make_mtl(mtl, obscon.upper(), zcat=mini, trimtozcat=True, trimcols=True)
        mii, zii = match(mtl['TARGETID'], zmtl['TARGETID'])
        for name in mtl.dtype.names:
            if name in ('RA', 'DEC'):
                continue
            mtl[name][mii] = zmtl[name][zii]
        mtl['TIMESTAMP'][mii] = timestamp

        # trimtozcat drops observations that were no good, which still have to be recorded.
        tidmiss = list(set(np.asarray(mini['TARGETID']).tolist())
                       - set(np.asarray(zmtl['TARGETID']).tolist()))
        badmiss = mini[match_to(mini['TARGETID'], tidmiss)] if tidmiss else mini[:0]
        if len(badmiss) and np.any(np.asarray(badmiss['ZWARN']) & bad == 0):
            raise ValueError('tile {}: make_mtl skipped observations that are not bad'.format(tileid))
        if len(badmiss):
            mii, zii = match(mtl['TARGETID'], badmiss['TARGETID'])
            # Never let a bad observation change how many observations a target has had.
            for name in set(badmiss.dtype.names) - {'NUMOBS', 'NUMOBS_MORE', 'RA', 'DEC'}:
                mtl[name][mii] = badmiss[name][zii]
            mtl['TIMESTAMP'][mii] = timestamp

        done.append(np.asarray(mtl[mtl['ZTILEID'] == tileid]))
        if tileid in reproctiles:
            timedict[tileid] = timestamp

    state.absorb_rows(np.concatenate(done))
    return timedict

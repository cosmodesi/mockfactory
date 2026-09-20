"""
Bitweights from a set of alternative realizations.

Each realization says, for every target, whether it was observed. Stacking that over
realizations gives a bit string per target, and its mean is the probability of observation.
The inverse of that probability is the weight that corrects a clustering measurement for
fiber assignment incompleteness.

The realizations must share their targets and differ only in their subpriorities, which is
what :func:`mockfactory.desi.altmtl.ledger.initialize_realization` arranges.
"""

import os
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.bitweights')


def pack_bitweights(observed):
    """
    Pack a boolean observation array into 64-bit integers.

    Parameters
    ----------
    observed : array
        Boolean array of shape ``(ntargets, nrealizations)``, true where the target was
        observed in that realization.

    Returns
    -------
    bitweights : array
        Integer array of shape ``(ntargets, (nrealizations + 63) // 64)``.
    """
    observed = np.asarray(observed)
    if observed.ndim != 2:
        raise ValueError('expected an array of shape (ntargets, nrealizations), got {}'.format(observed.shape))
    ntargets, nrealizations = observed.shape
    nbits = 64
    nout = (nrealizations + nbits - 1) // nbits

    bits = np.zeros((ntargets, 8), dtype='i')
    bitweights = np.zeros(ntargets, dtype='i8')
    output = np.zeros((ntargets, nout), dtype='i8')
    iout = 0
    for irealization in range(nrealizations):
        bits[observed[:, irealization], irealization % 8] = 1
        packed = np.array(np.packbits(bits[:, ::-1]), dtype='i8')
        bitweights |= np.left_shift(packed, 8 * ((irealization % nbits) // 8))
        if (irealization + 1) % nbits == 0 or irealization + 1 == nrealizations:
            output[:, iout] = bitweights
            bitweights[:] = 0
            iout += 1
        if (irealization + 1) % 8 == 0:
            bits[:] = 0
    return output


def unpack_bitweights(bitweights, nrealizations=None):
    """Undo :func:`pack_bitweights`, returning the boolean observation array."""
    bitweights = np.atleast_2d(bitweights)
    nbits = 64
    if nrealizations is None: nrealizations = nbits * bitweights.shape[1]
    observed = np.zeros((bitweights.shape[0], nrealizations), dtype='?')
    for irealization in range(nrealizations):
        word, bit = divmod(irealization, nbits)
        observed[:, irealization] = (bitweights[:, word] >> bit) & 1 > 0
    return observed


def _read_observed(altmtl_dir, healpixels, survey='main', obscon='dark', good_tilelocid=None):
    """
    Return the targets of one realization and whether each was observed.

    Parameters
    ----------
    good_tilelocid : array, default=None
        Tile-locations whose spectra are usable, as ``10000 * tileid + location``. Without it
        every observation counts, including those on fibers the real survey threw away, and the
        probability of observation comes out too high.
    """
    from desitarget import io
    from .compat import supported
    from .ledger import MTL_NSIDE, get_ledger_dir

    ledger = io.read_mtl_in_hp(get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon), MTL_NSIDE,
                               healpixels, unique=True, isodate=None, returnfn=False, initial=False,
                               leq=False, **supported(io.read_mtl_in_hp, tabform='ascii.ecsv'))
    # Not np.sort(..., order=...): the ledgers can come back as a masked record array, which
    # does not take the copy order that call needs.
    ledger = ledger[np.argsort(ledger['TARGETID'])]
    observed = ledger['NUMOBS'] > 0.5
    if good_tilelocid is not None:
        observed &= np.isin(_get_tilelocid(altmtl_dir, ledger, survey=survey), good_tilelocid)
    return ledger['TARGETID'], observed


def _get_tilelocid(altmtl_dir, ledger, survey='main'):
    """Return, for each ledger entry, the tile-location it was observed at, or -1."""
    import glob
    import fitsio
    from .assignment import get_fa_dir

    tilelocid = np.full(len(ledger), -1, dtype='i8')
    ztileid = ledger['ZTILEID']
    for tileid in np.unique(ztileid[ztileid != -1]):
        fns = glob.glob(os.path.join(get_fa_dir(altmtl_dir, '*', survey=survey),
                                     'fba-{}.fits'.format(utils.tile_string(tileid))))
        if not fns:
            raise ValueError('no alternative assignment found for tile {:d} under {}'.format(
                tileid, altmtl_dir))
        assignment = fitsio.read(fns[0], ext='FASSIGN', columns=['TARGETID', 'LOCATION'])
        mask = ztileid == tileid
        index = _match_to(assignment['TARGETID'], ledger['TARGETID'][mask])
        found = index >= 0
        selection = np.flatnonzero(mask)[found]
        tilelocid[selection] = 10000 * tileid + assignment['LOCATION'][index[found]]
    return tilelocid


def _match_to(source, targetid):
    """Return, for each of ``targetid``, its index in ``source``, or -1."""
    argsort = np.argsort(source)
    index = np.searchsorted(source, targetid, sorter=argsort)
    index = argsort[np.clip(index, 0, source.size - 1)]
    index[source[index] != targetid] = -1
    return index


def compute_bitweights(base_dir, realizations, healpixels, survey='main', obscon='dark',
                       good_tilelocid=None):
    """
    Stack the observations of several realizations into bitweights.

    Parameters
    ----------
    base_dir : str
        Directory holding the realizations. May hold a ``{:d}``-style field, else realizations
        are looked up as ``Univ000``, ``Univ001``, and so on.

    realizations : int, list
        Number of realizations, or the list of realization indices.

    healpixels : int, list
        Healpixels to process.

    survey : str, default='main'
        Survey that was replayed.

    obscon : str, default='dark'
        Observing conditions.

    good_tilelocid : array, default=None
        Tile-locations whose spectra are usable.

    Returns
    -------
    targetid : array
        Target identifiers.

    bitweights : array
        Packed observation bits, of shape ``(ntargets, (nrealizations + 63) // 64)``.

    prob_obs : array
        Fraction of realizations in which each target was observed.
    """
    if np.ndim(realizations) == 0: realizations = list(range(realizations))
    if np.ndim(healpixels) == 0: healpixels = [healpixels]

    targetid, observed = None, []
    for realization in realizations:
        if '{' in base_dir:
            altmtl_dir = base_dir.format(realization)
        else:
            altmtl_dir = utils.get_universe_dir(base_dir, realization=realization)
        tids, obs = _read_observed(altmtl_dir, healpixels, survey=survey, obscon=obscon,
                                   good_tilelocid=good_tilelocid)
        if targetid is None:
            targetid = tids
        elif not np.array_equal(targetid, tids):
            raise ValueError('realization {:d} holds different targets than the first one; '
                             'bitweights only mean something for realizations of the same '
                             'catalog'.format(realization))
        observed.append(obs)

    observed = np.column_stack(observed)
    prob_obs = observed.sum(axis=1) / observed.shape[1]
    logger.info('{:d} targets over {:d} realizations: mean probability of observation {:.4f}, '
                '{:d} never observed.'.format(targetid.size, observed.shape[1], prob_obs.mean(),
                                              int((prob_obs == 0.).sum())))
    return targetid, pack_bitweights(observed), prob_obs


def write_bitweights(base_dir, realizations, healpixels, output_dir, survey='main', obscon='dark',
                     good_tilelocid=None, overwrite=False):
    """
    Compute the bitweights of each healpixel and write them out, one file per healpixel.

    Takes the same arguments as :func:`compute_bitweights`, plus the directory to write to.

    Returns
    -------
    fns : list
        Paths of the files that were written.
    """
    from astropy.table import Table

    if np.ndim(healpixels) == 0: healpixels = [healpixels]
    output_dir = os.path.join(output_dir, survey.lower(), obscon.lower())
    utils.mkdir(output_dir)

    fns = []
    for healpix in healpixels:
        fn = os.path.join(output_dir, '{}bw-{}-hp-{:d}.fits'.format(survey.lower(), obscon.lower(), healpix))
        if os.path.isfile(fn) and not overwrite:
            logger.info('{} already exists, not recomputing it.'.format(fn))
            fns.append(fn)
            continue
        targetid, bitweights, prob_obs = compute_bitweights(
            base_dir, realizations, [healpix], survey=survey, obscon=obscon,
            good_tilelocid=good_tilelocid)
        Table({'TARGETID': targetid, 'BITWEIGHTS': bitweights, 'PROB_OBS': prob_obs}).write(fn, overwrite=True)
        fns.append(fn)
    logger.info('Wrote {:d} bitweight file(s) to {}.'.format(len(fns), output_dir))
    return fns

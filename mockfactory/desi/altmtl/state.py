"""
The merged target list held in memory.

The healpix ledgers are the real survey's persistence format: append-only ecsv, one row per
change of state, written so that a survey running for years stays auditable and restartable.
A mock needs none of that. It needs, per tile, the current state of the targets in it; per
update, the state of the targets that were observed; and, at the end, how many times each
target was observed.

Holding that in memory removes the ledgers from the inner loop. Measured on a mock of 11.6
million targets, the ledgers are 2 GB of ecsv, reading one tile out of them costs 1.5 s, and
building them costs ten minutes per mock and a 2 GB copy per realization.

The full history is kept, not just the latest state of each target, because reprocessing
replays every observation of a target from its unobserved state.
"""

import os
import logging

import numpy as np

from . import utils


logger = logging.getLogger('altmtl.state')


class LedgerState(object):
    """
    Merged target list state of one realization, held in memory.

    Attributes
    ----------
    current : array
        Latest state of every target, sorted by ``TARGETID``.

    history : list
        Blocks of superseded rows, oldest first. Together with :attr:`current` these are the
        rows an append-only ledger would hold.
    """
    def __init__(self, current, nside=None):
        from .ledger import MTL_NSIDE

        self.nside = MTL_NSIDE if nside is None else nside
        current = np.asarray(current)
        # Sorted by target, so that a row can be found by binary search.
        self.current = current[np.argsort(current['TARGETID'])]
        self.history = []
        self._pixel = self._healpix(self.current['RA'], self.current['DEC'])

    def _healpix(self, ra, dec):
        import healpy as hp
        return hp.ang2pix(self.nside, np.radians(90. - np.asarray(dec)), np.radians(np.asarray(ra)),
                          nest=True)

    def __len__(self):
        return len(self.current)

    @property
    def healpixels(self):
        """Healpixels the targets fall in."""
        return np.unique(self._pixel)

    @classmethod
    def from_targets(cls, targets_fn, obscon='dark', survey='main', nside=None):
        """
        Build the initial state from a mock target catalog.

        This is the in-memory counterpart of :func:`desitarget.mtl.make_ledger`: the same
        initial priorities and numbers of observations, without writing a ledger.

        Parameters
        ----------
        targets_fn : str
            Path of the target catalog, as written by
            :func:`mockfactory.desi.altmtl.targets.write_targets`.

        obscon : str, default='dark'
            Observing conditions.

        survey : str, default='main'
            Survey the state belongs to.

        nside : int, default=None
            Healpix nside the state is indexed at. Defaults to the one the real ledgers use.

        Returns
        -------
        state : LedgerState
        """
        import fitsio
        from desitarget.mtl import make_mtl

        targets = fitsio.read(targets_fn)
        # make_mtl with no redshift catalog returns the unobserved state, which is what a
        # freshly built ledger holds.
        current = np.asarray(make_mtl(targets, obscon.upper(), trimcols=True))
        logger.info('Built state of {:d} targets from {}.'.format(len(current), targets_fn))
        return cls(current, nside=nside)

    @classmethod
    def from_ledgers(cls, altmtl_dir, survey='main', obscon='dark', nside=None):
        """Build the state by reading existing healpix ledgers, to pick a run up again."""
        from desitarget import io
        from .ledger import get_ledger_dir, get_healpixels, MTL_NSIDE

        if nside is None: nside = MTL_NSIDE
        ledger_dir = get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon)
        healpixels = get_healpixels(ledger_dir, obscon=obscon)
        current = io.read_mtl_in_hp(ledger_dir, nside, [int(h) for h in healpixels], unique=True,
                                    tabform='ascii.ecsv')
        logger.info('Read state of {:d} targets from {}.'.format(len(current), ledger_dir))
        return cls(np.asarray(current), nside=nside)

    def targets_in_tiles(self, tiles):
        """
        Return the current state of the targets covered by ``tiles``.

        Stands in for :func:`desitarget.io.read_targets_in_tiles`: the healpix index narrows
        the catalog down, and the tile geometry then makes the cut exact.
        """
        import desimodel.footprint

        pixels = desimodel.footprint.tiles2pix(self.nside, tiles=tiles)
        targets = self.current[np.isin(self._pixel, pixels)]
        mask = desimodel.footprint.is_point_in_desi(tiles, targets['RA'], targets['DEC'])
        return targets[mask]

    def update(self, zcat, obscon='dark', numobs_from_ledger=True, ext=False):
        """
        Fold a redshift catalog into the state.

        This is :func:`desitarget.mtl.update_ledger` without the ledger: the targets the
        catalog refers to are handed to :func:`desitarget.mtl.make_mtl`, which applies the
        survey's rules, and the rows it returns replace their previous state.

        Parameters
        ----------
        zcat : astropy.table.Table
            Redshift catalog, with target identifiers already relabelled to this realization.

        obscon : str, default='dark'
            Observing conditions.

        numobs_from_ledger : bool, default=True
            Whether to take the number of observations so far from the state rather than from
            the redshift catalog.

        ext : bool, default=False
            Whether to apply the 1b rules.

        Returns
        -------
        nupdated : int
            Number of targets whose state changed.
        """
        from astropy.table import Table
        from desitarget.mtl import make_mtl

        index = self._index_of(zcat['TARGETID'])
        found = index >= 0
        if not found.all():
            # A redshift for a target this realization does not hold: it sat on a fiber that
            # the alternative assignment gave to sky or to nothing.
            zcat = zcat[found]
            index = index[found]
        if not len(zcat):
            return 0

        targets = Table(self.current[index])
        if numobs_from_ledger:
            zcat = Table(zcat)
            zcat['NUMOBS'] = targets['NUMOBS'] + 1

        # trimtozcat keeps only the targets the catalog updated, and drops bad observations.
        updated = np.asarray(make_mtl(targets, obscon.upper(), zcat=zcat, trimtozcat=True,
                                      trimcols=True, ext=ext))
        if not len(updated):
            return 0

        index = self._index_of(updated['TARGETID'])
        if (index < 0).any():
            raise ValueError('make_mtl returned {:d} target(s) absent from the '
                             'state'.format(int((index < 0).sum())))
        # Keep what is being replaced: reprocessing replays a target's observations from its
        # unobserved state, so the superseded rows have to survive.
        self.history.append(self.current[index].copy())
        for name in updated.dtype.names:
            self.current[name][index] = updated[name]
        return len(updated)

    def _index_of(self, targetid):
        """Return the row of each of ``targetid`` in :attr:`current`, or -1."""
        targetid = np.asarray(targetid)
        ids = self.current['TARGETID']
        index = np.searchsorted(ids, targetid)
        index = np.clip(index, 0, ids.size - 1)
        index[ids[index] != targetid] = -1
        return index

    def numobs(self, targetid=None):
        """Return how many times each target was observed, for ``targetid`` or for all."""
        if targetid is None:
            return self.current['TARGETID'], self.current['NUMOBS']
        index = self._index_of(targetid)
        numobs = np.zeros(len(index), dtype=self.current['NUMOBS'].dtype)
        found = index >= 0
        numobs[found] = self.current['NUMOBS'][index[found]]
        return targetid, numobs

    def all_rows(self):
        """Every row this state holds, superseded ones first, as an append-only ledger would."""
        if not self.history:
            return self.current
        return np.concatenate(self.history + [self.current])

    def _ledger_meta(self, healpix, survey='main', obscon='dark'):
        """The header a ledger carries, which desitarget reads to make sense of the directory."""
        return {'DR': 0, 'EXTNAME': 'MTL', 'FILEHPX': int(healpix), 'FILENEST': True,
                'FILENSID': int(self.nside), 'INDIR': 'mockfactory.desi.altmtl',
                'OBSCON': obscon.upper(), 'OVERRIDE': False, 'SCND': False, 'SURVEY': survey}

    def write_ledgers(self, altmtl_dir, survey='main', obscon='dark', healpixels=None,
                      overwrite=True):
        """
        Write the state out as healpix ledgers, in the format the real survey uses.

        Anything downstream that expects ledgers reads these; the loop itself never does.

        Parameters
        ----------
        altmtl_dir : str
            Directory of the realization to write under.

        survey : str, default='main'
            Survey the ledgers belong to.

        obscon : str, default='dark'
            Observing conditions.

        healpixels : array, default=None
            Write only these healpixels. Defaults to all of them.

        overwrite : bool, default=True
            Whether to replace ledgers that already exist.

        Returns
        -------
        ledger_dir : str
            Directory the ledgers were written to.
        """
        from astropy.table import Table
        from .ledger import get_ledger_dir

        ledger_dir = get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon)
        utils.mkdir(ledger_dir)
        rows = self.all_rows()
        pixel = self._healpix(rows['RA'], rows['DEC'])
        if healpixels is None:
            healpixels = np.unique(pixel)
        nwritten = 0
        for healpix in np.atleast_1d(healpixels):
            fn = os.path.join(ledger_dir, 'mtl-{}-hp-{:d}.ecsv'.format(obscon.lower(), int(healpix)))
            if os.path.isfile(fn) and not overwrite:
                continue
            block = rows[pixel == healpix]
            # An append-only ledger is ordered by the time each row was written.
            block = block[np.argsort(block['TIMESTAMP'], kind='stable')]
            table = Table(block)
            table.meta.update(self._ledger_meta(healpix, survey=survey, obscon=obscon))
            table.write(fn, format='ascii.ecsv', overwrite=True)
            nwritten += 1
        logger.info('Wrote {:d} healpix ledgers to {}.'.format(nwritten, ledger_dir))
        return ledger_dir

    def absorb_ledgers(self, ledger_dir, healpixels, survey='main', obscon='dark'):
        """
        Replace everything this state holds for ``healpixels`` with what those ledgers hold.

        Used after handing a few healpixels to desitarget for reprocessing: the ledgers are
        then the authority on those targets, both their latest state and their history.

        Returns
        -------
        nrows : int
            Number of rows read back.
        """
        from desitarget import io

        healpixels = [int(healpix) for healpix in np.atleast_1d(healpixels)]
        rows = np.asarray(io.read_mtl_in_hp(ledger_dir, self.nside, healpixels, unique=False,
                                            tabform='ascii.ecsv'))
        # The last row of a target is its state now; the others are what it passed through.
        order = np.lexsort((np.arange(len(rows)), rows['TARGETID']))
        sorted_targetid = rows['TARGETID'][order]
        is_last = np.empty(len(order), dtype='?')
        is_last[-1:] = True
        is_last[:-1] = sorted_targetid[1:] != sorted_targetid[:-1]
        latest, superseded = rows[order[is_last]], rows[order[~is_last]]

        # Drop what this state held for those healpixels, then put the new rows in its place.
        self.history = [block[~np.isin(self._healpix(block['RA'], block['DEC']), healpixels)]
                        for block in self.history]
        self.history = [block for block in self.history if len(block)]
        if len(superseded): self.history.append(superseded)

        keep = ~np.isin(self.current['TARGETID'], latest['TARGETID'])
        current = np.concatenate([self.current[keep], latest.astype(self.current.dtype)])
        self.current = current[np.argsort(current['TARGETID'])]
        self._pixel = self._healpix(self.current['RA'], self.current['DEC'])
        return len(rows)

    def healpixels_of(self, ra, dec):
        """Return the distinct healpixels the given positions fall in."""
        return np.unique(self._healpix(ra, dec))

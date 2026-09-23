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

from .targets import read_targets

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
        # Healpix of every row, kept alongside it: reprocessing asks for a few pixels at a
        # time, and recomputing it over the whole catalog each time is O(catalog) per action.
        self._pixel = self._healpix(self.current['RA'], self.current['DEC'])
        self._history_pixel = []
        # Rows ordered by healpix, built on demand: a tile asks for a handful of pixels, and
        # scanning the whole catalog for them costs the same as the assignment itself.
        self._pixel_index = None

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

    def _set_pixel(self, index, pixel):
        """
        Record the healpix of the rows at ``index``, keeping the index if none of them moved.

        Rebuilding the index sorts the whole catalog, seconds at tens of millions of targets,
        and reprocessing writes a few thousand rows at a time. Those rows keep the positions
        they had, since the replay preserves them, so almost always nothing moves and the
        index still stands.
        """
        pixel = np.asarray(pixel)
        if self._pixel_index is not None and not np.array_equal(self._pixel[index], pixel):
            self._pixel_index = None
        self._pixel[index] = pixel

    def _rows_of_healpixels(self, healpixels):
        """
        Return the rows of :attr:`current` that fall in ``healpixels``.

        The rows are indexed by healpix once and then looked up by binary search, rather than
        the whole catalog being scanned for every tile. With tens of millions of targets and a
        worker per core, that scan is memory traffic the assignment has to queue behind.
        """
        if self._pixel_index is None:
            order = np.argsort(self._pixel, kind='stable')
            self._pixel_index = (order, self._pixel[order])
        order, sorted_pixel = self._pixel_index
        healpixels = np.unique(np.asarray(healpixels))
        start = np.searchsorted(sorted_pixel, healpixels, side='left')
        stop = np.searchsorted(sorted_pixel, healpixels, side='right')
        blocks = [order[i:j] for i, j in zip(start, stop) if j > i]
        if not blocks:
            return np.zeros(0, dtype='i8')
        return np.concatenate(blocks) if len(blocks) > 1 else blocks[0]

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

        targets = read_targets(targets_fn)
        # make_mtl with no redshift catalog returns the unobserved state, which is what a
        # freshly built ledger holds.
        current = np.asarray(make_mtl(targets, obscon.upper(), trimcols=True))
        logger.info('Built state of {:d} targets from {}.'.format(len(current), targets_fn))
        return cls(current, nside=nside)

    @classmethod
    def from_ledgers(cls, altmtl_dir, survey='main', obscon='dark', nside=None):
        """Build the state by reading existing healpix ledgers, to pick a run up again."""
        from desitarget import io
        from .compat import supported
        from .ledger import get_ledger_dir, get_healpixels, MTL_NSIDE

        if nside is None: nside = MTL_NSIDE
        ledger_dir = get_ledger_dir(altmtl_dir, survey=survey, obscon=obscon)
        healpixels = get_healpixels(ledger_dir, obscon=obscon)
        current = io.read_mtl_in_hp(ledger_dir, nside, [int(h) for h in healpixels], unique=True,
                                    **supported(io.read_mtl_in_hp, tabform='ascii.ecsv'))
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
        targets = self.current[self._rows_of_healpixels(pixels)]
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
        from .compat import supported

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
                                      trimcols=True, **supported(make_mtl, ext=ext)))
        if not len(updated):
            return 0

        index = self._index_of(updated['TARGETID'])
        if (index < 0).any():
            raise ValueError('make_mtl returned {:d} target(s) absent from the '
                             'state'.format(int((index < 0).sum())))
        # Keep what is being replaced: reprocessing replays a target's observations from its
        # unobserved state, so the superseded rows have to survive.
        superseded = self.current[index].copy()
        self.history.append(superseded)
        self._history_pixel.append(self._pixel[index].copy())
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

    @property
    def nrows(self):
        """Number of rows held, counting superseded ones."""
        return len(self.current) + sum(len(block) for block in self.history)

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
        if healpixels is None:
            healpixels = self.healpixels
        healpixels = np.atleast_1d(healpixels)
        # Cut the catalog down to the requested pixels in one pass, then group within the
        # remainder: reprocessing asks for a handful of pixels out of thousands.
        rows, pixel = [], []
        for block, block_pixel in zip(self.history, self._history_pixel):
            mask = np.isin(block_pixel, healpixels)
            if mask.any():
                rows.append(block[mask])
                pixel.append(block_pixel[mask])
        mask = np.isin(self._pixel, healpixels)
        rows.append(self.current[mask])
        pixel.append(self._pixel[mask])
        rows = np.concatenate(rows) if len(rows) > 1 else rows[0]
        pixel = np.concatenate(pixel) if len(pixel) > 1 else pixel[0]

        nwritten = 0
        for healpix in healpixels:
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
        from .compat import supported

        healpixels = [int(healpix) for healpix in np.atleast_1d(healpixels)]
        rows = np.asarray(io.read_mtl_in_hp(ledger_dir, self.nside, healpixels, unique=False,
                                            **supported(io.read_mtl_in_hp, tabform='ascii.ecsv')))
        # The last row of a target is its state now; the others are what it passed through.
        order = np.lexsort((np.arange(len(rows)), rows['TARGETID']))
        sorted_targetid = rows['TARGETID'][order]
        is_last = np.empty(len(order), dtype='?')
        is_last[-1:] = True
        is_last[:-1] = sorted_targetid[1:] != sorted_targetid[:-1]
        latest, superseded = rows[order[is_last]], rows[order[~is_last]]

        # Drop what this state held for those healpixels, then put the new rows in its place.
        keep = [~np.isin(pixel, healpixels) for pixel in self._history_pixel]
        self.history = [block[mask] for block, mask in zip(self.history, keep) if mask.any()]
        self._history_pixel = [pixel[mask] for pixel, mask in zip(self._history_pixel, keep)
                               if mask.any()]
        if len(superseded):
            self.history.append(superseded)
            self._history_pixel.append(self._healpix(superseded['RA'], superseded['DEC']))

        # Write the new states over the rows they replace, rather than rebuilding the whole
        # catalog: reprocessing touches a few thousand targets out of tens of millions, and a
        # concatenate plus argsort of everything costs far more than the update itself.
        latest = latest.astype(self.current.dtype)
        index = self._index_of(latest['TARGETID'])
        known = index >= 0
        for name in latest.dtype.names:
            self.current[name][index[known]] = latest[name][known]
        self._pixel[index[known]] = self._healpix(latest['RA'][known], latest['DEC'][known])
        self._pixel_index = None
        if not known.all():
            # A target the state did not hold. It should not happen, since reprocessing only
            # revisits targets that were already there, so say so rather than absorb it.
            raise ValueError('reprocessing returned {:d} target(s) absent from the '
                             'state'.format(int((~known).sum())))
        return len(rows)

    def build_index(self):
        """
        Build the healpix index, if it is not already there.

        It is built on demand, and the demand comes from
        :meth:`targets_in_tiles`, which is called in the workers that assign tiles and nowhere
        else: an update is keyed on the target identifier and never needs it. So a forked
        worker finds it missing, argsorts the whole catalog to build it, uses it for its own
        tiles and takes it with it when it exits, and the next worker does the same. At tens of
        millions of targets that is five seconds a worker, against sixty milliseconds once it
        exists, and a pool of them argsorting at once is worse than the sum of its parts.

        Calling this in the parent before forking means the index is built once and inherited.
        """
        if self._pixel_index is None:
            self.rows_in_healpixels(self._pixel[:1])
        return self

    def healpixels_of(self, ra, dec):
        """Return the distinct healpixels the given positions fall in."""
        return np.unique(self._healpix(ra, dec))

    def rows_in_healpixels(self, healpixels):
        """
        Return every row this state holds for ``healpixels``, superseded ones included.

        This is what reading those healpix ledgers without taking the latest row of each
        target would give, and it is what a replay from the unobserved state needs.
        """
        healpixels = np.atleast_1d(healpixels)
        rows = []
        for block, pixel in zip(self.history, self._history_pixel):
            mask = np.isin(pixel, healpixels)
            if mask.any(): rows.append(block[mask])
        rows.append(self.current[self._rows_of_healpixels(healpixels)])
        return np.concatenate(rows) if len(rows) > 1 else rows[0]

    def absorb_rows(self, rows):
        """
        Fold rows produced by a replay back in, newest last.

        The last row of a target is its state now; the others are what it passed through, and
        so is whatever the state held for it before, since the replay supersedes all of it.

        Returns
        -------
        nupdated : int
            Number of targets whose state changed.
        """
        from .reprocess import last_of_each

        if not len(rows):
            return 0
        rows = np.asarray(rows).astype(self.current.dtype)
        is_last = np.zeros(len(rows), dtype='?')
        is_last[last_of_each(rows['TARGETID'])] = True
        latest, superseded = rows[is_last], rows[~is_last]

        index = self._index_of(latest['TARGETID'])
        if (index < 0).any():
            raise ValueError('the replay returned {:d} target(s) absent from the state'.format(
                int((index < 0).sum())))
        # What the state held for these targets is now superseded too.
        blocks, pixels = [self.current[index]], [self._pixel[index]]
        if len(superseded):
            blocks.append(superseded)
            pixels.append(self._healpix(superseded['RA'], superseded['DEC']))
        self.history.append(np.concatenate(blocks) if len(blocks) > 1 else blocks[0])
        self._history_pixel.append(np.concatenate(pixels) if len(pixels) > 1 else pixels[0])

        for name in latest.dtype.names:
            self.current[name][index] = latest[name]
        self._set_pixel(index, self._healpix(latest['RA'], latest['DEC']))
        return len(latest)

"""
Convert the legacy survey brick pixel maps this package reads into HDF5, once.

Sampling a pixel-level quantity at catalog positions costs one decompression of a whole
3600 x 3600 brick per brick touched, and the legacy survey ships those as tile compressed
HCOMPRESS. That codec is slow: measured back to back on one brick, a full read runs at
150-160 MB/s of pixels, against 2100-3100 MB/s for the same array in HDF5 with zstd at level 1,
which is also the smaller file (0.07 MB against 0.43 for maskbits). There is no trade to make,
so a pass over the bricks is decompression bound at about a fifteenth of the speed it could be.

One pass over DR9 converts them, and every later query reads the fast copy:

    347 206 bricks, 93 548 north and 253 658 south, four quantities each, about 260 GB

The layout groups bricks by the three character directory the legacy survey already shards on,
so the result is ~640 files of a few hundred megabytes rather than 1.4 million small ones,
which is what a parallel file system wants:

    <output_dir>/<region>/<prefix>.h5
        <brickname>/maskbits          int16, chunked, zstd
        <brickname>/nexp-g            int16
        <brickname>/nexp-r
        <brickname>/nexp-z
        <brickname>.attrs             the world coordinate system cards, so a reader can turn
                                      right ascension and declination into pixels without the
                                      original file

A quantity whose source file does not exist is **left out** rather than written as zeros: an
absent nexp band means the brick has no coverage in it, which a caller must be able to tell
apart from a measured zero.

Chunks of 600 are the default: against chunks of 100 they read a whole brick 4.5 times faster
and a scattered handful of positions only 1.2 times slower, at half the size.

    salloc -N 1 -C cpu -q interactive -t 02:00:00 -A desi
    srun -n 128 python -m mockfactory.desi.scripts.convert_bricks_to_h5 \
        --output-dir $SCRATCH/legacysurvey/dr9-hdf5

Ranks take whole prefixes, so no two write the same file. Re-running skips prefixes already
finished, which is what makes a long pass restartable.
"""

import argparse
import logging
import os
import time

import numpy as np

logger = logging.getLogger('convert_bricks')

DR9 = '/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9'
#: Source brick file, by region, brick name and quantity.
BRICK_FN = os.path.join(DR9, '{region}', 'coadd', '{prefix}', '{brickname}',
                        'legacysurvey-{brickname}-{quantity}.fits.fz')
#: List of the bricks that carry data, per region.
BRICKS_FN = os.path.join(DR9, '{region}', 'survey-bricks-dr9-{region}.fits.gz')
QUANTITIES = ('maskbits', 'nexp-g', 'nexp-r', 'nexp-z')
#: Cards a reader needs to turn sky coordinates into pixels.
WCS_KEYS = ('NAXIS1', 'NAXIS2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2',
            'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2')
#: The projection, which every legacy survey brick shares, so it is stored once per shard
#: rather than on each brick. Verified against the source headers, and a world coordinate
#: system rebuilt from these cards reproduces the original pixel mapping exactly.
PROJECTION = {'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN'}


def get_bricknames(region):
    """Brick names of ``region`` that carry data, sorted, grouped by their shard prefix."""
    import fitsio

    bricknames = fitsio.read(BRICKS_FN.format(region=region), columns=['brickname'])['brickname']
    bricknames = np.sort(np.asarray(bricknames).astype('U8'))
    prefixes = np.array([name[:3] for name in bricknames])
    # Sorting the names sorts their prefixes too, so the shards are contiguous runs
    edges = np.flatnonzero(prefixes[:-1] != prefixes[1:]) + 1
    return {group[0][:3]: group for group in np.split(bricknames, edges)}


def convert_prefix(region, prefix, bricknames, output_fn, quantities, chunk, clevel):
    """Write one shard: every brick of ``prefix``, every quantity that exists."""
    import fitsio
    import h5py
    import hdf5plugin

    compression = hdf5plugin.Zstd(clevel=clevel)
    nwritten, nmissing = 0, 0
    tmp_fn = output_fn + '.tmp'
    with h5py.File(tmp_fn, 'w') as h5:
        h5.attrs.update(PROJECTION)
        for brickname in bricknames:
            group = h5.create_group(str(brickname))
            for quantity in quantities:
                fn = BRICK_FN.format(region=region, prefix=prefix, brickname=brickname,
                                     quantity=quantity)
                if not os.path.isfile(fn):
                    # An absent band is absent coverage, and must stay distinguishable from zero
                    nmissing += 1
                    continue
                with fitsio.FITS(fn) as f:
                    data = f[1].read()
                    header = f[1].read_header()
                shape = tuple(min(c, s) for c, s in zip((chunk, chunk), data.shape))
                group.create_dataset(quantity, data=data, chunks=shape, **compression)
                if not group.attrs:
                    for key in WCS_KEYS:
                        if key in header:
                            group.attrs[key] = header[key]
                nwritten += 1
    os.replace(tmp_fn, output_fn)
    return nwritten, nmissing


def read_brick_quantity(output_dir, region, brickname, quantity):
    """
    Read one converted brick map back, or ``None`` where the source had no coverage.

    Returned with the world coordinate system cards, which is what a caller needs to turn sky
    coordinates into pixel indices.
    """
    import h5py
    import hdf5plugin  # noqa: F401  registers the codec

    fn = os.path.join(output_dir, region, brickname[:3] + '.h5')
    with h5py.File(fn, 'r') as h5:
        group = h5[brickname]
        wcs = dict(h5.attrs)
        wcs.update(group.attrs)
        if quantity not in group:
            return None, wcs
        return group[quantity][:], wcs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True, help='where the converted bricks go')
    parser.add_argument('--regions', nargs='*', default=['north', 'south'])
    parser.add_argument('--quantities', nargs='*', default=list(QUANTITIES))
    parser.add_argument('--chunk', type=int, default=600, help='chunk side, in pixels')
    parser.add_argument('--clevel', type=int, default=1, help='zstd level')
    parser.add_argument('--max-prefixes', type=int, default=None,
                        help='stop after this many shards per region, for a trial run')
    parser.add_argument('--overwrite', action='store_true',
                        help='redo shards that are already there')
    args = parser.parse_args()

    from mpi4py import MPI
    from mockfactory import setup_logging

    setup_logging()
    mpicomm = MPI.COMM_WORLD
    for region in args.regions:
        groups = get_bricknames(region) if mpicomm.rank == 0 else None
        groups = mpicomm.bcast(groups, root=0)
        prefixes = sorted(groups)
        if args.max_prefixes is not None:
            prefixes = prefixes[:args.max_prefixes]
        if mpicomm.rank == 0:
            logger.info('{}: {:d} bricks over {:d} shards.'
                        .format(region, sum(len(groups[p]) for p in prefixes), len(prefixes)))
            os.makedirs(os.path.join(args.output_dir, region), exist_ok=True)
        mpicomm.Barrier()

        start = time.time()
        for index in range(mpicomm.rank, len(prefixes), mpicomm.size):
            prefix = prefixes[index]
            output_fn = os.path.join(args.output_dir, region, prefix + '.h5')
            if os.path.isfile(output_fn) and not args.overwrite:
                continue
            nwritten, nmissing = convert_prefix(region, prefix, groups[prefix], output_fn,
                                                args.quantities, args.chunk, args.clevel)
            logger.info('{} {}: {:d} bricks, {:d} maps, {:d} absent, {:.1f} MB, {:.0f} s elapsed.'
                        .format(region, prefix, len(groups[prefix]), nwritten, nmissing,
                                os.path.getsize(output_fn) / 1e6, time.time() - start))
        mpicomm.Barrier()
        if mpicomm.rank == 0:
            logger.info('{} done in {:.0f} s.'.format(region, time.time() - start))


if __name__ == '__main__':
    main()

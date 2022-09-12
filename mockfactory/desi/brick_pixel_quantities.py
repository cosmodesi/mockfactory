"""
Script to read brick pixel-level quantities, such as maskbits.

This example can be run with `srun -n 16 python brick_pixel_quantities.py`,
but one will typically import:

```
from mockfactory.desi import get_brick_pixel_quantities
```

For an example, see desi/from_box_to_desi_cutsky script.
"""

import os
import logging

import fitsio
import numpy as np
from mpi4py import MPI


logger = logging.getLogger('Bricks')


def get_brick_pixel_quantities(ra, dec, columns, mpicomm=MPI.COMM_WORLD):
    """
    Return value of brick pixel-level qunantity at input RA/Dec, expected to be scattered on all MPI processes.
    Based on Rongpu Zhou's code:
    https://github.com/rongpu/desi-examples/blob/master/bright_star_mask/read_pixel_maskbit.py

    Parameters
    ----------
    ra : array
        Right ascension (degree).

    dec : array
        Declination (degree).

    columns : dict
        Dictionary of column name: {'fn': path to brick file, 'dtype': dtype (optional), 'default': default value in case brick does not exist (optional)}.
        Path to brick file can contain keywords 'region' and 'brickname'.
        One can also provide callable: kwargs; callable will be called on each brick as callable(ra, dec, brickname, region ('north' or 'south'), **kwargs), and must return a dictionary of arrays (brick quantities); for example desitarget.randoms.quantities_at_positions_in_a_brick: {'drdir': '/global/project/projectdirs/cosmo/data/legacysurvey/dr9/{region}'}.
        To register brickname, brickid, or photsys (N or S), just pass e.g. 'brickname': None.

    mpicomm : MPI communicator, default=None
        The current MPI communicator.

    Returns
    -------
    di : dict
        Dictionary with requested columns.
    """
    if not columns:
        return {}

    ra, dec = np.asarray(ra), np.asarray(dec)
    size, shape = ra.size, ra.shape
    if not mpicomm.allreduce(size):
        return {}
    ra, dec = np.ravel(ra), np.ravel(dec)

    import mpytools as mpy
    import mpsort

    def _dict_to_array(data):
        """
        Return dict as numpy array.

        Parameters
        ----------
        data : dict
            Data dictionary of name: array.

        Returns
        -------
        array : array
        """
        array = [(name, data[name]) for name in data]
        array = np.empty(array[0][1].shape[0], dtype=[(name, col.dtype, col.shape[1:]) for name, col in array])
        for name in data: array[name] = data[name]
        return array

    def _one_brick(fn, ra, dec, dtype=None, default=None):
        """Extract quantity associated to a (RA, Dec) position from a legacy imaging brick."""
        if os.path.isfile(fn):
            # Read data and header
            img, header = fitsio.read(fn, header=True)

            # Convert ra, dec coordinates to brick coordinates
            coadd_x, coadd_y = wcs.WCS(header).wcs_world2pix(ra, dec, 0)
            coadd_x, coadd_y = np.round(coadd_x).astype(int), np.round(coadd_y).astype(int)

            # Extract mask information
            return np.asarray(img[coadd_y, coadd_x], dtype=dtype)

        # Sometimes we can have objects outside DR9 footprint
        return np.full(ra.size, default, dtype=dtype)

    # Load bricks class
    from desiutil import brick
    from astropy import wcs
    bricks = brick.Bricks()

    # Create unique identification as index column
    cumsize = np.cumsum([0] + mpicomm.allgather(ra.size))[mpicomm.rank]
    index = cumsize + np.arange(ra.size)
    brickid_data = _dict_to_array({'ra': ra, 'dec': dec, 'brickname': bricks.brickname(ra, dec), 'brickid': bricks.brickid(ra, dec), 'index': index})

    # Let's group particles by brickid, with ~ similar number of brickids on each rank
    # Caution: this may produce memory unbalance between different processes
    # hence potential memory error, which may be avoided using some criterion to rebalance load at the cost of less efficiency
    unique_brickid, counts_brickid = np.unique(brickid_data['brickid'], return_counts=True)

    # Proceed rank-by-rank to save memory
    for irank in range(1, mpicomm.size):
        unique_brickid_irank = mpy.sendrecv(unique_brickid, source=irank, dest=0, tag=0, mpicomm=mpicomm)
        counts_brickid_irank = mpy.sendrecv(counts_brickid, source=irank, dest=0, tag=0, mpicomm=mpicomm)
        if mpicomm.rank == 0:
            unique_brickid, counts_brickid = np.concatenate([unique_brickid, unique_brickid_irank]), np.concatenate([counts_brickid, counts_brickid_irank])
            unique_brickid, inverse = np.unique(unique_brickid, return_inverse=True)
            counts_brickid = np.bincount(inverse, weights=counts_brickid).astype(int)

    # Compute the number particles that each rank must contain after sorting
    nbr_particles = None
    if mpicomm.rank == 0:
        nbr_bricks = [(irank * unique_brickid.size // mpicomm.size, (irank + 1) * unique_brickid.size // mpicomm.size) for irank in range(mpicomm.size)]
        nbr_particles = [np.sum(counts_brickid[nbr_brick_low:nbr_brick_high], dtype='i8') for nbr_brick_low, nbr_brick_high in nbr_bricks]
        nbr_bricks = np.diff(nbr_bricks, axis=-1)
        logger.info(f'Number of bricks to read per rank = {np.min(nbr_bricks)} - {np.max(nbr_bricks)} (min - max).')

    # Send the number particles that each rank must contain after sorting
    nbr_particles = mpicomm.scatter(nbr_particles, root=0)
    assert mpicomm.allreduce(nbr_particles) == mpicomm.allreduce(brickid_data.size), 'float in bincount messes up total particle counts'

    # Sort data to have same number of bricks in each rank
    brickid_data_tmp = np.empty_like(brickid_data, shape=nbr_particles)
    mpsort.sort(brickid_data, orderby='brickid', out=brickid_data_tmp)
    data_tmp = {}

    for brickname in np.unique(brickid_data_tmp['brickname']):
        mask_brick = brickid_data_tmp['brickname'] == brickname
        ra_tmp, dec_tmp = brickid_data_tmp['ra'][mask_brick], brickid_data_tmp['dec'][mask_brick]
        region = 'north' if bricks.brick_radec(ra_tmp[0], dec_tmp[0])[1] > 32.375 else 'south'
        tmp = {}
        for name, attrs in columns.items():
            if isinstance(name, str):
                if attrs is None:
                    if name.lower() == 'photsys':
                        tmp[name] = np.full(ra_tmp.size, region[0].upper())
                    elif name.lower() in brickid_data_tmp.dtype.names:
                        tmp[name] = brickid_data_tmp[name.lower()][mask_brick]
                    else:
                        raise ValueError('Unknown column {}'.format(name))
                else:
                    attrs = dict(attrs)
                    fn = attrs.pop('fn', None)
                    tmp[name] = _one_brick(fn.format(region=region, brickname=brickname), ra_tmp, dec_tmp, **attrs)
            else:
                attrs = dict(attrs)
                for key, value in attrs.items():
                    if isinstance(value, str): attrs[key] = value.format(region=region)
                tmp.update(name(ra_tmp, dec_tmp, brickname, **attrs))
        for name, value in tmp.items():
            if name not in data_tmp:
                data_tmp[name] = np.empty_like(value, shape=brickid_data_tmp.size)
            data_tmp[name][mask_brick] = value

    index_name = '_'.join(data_tmp.keys()) + '_index'  # just to make sure this is distinct from all other column names
    data_tmp[index_name] = brickid_data_tmp['index']
    data_tmp = _dict_to_array(data_tmp)

    dtypes = mpicomm.allgather((data_tmp.dtype, index_name) if data_tmp.size else (None, None))
    for dtype, index_name in dtypes:
        if dtype is not None: break
    if dtype is None:
        return {}
    if not data_tmp.size:
        data_tmp = np.empty(0, dtype=dtype)

    # Collect the data in the intial order
    data = np.empty(shape=size, dtype=dtype)
    mpsort.sort(data_tmp, orderby=index_name, out=data)

    # Check if we find the correct initial order
    assert np.all(data[index_name] == index)

    return {name: data[name].reshape(shape) for name in data.dtype.names if name != index_name}


if __name__ == '__main__':

    from mockfactory import RandomCutskyCatalog, setup_logging

    setup_logging()

    mpicomm = MPI.COMM_WORLD

    if mpicomm.rank == 0: logger.info('Run simple example to illustrate how to get pixel-level quantities.')

    # Generate example cutsky catalog, scattered on all processes
    cutsky = RandomCutskyCatalog(rarange=(25., 30.), decrange=(1., 2.), size=10000, seed=44, mpicomm=mpicomm)
    ra, dec = cutsky['RA'], cutsky['DEC']
    # to test when empty catalog is given with MPI
    if mpicomm.rank == 1: ra, dec = [], []

    start = MPI.Wtime()
    # to collect only maskbits, uncomment columns['maskbits'] and comment the rest
    columns = {}
    # default = outside brick primary
    # columns['maskbits'] = {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}
    # columns['elg_mask'] = {'fn': '/global/cfs/cdirs/desi/survey/catalogs/brickmasks/ELG/v1/{region}/coadd/{brickname:.3s}/{brickname}/{brickname}-elgmask.fits.gz', 'dtype': 'i2', 'default': 0}
    columns['lrg_mask'] = {'fn': '/global/cfs/cdirs/desi/survey/catalogs/brickmasks/LRG/v1.1/{region}/coadd/{brickname:.3s}/{brickname}/{brickname}-lrgmask.fits.gz', 'dtype': 'i2', 'default': 0}
    from desitarget import randoms
    columns[randoms.quantities_at_positions_in_a_brick] = {'drdir': '/global/project/projectdirs/cosmo/data/legacysurvey/dr9/{region}/', 'aprad': 1e-9}  # skip apflux
    columns['brickname'] = None
    columns['photsys'] = None
    catalog = get_brick_pixel_quantities(ra, dec, columns, mpicomm=mpicomm)
    if mpicomm.rank == 0:
        logger.info('Output columns are {}.'.format(list(catalog.keys())))
        logger.info('Pixel-level quantities read in {:2.2f} s.'.format(MPI.Wtime() - start))

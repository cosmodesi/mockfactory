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
    import mpytools as mpy
    from desiutil import brick
    from astropy import wcs

    if not columns:
        return {}

    ra, dec = np.asarray(ra), np.asarray(dec)
    size, shape = ra.size, ra.shape
    if not mpicomm.allreduce(size):
        return {}
    ra, dec = np.ravel(ra), np.ravel(dec)

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
    bricks = brick.Bricks()

    # Create mpytools.Catalog to sort it across processes in the brick id
    brickid_data = mpy.Catalog({'ra': ra, 'dec': dec, 'brickname': bricks.brickname(ra, dec), 'brickid': bricks.brickid(ra, dec)})
    # Create unique identification as index column
    brickid_data['index'] = brickid_data.cindex()
    # Copy unique identification to perform sanity check at the end
    index = brickid_data['index']

    # Sort data to have same number of bricks in each rank
    brickid_data = brickid_data.csort('brickid', size='orderby_counts')
    nbr_bricks = mpy.gather(np.unique(brickid_data['brickid']).size, mpiroot=0)
    if mpicomm.rank == 0: logger.info(f'Number of bricks to read per rank = {np.min(nbr_bricks)} - {np.max(nbr_bricks)} (min - max).')

    # Collect the brick pixel quantities in each brickname
    data = {}
    for brickname in np.unique(brickid_data['brickname']):
        mask_brick = brickid_data['brickname'] == brickname
        ra_tmp, dec_tmp = brickid_data['ra'][mask_brick], brickid_data['dec'][mask_brick]
        region = 'north' if bricks.brick_radec(ra_tmp[0], dec_tmp[0])[1] > 32.375 else 'south'
        tmp = {}
        for name, attrs in columns.items():
            if isinstance(name, str):
                if attrs is None:
                    if name.lower() == 'photsys':
                        tmp[name] = np.full(ra_tmp.size, region[0].upper())
                    elif name.lower() in brickid_data.columns():
                        tmp[name] = brickid_data[name.lower()][mask_brick]
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
            if name not in data:
                data[name] = np.empty_like(value, shape=brickid_data.size)
            data[name][mask_brick] = value

    index_name = '_'.join(data.keys()) + '_index'  # just to make sure this is distinct from all other column names
    data[index_name] = brickid_data['index']
    # Convert dict to mpytools.Catalog in order to sort it across processes to recover the initial order
    data = mpy.Catalog(data)

    dtypes = mpicomm.allgather((data.to_array().dtype, data.columns(), index_name) if data.size else (None, None, None))
    for dtype, names, index_name in dtypes:
        if dtype is not None: break
    if dtype is None:
        return {}
    if not data.size:
        data = mpy.Catalog({name: np.empty(0, dtype=dtype[name]) for name in names})

    # Collect the data in the intial order
    data = data.csort(index_name, size=index.size)
    # Check if we find the correct initial order
    assert np.all(data[index_name] == index)

    return {name: data[name].reshape(shape) for name in data.columns() if name != index_name}


if __name__ == '__main__':

    from mockfactory import RandomCutskyCatalog, setup_logging

    setup_logging()

    mpicomm = MPI.COMM_WORLD

    if mpicomm.rank == 0: logger.info('Run simple example to illustrate how to get pixel-level quantities.')

    # Generate example cutsky catalog, scattered on all processes
    #cutsky = RandomCutskyCatalog(rarange=(28., 30.), decrange=(1., 2.), csize=10000, seed=44, mpicomm=mpicomm)
    cutsky = RandomCutskyCatalog(rarange=(28., 30.), decrange=(-10, 35), csize=100000, seed=44, mpicomm=mpicomm)
    ra, dec = cutsky['RA'], cutsky['DEC']
    # to test when empty catalog is given with MPI
    #if mpicomm.rank == 1: ra, dec = [], []

    start = MPI.Wtime()
    # to collect only maskbits, uncomment columns['maskbits'] and comment the rest
    columns = {}
    # default = outside brick primary
    columns['maskbits'] = {'fn': '/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}
    #columns['nobs_r'] = {'fn': '/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-nexp-g.fits.fz', 'dtype': 'i2', 'default': 1}
    
    
    # columns['elg_mask'] = {'fn': '/dvs_ro/cfs/cdirs/desi/survey/catalogs/brickmasks/ELG/v1/{region}/coadd/{brickname:.3s}/{brickname}/{brickname}-elgmask.fits.gz', 'dtype': 'i2', 'default': 0}
    #columns['lrg_mask'] = {'fn': '/dvs_ro/cfs/cdirs/desi/survey/catalogs/brickmasks/LRG/v1.1/{region}/coadd/{brickname:.3s}/{brickname}/{brickname}-lrgmask.fits.gz', 'dtype': 'i2', 'default': 0}
    
    #from desitarget import randoms
    #columns[randoms.quantities_at_positions_in_a_brick] = {'drdir': '/global/project/projectdirs/cosmo/data/legacysurvey/dr9/{region}/', 'aprad': 1e-9}  # skip apflux
    
    columns['brickname'] = None
    columns['photsys'] = None
    
    catalog = get_brick_pixel_quantities(ra, dec, columns, mpicomm=mpicomm)
    if mpicomm.rank == 0:
        logger.info('Output columns are {}.'.format(list(catalog.keys())))
        logger.info('Pixel-level quantities read in {:2.2f} s.'.format(MPI.Wtime() - start))
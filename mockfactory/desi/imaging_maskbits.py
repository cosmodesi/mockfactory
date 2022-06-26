"""
Script to apply imaging maskbits.
This example can be run with srun -n 16 python imaging_maskbits.py.
"""

import os
import logging

import fitsio
import numpy as np
from astropy import wcs
from mpi4py import MPI

from desiutil import brick


logger = logging.getLogger('maskbits')



def get_maskbits(ra, dec, maskbits_fn='/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', dtype='i2', mpicomm=MPI.COMM_WORLD):
    """
    Return value of bit mask at input RA/Dec, expected to be scattered on all MPI processes.
    Based on Rongpu Zhou's code:
    https://github.com/rongpu/desi-examples/blob/master/bright_star_mask/read_pixel_maskbit.py

    Parameters
    ----------
    ra : array
        Right ascension.

    dec : array
        Declination.

    maskbits_fn : str
        Path to maskbits files, with keywords 'region' and 'brickname'.

    dtype : np.dtype, str, default='i2'
        Output type.

    mpicomm : MPI communicator, default=None
        The current MPI communicator.
    """
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

    def get_brick_maskbits(maskbits_fn, ra, dec):
        """Extract maskbit associated to a (RA, Dec) position from a legacy imaging brick."""
        if os.path.isfile(maskbits_fn):
            # Read data and header
            maskbits_img, header = fitsio.read(maskbits_fn, header=True)

            # Convert ra, dec coordinates to brick coordinates
            coadd_x, coadd_y = wcs.WCS(header).wcs_world2pix(ra, dec, 0)
            coadd_x, coadd_y = np.round(coadd_x).astype(int), np.round(coadd_y).astype(int)

            # Extract mask information
            maskbits = maskbits_img[coadd_y, coadd_x]

        else:
            #raise ValueError
            # Sometimes we can have objects outside DR9 footprint:
            # remove these objects setting maskbits 0 (NPRIMARY pixel)
            maskbits = 2**0 * np.ones(ra.size, dtype=dtype)

        return maskbits

    # If not empty, collect the information from brick
    ra, dec = np.asarray(ra), np.asarray(dec)
    maskbits = np.ones_like(ra, dtype='i2')
    if ra.size != 0:
        # Collect brickid
        bricks = brick.Bricks()
        cumsize = np.cumsum([0] + mpicomm.allgather(ra.size))[mpicomm.rank]
        index = cumsize + np.arange(ra.size)
        data = _dict_to_array({'ra': ra, 'dec': dec, 'brickname': bricks.brickname(ra, dec), 'brickid': bricks.brickid(ra, dec), 'maskbits': maskbits, 'index': index})
        #data = _dict_to_array({'ra': ra, 'dec': dec, 'brickname': bricks.brickname(ra, dec), 'brickid': index[::-1], 'maskbits': maskbits, 'index': index})
        import mpsort
        mpsort.sort(data, orderby='brickid')

        for brickname in np.unique(data['brickname']):
            mask_brick = data['brickname'] == brickname
            region = 'north' if bricks.brick_radec(data['ra'][mask_brick][0], data['dec'][mask_brick][0])[1] > 32.375 else 'south'
            data['maskbits'][mask_brick] = get_brick_maskbits(maskbits_fn.format(region=region, brickname=brickname), data['ra'][mask_brick], data['dec'][mask_brick])

        mpsort.sort(data, orderby='index')
        assert np.all(data['index'] == index)
        maskbits = data['maskbits']

    return maskbits


if __name__ == '__main__':

    from mockfactory import RandomCutskyCatalog, setup_logging

    mpicomm = MPI.COMM_WORLD

    setup_logging()

    if mpicomm.rank == 0:
        logger.info('Run simple example to illustrate how to apply DR9 maskbits.')

    # Generate example cutsky catalog, scattered on all processes
    ra, dec = 0., 0.
    cutsky = RandomCutskyCatalog(rarange=(20., 30.), decrange=(-0.5, 2.), size=10000, seed=44, mpicomm=mpicomm)
    start = MPI.Wtime()
    maskbits = get_maskbits(cutsky['RA'], cutsky['DEC'], mpicomm=mpicomm)
    if mpicomm.rank == 0:
        logger.info(f'Apply DR9 maskbits done in {MPI.Wtime() - start:2.2f} s.')

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
            # Sometimes we can have objects outside DR9 footprint:
            # remove these objects setting maskbits 0 (NPRIMARY pixel)
            maskbits = 2**0 * np.ones(ra.size, dtype=dtype)

        return maskbits

    ra, dec = np.asarray(ra), np.asarray(dec)

    # Load bricks class
    bricks = brick.Bricks()

    # Create unique identification as index column
    cumsize = np.cumsum([0] + mpicomm.allgather(ra.size))[mpicomm.rank]
    index = cumsize + np.arange(ra.size)
    data = _dict_to_array({'ra': ra, 'dec': dec, 'brickname': bricks.brickname(ra, dec), 'brickid': bricks.brickid(ra, dec), 'maskbits': np.ones_like(ra, dtype='i2'), 'index': index})

    # Let's group particles by brickid, with ~ similar number of brickids on each rank
    # Caution: this may produce memory unbalance between different processes
    # hence potential memory error, which may be avoided using some criterion to rebalance load at the cost of less efficiency
    unique_brickid, counts_brickid = np.unique(data['brickid'], return_counts=True)

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
    assert mpicomm.allreduce(nbr_particles) == mpicomm.allreduce(data.size), 'float in bincount messes up total particle counts'

    # Sort data to have same number of bricks in each rank
    data_tmp = np.empty_like(data, shape=nbr_particles)
    mpsort.sort(data, orderby='brickid', out=data_tmp)

    for brickname in np.unique(data_tmp['brickname']):
        mask_brick = data_tmp['brickname'] == brickname
        region = 'north' if bricks.brick_radec(data_tmp['ra'][mask_brick][0], data_tmp['dec'][mask_brick][0])[1] > 32.375 else 'south'
        data_tmp['maskbits'][mask_brick] = get_brick_maskbits(maskbits_fn.format(region=region, brickname=brickname), data_tmp['ra'][mask_brick], data_tmp['dec'][mask_brick])

    # Collect the data in the intial order
    mpsort.sort(data_tmp, orderby='index', out=data)

    # Check if we find the correct inital order
    assert np.all(data['index'] == index)

    return data['maskbits']


if __name__ == '__main__':

    from mockfactory import RandomCutskyCatalog, setup_logging

    setup_logging()

    mpicomm = MPI.COMM_WORLD

    if mpicomm.rank == 0: logger.info('Run simple example to illustrate how to apply DR9 maskbits.')

    # Generate example cutsky catalog, scattered on all processes
    cutsky = RandomCutskyCatalog(rarange=(20., 30.), decrange=(-0.5, 2.), size=10000, seed=44, mpicomm=mpicomm)
    ra, dec = cutsky['RA'], cutsky['DEC']
    # to test when empty catalog is given with MPI
    if mpicomm.rank == 1: ra, dec = [], []

    start = MPI.Wtime()
    maskbits = get_maskbits(ra, dec, mpicomm=mpicomm)
    if mpicomm.rank == 0: logger.info(f'Apply DR9 maskbits done in {MPI.Wtime() - start:2.2f} s.')

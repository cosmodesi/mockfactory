"""Script by Antoine Rocher to transform cubic mock to DESI cutsky mock."""

import os
import time
import logging
import argparse
from itertools import chain

import numpy as np


logger = logging.getLogger('DESI SV3 survey')

# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def get_region(dec):
    if dec > 32.375:
        return 'N'
    return 'S'


def read_rosettes():
    # Rosette coordinates
    rosettes = Table.read('rosettes_sv3.ecsv')
    tiles_rosettes = {rosette['ROSETTE_NUMBER']: [int(tid) for tid in rosette['TILEID'].split(',')] for rosette in rosettes}
    direction_rosettes = {rosette['ROSETTE_NUMBER']: (rosette['RA'], rosette['DEC']) for rosette in rosettes}
    region_rosettes = {rosette: get_region(radec[1]) for rosette, radec in direction_rosettes.items()}
    return tiles_rosettes, direction_rosettes, region_rosettes


def catalog_fn(name, rosettes imock=0):
    # name is 'data', 'randoms'; rosettes is list of rosette numbers
    return os.path.join(outdir, '{}_rosettes-{}_{:d}.fits'.format(name, '-'.join([str(rosette) for rosette in rosettes]), imock))


if __name__ == '__main__':
    """
    Example of how to go from cubic box simulation to DESI SV3 cutsky using MPI.
    The 20 SV3 rosettes are split in pairs and isolated rosettes, each of those configuration being cut out of the same box.
    The covariance matrix of the SV3 sample may be estimated as the average of the covariance matrices (estimated over many simulation boxes)
    of the clustering statistics of each pair of rosettes, rescaled by 1 / 10.

    Can be launched with MPI:

        salloc -N 2 -C haswell -t 00:30:00 --qos interactive -L SCRATCH,project
        srun -n 64 python from_box_to_desi_sv3_cutsky.py

    The first run will take a bit of time, to generate the set of lattice vectors for box remapping.
    Most time consuming step is reading maskbits.

    One can generate e.g. 10 mocks with:

        srun -n 64 python from_box_to_desi_sv3_cutsky.py --start 0 --stop 10
    """
    from astropy.table import Table
    import desimodel.footprint
    from mockfactory.remap import Cuboid
    from mockfactory.make_survey import Catalog, DistanceToRedshift, box_to_cutsky, RandomBoxCatalog, BoxCatalog, EuclideanIsometry, RedshiftDensityInterpolator, TabulatedRadialMask
    from mockfactory.desi import get_brick_pixel_quantities
    from mockfactory import utils, setup_logging
    from mpytools.random import MPIRandomState
    from cosmoprimo.fiducial import DESI

    setup_logging()

    parser = argparse.ArgumentParser(help='Generate DESI SV3 cutsky mocks')
    parser.add_argument('--start', type=int, required=False, default=0, help='First mock to generate')
    parser.add_argument('--stop', type=int, required=False, default=1, help='Last (exclusive) mock to generate')
    args = parser.parse_args()


    # Add maskbits?
    # This step can be long. Large sky coverage need several nodes to be executed in small amounts of time ( ~50 bricks per process)
    # collect only maskbits, see mockfactory/desi/brick_pixel_quantities for other quantities as PSFSIZE_R or LRG_mask
    add_brick_quantities = {'maskbits': {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}}
    # Set as False to save time.
    #add_brick_quantities = False
    plot_debug = False

    # Initialize cosmo
    cosmo = DESI()
    d2z = DistanceToRedshift(distance=cosmo.comoving_radial_distance)

    # Output directory
    outdir = '_tests'
    utils.mkdir(outdir)

    ### Box-specific ###
    boxsize = 500.
    boxcenter = 0.
    nbar = 2e-3  # here we generate a mock; density (in (Mpc/h)^(-3)) must be larger than peak data density

    lattice_fn = os.path.join(outdir, 'lattice_max3.npy')
    # Load lattice if precomputed
    try:
        lattice = np.load(lattice_fn, allow_pickle=True)[()]
    except IOError:
        lattice = Cuboid.generate_lattice_vectors(maxint=3, maxcomb=1, sort=False, boxsize=boxsize)
        np.save(lattice_fn, lattice)

    ### Tracer-specific ###
    z = 1.175  # redshift of AbacusSummit snapshot
    regions = ['N', 'S']
    ref_fn = {region: '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/EDAbeta/ELG_{}_clustering.dat.fits'.format(region) for region in regions}
    tiles_fn = '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/tiles-DARK.fits'
    zlim = (0.8, 1.6)
    bits = [1, 11, 12, 13]  # bits specific to ELGs (applied only if add_brick_quantities is provided)

    # RA, DEC direction of each rosette

    # cuboid size, roation along (global) los
    # To update cuboid size (if necessary), look at the available ones in lattice! (print(lattice.keys()))
    transform_rosettes = {}
    cuboidsize1 = (1224.744871391589, 456.4354645876384, 223.6067977499789)
    cuboidsize2 = (1224.744871391589, 353.5533905932738, 288.6751345948128)
    transform_rosettes[1, 2] = (cuboidsize1, 0.)
    transform_rosettes[12, 13] = (cuboidsize1, 0.)
    transform_rosettes[9, 10] = (cuboidsize2, 90.)
    transform_rosettes[8, 17] = (cuboidsize2, 0.)
    transform_rosettes[15, 18] = (cuboidsize2, 45.)
    transform_rosettes[4, 16] = (cuboidsize1, 90.)
    for rosette in [7, 14, 6, 11, 5, 0, 3, 19]:
        transform_rosettes[(rosette,)] = (cuboidsize1, 0.)

    randoms = RandomBoxCatalog(nbar=10. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=None)

    tiles = Table.read(tiles_fn)
    # Rosette coordinates
    tiles_rosettes, direction_rosettes, region_rosettes = read_rosettes()

    # Check rosettes are grouped in the same region
    rosettes_to_region = {rosettes: region_rosettes[rosettes[0]] for rosettes in all_rosettes}
    for rosettes in all_rosettes:
        if not all(region_rosettes[rosette] == rosettes_to_region[region]):
            raise ValueError('Rosettes must be grouped by regions')

    ndata = {}

    for region in regions:

        data = Catalog.read(ref_fn[region])
        # Measure n(z) without weight and define radial mask
        # Radial mask normalizes nbar such that no object is mask at the top of n(z)
        step = 0.01
        density = RedshiftDensityInterpolator(data['Z'], bins=np.arange(zlim[0] - step, zlim[1] + step + 1e-3, step=step), fsky=1., distance=cosmo.comoving_radial_distance)
        mask_radial = TabulatedRadialMask(z=density.z, nbar=density.nbar, zrange=zlim, interp_order=3)
        ndata[region] = ((data['Z'] > zlim[0]) & (data['Z'] < zlim[1])).csum()

    dmax = cosmo.comoving_radial_distance(zlim[-1])

    for imock in range(args.start, args.stop):
        #### Lognormal mock as a placeholder ###
        power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
        from mockfactory import LagrangianLinearMock
        mock = LagrangianLinearMock(power, nmesh=256, boxsize=boxsize, boxcenter=boxcenter, seed=imock + 1, unitary_amplitude=False)
        mpicomm, rank = mock.mpicomm, mock.mpicomm.rank
        mock.set_real_delta_field(bias=1.2 - 1.)  # Lagrangian bias
        mock.set_analytic_selection_function(nbar=nbar)
        mock.poisson_sample(seed=None)
        box = mock.to_catalog()
        # rsd_factor is the factor to multiply velocity with to get displacements (in position units)
        # For this mock it is just f, but it can be e.g. 1 / (a H); 1 / (100 a E) to use Mpc/h
        rsd_factor = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')  # growth rate

        # The following code requests a mockfactory.BoxCatalog to work.
        # mockfactory.BoxCatalog proposes different ways to read catalog in different formats with MPI
        # box = BoxCatalog.read(fn, filetype='fits', position='Position', velocity='Velocity', boxsize=boxsize, boxcenter=boxcenter, mpicomm=mpicomm)
        # We can also directly build BoxCatalog from an array (similar as dtype numpy array, or dictionary of numpy arrays) split on different ranks
        # data['Position'] should be of shape (N, 3) in 3D, ect...
        # box = BoxCatalog(data=data, columns=['Position', 'Velocity', 'Mass'], position='Position', velocity='Velocity', boxsize=boxsize, boxcenter=boxcenter, mpicomm=mpicomm)
        # To ensure that the box is centered: box.recenter()

        mock_data = {}
        for rosettes, (cuboidsize, los_rotation) in transform_rosettes.items():

            # If rosettes are not all in required regions, continue
            if rosettes_to_region[rosettes] not in regions: continue

            if mpicomm.rank == 0: logger.info('Processing rosettes {}.'.format(rosettes))
            tiles_in_rosettes = list(chain(*(tiles_rosettes[rosette] for rosette in rosettes)))

            los = utils.sky_to_cartesian(1., [direction_rosettes[rosette][0] for rosette in rosettes], [direction_rosettes[rosette][1] for rosette in rosettes])
            los = np.mean(los, axis=0)
            center_ra, center_dec = utils.cartesian_to_sky(los)[1:]
            los_rotation = EuclideanIsometry().rotation(los_rotation, axis=los)

            for catalog, name in zip([box] + ([randoms] if imock == 0 else []), ['data', 'randoms']):
                # Let's apply remapping to our catalog!
                remapped = catalog.remap(*lattice[cuboidsize][0])

                drange, rarange, decrange = box_to_cutsky(boxsize=remapped.boxsize, dmax=dmax)
                isometry = remapped.isometry_for_cutsky(drange=drange, rarange=np.array(rarange) + center_ra, decrange=np.array(decrange) + center_dec)[0]

                cutsky = remapped.cutsky_from_isometry(isometry, rdd=None)
                cutsky.isometry(los_rotation)

                if name == 'data':
                    # Apply rsd
                    position = cutsky['RSDPosition'] = cutsky.rsd_position(f=rsd_factor)
                else:
                    position = cutsky['Position']

                distance, cutsky['RA'], cutsky['DEC'] = utils.cartesian_to_sky(position)
                cutsky['Z'] = d2z(distance)

                # Apply tile selection
                mask_rosette, index_tile = desimodel.footprint.is_point_in_desi(tiles[np.isin(tiles['TILEID'], tiles_in_rosettes)], cutsky['RA'], cutsky['DEC'], return_tile_index=True)
                mask_rosette &= mask_radial(cutsky['Z'], seed=None)

                if plot_debug:
                    from matplotlib import pyplot as plt
                    n = 2
                    ra, dec = cutsky['RA'][::n].gather(mpiroot=0), cutsky['DEC'][::n].gather(mpiroot=0)
                    ra_masked, dec_masked = cutsky['RA'][mask_rosette][::n].gather(mpiroot=0), cutsky['DEC'][mask_rosette][::n].gather(mpiroot=0)
                    if mpicomm.rank == 0:
                        plt.scatter(ra, dec, alpha=0.1)
                        plt.scatter(ra_masked, dec_masked, alpha=0.1)
                        fig = plt.gcf()
                        fig.savefig(catalog_fn(name, rosettes).replace('.fits', '.png'))
                        plt.close(fig)

                cutsky = cutsky[mask_rosette]

                if add_brick_quantities:
                    tmp = get_brick_pixel_quantities(cutsky['RA'], cutsky['DEC'], add_brick_quantities)
                    for key, value in tmp.items(): cutsky[key] = value
                    if bits:  # apply maskbits
                        mask = True
                        for bit in bits: mask &= (cutsky['maskbits'] & 2**bit) == 0
                        cutsky = cutsky[mask]

                if name == 'data':  # store for downsampling
                    mock_data[rosettes] = cutsky
                else:
                    cutsky.write(catalog_fn(name, rosettes, imock=imock))

        for region in regions:  # downsample mock data to data
            mock_data_in_region = {rosettes: mock for rosettes, mock in mock_data.items() if rosettes_to_region[rosettes] == region}
            frac = ndata[region] / sum(mock.csize for mock in mock_data_in_region.values())
            if frac > 1. and mpicomm.rank == 0:
                logger.warning('Not enough density in mocks to match data in {}, ndata / nmock = {:.3f} > 1.'.format(region, frac))
            for rosettes, mock in mock_data_in_region.items():
                rng = MPIRandomState(mock.size, seed=None)
                mock = mock[rng.uniform() < frac]
                mock.write(catalog_fn('data', rosettes, imock=imock))

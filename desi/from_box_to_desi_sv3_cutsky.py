"""Script by Antoine Rocher to transform cubic mock to DESI cutsky mock."""

import os
import time
from itertools import chain

import numpy as np

import desimodel.footprint
from mockfactory.remap import Cuboid
from mockfactory.make_survey import  DistanceToRedshift, box_to_cutsky, RandomBoxCatalog, BoxCatalog, EuclideanIsometry, TabulatedRadialMask
from mockfactory import utils, setup_logging
from mockfactory.desi import get_brick_pixel_quantities
from cosmoprimo.fiducial import DESI


logger = logging.getLogger('DESI SV3 survey')

# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


if __name__ == '__main__':

    # Add maskbits?
    # This step can be long. Large sky coverage need several nodes to be executed in small amounts of time ( ~50 bricks per process)
    # collect only maskbits, see mockfactory/desi/brick_pixel_quantities for other quantities as PSFSIZE_R or LRG_mask
    add_brick_quantities = {'maskbits': {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}}
    # Set as False to save time.
    #add_brick_quantities = False

    # Initialize cosmo
    cosmo = DESI()
    d2z = DistanceToRedshift(distance=cosmo.comoving_radial_distance)

    # Output directory
    outdir = '_tests'

    def catalog_fn(name, rosettes):
        # name is 'data', 'randoms'; rosettes is list of rosette numbers
        return os.path.join(outdir, '{}_rosettes-{}.fits'.format(name, '-'.join([str(rosette) for rosette in rosettes])))

    ### Box-specific ###
    boxsize = 500.
    boxcenter = 0.

    # Load lattice if precomputed
    try:
        lattice_fn = '_tmp/lattice.npy'
        lattice = np.load(lattice_fn, allow_pickle=True)[()]
    except IOError:
        lattice = Cuboid.generate_lattice_vectors(maxint=4, maxcomb=1, sort=False, boxsize=boxsize)
        np.save(lattice_fn, lattice)

    ### Tracer-specific ###
    z = 1.175  # redshift of AbacusSummit snapshot
    regions = ['N', 'S']
    data_fn = {region: '/global/cfs/cdirs/desi/survey/catalogs/SV3/LSS/fuji/LSScats/EDAbeta/ELG_{}_clustering.dat.fits'.format(region) for region in regions}
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
    transform_rosettes[10, 9] = (cuboidsize2, 90.)
    transform_rosettes[17, 8] = (cuboidsize2, 0.)
    transform_rosettes[15, 18] = (cuboidsize2, 0.)
    transform_rosettes[4, 16] = (cuboidsize1, 90.)
    for rosette in [7, 14, 6, 11, 5, 0, 3, 19]:
        transform_rosettes[(rosette,)] = (cuboidsize1, 0.)

    #### Lognormal mock as a placeholder ###
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    mock = LagrangianLinearMock(power, nmesh=512, boxsize=boxsize, boxcenter=boxcenter, seed=None, unitary_amplitude=False)
    mpicomm, rank = mock.mpicomm, mock.mpicomm.rank
    mock.set_real_delta_field(bias=1.2 - 1.)  # Lagrangian bias
    mock.set_analytic_selection_function(nbar=1e-3)
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

    randoms = RandomBoxCatalog(nbar=2e-2, boxsize=boxsize, boxcenter=boxcenter, seed=None)

    tiles = fitsio.read(tiles_fn)
    # Rosette coordinates
    rosettes = Table.read('rosettes_sv3.ecsv')
    tiles_rosettes = {rosette['ROSETTE_NUMBER']: [int(tid) for tid in ','.split(rosette['TILEID'])] for rosette in rosettes}
    direction_rosettes = {rosette['ROSETTE_NUMBER']: (rosette['RA'], rosette['DEC']) for rosette in rosettes}

    ndata, rosettes_regions = {}, {}
    for region in regions:
        data = Catalog.read(data_fn[region])
        # Measure n(z) without weight and define radial mask
        # Radial mask normalizes nbar such that no object is mask at the top of n(z)
        density = RedshiftDensityInterpolator(data['Z'], bins=np.arange(zlim[0], zlim[1] + 1e-3, step=0.01), fsky=1., distance=cosmo.comoving_radial_distance)
        mask_radial = TabulatedRadialMask(z=density.z, nbar=density.nbar, interp_order=3)
        ndata[region] = data.csize
        dec_cut = 32.375
        if region == 'N':
            rosettes_regions[region] = [rosette if radec[1] > dec_cut for rosette, radec in direction_rosettes.items()]
        else:
            rosettes_regions[region] = [rosette if radec[1] < dec_cut for rosette, radec in direction_rosettes.items()]

    dmax = cosmo.comoving_radial_distance(zlim[-1])

    mock_data = {}
    for rosettes, (cuboidsize, los_rotation) in transform_rosettes.items():

        # If rosettes are not all in required regions, continue
        if not all(rosette in chain(*rosettes_regions.values()) for rosette in rosettes): continue

        logger.info('Processing rosettes {}.'.format(rosettes))
        tiles_in_rosette = list(chain(*tiles_rosettes[rosette] for rosette in rosettes)

        los = utils.sky_to_cartesian(1., [direction_rosettes[rosette][0] for rosette in rosettes], [direction_rosettes[rosette][1] for rosette in rosettes])
        los = np.mean(los, axis=0)
        center_ra, center_dec = utils.cartesian_to_sky(los)[2:]
        los_rotation = EuclideanIsometry().rotation(los_rotation, axis=los)

        for catalog, name in zip([box, randoms], ['data', 'randoms']):
            # Let's apply remapping to our catalog!
            remapped = catalog.remap(*lattice[cuboidsize][0])

            drange, rarange, decrange = box_to_cutsky(boxsize=remapped.boxsize, dmax=dmax)
            isometry, mask_radial, mask_angular = remapped.isometry_for_cutsky(drange=drange, rarange=np.array(rarange) + center_ra, decrange=np.array(decrange) + center_dec)

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
            mask_rosette, index_tile = desimodel.footprint.is_point_in_desi(tiles[np.isin(tiles['TILEID'], tiles_in_rosette)], cutsky['RA'], cutsky['DEC'], return_tile_index=True)
            mask_rosette &= mask_radial(cutsky['Z'], seed=None)
            cutsky = cutsky[mask_rosette]

            if add_brick_quantities:
                tmp = get_brick_pixel_quantities(cutsky['RA'], cutsky['DEC'], columns)
                for key, value in tmp.items(): cutsky[key] = value
                if bits:  # apply maskbits
                    cutsky = cutsky[all((cutsky['maskbits'] & 2**bit) == 0 for bit in bits)]

            if name == 'data':  # store for downsampling
                mock_data[rosettes] = cutsky
            else:
                cutsky.write(catalog_fn(name, rosettes))

    for region in regions:  # downsample mock data to data
        mock_data_in_region = {rosettes: mock for rosettes, mock in mock_data.items() if all(rosette in rosettes_regions[region] for rosette in rosettes)}
        frac = ndata[region] / sum(mock.csize for mock in mock_data_in_region.items())
        if frac < 1. and mpicomm.rank == 0:
            logger.warning('Not enough density in mocks to match data')
        for rosettes, mock in mock_data_in_region.items():
            rng = MPIRandomState(mock.size, seed=None)
            mock = mock[rng.uniform() < frac]
            mock.write(catalog_fn('data', rosettes))

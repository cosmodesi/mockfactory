import os
import sys
import logging

import numpy as np


logger = logging.getLogger('DESI Survey')

# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def z_to_chi(z, cosmo):
    """ Convert redshit into comoving distance at a given cosmology.
        The comoving distance is the radial comoving distance since we work in a flat Universe. """
    return cosmo.comoving_radial_distance(z)


def chi_to_z(chi, cosmo):
    """ Convert comoving distance into redshift at a given cosmology. """
    from mockfactory import DistanceToRedshift
    return DistanceToRedshift(cosmo.comoving_radial_distance)(chi)


def remap_the_box(BoxCatalog):
    """ Since the Box is periodic, we can transform the box into Parallelepiped
        following Jordan Carlson and Martin White's algorithm of arXiv:1003.3178.

        **Remark:** It is purely geomtric, we do not want the exact size of the box ect ..
                    We want to collect the desired reorganisation (ie) the lattice values.
    """
    from mockfactory.remap import Cuboid
    # remap the box --> cf notebook to see how we choose the vector:
    lattice = Cuboid.generate_lattice_vectors(maxint=1, maxcomb=1, sort=False,
                                              boxsize=np.array([5520, 5520, 5520]),
                                              cuboidranges=[[8000, 10000], [4000, 5000], [2000, 4000]])
    # Collect the desired lattice.values:
    u = list(lattice.values())[1][0]
    # Let's remap our catalog!
    BoxCatalog = BoxCatalog.remap(*u)
    # use z as depth to maximize the sky surface with remapped box: (x, y, z) --> (z, x, y)
    # (x, y, z) --> (z, y,-x)
    BoxCatalog.rotate(1, axis='y')
    # (z, y, -x) --> (z, x, y)
    BoxCatalog.rotate(1, axis='x')

    return BoxCatalog


def apply_rsd_and_cutsky(BoxCatalog, dmin, dmax, rsd_factor, add_ra=0, add_dec=0):
    """ Move the box and apply the desired cutsky after applying the RSD.

        **WARNING:** RSD need to be applied before the cutsky.
        **WARNING:** make the rotation in cartesian coordinates and not in shperical coordiantes.
                     In this case it is done inside `isometry_for_cutsky`

    Parameters
    ----------
    BoxCatalog: :BoxCatalog: class
        Box containing the simulation. It should be large enough for the desired dmax and dmin.
    dmin: float
        Minimal distance desired for the cutsky. Could be computed with `z_to_chi(z_min)`.
    dmax: float
        Maximal distance desired for the cutsky. Could be computed with `z_to_chi(z_max)`.
    rsd_factor: float
        Factor to apply when adding velocity displacement to the positions. It depends mon the velocity units choice.
    add_ra, add_dec : float
        Add angles to rotate the box. The box is centered around (RA,Dec) = (add_ra, add_dec) --> by default centered around (0, 0)

    Return
    ------
    cutsky: :CutskyCatalog:
        Catalog with desired cutsky and RSD position.
    """
    from mockfactory import box_to_cutsky, utils

    # Collect limit for the cone
    drange, rarange, decrange = box_to_cutsky(BoxCatalog.boxsize, dmax, dmin=dmin)

    # slice rarange et decrange:
    rarange = np.array(rarange) + add_ra
    decrange = np.array(decrange) + add_dec

    # For data, we want to apply RSD *before* selection function
    isometry, mask_radial, mask_angular = BoxCatalog.isometry_for_cutsky(drange, rarange, decrange)

    # First move data to its final position
    data_cutsky = BoxCatalog.cutsky_from_isometry(isometry)
    # Apply RSD
    data_cutsky['RSDPosition'] = data_cutsky.rsd_position(f=rsd_factor)
    # Collect d, ra, DEC
    data_cutsky['DISTANCE'], data_cutsky['RA'], data_cutsky['DEC'] = utils.cartesian_to_sky(data_cutsky['RSDPosition'])
    # Apply selection function (purely groemtric)
    mask = mask_radial(data_cutsky['DISTANCE']) & mask_angular(data_cutsky['RA'], data_cutsky['DEC'])

    # Ra goes from 0 to 360:
    data_cutsky['RA'][data_cutsky['RA'] > 360] = data_cutsky['RA'][data_cutsky['RA'] > 360] - 360

    return data_cutsky[mask]


def apply_radial_mask(cutsky, zmin=0., zmax=6., nz_filename='nz_qso_final.dat', seed=145):
    """ Match the input n(z) distribution between zmin and zmax. Here, we extract the largest
        number of galaxy as much as possible setting norm=1 / nz(z).max(axis=0)
        in `TabulatedRadialMask`.

        Parameters
        ----------
        cutsky: :CutskyCatalog: class
            Catalog containing at least a column 'Z'
        zmin: float
            Minimal redshift to consider in the n(z)
        zmax: float
            Maximum redshift to consider in the n(z)
        nz_filename: str
            Where the n(z) is saved. For now, only the final TS format is accepted.
        seed: int
            For reproductibility during the masking.

        Return
        ------
        cutsky: :CutskyCatalog:class
            Catalog with matched n(z) distribution.
        """
    from scipy.interpolate import interp1d
    from mockfactory import TabulatedRadialMask

    # load nz:
    zbin_min, zbin_max, n_z = np.loadtxt(nz_filename, skiprows=1).T
    zbin_mid = (zbin_min + zbin_max) / 2
    nz = interp1d(zbin_mid, n_z, kind='quadratic', bounds_error=False, fill_value=(0, 0))

    # define radial mask:
    z = np.linspace(zmin, zmax, 51)
    mask_radial = TabulatedRadialMask(z=z, nbar=nz(z), norm=1 / nz(z).max(axis=0))

    return cutsky[mask_radial(cutsky['Z'], seed=seed)]


def generate_redshifts(n_samples, cosmo, zmin=0., zmax=6., nz_filename='nz_qso_final.dat', seed=145):
    """Generate redshifts following the input n(z) distribution between zmin and zmax. This uses a naive implementation
        from `RadialMask`, can be improved if it takes too long.

        Parameters
        ----------
        n_samples: int
            number of redshift that you want to generate
        cosmo: cosmology class
            cosmology from cosmoprimo to convert redshift to distance.
        zmin: float
            Minimal redshift to consider in the n(z)
        zmax: float
            Maximum redshift to consider in the n(z)
        nz_filename: str
            Where the n(z) is saved. For now, only the final TS format is accepted.
        seed: int
            For reproductibility during the masking.

        Return
        ------
        z : float array
            Array of size n_samples which contains redshift following the input tabulated n(z)
        """
    from scipy.interpolate import interp1d
    from mockfactory import TabulatedRadialMask

    # load nz:
    zbin_min, zbin_max, n_z = np.loadtxt(nz_filename, skiprows=1).T
    zbin_mid = (zbin_min + zbin_max) / 2
    nz = interp1d(zbin_mid, n_z, kind='quadratic', bounds_error=False, fill_value=(0, 0))

    # define radial mask:
    z = np.linspace(zmin, zmax, 51)
    mask_radial = TabulatedRadialMask(z=z, nbar=nz(z), norm=1 / nz(z).max(axis=0))

    def distance(z):
        """ We generate randomly points in distance space and not in redshift space """
        return z_to_chi(z, cosmo)

    return mask_radial.sample(n_samples, distance, seed=seed)


def photometric_region_center(region):
    if region == 'N':
        ra, dec = 192.3, 56.0
    elif region == 'SNGC':
        ra, dec = 192, 13.0
    elif region == 'SSGC':
        ra, dec = 6.4, 5.3
    else:
        logger.error(f'photometric_region_centeris not defined for region={region}')
        sys.exit(1)
    return ra, dec


def match_dr9_desi_footprint(cutsky, region, release, program, rank, nside=256):
    """ Remove part of the cutsky to match as best as possible (precision is healpix map at nside)
        the DESI release (as y1) footprint and DR9 photometric footprint. """
    from regressis import DR9Footprint
    from regressis.utils import build_healpix_map
    from mockfactory.desi import is_in_desi_footprint

    # precompute the healpix number:
    _, cutsky['HPX'] = build_healpix_map(nside, cutsky['RA'], cutsky['DEC'], return_pix=True)
    if rank == 0: logger.info(f'Collect healpix number at nside={nside}')

    # Mask objects outside DESI footprint:
    is_in_desi = is_in_desi_footprint(cutsky['RA'], cutsky['DEC'], release=release, program=program)
    if rank == 0: logger.info(f'Create DESI {release}-{program} footprint mask')

    # Load DR9 footprint and create corresponding mask:
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=False, mask_around_des=False, cut_desi=False, verbose=(rank == 0))
    convert_dict = {'N': 'north', 'SNGC': 'south_mid_ngc', 'SSGC': 'south_mid_sgc', 'DES': 'des'}
    is_in_correct_photo = dr9_footprint(convert_dict[region])[cutsky['HPX']]
    if rank == 0: logger.info(f'Create Photo footprint mask for {convert_dict[region]} region')

    return cutsky[is_in_desi & is_in_correct_photo]


if __name__ == '__main__':
    """ Example of how go from Cubicbox simulation to DESI cutsky using MPI.

        Can be launch with MPI:
            salloc -N 4 -C haswell -t 00:30:00 --qos interactive -L SCRATCH,project
            srun -n 256 python from_box_to_desi_cutsky.py

        For debbuging purpose only:
            mpirun -n 4 python from_box_to_desi_cutsky.py
        """

    from mockfactory import LagrangianLinearMock, setup_logging
    from mockfactory.desi import get_brick_pixel_quantities

    # to remove the following warning from pmesh
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    from cosmoprimo.fiducial import DESI

    setup_logging()

    from mpi4py import MPI
    mpicomm = MPI.COMM_WORLD
    rank = mpicomm.Get_rank()
    start_ini = MPI.Wtime()

    # n(z)
    zmin, zmax, nz_filename = 0.8, 2.65, 'nz_qso_final.dat'
    # Choose the footprint: DA02, Y1, Y1-3pass, Y5 for dark or bight time
    release, program = 'y5', 'dark'
    # Do you want also to generate randoms?
    generate_randoms = False

    # load DESI fiducial cosmology
    cosmo = DESI(engine='class')
    # linear power spectrum at z=1.
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=1.)
    f = cosmo.sigma8_z(z=1., of='theta_cb') / cosmo.sigma8_z(z=1., of='delta_cb')  # growth rate

    # Generate a cubic lognormal mock:
    mock = LagrangianLinearMock(power, nmesh=256, boxsize=5500, boxcenter=[0, 0, 0], seed=42, unitary_amplitude=False)
    mock.set_real_delta_field(bias=(2 - 1))  # This is Lagrangian bias, Eulerian bias - 1
    mock.set_analytic_selection_function(nbar=1e-3)
    mock.poisson_sample(seed=792)
    box = mock.to_catalog()

    # for debug --> to be deleted
    box.write('box-before-remap', filetype='bigfile', group=f'{release}-{program}/', overwrite=True)

    # The following code request a mockfactory.BoxCatalog to work.
    # mockfactory.BoxCatalog proposes different way to read catalog in different format with MPI
    # We can also directly build BoxCatalog from an array (similar as dtype numpy array) splitted in different ranks
    # data['Position'] should be (x, y, z) in 3D, ect...
    # box = BoxCatalog(data=data, columns=['Position', 'Velocity', 'Mass'], boxsize=[size_x, size_y, size_z], boxcenter=[x, y, z], mpicomm=mpicomm)
    # To ensure that the box is centered: box.recenter()

    # In order to increase the sky coverage, remap the box:
    start = MPI.Wtime()
    box = remap_the_box(box)
    if rank == 0: logger.info(f"Remap done in {MPI.Wtime() - start:2.2f} s.")

    # With the SAME realisation match North, Decalz_north and Decalz_south region
    # The three regions are not idependent. Usefull to test the geometrical effect / imaging systematic effect on each region
    region_list = ['N']  # ‡, 'SNGC', 'SSGC']

    # Fix seed for reproductibility
    seed_nz, seed_nmock, seed_randoms = 79, 792, 4
    seeds_nz, seeds_nmock = {'N': seed_nz, 'SNGC': seed_nz * 50, 'SSGC': seed_nz * 794}, {'N': seed_nmock, 'SNGC': seed_nmock * 78, 'SSGC': seed_nmock * 7}
    seeds_randoms = {'N': seed_randoms, 'SNGC': seed_randoms * 12, 'SSGC': seed_randoms * 1654}

    for region in region_list:
        start = MPI.Wtime()

        # rotation of the box to match the best as possible each region
        add_ra, add_dec = photometric_region_center(region)
        if rank == 0: logger.info(f'Rotation to match region: {region} with add_ra: {add_ra} and add_dec: {add_dec}')

        # for debug --> to be deleted
        box.write('data-box', filetype='bigfile', group=f'{release}-{program}-{region}/', overwrite=True)

        # create the cutsky
        cutsky = apply_rsd_and_cutsky(box, z_to_chi(zmin, cosmo=cosmo), z_to_chi(zmax, cosmo=cosmo), f, add_ra=add_ra, add_dec=add_dec)

        # for debug --> to be deleted
        cutsky.write('data-test', filetype='bigfile', group=f'{release}-{program}-{region}/', overwrite=True)

        # convert distance to redshift
        cutsky['Z'] = chi_to_z(cutsky['DISTANCE'], cosmo=cosmo)
        # match the nz distribution
        cutsky = apply_radial_mask(cutsky, zmin, zmax, nz_filename=nz_filename, seed=seeds_nz[region])
        if rank == 0: logger.info(f"Remap + cutsky + RSD + radial mask done in {MPI.Wtime() - start:2.2f} s.")

        # match the desi footprint and apply the DR9 mask
        start = MPI.Wtime()
        desi_cutsky = match_dr9_desi_footprint(cutsky, region, release, program, rank)

        # # collect only maskbits, see mockfactory/desi/brick_pixel_quantities for other quantities as PSFSIZE_R or LRG_mask
        # # this step can be long. Large sky coverage need several nodes to be executed in small amounts of time ( ~50 bricks per process)
        # columns = {'maskbits': {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}}
        # desi_cutsky['MASKBITS'] = get_brick_pixel_quantities(desi_cutsky['RA'], desi_cutsky['DEC'], columns, mpicomm=mpicomm)['maskbits']
        # if rank == 0: logger.info(f"Match region: {region} and release footprint: {release} + apply DR9 maskbits done in {MPI.Wtime() - start:2.2f} s.")

        # save desi_cutsky into the same bigfile --> N / SNGC / SSGC
        start = MPI.Wtime()
        desi_cutsky.write('data-cutsky', columns=['RA', 'DEC', 'Z', 'DISTANCE'], filetype='bigfile', group=f'{release}-{program}-{region}/', overwrite=True)
        # desi_cutsky.write('data-cutsky', columns=['RA', 'DEC', 'Z', 'MASKBITS', 'DISTANCE', 'HPX'], filetype='bigfile', group=f'{release}-{program}-{region}/')
        if rank == 0: logger.info(f"Save done in {MPI.Wtime() - start:2.2f} s.\n")

        if generate_randoms:
            # generate associated randoms:
            from mockfactory import RandomCutskyCatalog, box_to_cutsky

            # we want 10 times more than the cutksy mock
            # since random are generated not directly on DESI footprint, we want to know the size of cutsky and not desi_cutsky
            nbr_randoms = int(cutsky.csize * 10)
            # collect limit for the cone
            _, rarange, decrange = box_to_cutsky(box.boxsize, z_to_chi(zmax, cosmo=cosmo), dmin=z_to_chi(zmin, cosmo=cosmo))
            if rank == 0: logger.info(f'Generate randoms for region={region} with seed={seeds_randoms[region]}')

            start = MPI.Wtime()
            randoms = RandomCutskyCatalog(rarange=add_ra + np.array(rarange), decrange=add_dec + np.array(decrange), size=nbr_randoms, seed=seeds_randoms[region], mpicomm=mpicomm)
            if rank == 0: logger.info(f"Generate randoms with RandomCutsky done in {MPI.Wtime() - start:2.2f} s.")

            # match the desi footprint and apply the DR9 mask
            start = MPI.Wtime()
            randoms = match_dr9_desi_footprint(randoms, region, release, program, rank)
            randoms['MASKBITS'] = get_brick_pixel_quantities(randoms['RA'], randoms['DEC'], columns, mpicomm=mpicomm)['maskbits']
            if rank == 0: logger.info(f"Match region: {region} and release footprint: {release} + apply DR9 maskbits done in {MPI.Wtime() - start:2.2f} s.")

            # use the naive implementation of mockfactory/make_survey/BaseRadialMask
            # draw numbers according to a uniform law until to find enough correct numbers
            # basically, this is the so-called 'methode du rejet'
            randoms['Z'] = generate_redshifts(randoms.size, cosmo, zmin, zmax, nz_filename=nz_filename, seed=seeds_randoms[region] * 276)

            # save randoms
            start = MPI.Wtime()
            randoms.write('randoms-cutsky', columns=['RA', 'DEC', 'Z', 'MASKBITS', 'HPX'], filetype='bigfile', group=f'{release}-{program}-{region}/')
            if rank == 0: logger.info(f"Save done in {MPI.Wtime() - start:2.2f} s.\n")

    if rank == 0: logger.info(f"Make survey took {MPI.Wtime() - start_ini:2.2f} s.")

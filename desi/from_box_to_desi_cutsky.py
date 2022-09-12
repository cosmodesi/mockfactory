import os
import logging

import numpy as np


logger = logging.getLogger('DESI Survey')

# To avoid error from NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


def remap_the_box(catalog):
    """
    Since the box is periodic, we can transform the box into a parallelepiped
    following Jordan Carlson and Martin White's algorithm of arXiv:1003.3178.

    It is purely geometric.
    """
    from mockfactory.remap import Cuboid
    # Remap the box, see nb/remap_examples.ipynb to see how we choose the vector:
    lattice = Cuboid.generate_lattice_vectors(maxint=1, maxcomb=1, sort=False,
                                              boxsize=[5500, 5500, 5500],
                                              cuboidranges=[[8000, 10000], [4000, 5000], [2000, 4000]])
    # Collect the desired lattice.values:
    u = list(lattice.values())[1][0]
    # Let's remap our catalog!
    catalog = catalog.remap(*u)
    # use z as depth to maximize the sky surface with remapped box: (x, y, z) --> (z, x, y)
    # (x, y, z) --> (z, y,-x)
    catalog.rotate(1, axis='y')  # 1 in units of pi / 2
    # (z, y, -x) --> (z, x, y)
    catalog.rotate(1, axis='x')

    return catalog


def apply_rsd_and_cutsky(catalog, dmin, dmax, rsd_factor, add_ra=0, add_dec=0):
    """
    Rotate the box to the final position, apply RSD and masks.

    Note
    ----
    RSD needs to be applied before applying the distance cuts.

    Parameters
    ----------
    catalog: BoxCatalog
        Box containing the simulation. Must be large enough for the desired ``dmax`` and ``dmin``.

    dmin : float
        Minimal distance desired for the cutsky. Can be computed with `cosmo.comoving_radial_distance(zmin)`.

    dmax : float
        Maximal distance desired for the cutsky. Can be computed with `cosmo.comoving_radial_distance(zmax)`.

    rsd_factor: float
        Factor to apply to ``catalog.velocity`` to obtain RSD displacement in positions units, to be added to ``catalog.position``.
        It depends on the choice of velocity units in ``catalog``.

    add_ra, add_dec : float, default=0.
        Add angles to rotate the box. The box is centered around (RA, Dec) = (add_ra, add_dec).

    Returns
    -------
    cutsky : CutskyCatalog
        Catalog with desired cutsky and RSD positions.
    """
    from mockfactory import box_to_cutsky, utils

    # Collect limit for the cone
    drange, rarange, decrange = box_to_cutsky(catalog.boxsize, dmax, dmin=dmin)

    # Slice rarange et decrange:
    rarange = np.array(rarange) + add_ra
    decrange = np.array(decrange) + add_dec

    # Collect isometry (transform) and masks to be applied
    isometry, mask_radial, mask_angular = catalog.isometry_for_cutsky(drange, rarange, decrange)
    # First move data to its final position
    data_cutsky = catalog.cutsky_from_isometry(isometry, rdd=None)
    # For data, we apply RSD *before* distance cuts
    data_cutsky['RSDPosition'] = data_cutsky.rsd_position(f=rsd_factor)
    # Collect distance, ra, dec
    data_cutsky['DISTANCE'], data_cutsky['RA'], data_cutsky['DEC'] = utils.cartesian_to_sky(data_cutsky['RSDPosition'], wrap=True)
    # Apply selection function (purely geometric)
    mask = mask_radial(data_cutsky['DISTANCE']) & mask_angular(data_cutsky['RA'], data_cutsky['DEC'])

    return data_cutsky[mask]


def apply_radial_mask(cutsky, zmin=0., zmax=6., nz_filename='nz_qso_final.dat', cosmo=None, seed=145):
    """
    Match the input n(z) distribution between ``zmin`` and ``zmax``.
    Here, we extract the largest number of galaxy as possible (as default).

    Parameters
    ----------
    cutsky: CutskyCatalog
        Catalog containing at least a column 'Z'.

    zmin: float, default=0.
        Minimal redshift to consider in the n(z).

    zmax: float, default=6.
        Maximum redshift to consider in the n(z).

    nz_filename: string, default='nz_qso_final.dat'
        Where the n(z) is saved, in ``cutsky.position`` units, e.g. (Mpc/h)^(-3). For now, only the final TS format is accepted.

    cosmo : Cosmology
        Cosmology of the input mock, to convert n(z) in ``nz_filename`` to mock units.

    seed : int, default=145
        Random seed, for reproductibility during the masking.

    Returns
    -------
    cutsky : CutskyCatalog
        Catalog with matched n(z) distribution.
    """
    from scipy.interpolate import interp1d
    from mockfactory import TabulatedRadialMask

    # Load nz
    zbin_min, zbin_max, n_z = np.loadtxt(nz_filename, skiprows=1).T
    zbin_mid = (zbin_min + zbin_max) / 2
    zedges = np.insert(zbin_max, 0, zbin_min[0])
    dedges = cosmo.comoving_radial_distance(zedges)
    volume = dedges[1:]**3 - dedges[:-1]**3
    nz = interp1d(zbin_mid, n_z / volume, kind='quadratic', bounds_error=False, fill_value=(0, 0))

    # Define radial mask
    z = np.linspace(zmin, zmax, 51)
    mask_radial = TabulatedRadialMask(z=z, nbar=nz(z))

    return cutsky[mask_radial(cutsky['Z'], seed=seed)]


def generate_redshifts(size, zmin=0., zmax=6., nz_filename='nz_qso_final.dat', cosmo=None, seed=145):
    """
    Generate redshifts following the input n(z) distribution between ``zmin`` and ``zmax``.

    Note
    ----
    This uses a naive implementation from `RadialMask`, can be improved if it takes too long.

    Parameters
    ----------
    size : int
        Number of redshifts to generate.

    zmin : float, default=0.
        Minimal redshift to consider in the n(z).

    zmax : float, default=6.
        Maximum redshift to consider in the n(z).

    nz_filename: string, default='nz_qso_final.dat'
        Where the n(z) is saved, in ``cutsky.position`` units, e.g. (Mpc/h)^(-3). For now, only the final TS format is accepted.

    cosmo : Cosmology
        Cosmology of the input mock, to convert n(z) in ``nz_filename`` to mock units.

    seed : int, default=145
        Random seed, for reproductibility during the masking.

    Returns
    -------
    z : array
        Array of size ``size`` of redshifts following the input tabulated n(z).
    """
    from scipy.interpolate import interp1d
    from mockfactory import TabulatedRadialMask

    # Load nz
    zbin_min, zbin_max, n_z = np.loadtxt(nz_filename, skiprows=1).T
    zbin_mid = (zbin_min + zbin_max) / 2
    zedges = np.insert(zbin_max, 0, zbin_min[0])
    dedges = cosmo.comoving_radial_distance(zedges)
    volume = dedges[1:]**3 - dedges[:-1]**3
    nz = interp1d(zbin_mid, n_z / volume, kind='quadratic', bounds_error=False, fill_value=(0, 0))

    # Define radial mask:
    z = np.linspace(zmin, zmax, 51)
    mask_radial = TabulatedRadialMask(z=z, nbar=nz(z))

    # We generate randomly points in redshift space directly, as this is the unit of n_z file
    return mask_radial.sample(size, cosmo.comoving_radial_distance, seed=seed)


def photometric_region_center(region):
    if region == 'N':
        ra, dec = 192.3, 56.0
    elif region == 'DN':
        ra, dec = 192, 13.0
    elif region == 'DS':
        ra, dec = 6.4, 5.3
    else:
        ValueError(f'photometric_region_center is not defined for region={region}')
    return ra, dec


def is_in_photometric_region(ra, dec, region, rank=0):
    """ DN=NNGC and DS = SNGC """
    region = region.upper()
    assert region in ['N', 'DN', 'DS', 'SNGC', 'SSGC', 'DES']

    DR9Footprint = None
    try:
        from regressis import DR9Footprint
    except ImportError:
        if rank == 0: logger.info('Regressis not found, falling back to RA/Dec cuts')

    if DR9Footprint is None:
        mask = np.ones_like(ra, dtype='?')
        if region == 'DES':
            raise ValueError('Do not know DES cuts, install regressis')
        dec_cut = 32.375
        if region == 'N':
            mask &= dec > dec_cut
        else:  # S
            mask &= dec < dec_cut
        if region in ['DN', 'DS']:
            mask_ra = (ra > 100 - dec)
            mask_ra &= (ra < 280 + dec)
            if region == 'DN':
                mask &= mask_ra
            else:  # DS
                mask &= dec > -25
                mask &= ~mask_ra
        return mask
    else:
        from regressis.utils import build_healpix_map
        # Precompute the healpix number
        nside = 256
        _, cutsky['HPX'] = build_healpix_map(nside, ra, dec, return_pix=True)

        # Load DR9 footprint and create corresponding mask
        dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=False, mask_around_des=False, cut_desi=False, verbose=(rank == 0))
        convert_dict = {'N': 'north', 'DN': 'south_mid_ngc', 'DS': 'south_mid_sgc', 'DES': 'des'}
        return dr9_footprint(convert_dict[region])[cutsky['HPX']]


def match_photo_desi_footprint(cutsky, region, release, program='dark', npasses=None, rank=0):
    """
    Remove part of the cutsky to match as best as possible (precision is healpix map at nside)
    the DESI release (e.g. y1) footprint and DR9 photometric footprint.
    """
    from mockfactory.desi import is_in_desi_footprint

    # Mask objects outside DESI footprint:
    is_in_desi = is_in_desi_footprint(cutsky['RA'], cutsky['DEC'], release=release, program=program, npasses=npasses)
    if rank == 0: logger.info(f'Create DESI {release}-{program} footprint mask')

    is_in_photo = is_in_photometric_region(cutsky['RA'], cutsky['DEC'], region, rank=rank)
    if rank == 0: logger.info(f'Create photometric footprint mask for {region} region')
    return cutsky[is_in_desi & is_in_photo]


if __name__ == '__main__':
    """
    Example of how to go from cubic box simulation to DESI cutsky using MPI.

    Can be launched with MPI:
        salloc -N 4 -C haswell -t 00:30:00 --qos interactive -L SCRATCH,project
        srun -n 256 python from_box_to_desi_cutsky.py

    For debbuging purpose only:
        mpirun -n 4 python from_box_to_desi_cutsky.py
    """
    from mockfactory import LagrangianLinearMock, setup_logging
    from mockfactory.desi import get_brick_pixel_quantities

    # To remove the following warning from pmesh (no need for pmesh version in cosmodesiconda)
    # import warnings
    # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    from cosmoprimo.fiducial import DESI

    setup_logging()

    from mpi4py import MPI
    start_ini = MPI.Wtime()

    # Output directory
    outdir = '_tests'

    # n(z)
    zmin, zmax, nz_filename = 0.8, 2.65, 'nz_qso_final.dat'

    # Choose the footprint: DA02, Y1, Y5 for dark or bight time
    release, program, npasses = 'y1', 'dark', 3
    programpasses = f'{program}-{npasses}' if npasses is not None else program

    # Do you want also to generate randoms?
    generate_randoms = True

    # Add maskbits?
    # This step can be long. Large sky coverage need several nodes to be executed in small amounts of time ( ~50 bricks per process)
    # collect only maskbits, see mockfactory/desi/brick_pixel_quantities for other quantities as PSFSIZE_R or LRG_mask
    add_brick_quantities = {'maskbits': {'fn': '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/{brickname}/legacysurvey-{brickname}-maskbits.fits.fz', 'dtype': 'i2', 'default': 1}}
    # Set as False to save time.
    add_brick_quantities = False

    # output format. For large dataset, fits is not always the best...
    fmt = 'fits'
    # fmt = 'bigfile'

    # Load DESI fiducial cosmology
    cosmo = DESI(engine='class')
    z = (zmin + zmax) / 2.
    # Linear power spectrum at median z
    power = cosmo.get_fourier().pk_interpolator().to_1d(z=z)
    f = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')  # growth rate
    bias = (2. - 1)  # this is Lagrangian bias, Eulerian bias - 1

    # Generate a cubic lognormal mock:
    mock = LagrangianLinearMock(power, nmesh=512, boxsize=5500, boxcenter=[0, 0, 0], seed=42, unitary_amplitude=False)
    mpicomm, rank = mock.mpicomm, mock.mpicomm.rank
    mock.set_real_delta_field(bias=bias)
    mock.set_analytic_selection_function(nbar=1e-5)
    mock.poisson_sample(seed=792)
    box = mock.to_catalog()

    # The following code requests a mockfactory.BoxCatalog to work.
    # mockfactory.BoxCatalog proposes different way to read catalog in different format with MPI
    # We can also directly build BoxCatalog from an array (similar as dtype numpy array, or dictionary of numpy arrays) split on different ranks
    # data['Position'] should be of shape (N, 3) in 3D, ect...
    # box = BoxCatalog(data=data, columns=['Position', 'Velocity', 'Mass'], boxsize=[size_x, size_y, size_z], boxcenter=[x, y, z], mpicomm=mpicomm)
    # To ensure that the box is centered: box.recenter()

    # In order to increase the sky coverage, remap the box:
    start = MPI.Wtime()
    box = remap_the_box(box)
    if rank == 0: logger.info(f'Remapping done in {MPI.Wtime() - start:2.2f} s.')

    # With the SAME realisation match North (BASS / MzLS), DECaLS NGC and DECaLS SGC region
    # The three regions are not independent. Useful to test the geometrical effect / imaging systematic effect on each region
    regions = ['N', 'DN', 'DS']

    # Fix random seed for reproductibility
    seed_mock, seed_randoms = 79, 792
    seeds_data_nz = {region: seed_mock + i for i, region in enumerate(regions)}
    seeds_randoms = {region: seed_randoms + i for i, region in enumerate(regions)}

    from mockfactory import DistanceToRedshift
    d2z = DistanceToRedshift(cosmo.comoving_radial_distance)

    for region in regions:
        start = MPI.Wtime()

        # rotation of the box to match the best as possible each region
        add_ra, add_dec = photometric_region_center(region)
        if rank == 0: logger.info(f'Rotation to match region: {region} with add_ra: {add_ra} and add_dec: {add_dec}')

        # Create the cutsky
        cutsky = apply_rsd_and_cutsky(box, cosmo.comoving_radial_distance(zmin), cosmo.comoving_radial_distance(zmax), f, add_ra=add_ra, add_dec=add_dec)

        # Convert distance to redshift
        cutsky['Z'] = d2z(cutsky['DISTANCE'])
        # Match the nz distribution
        cutsky = apply_radial_mask(cutsky, zmin, zmax, nz_filename=nz_filename, cosmo=cosmo, seed=seeds_data_nz[region])
        if rank == 0: logger.info(f'Remap + cutsky + RSD + radial mask done in {MPI.Wtime() - start:2.2f} s.')

        # Match the desi footprint and apply the DR9 mask
        start = MPI.Wtime()
        desi_cutsky = match_photo_desi_footprint(cutsky, region, release, program, npasses=npasses, rank=rank)
        if add_brick_quantities:
            tmp = get_brick_pixel_quantities(desi_cutsky['RA'], desi_cutsky['DEC'], add_brick_quantities, mpicomm=mpicomm)
            for key, value in tmp.items(): desi_cutsky[key.upper()] = value
        if rank == 0: logger.info(f'Match region: {region} and release footprint: {release} + bricks done in {MPI.Wtime() - start:2.2f} s.')

        start = MPI.Wtime()
        fn = os.path.join(outdir, 'data-cutsky')
        columns = desi_cutsky.columns(exclude=['RSDPosition'])
        if fmt == 'bigfile':
            # Save desi_cutsky into the same bigfile: N / DN / DS
            desi_cutsky.write(fn, columns=columns, filetype='bigfile', group=f'{release}-{programpasses}-{region}/', overwrite=True)
        else:
            desi_cutsky.write(f'{fn}-{release}-{programpasses}-{region}.{fmt}', columns=columns)
        if rank == 0: logger.info(f'Data saved in {MPI.Wtime() - start:2.2f} s.\n')

        if generate_randoms:
            # Generate associated randoms
            from mockfactory import RandomCutskyCatalog, box_to_cutsky

            # We want 10 times more than the cutsky mock
            nrand_over_data = 10
            # Since random are generated not directly on DESI footprint, we take the size of cutsky and not desi_cutsky
            nbr_randoms = int(cutsky.csize * nrand_over_data + 0.5)
            # Collect limit for the cone
            _, rarange, decrange = box_to_cutsky(box.boxsize, cosmo.comoving_radial_distance(zmax), dmin=cosmo.comoving_radial_distance(zmin))
            if rank == 0: logger.info(f'Generate randoms for region: {region} with seed: {seeds_randoms[region]}')

            start = MPI.Wtime()
            randoms = RandomCutskyCatalog(rarange=add_ra + np.array(rarange), decrange=add_dec + np.array(decrange), size=nbr_randoms, seed=seeds_randoms[region], mpicomm=mpicomm)
            if rank == 0: logger.info(f'Randoms generated in {MPI.Wtime() - start:2.2f} s.')

            # Match the desi footprint and apply the DR9 mask
            start = MPI.Wtime()
            randoms = match_photo_desi_footprint(randoms, region, release, program, npasses=npasses, rank=rank)
            if add_brick_quantities:
                tmp = get_brick_pixel_quantities(randoms['RA'], randoms['DEC'], add_brick_quantities, mpicomm=mpicomm)
                for key, value in tmp.items(): randoms[key.upper()] = value
            if rank == 0: logger.info(f'Match region: {region} and release footprint: {release} + bricks done in {MPI.Wtime() - start:2.2f} s.')

            # Use the naive implementation of mockfactory/make_survey/BaseRadialMask
            # draw numbers according to a uniform law until to find enough correct numbers
            # basically, this is the so-called 'methode du rejet'
            randoms['Z'] = generate_redshifts(randoms.size, zmin, zmax, nz_filename=nz_filename, cosmo=cosmo, seed=seeds_randoms[region] * 276)

            # save randoms
            start = MPI.Wtime()
            fn = os.path.join(outdir, 'randoms-cutsky')
            if fmt == 'bigfile':
                randoms.write(fn, filetype='bigfile', group=f'{release}-{programpasses}-{region}/', overwrite=True)
            else:
                randoms.write([f'{fn}-{release}-{programpasses}-{region}-{i:d}.{fmt}' for i in range(nrand_over_data)])

            if rank == 0: logger.info(f'Randoms saved in {MPI.Wtime() - start:2.2f} s.\n')

    if rank == 0: logger.info(f'Make survey took {MPI.Wtime() - start_ini:2.2f} s.')

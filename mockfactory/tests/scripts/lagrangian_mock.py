import os

import numpy as np
from cosmoprimo.fiducial import DESI

from mockfactory import LagrangianLinearMock, RandomBoxCatalog, DistanceToRedshift, utils, setup_logging


def main():
    z = 1.
    # Loading DESI fiducial cosmology
    cosmo = DESI()
    pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)

    dist = cosmo.comoving_radial_distance(z)
    f = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')  # growth rate
    bias, nbar, nmesh, boxsize = 2.0, 1e-3, 256, 1000.
    boxcenter = [dist, 0, 0]
    mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias - 1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=43)
    mock.set_rsd(f=f, los=None)
    data = mock.to_catalog()

    # We've got data, now turn to randoms
    randoms = RandomBoxCatalog(nbar=4. * nbar, boxsize=boxsize, seed=44)

    # Let us cut the above box to some geometry
    drange = [dist - boxsize / 3., dist + boxsize / 3.]
    rarange = [10, 20]
    decrange = [20, 30]

    data = data.cutsky(drange=drange, rarange=rarange, decrange=decrange)
    randoms = randoms.cutsky(drange=drange, rarange=rarange, decrange=decrange)

    distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)

    for catalog in [data, randoms]:
        catalog['Distance'], catalog['RA'], catalog['DEC'] = utils.cartesian_to_sky(catalog.position)
        catalog['Z'] = distance_to_redshift(catalog['Distance'])

    # Let us apply some redshift cuts
    from mockfactory.make_survey import TabulatedRadialMask

    z = np.linspace(0.85, 1.15, 51)
    nbar = np.exp(-200. * (z - 1.0)**2 / 2.)
    mask_radial = TabulatedRadialMask(z=z, nbar=nbar)

    base_dir = '_catalog'

    fn = os.path.join(base_dir, 'data.fits')
    mask = mask_radial(data['Z'], seed=84)
    data[mask].write(fn)

    fn = os.path.join(base_dir, 'randoms.fits')
    mask = mask_radial(randoms['Z'], seed=85)
    randoms[mask].write(fn)


if __name__ == '__main__':

    setup_logging()
    main()

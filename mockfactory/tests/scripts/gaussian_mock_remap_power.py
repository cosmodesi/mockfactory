import os

import numpy as np
from matplotlib import pyplot as plt
from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower

from mockfactory import LagrangianLinearMock, RandomBoxCatalog, DistanceToRedshift, utils, setup_logging
from mockfactory.remap import Cuboid


def kaiser_power(k, pklin, bias, f):
    pklin = bias**2 * pklin(k)
    beta = f / bias
    toret = []
    toret.append((1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pklin)
    toret.append((4. / 3. * beta + 4. / 7. * beta ** 2) * pklin)
    toret.append(8. / 35 * beta ** 2 * pklin)
    return np.array(toret)


def main():
    z = 1.
    # Loading DESI fiducial cosmology
    cosmo = DESI()
    pklin = cosmo.get_fourier().pk_interpolator().to_1d(z=z)

    dist = cosmo.comoving_radial_distance(z)
    f = cosmo.sigma8_z(z=z, of='theta_cb') / cosmo.sigma8_z(z=z, of='delta_cb')  # growth rate
    bias, nbar, nmesh, boxsize, boxcenter = 2.0, 1e-3, 256, 1000., 0.
    mock = LagrangianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
    # this is Lagrangian bias, Eulerian bias - 1
    mock.set_real_delta_field(bias=bias - 1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=43)
    data = mock.to_catalog()
    # We've got data, now turn to randoms
    randoms = RandomBoxCatalog(nbar=4. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=44)
    lattice = Cuboid.generate_lattice_vectors(maxint=2, maxcomb=1, sort=False, boxsize=boxsize)
    # Instantiate Cuboid
    cuboidsize = (3000.0, 2357.022603955158, 141.42135623730954) # this is the remapped box size take from lattice
    u = lattice[cuboidsize][0] # corresponding lattice vectors
    cuboid = Cuboid(*u, boxsize=boxsize)
    # Let's apply remapping to our catalog!
    data = data.remap(cuboid)
    randoms = randoms.remap(cuboid)
    los = 'x'
    data['Position'] = data.rsd_position(los=los)

    poles = CatalogFFTPower(data_positions1=data['Position'], randoms_positions1=randoms['Position'],
                            edges={'step': 0.01}, nmesh=200, los=los, resampler='tsc', interlacing=2, position_type='pos',
                            mpicomm=data.mpicomm).poles
    ells = (0, 2, 4)
    kth = poles.k[poles.k > 0]
    theory = kaiser_power(kth, pklin, bias, f)
    ax = plt.gca()
    for ill, ell in enumerate(ells):
        ax.plot(poles.k, poles.k * poles(ell=ell, complex=False), color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        ax.plot(kth, kth * theory[ill], linestyle='--', color='C{:d}'.format(ill))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    if data.mpicomm.rank == 0:
        plt.show()


if __name__ == '__main__':

    setup_logging()
    main()

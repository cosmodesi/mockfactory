import numpy as np
from matplotlib import pyplot as plt
from cosmoprimo.fiducial import DESI
from pypower import CatalogFFTPower

from mockfactory import EulerianLinearMock, RandomBoxCatalog, setup_logging


def kaiser_power(k, pklin, bias, f):
    pklin = bias**2 * pklin(k)
    beta = f / bias
    toret = []
    toret.append((1. + 2. / 3. * beta + 1. / 5. * beta ** 2) * pklin)
    toret.append((4. / 3. * beta + 4. / 7. * beta ** 2) * pklin)
    toret.append(8. / 35 * beta ** 2 * pklin)
    return np.array(toret)


def main():
    nmesh = 100
    boxsize = 500
    boxcenter = 0
    seed = 42
    f = 0.8
    bias = 2.
    los = 'x'
    pklin = DESI().get_fourier().pk_interpolator().to_1d(z=1)

    # unitary_amplitude forces amplitude to 1
    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)

    data = RandomBoxCatalog(nbar=4e-3, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.

    poles = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], edges={'step': 0.01},
                            los=los, boxsize=boxsize, boxcenter=boxcenter, nmesh=100,
                            resampler='tsc', interlacing=2,
                            position_type='pos', mpicomm=data.mpicomm).poles
    ells = (0, 2, 4)
    kth = poles.k[poles.k > 0]
    theory = kaiser_power(kth, pklin, bias, f)
    ax = plt.gca()
    for ill, ell in enumerate(ells):
        ax.plot(poles.k, poles.k * poles(ell=ell, complex=True), color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
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

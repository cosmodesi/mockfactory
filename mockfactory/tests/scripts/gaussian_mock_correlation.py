import numpy as np
from matplotlib import pyplot as plt
from cosmoprimo.fiducial import DESI
from pycorr import TwoPointCorrelationFunction

from mockfactory import EulerianLinearMock, RandomBoxCatalog, setup_logging


def kaiser_correlation(s, pklin, bias, f):
    beta = f / bias
    toret = []
    toret.append(bias**2 * (1. + 2. / 3. * beta + 1. / 5. * beta**2) * pklin.to_xi(fftlog_kwargs={'ell': 0})(s))
    toret.append(bias**2 * (4. / 3. * beta + 4. / 7. * beta**2) * pklin.to_xi(fftlog_kwargs={'ell': 2})(s))
    toret.append(bias**2 * 8. / 35 * beta**2 * pklin.to_xi(fftlog_kwargs={'ell': 4})(s))
    return np.array(toret)


def main(fft=False):
    nmesh = 100
    boxsize = 500
    boxcenter = 0
    seed = 42
    f = 0.9
    bias = 1.7
    z = 1.1
    los = 'x'
    pklin = DESI().get_fourier().pk_interpolator().to_1d(z=z)

    # unitary_amplitude forces amplitude to 1
    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=los)

    if fft:
        from pypower import MeshFFTCorr
        ells = (0, 2, 4)
        edges = np.linspace(0., 150, 51)
        result = MeshFFTCorr(mock.mesh_delta_r + 1, edges=edges, boxcenter=boxcenter, ells=ells, los=los).poles
        s, xiell = result(ell=ells, return_s=True)

    else:
        data = RandomBoxCatalog(nbar=4e-3, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
        data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.
        edges = (np.linspace(0., 50, 51), np.linspace(-1., 1., 101))
        result = TwoPointCorrelationFunction('smu', edges, data_positions1=data['Position'], data_weights1=data['Weight'],
                                            engine='corrfunc', los=los, boxsize=boxsize, position_type='pos', mpicomm=data.mpicomm, nthreads=4)
        ells = (0, 2, 4)
        s, xiell = result(ells=ells, return_sep=True)

    theory = kaiser_correlation(s, pklin, bias, f)
    ax = plt.gca()
    for ill, ell in enumerate(ells):
        ax.plot(s, s**2 * xiell[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
        ax.plot(s, s**2 * theory[ill], linestyle='--', color='C{:d}'.format(ill))
    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r'$s$')
    ax.set_ylabel(r'$s^{2} \xi_{\ell}(s)$ [$(\mathrm{Mpc}/h)^{2}$]')
    if mock.mpicomm.rank == 0:
        plt.show()


if __name__ == '__main__':

    setup_logging()
    main()

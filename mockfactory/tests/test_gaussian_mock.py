import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI
from nbodykit.lab import FFTPower, UniformCatalog, FKPCatalog, ConvolvedFFTPower

from mockfactory import EulerianLinearMock, LagrangianLinearMock, setup_logging

seed = 42
boxsize = 500*np.ones(3, dtype='f8')
los = np.array([0,0,1], dtype='f8')
nmesh = 64*np.ones(3, dtype='i8')
boxcenter = 40000.*los
ells = (0,)
redshift = 1.0
f = 0.8
bias = 2.0
nbar = 2e-3
z = 1.
power = DESI().get_fourier().pk_interpolator().to_1d(z=z)


def kaiser(k):
    pklin = bias**2*power(k)
    beta = f/bias
    toret = []
    toret.append((1. + 2./3.*beta + 1./5.*beta**2)*pklin)
    toret.append((4./3.*beta + 4./7.*beta**2)*pklin)
    toret.append(8./35*beta**2*pklin)
    return np.array(toret)


def plot_power_spectrum(result, model=None):
    poles = result.poles
    ells = result.attrs['poles']
    colors = ['C{:d}'.format(i) for i in range(len(ells))]
    for ell,color in zip(ells,colors):
        label = r'$\ell = {:d}$'.format(ell)
        pk = poles['power_{:d}'.format(ell)].real
        if ell == 0: pk = pk - poles.attrs['shotnoise']
        plt.plot(poles['k'], poles['k'] * pk, label=label, color=color)
    if model is not None:
        k = np.linspace(poles['k'].min()+1e-5,poles['k'].max(),100)
        model = model(k)
        for ill,color in enumerate(colors):
            plt.plot(k, k*model[ill], color=color, linestyle=':')

    # format the axes
    plt.legend(loc=0)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$k \ P_\ell$ [$h^{-2} \mathrm{Mpc}^2$]")

    if result.comm.rank == 0:
        plt.show()


def test_eulerian():

    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    #mock.set_rsd(f=f, los=los)
    mock.set_rsd(f=f)
    result = FFTPower(mock.mesh_delta_r, los=los, mode='2d', poles=ells, dk=0.01, kmin=0.)
    plot_power_spectrum(result, model=kaiser)

    data = UniformCatalog(nbar, boxsize, seed=seed)
    randoms = UniformCatalog(nbar, boxsize, seed=seed+1)
    for catalog in [data, randoms]:
        catalog['Position'] += mock.boxcenter - mock.boxsize/2.
        catalog['NZ'] = catalog['Weight']*nbar
        catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')
    data['Weight'] = mock.readout(data['Position'], field='1+delta', resampler='cic')

    fkp = FKPCatalog(data, randoms, nbar='NZ')
    mesh = fkp.to_mesh(position='Position', fkp_weight='WEIGHT_FKP', comp_weight='Weight', nbar='NZ', BoxSize=1000, Nmesh=128, resampler='cic', interlaced=True)
    result = ConvolvedFFTPower(mesh, poles=ells, dk=0.01)
    plot_power_spectrum(result, model=kaiser)


def test_lagrangian():

    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias-1)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed)
    mock.set_rsd(f=f)
    data = mock.to_nbodykit_catalog()

    randoms = UniformCatalog(nbar, boxsize, seed=seed+1)
    randoms['Position'] += mock.boxcenter
    for catalog in [data, randoms]:
        catalog['NZ'] = catalog['Weight']*nbar
        catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')

    fkp = FKPCatalog(data, randoms, nbar='NZ')
    mesh = fkp.to_mesh(position='Position', fkp_weight='WEIGHT_FKP', comp_weight='Weight', nbar='NZ', BoxSize=1000, Nmesh=128, resampler='tsc', interlaced=True)
    result = ConvolvedFFTPower(mesh, poles=ells, dk=0.01)
    plot_power_spectrum(result, model=kaiser)


if __name__ == '__main__':

    setup_logging()
    test_eulerian()
    #test_lagrangian()

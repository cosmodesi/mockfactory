import numpy as np
from matplotlib import pyplot as plt

from cosmoprimo.fiducial import DESI
from nbodykit.lab import FFTPower, UniformCatalog, FKPCatalog, ConvolvedFFTPower

from mockfactory import EulerianLinearMock, LagrangianLinearMock, utils, setup_logging

seed = 42
boxsize = 500*np.ones(3, dtype='f8')
los = np.array([0,0,1], dtype='f8')
nmesh = 100*np.ones(3, dtype='i8')
boxcenter = 40000.*los
ells = (0, 2)
redshift = 1.0
f = 0.8
bias = 2.0
nbar = 4e-3
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


def test_pm():
    from pmesh.pm import ParticleMesh
    pm = ParticleMesh(BoxSize=[2.]*3, Nmesh=[2]*3, dtype='f8')
    mesh = pm.create(type='real')
    positions = np.array([1.8]*3)[None,:]
    mesh.paint(positions, resampler='cic')
    print(mesh.value[:])
    values = mesh.readout(positions, resampler='nnb')
    print(values)
    for rslab, slab in zip(mesh.slabs.x,mesh.slabs):
        print('x', rslab[0])
        print('y', rslab[1])
        print('z', rslab[2])
        print('value', slab)


def test_readout():
    from pmesh.pm import ParticleMesh
    pm = ParticleMesh(BoxSize=[100.]*3, Nmesh=[100]*3, dtype='f8')
    mesh = pm.create(type='real')
    for rslab, slab in zip(mesh.slabs.x,mesh.slabs):
        mask = (rslab[0] <= 0) & (~np.isnan(rslab[1])) & (~np.isnan(rslab[2]))
        slab[mask] = 1.
    positions = np.array([np.linspace(0., pm.BoxSize[ii], 10) for ii in range(3)]).T
    layout = pm.decompose(positions, smoothing='nnb')
    positions = layout.exchange(positions)
    values = mesh.readout(positions, resampler='nnb')
    print(positions)
    print(values)


def test_uniform():

    nbar = 1e-3
    boxsize = 1000.
    first = UniformCatalog(nbar, boxsize, seed=42).to_mesh(Nmesh=nmesh, resampler='cic', interlaced=True)
    second = UniformCatalog(nbar, boxsize, seed=43).to_mesh(Nmesh=nmesh, resampler='cic', interlaced=True)
    result = FFTPower(first, second=second, los=los, mode='2d', poles=(0,), dk=0.01, kmin=0.)
    plot_power_spectrum(result, model=None)


def test_eulerian():
    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    #mock.set_rsd(f=f, los=los)
    mock.set_rsd(f=f)
    result = FFTPower(mock.mesh_delta_r, los=los, mode='2d', poles=ells, dk=0.01, kmin=0.)
    plot_power_spectrum(result, model=kaiser)

    mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    #mock.set_rsd(f=f, los=los)
    mock.set_rsd(f=f)

    data = UniformCatalog(nbar, boxsize, seed=seed)
    randoms = UniformCatalog(nbar, boxsize, seed=seed+1)
    for catalog in [data, randoms]:
        catalog['Position'] += mock.boxcenter - mock.boxsize/2.
        catalog['NZ'] = catalog['Weight']*nbar
        catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='ngp', compensate=True) + 1.

    fkp = FKPCatalog(data, randoms, nbar='NZ')
    mesh = fkp.to_mesh(position='Position', fkp_weight='WEIGHT_FKP', comp_weight='Weight', nbar='NZ', BoxSize=1000, Nmesh=100, resampler='cic', interlaced=True)
    result = ConvolvedFFTPower(mesh, poles=ells, dk=0.01)
    plot_power_spectrum(result, model=kaiser)
    #plot_power_spectrum(result, model=lambda k: [bias**2*power(k)]*3)


def test_lagrangian():

    bias = 2.

    mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias-1.)
    mock.set_analytic_selection_function(nbar=nbar)
    mock.poisson_sample(seed=seed, resampler='cic', compensate=True)
    #mock.set_rsd(f=f, los=los)
    mock.set_rsd(f=f)
    data = mock.to_nbodykit_catalog()

    randoms = UniformCatalog(nbar, boxsize, seed=seed+1)
    randoms['Position'] += mock.boxcenter - mock.boxsize/2.
    for catalog in [data, randoms]:
        catalog['NZ'] = catalog['Weight']*nbar
        catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')


    fkp = FKPCatalog(data, randoms, nbar='NZ')
    mesh = fkp.to_mesh(position='Position', fkp_weight='WEIGHT_FKP', comp_weight='Weight', nbar='NZ', BoxSize=1000, Nmesh=100, resampler='tsc', interlaced=True)
    result = ConvolvedFFTPower(mesh, poles=ells, dk=0.01)
    #plot_power_spectrum(result, model=kaiser)
    plot_power_spectrum(result, model=lambda k: [bias**2*power(k)]*3)

    from nbodykit.lab import LogNormalCatalog
    from nbodykit import cosmology
    cosmo = cosmology.Planck15
    data = LogNormalCatalog(power, bias=bias, cosmo=cosmo, redshift=redshift, nbar=nbar, BoxSize=boxsize, Nmesh=nmesh, seed=seed, unitary_amplitude=True)
    fref = cosmo.scale_independent_growth_rate(redshift)
    data['Displacement'] = data['VelocityOffset']/fref
    data['Position'] += mock.boxcenter - mock.boxsize/2.
    #data['Position'] -= data['Displacement']
    data['Position'] += f*utils.vector_projection(data['Displacement'], los)
    for catalog in [data, randoms]:
        catalog['NZ'] = catalog['Weight']*nbar
        catalog['WEIGHT_FKP'] = np.ones(catalog.size,dtype='f8')

    fkp = FKPCatalog(data, randoms, nbar='NZ')
    mesh = fkp.to_mesh(position='Position', fkp_weight='WEIGHT_FKP', comp_weight='Weight', nbar='NZ', BoxSize=1000, Nmesh=100, resampler='tsc', interlaced=True)
    ref = ConvolvedFFTPower(mesh, poles=ells, dk=0.01)
    kref = ref.poles['k']
    pkref = [ref.poles['power_{:d}'.format(ell)].real - (ell == 0) * ref.attrs['shotnoise'] for ell in ells]
    #plot_power_spectrum(result, model=kaiser)
    plot_power_spectrum(result, model=lambda k: [np.interp(k, kref, pk) for pk in pkref])


def test_mpi():
    nmesh = 64
    nbar = 1e-5
    from mpi4py import MPI
    from mockfactory.make_survey import RandomBoxCatalog

    def run_eulerian(mpicomm=MPI.COMM_WORLD):
        mock = EulerianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True, mpicomm=mpicomm)
        mock.set_real_delta_field(bias=bias)
        #mock.set_rsd(f=f, los=los)
        mock.set_rsd(f=f)
        data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed, mpicomm=mpicomm)
        data['Weight'] = mock.readout(data['Position'], field='delta', resampler='ngp', compensate=True) + 1.
        return data

    def run_lagrangian(mpicomm=MPI.COMM_WORLD):

        mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True, mpicomm=mpicomm)
        mock.set_real_delta_field(bias=bias-1.)
        mock.set_analytic_selection_function(nbar=nbar)
        mock.poisson_sample(seed=seed, resampler='cic', compensate=True)
        #mock.set_rsd(f=f, los=los)
        mock.set_rsd(f=f)
        data = mock.to_catalog()
        return data

    data = run_eulerian()
    positions, weights = data.gget('Position'), data.gget('Weight')

    if data.is_mpi_root():
        data = run_eulerian(mpicomm=MPI.COMM_SELF)
        ref_positions = data.get('Position')
        ref_weights = data.get('Weight')
        assert np.allclose(positions, ref_positions)
        assert np.allclose(weights, ref_weights)

    data = run_lagrangian()
    positions = data.gget('Position')

    if data.is_mpi_root():
        data = run_lagrangian(mpicomm=MPI.COMM_SELF)
        ref_positions = data.get('Position')
        assert np.allclose(positions, ref_positions)


if __name__ == '__main__':

    setup_logging()
    #test_uniform()
    #test_pm()
    #test_readout()
    test_eulerian()
    test_lagrangian()
    test_mpi()

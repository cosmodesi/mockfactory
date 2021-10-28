import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import LinearMesh, cosmology, FFTPower, LogNormalCatalog, UniformCatalog, FKPCatalog, ConvolvedFFTPower
from nbodykit import setup_logging

from mockfactory import GaussianFieldMesh, ConvolvedMeshFFTPower

seed = 42
BoxSize = 500*np.ones(3)
los = np.array([0,0,1])
Nmesh = 128*np.ones(3,dtype='i8')
#Nmesh = 64*np.ones(3,dtype='i8')
BoxCenter = 40000.*los
ells = (0,2,4)
redshift = 1.0
growth_rate = 0.8
bias = 2.0
nbar = 1e-1
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')


def kaiser(k):
    pklin = bias**2*Plin(k)
    beta = growth_rate/bias
    toret = []
    toret.append((1. + 2./3.*beta + 1./5.*beta**2)*pklin)
    toret.append((4./3.*beta + 4./7.*beta**2)*pklin)
    toret.append(8./35*beta**2*pklin)
    return np.array(toret)


def plot_power_spectrum(power, model=None):
    poles = power.poles
    ells = power.attrs['poles']
    colors = ['C{:d}'.format(i) for i in range(len(ells))]
    for ell,color in zip(ells,colors):
        label = r'$\ell=%d$' % (ell)
        P = poles['power_%d' %ell].real
        if ell == 0: P = P - poles.attrs['shotnoise']
        plt.plot(poles['k'],poles['k'] * P,label=label,color=color)
    if model is not None:
        k = np.linspace(poles['k'].min(),poles['k'].max(),100)
        model = model(k)
        for ill,color in enumerate(colors):
            plt.plot(k,k*model[ill],color=color,linestyle=':')

    # format the axes
    plt.legend(loc=0)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$k \ P_\ell$ [$h^{-2} \mathrm{Mpc}^2$]")

    if power.comm.rank == 0:
        plt.show()


def test_rsd():

    mesh = GaussianFieldMesh(Plin,Nmesh=Nmesh,BoxSize=BoxSize,seed=seed,unitary_amplitude=True)
    mesh.set_real_delta_field(bias=bias,BoxCenter=BoxCenter)
    mesh.set_linear_rsd_local_los(growth_rate=growth_rate)
    power = FFTPower(mesh.delta_r,los=los,mode='2d',poles=ells,dk=0.01,kmin=0.)
    plot_power_spectrum(power,model=kaiser)


def test_sample():

    mesh = GaussianFieldMesh(Plin,Nmesh=Nmesh,BoxSize=BoxSize,seed=seed,unitary_amplitude=True)
    mesh.set_real_delta_field(bias=bias,BoxCenter=BoxCenter)
    mesh.set_linear_rsd_local_los(growth_rate=growth_rate)

    data = UniformCatalog(nbar, BoxSize, seed=seed)
    randoms = UniformCatalog(nbar, BoxSize, seed=seed+1)
    for catalog in [data,randoms]:
        catalog['Position'] += mesh.attrs['BoxCenter']
        catalog['NZ'] = catalog['Weight']*nbar
    randoms['Weight'] = mesh.read_weights(randoms['Position'],resampler='nnb')

    fkp = FKPCatalog(data,randoms,nbar='NZ')
    mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=1000,Nmesh=256,resampler='tsc',interlaced=True)
    power = ConvolvedFFTPower(mesh,poles=ells,dk=0.01)
    plot_power_spectrum(power,model=kaiser)


def test_selection_function():

    mesh = GaussianFieldMesh(Plin,Nmesh=Nmesh,BoxSize=BoxSize,seed=seed,unitary_amplitude=True)
    mesh.set_real_delta_field(bias=bias,BoxCenter=BoxCenter)
    mesh.set_linear_rsd_local_los(growth_rate=growth_rate)
    randoms = UniformCatalog(nbar,BoxSize,seed=seed)
    randoms['Position'] += mesh.attrs['BoxCenter']
    mesh.set_sampled_selection_function(randoms['Position'],resampler='nnb')
    #mesh.set_sampled_selection_function(randoms['Position'],resampler='cic')
    #mesh.set_analytic_selection_function(nbar)
    power = ConvolvedMeshFFTPower(mesh.to_mesh(),ells,dk=0.01)
    plot_power_spectrum(power,model=kaiser)


if __name__ == '__main__':

    setup_logging()
    #test_rsd()
    #test_sample()
    test_selection_function()
    """
    import timeit
    d = {}
    d['harmonic'] = {'stmt':"mesh.set_linear_rsd_local_los_harmonic(growth_rate=growth_rate)",'number':1}
    d['cartesian'] = {'stmt':"mesh.set_linear_rsd_local_los(growth_rate=growth_rate)",'number':1}

    for key,value in d.items():
        dt = timeit.timeit(**value,globals={**globals(),**locals()})/value['number']*1e3
        print('{} takes {:.3f} milliseconds'.format(key,dt))
    """

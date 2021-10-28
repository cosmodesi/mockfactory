import numpy as np
from matplotlib import pyplot as plt
from nbodykit.lab import cosmology, FFTPower, LogNormalCatalog
from nbodykit import setup_logging

from mockfactory import LogNormalFieldMesh

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
nbar = 1e-3
cosmo = cosmology.Planck15
Plin = cosmology.LinearPower(cosmo, redshift=redshift, transfer='EisensteinHu')


def plot_power_spectrum(*powers):
    for power,linestyle in zip(powers,['-',':']):
        poles = power.poles
        ells = power.attrs['poles']
        colors = ['C{:d}'.format(i) for i in range(len(ells))]
        for ell,color in zip(ells,colors):
            label = r'$\ell=%d$' % (ell)
            P = poles['power_%d' %ell].real
            if ell == 0: P = P - poles.attrs['shotnoise']
            plt.plot(poles['k'],poles['k'] * P,label=label,color=color,linestyle=linestyle)

    # format the axes
    plt.legend(loc=0)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$k \ P_\ell$ [$h^{-2} \mathrm{Mpc}^2$]")

    if power.comm.rank == 0:
        plt.show()


def test_ref():
    catalog = LogNormalCatalog(Plin=Plin,nbar=nbar,BoxSize=BoxSize,Nmesh=Nmesh,bias=bias,seed=seed)
    mesh = catalog.to_mesh()
    power = FFTPower(mesh,los=los,mode='2d',poles=ells,dk=0.01,kmin=0.)
    plot_power_spectrum(power)


def test_lognormal():
    catalog = LogNormalCatalog(Plin=Plin,nbar=nbar,BoxSize=BoxSize,Nmesh=Nmesh,bias=bias,seed=seed)
    f = Plin.cosmo.scale_independent_growth_rate(catalog.attrs['redshift'])
    catalog['Position'] += growth_rate/f*np.sum(catalog['VelocityOffset']*los,axis=-1)[:,None]*los
    catalog['Position'] %= BoxSize
    mesh = catalog.to_mesh()
    power_ref = FFTPower(mesh,los=los,mode='2d',poles=ells,dk=0.01,kmin=0.)

    mesh = LogNormalFieldMesh(Plin,Nmesh=Nmesh,BoxSize=BoxSize,seed=seed,unitary_amplitude=False)
    mesh.set_real_delta_field(bias=bias-1,lognormal_transform=True,BoxCenter=BoxCenter)
    mesh.set_displacement_field(growth_rate=None)
    mesh.set_analytic_selection_function(nbar)
    catalog = mesh.to_catalog(displacement=True,resampler='nnb')
    catalog['Position'] -= BoxCenter
    catalog['Position'] += catalog['Displacement']
    catalog['Position'] += growth_rate*np.sum(catalog['Displacement']*los,axis=-1)[:,None]*los
    catalog['Position'] %= BoxSize
    mesh = catalog.to_mesh()
    power = FFTPower(mesh,los=los,mode='2d',poles=ells,dk=0.01,kmin=0.)
    plot_power_spectrum(power,power_ref)



if __name__ == '__main__':

    setup_logging()
    test_lognormal()

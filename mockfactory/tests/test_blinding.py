import os

import numpy as np
from matplotlib import pyplot as plt
from nbodykit import setup_logging
from nbodykit.transform import SkyToCartesian
from nbodykit.utils import ScatterArray, GatherArray
from nbodykit.lab import *

from cosmopipe.lib.catalog import Catalog, RandomCatalog
from cosmopipe.lib.catalog import utils

from mockfactory.make_survey import box_to_cutsky, distance


base_dir = '_catalog'
data_fn = os.path.join(base_dir,'lognormal_data.npy')
randoms_fn = os.path.join(base_dir,'lognormal_randoms.npy')

cosmo = cosmology.Planck15
fraction = 40

def plot_power_spectrum(*powers):
    for ipower,(power,linestyle) in enumerate(zip(powers,['-','--'])):
        poles = power.poles
        ells = power.attrs['poles']
        colors = ['C{:d}'.format(i) for i in range(len(ells))]
        for ell,color in zip(ells,colors):
            P = poles['power_%d' %ell].real
            if ell == 0: P = P - poles.attrs['shotnoise']
            label = None
            if ipower == 0:
                label = r'$\ell=%d$' % (ell)
            elif ell == 0:
                label = '$\\alpha = {}\\%$ with $z$ shuffled'.format(fraction)
            plt.plot(poles['k'],poles['k'] * P,label=label,color=color,linestyle=linestyle)
    factor = (1 - fraction/100.)**2
    for ell,color in zip(ells,colors):
        P = poles['power_%d' %ell].real
        if ell == 0: P = P - poles.attrs['shotnoise']
        label = None
        if ell == 0:
            label = '$\\alpha = {}\\%$ with $z$ shuffled / $(1 - \\alpha)^{{2}}$'.format(fraction)
        plt.plot(poles['k'],1./factor * poles['k'] * P,label=label,color=color,linestyle=':')

    # format the axes
    plt.legend(loc=0,ncol=2)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$k \ P_\ell(k)$ [$h^{-2} \mathrm{Mpc}^2$]")

    if power.comm.rank == 0:
        plt.savefig('test_blinding_{}.png'.format(fraction),bbox_inches='tight',pad_inches=0.1,dpi=200)
        # plt.show()


def generate_lognormal():
    redshift = 1.
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='CLASS')
    BoxSize = 600.
    nbar = 1e-3
    #nbar = 1e-5
    bias = 2.0
    #Nmesh = 256
    Nmesh = 128
    seed = 42
    cat = LogNormalCatalog(Plin=Plin,nbar=nbar,BoxSize=BoxSize,Nmesh=Nmesh,bias=bias,seed=seed,unitary_amplitude=False)
    cat = Catalog.from_nbodykit(cat)
    cat['Position'] -= cat.attrs['BoxSize'] / 2.
    cat['Position'][:,0] += cosmo.comoving_distance(redshift)
    dmax = cat.attrs['BoxSize'][0]/2. + cosmo.comoving_distance(redshift)
    deltara, deltadec, dmin = box_to_cutsky(cat.attrs['BoxSize'],dmax)
    distance,ra,dec = utils.cartesian_to_sky(cat['Position'])
    los = cat['Position']/distance[:,None]
    distance_to_redshift = utils.DistanceToRedshift(cosmo.comoving_distance)
    cat['Z_COSMO'] = distance_to_redshift(distance)
    cat['Position'] = cat['Position'] + utils.vector_projection(cat['VelocityOffset'],cat['Position'])
    distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'],wrap=False)
    #assert np.allclose(cat['RA'],ra) and np.allclose(cat['DEC'],dec)
    mask = (cat['RA'] >= -deltara/2.) & (cat['RA'] <= deltara/2.) & (cat['DEC'] >= -deltadec/2.) & (cat['DEC'] <= deltadec/2.) & (distance_rsd > dmin) & (distance_rsd < dmax)
    cat['Z'] = distance_to_redshift(distance_rsd)
    cat['DZ'] = cat['Z'] - cat['Z_COSMO']
    cat['NZ'] = cat.ones()*nbar
    """
    mesh = cat.to_nbodykit().to_mesh()
    power = FFTPower(mesh,los=[1,0,0],mode='2d',poles=ells,dk=0.01,kmin=0.)
    plot_power_spectrum(power)
    """
    cat = cat[mask]
    cat.save(data_fn)
    cat.save_fits(data_fn.replace('.npy','.fits'))

    cat = RandomCatalog(BoxSize=BoxSize,BoxCenter=0.,nbar=10*nbar,seed=seed,mpistate='scattered')
    cat['Position'][:,0] += cosmo.comoving_distance(redshift)
    distance_rsd,cat['RA'],cat['DEC'] = utils.cartesian_to_sky(cat['Position'],wrap=False)
    mask = (cat['RA'] >= -deltara/2.) & (cat['RA'] <= deltara/2.) & (cat['DEC'] >= -deltadec/2.) & (cat['DEC'] <= deltadec/2.) & (distance_rsd > dmin) & (distance_rsd < dmax)
    cat['Z'] = distance_to_redshift(distance_rsd)
    cat['NZ'] = cat.ones()*nbar
    cat = cat[mask]
    cat.save(randoms_fn)
    cat.save_fits(randoms_fn.replace('.npy','.fits'))


def test_blinding():

    BoxSize = 2000.
    Nmesh = 256

    ells = (0,)
    data = FITSCatalog(data_fn.replace('.npy','.fits'))
    randoms = FITSCatalog(randoms_fn.replace('.npy','.fits'))
    #data = Catalog.load(data_fn)[::n].to_nbodykit()
    #randoms = Catalog.load(randoms_fn)[::n].to_nbodykit()
    data['Position'] = SkyToCartesian(data['RA'],data['DEC'],data['Z'],cosmo)
    randoms['Position'] = SkyToCartesian(randoms['RA'],randoms['DEC'],randoms['Z'],cosmo=cosmo)
    fkp = FKPCatalog(data,randoms,nbar='NZ')
    mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
    power = ConvolvedFFTPower(mesh,poles=ells,dk=0.01)

    z = GatherArray(data['Z'].compute(),comm=data.comm)
    nz = GatherArray(data['NZ'].compute(),comm=data.comm)

    if data.comm.rank == 0:
        rng = np.random.RandomState(seed=42)
        nreplace = int(np.rint(fraction/100.*z.size))
        print(nreplace)
        indices1 = rng.choice(z.size,size=nreplace,replace=False)
        indices2 = rng.choice(z.size,size=nreplace,replace=False)
        z[indices1] = z[indices2]
        nz[indices1] = nz[indices2]
    else:
        z,nz = None,None

    counts = data.comm.allgather(data['Z'].size)
    data['Z'] = ScatterArray(z,comm=data.comm,counts=counts)
    data['NZ'] = ScatterArray(nz,comm=data.comm,counts=counts)

    data['Position'] = SkyToCartesian(data['RA'],data['DEC'],data['Z'],cosmo)
    fkp = FKPCatalog(data,randoms,nbar='NZ')
    mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
    power_blind = ConvolvedFFTPower(mesh,poles=ells,dk=0.01)

    plot_power_spectrum(power,power_blind)


if __name__ == '__main__':

    setup_logging()
    #generate_lognormal()
    test_blinding()

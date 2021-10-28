import os

import numpy as np
from matplotlib import pyplot as plt
from nbodykit import setup_logging
from nbodykit.transform import SkyToCartesian
from nbodykit.utils import ScatterArray, GatherArray
from nbodykit.lab import *

from mockfactory.make_survey import *
from mockfactory.mpi import MPIRandomState


base_dir = '_catalog'
data_fn = os.path.join(base_dir,'lognormal_data.fits')
randoms_fn = os.path.join(base_dir,'lognormal_randoms.fits')
data_masked_fn = os.path.join(base_dir,'lognormal_masked_data.fits')
randoms_masked_fn = os.path.join(base_dir,'lognormal_masked_randoms.fits')

cosmo = cosmology.Planck15
nbar = 1e-3
nbar_randoms = 10*nbar
masked_fraction = 1./4.
masked_downsampling = 0.6
nside = 256


def generate_lognormal():
    redshift = 1.
    Plin = cosmology.LinearPower(cosmo,redshift,transfer='CLASS')
    BoxSize = 600.
    bias = 2.0
    #Nmesh = 256
    Nmesh = 128
    seed = 42
    catalog = LogNormalCatalog(Plin=Plin,nbar=nbar,BoxSize=BoxSize,Nmesh=Nmesh,bias=bias,seed=seed,unitary_amplitude=True)
    catalog = SurveyCatalog.from_nbodykit(catalog)
    catalog['Velocity'] = catalog['VelocityOffset']
    del catalog['VelocityOffset']
    catalog.translate(-catalog.attrs['BoxSize'] / 2.)
    catalog.translate_along_axis('x',cosmo.comoving_distance(redshift))
    catalog.flush()
    dmax = catalog.attrs['BoxSize'][0]/2. + cosmo.comoving_distance(redshift)
    deltara,deltadec,dmin = box_to_cutsky(catalog.attrs['BoxSize'],dmax)
    distance,ra,dec = cartesian_to_sky(catalog.Position)
    distance_to_redshift = DistanceToRedshift(cosmo.comoving_distance)
    catalog['Z_COSMO'] = distance_to_redshift(distance)
    catalog['RSDPosition'] = catalog.RSDPosition()
    distance_rsd,catalog['RA'],catalog['DEC'] = cartesian_to_sky(catalog['RSDPosition'],wrap=False)
    catalog['Z'] = distance_to_redshift(distance_rsd)
    catalog['DZ'] = catalog['Z'] - catalog['Z_COSMO']
    catalog['NZ'] = catalog.ones()*nbar
    #for field in ['RA','DEC']:
    #    print(field,catalog[field].min(),catalog[field].max())
    #print('distance',distance_rsd.min(),distance_rsd.max(),dmin,dmax)
    mask = (catalog['RA'] >= -deltara/2.) & (catalog['RA'] <= deltara/2.) & (catalog['DEC'] >= -deltadec/2.) & (catalog['DEC'] <= deltadec/2.) & (distance_rsd > dmin) & (distance_rsd < dmax)
    #catalog = catalog[mask]
    catalog.save_fits(data_fn)

    catalog = RandomCatalog(BoxSize=BoxSize,BoxCenter=0.,nbar=nbar_randoms,seed=seed)
    catalog.translate_along_axis('x',cosmo.comoving_distance(redshift))
    catalog.flush()
    distance_rsd,catalog['RA'],catalog['DEC'] = cartesian_to_sky(catalog['Position'],wrap=False)
    catalog['Z'] = distance_to_redshift(distance_rsd)
    catalog['NZ'] = catalog.ones()*nbar
    mask = (catalog['RA'] >= -deltara/2.) & (catalog['RA'] <= deltara/2.) & (catalog['DEC'] >= -deltadec/2.) & (catalog['DEC'] <= deltadec/2.) & (distance_rsd > dmin) & (distance_rsd < dmax)
    #catalog = catalog[mask]
    #catalog.save(randoms_fn)
    catalog.save_fits(randoms_fn)


def mask_catalogs():
    import healpy
    hpmap = np.ones(healpy.nside2npix(nside),dtype='f8')
    hpind = np.arange(hpmap.size)
    hpmap[hpind % int(np.rint(1./masked_fraction)) == 0] = 1. - masked_downsampling

    hpmask = HealpixAngularMask(hpmap,nest=False)
    data = Catalog.load_fits(data_fn)
    data['prob'] = hpmask.prob(data['RA'],data['DEC'])
    rng = MPIRandomState(data.size,mpicomm=data.mpicomm,seed=69)
    data['random_weight'] = rng.uniform(0.2,1.8)
    mask = hpmask(data['RA'],data['DEC'])
    print('Selecting {:d}/{:d} = {:.4f} data'.format(mask.sum(),mask.size,mask.sum()*1./mask.size))
    data.save_fits(data_fn)
    data[mask].save_fits(data_masked_fn)
    randoms = Catalog.load_fits(randoms_fn)
    randoms['prob'] = hpmask.prob(randoms['RA'],randoms['DEC'])
    rng = MPIRandomState(randoms.size,mpicomm=randoms.mpicomm,seed=69)
    randoms['random_weight'] = rng.uniform(0.2,1.8)
    mask = hpmask(randoms['RA'],randoms['DEC'])
    print('Selecting {:d}/{:d} = {:.4f} randoms'.format(mask.sum(),mask.size,mask.sum()*1./mask.size))
    randoms.save_fits(randoms_fn)
    randoms[mask].save_fits(randoms_masked_fn)


ref_fn = os.path.join(base_dir,'ref.json')
masked_fn = os.path.join(base_dir,'masked.json')
masked_globalnz_fn = os.path.join(base_dir,'masked_globalnz.json')
masked_truenz_fn = os.path.join(base_dir,'masked_truenz.json')
weight_fn = os.path.join(base_dir,'weight.json')
weight_nz_fn = os.path.join(base_dir,'weight_nz.json')
ranweight_fn = os.path.join(base_dir,'ranweight.json')
ranweight_nz_fn = os.path.join(base_dir,'ranweight_nz.json')

ref_meshnorm_fn = os.path.join(base_dir,'ref_meshnorm.json')
masked_meshnorm_fn = os.path.join(base_dir,'masked_meshnorm.json')
weight_meshnorm_fn = os.path.join(base_dir,'weight_meshnorm.json')
ranweight_meshnorm_fn = os.path.join(base_dir,'ranweight_meshnorm.json')


def estimate_power_spectra():

    BoxSize = 1000.
    Nmesh = 300
    ells = (0,)

    def get_power(data, randoms):
        data = data.to_nbodykit()
        randoms = randoms.to_nbodykit()
        fkp = FKPCatalog(data,randoms,nbar='NZ')
        mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFFTPower(mesh,poles=ells,dk=0.01)

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    get_power(data,randoms).save(ref_fn)

    data = Catalog.load_fits(data_masked_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_masked_fn)
    get_power(data,randoms).save(masked_fn)

    data = Catalog.load_fits(data_masked_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_masked_fn)
    data['NZ'] *= (1. - masked_downsampling*masked_fraction)
    randoms['NZ'] *= (1. - masked_downsampling*masked_fraction)
    get_power(data,randoms).save(masked_globalnz_fn)

    data = Catalog.load_fits(data_masked_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_masked_fn)
    data['NZ'] *= data['prob']
    randoms['NZ'] *= randoms['prob']
    get_power(data,randoms).save(masked_truenz_fn)

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    data['Weight'] = data['prob']
    randoms['Weight'] = randoms['prob']
    get_power(data,randoms).save(weight_fn)

    data['NZ'] *= data['prob']
    randoms['NZ'] *= randoms['prob']
    get_power(data,randoms).save(weight_nz_fn)

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    randoms['Weight'] = randoms['random_weight']
    get_power(data,randoms).save(ranweight_fn)

    #data['NZ'] *= data['random_weight']
    #randoms['NZ'] *= randoms['random_weight']
    #get_power(data,randoms).save(ranweight_nz_fn)

def plot_power_spectra():

    power = ConvolvedFFTPower.load(ref_fn)
    poles_ref = power.poles

    def plot_ps(power, single_label=None, linestyle='-'):
        poles = power.poles
        ells = power.attrs['poles']
        colors = ['C{:d}'.format(i) for i in range(len(ells))]
        for ell,color in zip(ells,colors):
            pk = poles['power_{:d}'.format(ell)].real
            pk_ref = poles_ref['power_{:d}'.format(ell)].real
            if ell == 0:
                pk = pk - poles.attrs['shotnoise']
                pk_ref = pk_ref - poles_ref.attrs['shotnoise']
            label = None
            if single_label:
                if ell == 0:
                    label = single_label
            else:
                label = '$\ell = {:d}$'.format(ell)
            plt.plot(poles['k'],pk/pk_ref,label=label,color=color,linestyle=linestyle)

    def plot_attrs():
        plt.axhline(1.,0.,1.,color='k',linestyle='--')
        plt.legend(loc=0,ncol=1)
        plt.xlabel('$k$ [$h \ \mathrm{Mpc}^{-1}$]')
        plt.ylabel('$P(k) / P_{\mathrm{ref}}(k)$')
        plt.show()

    power = ConvolvedFFTPower.load(masked_fn)
    plot_ps(power,single_label='masked by {:.4f}, unscaled $n(z)$'.format(masked_fraction),linestyle='-')
    power = ConvolvedFFTPower.load(masked_globalnz_fn)
    plot_ps(power,single_label='masked by {:.4f}, global scaled $n(z)$'.format(masked_fraction),linestyle='--')
    power = ConvolvedFFTPower.load(masked_truenz_fn)
    plot_ps(power,single_label='masked by {:.4f}, local scaled $n(z)$'.format(masked_fraction),linestyle=':')
    plot_attrs()

    power = ConvolvedFFTPower.load(weight_fn)
    plot_ps(power,single_label='weights, unscaled $n(z)$',linestyle='-')
    power = ConvolvedFFTPower.load(weight_nz_fn)
    plot_ps(power,single_label='weights, scaled $n(z)$',linestyle='--')
    plot_attrs()

    power = ConvolvedFFTPower.load(ranweight_fn)
    plot_ps(power,single_label='random weights, unscaled $n(z)$',linestyle='-')
    #power = ConvolvedFFTPower.load(ranweight_nz_fn)
    #plot_ps(power,single_label='random weights, scaled $n(z)$',linestyle='--')
    plot_attrs()


def estimate_mesh_norm_power_spectra():

    BoxSize = 1000.
    Nmesh = 300
    ells = (0,)

    def get_power(data, randoms):
        data = data.to_nbodykit()
        randoms = randoms.to_nbodykit()
        fkp = FKPCatalog(data,randoms,nbar='NZ')
        mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFFTPower(mesh,poles=ells,dk=0.01)

    """
    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    get_power(data,randoms).save(ref_fn)
    """
    #Nmesh = 300
    #BoxSize = 600.

    def get_power(data, randoms):
        from mockfactory.convolved_fkp_power import FKPCatalog, ConvolvedFFTPower
        data = data.to_nbodykit()
        randoms = randoms.to_nbodykit()
        fkp = FKPCatalog(data,randoms)
        mesh = fkp.to_mesh(position='Position',comp_weight='Weight',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFFTPower(mesh,poles=ells,edges={'step':0.01})

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    get_power(data,randoms).save(ref_meshnorm_fn)

    """
    data = Catalog.load_fits(data_masked_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_masked_fn)
    get_power(data,randoms).save(masked_meshnorm_fn)

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    data['Weight'] = data['prob']
    randoms['Weight'] = randoms['prob']
    get_power(data,randoms).save(weight_meshnorm_fn)

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    randoms['Weight'] = randoms['random_weight']
    get_power(data,randoms).save(ranweight_meshnorm_fn)
    """


def plot_mesh_norm_power_spectra():

    power_ref = ConvolvedFFTPower.load(ref_fn)
    poles_ref = power_ref.poles

    def plot_ps(power, single_label=None, linestyle='-'):
        poles = power.poles
        ells = power.attrs['poles']
        colors = ['C{:d}'.format(i) for i in range(len(ells))]
        for ell,color in zip(ells,colors):
            pk = poles['power_{:d}'.format(ell)].real
            pk_ref = poles_ref['power_{:d}'.format(ell)].real
            if ell == 0:
                pk = pk - poles.attrs['shotnoise']
                pk_ref = pk_ref - poles_ref.attrs['shotnoise']
            label = None
            if single_label:
                if ell == 0:
                    label = single_label
            else:
                label = '$\ell = {:d}$'.format(ell)
            mask = (poles_ref['k'] > poles['k'].min()) & (poles_ref['k'] < poles['k'].max())
            plt.plot(poles_ref['k'][mask],np.interp(poles_ref['k'][mask],poles['k'],pk)/pk_ref[mask],label=label,color=color,linestyle=linestyle)

    def plot_attrs():
        plt.axhline(1.,0.,1.,color='k',linestyle='--')
        plt.legend(loc=0,ncol=1)
        plt.xlabel('$k$ [$h \ \mathrm{Mpc}^{-1}$]')
        plt.ylabel('$P(k) / P_{\mathrm{ref}}(k)$')
        plt.show()

    power = ConvolvedFFTPower.load(ref_meshnorm_fn)
    #print(power.attrs['randoms.norm']/power_ref.attrs['randoms.norm'],power_ref.attrs['shotnoise']/power.attrs['shotnoise'])
    print(power.attrs['randoms.norm'],power_ref.attrs['randoms.norm'],600**3*1e-6)
    plot_ps(power,single_label='mesh norm',linestyle='-')
    plot_attrs()
    exit()
    power = ConvolvedFFTPower.load(masked_meshnorm_fn)
    plot_ps(power,single_label='masked by {:.4f}'.format(masked_fraction),linestyle='--')
    power = ConvolvedFFTPower.load(weight_meshnorm_fn)
    plot_ps(power,single_label='weights',linestyle=':')
    power = ConvolvedFFTPower.load(ranweight_meshnorm_fn)
    plot_ps(power,single_label='random weights',linestyle='-.')
    plot_attrs()


def test_shotnoise():

    BoxSize = 600.
    Nmesh = 256
    ells = (0,)
    seed = 42

    data = RandomCatalog(BoxSize=BoxSize,BoxCenter=0.,nbar=nbar,seed=seed)

    def get_power(data):
        mesh = data.to_nbodykit().to_mesh(Nmesh=Nmesh,resampler='tsc',compensated=True,interlaced=True)
        return FFTPower(mesh,mode='2d',poles=ells,dk=0.01)

    power_ref = get_power(data)

    rng = MPIRandomState(data.size,mpicomm=data.mpicomm)
    data['Weight'] = rng.uniform(0.,1.)
    power = get_power(data)

    def plot_ps(power, single_label=None, linestyle='-'):
        poles = power.poles
        ells = power.attrs['poles']
        colors = ['C{:d}'.format(i) for i in range(len(ells))]
        for ell,color in zip(ells,colors):
            pk = poles['power_{:d}'.format(ell)].real
            if ell == 0: pk = pk - poles.attrs['shotnoise']
            label = None
            if single_label:
                if ell == 0:
                    label = single_label
            else:
                label = '$\ell = {:d}$'.format(ell)
            plt.plot(poles['k'],poles['k'] * pk,label=label,color=color,linestyle=linestyle)

    plot_ps(power_ref,single_label='no weight',linestyle='-')
    plot_ps(power,single_label='random weight',linestyle='--')
    plt.show()


def test_resolution():

    BoxSize = 1000.
    Nmesh = 300
    ells = (0,)

    def get_power(data, randoms):
        data = data.to_nbodykit()
        randoms = randoms.to_nbodykit()
        fkp = FKPCatalog(data,randoms,nbar='NZ')
        mesh = fkp.to_mesh(position='Position',comp_weight='Weight',nbar='NZ',BoxSize=BoxSize,Nmesh=Nmesh,resampler='tsc',interlaced=True)
        return ConvolvedFFTPower(mesh,poles=ells,dk=0.01)

    data = Catalog.load_fits(data_fn)
    data['Position'] = data['RSDPosition']
    randoms = Catalog.load_fits(randoms_fn)
    power_ref = get_power(data,randoms)

    Nmesh = 100
    power = get_power(data,randoms)

    poles_ref = power_ref.poles

    def plot_ps(power, single_label=None, linestyle='-'):
        poles = power.poles
        ells = power.attrs['poles']
        colors = ['C{:d}'.format(i) for i in range(len(ells))]
        for ell,color in zip(ells,colors):
            pk = poles['power_{:d}'.format(ell)].real
            pk_ref = poles_ref['power_{:d}'.format(ell)].real
            if ell == 0:
                pk = pk - poles.attrs['shotnoise']
                pk_ref = pk_ref - poles_ref.attrs['shotnoise']
            label = None
            if single_label:
                if ell == 0:
                    label = single_label
            else:
                label = '$\ell = {:d}$'.format(ell)
            mask = (poles_ref['k'] > poles['k'].min()) & (poles_ref['k'] < poles['k'].max())
            plt.plot(poles_ref['k'][mask],np.interp(poles_ref['k'][mask],poles['k'],pk)/pk_ref[mask],label=label,color=color,linestyle=linestyle)

    def plot_attrs():
        plt.axhline(1.,0.,1.,color='k',linestyle='--')
        plt.legend(loc=0,ncol=1)
        plt.xlabel('$k$ [$h \ \mathrm{Mpc}^{-1}$]')
        plt.ylabel('$P(k) / P_{\mathrm{ref}}(k)$')
        plt.show()

    plot_ps(power,single_label='lower resolution',linestyle='-')
    plot_attrs()




if __name__ == '__main__':

    setup_logging()
    #generate_lognormal()
    #mask_catalogs()
    #estimate_power_spectra()
    #plot_power_spectra()
    estimate_mesh_norm_power_spectra()
    plot_mesh_norm_power_spectra()
    #test_shotnoise()
    #test_resolution()

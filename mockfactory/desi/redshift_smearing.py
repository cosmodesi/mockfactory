import os
import logging

import numpy as np
from scipy import constants, stats

from mockfactory import utils
from mockfactory import RVS2DRedshiftSmearing


logger = logging.getLogger('Redshift Smearing')


<<<<<<< HEAD
def RedshiftSmearingRVS(tracer='QSO', fn=('data/qso_redshift_smearing_sv1.ecsv', 'data/qso_redshift_smearing_sv3.ecsv')):
    """
    Return redshifts ``z``, list of continuous random variates for ``z``  
=======
def TracerRedshiftSmearingRVS(tracer='QSO', fn=None):
    """
    Return redshifts ``z``, list of continuous random variates for ``z``
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9
    (QSOs are Laplace and Gaussian, ELGs and BGS are Lorentzian, LRGs are Gaussian)
    weights, and optional ``dz`` transform, given input tabulated file(s) of redshift errors.
    """
    from astropy.table import Table, vstack

    if fn is None:
        dirname = os.path.join(os.path.dirname(__file__), 'data')
        if tracer == 'QSO':
            fn = ['{}_redshift_smearing_{}.ecsv'.format(tracer, sv) for sv in ['sv1', 'sv3']]
        else:
            fn = ['{}_redshift_smearing_sv1.ecsv'.format(tracer)]
        fn = [os.path.join(dirname, ff) for ff in fn]

    if not utils.is_sequence(fn): fn = [fn]

    # Concatenate, keeping unique redshifts
    table = Table()
    for ff in fn:
        tt = Table.read(ff)
        tt.sort('mean_z')
        if len(table):
            table = vstack([table[table['mean_z'] < tt['mean_z'][0]], tt])
        else:
            table = tt
    rvs_nongaussian, rvs_gaussian, laz = [], [], []
    for iz, z in enumerate(table['mean_z']):
        if tracer == 'QSO':
            A0, x0, s0, sg, la = table['val_fit'][iz]
<<<<<<< HEAD
            rvs_nongaussian.append(stats.laplace(x0, s0))
            rvs_gaussian.append(stats.norm(x0, sg))
        else:
            sigma,x0, p,mu, la = table['val_fit'][iz]
            rvs_nongaussian.append(stats.cauchy(scale=p/2, loc=mu))
            rvs_gaussian.append(stats.norm(scale=sigma,loc=x0))
        laz.append(la)
    laz = np.array(laz)
    dztransform = lambda z, dz: dz / (constants.c / 1e3) / (1. + z)  # file units was km / s
    return np.asarray(table['mean_z']), [rvs_nongaussian, rvs_gaussian], [laz, 1. - laz], dztransform
=======
            s0, sg = s0 / np.sqrt(2), sg / np.sqrt(2)
            rvs_nongaussian.append(stats.laplace(x0, s0))
            rvs_gaussian.append(stats.norm(x0, sg))
        else:
            sigma, x0, p, mu, la = table['val_fit'][iz]
            rvs_nongaussian.append(stats.cauchy(scale=p / 2, loc=mu))
            rvs_gaussian.append(stats.norm(scale=sigma, loc=x0))
        laz.append(la)
    laz = np.array(laz)
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9

    if tracer == 'QSO':

<<<<<<< HEAD
def TracerRedshiftSmearing(tracer='QSO', fn=('data/qso_redshift_smearing_sv1.ecsv', 'data/qso_redshift_smearing_sv3.ecsv')):
    """
    Return :class:`RVS2DRedshiftSmearing` instance given input tabulate file of redshift errors.
    Redshift errors can be sampled through: ``dz = rs.sample(z, seed=42)``.
    """
    z, rvs, weights, dztransform = RedshiftSmearingRVS(tracer=tracer, fn=fn)
    if tracer == 'QSO':
        dzscale = 5e3
    elif (tracer == 'ELG')|(tracer == 'BGS'):
        dzscale = 150
    elif tracer == 'LRG':
        dzscale = 200
    return RVS2DRedshiftSmearing.average([RVS2DRedshiftSmearing(z, rv, dzsize=10000, dzscale=dzscale, dztransform=dztransform) for rv in rvs], weights=weights)
    
=======
        def dztransform(z, dz):
            return dz / (constants.c / 1e3) / (1. + z)  # file unit is dz (1 + z) c [km / s]

    else:

        def dztransform(z, dz):
            return dz / (constants.c / 1e3) * (1. + z)  # file unit is c dz / (1 + z) [km / s]

    return np.asarray(table['mean_z']), [rvs_nongaussian, rvs_gaussian], [laz, 1. - laz], dztransform
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9


def TracerRedshiftSmearing(tracer='QSO', fn=None):
    """
    Return :class:`RVS2DRedshiftSmearing` instance given input tabulate file of redshift errors.
    Redshift errors can be sampled through: ``dz = rs.sample(z, seed=42)``.
    """
<<<<<<< HEAD
    if tracer in ['BGS', 'LRG', 'ELG']:
        if fn is None:
            import os
            import mockfactory.desi
            dir = os.path.dirname(mockfactory.desi.__file__)
            fn  = os.path.join(dir, 'data/{}_redshift_smearing_sv1.ecsv'.format(tracer))
        return TracerRedshiftSmearing(tracer=tracer, fn=fn)

    elif tracer == 'QSO':
        if fn is None:
            import os
            import mockfactory.desi
            dir = os.path.dirname(mockfactory.desi.__file__)
            fn = (os.path.join(dir, 'data/qso_redshift_smearing_sv1.ecsv'), os.path.join(dir, 'data/qso_redshift_smearing_sv3.ecsv'))
        return TracerRedshiftSmearing(fn=fn)

    else:
        raise ValueError(f'{tracer} redshift smearing does not exists')
=======
    z, rvs, weights, dztransform = TracerRedshiftSmearingRVS(tracer=tracer, fn=fn)
    if tracer == 'QSO':
        dzscale = 5e3
    elif tracer in ['ELG', 'BGS']:
        dzscale = 150
    elif tracer == 'LRG':
        dzscale = 200
    else:
        raise ValueError(f'{tracer} redshift smearing does not exist')
    return RVS2DRedshiftSmearing.average([RVS2DRedshiftSmearing(z, rv, dzsize=10000, dzscale=dzscale, dztransform=dztransform) for rv in rvs], weights=weights)
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9


if __name__ == '__main__':

    from argparse import ArgumentParser
    from matplotlib import pyplot as plt
    from mockfactory import setup_logging

    setup_logging()

<<<<<<< HEAD
    from argparse import ArgumentParser
=======
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9
    def parse_args():
        parser = ArgumentParser()
        parser.add_argument(
            "--tracer", help="the tracer for redshift smearing: QSO, LRG, ELG, BGS",
            type=str, default='QSO', required=True,
        )
        args = None
        args = parser.parse_args()
        return args
<<<<<<< HEAD
=======

>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9
    args = parse_args()
    tracer = args.tracer

    # Instantiate redshift smearing class
<<<<<<< HEAD
    if tracer == 'QSO':
        fn = ('data/qso_redshift_smearing_sv1.ecsv', 'data/qso_redshift_smearing_sv3.ecsv')
    else:
        fn = 'data/{}_redshift_smearing_sv1.ecsv'.format(tracer)
    rs = TracerRedshiftSmearing(tracer=tracer, fn=fn)

    # Load random variates, to get pdf to compare to
    z, rvs, weights, dztransform = RedshiftSmearingRVS(tracer=tracer, fn=fn)
=======
    rs = TracerRedshiftSmearing(tracer=tracer)

    # Load random variates, to get pdf to compare to
    z, rvs, weights, dztransform = TracerRedshiftSmearingRVS(tracer=tracer)
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9

    # z slices where to plot distributions
    lz = np.linspace(z[0], z[-1], 15)
    # Tabulated dz where to evaluate pdf
    if tracer == 'QSO':
<<<<<<< HEAD
        dzscale = 5e3
    elif (tracer == 'ELG')|(tracer == 'BGS'):
        dzscale = 150
    elif tracer == 'LRG':
        dzscale = 200
    ldz = np.linspace(-dzscale, dzscale, 1000)
=======
        dvscale = 5e3
    elif tracer in ['ELG', 'BGS']:
        dvscale = 150
    elif tracer == 'LRG':
        dvscale = 200

    #unit = 'dz'
    unit = 'dv [km/s]'
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9

    fig, lax = plt.subplots(3, 5, figsize=(20, 10))
    lax = lax.flatten()

    for zz, ax in zip(lz, lax):
        # Generate dz at given z
        dz = rs.sample(np.full(100000, zz), seed=42)
        # Array where to evaluate analytic pdf
        dvpdf = np.linspace(-dvscale, dvscale, 1000)
        # unit = 'dz'
        xmin, xmax = [dztransform(zz, dz) for dz in [-dvscale, dvscale]]
        jacpdf = 1. / dztransform(zz, 1.)
        xpdf = dztransform(zz, dvpdf)
        if 'dv' in unit:
            scale = constants.c / 1e3 / (1 + zz)
            dz *= scale
            xmin, xmax = [scale * dz for dz in [xmin, xmax]]
            jacpdf *= 1. / scale
            xpdf *= scale
        ax.hist(dz, density=True, histtype='step', color='k', bins=np.linspace(xmin, xmax, 50))
        # Compute interpolated pdf
        iz = np.searchsorted(z, zz, side='right') - 1
        alpha = (zz - z[iz]) / (z[iz + 1] - z[iz]) if iz < len(z) - 1 else 0.
        pdf = (1. - alpha) * sum(weight[iz] * rv[iz].pdf(dvpdf) for rv, weight in zip(rvs, weights))
        if alpha != 0.:
<<<<<<< HEAD
            pdf += alpha * sum(weight[iz + 1] * rv[iz + 1].pdf(ldz) for rv, weight in zip(rvs, weights))
        ax.plot(dztransform(zz, ldz), constants.c / 1e3 * (1. + zz) * pdf, color='r')
        ax.set_xlabel('$dz$')
        ax.set_xlim(dztransform(zz, -dzscale),dztransform(zz, dzscale))
=======
            pdf += alpha * sum(weight[iz + 1] * rv[iz + 1].pdf(dvpdf) for rv, weight in zip(rvs, weights))
        ax.plot(xpdf, jacpdf * pdf, color='r')
        ax.set_xlabel(f'${unit}$')
        ax.set_xlim(xmin, xmax)
>>>>>>> 93ec62f3d661730962733f3d311809ed2b8581a9

    if rs.mpicomm.rank == 0:
        plt.show()

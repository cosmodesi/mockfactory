import os
import logging

import numpy as np
from scipy import constants, stats

from mockfactory import utils
from mockfactory import RVS2DRedshiftSmearing


logger = logging.getLogger('Redshift Smearing')


def TracerRedshiftSmearingRVS(tracer='QSO', fn=None):
    """
    Return redshifts ``z``, list of continuous random variates for ``z``
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
            s0, sg = s0 / np.sqrt(2), sg / np.sqrt(2)
            rvs_nongaussian.append(stats.laplace(x0, s0))
            rvs_gaussian.append(stats.norm(x0, sg))
        else:
            sigma, x0, p, mu, la = table['val_fit'][iz]
            rvs_nongaussian.append(stats.cauchy(scale=p / 2, loc=mu))
            rvs_gaussian.append(stats.norm(scale=sigma, loc=x0))
        laz.append(la)
    laz = np.array(laz)

    if tracer == 'QSO':

        def dztransform(z, dz):
            return dz / (constants.c / 1e3) / (1. + z)  # file unit is dz (1 + z) c [km / s]

    else:

        def dztransform(z, dz):
            return dz / (constants.c / 1e3) * (1. + z)  # file unit is c dz / (1 + z) [km / s]

    return np.asarray(table['mean_z']), [rvs_nongaussian, rvs_gaussian], [laz, 1. - laz], dztransform


def TracerRedshiftSmearing(tracer='QSO', fn=None):
    """
    Return :class:`RVS2DRedshiftSmearing` instance given input tabulate file of redshift errors.
    Redshift errors can be sampled through: ``dz = rs.sample(z, seed=42)``.
    """
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


if __name__ == '__main__':

    from argparse import ArgumentParser
    from matplotlib import pyplot as plt
    from mockfactory import setup_logging

    def collect_argparser():
        parser = ArgumentParser(description="Load and display the redshift smearing for args.tracer")
        parser.add_argument("--tracer", type=str, required=True, default='QSO',
                            help="the tracer for redshift smearing: QSO, LRG, ELG, BGS")
        return parser.parse_args()

    setup_logging()
    args = collect_argparser()

    # Instantiate redshift smearing class
    rs = TracerRedshiftSmearing(tracer=args.tracer)

    # Load random variates, to get pdf to compare to
    z, rvs, weights, dztransform = TracerRedshiftSmearingRVS(tracer=args.tracer)

    # z slices where to plot distributions
    lz = np.linspace(z[0], z[-1], 15)
    # Tabulated dz where to evaluate pdf
    if args.tracer == 'QSO':
        dvscale = 5e3
    elif args.tracer in ['ELG', 'BGS']:
        dvscale = 150
    elif args.tracer == 'LRG':
        dvscale = 200

    # unit = 'dz'
    unit = 'dv [km/s]'

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
            pdf += alpha * sum(weight[iz + 1] * rv[iz + 1].pdf(dvpdf) for rv, weight in zip(rvs, weights))
        ax.plot(xpdf, jacpdf * pdf, color='r')
        ax.set_xlabel(f'${unit}$')
        ax.set_xlim(xmin, xmax)

    if rs.mpicomm.rank == 0:
        plt.tight_layout()
        plt.show()

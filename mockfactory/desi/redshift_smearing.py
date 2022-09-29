import logging

import numpy as np
from scipy import constants, stats

from mockfactory import utils
from mockfactory import RVS2DRedshiftSmearing


logger = logging.getLogger('Redshift Smearing')


def QSORedshiftSmearingRVS(fn=('data/qso_redshift_smearing_sv1.ecsv', 'data/qso_redshift_smearing_sv3.ecsv')):
    """
    Return redshifts ``z``, list of continuous random variates (Laplace and Gaussian) for ``z``,
    weights, and optional ``dz`` transform, given input tabulated file(s) of redshift errors.
    """
    from astropy.table import Table, vstack
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
    rvs_laplace, rvs_gaussian, laz = [], [], []
    for iz, z in enumerate(table['mean_z']):
        A0, x0, s0, sg, la = table['val_fit'][iz]
        rvs_laplace.append(stats.laplace(x0, s0))
        rvs_gaussian.append(stats.norm(x0, sg))
        laz.append(la)
    laz = np.array(laz)
    dztransform = lambda z, dz: dz / (constants.c / 1e3) / (1. + z)  # file units was km / s
    return np.asarray(table['mean_z']), [rvs_laplace, rvs_gaussian], [laz, 1. - laz], dztransform


def QSORedshiftSmearing(fn=('data/qso_redshift_smearing_sv1.ecsv', 'data/qso_redshift_smearing_sv3.ecsv')):
    """
    Return :class:`RVS2DRedshiftSmearing` instance given input tabulate file of redshift errors.
    Redshift errors can be sampled through: ``dz = rs.sample(z, seed=42)``.
    """
    z, rvs, weights, dztransform = QSORedshiftSmearingRVS(fn=fn)
    return RVS2DRedshiftSmearing.average([RVS2DRedshiftSmearing(z, rv, dzsize=10000, dzscale=5e3, dztransform=dztransform) for rv in rvs], weights=weights)


def RedshiftSmearing(tracer='QSO', fn=None):
    """
    Return :class:`RVS2DRedshiftSmearing` instance given the desired tracer and input tabulate file of redshift errors.
    Redshift errors can be sampled through: ``dz = rs.sample(z, seed=42)``.
    """
    if tracer in ['BGS', 'LRG', 'ELG']:
        raise ValueError(f'Redshift Smearing is not ready for {tracer} tracer')

    if tracer == 'QSO':
        if fn is None:
            import os
            import mockfactory.desi
            dir = os.path.dirname(mockfactory.desi.__file__)
            fn = (os.path.join(dir, 'data/qso_redshift_smearing_sv1.ecsv'), os.path.join(dir, 'data/qso_redshift_smearing_sv3.ecsv'))
        return QSORedshiftSmearing(fn=fn)

    else:
        raise ValueError(f'Redshift Smearing is not ready for {tracer} tracer')


if __name__ == '__main__':

    from matplotlib import pyplot as plt

    from mockfactory import setup_logging

    setup_logging()

    # Instantiate redshift smearing class
    fn = ('data/qso_redshift_smearing_sv1.ecsv', 'data/qso_redshift_smearing_sv3.ecsv')
    rs = QSORedshiftSmearing(fn=fn)

    # Load random variates, to get pdf to compare to
    z, rvs, weights, dztransform = QSORedshiftSmearingRVS(fn=fn)

    # z slices where to plot distributions
    lz = np.linspace(z[0], z[-1], 15)
    # Tabulated dz where to evaluate pdf
    ldz = np.linspace(-5e4, 5e4, 1000)

    fig, lax = plt.subplots(3, 5, figsize=(20, 10))
    lax = lax.flatten()
    for zz, ax in zip(lz, lax):
        # Generate dz at given z
        dz = rs.sample(np.full(100000, zz), seed=42)
        ax.hist(dz, density=True, histtype='step', color='k', bins=100)
        # Compute interpolated pdf
        iz = np.searchsorted(z, zz, side='right') - 1
        alpha = (zz - z[iz]) / (z[iz + 1] - z[iz]) if iz < len(z) - 1 else 0.
        pdf = (1. - alpha) * sum(weight[iz] * rv[iz].pdf(ldz) for rv, weight in zip(rvs, weights))
        if alpha != 0.:
            pdf += alpha * sum(weight[iz + 1] * rv[iz + 1].pdf(ldz) for rv, weight in zip(rvs, weights))
        ax.plot(dztransform(zz, ldz), constants.c / 1e3 * (1. + zz) * pdf, color='r')
        ax.set_xlabel('$dz$')
    if rs.mpicomm.rank == 0:
        plt.show()

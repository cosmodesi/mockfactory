import numpy as np
from scipy import constants, stats

from mockfactory.utils import BaseClass
from mockfactory import RVS2DRedshiftSmearing


def QSORedshiftSmearingRVS(fn='data/qso_redshift_smearing.ecsv'):
    """
    Return redshifts ``z``, list of continuous random variates (Laplace and Gaussian) for ``z``,
    weights, and optional ``dz`` transform, given input tabulated file of redshift errors.
    """
    from astropy.table import Table
    table = Table.read(fn)
    rvs_laplace, rvs_gaussian, laz = [], [], []
    for iz, z in enumerate(table['mean_z']):
        A0, x0, s0, sg, la = table['val_fit'][iz]
        rvs_laplace.append(stats.laplace(x0, s0))
        rvs_gaussian.append(stats.norm(x0, sg))
        laz.append(la)
    laz = np.array(laz)
    dztransform = lambda z, dz: dz / constants.c / (1. + z)
    return np.asarray(table['mean_z']), [rvs_laplace, rvs_gaussian], [laz, 1. - laz], dztransform


def QSORedshiftSmearing(fn='data/qso_redshift_smearing.ecsv'):
    """
    Return :class:`RVS2DRedshiftSmearing` instance given input tabulate file of redshift errors.
    Redshift errors can be sampled through: ``dz = rs.sample(z, seed=42)``.
    """
    z, rvs, weights, dztransform = QSORedshiftSmearingRVS(fn=fn)
    return RVS2DRedshiftSmearing.average([RVS2DRedshiftSmearing(z, rv, dzsize=10000, dzscale=5e3, dztransform=dztransform) for rv in rvs], weights=weights)


if __name__ == '__main__':

    from mockfactory import setup_logging
    from matplotlib import pyplot as plt

    setup_logging()
    fn = 'data/qso_redshift_smearing.ecsv'
    z, rvs, weights, dztransform = QSORedshiftSmearingRVS(fn=fn)
    dz = np.linspace(-5e4, 5e4, 1000)
    rs = QSORedshiftSmearing(fn=fn)

    fig, lax = plt.subplots(2, 5, figsize=(20, 10))
    lax = lax.flatten()
    for iz, zz in enumerate(z):
        s = rs.sample(np.full(100000, zz), seed=42)
        lax[iz].hist(s, density=True, histtype='step', color='k', bins=100)
        lax[iz].plot(dztransform(zz, dz), constants.c * (1. + zz) * sum(weight[iz] * rv[iz].pdf(dz) for rv, weight in zip(rvs, weights)), color='r')
        lax[iz].set_xlabel('$dz$')
    if rs.mpicomm.rank == 0:
        plt.show()

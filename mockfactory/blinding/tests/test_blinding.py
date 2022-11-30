import numpy as np
import pytest

import mpytools as mpy
from mockfactory import EulerianLinearMock, RandomBoxCatalog, setup_logging
from mockfactory.blinding import CutskyCatalogBlinding, get_cosmo_blind
from cosmoprimo.fiducial import DESI


def test_blinding():
    from pypower import CatalogFFTPower

    nmesh, boxsize, nbar, bias, z = 128, 1000., 5e-4, 2., 1.
    seed = 42
    cosmo = DESI()
    fo = cosmo.get_fourier()
    pklin = fo.pk_interpolator().to_1d(z=z)
    f = fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb')
    boxcenter = [0, 0, cosmo.comoving_radial_distance(z)]

    mock = EulerianLinearMock(pklin, nmesh=nmesh, boxsize=boxsize, boxcenter=boxcenter, seed=seed, unitary_amplitude=True)
    mock.set_real_delta_field(bias=bias)
    mock.set_rsd(f=f, los=None)

    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    data['Weight'] = mock.readout(data['Position'], field='delta', resampler='tsc', compensate=True) + 1.
    randoms = RandomBoxCatalog(nbar=5. * nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)

    cosmo_blind = get_cosmo_blind(cosmo.clone(w0_fld=-0.8, wa_fld=0.5), z=z, seed=42, params={'f': 0.05, 'fnl': 10.})
    assert 'f' in cosmo_blind._derived and 'fnl' in cosmo_blind._derived
    cosmo_blind._derived['f'] = 0.8 * f
    cosmo_blind._derived['fnl'] = -200.
    blinding = CutskyCatalogBlinding(cosmo_fid=cosmo, cosmo_blind=cosmo_blind, bias=bias, z=z)
    data_png = data.deepcopy()
    data_png['Weight'] = blinding.png(data['Position'], data_weights=data['Weight'], randoms_positions=randoms['Position'])
    data_rsd = data_png.deepcopy()
    data_rsd['Position'] = blinding.rsd(data_png['Position'], data_weights=data['Weight'], randoms_positions=randoms['Position'])
    data_ap, randoms_ap = data_rsd.deepcopy(), randoms.deepcopy()
    for catalog in [data_ap, randoms_ap]:
        catalog['Position'] = blinding.ap(catalog['Position'])

    ells = (0, 2)
    kwargs = dict(edges={'step': 0.01}, los=None, ells=ells, boxsize=2. * boxsize, boxcenter=boxcenter, nmesh=nmesh,
                  resampler='tsc', interlacing=2, position_type='pos', mpicomm=data.mpicomm)
    poles = CatalogFFTPower(data_positions1=data['Position'], data_weights1=data['Weight'], randoms_positions1=randoms['Position'], **kwargs).poles
    poles_png = CatalogFFTPower(data_positions1=data_png['Position'], data_weights1=data_png['Weight'], randoms_positions1=randoms['Position'], **kwargs).poles
    poles_rsd = CatalogFFTPower(data_positions1=data_rsd['Position'], data_weights1=data_rsd['Weight'], randoms_positions1=randoms['Position'], **kwargs).poles
    poles_ap = CatalogFFTPower(data_positions1=data_ap['Position'], data_weights1=data_ap['Weight'], randoms_positions1=randoms_ap['Position'], **kwargs).poles
    figsize = (12, 3)
    from matplotlib import pyplot as plt
    fig, lax = plt.subplots(1, len(ells), sharex=False, sharey=False, figsize=figsize, squeeze=True)
    for ax, ell in zip(lax, ells):
        ax.set_title(r'$\ell = {:d}$'.format(ell))
        ax.plot(poles.k, poles.k * poles(ell=ell), label='fid')
        ax.plot(poles_png.k, poles_png.k * poles_png(ell=ell), label='+ png')
        ax.plot(poles_rsd.k, poles_rsd.k * poles_rsd(ell=ell), label='+ rsd')
        ax.plot(poles_ap.k, poles_ap.k * poles_ap(ell=ell), label='+ ap')
        ax.set_ylabel(r'$k P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        ax.legend()
    if data.mpicomm.rank == 0:
        plt.show()


def test_misc():
    from pypower.utils import cartesian_to_sky
    boxsize, nbar, bias, z = 1000., 5e-4, 2., 1.
    seed = 42
    cosmo = DESI()
    fo = cosmo.get_fourier()
    f = fo.sigma8_z(z, of='theta_cb') / fo.sigma8_z(z, of='delta_cb')
    boxcenter = [0, 0, cosmo.comoving_radial_distance(z)]
    data = RandomBoxCatalog(nbar=nbar, boxsize=boxsize, boxcenter=boxcenter, seed=seed)
    cosmo_blind = get_cosmo_blind(cosmo, z=z)
    cosmo_blind._derived['f'] = f * 0.8
    blinding = CutskyCatalogBlinding(cosmo_fid=cosmo, cosmo_blind=cosmo_blind, bias=bias, z=z)
    blinded_positions = blinding.shuffle(data['Position'], seed=seed)
    assert not np.all(blinded_positions == data['Position']) and np.any(blinded_positions == data['Position'])
    blinded_positions = mpy.gather(blinded_positions)
    data_positions = data.cget('Position')
    data_positions = blinding.shuffle(data_positions, seed=seed, mpiroot=0)
    if data.mpicomm.rank == 0:
        assert np.allclose(data_positions, blinded_positions)
    cosmo_blind._derived['f'] = f * 1.2
    with pytest.raises(ValueError):
        data['Position'] = blinding.shuffle(data['Position'])
    cosmo_blind._derived['f'] = f * 0.8

    positions = data['Position']
    positions_gathered = data.cget('Position')
    mpicomm = data.mpicomm
    from mockfactory import DistanceToRedshift
    d2z = DistanceToRedshift(cosmo.comoving_radial_distance)

    for method, kwargs in zip(['shuffle', 'ap'], [{'seed': 42}, {}]):
        method = getattr(blinding, method)
        tmp = mpy.gather(method(positions, position_type='pos', **kwargs))
        tmp_gathered = method(positions_gathered, position_type='pos', mpiroot=0, **kwargs)
        if mpicomm.rank == 0:
            assert np.allclose(tmp, tmp_gathered)
        tmp_xyz = method(positions.T, position_type='xyz', **kwargs)
        tmp_xyz = [mpy.gather(tt) for tt in tmp_xyz]
        tmp_xyz_gathered = method(positions_gathered.T if mpicomm.rank == 0 else None, position_type='xyz', mpiroot=0, **kwargs)
        assert np.allclose(positions, data['Position'])
        if mpicomm.rank == 0:
            assert np.allclose(tmp_xyz_gathered, tmp_gathered.T)
            assert np.allclose(tmp_xyz, tmp_xyz_gathered)
        rdd = cartesian_to_sky(positions.T)
        rdd_gathered = None
        if mpicomm.rank == 0:
            rdd_gathered = cartesian_to_sky(positions_gathered.T)
        tmp_rdd = method(rdd, position_type='rdd', **kwargs)
        tmp_rdd = [mpy.gather(tt) for tt in tmp_rdd]
        tmp_rdd_gathered = method(rdd_gathered, position_type='rdd', mpiroot=0, **kwargs)
        if mpicomm.rank == 0:
            assert np.allclose(tmp_rdd_gathered, cartesian_to_sky(tmp_gathered.T), rtol=1e-6)
            assert np.allclose(tmp_rdd, tmp_rdd_gathered)
        rdz = rdd[:2] + [d2z(rdd[2])]
        rdz_gathered = None
        if mpicomm.rank == 0:
            rdz_gathered = rdd_gathered[:2] + [d2z(rdd_gathered[2])]
        tmp_rdz = method(rdz, position_type='rdz', **kwargs)
        tmp_rdz = [mpy.gather(tt) for tt in tmp_rdz]
        tmp_rdz_gathered = method(rdz_gathered, position_type='rdz', mpiroot=0, **kwargs)
        if mpicomm.rank == 0:
            assert np.allclose(tmp_rdz_gathered, list(tmp_rdd_gathered)[:2] + [d2z(tmp_rdd_gathered[2])], rtol=1e-5)
            assert np.allclose(tmp_rdz, tmp_rdz_gathered)


if __name__ == '__main__':

    setup_logging()
    test_blinding()
    test_misc()

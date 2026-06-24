import inspect

import numpy as np
import pytest


def _require_pyrecon_mpi():
    pyrecon = pytest.importorskip('pyrecon')
    signature = inspect.signature(pyrecon.IterativeFFTReconstruction.__init__)
    if 'data_positions' not in signature.parameters:
        pytest.skip('pyrecon@mpi API is required for backend comparison')
    return pyrecon


def _make_case():
    from cosmoprimo.fiducial import DESI

    z, bias, seed = 1., 2., 11
    cosmo = DESI()
    fourier = cosmo.get_fourier()
    f = fourier.sigma8_z(z, of='theta_cb') / fourier.sigma8_z(z, of='delta_cb')
    cosmo_blind = cosmo.clone()
    cosmo_blind._derived['f'] = 0.8 * f
    cosmo_blind._derived['fnl'] = 30.

    rng = np.random.RandomState(seed)
    boxcenter = np.array([0., 0., cosmo.comoving_radial_distance(z)])
    data_positions = rng.uniform(-50., 50., size=(500, 3)) + boxcenter
    randoms_positions = rng.uniform(-50., 50., size=(2500, 3)) + boxcenter
    data_weights = np.clip(1. + 0.1 * rng.normal(size=len(data_positions)), 0.5, 1.5)
    randoms_weights = np.ones(len(randoms_positions))
    kwargs = dict(nmesh=24, boxsize=140., boxcenter=boxcenter)
    return cosmo, cosmo_blind, z, bias, f, data_positions, data_weights, randoms_positions, randoms_weights, kwargs


def _pyrecon_reconstruction(pyrecon, data_positions, data_weights, randoms_positions, randoms_weights,
                            f, bias, smoothing_radius=15., **kwargs):
    return pyrecon.IterativeFFTReconstruction(data_positions=data_positions, data_weights=data_weights,
                                             randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                                             f=f, bias=bias, position_type='pos', mpiroot=None,
                                             smoothing_radius=smoothing_radius, threshold_randoms=('noise', 0.01),
                                             **kwargs)


def _pyrecon_rsd_reference(pyrecon, data_positions, data_weights, randoms_positions, randoms_weights,
                           f, f_blind, bias, smoothing_radius=15., **kwargs):
    recon = _pyrecon_reconstruction(pyrecon, data_positions, data_weights, randoms_positions, randoms_weights,
                                    f=f, bias=bias, smoothing_radius=smoothing_radius, **kwargs)
    shifts = recon.read_shifts(data_positions, position_type='pos', mpiroot=None, field='rsd')
    return data_positions + (f_blind / recon.f - 1.) * shifts


def _pyrecon_png_reference(pyrecon, cosmo, cosmo_blind, z, data_positions, data_weights,
                           randoms_positions, randoms_weights, f, bias, smoothing_radius=30.,
                           shotnoise_correction=False, **kwargs):
    recon = _pyrecon_reconstruction(pyrecon, data_positions, data_weights, randoms_positions, randoms_weights,
                                    f=f, bias=bias, smoothing_radius=smoothing_radius, **kwargs)
    sigma1 = recon.smoothing_radius
    shifts = recon.read_shifts(data_positions, position_type='pos', mpiroot=None, field='rsd')
    shifted_positions = data_positions - shifts

    recon.mesh_data = recon.mesh_randoms = None
    recon.assign_data(shifted_positions, weights=data_weights, position_type='pos', mpiroot=None)
    recon.assign_randoms(randoms_positions, weights=randoms_weights, position_type='pos', mpiroot=None)
    recon.set_density_contrast(smoothing_radius=smoothing_radius)

    sigma2 = recon.smoothing_radius
    mesh = recon.mesh_delta.r2c()
    b1 = recon.bias
    bfnl = 2 * 1.686 * (b1 - 1.) * cosmo_blind._derived.get('fnl', 0.)

    pk_prim = cosmo.get_primordial().pk_interpolator(mode='scalar')
    pk_lin = cosmo.get_fourier().pk_interpolator(of='theta_cb').to_1d(z=z)

    def Tk(k):
        pphi_prim = 9 / 25 * 2 * np.pi**2 / k**3 * pk_prim(k) / cosmo.h**3
        return (pk_lin(k) / pphi_prim)**0.5

    for kslab, slab in zip(mesh.slabs.x, mesh.slabs):
        k = sum(kk.real**2 for kk in kslab)**0.5
        nonzero = k != 0.
        slab[nonzero] *= bfnl / Tk(k[nonzero])
        slab[~nonzero] = 0.

    if shotnoise_correction:

        def S1(k):
            return np.exp(- 0.5 * k**2 * sigma1**2)

        def S2(k):
            return np.exp(- 0.5 * k**2 * sigma2**2)

        recon.mesh_data = None
        recon.assign_data(data_positions, weights=data_weights * data_weights, position_type='pos', mpiroot=None)
        sum_w2 = recon.mesh_data

        recon.mesh_data = None
        recon.assign_data(data_positions, weights=data_weights, position_type='pos', mpiroot=None)
        sum_wd = recon.mesh_data

        recon.mesh_data = None
        recon.assign_data(randoms_positions, weights=randoms_weights, position_type='pos', mpiroot=None)
        alpha = np.sum(data_weights) / np.sum(randoms_weights)
        nbar = alpha / np.prod(recon.cellsize) * recon.mesh_data

        sum_w2[sum_w2 == 0.] = 1.
        inv_shotnoise = recon._smooth_gaussian(sum_wd * nbar / sum_w2)
        shotnoise = 1 / recon._readout(inv_shotnoise, randoms_positions)

        mu_pivot = 0.6
        k_pivot = 4e-3 if bfnl >= 0 else 8e-3
        mask = S1(pk_lin.k) > 1e-4
        sigma_d_2 = pk_lin.clone(k=pk_lin.k[mask], pk=(S1(pk_lin.k)**2 * pk_lin(pk_lin.k))[mask]).sigma_d()**2
        X_tilde = (b1 + f * mu_pivot**2) * (b1 + (1. - S1(k_pivot)) * f * mu_pivot**2) * S2(k_pivot) * pk_lin(k_pivot) + S2(k_pivot) * shotnoise * np.exp(- 0.5 * k_pivot**2 * mu_pivot**2 * f**2 * sigma_d_2)
        Y_tilde = (b1 + (1. - S1(k_pivot)) * f * mu_pivot**2)**2 * S2(k_pivot)**2 * pk_lin(k_pivot) + S2(k_pivot)**2 * shotnoise
        expected_pivot = 2 * bfnl / Tk(k_pivot) * b1 * (b1 + f * mu_pivot**2) * pk_lin(k_pivot) + (bfnl / Tk(k_pivot))**2 * b1**2 * pk_lin(k_pivot)
        shotnoise_factor = (- X_tilde + np.sqrt(X_tilde**2 + Y_tilde * expected_pivot)) / Y_tilde / (bfnl / Tk(k_pivot))
    else:
        shotnoise_factor = 1.

    weights = recon._readout(mesh.c2r(), randoms_positions)
    return randoms_weights * (1. - shotnoise_factor * weights)


def test_jax_recon_matches_pyrecon_for_blinding():
    from jax import config
    config.update('jax_enable_x64', True)

    pytest.importorskip('jaxrecon')
    pyrecon = _require_pyrecon_mpi()

    from mockfactory.blinding import CutskyCatalogBlinding

    cosmo, cosmo_blind, z, bias, f, data_positions, data_weights, randoms_positions, randoms_weights, kwargs = _make_case()
    blinding = CutskyCatalogBlinding(cosmo_fid=cosmo, cosmo_blind=cosmo_blind, bias=bias, z=z, dtype='f8')

    rsd_jax = blinding.rsd(data_positions, data_weights=data_weights, randoms_positions=randoms_positions,
                           randoms_weights=randoms_weights, **kwargs)
    rsd_pyrecon = _pyrecon_rsd_reference(pyrecon, data_positions, data_weights, randoms_positions, randoms_weights,
                                         f=f, f_blind=cosmo_blind._derived['f'], bias=bias, **kwargs)
    assert np.allclose(rsd_jax, rsd_pyrecon, atol=1e-10, rtol=1e-10)

    for shotnoise_correction in [False, True]:
        weights_jax = blinding.png(data_positions, data_weights=data_weights, randoms_positions=randoms_positions,
                                   randoms_weights=randoms_weights, shotnoise_correction=shotnoise_correction, **kwargs)
        weights_pyrecon = _pyrecon_png_reference(pyrecon, cosmo, cosmo_blind, z, data_positions, data_weights,
                                                 randoms_positions, randoms_weights, f=f, bias=bias,
                                                 shotnoise_correction=shotnoise_correction, **kwargs)
        assert np.allclose(weights_jax, weights_pyrecon, atol=1e-10, rtol=1e-10)

"""Utilities to blind cutsky catalogs."""

import numpy as np

import mpytools as mpy
from mpytools import CurrentMPIComm

from mockfactory.utils import BaseClass
from mockfactory import utils


def get_cosmo(cosmo):
    """
    Return :class:`cosmoprimo.Cosmology` instance given either such an instance, a string (name of fiducial cosmology),
    a tuple (name, dictionary of parameters) or a dictionary of parameters.
    """
    if cosmo is None:
        return cosmo
    import cosmoprimo
    if isinstance(cosmo, cosmoprimo.Cosmology):
        return cosmo
    if isinstance(cosmo, str):
        cosmo = (cosmo, {})
    if isinstance(cosmo, tuple):
        return getattr(cosmoprimo.fiducial, cosmo[0])(**cosmo[1])
    return cosmoprimo.Cosmology(**cosmo)


def _get_from_cosmo(cosmo, name, z=None):
    # Return quantity of name ``name`` from input :class:`cosmoprimo.Cosmology` instance ``©osmo``:
    # Omega's, quantity in ``cosmo._derived`` (typically precomputed f or fnl), 'f' or 'fnl'
    def check_z():
        if z is None:
            raise ValueError('z is None!')

    if name.startswith('omega'):
        return _get_from_cosmo(cosmo, 'O' + name[1:], z=z) * cosmo.h ** 2
    if name.startswith('Omega'):
        if z is None:
            name = name[:5] + '0' + name[5:]
            return getattr(cosmo, name)
        check_z()
        return getattr(cosmo, name)(z)
    if name in cosmo._derived:
        return cosmo._derived[name]
    if name == 'fsigma8':
        check_z()
        return cosmo.get_fourier().sigma8_z(z, of='theta_cb')
    if name == 'sigma8':
        check_z()
        return cosmo.get_fourier().sigma8_z(z, of='delta_cb')
    if name == 'f':
        return _get_from_cosmo(cosmo, 'fsigma8', z=z) / _get_from_cosmo(cosmo, 'sigma8', z=z)
    if name == 'fnl':
        return 0.
    return getattr(cosmo, name)


def get_cosmo_blind(cosmo_fid, seed=42, params=None, z=None):
    """
    Generate blind cosmology from input fiducial cosmology ``cosmo_fid``.

    Parameters
    ----------
    cosmo_fid : string, tuple, dict, cosmoprimo.Cosmology
        Fiducial cosmology (see :func:`get_cosmo`).

    seed : int, default=42
        Random seed, for reproducibility.

    params : dict, default={'w0_fld': 0.05, 'wa_fld': 0.2, 'Omega_m': 0.02, 'f': 0.05, 'fnl': 10}
        Dictionary of cosmological parameters to blind and associated uncertainty.
        Blinded parameters will be drawn with a uniform distribution of width the provided uncertainty
        on each side of the fiducial parameter value.

    z : float, default=None
        Redshift at which the growth rate f must be computed, if f is to be blinded.
        See e.g. :func:`get_z` to compute it.

    Returns
    -------
    cosmo_blind : cosmoprimo.Cosmology
        Blinded cosmology, with ``f`` and ``fnl`` in ``cosmo_blind._derived`` if these are blinded parameters.
    """
    if params is None:
        params = {'w0_fld': 0.05, 'wa_fld': 0.2, 'Omega_m': 0.02, 'f': 0.05, 'fnl': 10}
    cosmo_fid = get_cosmo(cosmo_fid)
    rng = np.random.RandomState(seed=seed)
    update, derived = {}, {}
    for param, sigma in params.items():
        tmp = _get_from_cosmo(cosmo_fid, param, z=z) + 2 * sigma * (rng.uniform() - 0.5)
        if param in ['f', 'fnl']:
            derived[param] = tmp
        else:
            update[param] = tmp
    cosmo_blind = cosmo_fid.clone(**update)
    cosmo_blind._derived.update(derived)
    return cosmo_blind


@CurrentMPIComm.enable
def get_z(positions, weights=None, position_type='pos', cosmo=None, mpicomm=None, mpiroot=None):
    """
    Compute effective redshift.

    Parameters
    ----------
    positions : list, array
        Positions. See ``position_type``.

    weights : array, default=None
        Optionally, weights.

    position_type : string, default='pos'
        Type of input positions, one of:

            - "pos": Cartesian positions of shape (N, 3)
            - "xyz": Cartesian positions of shape (3, N)
            - "rdd": RA/Dec in degree, distance of shape (3, N)
            - "rdz": RA/Dec in degree, redshift of shape (3, N)

        This can be overriden for each of the blinding methods.

    cosmo : cosmoprimo.Cosmology
        Cosmology to transform redshifts to distances, required in case ``position_type`` is not "rdz".
        ``cosmo.comoving_radial_distance`` only is used.

    mpiroot : int, default=None
        If ``None``, input positions and weights are assumed to be scattered across all ranks.
        Else the MPI rank where input positions and weights are gathered.

    mpicomm : MPI communicator, default=mpi.COMM_WORLD
        The MPI communicator.
    """
    d2z = position_type != 'rdz'
    if not d2z:
        position_type = 'rdd'  # no need to apply z -> d transform
    positions = _format_positions(positions, position_type=position_type, mpicomm=mpicomm, mpiroot=mpiroot)
    dist, ra, dec = utils.cartesian_to_sky(positions)
    if d2z:
        from mockfactory import DistanceToRedshift
        d2z = DistanceToRedshift(cosmo.comoving_radial_distance)
        dist = d2z(dist)
    return mpy.caverage(dist, weights=weights, mpicomm=mpicomm)


def _format_positions(positions, position_type='xyz', dtype=None, copy=True, cosmo=None, mpicomm=None, mpiroot=None):
    # Format input array of positions
    # position_type in ["xyz", "rdd", "pos"]

    def __format_positions(positions):
        if position_type == 'pos':  # array of shape (N, 3)
            positions = np.array(positions, dtype=dtype) if copy else np.asarray(positions, dtype=dtype)
            if not np.issubdtype(positions.dtype, np.floating):
                return None, 'Input position arrays should be of floating type, not {}'.format(positions.dtype)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            return positions, None
        # Array of shape (3, N)
        positions = list(positions)
        for ip, p in enumerate(positions):
            # Cast to the input dtype if exists (may be set by previous positions)
            positions[ip] = np.array(p, dtype=dtype) if copy else np.asarray(p, dtype=dtype)
        size = len(positions[0])
        dt = positions[0].dtype
        if not np.issubdtype(dt, np.floating):
            return None, 'Input position arrays should be of floating type, not {}'.format(dt)
        for p in positions[1:]:
            if len(p) != size:
                return None, 'All position arrays should be of the same size'
            if p.dtype != dt:
                return None, 'All position arrays should be of the same type, you can e.g. provide dtype'
        if len(positions) != 3:
            return None, 'For position type = {}, please provide a list of 3 arrays for positions (found {:d})'.format(position_type, len(positions))
        if position_type == 'rdd':  # RA, Dec, distance
            positions = utils.sky_to_cartesian(positions[2], *positions[:2], degree=True).T
        elif position_type == 'rdz':  # RA, Dec, Z
            positions = utils.sky_to_cartesian(cosmo.comoving_radial_distance(positions[2]), *positions[:2], degree=True).T
        elif position_type != 'xyz':
            return None, 'Position type should be one of ["pos", "xyz", "rdz", "rdd"]'
        return np.asarray(positions).T, None

    error = None
    if mpiroot is None or (mpicomm.rank == mpiroot):
        if positions is not None and (position_type == 'pos' or not all(position is None for position in positions)):
            positions, error = __format_positions(positions)  # return error separately to raise on all processes
    if mpicomm is not None:
        error = mpicomm.allgather(error)
    else:
        error = [error]
    errors = [err for err in error if err is not None]
    if errors:
        raise ValueError(errors[0])
    if mpiroot is not None and mpicomm.bcast(positions is not None if mpicomm.rank == mpiroot else None, root=mpiroot):
        positions = mpy.scatter(positions, mpicomm=mpicomm, mpiroot=mpiroot)
    return positions


def _format_weights(weights, size=None, dtype=None, copy=True, mpicomm=None, mpiroot=None):
    # Format input weights.
    def __format_weights(weights):
        if weights is None:
            return weights
        weights = weights.astype(dtype, copy=copy)
        return weights

    weights = __format_weights(weights)
    if mpiroot is None:
        is_none = mpicomm.allgather(weights is None)
        if any(is_none) and not all(is_none):
            raise ValueError('mpiroot = None but weights are None on some ranks')
    else:
        weights = mpy.scatter(weights, mpicomm=mpicomm, mpiroot=mpiroot)

    if size is not None and weights is not None and len(weights) != size:
        raise ValueError('Weight arrays should be of the same size as position arrays')
    return weights


def _format_output_positions(positions, position_type='pos', cosmo=None, mpicomm=None, mpiroot=None):
    # Transform output posiitons to input format (position_type and gathered or not)
    toret = positions
    if mpiroot is not None:  # positions returned, gather on the same rank
        toret = mpy.gather(toret, mpicomm=mpicomm, mpiroot=mpiroot)
    if toret is not None:
        if position_type == 'rdz':
            dist, ra, dec = utils.cartesian_to_sky(toret)
            from mockfactory import DistanceToRedshift
            dist = DistanceToRedshift(cosmo.comoving_radial_distance)(dist)
            toret = [ra, dec, dist]
        elif position_type == 'rdd':
            dist, ra, dec = utils.cartesian_to_sky(toret)
            toret = [ra, dec, dist]
        elif position_type == 'xyz':
            toret = toret.T
    return toret


OPT = '-fopenmp -pedantic -Wall -Wextra -O3 -std=c99'


def _format_output_weights(weights, mpicomm=None, mpiroot=None):
    # Transform output weights to input format (position_type and gathered or not)
    toret = weights
    if mpiroot is not None and toret is not None:  # positions returned, gather on the same rank
        toret = mpy.gather(toret, mpicomm=mpicomm, mpiroot=mpiroot)
    return toret


def _gather_to_all(array, mpicomm=None):
    """Gather an MPI-scattered array and broadcast the gathered copy."""
    if array is None or mpicomm is None or mpicomm.size == 1:
        return array
    array = mpy.gather(array, mpicomm=mpicomm, mpiroot=0)
    if mpicomm.rank == 0:
        array = np.asarray(array)
    return mpicomm.bcast(array, root=0)


def _pop_recon_kwargs(kwargs, default_cellsize):
    kwargs = dict(kwargs)
    if not any(name in kwargs for name in ['nmesh', 'meshsize', 'cellsize']):
        kwargs['cellsize'] = default_cellsize

    mesh_kwargs = {}
    if 'nmesh' in kwargs:
        mesh_kwargs['meshsize'] = kwargs.pop('nmesh')
    for name in ['meshsize', 'boxsize', 'boxcenter', 'cellsize', 'boxpad', 'check', 'approximate',
                 'dtype', 'primes', 'divisors', 'sharding_mesh', 'fft_backend']:
        if name in kwargs:
            mesh_kwargs[name] = kwargs.pop(name)

    recon_kwargs = {}
    for name, default in [('los', None), ('resampler', 'cic'), ('halo_add', 0),
                          ('threshold_randoms', ('noise', 0.01)), ('niterations', 3)]:
        recon_kwargs[name] = kwargs.pop(name, default)

    if kwargs:
        raise TypeError('Unknown reconstruction keyword argument(s): {}'.format(', '.join(sorted(kwargs))))
    return mesh_kwargs, recon_kwargs


def _get_reconstruction_class(recon):
    from jaxrecon import zeldovich
    if not isinstance(recon, str):
        return recon
    try:
        cls = getattr(zeldovich, recon)
    except AttributeError as exc:
        raise ValueError('Unknown jax-recon reconstruction {!r}'.format(recon)) from exc
    return cls


def _make_particle_field(positions, weights=None, attrs=None, mpicomm=None):
    from jaxpower import ParticleField
    positions = _gather_to_all(positions, mpicomm=mpicomm)
    weights = _gather_to_all(weights, mpicomm=mpicomm)
    return ParticleField(positions, weights, attrs=attrs)


def _build_reconstruction(data_positions, data_weights=None, randoms_positions=None, randoms_weights=None,
                          f=None, bias=None, smoothing_radius=15., dtype=None, mpicomm=None,
                          default_cellsize=7., recon='IterativeFFTReconstruction', **kwargs):
    from jaxpower import FKPField, get_mesh_attrs

    ReconstructionAlgorithm = _get_reconstruction_class(recon)
    mesh_kwargs, recon_kwargs = _pop_recon_kwargs(kwargs, default_cellsize=default_cellsize)
    if dtype is not None:
        mesh_kwargs.setdefault('dtype', dtype)

    data_positions_all = _gather_to_all(data_positions, mpicomm=mpicomm)
    randoms_positions_all = _gather_to_all(randoms_positions, mpicomm=mpicomm)
    positions = [pos for pos in [data_positions_all, randoms_positions_all] if pos is not None]
    attrs = get_mesh_attrs(*positions, **mesh_kwargs)

    data = _make_particle_field(data_positions, data_weights, attrs=attrs, mpicomm=mpicomm)
    randoms = None
    if randoms_positions is not None:
        randoms = _make_particle_field(randoms_positions, randoms_weights, attrs=attrs, mpicomm=mpicomm)
    particles = FKPField(data, randoms, attrs=attrs) if randoms is not None else data
    kwargs_recon = dict(resampler=recon_kwargs['resampler'], halo_add=recon_kwargs['halo_add'],
                        smoothing_radius=smoothing_radius, threshold_randoms=recon_kwargs['threshold_randoms'])
    if ReconstructionAlgorithm.__name__ in ['IterativeFFTReconstruction', 'IterativeFFTParticleReconstruction']:
        kwargs_recon['niterations'] = recon_kwargs['niterations']
    recon = ReconstructionAlgorithm(particles, growth_rate=f, bias=bias, los=recon_kwargs['los'], **kwargs_recon)
    return recon, attrs, recon_kwargs


def _paint_particles(positions, weights=None, attrs=None, resampler='cic', halo_add=0, mpicomm=None):
    particles = _make_particle_field(positions, weights=weights, attrs=attrs, mpicomm=mpicomm)
    mesh = particles.paint(resampler=resampler, compensate=False, interlacing=0, halo_add=halo_add, out='real')
    return mesh, particles


def _get_threshold_randoms(randoms, threshold_randoms=0.01):
    if randoms is None or threshold_randoms is None:
        return None
    if isinstance(threshold_randoms, tuple):
        threshold_method, threshold_value = threshold_randoms
    else:
        threshold_method, threshold_value = 'noise', threshold_randoms
    if threshold_method not in ['noise', 'mean']:
        raise ValueError('threshold_randoms method must be "noise" or "mean"')
    if threshold_method == 'noise':
        return threshold_value * (randoms.weights**2).sum() / randoms.sum()
    return threshold_value * randoms.sum() / randoms.size


def _density_contrast(mesh_data, mesh_randoms=None, randoms=None, bias=1., smoothing_radius=15., threshold_randoms=0.01):
    from jaxrecon.zeldovich import estimate_mesh_delta
    threshold_randoms = _get_threshold_randoms(randoms, threshold_randoms=threshold_randoms)
    return estimate_mesh_delta(mesh_data, mesh_randoms=mesh_randoms, threshold_randoms=threshold_randoms,
                               smoothing_radius=smoothing_radius) / bias


def _apply_png_transfer(mesh, bfnl, Tk):
    """Apply PNG transfer function to a jaxpower complex mesh."""
    import jax.numpy as jnp
    k = sum(np.asarray(kk)**2 for kk in mesh.attrs.kcoords(sparse=True))**0.5
    transfer = np.zeros(k.shape, dtype=np.asarray(mesh.value).real.dtype)
    nonzero = k != 0.
    transfer[nonzero] = bfnl / Tk(k[nonzero])
    return mesh * jnp.asarray(transfer)


def _replace_mesh_zeros(mesh):
    """Replace zero mesh values by one."""
    import jax.numpy as jnp
    return mesh.clone(value=jnp.where(mesh.value == 0., 1., mesh.value))


def _smooth_mesh(mesh, smoothing_radius=15.):
    from jaxrecon.zeldovich import kernel_gaussian
    return (mesh.r2c() * kernel_gaussian(mesh.attrs, smoothing_radius=smoothing_radius)).c2r()


def _read_mesh(mesh, positions, resampler='cic', halo_add=0):
    return np.asarray(mesh.read(positions, resampler=resampler, compensate=False, halo_add=halo_add))


def _gradient_shifts(mesh, positions, resampler='cic', halo_add=0):
    """Return gradient readouts from a complex jaxpower mesh."""
    import jax.numpy as jnp
    kcoords = mesh.attrs.kcoords(sparse=True)
    k2 = sum(kk**2 for kk in kcoords)
    k2 = jnp.where(k2 == 0., 1., k2)
    disps = []
    for iaxis in range(mesh.attrs.ndim):
        psi = (mesh * (1j * kcoords[iaxis] / k2)).c2r()
        disps.append(_read_mesh(psi, positions, resampler=resampler, halo_add=halo_add))
    return np.column_stack(disps)


class CutskyCatalogBlinding(BaseClass):
    """
    Apply catalog-level blinding. A typical blinding procedure would be:

    .. code-block:: python

        cosmo_fid = 'DESI'
        cosmo_blind = get_cosmo_blind(cosmo_fid)
        # position_type = 'pos' ((N, 3) Cartesian positions), 'xyz' ((3, N) Cartesian positions), 'rdd' (RA, DEC, distance), 'rdz' (RA, DEC, Z)
        blinding = CutskyCatalogBlinding(cosmo_fid=cosmo_fid, cosmo_blind=cosmo_blind, bias=1.4, z=1.5, position_type='rdz')
        # data_png_weights are data_weights modified to include (local) PNG blinding
        randoms_png_weights = blinding.png(data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights)
        # For RSD blinding use randoms_weights instead of randoms_png_weights to avoid coupling between png and rsd blinding
        # (though this should not be too problematic if blinded fnl is not unrealistic)
        data_positions = blinding.rsd(data_positions, data_weights=data_weights, randoms_positions=randoms_positions, randoms_weights=randoms_weights)
        # Alcock-Paczynski-type blinding
        data_positions, randoms_positions = blinding.ap(data_positions), blinding.ap(randoms_positions)
        # Blinded output is:
        # - data: data_positions, data_weights
        # - randoms: randoms_positions, randoms_png_weights

    Note
    ----
    :meth:`rsd` and :meth:`png` require ``jax-recon``.
    """
    @CurrentMPIComm.enable
    def __init__(self, cosmo_fid='DESI', cosmo_blind='DESI', bias=None, z=None, position_type='pos', dtype=None, mpiroot=None, mpicomm=None):
        """
        Initialize :class:`CutskyCatalogBlinding`.

        Parameters
        ----------
        cosmo_fid : string, tuple, dict, cosmoprimo.Cosmology
            Fiducial cosmology (see :func:`get_cosmo`).

        cosmo_blind : cosmoprimo.Cosmology
            Blinded cosmology (see :func:`get_cosmo_blind`).

        bias : float, default=None
            Tracer bias. This is required to apply either :meth:`png` or :meth:`rsd` blinding.

        z : float, default=None
            Effective redshift. This is required to compute the fiducial growth rate f, when applying either :meth:`png` or :meth:`rsd` blinding.
            See e.g. :func:`get_z` to compute it.

        position_type : string, default='pos'
            Type of input positions, one of:

                - "pos": Cartesian positions of shape (N, 3)
                - "xyz": Cartesian positions of shape (3, N)
                - "rdd": RA/Dec in degree, distance of shape (3, N)
                - "rdz": RA/Dec in degree, redshift of shape (3, N)

            This can be overriden for each of the blinding methods.

        mpiroot : int, default=None
            If ``None``, input positions and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input positions and weights are gathered.

        mpicomm : MPI communicator, default=mpi.COMM_WORLD
            The MPI communicator.
        """
        self.mpicomm = mpicomm
        self.cosmo_fid = get_cosmo(cosmo_fid)
        self.cosmo_blind = get_cosmo(cosmo_blind)
        self.bias = bias
        self.z = z
        self.position_type = position_type
        self.mpiroot = mpiroot
        self.dtype = dtype

    def ap(self, positions, **kwargs):
        """
        Apply Alcock-Paczynski-type blinding.

        Parameters
        ----------
        positions : list, array
            Positions, of shape (N, 3) or (3, N) depending on :attr:`position_type`.

        kwargs : dict
            ``position_type``, ``mpiroot`` can be provided to override default :attr:`position_type`, :attr:`mpiroot`.

        Returns
        -------
        positions : list, array
            AP-blinded positions, of same type as input.
        """
        position_type = kwargs.pop('position_type', self.position_type)
        mpiroot = kwargs.pop('mpiroot', self.mpiroot)
        d2z = position_type != 'rdz'
        # No need to apply z -> d transform if position_type == 'rdz'
        positions = _format_positions(positions, position_type=position_type if d2z else 'rdd', dtype=self.dtype, mpicomm=self.mpicomm, mpiroot=mpiroot)
        dist, ra, dec = utils.cartesian_to_sky(positions)
        if d2z:
            from mockfactory import DistanceToRedshift
            d2z = DistanceToRedshift(self.cosmo_fid.comoving_radial_distance)
            z = d2z(dist)
        else:
            z = dist
        blind_dist = self.cosmo_blind.comoving_radial_distance(z)
        positions = positions * blind_dist[..., None] / dist[..., None]
        return _format_output_positions(positions, position_type=position_type, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)

    def shuffle(self, positions, seed=None, **kwargs):
        """
        Shuffle some fraction of redshifts / distances, to decrease power and hence bling the growth rate f.
        For this method to be applicable, the blinded f must be lower than the fiducial f.

        Parameters
        ----------
        positions : list, array
            Positions, of shape (N, 3) or (3, N) depending on :attr:`position_type`.

        seed : int, default=None
            Random seed.

        kwargs : dict
            ``position_type``, ``mpiroot`` can be provided to override default :attr:`position_type`, :attr:`mpiroot`.

        Returns
        -------
        positions : list, array
            Positions with shuffled redshifts / distances, of same type as input.
        """
        position_type = kwargs.pop('position_type', self.position_type)
        mpiroot = kwargs.pop('mpiroot', self.mpiroot)
        isrdd = position_type in ['rdd', 'rdz']
        if isrdd: position_type = 'xyz'
        positions = _format_positions(positions, position_type=position_type, dtype=self.dtype, copy=True, mpicomm=self.mpicomm, mpiroot=mpiroot)
        size = len(positions)
        frac = (_get_from_cosmo(self.cosmo_blind, 'f') / _get_from_cosmo(self.cosmo_fid, 'f', z=self.z)) ** 2
        if frac > 1.:
            raise ValueError('Blinded f is greater than fiducial f, cannot apply shuffle()')
        rng = mpy.random.MPIRandomState(size, seed=seed)
        mask = rng.uniform() < frac
        if isrdd:
            dist_masked = positions[mask, 2]
        else:
            dist_masked = utils.distance(positions[mask])
        # Let's gather the redshifts / distances to shuffle on the first rank
        # Not found anything smarter yet
        dist_masked_shuffled = mpy.gather(dist_masked, mpicomm=self.mpicomm, mpiroot=0)
        if self.mpicomm.rank == 0:
            rng = np.random.RandomState(seed=seed)
            rng.shuffle(dist_masked_shuffled)
        dist_masked_shuffled = mpy.scatter(dist_masked_shuffled, size=mask.sum(), mpicomm=self.mpicomm, mpiroot=0)
        if isrdd:
            positions[mask, 2] = dist_masked_shuffled
        else:
            positions[mask, ...] *= dist_masked_shuffled[..., None] / dist_masked[..., None]
        return _format_output_positions(positions, position_type=position_type, mpicomm=self.mpicomm, mpiroot=mpiroot)

    def rsd(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None,
            recon='IterativeFFTReconstruction', smoothing_radius=15., **kwargs):
        """
        Apply RSD blinding, changing RSD displacements of input positions according to blinded f.

        Parameters
        ----------
        data_positions : list, array
            Data positions, of shape (N, 3) or (3, N) depending on :attr:`position_type`.

        data_weights : array, default=None
            Optionally, data weights.

        randoms_positions : list, array, default=None.
            Optionally, randoms positions, of shape (N, 3) or (3, N) depending on :attr:`position_type`.

        randoms_weights : array, default=None
            Optionally, randoms weights.

        recon : str, default='IterativeFFTReconstruction'
            Name of jax-recon reconstruction algorithm.

        smoothing_radius : float, default=15.
            Smoothing radius for reconstruction.

        kwargs : dict
            Optionally, reconstruction parameters: ``cellsize`` (defaults to 7.), etc.
            ``position_type``, ``mpiroot`` can be provided to override default :attr:`position_type`, :attr:`mpiroot`.

        Returns
        -------
        data_positions : list, array
            Data positions with blinded RSD, of same type as input.
        """
        position_type = kwargs.pop('position_type', self.position_type)
        mpiroot = kwargs.pop('mpiroot', self.mpiroot)
        data_positions = _format_positions(data_positions, position_type=position_type, dtype=self.dtype, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)
        data_weights = _format_weights(data_weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
        randoms_positions = _format_positions(randoms_positions, position_type=position_type, dtype=self.dtype, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)
        randoms_weights = _format_weights(randoms_weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
        f = _get_from_cosmo(self.cosmo_fid, 'f', z=self.z)
        recon, attrs, recon_kwargs = _build_reconstruction(data_positions, data_weights=data_weights,
                                                           randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                                                           f=f, bias=self.bias, smoothing_radius=smoothing_radius,
                                                           dtype=self.dtype, mpicomm=self.mpicomm, default_cellsize=7.,
                                                           recon=recon, **kwargs)
        del attrs, recon_kwargs
        shifts = np.asarray(recon.read_shifts(data_positions, field='rsd'))
        f_blind = _get_from_cosmo(self.cosmo_blind, 'f')
        # Change RSD displacements depending on blind f
        data_positions = data_positions + (f_blind / f - 1.) * shifts
        return _format_output_positions(data_positions, position_type=position_type, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)

    def png(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, method='randoms_weights',
            recon='IterativeFFTReconstruction', smoothing_radius=30., shotnoise_correction=False, **kwargs):
        r"""
        Apply local primordial non-Gaussianity blinding, computing weights to apply scale-dependent bias on large scales.
        The rationale is to change the real-space Fourier galaxy density contrast: :math:`b_{1} \delta(\mathbf{k})` such that it becomes
        :math:`(b_{1} + b_{\phi} f_{NL}^{\mathrm{loc}} \alpha(k)) \delta(\mathbf{k})`.
        The real-space Fourier density contrast :math:`\delta(\mathbf{k})` is obtained through reconstruction,
        and we return for each random (resp. data) point the weight :math:`1 - w_{NL}` (resp. :math:`1 + w_{NL}`)
        where :math:`w_{NL} = b_{\phi} f_{NL}^{\mathrm{loc}} \alpha \delta` (transformed in configuration space).

        Parameters
        ----------
        data_positions : list, array
            Data positions, of shape (N, 3) or (3, N) depending on :attr:`position_type`.

        data_weights : array, default=None
            Optionally, data weights.

        randoms_positions : list, array, default=None.
            Optionally, randoms positions, of shape (N, 3) or (3, N) depending on :attr:`position_type`.

        randoms_weights : array, default=None
            Optionally, randoms weights.

        method : str, default='randoms_weights'
            If 'randoms_weights', apply weights to randoms.
            If 'data_weigths', apply weights to data.

        recon : str, default='IterativeFFTReconstruction'
            Name of jax-recon reconstruction algorithm.

        smoothing_radius : float, default=30.
            Smoothing radius for reconstruction. Larger than for RSD blinding, as we only need large scale RSD to be resolved.

        shotnoise_correction : bool, default=False
            If ``True``, apply shotnoise correction to avoid excess of power at large scales. Requires randoms to work.

        kwargs : dict
            Optionally, reconstruction parameters: ``cellsize`` (defaults to 15.), etc.
            ``position_type``, ``mpiroot`` can be provided to override default :attr:`position_type`, :attr:`mpiroot`.

        Returns
        -------
        randoms_weights : array
            Randoms weights, including blinded PNG signal.
        """
        available_methods = ['data_weights', 'randoms_weights', 'data_positions', 'randoms_positions']
        if method not in available_methods:
            raise ValueError('blinding method {} must be one of {}'.format(method, available_methods))
        position_type = kwargs.pop('position_type', self.position_type)
        mpiroot = kwargs.pop('mpiroot', self.mpiroot)
        data_positions = _format_positions(data_positions, position_type=position_type, dtype=self.dtype, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)
        data_weights = _format_weights(data_weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
        randoms_positions = _format_positions(randoms_positions, position_type=position_type, dtype=self.dtype, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)
        randoms_weights = _format_weights(randoms_weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
        f = _get_from_cosmo(self.cosmo_fid, 'f', z=self.z)
        recon, attrs, recon_kwargs = _build_reconstruction(data_positions, data_weights=data_weights,
                                                           randoms_positions=randoms_positions, randoms_weights=randoms_weights,
                                                           f=f, bias=self.bias, smoothing_radius=smoothing_radius,
                                                           dtype=self.dtype, mpicomm=self.mpicomm, default_cellsize=15.,
                                                           recon=recon, **kwargs)
        resampler, halo_add = recon_kwargs['resampler'], recon_kwargs['halo_add']
        threshold_randoms = recon_kwargs['threshold_randoms']
        sigma1 = smoothing_radius
        shifts = np.asarray(recon.read_shifts(data_positions, field='rsd'))
        shifted_positions = data_positions - shifts
        mesh_data, _ = _paint_particles(shifted_positions, weights=data_weights, attrs=attrs,
                                            resampler=resampler, halo_add=halo_add, mpicomm=self.mpicomm)
        mesh_randoms, randoms = None, None
        if randoms_positions is not None:
            mesh_randoms, randoms = _paint_particles(randoms_positions, weights=randoms_weights, attrs=attrs,
                                                         resampler=resampler, halo_add=halo_add, mpicomm=self.mpicomm)

        if 'weights' not in method and shotnoise_correction:
            raise ValueError('No shot noise correction when blinding is based on particle shifts')

        mesh_delta = _density_contrast(mesh_data, mesh_randoms=mesh_randoms, randoms=randoms, bias=self.bias,
                                           smoothing_radius=smoothing_radius, threshold_randoms=threshold_randoms)
        sigma2 = smoothing_radius
        mesh = mesh_delta.r2c()
        b1 = self.bias
        bfnl = 2 * 1.686 * (b1 - 1.) * _get_from_cosmo(self.cosmo_blind, 'fnl')

        pk_prim = self.cosmo_fid.get_primordial().pk_interpolator(mode='scalar')
        pk_lin = self.cosmo_fid.get_fourier().pk_interpolator(of='theta_cb').to_1d(z=self.z)

        def Tk(k):
            pphi_prim = 9 / 25 * 2 * np.pi**2 / k**3 * pk_prim(k) / self.cosmo_fid.h**3
            return (pk_lin(k) / pphi_prim)**0.5

        mesh = _apply_png_transfer(mesh, bfnl, Tk)

        if shotnoise_correction:

            def S1(k):
                return np.exp(- 0.5 * k**2 * sigma1**2)

            def S2(k):
                return np.exp(- 0.5 * k**2 * sigma2**2)

            sum_w2, _ = _paint_particles(data_positions, weights=data_weights * data_weights if data_weights is not None else None,
                                             attrs=attrs, resampler=resampler, halo_add=halo_add, mpicomm=self.mpicomm)

            sum_wd, _ = _paint_particles(data_positions, weights=data_weights, attrs=attrs,
                                             resampler=resampler, halo_add=halo_add, mpicomm=self.mpicomm)

            if randoms_positions is not None:
                mesh_nbar, _ = _paint_particles(randoms_positions, weights=randoms_weights, attrs=attrs,
                                                    resampler=resampler, halo_add=halo_add, mpicomm=self.mpicomm)
                alpha = mpy.csum(data_weights if data_weights is not None else len(data_positions), mpicomm=self.mpicomm) / mpy.csum(randoms_weights if randoms_weights is not None else len(randoms_positions), mpicomm=self.mpicomm)
                nbar = alpha / np.prod(np.asarray(attrs.cellsize)) * mesh_nbar
            else:
                nbar = mpy.csum(data_weights if data_weights is not None else len(data_positions), mpicomm=self.mpicomm) / np.prod(np.asarray(attrs.boxsize))

            sum_w2 = _replace_mesh_zeros(sum_w2)  # just to avoid NaN's below
            inv_shotnoise = sum_wd * nbar / sum_w2
            inv_shotnoise = _smooth_mesh(inv_shotnoise, smoothing_radius=smoothing_radius)

            # compute the corrective factor at k_pivot
            mu_pivot = 0.6
            k_pivot = 4e-3 if bfnl >= 0 else 8e-3

            if 'data' in method:
                shotnoise = 1 / _read_mesh(inv_shotnoise, data_positions, resampler=resampler, halo_add=halo_add)
            elif 'randoms' in method:
                shotnoise = 1 / _read_mesh(inv_shotnoise, randoms_positions, resampler=resampler, halo_add=halo_add)
            else:
                shotnoise = 0.

            mask = S1(pk_lin.k) > 1e-4  # to avoid error during the interpolation...
            sigma_d_2 = pk_lin.clone(k=pk_lin.k[mask], pk=(S1(pk_lin.k)**2 * pk_lin(pk_lin.k))[mask]).sigma_d()**2

            X_tilde = (b1 + f * mu_pivot**2) * (b1 + (1. - S1(k_pivot)) * f * mu_pivot**2) * S2(k_pivot) * pk_lin(k_pivot) + S2(k_pivot) * shotnoise * np.exp(- 0.5 * k_pivot**2 * mu_pivot**2 * f**2 * sigma_d_2)
            Y_tilde = (b1 + (1. - S1(k_pivot)) * f * mu_pivot**2)**2 * S2(k_pivot)**2 * pk_lin(k_pivot) + S2(k_pivot)**2 * shotnoise
            expected_pivot = 2 * bfnl / Tk(k_pivot) * b1 * (b1 + f * mu_pivot**2) * pk_lin(k_pivot) + (bfnl / Tk(k_pivot))**2 * b1**2 * pk_lin(k_pivot)

            # two solutions, keep the positive one
            shotnoise_factor = (- X_tilde + np.sqrt(X_tilde**2 + Y_tilde * expected_pivot)) / Y_tilde / (bfnl / Tk(k_pivot))
            # if recon.mpicomm.rank == 0: print(pk_lin.sigma_d(), shotnoise, W, X_tilde, Y_tilde, bfnl / Tk(k_pivot), expected_pivot, shotnoise_factor)
        else:
            shotnoise_factor = 1.

        if 'weights' in method:
            mesh = mesh.c2r()
            if 'data' in method:
                weights = _read_mesh(mesh, data_positions, resampler=resampler, halo_add=halo_add)
                weights = (1. if data_weights is None else data_weights) * (1. + shotnoise_factor * weights)
            elif 'randoms' in method:
                weights = _read_mesh(mesh, randoms_positions, resampler=resampler, halo_add=halo_add)
                weights = (1. if randoms_weights is None else randoms_weights) * (1. - shotnoise_factor * weights)
            return _format_output_weights(weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
        else:
            positions = data_positions if 'data' in method else randoms_positions
            shifts = _gradient_shifts(mesh, positions, resampler=resampler, halo_add=halo_add)
            shifts -= mpy.cmean(shifts)
            positions = positions + (shifts if 'data' in method else - shifts)
            return _format_output_positions(positions, position_type=position_type, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)

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
            positions = np.array(positions, dtype=dtype, copy=copy)
            if not np.issubdtype(positions.dtype, np.floating):
                return None, 'Input position arrays should be of floating type, not {}'.format(positions.dtype)
            if positions.shape[-1] != 3:
                return None, 'For position type = {}, please provide a (N, 3) array for positions'.format(position_type)
            return positions, None
        # Array of shape (3, N)
        positions = list(positions)
        for ip, p in enumerate(positions):
            # Cast to the input dtype if exists (may be set by previous positions)
            positions[ip] = np.array(p, dtype=dtype, copy=copy)
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
    :meth:`rsd` and :meth:`png` require pip install git+https://github.com/cosmodesi/pyrecon@mpi.
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

    def rsd(self, data_positions, data_weights=None, randoms_positions=None, randoms_weights=None, recon='IterativeFFTReconstruction', smoothing_radius=15., **kwargs):
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

        recon : string, pyrecon.BaseReconstruction, default='IterativeFFTReconstruction'
            Name of reconstruction algorithm, or (already run) reconstruction instance,
            in which case input ``randoms_positions``, ``randoms_weights`` are ignored.

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
        # Run reconstruction
        if isinstance(recon, str):
            import pyrecon
            ReconstructionAlgorithm = getattr(pyrecon, recon)
            data_weights = _format_weights(data_weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
            f = _get_from_cosmo(self.cosmo_fid, 'f', z=self.z)
            if not any(name in kwargs for name in ['nmesh', 'cellsize']):
                kwargs['cellsize'] = 7.
            kwargs.setdefault('smoothing_radius', smoothing_radius)
            randoms_positions = _format_positions(randoms_positions, position_type=position_type, dtype=self.dtype, cosmo=self.cosmo_fid, mpicomm=self.mpicomm, mpiroot=mpiroot)
            randoms_weights = _format_weights(randoms_weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
            recon = ReconstructionAlgorithm(data_positions=data_positions, data_weights=data_weights,
                                            randoms_positions=randoms_positions, randoms_weights=randoms_weights, f=f, bias=self.bias,
                                            position_type='pos', mpicomm=self.mpicomm, mpiroot=None, **kwargs)
        shifts = recon.read_shifts(data_positions, position_type='pos', mpiroot=None, field='rsd')
        f_blind = _get_from_cosmo(self.cosmo_blind, 'f')
        # Change RSD displacements depending on blind f
        data_positions = data_positions + (f_blind / recon.f - 1.) * shifts
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

        recon : str, pyrecon.BaseReconstruction, default='IterativeFFTReconstruction'
            Name of reconstruction algorithm, or (already run) reconstruction instance,
            in which case input ``randoms_positions``, ``randoms_weights`` are ignored.

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
        if recon is None:
            recon = 'IterativeFFTReconstruction'
        if isinstance(recon, str):
            import pyrecon
            ReconstructionAlgorithm = getattr(pyrecon, recon)
            f = _get_from_cosmo(self.cosmo_fid, 'f', z=self.z)
            if not any(name in kwargs for name in ['nmesh', 'cellsize']):
                kwargs['cellsize'] = 15.
            recon = ReconstructionAlgorithm(data_positions=data_positions, data_weights=data_weights,
                                            randoms_positions=randoms_positions, randoms_weights=randoms_weights, f=f, bias=self.bias,
                                            position_type='pos', mpicomm=self.mpicomm, mpiroot=None,
                                            smoothing_radius=smoothing_radius, **kwargs)
        sigma1 = recon.smoothing_radius
        shifts = recon.read_shifts(data_positions, position_type='pos', mpiroot=None, field='rsd')
        shifted_positions = data_positions - shifts
        # Just to make sure meshes do not exist anymore
        recon.mesh_data = recon.mesh_randoms = None
        recon.assign_data(shifted_positions, weights=data_weights, position_type='pos', mpiroot=None)

        if randoms_positions is not None:
            recon.assign_randoms(randoms_positions, weights=randoms_weights, position_type='pos', mpiroot=None)

        if 'weights' not in method and shotnoise_correction:
            raise ValueError('No shot noise correction when blinding is based on particle shifts')

        recon.set_density_contrast(smoothing_radius=smoothing_radius)  # divides by bias
        sigma2 = recon.smoothing_radius
        mesh = recon.mesh_delta.r2c()
        b1 = recon.bias
        bfnl = 2 * 1.686 * (b1 - 1.) * _get_from_cosmo(self.cosmo_blind, 'fnl')

        pk_prim = self.cosmo_fid.get_primordial().pk_interpolator(mode='scalar')
        pk_lin = self.cosmo_fid.get_fourier().pk_interpolator(of='theta_cb').to_1d(z=self.z)

        def Tk(k):
            pphi_prim = 9 / 25 * 2 * np.pi**2 / k**3 * pk_prim(k) / self.cosmo_fid.h**3
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
            recon.assign_data(data_positions, weights=data_weights * data_weights if data_weights is not None else None, position_type='pos', mpiroot=None)
            sum_w2 = recon.mesh_data

            recon.mesh_data = None
            recon.assign_data(data_positions, weights=data_weights, position_type='pos', mpiroot=None)
            sum_wd = recon.mesh_data

            if randoms_positions is not None:
                recon.mesh_data = None
                recon.assign_data(randoms_positions, weights=randoms_weights, position_type='pos', mpiroot=None)
                alpha = mpy.csum(data_weights if data_weights is not None else len(data_positions), mpicomm=self.mpicomm) / mpy.csum(randoms_weights if randoms_weights is not None else len(randoms_positions), mpicomm=self.mpicomm)
                nbar = alpha / np.prod(recon.cellsize) * recon.mesh_data
            else:
                nbar = mpy.csum(data_weights if data_weights is not None else len(data_positions), mpicomm=self.mpicomm) / np.prod(recon.boxsize)

            sum_w2[sum_w2 == 0.] = 1.  # just to avoid NaN's below
            inv_shotnoise = sum_wd * nbar / sum_w2
            inv_shotnoise = recon._smooth_gaussian(inv_shotnoise)

            # compute the corrective factor at k_pivot
            mu_pivot = 0.6
            k_pivot = 4e-3 if bfnl >= 0 else 8e-3

            if 'data' in method:
                shotnoise = 1 / recon._readout(inv_shotnoise, data_positions)
            elif 'randoms' in method:
                shotnoise = 1 / recon._readout(inv_shotnoise, randoms_positions)
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
                weights = recon._readout(mesh, data_positions)
                weights = (1. if data_weights is None else data_weights) * (1. + shotnoise_factor * weights)
            elif 'randoms' in method:
                weights = recon._readout(mesh, randoms_positions)
                weights = (1. if randoms_weights is None else randoms_weights) * (1. - shotnoise_factor * weights)
            return _format_output_weights(weights, mpicomm=self.mpicomm, mpiroot=mpiroot)
        else:
            positions = data_positions if 'data' in method else randoms_positions
            disps = []
            for iaxis in range(mesh.ndim):
                psi = mesh.copy()
                for kslab, slab in zip(psi.slabs.x, psi.slabs):
                    k2 = sum(kk**2 for kk in kslab)
                    k2[k2 == 0.] = 1.  # avoid dividing by zero
                    slab[...] *= 1j * kslab[iaxis] / k2
                psi = psi.c2r()
                disps.append(recon._readout(psi, positions))
            shifts = np.column_stack(disps)
            shifts -= mpy.cmean(shifts)
            positions = positions + (shifts if 'data' in method else - shifts)
            return _format_output_weights(positions, mpicomm=self.mpicomm, mpiroot=mpiroot)

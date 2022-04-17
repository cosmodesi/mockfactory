import numpy as np

import mpytools as mpy
from mpytools import CurrentMPIComm

from .utils import BaseClass
from . import utils


def _make_array(value, shape, dtype='f8'):
    # Return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret


def _get_los(los):
    # Return line of sight 3-vector
    if isinstance(los, str):
        los = 'xyz'.index(los)
    if np.ndim(los) == 0:
        ilos = los
        los = np.zeros(3, dtype='f8')
        los[ilos] = 1.
    los = np.array(los, dtype='f8')
    return los / utils.distance(los)


def _compensate_mesh(mesh, resampler='nnb'):
    # Compensate particle-mesh assignment by applying window
    # See https://arxiv.org/abs/astro-ph/0409240.
    p = {'nnb': 1, 'ngp': 1, 'cic': 2, 'tsc': 3, 'pcs': 4}[resampler]

    def window(x):
        toret = 1.
        for xi in x:
            toret = toret * np.sinc(0.5 * xi / np.pi) ** p
        return toret

    mesh = mesh.r2c()
    cellsize = mesh.pm.BoxSize / mesh.pm.Nmesh
    for k, slab in zip(mesh.slabs.x, mesh.slabs):
        slab[...] /= window(ki * ci for ki, ci in zip(k, cellsize))
    return mesh.c2r()


def _transform_rslab(rslab, boxsize):
    # We do not use the same conventions as pmesh:
    # rslab < 0 is sent back to [boxsize/2, boxsize]
    toret = []
    for ii, rr in enumerate(rslab):
        mask = rr < 0.
        rr[mask] += boxsize[ii]
        toret.append(rr)
    return toret


class SetterProperty(object):
    """
    Attribute setter, runs ``func`` when setting a class attribute.
    Taken from https://stackoverflow.com/questions/17576009/python-class-property-use-setter-but-evade-getter
    """
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)


class BaseGaussianMock(BaseClass):
    """
    Base template Gaussian mock class.
    Mock engines should extend this class, by (at least) implementing :meth:`set_rsd`.

    A standard run would be:

    .. code-block:: python

        # MyMock is your mock engine
        mock = MyMock(power, boxsize, nmesh)
        mock.set_real_delta_field()
        # from here you can get the real-space delta field
        # mock.readout(positions, field='1+delta')
        mock.set_rsd(f, los=los)
        # from here you can get the redshift-space delta field
        mock.set_analytic_selection_function(nbar, interlaced=False)
        # from here you can get the density field
        # mock.readout(positions, field='nbar*(1+delta)')
        mock.poisson_sample(seed)
        # from here you have mock.positions


    Attributes
    ----------
    mesh_delta_k : pm.ComplexField
        Density fluctuations in Fourier space.

    mesh_delta_r : pm.RealField
        Density fluctuations in real space.

    nbar : pm.RealField
        Selection function in real space.

    positions : array of shape (N, 3)
        Cartesian positions sampling the density field.
    """
    @CurrentMPIComm.enable
    def __init__(self, power, nmesh=None, boxsize=None, cellsize=None, boxcenter=0., los=None,
                 seed=None, unitary_amplitude=False, inverted_phase=False, dtype='f4', mpicomm=None):
        r"""
        Initialize :class:`GaussianFieldMesh` and set :attr:`mesh_delta_k`.

        Parameters
        ----------
        power : callable
            The callable power spectrum function.

        nmesh : int, 3-vector of int, default=None
            Number of mesh cells per side.

        boxsize : float, 3-vector of floats, default=None
            Box size.

        cellsize : float, 3-vector of floats, default=None
            Physical size of mesh cells.
            If not ``None``, and mesh size ``nmesh`` is not ``None``, used to set ``boxsize`` as ``nmesh * cellsize``.
            If ``nmesh`` is ``None``, it is set as (the nearest integer(s) to) ``boxsize/cellsize``.

        boxcenter : float, 3-vector of floats, default=0.
            Box center.

        los : 'x', 'y', 'z'; int, 3-vector of int, default=None
            Line of sight :math:`\hat{\eta}` used to paint anisotropic power spectrum.
            If provided, ``power`` should depend :math:`(k,\hat{k} \cdot \hat{\eta})`.

        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.

        unitary_amplitude: bool, default=False
            ``True`` to remove variance from the complex field by fixing the
            amplitude to :math:`P(k)` and only the phase is random.

        inverted_phase : bool, default=False
            ``True`` to invert phase of the complex field.

        dtype : string, np.dtype, defaut='f8'
            Type for :attr:`mesh_delta_k`.

        mpicomm : MPI communicator, default=None
            The MPI communicator.
        """
        self.power = power
        self.mpicomm = mpicomm
        self.mpiroot = 0
        # set the seed randomly if it is None
        if seed is None: seed = mpy.random.bcast_seed(size=None)
        self.attrs = {}
        self.attrs['seed'] = seed
        self.attrs['unitary_amplitude'] = unitary_amplitude
        self.attrs['inverted_phase'] = inverted_phase
        self.dtype = np.dtype(dtype)

        if boxsize is None:
            if cellsize is not None and nmesh is not None:
                boxsize = nmesh * cellsize
            else:
                raise ValueError('boxsize (or cellsize) must be specified')

        if nmesh is None:
            if cellsize is not None:
                nmesh = np.rint(boxsize / cellsize).astype(int)
            else:
                raise ValueError('nmesh (or cellsize) must be specified')

        self.boxcenter = boxcenter
        from pmesh.pm import ParticleMesh
        self.pm = ParticleMesh(BoxSize=_make_array(boxsize, 3, dtype='f8'), Nmesh=_make_array(nmesh, 3, dtype='i8'), dtype=self.dtype, comm=self.mpicomm)

        if los is not None:
            los = self._get_los(los)
        self.attrs['los'] = los
        self.set_complex_delta_field()

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    @property
    def boxsize(self):
        """Box size."""
        return self.pm.BoxSize

    @property
    def nmesh(self):
        """Mesh size."""
        return self.pm.Nmesh

    @property
    def ndim(self):
        return self.pm.ndim

    @SetterProperty
    def boxcenter(self, boxcenter):
        """Set box center."""
        self.__dict__['boxcenter'] = _make_array(boxcenter, 3, dtype='f8')

    def set_complex_delta_field(self):
        """Set :attr:`mesh_delta_k`; the end-user does not neet to call this method."""

        mesh_delta_k = self.pm.generate_whitenoise(self.attrs['seed'], type='untransposedcomplex', unitary=self.attrs['unitary_amplitude'])
        if self.is_mpi_root():
            self.log_info('White noise generated.')

        if self.attrs['inverted_phase']: mesh_delta_k[...] *= -1
        # volume factor needed for normalization
        norm = 1.0 / self.boxsize.prod()
        # iterate in slabs over fields
        # loop over the mesh, slab by slab
        los = self.attrs['los']
        for kslab, delta_slab in zip(mesh_delta_k.slabs.x, mesh_delta_k.slabs):
            # the square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            k = (k2**0.5).ravel()
            mask_nonzero = k != 0.
            power = np.zeros_like(k)
            if los is not None:
                mu = sum(kk * ll for kk, ll in zip(kslab, los)).ravel() / k
                power[mask_nonzero] = self.power(k[mask_nonzero], mu[mask_nonzero])
            else:
                power[mask_nonzero] = self.power(k[mask_nonzero])

            # multiply complex field by sqrt of power
            delta_slab[...].flat *= (power * norm)**0.5
        self.mesh_delta_k = mesh_delta_k

    def set_real_delta_field(self, bias=None, lognormal_transform=False):
        r"""
        Set the density contrast in real space :attr:`mesh_delta_r`.

        Parameters
        ----------
        bias : callable, float, default=None
            Eulerian bias (optionally including growth factor).
            If a callable, take the (flattened) :math:`\delta' field and (flattened) distance to the observer as input.
            Else, a float to multiply the :math:`\delta' field.
            If ``None``, defaults to no bias.

        lognormal_transform : bool, default=False
            Whether to apply a lognormal transform to the (biased) real field, i.e. :math:`\exp{(\delta)}'
        """
        mesh_delta_r = self.mesh_delta_k.c2r()
        offset = self.boxcenter - self.boxsize / 2.  # + 0.5*self.boxsize / self.nmesh
        if bias is not None:
            if callable(bias):
                for islabs in zip(mesh_delta_r.slabs.x, mesh_delta_r.slabs):
                    rslab, delta_slab = islabs[:2]
                    rslab = _transform_rslab(rslab, self.boxsize)
                    rnorm = np.sum((r + o)**2 for r, o in zip(rslab, offset))**0.5
                    delta_slab[...].flat = bias(delta_slab.flatten(), rnorm.flatten())
            else:
                mesh_delta_r *= bias
        if lognormal_transform:
            mesh_delta_r[:] = np.exp(mesh_delta_r.value)
            mesh_delta_r[:] /= mesh_delta_r.cmean(dtype='f8')
            mesh_delta_r[:] -= 1.
        self.mesh_delta_r = mesh_delta_r

    def set_analytic_selection_function(self, nbar, interlacing=False):
        """
        Set mesh mean density :attr:`nbar` with analytic selection function.

        Parameters
        ----------
        nbar : callable, float, default=None
            Analytic selection function, i.e. mean density.
            If a callable, take the distance, right ascension and declination as inputs.
            Else, a float.

        interlacing : bool, int, default=2
            Whether to use interlacing to reduce aliasing when applying selection function on the mesh.
            If positive int, the interlacing order (minimum: 2).
        """
        cellsize = self.boxsize / self.nmesh
        dv = np.prod(cellsize)
        self.attrs['shotnoise'] = 0.
        if not callable(nbar):
            self.nbar = nbar
            self.attrs['norm'] = np.prod(self.nmesh) * self.nbar**2 * dv
            return
        self.nbar = self.pm.create(type='real')

        def cartesian_to_sky(*position):
            dist = sum(pos**2 for pos in position)**0.5
            ra = np.arctan2(position[1], position[0]) % (2. * np.pi)
            dec = np.arcsin(position[2] / dist)
            conversion = np.pi / 180.
            return dist, ra / conversion, dec / conversion

        offset = self.boxcenter - self.boxsize / 2.

        for rslab, slab in zip(self.nbar.slabs.x, self.nbar.slabs):
            rslab = _transform_rslab(rslab, self.boxsize)
            dist, ra, dec = cartesian_to_sky(*[(r + o).ravel() for r, o in zip(rslab, offset)])
            slab[...].flat = nbar(dist, ra, dec)
        if interlacing:
            interlacing = int(interlacing)
            if interlacing == 1:
                if self.is_mpi_root():
                    self.log_warning('Provided interlacing is {}; setting it to 2.'.format(interlacing))
                interlacing = 2
            shifts = np.arange(interlacing) * 1. / interlacing
            # remove 0 shift, already computed
            shifts = shifts[1:]
            self.nbar = self.nbar.r2c()
            for shift in shifts:
                # paint to two shifted meshes
                nbar_shifted = self.pm.create(type='real')
                offset -= 0.5 * cellsize  # shift nbar by 0.5*cellsize
                for rslab, slab in zip(nbar_shifted.slabs.x, nbar_shifted.slabs):
                    rslab = _transform_rslab(rslab, self.boxsize)
                    dist, ra, dec = cartesian_to_sky(*[(r + o).ravel() for r, o in zip(rslab, offset)])
                    slab[...].flat = nbar(dist, ra, dec)
                nbar_shifted = nbar_shifted.r2c()
                for k, s1, s2 in zip(self.nbar.slabs.x, self.nbar.slabs, nbar_shifted.slabs):
                    kc = sum(k[i] * cellsize[i] for i in range(3))
                    s1[...] = s1[...] + s2[...] * np.exp(shift * 1j * kc)
            self.nbar = self.nbar.c2r()
            self.nbar[:] /= interlacing
        self.attrs['norm'] = (self.nbar**2).csum() * dv

    def set_real_white_noise(self, seed=None):
        """
        Add white noise to :attr:`mesh_delta_r`, with amplitude :attr:`nbar`.

        Parameters
        ----------
        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.
        """
        if seed is None: seed = mpy.random.bcast_seed(size=None)
        noise = self.pm.generate_whitenoise(seed, type='real')
        # noise has amplitude of 1
        # multiply by expected density
        vol_per_cell = np.prod(self.boxsize / self.nmesh)
        self.mesh_delta_r += noise / (self.nbar * vol_per_cell) ** 0.5

    def readout(self, positions, field='delta', resampler='nnb', compensate=False):
        """
        Read density field at input positions.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        field : string, pm.RealField, default='1+delta'
            Mesh or type of field to read, either 'delta', 'nbar*delta', 'nbar*(1+delta)', 'nbar'.
            Fields with 'nbar' require calling :meth:`set_analytic_selection_function` first.

        resampler : string, default='nnb'
            Resampler to interpolate the field at input positions.
            e.g. 'nnb', 'cic', 'tsc', 'pcs'...

        compensate : bool, default=False
            Whether to compensate for smooting due to resampling, to make the power spectrum
            of output (positions, values) match that of ``field`` (up to sampling noise).

        Returns
        -------
        values : array of shape (N,)
            Field values interpolated at input positions.
        """
        if not isinstance(field, str):
            mesh = field
        elif field == 'delta':
            mesh = self.mesh_delta_r
        elif field == 'nbar*delta':
            mesh = self.nbar * self.mesh_delta_r
        elif field == 'nbar*(1+delta)':
            mesh = self.nbar * (self.mesh_delta_r + 1)
        elif field == 'nbar':
            mesh = self.nbar
        else:
            raise ValueError('Unknown field {}'.format(field))

        resampler = resampler.lower()

        if compensate:
            mesh = _compensate_mesh(mesh, resampler=resampler)

        if resampler == 'ngp': resampler = 'nnb'
        # half cell shift already included in resampling
        positions = positions - self.boxcenter + self.boxsize / 2.
        layout = self.pm.decompose(positions, smoothing=resampler)
        positions = layout.exchange(positions)
        values = mesh.readout(positions, resampler=resampler)
        return layout.gather(values, mode='sum', out=None)

    def to_nbodykit_mesh(self):
        """Export density fied ``self.mesh_delta_r*self.nbar`` to :class:`nbodykit.source.mesh.field.FieldMesh`."""
        from nbodykit.source.mesh.field import FieldMesh
        toret = FieldMesh(self.mesh_delta_r * self.nbar)
        toret.attrs.update(self.attrs)
        toret.attrs['BoxSize'] = self.boxsize
        toret.attrs['BoxCenter'] = self.boxcenter
        return toret

    def poisson_sample(self, seed=None):
        """
        Poisson sample density field and set :attr:`position`.

        Parameters
        ----------
        seed : int, default=None
            Random seed.
        """
        import mpsort
        seed1, seed2 = mpy.random.bcast_seed(seed=seed, size=2, mpicomm=self.mpicomm)
        # mean number of objects per cell
        cellsize = self.boxsize / self.nmesh
        dv = np.prod(cellsize)
        # number of objects in each cell (per rank, as a RealField)
        cellmean = (1. + self.mesh_delta_r) * self.nbar * dv
        # create a random state with the input seed
        rng = mpy.random.MPIRandomState(size=self.mesh_delta_r.size, seed=seed1, mpicomm=self.mpicomm)
        # generate poissons. Note that we use ravel/unravel to
        # maintain MPI invariance.
        number_ravel = rng.poisson(lam=cellmean.ravel())
        number = self.pm.create(type='real')
        number.unravel(number_ravel)

        ntotal = int(number.csum() + 0.5)
        if self.is_mpi_root():
            self.log_info('Poisson sampling done, total number of objects is {:d}.'.format(ntotal))

        # create uniform grid of particles, one per grid point, in BoxSize coordinates
        positions, ids = self.pm.generate_uniform_particle_grid(shift=0, return_id=True)
        # no need to do decompose because pos_mesh is strictly within the
        # local volume of the RealField.
        number_per_cell = number.readout(positions, resampler='nnb')
        # fight round off errors, if any
        number_per_cell = np.int64(number_per_cell + 0.5)
        # create the correct number of particles at each cell
        positions = positions.repeat(number_per_cell, axis=0)
        ids = ids.repeat(number_per_cell, axis=0)

        if self.is_mpi_root():
            self.log_info('Catalog produced. Assigning in cell shift.')

        positions = mpsort.sort(positions, orderby=ids, comm=self.mpicomm)

        rng_shift = mpy.random.MPIRandomState(size=len(positions), seed=seed2, mpicomm=self.mpicomm)
        in_cell_shift = np.array([rng_shift.uniform(0, c) for c in cellsize]).T

        positions[...] += in_cell_shift
        # for ii in range(3):
        #     positions[positions[:,ii] >= self.boxsize[ii]/2,ii] -= self.boxsize[ii]
        # print(positions.min(axis=0), positions.max(axis=0))
        # positions[...] %= self.boxsize
        positions[...] += self.boxcenter - self.boxsize / 2.

        self.position = positions

    def to_catalog(self):
        """
        Export as :class:`make_survey.BoxCatalog`.

        Note
        ----
        :meth:`poisson_sample` must be called first.
        """
        source = {'Position': self.position}
        from .make_survey import BoxCatalog
        return BoxCatalog(source, position='Position', boxsize=self.boxsize, boxcenter=self.boxcenter, attrs=self.attrs)

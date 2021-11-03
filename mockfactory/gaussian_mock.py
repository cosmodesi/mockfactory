import logging

import numpy as np

from pmesh.pm import RealField, ComplexField, ParticleMesh
import mpsort

from .mpi import MPIRandomState, CurrentMPIComm
from .utils import BaseClass
from . import mpi, utils


def _make_array(value, shape, dtype='f8'):
    # return numpy array filled with value
    toret = np.empty(shape, dtype=dtype)
    toret[...] = value
    return toret

def _get_los(los):
    # return line of sight 3-vector
    if isinstance(los,str):
        los = 'xyz'.index(los)
    if np.ndim(los) == 0:
        ilos = los
        los = np.zeros(3, dtype='f8')
        los[ilos] = 1.
    los = np.asarray(los)
    return los


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


def cartesian_to_sky(position, wrap=True, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (3, N)
        Position in cartesian coordinates.

    wrap : bool, default=True
        Whether to wrap RA in :math:`[0, 2 \pi]`.

    degree : bool, default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    dist : array
        Distance.

    ra : array
        Right Ascension.

    dec : array
        Declination.
    """
    dist = sum(pos**2 for pos in position)**0.5
    ra = np.arctan2(position[:,1], position[:,0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(position[:,2]/dist)
    conversion = np.pi/180. if degree else 1.
    return dist, ra/conversion, dec/conversion


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
    def __init__(self, power, boxsize, nmesh, boxcenter=0., los=None,
                 seed=None, unitary_amplitude=False, inverted_phase=False,
                 dtype='f4', mpicomm=None):
        r"""
        Initialize :class:`GaussianFieldMesh` and set :attr:`mesh_delta_k`.

        Parameters
        ----------
        power : callable
            The callable power spectrum function.

        boxsize : float, 3-vector of floats
            Box size.

        nmesh : int, 3-vector of int
            Number of mesh cells per side.

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
        # set the seed randomly if it is None
        if seed is None: seed = mpi.bcast_seed(size=None)
        self.attrs = {}
        self.attrs['seed'] = seed
        self.attrs['unitary_amplitude'] = unitary_amplitude
        self.attrs['inverted_phase'] = inverted_phase
        self.dtype = np.dtype(dtype)

        if nmesh is None or boxsize is None:
            raise ValueError('Both nmesh and boxsize must be provided to initialize ParticleMesh')

        self.boxcenter = boxcenter
        self.pm = ParticleMesh(BoxSize=_make_array(boxsize, 3, dtype='f8'), Nmesh=_make_array(nmesh, 3, dtype='i8'), dtype=self.dtype, comm=self.mpicomm)

        if los is not None:
            los = self._get_los(los)
        self.attrs['los'] = los
        self.set_complex_delta_field()

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
        if self.mpicomm.rank == 0:
            self.log_info('White noise generated')

        if self.attrs['inverted_phase']: mesh_delta_k[...] *= -1
        # volume factor needed for normalization
        norm = 1.0 / self.boxsize.prod()
        # iterate in slabs over fields
        # loop over the mesh, slab by slab
        los = self.attrs['los']
        for kslab,delta_slab in zip(mesh_delta_k.slabs.x,mesh_delta_k.slabs):
            # the square of the norm of k on the mesh
            k2 = sum(kk**2 for kk in kslab)
            mask_zero = k2 == 0.
            # avoid dividing by zero, set a value that is non-zero and likeliy to be in power interpolation range
            k2[mask_zero] = k2.flat[1]**2
            k = (k2**0.5).flatten()
            if los is not None:
                mu = sum(kk*ll for kk, ll in zip(kslab, los)).flatten()/k
                power = self.power(k, mu)
            else:
                power = self.power(k)

            # multiply complex field by sqrt of power
            delta_slab[...].flat *= (power*norm)**0.5

            # set k == 0 to zero (zero config-space mean)
            delta_slab[mask_zero] = 0.
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
        offset = self.boxcenter + 0.5*self.boxsize / self.nmesh
        if bias is not None:
            if callable(bias):
                for islabs in zip(mesh_delta_r.slabs.x, mesh_delta_r.slabs):
                    rslab, delta_slab = islabs[:2]
                    rnorm = np.sum((r + o)**2 for r,o in zip(rslab,offset))**0.5
                    delta_slab[...].flat = bias(delta_slab[...].flat, rnorm.flatten())
            else:
                mesh_delta_r *= bias
        if lognormal_transform:
            mesh_delta_r[:] = np.exp(mesh_delta_r.value)
            mesh_delta_r[:] /= mesh_delta_r.cmean(dtype='f8')
            mesh_delta_r[:] -= 1.
        self.mesh_delta_r = mesh_delta_r

    def set_analytic_selection_function(self, nbar, interlaced=False):
        """
        Set mesh mean density :attr:`nbar` with analytic selection function.

        Parameters
        ----------
        nbar : callable, float, default=None
            Analytic selection function, i.e. mean density.
            If a callable, take the distance, right ascension and declination as inputs.
            Else, a float.

        interlaced : bool, default=False
            ``True`` to apply interlacing correction.
        """
        cellsize = self.boxsize/self.nmesh
        dv = np.prod(cellsize)
        self.attrs['shotnoise'] = 0.
        if not callable(nbar):
            self.nbar = nbar
            self.attrs['norm'] = np.prod(self.nmesh)*self.nbar**2*dv
            return
        self.nbar = self.pm.create(type='real')
        for rslab, slab in zip(self.nbar.slabs.x,self.nbar.slabs):
            dist,ra,dec = cartesian_to_sky([r.flatten() for r in rslab])
            slab[...].flat = nbar(dist, ra, dec)
        if interlaced:
            nbar2 = self.nbar.copy()
            #shifted = pm.affine.shift(0.5)
            offset = 0.5*cellsize
            for rslab, slab in zip(nbar2.slabs.x,nbar2.slabs):
                dist,ra,dec = cartesian_to_sky([(r + o).flatten() for r,o in zip(rslab,offset)])
                slab[...].flat = nbar(dist, ra, dec)
            c1 = self.nbar.r2c()
            c2 = nbar2.r2c()
            # and then combine
            for k, s1, s2 in zip(c1.slabs.x, c1.slabs, c2.slabs):
                kcellsize = sum(k[i] * cellsize[i] for i in range(self.nbar.ndim))
                s1[...] = s1[...] * 0.5 + s2[...] * 0.5 * np.exp(0.5 * 1j * kcellsize)
            # FFT back to real-space
            c1.c2r(out=self.nbar)
        self.attrs['norm'] = (self.nbar**2).csum()*dv

    def set_real_white_noise(self, seed=None):
        """
        Add white noise to :attr:`mesh_delta_r`, with amplitude :attr:`nbar`.

        Parameters
        ----------
        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.
        """
        if seed is None: seed = mpi.bcast_seed(size=None)
        noise = self.pm.generate_whitenoise(seed, type='real')
        # noise has amplitude of 1
        # multiply by expected density
        dv = np.prod(self.pm.boxsize/self.pm.nmesh)
        self.mesh_delta_r += noise*self.nbar*dv

    def readout(self, positions, field='1+delta', resampler='nnb'):
        """
        Read density field at input positions.

        Parameters
        ----------
        positions : array of shape (N,3)
            Cartesian positions.

        field : string, default='1+delta'
            Type of field to read, either '1+delta', 'nbar*delta', 'nbar*(1+delta)', 'nbar'.
            Fields with 'nbar' require calling :meth:`set_analytic_selection_function` first.

        resampler : string, default='nnb'
            Resampler to interpolate the field at input positions.
            e.g. 'nnb', 'cic', 'tsc', 'pcs'...

        Returns
        -------
        values : array of shape (N,)
            Field values interpolated at input positions.
        """
        # half cell shift already included in resampling
        if field == '1+delta':
            return self.mesh_delta_r.readout(positions - self.boxcenter, resampler=resampler) + 1.
        if field == 'nbar*delta':
            return (self.nbar*self.mesh_delta_r).readout(positions - self.boxcenter, resampler=resampler)
        if field == 'nbar*(1+delta)':
            return (self.nbar*(self.mesh_delta_r + 1)).readout(positions - self.boxcenter, resampler=resampler)
        if field == 'nbar':
            return self.nbar.readout(positions - self.boxcenter, resampler=resampler)
        raise ValueError('Unknown field {}'.format(field))

    def to_nbodykit_mesh(self):
        """Export density fied ``self.mesh_delta_r*self.nbar`` to :class:`nbodykit.source.mesh.field.FieldMesh`."""
        from nbodykit.source.mesh.field import FieldMesh
        toret = FieldMesh(self.mesh_delta_r*self.nbar)
        toret.attrs.update(self.attrs)
        toret.attrs['BoxSize'] = self.boxsize
        toret.attrs['BoxCenter'] = self.boxcenter
        return toret

    def poisson_sample(self, seed=None):
        """
        Poisson sample density field and set :attr:`positions`.

        Parameters
        ----------
        seed : int, default=None
            Random seed.
        """
        import mpsort
        seed1, seed2 = mpi.bcast_seed(seed=seed, size=2)
        # mean number of objects per cell
        cellsize = self.boxsize / self.nmesh
        dv = np.prod(cellsize)
        # number of objects in each cell (per rank, as a RealField)
        cellmean = (self.mesh_delta_r + 1.) * self.nbar*dv
        # create a random state with the input seed
        rng = MPIRandomState(seed=seed1, mpicomm=self.mpicomm, size=self.mesh_delta_r.size)
        # generate poissons. Note that we use ravel/unravel to
        # maintain MPI invariance.
        number_ravel = rng.poisson(lam=cellmean.ravel())
        number = self.pm.create(type='real')
        number.unravel(number_ravel)

        ntotal = int(number.csum() + 0.5)
        if self.mpicomm.rank == 0:
            self.log_info('Poisson sampling done, total number of objects is {:d}'.format(ntotal))

        # create uniform grid of particles, one per grid point, in BoxSize coordinates
        positions, ids = self.pm.generate_uniform_particle_grid(shift=0.0, return_id=True)
        # no need to do decompose because pos_mesh is strictly within the
        # local volume of the RealField.
        number_per_cell = number.readout(positions, resampler='nnb')
        # fight round off errors, if any
        number_per_cell = np.int64(number_per_cell + 0.5)
        # create the correct number of particles at each cell
        positions = positions.repeat(number_per_cell, axis=0)
        ids = ids.repeat(number_per_cell, axis=0)

        if self.mpicomm.rank == 0:
            self.log_info('Catalog produced. Assigning in cell shift.')

        positions = mpsort.sort(positions, orderby=ids, comm=self.mpicomm)

        rng_shift = MPIRandomState(seed=seed2, mpicomm=self.mpicomm, size=len(positions))
        in_cell_shift = np.array([rng_shift.uniform(0, c) for c in cellsize]).T

        positions[...] += in_cell_shift
        positions[...] %= self.boxsize
        positions[...] += self.boxcenter

        self.positions = positions

    def to_nbodykit_catalog(self):
        """
        Export as :class:`nbodykit.source.catalog.ArrayCatalog`.

        Note
        ----
        :meth:`poisson_sample` must be called first.
        """
        source = {'Position':self.positions}
        from nbodykit.source.catalog import ArrayCatalog
        return ArrayCatalog(source, **self.attrs)

import os
import logging
import functools
import itertools

import numpy as np
from scipy import interpolate, optimize

from . import mpi, utils
from .mpi import CurrentMPIComm, MPIRandomState
from .utils import BaseClass


def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix transforming 3D vector ``a`` to 3D vector ``b``.

    >>> a = np.array([0.,1.,2.])
    >>> b = np.array([0.,2.,1.])
    >>> rot = rotation_matrix_from_vectors(a, b):
    >>> assert np.allclose(rot.dot(a),b)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a /= utils.distance(a)
    b /= utils.distance(b)
    v = np.cross(a,b)
    c = np.dot(a,b)
    s = utils.distance(v)
    I = np.identity(3,dtype='f8')
    k = np.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])
    if s == 0.: return I
    return I + k + np.matmul(k,k) * ((1.-c)/(s**2))


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


class EuclideanIsometryError(Exception):

    """Error raised when issue with euclidean isometry."""


class EuclideanIsometry(BaseClass):

    """Class to manage a series of euclidean isometries (translations and rotations)."""

    def __init__(self):
        """Initialize :class:`EuclideanIsometry`."""
        self._rotation = np.eye(3, dtype='f8')
        self._translation = np.zeros(3, dtype='f8')

    def is_identity(self, translational_invariant=False):
        """Whether current isometry is an identity."""
        toret = np.all(self._rotation == np.eye(3, dtype='f8'))
        if not translational_invariant:
            toret &= np.all(self._translation == 0.)
        return toret

    def transform(self, vector, translational_invariant=False):
        """
        Apply current isometry to input vector.

        Parameters
        ----------
        vector : array of shape (N, 3)
            Input vector.

        translational_invariant : bool, default=False
            Whether vector is translational-invariant (in which case no translation is applied),
            as is e.g. the case of velocities.

        Returns
        -------
        vector : array of shape (N, 3)
            Output vector, with isometry applied.
        """
        toret = np.tensordot(vector, self._rotation, axes=((-1,),(1,)))
        if not translational_invariant: toret += self._translation
        return toret

    @staticmethod
    def _get_axis(axis):
        if axis not in range(3):
            axis = 'xyz'.index(axis)
        return axis

    def rotation(self, angle=0., axis=None, degree=True, frame='origin'):
        """
        Register rotation around axis.

        Parameters
        ----------
        angle : float, array, default=0.
            (Signed) angle (around ``axis``).

        axis : int, string, default=None
            Axis number, or 'x' (0), 'y' (1) or 'z' (2),
            or ``None``, in which case ``angle`` is broadcast to all axes.

        degree : bool, default=True
            Whether input ``angle`` is in degree.

        frame : string, default='origin'
            Either 'origin', in which case rotation is w.r.t. origin :math:`(0,0,0)`,
            or 'current', in which case rotation is w.r.t. current position.
        """
        if degree: angle *= np.pi/180.
        if axis is not None:
            axis = self._get_axis(axis)
            angle = [angle if ax == axis else 0. for ax in range(3)]
        angles = _make_array(angle, 3, dtype='f8')
        matrix = np.eye(3, dtype='f8')
        for axis, angle in enumerate(angles):
            c, s = np.cos(angle), np.sin(angle)
            if axis == 0: mat = [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
            if axis == 1: mat = [[c,0.,s],[0,1.,0.],[-s,0.,c]]
            if axis == 2: mat = [[c,-s,0],[s,c,0],[0.,0,1.]]
            matrix = np.asarray(mat).dot(matrix)
        self._rotation = matrix.dot(self._rotation)
        if frame == 'origin':
            self._translation = matrix.dot(self._translation)
        elif frame != 'current':
            raise EuclideanIsometryError('center must be "origin" or "current"')

    def translation(self, shift=0., axis=None, frame='origin'):
        """
        Register translation.

        Parameters
        ----------
        shift : float, array, default=0.
            Shift (along ``axis``).

        axis : int, string, default=None
            Axis number, or 'x' (0), 'y' (1) or 'z' (2),
            or ``None``, in which case ``shift`` is broadcast to all axes.

        frame : string, default='origin'
            Either 'origin', in which case translation is w.r.t. origin frame.
            or 'current', in which case translation is w.r.t. current rotation.
        """
        if axis is not None:
            axis = self._get_axis(axis)
            shift = [shift if ax == axis else 0. for ax in range(3)]
        shift = _make_array(shift, 3)
        if frame == 'current':
            shift = np.tensordot(shift, self._rotation, axes=((1,),(1,)))
        elif frame != 'origin':
            raise EuclideanIsometryError('center must be "origin" or "current"')
        self._translation += shift

    def reset_rotation(self, frame='origin'):
        """
        Reset rotation.

        Parameters
        ----------
        frame : string, default='origin'
            Either 'origin', in which case rotation is undone w.r.t. origin :math:`(0,0,0)`,
            or 'current', in which case rotation is undone w.r.t. current position.
        """
        if frame == 'origin':
            self._translation = self._rotation.T.dot(self._translation)
        elif frame != 'current':
            raise EuclideanIsometryError('center must be "origin" or "current"')
        self._rotation = np.eye(self._rotation.shape[0], dtype=self._rotation.dtype)

    def reset_translate(self):
        """Reset translation."""
        self._translation[:] = 0.

    @classmethod
    def concatenate(cls, *others):
        """Return isometry corresponding to the successive input isometries."""
        if not others: return cls()
        new = cls(others[0])
        for other in others[1:]:
            new._rotation = other._rotation.dot(new._rotation)
            new._translation += other._translation
        return new

    def __radd__(self, other):
        """Operation corresponding to ``other + self``."""
        if other in [[], 0, None]:
            return self.copy()
        return self.__add__(other)

    def __iadd__(self, other):
        """Operation corresponding to ``self += other``."""
        self.extend(other)
        return self

    def __add__(self, other):
        """Addition of two isometries instances is defined as concatenation."""
        return self.concatenate(self,other)


def box_to_cutsky(boxsize, dmax):
    r"""
    Given input box size ``boxsize``, and maximum distance to the observer desired ``dmax``,
    return the maximum distance, RA, Dec ranges.

    Parameters
    ----------
    boxsize : float, array
        Box size.

    dmax : float
        Maximum distance to the observer.

    Returns
    -------
    drange : tuple
        Maximum distance range.

    rarange : tuple
        RA range.

    decrange : tuple
        Dec range.
    """
    boxsize = _make_array(boxsize, 3)
    deltara = np.arcsin(boxsize[1]/2./dmax)
    deltadec = np.arcsin(boxsize[2]/2./dmax)
    dmin = (dmax - boxsize[0])/min(np.cos(deltara), np.cos(deltadec))
    deltara *= 180./np.pi
    deltadec *= 180./np.pi
    return [dmin, dmax], [-deltara, deltara], [-deltadec, deltadec]


def cutsky_to_box(drange, rarange, decrange, return_isometry=False):
    """
    Compute minimal box size required to cover input distance, RA, Dec ranges.

    Parameters
    ----------
    drange : tuple, array
        Distance range (dmin, dmax).

    rarange : tuple, array
        RA range.

    decrange : tuple, array
        Dec range.

    return_isometry : bool, default=False
        If ``True``, also return :class:`EuclideanIsometry` instance
        to apply to the initial positions in the box (centered at (0,0,0)).
        Translation along 'x' and rotation about 'y' and 'z' axes.

    Returns
    -------
    boxsize : array
        Minimal box size.

    isometry : EuclideanIsometry
        If ``return_isometry`` is ``True``, :class:`EuclideanIsometry` instance
        to apply to the initial positions.
    """
    rarange = utils.wrap_angle(rarange, degree=True)
    if rarange[1] < rarange[0]: rarange[0] -= 360.
    deltara = abs(rarange[1]-rarange[0])/2.*np.pi/180.
    deltadec = abs(decrange[1]-decrange[0])/2.*np.pi/180.
    boxsize = np.empty(3, dtype='f8')
    boxsize[1] = 2.*drange[1]*np.sin(deltara)
    boxsize[2] = 2.*drange[1]*np.sin(deltadec)
    boxsize[0] = drange[1] - drange[0]*min(np.cos(deltara),np.cos(deltadec))
    if return_isometry:
        isometry = EuclideanIsometry()
        isometry.translation(drange[1]-boxsize[0]/2., axis='x', frame='origin')
        isometry.rotation(-(decrange[0]+decrange[1])/2., axis='y', degree=True, frame='origin') # minus because of direction convention
        isometry.rotation((rarange[0]+rarange[1])/2., axis='z', degree=True, frame='origin')
        return boxsize, isometry
    return boxsize


class DistanceToRedshift(BaseClass):

    """Class that holds a conversion distance -> redshift."""

    def __init__(self, distance, zmax=100., nz=2048, interp_order=3):
        """
        Initialize :class:`DistanceToRedshift`.
        Creates an array of redshift -> distance in log(redshift) and instantiates
        a spline interpolator distance -> redshift.

        Parameters
        ----------
        distance : callable
            Callable that provides distance as a function of redshift (array).

        zmax : float, default=100.
            Maximum redshift for redshift <-> distance mapping.

        nz : int, default=2048
            Number of points for redshift <-> distance mapping.

        interp_order : int, default=3
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.
        """
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        zgrid = np.logspace(-8,np.log10(self.zmax),self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        self.interp = interpolate.UnivariateSpline(self.rgrid,self.zgrid,k=interp_order,s=0)

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)


class RedshiftDensityInterpolator(BaseClass):
    """
    Class that computes and interpolates a redshift density histogram :math:`n(z)` from an array of redshift and optionally weights.
    Adapted from: https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/zhist.py
    """

    @CurrentMPIComm.enable
    def __init__(self, redshifts, weights=None, bins=None, fsky=1., radial_distance=None, interp_order=1, mpicomm=None):
        r"""
        Initialize :class:`RedshiftDensityInterpolator`.

        Parameters
        ----------
        redshifts : array
            Array of redshifts.

        weights : array, default=None
            Array of weights, same shape as ``redshifts``. Defaults to 1.

        bins : int, array, string, default=None
            If `bins` is an integer, it defines the number of equal-width
            bins in the given range. If `bins` is a sequence, it defines the bin
            edges, including the rightmost edge, allowing for non-uniform bin widths.
            If 'scott', Scott's rule is used to estimate the optimal bin width
            from the input data. Defaults to 'scott'.

        fsky : float, default=1
            The sky area fraction, which is used in the volume calculation when normalizing :math:`n(z)`.
            ``1`` corresponds to full-sky: :math:`4 \pi` or :math:`\simeq 41253\; \mathrm{deg}^{2}`.

        radial_distance : callable, default=None
            Radial distance to use when converting redshifts into comoving distance.
            If ``None``, ``redshifts`` and optionally ``bins`` are assumed to be in distance units.

        interp_order : int, default=1
            Interpolation order, e.g. ``1`` for linear interpolation, ``3`` for cubic splines.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        def zrange(redshifts):
            if self.is_mpi_scattered():
                return mpi.min_array(redshifts,mpicomm=self.mpicomm),mpi.max_array(redshifts,mpicomm=self.mpicomm)
            else:
                zrange = None
                if self.is_mpi_root():
                    zrange = np.min(redshifts),np.max(redshifts)
                return self.mpicomm.bcast(zrange,root=self.mpiroot)

        if bins is None or bins == 'scott':
            # scott's rule
            if self.is_mpi_scatter():
                var = mpi.var_array(redshifts,aweights=weights,ddof=1,mpicomm=self.mpicomm)
                gsize = mpi.size_array(redshifts)
            else:
                var,gsize = None,None
                if self.is_mpi_root():
                    var = np.cov(redshifts,aweights=weights,ddof=1)
                    gsize = redshifts.size
                var,gsize = self.mpicomm.bcast((var,gsize),root=self.mpiroot)
            sigma = np.sqrt(var)
            dz = sigma * (24. * np.sqrt(np.pi) / gsize) ** (1. / 3)
            zrange = zrange(redshifts)
            nbins = np.ceil((maxval - minval) * 1. / dx)
            nbins = max(1, nbins)
            edges = minval + dx * np.arange(nbins + 1)

        if np.ndim(bins) == 0:
            bins = np.linspace(*zrange(redshifts),num=bins+1,endpoint=True)

        counts = np.histogram(redshifts,weights=weights,bins=bins)
        if self.is_mpi_scattered():
            counts = mpicomm.allreduce(counts,op=mpi.MPI.SUM)

        if radial_distance is not None:
            dbins = radial_distance(bins)
        else:
            dbins = bins
        dvol = fsky*4./3.*np.pi*(dbins[1:]**3 - dbins[:-1]**3)
        self.z = (bins[:-1] + bins[1:])/2.
        self.density = counts/dvol
        self.spline = interpolate.UnivariateSpline(self.z,self.density,k=interp_order,s=0)

    def __call__(self, z):
        """Return density at redshift ``z`` (scalar or array)."""
        return self.spline(z)


def _multiple_columns(column):
    return isinstance(column, (list,tuple))


def vectorize_columns(func):
    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not _multiple_columns(column):
            return func(self,column,**kwargs)
        toret = [func(self,col,**kwargs) for col in column]
        if all(t is None for t in toret): # in case not broadcast to all ranks
            return None
        return np.asarray(toret)
    return wrapper



class BaseCatalog(BaseClass):

    _attrs = ['attrs']

    """Class that represents a catalog, as a dictionary of columns stored as arrays."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, attrs=None, mpicomm=None, mpiroot=0):
        """
        Initialize :class:`BaseCatalog`.

        Parameters
        ----------
        data : dict, BaseCatalog
            Dictionary of {name: array}.

        columns : list, default=None
            List of column names.
            Defaults to ``data.keys()``.

        attrs : dict
            Other attributes.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
        self.data = {}
        if columns is None:
            columns = list((data or {}).keys())
        if data is not None:
            for name in columns:
                self[name] = data[name]
        self.attrs = attrs or {}
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    @classmethod
    def from_nbodykit(cls, catalog, columns=None):
        """
        Build new catalog from **nbodykit**.

        Parameters
        ----------
        catalog : nbodykit.base.catalog.CatalogSource
            **nbodykit** catalog.

        columns : list, default=None
            Columns to import. Defaults to all columns.

        Returns
        -------
        catalog : BaseCatalog
        """
        if columns is None: columns = catalog.columns
        data = {col: catalog[col].compute() for col in columns}
        return cls(data, mpicomm=catalog.comm, mpiroot=0, attrs=catalog.attrs)

    def to_nbodykit(self, columns=None):
        """
        Return catalog in **nbodykit** format.

        Parameters
        ----------
        columns : list, default=None
            Columns to export. Defaults to all columns.

        Returns
        -------
        catalog : nbodykit.source.catalog.ArrayCatalog
        """
        if columns is None: columns = self.columns()
        source = {col:self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        attrs = {key:value for key,value in self.attrs.items() if key != 'fitshdr'}
        return ArrayCatalog(source, **attrs)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        keys = list(self.data.keys())
        if not keys or self[keys[0]] is None:
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def gsize(self):
        """Return catalog global size, i.e. sum of size in each process."""
        return self.mpicomm.allreduce(len(self))

    def columns(self, include=None, exclude=None):
        """
        Return catalog column names, after optional selections.

        Parameters
        ----------
        include : list, string, default=None
            Single or list of *regex* patterns to select column names to include.
            Defaults to all columns.

        exclude : list, string, default=None
            Single or list of *regex* patterns to select column names to exclude.
            Defaults to no columns.

        Returns
        -------
        columns : list
            Return catalog column names, after optional selections.
        """
        toret = None

        if self.is_mpi_root():
            toret = allcols = list(self.data.keys())

            def toregex(name):
                return name.replace('.','\.').replace('*','(.*)')

            if include is not None:
                if not isinstance(include,(tuple,list)):
                    include = [include]
                toret = []
                for inc in include:
                    inc = toregex(inc)
                    for col in allcols:
                        if re.match(inc,str(col)):
                            toret.append(col)
                allcols = toret

            if exclude is not None:
                if not isinstance(exclude,(tuple,list)):
                    exclude = [exclude]
                toret = []
                for exc in exclude:
                    exc = toregex(exc)
                    for col in allcols:
                        if re.match(exc,str(col)) is None:
                            toret.append(col)

        return self.mpicomm.bcast(toret,root=self.mpiroot)

    def __contains__(self, column):
        """Whether catalog contains column name ``column``."""
        return column in self.data

    def gindices(self):
        """Row numbers in the global catalog."""
        sizes = self.mpicomm.allgather(len(self))
        sizes = [0] + np.cumsum(sizes[:1]).tolist()
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, dtype='f8'):
        """Return array of size :attr:`size` filled with zero."""
        return np.zeros(len(self),dtype=dtype)

    def ones(self, dtype='f8'):
        """Return array of size :attr:`size` filled with one."""
        return np.ones(len(self),dtype=dtype)

    def full(self, fill_value, dtype='f8'):
        """Return array of size :attr:`size` filled with ``fill_value``."""
        return np.full(len(self),fill_value,dtype=dtype)

    def falses(self):
        """Return array of size :attr:`size` filled with ``False``."""
        return self.zeros(dtype=np.bool_)

    def trues(self):
        """Return array of size :attr:`size` filled with ``True``."""
        return self.ones(dtype=np.bool_)

    def nans(self):
        """Return array of size :attr:`size` filled with :attr:`numpy.nan`."""
        return self.ones()*np.nan

    def get(self, column, *args, **kwargs):
        """Return catalog (local) column ``column`` if exists, else return provided default."""
        has_default = False
        if args:
            if len(args) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = args[0]
        if kwargs:
            if len(kwargs) > 1:
                raise SyntaxError('Too many arguments!')
            has_default = True
            default = kwargs['default']
        if column not in self.data and has_default:
            return default
        return self.data[column]

    def set(self, column, item):
        """Set column of name ``column``."""
        self.data[column] = item

    def gget(self, column, mpiroot=None):
        """
        Return on process rank ``root`` catalog global column ``column`` if exists, else return provided default.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` return result on all processes.
        """
        if mpiroot is None: mpiroot = Ellipsis
        return mpi.gather_array(self[column], mpicomm=self.mpicomm, root=mpiroot)

    def gslice(self, *args):
        """
        Perform global slicing of catalog,
        e.g. ``catalog.gslice(0,100,1)`` will return a new catalog of global size ``100``.
        Same reference to :attr:`attrs`.
        """
        sl = slice(*args)
        new = self.copy()
        for col in self.columns():
            self_value = self.gget(col,mpiroot=self.mpiroot)
            new[col] = mpi.scatter_array(self_value if self.is_mpi_root() else None,mpiroot=self.mpiroot,mpicomm=self.mpicomm)
        return new

    def to_array(self, columns=None, struct=True):
        """
        Return catalog as *numpy* array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all catalog columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, *numpy* will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if columns is None:
            columns = self.columns()
        if struct:
            toret = np.empty(self.size,dtype=[(col,self[col].dtype,self[col].shape[1:]) for col in columns])
            for col in columns: toret[col] = self[col]
            return toret
        return np.array([self[col] for col in columns])

    @classmethod
    @CurrentMPIComm.enable
    def from_array(cls, array, columns=None, mpicomm=None, mpiroot=0, **kwargs):
        """
        Build :class:`BaseCatalog` from input ``array``.

        Parameters
        ----------
        columns : list
            List of columns to read from array.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpistate : string, mpi.CurrentMPIState
            MPI state of the input array: 'scattered', 'gathered', 'broadcast'?

        mpicomm : MPI communicator, default=None
            MPI communicator.

        kwargs : dict
            Other arguments for :meth:`__init__`.

        Returns
        -------
        catalog : BaseCatalog
        """
        isstruct = None
        if mpicomm.rank == mpiroot:
            isstruct = array.dtype.names is not None
            if isstruct:
                if columns is None: columns = array.dtype.names
        isstruct = mpicomm.bcast(isstruct,root=mpiroot)
        columns = mpicomm.bcast(columns,root=mpiroot)
        new = cls(data=dict.fromkeys(columns),mpiroot=mpiroot,mpicomm=mpicomm,**kwargs)
        if isstruct:
            new.data = {col:array[col] for col in columns}
        else:
            new.data = {col:arr for col,arr in zip(columns,array)}
        return new

    def copy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
        new = super(BaseCatalog,self).__copy__()
        if columns is None: columns = self.columns()
        new.data = {col:self[col] if col in self else None for col in columns}
        import copy
        for name in new._attrs:
            if hasattr(self, name):
                tmp = copy.copy(getattr(self, name))
                setattr(new, name, tmp)
        return new

    def deepcopy(self):
        """Return deep copy."""
        import copy
        return copy.deepcopy(self)

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        state = {'data':data}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.__dict__.update(state)

    @classmethod
    @CurrentMPIComm.enable
    def from_state(cls, state, mpiroot=0, mpicomm=None):
        """Create class from state."""
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def __getitem__(self, name):
        """Get catalog column ``name`` if string, else return copy with local slice."""
        if isinstance(name,str):
            return self.get(name)
        new = self.copy()
        new.attrs = self.attrs.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __setitem__(self, name, item):
        """Set catalog column ``name`` if string, else set slice ``name`` of all columns to ``item``."""
        if isinstance(name,str):
            return self.set(name,item)
        for col in self.data:
            self[col][name] = item

    def __delitem__(self, name):
        """Delete column ``name``."""
        del self.data[name]

    def __repr__(self):
        """Return string representation of catalog, including global size and columns."""
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__,self.gsize,self.columns())

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate catalogs together.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs
        new_columns = new.columns()

        for other in others:
            other_columns = other.columns()
            assert new.mpicomm is other.mpicomm
            if new_columns and other_columns and set(other_columns) != set(new_columns):
                raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns,new_columns))

        for column in new_columns:
            columns = [other.gget(column,root=new.mpiroot) for other in others]
            if new.is_mpi_root():
                new[column] = np.concatenate(columns,axis=0)
            new[column] = mpi.scatter_array(new[column] if new.is_mpi_root() else None,root=new.mpiroot,mpicomm=new.mpicomm)
        return new

    def extend(self, other):
        """Extend catalog with ``other``."""
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and columns? (ignoring :attr:`attrs`)"""
        if not isinstance(other,self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        assert self.mpicomm == other.mpicomm
        toret = True
        for col in self_columns:
            self_value = self.gget(col,mpiroot=self.mpiroot)
            other_value = other.gget(col,mpiroot=self.mpiroot)
            if self.is_mpi_root():
                if not np.all(self_value == other_value):
                    toret = False
                    break
        return self.mpicomm.bcast(toret,root=self.mpiroot)

    @classmethod
    @CurrentMPIComm.enable
    def load_fits(cls, filename, columns=None, ext=None, mpiroot=0, mpicomm=None):
        """
        Load catalog in *fits* binary format from disk.

        Parameters
        ----------
        columns : list, default=None
            List of column names to read. Defaults to all columns.

        ext : int, default=None
            *fits* extension. Defaults to first extension with data.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        if mpicomm.rank == mpiroot:
            cls.log_info('Loading {}.'.format(filename))
        import fitsio
        # Stolen from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/fits.py
        msg = 'Input FITS file {}'.format(filename)
        with fitsio.FITS(filename) as file:
            if ext is None:
                for i, hdu in enumerate(file):
                    if hdu.has_data():
                        ext = i
                        break
                if ext is None:
                    raise IOError('{} has no binary table to read'.format(msg))
            else:
                if isinstance(ext,str):
                    if ext not in file:
                        raise IOError('{} does not contain extension with name {}'.format(msg,ext))
                elif ext >= len(file):
                    raise IOError('{} extension {} is not valid'.format(msg,ext))
            file = file[ext]
            # make sure we crash if data is wrong or missing
            if not file.has_data() or file.get_exttype() == 'IMAGE_HDU':
                raise ValueError('{} extension {} is not a readable binary table'.format(msg,ext))
            size = file.get_nrows()
            start = mpicomm.rank * size // mpicomm.size
            stop = (mpicomm.rank + 1) * size // mpicomm.size
            new = file.read(ext=ext,columns=columns,rows=range(start,stop))
            header = file.read_header()
            header.clean()
            attrs = dict(header)
            attrs['fitshdr'] = header
            new = cls.from_array(new, attrs=attrs, mpiroot=mpiroot, mpicomm=mpicomm)
        return new

    def save_fits(self, filename):
        """Save catalog to ``filename`` as *fits* file. Possible to change fitsio to write by chunks?."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        import fitsio
        array = self.to_array(struct=True)
        array = mpi.gather_array(array,mpicomm=self.mpicomm,root=self.mpiroot)
        if self.is_mpi_root():
            fitsio.write(filename,array,header=self.attrs.get('fitshdr',None),clobber=True)

    @classmethod
    @CurrentMPIComm.enable
    def load_hdf5(cls, filename, group='/', columns=None, mpiroot=0, mpicomm=None):
        """
        Load catalog in *hdf5* binary format from disk.

        Parameters
        ----------
        group : string, default='/'
            HDF5 group where columns are located.

        columns : list, default=None
            List of column names to read. Defaults to all columns in ``group``.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        if mpicomm.rank == mpiroot:
            cls.log_info('Loading {}.'.format(filename))
        import h5py
        with h5py.File(filename, 'r') as file:
            attrs = dict(file.attrs)
            grp = file[group]
            data = {}

            def set_dataset(name, obj):
                data[name] = obj

            grp.visititems(set_dataset)
            if columns is not None:
                data = {name:value for name,value in data.items() if name in columns}
                if set(data.keys()) != set(columns):
                    raise ValueError('Could not find columns {}'.format(set(columns) - set(data.keys())))
            for name in data:
                gsize = data[name].size
                break
            size = mpi.local_size(gsize, mpicomm=mpicomm)
            csizes = np.cumsum([0] + mpicomm.allgather(size))
            rank = mpicomm.rank
            for name in data:
                if data[name].size != gsize:
                    raise ValueError('Column {} has different length (expected {:d}, found {:d})'.format(name, gsize, data[name].size))
                data[name] = data[name][csizes[rank]:csizes[rank+1]]
        return cls(data=data, attrs=attrs)

    def save_hdf5(self, filename, group='/'):
        """Save catalog to ``filename`` as hdf5* file."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        driver = 'mpio'
        kwargs = {'comm': self.mpicomm}
        import h5py
        try:
            h5py.File(filename, 'w', driver=driver, **kwargs)
        except ValueError:
            driver = None
            kwargs = {}
        with h5py.File(filename, 'w', driver=driver, **kwargs) as file:
            csizes = np.cumsum([0] + self.mpicomm.allgather(self.size))
            gsize = csizes[-1]
            grp = file
            if group != '/':
                grp = file.create_group(group)
            grp.attrs.update(self.attrs)
            for name in self.columns():
                if driver == 'mpio':
                    rank = self.mpicomm.rank
                    dset = grp.create_dataset(name, shape=(gsize,)+self[name].shape[1:], dtype=self[name].dtype)
                    dset[csizes[rank]:csizes[rank+1]] = self[name]
                else:
                    dset = grp.create_dataset(name, data=self.gget(name))

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, filename, columns=None, mpiroot=0, mpicomm=None):
        """
        Load catalog in *npy* binary format from disk.

        Parameters
        ----------
        columns : list, default=None
            List of column names to read. Defaults to all columns.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        if mpicomm.rank == mpiroot:
            cls.log_info('Loading {}.'.format(filename))
            state = np.load(filename, allow_pickle=True)[()]
            data = state.pop('data')
            if columns is None: columns = list(data.keys())
        else:
            state = None
        state = mpicomm.bcast(state, root=mpiroot)
        columns = mpicomm.bcast(columns, root=mpiroot)
        state['data'] = {}
        for name in columns:
            state['data'][name] = mpi.scatter_array(data[name] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, root=mpiroot)
        return cls.from_state(state, mpicomm=mpicomm, mpiroot=mpiroot)

    def save(self, filename):
        """Save catalog to ``filename`` as *npy* file."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        state['data'] = {name: self.gget(name) for name in self.columns()}
        if self.is_mpi_root():
            np.save(filename, state, allow_pickle=True)

    @vectorize_columns
    def sum(self, column, axis=0):
        """Return global sum of column(s) ``column``."""
        return mpi.sum_array(self[column],axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def average(self, column, weights=None, axis=0):
        """Return global average of column(s) ``column``, with weights ``weights`` (defaults to ``1``)."""
        return mpi.average_array(self[column],weights=weights,axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def mean(self, column, axis=0):
        """Return global mean of column(s) ``column``."""
        return self.average(column,axis=axis)

    @vectorize_columns
    def minimum(self, column, axis=0):
        """Return global minimum of column(s) ``column``."""
        return mpi.min_array(self[column],axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def maximum(self, column, axis=0):
        """Return global maximum of column(s) ``column``."""
        return mpi.max_array(self[column],axis=axis,mpicomm=self.mpicomm)


class ParticleCatalog(BaseCatalog):

    _attrs = BaseCatalog._attrs + ['_position', '_velocity', '_vectors', '_translational_invariants']

    """A catalog of particles, with 'Position' and 'Velocity'."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, position='Position', velocity='Velocity', vectors=None, translational_invariants=None, **kwargs):
        """
        Initialize :class:`ParticleCatalog`.

        Parameters
        ----------
        data : dict, BaseCatalog
            Dictionary of {name: array}.

        columns : list, default=None
            List of column names.
            Defaults to ``data.keys()``.

        position : string, default='Position'
            Column name of positions in ``data``.

        velocity : string, default='Velocity'
            Column name of velocities in ``data``.

        vectors : list, tuple, set, default=None
            Names of columns which live in Cartesian space
            (position and velocity columns will be added).

        translational_invariants : list, tuple, set, default=None
            Names of columns (of ``vectors``) which are invariant under translation, e.g. velocities.

        kwargs : dict
            Other optional arguments, see :class:`ParticleCatalog`.
        """
        super(ParticleCatalog, self).__init__(data=data, columns=columns, **kwargs)
        self._position = position
        self._velocity = velocity
        self._vectors = set(vectors or [])
        self._translational_invariants = set(translational_invariants or [])
        self._vectors |= set([self._position, self._velocity])
        self._translational_invariants |= set([self._velocity])

    @property
    def vectors(self):
        """Names of current columns which live in Cartesian space."""
        return set(self._vectors) & set(self.columns())

    @property
    def translational_invariants(self):
        """Names of current columns (of :attr:`vectors`) which are invariant under translation, e.g. velocities."""
        return set(self._translational_invariants) & set(self.vectors)

    @property
    def position(self):
        """Cartesian positions."""
        return self[self._position]

    @property
    def velocity(self):
        """Velocities."""
        return self[self._velocity]

    def distance(self):
        """Distance."""
        return utils.distance(self.position)

    def rsd_position(self, f=1., los=None):
        """
        Return :attr:`position` with redshift space distortions applied.

        Parameters
        ----------
        f : callable, float
            Relation between the velocity and the RSD displacement.
            If a callable, take the (flattened) distance to the observer as input, i.e. :math:`f(r) \psi'.
            Else, a float to multiply the velocity.

        los : 'x', 'y', 'z'; int, 3-vector of int, default=None
            Line of sight :math:`\hat{\eta}` for RSD.
            If ``None``, use local line of sight.
        """
        if los is None:
            los = self.position/self.distance()
        else:
            los = _get_los(los)
        rsd = utils.vector_projection(self.velocity, los)
        iscallable = callable(f)
        if iscallable:
            rsd *= f(self.distance())
        else:
            rsd *= f
        return position + rsd


class CutskyCatalog(ParticleCatalog):

    """A catalog of particles, with a survey (cutsky) geometry."""

    def isometry(self, isometry):
        """
        Apply input isometry to catalog.

        Parameters
        ----------
        isometry : EuclideanIsometry
            Isometry to apply to catalog.
        """
        for name in self.vectors:
            self[name] = isometry.transform(self[name], translational_invariant=name in self._translational_invariants)


class BoxCatalog(ParticleCatalog):

    _attrs = ParticleCatalog._attrs + ['_boxsize', '_boxcenter']

    """A catalog of particles, with a box geometry."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, boxsize=None, boxcenter=0., **kwargs):
        """
        Initialize :class:`BoxCatalog`.

        Parameters
        ----------
        data : dict, BaseCatalog
            Dictionary of {name: array}.

        columns : list, default=None
            List of column names.
            Defaults to ``data.keys()``.

        boxsize : float, 3-vector of floats
            Box size.

        boxcenter : float, 3-vector of floats, default=0.
            Box center.

        kwargs : dict
            Other optional arguments, see :class:`ParticleCatalog`.
        """
        super(BoxCatalog, self).__init__(data=data, columns=columns, **kwargs)
        if boxsize is None:
            boxsize = self.attrs['boxsize']
        if boxcenter is None:
            boxcenter = self.attrs.get('boxcenter', 0.)
        self._boxsize = _make_array(boxsize, 3)
        self._boxcenter = _make_array(boxcenter, 3)

    def recenter(self):
        """Recenter box at origin."""
        operation = EuclideanIsometry()
        operation.translation(-self.boxcenter)
        self.boxcenter = operation.transform(self.boxcenter)
        for name in self.vectors:
            self[name] = operation.transform(self[name], translational_invariant=name in self._translational_invariants)

    @property
    def boxsize(self):
        """Box size."""
        return self._boxsize

    @boxsize.setter
    def boxsize(self, boxsize):
        """Set box size."""
        self._boxsize = _make_array(boxsize, 3, dtype='f8')

    @property
    def boxcenter(self):
        """Box center."""
        return self._boxcenter

    @boxcenter.setter
    def boxcenter(self, boxcenter):
        """Set box center."""
        self._boxcenter = _make_array(boxcenter, 3, dtype='f8')

    def glos(self):
        """Return unit vector to the box center."""
        return self.boxcenter/utils.distance(self.boxcenter)

    def remap(self, cuboid):
        """
        Remap box catalog.

        Parameters
        ----------
        cuboid : :class:`remap.Cuboid`
            Cuboid instance for remapping.
        """
        ofileset = self.boxcenter - self.boxsize/2.
        for name in self.vectors:
            if name not in self._translational_invariants:
                self[name] = cuboid.transform(self[name] - ofileset, boxsize=self.boxsize) + ofileset

    def subbox(self, ranges=(0,1), boxsize_unit=True):
          """
          Return new catalog brangeed to input ranges.

          Parameters
          ----------
          ranges : tuple, list of tuples
              Cartesian ranges (min, max) for the new catalog.
              Can be provided for each axis, with a list of tuples.

          boxsize_unit : bool, default=True
              ``True`` if input ranges are in units of :attr:`boxsize` (typically between 0 and 1).
              ``False`` if input ranges are in Cartesian space, with the same unit as :attr:`boxsize`
              (typically between ``self.boxcenter - self.boxsize/2.`` and ``self.boxcenter + self.boxsize/2.``).

          Returns
          -------
          new : BoxCatalog
              Catalog cut to provided ranges.
          """
          if np.ndim(ranges[0]) == 0:
              ranges = [ranges]*3
          if len(ranges) != 3:
              raise ValueError('Provide ranges for each axis')
          mask = self.trues()
          ofileset = [0]*3
          if boxsize_unit:
              ofileset = self.boxcenter - self.boxsize/2.
          current_box = [self.boxcenter - self.boxsize/2., self.boxcenter + self.boxsize/2.]
          box = [np.empty(3, dtype='f8') for i in range(2)]
          for ii, brange in enumerate(ranges):
              if boxsize_unit:
                  brange = [self.boxsize[ii] * bb + ofileset[ii] for bb in brange]
              box[0][ii] = max(current_box[0][ii], brange[0])
              box[1][ii] = min(current_box[1][ii], brange[1])
              mask &= (self.position[:,ii] >= brange[0]) & (self.position[:,ii] < brange[1])
          new = self[mask]
          new.boxsize = np.asarray(box[1]) - np.asarray(box[0])
          new.boxcenter = (np.asarray(box[1]) + np.asarray(box[0]))/2.
          return new

    def pad(self, factor=1.1):
        """
        Return new catalog padded by input factor.
        Peridodic boundary conditions are assumed -- and the output catalog is not periodic anymore,
        except if ``factor`` is an integer.

        Parameters
        ----------
        factor : tuple, list, float
            Padding factor (optionally along each axis).

        Returns
        -------
        new : BoxCatalog
            Padded catalog.
        """
        factors = _make_array(factor, 3, dtype='f8')
        new = self.copy()
        # only boxsize changes; boxcenter does not move as box is padded by the same amount on both sides
        new.boxsize *= factors
        position = self._position
        shifts = [np.arange(-np.ceil(factor)+1, np.ceil(factor)) for factor in factors]
        data = {col:[] for col in new}
        for shift in itertools.product(shifts):
            tmp = {col: self[col] + self.boxsize*shift for col in replicate}
            mask = (tmp[position] >= new.boxcenter - new.boxsize/2.) & (tmp[position] <= new.boxcenter + new.boxsize/2.)
            mask = np.all(mask, axis=-1)
            for col in new:
                if col in self._vectors and col not in self._translational_invariants:
                    data[col].append(tmp[col][mask])
                else:
                    data[col].append(self[col][mask])
        for col in new:
            new[col] = np.concatenate(data[col], axis=0)
        return new


    def cutsky(self, drange, rarange, decrange, external_margin=None, internal_margin=None, noutput=1):
        """
        Cut box to sky geometry.

        Parameters
        ----------
        drange : tuple, array
            Distance range (dmin, dmax).

        rarange : tuple, array
            RA range.

        decrange : tuple, array
            Dec range.

        external_margin : float, tuple, list, default=None
            Margin to apply on box edges to avoid spurious large scale correlations
            when no periodic boundary conditions are applied.
            If ``None``, set to ``internal_margin`` and maximize margin.

        internal_margin : float, tuple, list, default=None
            Margin between subboxes to reduce correlation between output catalogs.
            If ``None``, set to ``external_margin`` and maximize margin.

        noutput : int, default=1
            Number of output catalogs.
            If ``1``, one catalog is output.
            Else, if ``None``, cut maximum number of disjoint catalogs.
            Else, return this number of catalogs.

        Returns
        -------
        catalog : CutskyCatalog
        """
        if np.ndim(external_margin) == 0:
            external_margin = [external_margin]*3
        if np.ndim(internal_margin) == 0:
            internal_margin = [internal_margin]*3
        boxsize, isometry = cutsky_to_box(drange, rarange, decrange, return_isometry=True)
        boxes = []
        if noutput == 1:
            em = np.array([0. if m is None else m for m in external_margin])
            factor = (self.boxsize - 2.*em) / boxsize
            if not np.all(factor > 1.):
                raise ValueError('boxsize {} with margin {} is too small for input survey geometry which requires {}'.format(self.boxsize, external_margin, boxsize))
            box = self.subbox()
            box.recenter()
            boxes.append(box)
        else:
            ranges = []
            for iaxis in range(3):
                # boxsize * nboxes + internal_margin * (nboxes - 1) + 2 * external_margin = self.boxsize
                em = external_margin[iaxis] or 0.
                im = internal_margin[iaxis] or 0.
                nboxes = int((self.boxsize[iaxis] - 2.*em + im)/(boxsize[iaxis] + im))
                if external_margin[iaxis] is None and internal_margin[iaxis] is None:
                    tm = self.boxsize[iaxis] - boxsize[iaxis] * nboxes
                    em = im = tm / (nboxes + 1.)
                if external_margin[iaxis] is None: em = im
                if internal_margin[iaxis] is None: im = em
                if (em < 0) or (im < 0):
                    raise ValueError('margins must be > 0')
                if (em > 0.) or (im > 0.):
                    # extend margins to occupy full volume (same rescaling factor assumed for internal and external)
                    rescale_margin = (self.boxsize[iaxis] - boxsize[iaxis] * nboxes)/(im * (nboxes - 1) + 2 * em)
                    assert rescale_margin >= 1.
                    im = rescale_margin * im
                    em = rescale_margin * em
                tmp1 = em/self.boxsize[iaxis] + (boxsize[iaxis] + im) * np.arange(nboxes)/self.boxsize[iaxis]
                tmp2 = tmp1 + boxsize[iaxis]/self.boxsize[iaxis]
                ranges.append(list(zip(tmp1, tmp2))) # in boxsize_unit
            nmax = np.prod([len(r) for r in ranges])
            if noutput is not None and nmax < noutput:
                raise ValueError('can only cut {:d} catalogs from box, not noutput = {:d}'.format(nmax, noutput))
            ioutput = 0
            for ranges in itertools.product(*ranges):
                if noutput is not None and ioutput >= noutput: break
                box = self.subbox(ranges=ranges, boxsize_unit=True)
                box.recenter()
                boxes.append(box)
                ioutput += 1

        toret = []
        for box in boxes:
            catalog = CutskyCatalog(data=box.data, position=self._position, velocity=self._velocity, vectors=self.vectors, translational_invariants=self._translational_invariants, attrs=self.attrs)
            catalog.isometry(isometry)
            dist, ra, dec = utils.cartesian_to_sky(catalog.position, degree=True, wrap=False)
            mask_radial = UniformRadialMask(zrange=drange)
            mask_angular = UniformAngularMask(rarange=rarange, decrange=decrange)
            mask = mask_radial(dist) & mask_angular(ra, dec)
            catalog = catalog[mask]
            toret.append(catalog)
        if noutput == 1:
            toret = toret[0]
        return toret


class RandomBoxCatalog(BoxCatalog):

    """A catalog of random particles, with a box geometry."""

    @CurrentMPIComm.enable
    def __init__(self, boxsize=None, boxcenter=0., size=None, nbar=None, seed=None, **kwargs):
        """
        Initialize :class:`RandomBoxCatalog`, with a random sampling in 3D space.
        Set column ``position``.

        Parameters
        ----------
        boxsize : float, 3-vector of floats
            Box size.

        boxcenter : float, 3-vector of floats, default=0.
            Box center.

        size : float, default=None
            Global catalog size.

        nbar : float, default=None
            If ``size`` is ``None``, global catalog size is obtained as the nearest integer to ``nbar * volume``
            where ``volume`` is the box volume.

        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.

        kwargs : dict
            Other optional arguments, see :class:`ParticleCatalog`.
        """
        super(RandomBoxCatalog,self).__init__(data={}, boxsize=boxsize, boxcenter=boxcenter, **kwargs)
        self.attrs['seed'] = seed

        if size is None:
            size = int(nbar*np.prod(self.boxsize) + 0.5)
        size = mpi.local_size(size, mpicomm=self.mpicomm)
        rng = MPIRandomState(size=size, seed=seed, mpicomm=self.mpicomm)

        self[self._position] = np.array([rng.uniform(self.boxcenter[i] - self.boxsize[i]/2., self.boxcenter[i] + self.boxsize[i]/2.) for i in range(3)]).T


class RandomCutskyCatalog(CutskyCatalog):

    """A catalog of random particles, with a cutsky geometry."""

    @CurrentMPIComm.enable
    def __init__(self, rarange=(0.,360.), decrange=(-90.,90.), drange=None, size=None, nbar=None, seed=None, **kwargs):
        """
        Initialize :class:`RandomCutskyCatalog`, with a uniform sampling on the sky and as a function of distance.
        Set columns 'RA' (degree), 'DEC' (degree), 'Distance' and ``position``.

        Parameters
        ----------
        rarange : tuple, default=(0, 360)
            Range (min, max) of right ascension (degree).

        decrange : tuple
            Range (min, max) of declination (degree).

        drange : tuple, default=None
            Range (min, max) of distance.
            If ``None``, positions will be on the unit sphere (at distance 1).

        size : float, default=None
            Global catalog size.

        nbar : float, default=None
            If ``size`` is ``None``, global catalog size is obtained as the nearest integer to ``nbar * area``
            where ``area`` is the sky area.

        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.

        kwargs : dict
            Other optional arguments, see :class:`ParticleCatalog`.
        """
        super(RandomCutskyCatalog,self).__init__(data={}, **kwargs)
        area = utils.radecbox_area(rarange, decrange)
        if size is None:
            size = int(nbar*area + 0.5)
        self.attrs['seed'] = seed
        self.attrs['area'] = area

        size = mpi.local_size(size, mpicomm=self.mpicomm)

        seed1, seed2 = mpi.bcast_seed(seed=seed, size=2, mpicomm=self.mpicomm)
        mask = UniformAngularMask(rarange=rarange, decrange=decrange, mpicomm=self.mpicomm)
        self['RA'], self['DEC'] = mask.sample(size, seed=seed1)
        if drange is None:
            self['Distance'] = self.ones(dtype=self['RA'].dtype)
        else:
            mask = UniformRadialMask(zrange=drange, mpicomm=self.mpicomm)
            self['Distance'] = mask.sample(size, distance=lambda z: z, seed=seed2)
        self['Position'] = utils.sky_to_cartesian(self['Distance'], self['RA'], self['DEC'], degree=True)


class BaseMask(BaseClass):
    r"""
    Base template class to apply selection function.
    Subclasses should at least implement :meth:`prob`.
    """
    @CurrentMPIComm.enable
    def __init__(self, mpicomm=None, mpiroot=0):
        """
        Initialize :class:`BaseRadialMask`.

        Parameters
        ----------
        zrange : tuple, list
            Redshift range.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    def prob(self, *args, **kwargs):
        """Return selection probability; to be implemented in the subclass."""
        raise NotImplementedError('Implement method "prob" in your "{}"-inherited class'.format(self.__class__.___name__))

    def __call__(self, *args, seed=None, **kwargs):
        """
        Apply selection function to input redshifts with constant volume-density (i.e. distance**2 distance-density).

        Parameters
        ----------
        args : arrays of shape (N,)
            Variables (e.g. redshifts) for :meth:`prob`.

        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.

        kwargs : dict
            Optional arguments for :meth:`prob`.
        """
        prob = self.prob(*args, **kwargs)
        rng = MPIRandomState(size=prob.size, seed=seed, mpicomm=self.mpicomm)
        return prob >= rng.uniform(low=0., high=1.)


class MaskCollection(dict,BaseMask):
    """
    Dictionary of masks.
    Useful to apply chunkwise selection functions.
    """
    def prob(self, chunk, *args, **kwargs):
        """
        Return selection probability.

        Parameters
        ----------
        chunk : array of shape (N,)
            Chunk labels.

        args : arrays of shape (N,)
            Other variables, e.g. redshifts.

        kwargs : dict
            Optional arguments for :meth:`prob` of masks.

        Returns
        -------
        prob : array
            Selection probability.
        """
        chunk = np.asarray(chunk)
        prob = np.ones_like(chunk, dtype='f8')
        for ichunkz, density in self.items():
            mask = chunk == ichunkz
            if mask.any():
                prob[...,mask] = density.prob(*[arg[...,mask] for arg in args], **kwargs)
        return prob


class BaseRadialMask(BaseMask):
    r"""
    Base template class to apply :math:`n(z)` (in 3D volume unit) selection.
    Subclasses should at least implement :meth:`prob`.
    """
    @CurrentMPIComm.enable
    def __init__(self, zrange=None, mpicomm=None, mpiroot=0):
        """
        Initialize :class:`BaseRadialMask`.

        Parameters
        ----------
        zrange : tuple, list, default=None
            Redshift range.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
        self.zrange = tuple(zrange) if zrange is not None else (0., np.inf)
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot

    def sample(self, size, distance, seed=None):
        """
        Draw redshifts from radial selection function.
        This is a very naive implementation, drawing uniform samples and masking them with :meth:`__call__`.

        Parameters
        ----------
        size : int
            Local number of redshifts to sample.

        distance : callable
            Callable that provides distance as a function of redshift (array).
            Redshifts are sampled by applying :meth:`__call__` distribution with constant volume-density
            (i.e. distance-density following distance**2).

        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.

        Returns
        -------
        z : array of shape (size,)
            Array of sampled redshifts.
        """
        drange = distance(self.zrange)
        def sample(size, seed=None):
            rng = MPIRandomState(size=size, seed=seed, mpicomm=self.mpicomm)
            dist = rng.uniform(drange[0], drange[1])
            prob = dist**2 # jacobian, d^3 r = r^2 dr
            prob /= prob.max()
            mask = prob >= rng.uniform(low=0., high=1.)
            z = DistanceToRedshift(distance, zmax=self.zrange[-1] + 0.1)(dist[mask])
            mask = self(z)
            return z[mask]

        z = []
        dsize = size
        while dsize > 0:
            std = 1./np.sqrt(dsize)
            newsize = int(dsize*(1. + 3.*std) + 100.)
            seed = mpi.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None) + 1
            tmpz = sample(newsize, seed=seed)
            dsize -= tmpz.size
            z.append(tmpz)

        return np.concatenate(z, axis=0)[:size]


class UniformRadialMask(BaseRadialMask):

    r"""Uniform :math:`n(z)` selection."""

    @CurrentMPIComm.enable
    def __init__(self, nbar=1., **kwargs):
        self.nbar = nbar
        super(UniformRadialMask, self).__init__(**kwargs)

    def prob(self, z):
        """Uniform probability within :attr:`zrange`."""
        z = np.asarray(z)
        prob = np.clip(self.nbar, 0., 1.)*np.ones_like(z)
        mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
        prob[~mask] = 0.
        return prob


class TabulatedRadialMask(BaseRadialMask):

    r"""Tabulated :math:`n(z)` selection."""

    @CurrentMPIComm.enable
    def __init__(self, z, nbar=None, zrange=None, filename=None, norm=None, mpicomm=None, mpiroot=0):
        """
        Initialize :class:`TabulatedRadialMask`.

        Parameters
        ----------
        z : array, default=None
            Redshift array.

        nbar : array, default=None
            Density array.

        zrange : tuple, list, default=None
            Redshift range to restrict to.
            Defaults to ``(z[0], z[-1])``.

        filename : string, default=None
            If provided, file name to load ``z``, ``nbar`` from.

        norm : float, default=None
            Normalization factor to apply to ``nbar``.
            See :meth:`prepare`.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
        if filename is not None:
            z, nbar = None, None
            if self.is_mpi_root():
                self.log_info('Loading density file: {}.'.format(filename))
                z, nbar = np.loadtxt(filename, unpack=True)
            z = mpi.broadcast_array(z, mpicomm=self.mpicomm, root=self.mpiroot)
            nbar = mpi.broadcast_array(nbar, mpicomm=self.mpicomm, root=self.mpiroot)
        self.z, self.nbar = np.asarray(z), np.asarray(nbar)
        if not np.all(self.nbar >= 0.):
            raise ValueError('Provided nbar should be all positive.')
        zmin, zmax = self.z[0], self.z[-1]
        if zrange is None: zrange = zmin, zmax
        super(TabulatedRadialMask, self).__init__(zrange=zrange, mpicomm=mpicomm, mpiroot=mpiroot)
        if not ((zmin <= self.zrange[0]) & (zmax >= self.zrange[1])):
            raise ValueError('Redshift range is {:.2f} - {:.2f} but the limiting range is {:.2f} - {:.2f}.'.format(zmin,zmax,self.zrange[0],self.zrange[1]))
        self.prepare(norm=norm)

    def prepare(self, norm=None):
        """
        Set normalization factor :attr:`norm` and interpolation :attr:`interp`.

        Parameters
        ----------
        norm : float, default=None
            Factor to scale :attr:`nbar`.
            Defaults to maximum :attr:`nbar` in redshift range.
        """
        if norm is None: norm = 1./self.nbar[self.zmask].max(axis=0)
        self.norm = norm
        self._set_interp()
        return self.norm

    def _set_interp(self):
        # Set :attr:`interp`, the end user does not need to call it
        prob = np.clip(self.norm*self.nbar, 0., 1.)
        self.interp = interpolate.Akima1DInterpolator(self.z, prob, axis=0)

    @property
    def zmask(self):
        """Mask to limit :attr:`z`, :attr:`nz` to :attr:`zrange`."""
        return (self.z >= self.zrange[0]) & (self.z <= self.zrange[-1])

    def prob(self, z):
        """Return probability to select input redshift ``z``."""
        prob = self.interp(z)
        mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
        prob[~mask] = 0.
        return prob

    def integral(self, z=None, weights=None, normalize_weights=True):
        """
        Return integral of :meth:`prob`, i.e. the fraction of redshifts that will be selected.

        Parameters
        ----------
        z : array, default=None
            Redshift sample to evaluate :attr:`prob` at.
            If ``None``, numerically integrate :attr:`interp` over :attr:`zrange`.

        weights : array, default=None
            If ``z`` is provided, associate weights (default to 1.).

        normalize_weights : bool, default=True
            Whether to normalize input ``weights``.

        Returns
        -------
        integral : float
            Integral of :meth:`prob`.
        """
        if z is None and weights is None:
            return self.interp.integrate(self.zrange[0], self.zrange[1])
        if weights is not None and z is None:
            raise ValueError('Provide z when giving w')
        if weights is None:
            weights = 1.
            if normalize_weights:
                weights = weights/self.mpicomm.allreduce(z.size)
        elif normalize_weights:
            weights = weights/mpi.sum_array(weights, mpicomm=mpicomm)
        return mpi.sum_array(self.prob(z)*weights, mpicomm=self.mpicomm)

    def normalize(self, target, z=None, weights=None):
        """
        Normalize :attr:`nbar`.

        Parameters
        ----------
        target : float
            Target integral, i.e. the fraction of redshifts that will be selected.

        z : array, default=None
            Redshift sample to evaluate :attr:`prob` at.
            If ``None``, numerically integrate :attr:`interp` over :attr:`zrange`.

        weights : array, default=None
            If ``z`` is provided, associate weights (default to 1.).
        """
        if not ((target >= 0) & (target <= 1.)):
            raise ValueError('Input norm should be in (0, 1)')

        if weights is not None:
            weights = weights/mpi.sum_array(weights, mpicomm=mpicomm)

        def normalization(norm):
            self.norm = norm
            self._set_interp()
            return self.integral(z, weights, normalize_weights=False)/target - 1.

        min_ = self.nbar[self.nbar>0.].min()
        norm = optimize.brentq(normalization, 0., 1/min_) # the lowest point of n(z) is limited by 1.
        norm = self.mpicomm.bcast(norm, root=self.mpiroot)

        self.prepare(norm=norm)
        error = self.integral(z, weights, normalize_weights=False) - target
        if self.is_mpi_root():
            self.log_info('Norm is: {:.12g}.'.format(self.norm))
            self.log_info('Expected error: {:.5g}.'.format(error))

    def convert_to_cosmo(self, distance_self, distance_target, zedges=None):
        """
        Rescale volume-density :attr:`nbar` to target cosmology.

        Parameters
        ----------
        distance_self : callable
            Callable that provides distance as a function of redshift (array) in current cosmology.

        distance_other : callable
            Callable that provides distance as a function of redshift (array) in target cosmology.

        zedges : array, default=None
            The original redshift edges in case of binned :attr:`nbar` where shell volumes were proportional
            to ``distance_self(zedges[1:])**3-distance_self(dedges[:-1])**3``, in which case a perfect rescaling is achieved.
            Defaults to edges falling in the middle of :attr:`z` points.
        """
        if zedges is None:
            zedges = (self.z[:-1] + self.z[1:])/2.
            zedges = np.concatenate([self.z[0]],zedges,[self.z[-1]])
        dedges = distance_self(zedges)
        volume_self = dedges[1:]**3-dedges[:-1]**3
        dedges = distance_target(zedges)
        volume_target = dedges[1:]**3-dedges[:-1]**3
        self.nbar = self.nbar*volume_self/volume_target
        self.prepare()


class BaseAngularMask(BaseMask):
    r"""
    Base template class to apply :math:`n(ra, dec)` selection.
    Subclasses should at least implement :meth:`prob`.
    """
    @CurrentMPIComm.enable
    def __init__(self, rarange=None, decrange=None, mpicomm=None, mpiroot=0):
        """
        Initialize :class:`BaseRadialMask`.

        Parameters
        ----------
        rarange : tuple, list, default=None
            Right ascension range.

        decrange : tuple, list, default=None
            Declination range.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
        if rarange is not None:
            rarange = utils.wrap_angle(rarange, degree=True)
            if rarange[1] < rarange[0]: rarange[0] -= 360.
            self.rarange = tuple(rarange)
        else:
            self.rarange = (0., 360.)
        if decrange is not None:
            self.decrange = tuple(decrange)
        else:
            self.decrange = (-90., 90.)
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot

    def sample(self, size, seed=None):
        """
        Draw ra, dec from angular selection function.
        This is a very naive implementation, drawing uniform samples and masking them with :meth:`__call__`.

        Parameters
        ----------
        size : int
            Local number of RA, Dec positions to sample.

        seed : int, default=None
            The global random seed, used to set the seeds across all ranks.

        Returns
        -------
        ra : array of shape (size,)
            Array of sampled RA, Dec.
        """
        def sample(size, seed=None):
            rng = MPIRandomState(size=size, seed=seed, mpicomm=self.mpicomm)
            ra = rng.uniform(low=self.rarange[0], high=self.rarange[1])
            urange = np.sin(np.asarray(self.decrange)*np.pi/180.)
            dec = np.arcsin(rng.uniform(low=urange[0], high=urange[1]))/(np.pi/180.) % 360.
            mask = self(ra, dec)
            return ra[mask], dec[mask]

        ra, dec = [], []
        dsize = size
        while dsize > 0:
            std = 1./np.sqrt(dsize)
            newsize = int(dsize*(1. + 3.*std) + 100.)
            seed = mpi.bcast_seed(seed=seed, mpicomm=self.mpicomm, size=None) + 1
            tmpra, tmpdec = sample(newsize, seed=seed)
            dsize -= tmpra.size
            ra.append(tmpra)
            dec.append(tmpdec)

        return np.concatenate(ra, axis=0)[:size], np.concatenate(dec, axis=0)[:size]


class UniformAngularMask(BaseAngularMask):

    r"""Uniform :math:`n(ra, dec)` selection."""

    @CurrentMPIComm.enable
    def __init__(self, nbar=1., **kwargs):
        self.nbar = nbar
        super(UniformAngularMask, self).__init__(**kwargs)

    def prob(self, ra, dec):
        """Uniform probability within :attr:`rarange` and :attr:`decrange`."""
        ra, dec = utils.wrap_angle(ra, degree=True), np.asarray(dec)
        prob = np.clip(self.nbar, 0., 1.)*np.ones_like(ra)
        mask = (ra >= self.rarange[0]) & (ra <= self.rarange[-1])
        mask &= (dec >= self.decrange[0]) & (dec <= self.decrange[-1])
        prob[~mask] = 0
        return prob


try:
    import pymangle
    HAVE_PYMANGLE = True
except ImportError:
    HAVE_PYMANGLE = False

if HAVE_PYMANGLE:

    class MangleAngularMask(BaseAngularMask):

        """Angular mask based on Mangle."""

        def __init__(self, nbar=None, filename=None, **kwargs):
            """
            Initialize :class:`MangleAngularMask`.

            Parameters
            ----------
            nbar : pymangle.Mangle, default=None
                Mangle mask with method :meth:`pymangle.Mangle.polyid_and_weight`

            filename : string, default=None
                If provided, file name to load Mangle mask from.
                This requires **pymangle**.
            """
            super(MangleAngularMask,self).__init__(**kwargs)
            if filename is not None:
                if self.is_mpi_root():
                    self.log_info('Loading Mangle geometry file: {}.'.format(filename))
                self.nbar = pymangle.Mangle(filename)
            else:
                self.nbar = np.asarray(nbar)

        def prob(self, ra, dec):
            """
            Return selection probability.

            Parameters
            ----------
            ra : array
                Right ascension (degree).

            dec : array
                Declination (degree).

            Returns
            -------
            prob : array
                Selection probability.
            """
            ra, dec = utils.wrap_angle(ra, degree=True), np.asarray(dec)
            ids, prob = self.nbar.polyid_and_weight(ra, dec)
            mask = ids != -1
            mask &= (ra >= self.rarange[0]) & (ra <= self.rarange[-1])
            mask &= (dec >= self.decrange[0]) & (dec <= self.decrange[-1])
            prob[~mask] = 0.
            return prob


try:
    import healpy
    HAVE_HEALPY = True
except ImportError:
    HAVE_HEALPY = False

if HAVE_HEALPY:

    class HealpixAngularMask(BaseAngularMask):
        """
        Angular mask based on Healpix.
        This requires healpy.
        """
        def __init__(self, nbar=None, filename=None, nest=False, **kwargs):
            """
            Initialize :class:`MangleAngularMask`.

            Parameters
            ----------
            nbar : array
                Healpix nbar for each Healpix index.

            filename : string, default=None
                If provided, file name to load Healpix map from.

            nest : bool, default=False
                Nested scheme?
            """
            super(HealpixAngularMask,self).__init__(**kwargs)
            if filename is not None:
                if self.is_mpi_root():
                    self.log_info('Loading healpy geometry file: {}.'.format(filename))
                self.nbar = healpy.fitsfunc.read_map(filename, nest=nest, **kwargs)
            else:
                self.nbar = np.asarray(nbar)
            self.nside = healpy.npix2nside(self.nbar.size)
            self.nest = nest

        def prob(self, ra, dec):
            """
            Return selection probability.

            Parameters
            ----------
            ra : array
                Right ascension (degree).

            dec : array
                Declination (degree).

            Returns
            -------
            prob : array
                Selection probability.
            """
            ra, dec = utils.wrap_angle(ra, degree=True), np.asarray(dec)
            theta, phi = (-dec+90.)*np.pi/180., ra*np.pi/180.
            prob = self.nbar[healpy.ang2pix(self.nside,theta,phi,nest=self.nest,lonlat=False)]
            mask = (ra >= self.rarange[0]) & (ra <= self.rarange[-1])
            mask &= (dec >= self.decrange[0]) & (dec <= self.decrange[-1])
            prob[~mask] = 0.
            return prob

"""
Utilities to cut survey mocks from boxes.
A Python analogous of Cameron K. McBride https://github.com/cmcbride/make_survey.
"""

import os
import logging
import functools
import itertools

import numpy as np
from scipy import interpolate, optimize

from . import mpi, utils
from .mpi import CurrentMPIComm, MPIRandomState
from .catalog import BaseCatalog, BaseFile, FitsFile, HDF5File
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
    if isinstance(los, str):
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
            shift = np.tensordot(shift, self._rotation, axes=((-1,),(1,)))
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
        new = cls()
        if not others: return new
        for other in others:
            new._rotation = other._rotation.dot(new._rotation)
            new._translation = np.tensordot(new._translation, other._rotation, axes=((-1,),(1,))) + other._translation
            #new._translation += other._translation
        return new

    def extend(self, other):
        """Add ``other`` isometry to ``self``."""
        new = self.concatenate(self, other)
        self.__dict__.update(new.__dict__)

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
        return self.concatenate(self, other)


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
    # if e.g. rarange = (300, 40), we want to point to the middle of |-60, 40]
    if rarange[0] > rarange[1]: rarange[0] -= 360
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
            Interpolation order, e.g. 1 for linear interpolation, 3 for cubic splines.
        """
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        zgrid = np.logspace(-8,np.log10(self.zmax),self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        self.interp = interpolate.UnivariateSpline(self.rgrid, self.zgrid, k=interp_order, s=0, ext='raise')

    def __call__(self, distance):
        """Return (interpolated) redshift at distance ``distance`` (scalar or array)."""
        return self.interp(distance)


class RedshiftDensityInterpolator(BaseClass):
    """
    Class that computes and interpolates a redshift density histogram :math:`n(z)` from an array of redshift and optionally weights.
    Adapted from: https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/zhist.py

    Attributes
    ----------
    z : array
        Tabulated redshift array.

    nbar : array
        Tabulated density array.

    spline : sp.interpolate.UnivariateSpline
        Spline interpolator.
    """

    @CurrentMPIComm.enable
    def __init__(self, z, weights=None, bins=None, fsky=1., radial_distance=None, interp_order=1, mpicomm=None, mpiroot=None):
        r"""
        Initialize :class:`RedshiftDensityInterpolator`.

        Parameters
        ----------
        z : array
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

        mpiroot : int, default=None
            If ``None``, input z and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input z and weights are gathered.
        """
        self.mpicomm = mpicomm

        if mpiroot is not None:
            if not self.mpicomm.bcast(z is None, root=mpiroot):
                z = mpi.scatter_array(z, mpicomm=self.mpicomm, root=mpiroot)
            if not self.mpicomm.bcast(weights is None, root=mpiroot):
                weights = mpi.scatter_array(weights, mpicomm=self.mpicomm, root=mpiroot)

        def zrange(z):
            return mpi.min_array(z, mpicomm=self.mpicomm), mpi.max_array(z, mpicomm=self.mpicomm)

        if bins is None or isinstance(bins, str) and bins == 'scott':
            # scott's rule
            var = mpi.var_array(z, aweights=weights, ddof=1, mpicomm=self.mpicomm)
            size = mpi.size_array(z)
            sigma = np.sqrt(var)
            dx = sigma * (24. * np.sqrt(np.pi) / size) ** (1. / 3)
            minval, maxval = zrange(z)
            nbins = np.ceil((maxval - minval) * 1. / dx)
            nbins = max(1, nbins)
            bins = minval + dx * np.arange(nbins + 1)

        elif np.ndim(bins) == 0:
            bins = np.linspace(*zrange(z), num=bins+1, endpoint=True)

        counts = np.histogram(z, weights=weights, bins=bins)[0]
        counts = self.mpicomm.allreduce(counts, op=mpi.MPI.SUM)

        if radial_distance is not None:
            dbins = radial_distance(bins)
        else:
            dbins = bins
        dvol = fsky*4./3.*np.pi*(dbins[1:]**3 - dbins[:-1]**3)
        self.z = (bins[:-1] + bins[1:])/2.
        self.nbar = counts/dvol
        self.spline = interpolate.UnivariateSpline(self.z, self.nbar, k=interp_order, s=0, ext='zeros')

    def __call__(self, z):
        """Return density at redshift ``z`` (scalar or array)."""
        return self.spline(z)


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
            los = self.position/self.distance()[:,None]
        else:
            los = _get_los(los)
        rsd = utils.vector_projection(self.velocity, los)
        iscallable = callable(f)
        if iscallable:
            rsd *= f(self.distance())
        else:
            rsd *= f
        return self.position + rsd


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

    def remap(self, *args):
        """
        Remap box catalog.

        Parameters
        ----------
        args : :class:`remap.Cuboid`, 3-vectors
            Cuboid instance for remapping, or 3 lattice vectors.

        Returns
        -------
        new : BoxCatalog
            Remapped catalog.
        """
        from .remap import Cuboid
        if not args:
            raise ValueError('Provide either remap.Cuboid instance or 3 lattice vectors')
        if len(args) == 1:
            if isinstance(args[0], Cuboid):
                cuboid = args[0]
            else:
                args = args[0]
        if len(args) > 1:
            cuboid = Cuboid(*args, boxsize=self.boxsize)
        offset = self.boxcenter - self.boxsize/2.
        new = self.copy()
        new.boxsize = cuboid.cuboidsize
        cuboidoffset = new.boxcenter - new.boxsize/2.
        for name in self.vectors:
            if name not in self._translational_invariants:
                new[name] = cuboid.transform(self[name] - offset) + cuboidoffset
        return new

    def subbox(self, ranges=(0,1), boxsize_unit=True):
        """
        Return new catalog limited to input ranges.

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
        offset = [0]*3
        if boxsize_unit:
            offset = self.boxcenter - self.boxsize/2.
        current_box = [self.boxcenter - self.boxsize/2., self.boxcenter + self.boxsize/2.]
        box = [np.empty(3, dtype='f8') for i in range(2)]
        for ii, brange in enumerate(ranges):
            if boxsize_unit:
                brange = [self.boxsize[ii] * bb + offset[ii] for bb in brange]
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

    def isometry_for_cutsky(self, drange, rarange, decrange, external_margin=None, internal_margin=None, noutput=1):
        """
        Return operations (isometry, radial mask and angular mask) to cut box to sky geometry.

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
            Number of output sky-cuts.
            If ``1``, one :class:`EuclideanIsometry` is output.
            Else, if ``None``, maximum number of disjoint sky-cuts.
            Else, return this number of :class:`EuclideanIsometry`.

        Returns
        -------
        isometries : EuclideanIsometry or list of EuclideanIsometry instances
            Isometries to apply to current instance to move it (using :meth:`isometry`) to the desired sky location.
            If ``noutput`` is 1, only one :class:`EuclideanIsometry` instance, else a list of such instances.

        mask_radial : UniformRadialMask
            Radial mask (function of distance) to apply to the shifted catalog.

        mask_angular : UniformAngularMask
            Angular mask (function of RA, Dec) to apply to the shifted catalog.
        """
        if np.ndim(external_margin) == 0:
            external_margin = [external_margin]*3
        if np.ndim(internal_margin) == 0:
            internal_margin = [internal_margin]*3
        boxsize, origin_isometry = cutsky_to_box(drange, rarange, decrange, return_isometry=True)
        mask_radial = UniformRadialMask(zrange=drange)
        mask_angular = UniformAngularMask(rarange=rarange, decrange=decrange)
        if noutput == 1:
            em = np.array([0. if m is None else m for m in external_margin])
            factor = (self.boxsize - 2.*em) / boxsize
            if not np.all(factor > 1.):
                raise ValueError('boxsize {} with margin {} is too small for input survey geometry which requires {}'.format(self.boxsize, external_margin, boxsize))
            isometry = EuclideanIsometry()
            isometry.translation(-self.boxcenter)
            isometry += origin_isometry
            return isometry, mask_radial, mask_angular

        centers = []
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
            low = em + (boxsize[iaxis] + im) * np.arange(nboxes)
            high = low + boxsize[iaxis]
            center = (low + high)/2. - self.boxsize[iaxis]/2. + self.boxcenter[iaxis]
            centers.append(center)
        nmax = np.prod([len(center) for center in centers])
        if noutput is not None and nmax < noutput:
            raise ValueError('can only cut {:d} catalogs from box, not noutput = {:d}'.format(nmax, noutput))
        isometries = []
        for center in itertools.product(*centers):
            if noutput is not None and len(isometries) >= noutput: break
            isometry = EuclideanIsometry()
            isometry.translation(-np.array(center))
            isometry += origin_isometry
            isometries.append(isometry)

        return isometries, mask_radial, mask_angular

    def cutsky_from_isometry(self, isometry, mask_radial=None, mask_angular=None, dradec=('Distance', 'RA', 'DEC')):
        """
        Cut box to sky geometry, starting from distance, RA, and Dec ranges.
        Typically called with outputs of :meth:`isometry_for_cutsky`.

        Parameters
        ----------
        isometries : EuclideanIsometry or list of EuclideanIsometry instances
            Isometries to apply to current instance to move it (using :meth:`isometry`) to the desired sky location.
            If a single :class:`EuclideanIsometry` instance, a single catalog is returned.

        mask_radial : UniformRadialMask, default=None
            Radial mask (function of distance) to apply to the shifted catalog.
            If not provided, no radial mask is applied.

        mask_angular : UniformAngularMask, default=None
            Angular mask (function of RA, Dec) to apply to the shifted catalog.
            If not provided, no angular mask is applied.

        dradec : tuple or list of strings
            Names of columns to store distance, RA and Dec in output catalog.
            If ``None``, columns distance, RA and Dec are not added to output catalog.

        Returns
        -------
        catalog : CutskyCatalog or list of CutskyCatalog
            One catalog if a single :class:`EuclideanIsometry` instance provided, else a list of such catalogs.
        """
        islist = isinstance(isometry, (tuple, list))
        if islist:
            return [self.cutsky_from_isometry(isom, mask_radial=mask_radial, mask_angular=mask_angular) for isom in isometry]

        catalog = self.deepcopy()
        catalog = CutskyCatalog(data=catalog.data, position=catalog._position, velocity=catalog._velocity, vectors=catalog.vectors,
                                translational_invariants=catalog._translational_invariants, attrs=catalog.attrs)
        catalog.isometry(isometry)
        dist, ra, dec = utils.cartesian_to_sky(catalog.position, degree=True, wrap=False)
        if dradec is not None:
            for name, array in zip(dradec, [dist, ra, dec]):
                catalog[name] = array

        if mask_radial is not None:
            mask = mask_radial(dist)
            if mask_angular is not None:
                mask &= mask_angular(ra, dec)
            catalog = catalog[mask]
        elif mask_angular is not None:
            mask = mask_angular(ra, dec)
            catalog = catalog[mask]
        return catalog

    def cutsky(self, drange, rarange, decrange, external_margin=None, internal_margin=None, noutput=1, mask_radial=True, mask_angular=True, dradec=('Distance', 'RA', 'DEC')):
        """
        Cut box to sky geometry, starting from distance, RA, and Dec ranges.
        Basically a shortcut to :meth:`isometry_for_cutsky` and :meth:`cutsky_from_isometry`.

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

        mask_radial : UniformRadialMask, default=None
            Whether to apply radial mask.

        mask_angular : UniformAngularMask, default=None
            Whether to apply angular mask.

        dradec : tuple or list of strings
            Names of columns to store distance, RA and Dec in output catalog.
            If ``None``, columns distance, RA and Dec are not added to output catalog.

        Returns
        -------
        catalog : CutskyCatalog or list of CutskyCatalog
            One catalog if a single :class:`EuclideanIsometry` instance provided, else a list of such catalogs.
        """
        result = self.isometry_for_cutsky(drange, rarange, decrange,
                                          external_margin=external_margin, internal_margin=internal_margin,
                                          noutput=noutput)
        isometry = result[0]
        mask_radial = result[1] if mask_radial else None
        mask_angular = result[2] if mask_angular else None
        return self.cutsky_from_isometry(isometry, mask_radial=mask_radial, mask_angular=mask_angular)


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
    def __init__(self, mpicomm=None):
        """
        Initialize :class:`BaseRadialMask`.

        Parameters
        ----------
        zrange : tuple, list
            Redshift range.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.mpicomm = mpicomm
        self.mpiroot = 0

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


class MaskCollection(BaseMask, dict):
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
    def __init__(self, zrange=None, mpicomm=None):
        """
        Initialize :class:`BaseRadialMask`.

        Parameters
        ----------
        zrange : tuple, list, default=None
            Redshift range.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.zrange = tuple(zrange) if zrange is not None else (0., np.inf)
        self.mpicomm = mpicomm
        self.mpiroot = 0

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
    def __init__(self, z, nbar=None, zrange=None, filename=None, norm=None, mpicomm=None):
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
        """
        self.mpicomm = mpicomm
        self.mpiroot = 0
        if filename is not None:
            z, nbar = None, None
            if self.is_mpi_root():
                self.log_info('Loading density file: {}.'.format(filename))
                z, nbar = np.loadtxt(filename, unpack=True)
            z = mpi.broadcast_array(z, mpicomm=self.mpicomm, root=self.mpiroot)
            nbar = mpi.broadcast_array(nbar, mpicomm=self.mpicomm, root=self.mpiroot)
        self.z = mpi.broadcast_array(np.asarray(z), mpicomm=self.mpicomm, root=self.mpiroot)
        self.nbar = mpi.broadcast_array(np.asarray(nbar), mpicomm=self.mpicomm, root=self.mpiroot)
        if not np.all(self.nbar >= 0.):
            raise ValueError('Provided nbar should be all positive.')
        zmin, zmax = self.z[0], self.z[-1]
        if zrange is None: zrange = zmin, zmax
        super(TabulatedRadialMask, self).__init__(zrange=zrange, mpicomm=mpicomm)
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

    def integral(self, z=None, weights=None, normalize_weights=True, mpiroot=None):
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

        mpiroot : int, default=None
            If ``None``, input z and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input z and weights are gathered.

        Returns
        -------
        integral : float
            Integral of :meth:`prob`.
        """
        if mpiroot is not None:
            if not self.mpicomm.bcast(z is None, root=mpiroot):
                z = mpi.scatter_array(z, mpicomm=self.mpicomm, root=mpiroot)
            if not self.mpicomm.bcast(weights is None, root=mpiroot):
                weights = mpi.scatter_array(weights, mpicomm=self.mpicomm, root=mpiroot)
        if z is None and weights is None:
            return self.interp.integrate(self.zrange[0], self.zrange[1])
        if weights is not None and z is None:
            raise ValueError('Provide z when giving weights')
        if weights is None:
            weights = 1.
            if normalize_weights:
                weights = weights/self.mpicomm.allreduce(z.size)
        elif normalize_weights:
            weights = weights/mpi.sum_array(weights, mpicomm=mpicomm)
        return mpi.sum_array(self.prob(z)*weights, mpicomm=self.mpicomm)

    def normalize(self, target, z=None, weights=None, mpiroot=None):
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

        mpiroot : int, default=None
            If ``None``, input z and weights are assumed to be scattered across all ranks.
            Else the MPI rank where input z and weights are gathered.
        """
        if not ((target >= 0) & (target <= 1.)):
            raise ValueError('Input norm should be in (0, 1)')

        if mpiroot is not None:
            if not self.mpicomm.bcast(z is None, root=mpiroot):
                z = mpi.scatter_array(z, mpicomm=self.mpicomm, root=mpiroot)
            if not self.mpicomm.bcast(weights is None, root=mpiroot):
                weights = mpi.scatter_array(weights, mpicomm=self.mpicomm, root=mpiroot)

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
        error = self.integral(z, weights, normalize_weights=False, mpiroot=None) - target
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
            zedges = np.concatenate([[self.z[0]],zedges,[self.z[-1]]], axis=0)
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
        """
        if rarange is not None:
            rarange = utils.wrap_angle(rarange, degree=True)
            # if e.g. rarange = (300, 40), we want to select RA > 300 or RA < 40
            self.rarange = tuple(rarange)
        else:
            self.rarange = (0., 360.)
        if decrange is not None:
            self.decrange = tuple(decrange)
        else:
            self.decrange = (-90., 90.)
        self.mpicomm = mpicomm
        self.mpiroot = 0

    def _mask_ranges(self, ra, dec):
        # input ra assumed in [0., 360.] and dec in [-90., 90.]
        if self.rarange[0] <= self.rarange[-1]:
            mask = (ra >= self.rarange[0]) & (ra <= self.rarange[-1])
        else:
            mask = (ra >= self.rarange[0]) | (ra <= self.rarange[-1])
        mask &= (dec >= self.decrange[0]) & (dec <= self.decrange[-1])
        return mask

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
            rarange = list(self.rarange)
            # if e.g. rarange = (300, 40), we want to generate RA > -60 and RA < 40
            if rarange[0] > rarange[1]: rarange[0] -= 360
            ra = rng.uniform(low=rarange[0], high=rarange[1]) % 360.
            urange = np.sin(np.asarray(self.decrange)*np.pi/180.)
            dec = np.arcsin(rng.uniform(low=urange[0], high=urange[1]))/(np.pi/180.)
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
        mask = self._mask_ranges(ra, dec)
        prob[~mask] = 0
        return prob


try: import pymangle
except ImportError: pymangle = None


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
        if pymangle is None:
            raise ImportError('Install pymangle')
        super(MangleAngularMask,self).__init__(**kwargs)
        if filename is not None:
            if self.is_mpi_root():
                self.log_info('Loading Mangle geometry file: {}.'.format(filename))
            self.nbar = pymangle.Mangle(filename)
        else:
            self.nbar = mpi.broadcast_array(np.asarray(nbar), mpicomm=self.mpicomm, root=self.mpiroot)

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
        mask = self._mask_ranges(ra, dec)
        prob[~mask] = 0.
        return prob


try: import healpy
except ImportError: healpy = None


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
        if healpy is None:
            raise ImportError('Install healpy')
        super(HealpixAngularMask,self).__init__(**kwargs)
        if filename is not None:
            if self.is_mpi_root():
                self.log_info('Loading healpy geometry file: {}.'.format(filename))
            nbar = healpy.fitsfunc.read_map(filename, nest=nest, **kwargs)
        self.nbar = mpi.broadcast_array(np.asarray(nbar), mpicomm=self.mpicomm, root=0)
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
        mask = self._mask_ranges(ra, dec)
        prob[~mask] = 0.
        return prob

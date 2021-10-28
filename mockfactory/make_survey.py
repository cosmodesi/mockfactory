import os
import logging
import functools
import itertools
from collections import UserDict

import numpy as np
from scipy import interpolate, optimize

from . import mpi
from .mpi import SetMPIComm


def distance(position):
    """Return cartesian distance, taking coordinates along ``position`` last axis."""
    return np.sqrt((position**2).sum(axis=-1))


def rotation_matrix_from_vectors(a, b):
    """
    Return rotation matrix transforming 3D vector ``a`` to 2D vector ``b``.

    >>> a = np.array([0.,1.,2.])
    >>> b = np.array([0.,2.,1.])
    >>> rot = rotation_matrix_from_vectors(a, b):
    >>> assert np.allclose(rot.dot(a),b)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    a /= distance(a)
    b /= distance(b)
    v = np.cross(a,b)
    c = np.dot(a,b)
    s = distance(v)
    I = np.identity(3,dtype='f8')
    k = np.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]])
    if s == 0.: return I
    return I + k + np.matmul(k,k) * ((1.-c)/(s**2))


def cartesian_to_sky(position, wrap=True, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (N, 3)
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
    dist = distance(position)
    ra = np.arctan2(position[:,1], position[:,0])
    if wrap: ra %= 2.*np.pi
    dec = np.arcsin(position[:,2]/dist)
    conversion = np.pi/180. if degree else 1.
    return dist, ra/conversion, dec/conversion


def sky_to_cartesian(dist, ra, dec, degree=True, dtype=None):
    """
    Transform distance, RA, Dec into cartesian coordinates.

    Parameters
    ----------
    dist : array of shape (N,)
        Distance.

    ra : array of shape (N,)
        Right Ascension.

    dec : array of shape (N,)
        Declination.

    degree : default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    dtype : numpy.dtype, default=None
        :class:`numpy.dtype` for returned array.

    Returns
    -------
    position : array of shape (N,)
        Position in cartesian coordinates.
    """
    conversion = 1.
    if degree: conversion = np.pi/180.
    position = [None]*3
    cos_dec = np.cos(dec*conversion)
    position[0] = cos_dec*np.cos(ra*conversion)
    position[1] = cos_dec*np.sin(ra*conversion)
    position[2] = np.sin(dec*conversion)
    return (dist*np.asarray(position,dtype=dtype)).T


def cutsky_to_box(drange, rarange, decrange):
    """Translation along x and rotation about z and y."""
    if rarange[0] > rarange[1]: rarange[0] -= 360.
    deltara = abs(rarange[1]-rarange[0])/2.*np.pi/180.
    deltadec = abs(decrange[1]-decrange[0])/2.*np.pi/180.
    boxsize = np.empty((3),dtype='f8')
    boxsize[1] = 2.*drange[1]*np.sin(deltara)
    boxsize[2] = 2.*drange[1]*np.sin(deltadec)
    boxsize[0] = drange[1] - drange[0]*min(np.cos(deltara),np.cos(deltadec))
    operations = [{'method':'translate_along_axis','kwargs':{'axis':'x','translate':drange[1]-boxsize[0]/2.}}]
    operations += [{'method':'rotate_about_origin_axis','kwargs':{'axis':'y','angle':(decrange[0]+decrange[1])/2.}}]
    operations += [{'method':'rotate_about_origin_axis','kwargs':{'axis':'z','angle':(rarange[0]+rarange[1])/2.}}]
    return boxsize, operations


def box_to_cutsky(boxsize, dmax):
    deltara = np.arcsin(boxsize[1]/2./dmax)
    deltadec = np.arcsin(boxsize[2]/2./dmax)
    dmin = (dmax-boxsize[0])/min(np.cos(deltara),np.cos(deltadec))
    return deltara*2.*180./np.pi,deltadec*2.*180./np.pi,dmin


def radecbox_area(ramin, ramax, decmin, decmax):
    """
    Return area of ra, dec box.

    Parameters
    ----------
    ramin : float, array-like
        Minimum right ascension (degree).

    ramax : float, array-like
        Maximum right ascension (degree).

    decmin : float, array-like
        Minimum declination (degree).

    decmax : float, array-like
        Maximum declination (degree).

    Returns
    -------
    area : float, ndarray.
        Area (degree^2).
    """
    decfrac = np.diff(np.rad2deg(np.sin(np.deg2rad([decmin,decmax]))),axis=0)
    rafrac = np.diff([ramin,ramax],axis=0)
    area = decfrac*rafrac
    if np.isscalar(ramin):
        return area[0]
    return area


def vector_projection(vector, direction):
    r"""
    Vector components of given vectors in a given direction.

    .. math::
       \mathbf{v}_\mathbf{d} &= (\mathbf{v} \cdot \hat{\mathbf{d}}) \hat{\mathbf{d}} \\
       \hat{\mathbf{d}} &= \frac{\mathbf{d}}{\|\mathbf{d}\|}

    Adapted from https://github.com/bccp/nbodykit/blob/master/nbodykit/transform.py

    Parameters
    ----------
    vector : array
        Array of vectors to be projected (along last dimension).

    direction : array
        Projection direction, 1D or 2D (if different direction for each input ``vector``) array.
        It will be normalized.

    Returns
    -------
    projection : array
        Vector components of the given vectors in the given direction.
        Same shape as input ``vector``.
    """
    direction = np.asarray(direction, dtype='f8')
    direction = direction / (direction ** 2).sum(axis=-1)[:, None] ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction

    return projection


class DistanceToRedshift(object):

    def __init__(self, distance, zmax=100., nz=2048):
        self.distance = distance
        self.zmax = zmax
        self.nz = nz
        self.compute()

    def compute(self):
        zgrid = np.logspace(-8,np.log10(self.zmax),self.nz)
        self.zgrid = np.concatenate([[0.], zgrid])
        self.rgrid = self.distance(self.zgrid)
        self.interp = interpolate.Akima1DInterpolator(self.rgrid,self.zgrid,axis=0)

    def __call__(self, distance):
        return self.interp(distance)


class RedshiftDensityInterpolator(object):

    logger = logging.getLogger('RedshiftDensityInterpolator')

    @SetMPIComm
    def __init__(self, redshifts, weights=None, bins=None, fsky=1., radial_distance=None, interp_order=1, mpicomm=None):
        """
        Inspired from: https://github.com/bccp/nbodykit/blob/master/nbodykit/algorithms/zhist.py
        """
        self.mpicomm = mpicomm
        def zrange(redshifts):
            return mpi.min_array(redshifts,mpicomm=self.mpicomm),mpi.max_array(redshifts,mpicomm=self.mpicomm)

        if bins is None or bins == 'scott':
            # scott's rule
            var = mpi.var_array(redshifts,aweights=weights,ddof=1,mpicomm=self.mpicomm)
            gsize = mpi.size_array(redshifts)
            sigma = np.sqrt(var)
            dz = sigma * (24. * np.sqrt(np.pi) / gsize) ** (1. / 3)
            zrange = zrange(redshifts)
            nbins = np.ceil((maxval - minval) * 1. / dx)
            nbins = max(1, nbins)
            edges = minval + dx * np.arange(nbins + 1)

        if np.ndim(bins) == 0:
            bins = np.linspace(*zrange(redshifts),num=bins+1,endpoint=True)

        counts = np.histogram(redshifts,weights=weights,bins=bins)
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
        return self.spline(z)



def _multiple_columns(column):
    return isinstance(column,list)


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



class Catalog(object):

    logger = logging.getLogger('Catalog')

    @SetMPIComm
    def __init__(self, data=None, columns=None, attrs=None, mpicomm=None, mpiroot=0):
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
        return self.mpicomm.rank == self.mpiroot

    @classmethod
    def from_nbodykit(cls, catalog, columns=None):
        if columns is None: columns = catalog.columns
        data = {col: catalog[col].compute() for col in columns}
        return cls(data,mpicomm=catalog.comm,mpiroot=0,attrs=catalog.attrs)

    def to_nbodykit(self, columns=None):
        if columns is None: columns = self.columns()
        source = {col:self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        attrs = {key:value for key,value in self.attrs.items() if key != 'fitshdr'}
        return ArrayCatalog(source,**attrs)

    def __len__(self):
        keys = list(self.data.keys())
        if not keys or self[keys[0]] is None:
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        return len(self)

    @property
    def gsize(self):
        return self.mpicomm.allreduce(len(self))

    def columns(self, include=None, exclude=None):
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
        return column in self.data

    def indices(self):
        sizes = self.mpicomm.allgather(len(self))
        sizes = [0] + np.cumsum(sizes[:1]).tolist()
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, dtype='f8'):
        return np.zeros(len(self),dtype=dtype)

    def ones(self, dtype='f8'):
        return np.ones(len(self),dtype=dtype)

    def full(self, fill_value, dtype='f8'):
        return np.full(len(self),fill_value,dtype=dtype)

    def falses(self):
        return self.zeros(dtype=np.bool_)

    def trues(self):
        return self.ones(dtype=np.bool_)

    def nans(self):
        return self.ones()*np.nan

    def get(self, column, *args, **kwargs):
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
        self.data[column] = item

    def gget(self, column, mpiroot=None):
        if mpiroot is None: mpiroot = Ellipsis
        return mpi.gather_array(self[column],mpiroot=Ellipsis,mpicomm=self.mpicomm)

    def gslice(self, *args):
        sl = slice(*args)
        new = self.copy()
        for col in self.columns():
            self_value = self.gget(col,mpiroot=self.mpiroot)
            new[col] = mpi.scatter_array(self_value if self.is_mpi_root() else None,mpiroot=self.mpiroot,mpicomm=self.mpicomm)
        return new

    def to_array(self, columns=None, struct=True):
        if columns is None:
            columns = self.columns()
        if struct:
            toret = np.empty(self.size,dtype=[(col,self[col].dtype,self[col].shape[1:]) for col in columns])
            for col in columns: toret[col] = self[col]
            return toret
        return np.array([self[col] for col in columns])

    @classmethod
    @SetMPIComm
    def from_array(cls, array, columns=None, mpicomm=None, mpiroot=0, **kwargs):
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

    def __copy__(self):
        """Return shallow copy of ``self``."""
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        new.data = {col:self[col] for col in self}
        return new

    def copy(self, columns=None):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        if columns is None: columns = self.columns()
        new.data = {col:self[col] for col in columns}
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data,'attrs':self.attrs}

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.data = state['data'].copy()
        self.attrs = state['attrs']

    def __getitem__(self, name):
        if isinstance(name,str):
            return self.get(name)
        new = self.copy()
        new.attrs = self.attrs.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __setitem__(self, name, item):
        if isinstance(name,str):
            return self.set(name,item)
        for col in self.data:
            self[col][name] = item

    def __delitem__(self, name):
        del self.data[name]

    def __repr__(self):
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__,self.gsize,self.columns())

    def extend(self, other):
        self_columns = self.columns()
        other_columns = other.columns()
        assert self.mpicomm is other.mpicomm

        if self_columns and other_columns and set(other_columns) != set(self_columns):
            raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns,self_columns))

        for col in other_columns:
            if col not in self_columns:
                self_value = None
            else:
                self_value = self.gget(col,mpiroot=self.mpiroot)
            other_value = other.gget(col,mpiroot=self.mpiroot)
            self[col] = None
            if self.is_mpi_root():
                if self_value is not None:
                    self[col] = np.concatenate([self_value,other_value],axis=0)
                else:
                    self[col] = other_value.copy()
            self[col] = mpi.scatter_array(self[col] if self.is_mpi_root() else None,mpiroot=self.mpiroot,mpicomm=self.mpicomm)
        return self

    def __eq__(self, other):
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
    @SetMPIComm
    def load_fits(cls, filename,  columns=None, ext=None, mpiroot=0, mpicomm=None):
        """Load class from disk."""
        if mpicomm.rank == mpiroot:
            cls.logger.info('Loading {}.'.format(filename))
        import fitsio
        # Stolen from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/fits.py
        msg = 'Input FITS file {}'.format(filename)
        with fitsio.FITS(filename) as ff:
            if ext is None:
                for i, hdu in enumerate(ff):
                    if hdu.has_data():
                        ext = i
                        break
                if ext is None:
                    raise IOError('{} has no binary table to read'.format(msg))
            else:
                if isinstance(ext,str):
                    if ext not in ff:
                        raise IOError('{} does not contain extension with name {}'.format(msg,ext))
                elif ext >= len(ff):
                    raise IOError('{} extension {} is not valid'.format(msg,ext))
            ff = ff[ext]
            # make sure we crash if data is wrong or missing
            if not ff.has_data() or ff.get_exttype() == 'IMAGE_HDU':
                raise ValueError('{} extension {} is not a readable binary table'.format(msg,ext))
            size = ff.get_nrows()
            start = mpicomm.rank * size // mpicomm.size
            stop = (mpicomm.rank + 1) * size // mpicomm.size
            new = ff.read(ext=ext,columns=columns,rows=range(start,stop))
            header = ff.read_header()
            header.clean()
            attrs = dict(header)
            attrs['fitshdr'] = header
            new = cls.from_array(new,mpiroot=mpiroot,mpicomm=mpicomm,attrs=attrs)
        return new

    def save_fits(self, filename):
        """Save as fits file. Should be possible to change fitsio to write by chunks?."""
        import fitsio
        if self.is_mpi_root():
            self.logger.info('Saving to {}.'.format(filename))
        array = self.to_array(struct=True)
        array = mpi.gather_array(array,mpiroot=self.mpiroot,mpicomm=self.mpicomm)
        fitsio.write(filename,array,header=self.attrs.get('fitshdr',None),clobber=True)

    @vectorize_columns
    def sum(self, column, axis=0):
        return mpi.sum_array(self[column],axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def average(self, column, weights=None, axis=0):
        return mpi.average_array(self[column],weights=weights,axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def mean(self, column, axis=0):
        return self.average(column,axis=axis)

    @vectorize_columns
    def minimum(self, column, axis=0):
        return mpi.min_array(self[column],axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def maximum(self, column, axis=0):
        return mpi.max_array(self[column],axis=axis,mpicomm=self.mpicomm)

    def distance(self, column='Position'):
        return distance(self[column])

    def box(self, column='Position'):
        return (self.minimum(position),self.maximum(position))

    def boxsize(self, position='Position'):
        lbox = np.diff(self.box(position=position,axis=axis),axis=0)[0]
        return np.sqrt((lbox**2).sum(axis=0))


class SurveyCatalog(Catalog):

    logger = logging.getLogger('SurveyCatalog')

    @SetMPIComm
    def __init__(self, data=None, columns=None, BoxSize=None, BoxCenter=None, attrs=None, mpicomm=None, mpiroot=0):
        super(SurveyCatalog,self).__init__(data=data,columns=columns,attrs=attrs,mpicomm=mpicomm,mpiroot=mpiroot)
        if BoxSize is None:
            BoxSize = self.attrs['BoxSize']
        if BoxCenter is None:
            BoxCenter = self.attrs.get('BoxCenter',0.)
        self.BoxSize = BoxSize
        self._boxcenter = np.empty(3,dtype='f8')
        self._boxcenter[:] = BoxCenter
        self._rotation = np.eye(3,dtype='f8')
        self._translation =  self._boxcenter.copy()
        self._compute_position = True
        self._compute_velocity = True

    @property
    def BoxSize(self):
        return self.attrs['BoxSize']

    @BoxSize.setter
    def BoxSize(self, BoxSize):
        self.attrs['BoxSize'] = np.empty(3,dtype='f8')
        self.attrs['BoxSize'][:] = BoxSize

    @property
    def Position(self):
        if self._compute_position:
            self._position = np.tensordot(self['Position']-self._boxcenter,self._rotation,axes=((1,),(1,))) + self._translation
        self._compute_position = False
        return self._position

    @property
    def Velocity(self):
        if self._compute_velocity:
            self._velocity = np.tensordot(self['Velocity'],self._rotation,axes=((1,),(1,)))
        self._compute_velocity = False
        return self._velocity

    def rotate_about_origin_axis(self, axis=0, angle=0., degree=True):
        if degree: angle *= np.pi/180.
        if not isinstance(axis,int): axis = 'xyz'.index(axis)
        c,s = np.cos(angle),np.sin(angle)
        if axis == 0: matrix = [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
        if axis == 1: matrix = [[c,0.,s],[0,1.,0.],[-s,0.,c]]
        if axis == 2: matrix = [[c,-s,0],[s,c,0],[0.,0,1.]]
        matrix = np.asarray(matrix)
        self._rotation = matrix.dot(self._rotation)
        self._translation = matrix.dot(self._translation)
        self._compute_position = True
        self._compute_velocity = True

    def rotate_about_center_axis(self, axis=0, angle=0., degree=True):
        if degree: angle *= np.pi/180.
        if not isinstance(axis,int): axis = 'xyz'.index(axis)
        c,s = np.cos(angle),np.sin(angle)
        if axis == 0: matrix = [[1.,0.,0.],[0.,c,-s],[0.,s,c]]
        if axis == 1: matrix = [[c,0.,s],[0,1.,0.],[-s,0.,c]]
        if axis == 2: matrix = [[c,-s,0],[s,c,0],[0.,0,1.]]
        matrix = np.asarray(matrix)
        self._rotation = matrix.dot(self._rotation)
        self._compute_position = True
        self._compute_velocity = True

    def rotate_about_origin(self, angles=(), degree=True):
        assert len(angles) <= 3
        for axis,angle in enumerate(angles):
            self.rotate_about_origin_axis(axis,angle,degree=degree)

    def rotate_about_center(self, angles=(), degree=True):
        assert len(angles) <= 3
        for axis,angle in enumerate(angles):
            self.rotate_about_center_axis(axis,angle,degree=degree)

    def translate(self, translate=0.):
        shift = np.empty(3,dtype='f8')
        shift[:] = translate
        self._translation += shift
        self._compute_position = True

    def translate_along_axis(self, axis=0, translate=0.):
        if isinstance(axis,str): axis = 'xyz'.index(axis)
        self._translation[axis] += translate
        self._compute_position = True

    def reset_rotate_about_center(self):
        self._rotation = np.eye(self._rotation.shape[0],dtype=self._rotation.dtype)
        self._compute_position = True
        self._compute_velocity = True

    def reset_rotate_about_origin(self):
        self._translation = self._rotation.T.dot(self._translation)
        self._rotation = np.eye(self._rotation.shape[0],dtype=self._rotation.dtype)
        self._compute_position = True
        self._compute_velocity = True

    def reset_translate(self):
        self._translation[:] = self._boxcenter[:]
        self._compute_position = True

    def recenter(self):
        self._translation[:] = 0.
        self._compute_position = True

    def recenter_position(self, position):
        return np.tensordot(position-self._translation,self._rotation.T,axes=((1,),(1,)))

    def reset_position(self, position):
        return self.recenter_position(position) + self._boxcenter

    def reset_velocity(self, velocity):
        return np.tensordot(self['Velocity'],self._rotation.T,axes=((1,),(1,)))

    def distance(self):
        return distance(self.Position)

    def flush(self):
        self['Position'] = self.Position
        if 'Velocity' in self: self['Velocity'] = self.Velocity
        self._rotation = np.eye(self._rotation.shape[0],dtype=self._rotation.dtype)
        self._boxcenter[:] = self._translation[:]

    @property
    def glos(self):
        return self._translation/np.sqrt((self._translation**2).sum(axis=-1))

    def cartesian_to_sky(self, wrap=True, degree=True):
        return cartesian_to_sky(self.Position,wrap=wrap,degree=degree)

    def RSDPosition(self, los='local'):

        if np.ndim(los) > 0:
            unit_vector = np.array(los,dtype='f8')
            unit_vector /= distance(unit_vector)
        elif los == 'local':
            unit_vector = self.Position/distance(self.Position)[:,None]
        elif los == 'global':
            unit_vector = self.glos
        else:
            axis = los
            if isinstance(los,str): axis = 'xyz'.index(axis)
            unit_vector = np.zeros(3,dtype='f8')
            unit_vector[axis] = 1.
            unit_vector = np.tensordot(unit_vector,self._rotation,axes=((0,),(1,)))

        return self.Position + vector_projection(self['Velocity'],unit_vector)

    def remap(self, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        from .remap import Cuboid
        cuboid = Cuboid(u1=u1,u2=u2,u3=u3)
        return cuboid.transform(self.position,BoxSize=self.BoxSize)

    def subvolume(self, ranges=([0,1],[0,1],[0,1])):
        if np.ndim(ranges[0]) == 0: ranges = [ranges]*3
        position = self.Position
        mask = self.trues()
        for i,r in enumerate(ranges): mask &= (position[:,i] >= r[0]) & (position[:,i] <= r[1])
        new = self[mask]
        new.BoxSize = np.diff(ranges,axis=-1)[:,0]
        new._boxcenter = np.array([r[0] for r in ranges],dtype='f8') + new.BoxSize/2. + self._boxcenter - self._translation
        return new

    def apply_operation(self,*operations):
        for operation in operations: getattr(self,operation['method'])(**operation['kwargs'])

    def replicate(self, factor=1.1, replicate=()):
        factors = np.empty(3,dtype='f8')
        factors[:] = factor
        new = self.copy()
        new.BoxSize *= factors
        position = 'Position'
        if position not in replicate: replicate.append(position)
        shifts = [np.arange(-np.ceil(factor)+1,np.ceil(factor)) for factor in factors]
        data = {key:[] for key in new}
        for shift in itertools.product(shifts):
            tmp = {col:self[col] + self.BoxSize*shift for col in replicate}
            mask = (tmp[position] >= -new.BoxSize/2. + self._boxcenter) & (tmp[position] <= new.BoxSize/2. + self._boxcenter)
            mask = np.all(mask,axis=-1)
            for col in new:
                if col in replicate: data[col].append(tmp[col][mask])
                else: data[col].append(self[col][mask])
        for col in new:
            new[col] = np.concatenate(data[col],axis=0)
        new._compute_position = True
        new._compute_velocity = True
        return new


class RandomCatalog(SurveyCatalog):

    logger = logging.getLogger('RandomCatalog')

    @SetMPIComm
    def __init__(self, BoxSize=None, BoxCenter=None, size=None, nbar=None, seed=None, mpicomm=None, mpiroot=0, attrs=None):
        super(RandomCatalog,self).__init__(data={},BoxSize=BoxSize,BoxCenter=BoxCenter,mpicomm=mpicomm,mpiroot=mpiroot,attrs=attrs)
        self.attrs['seed'] = seed

        if size is None:
            size = np.random.RandomState(seed=seed).poisson(nbar*np.prod(self.BoxSize))
        size = mpi.local_size(size,mpicomm=mpicomm)
        rng = mpi.MPIRandomState(size,seed=seed)

        self.data['Position'] = np.array([rng.uniform(-self.BoxSize[i]/2.+self._boxcenter[i],self.BoxSize[i]/2.+self._boxcenter[i]) for i in range(3)]).T


class RandomSkyCatalog(Catalog):

    logger = logging.getLogger('RandomSkyCatalog')

    @SetMPIComm
    def __init__(self, rarange=(0.,360.), decrange=(-90.,90.), nbar=None, seed=None, wrap=True, mpicomm=None, mpiroot=0, attrs=None):

        super(RandomSkyCatalog,self).__init__(data={},mpicomm=mpicomm,mpiroot=mpiroot,attrs=attrs)
        area = radecbox_area(*rarange,*decrange)
        if size is None:
            size = np.random.RandomState(seed=seed).poisson(nbar*area)
        self.attrs['seed'] = seed
        self.attrs['area'] = area

        size = mpi.local_size(size,mpicomm=mpicomm)
        rng = mpi.MPIRandomState(size,seed=seed,mpicomm=mpicomm)
        self.data['RA'] = rng.uniform(low=rarange[0],high=rarange[1],size=size)
        urange = np.sin(np.asarray(decrange)*np.pi/180.)
        self.data['DEC'] = np.arcsin(rng.uniform(low=urange[0],high=urange[1],size=size))/(np.pi/180.)
        if wrap: self.data['RA'] %= 360.


class BaseRedshiftDensityMask(object):

    logger = logging.getLogger('BaseRedshiftDensityMask')

    @SetMPIComm
    def __init__(self, seed=None, mpicomm=None, mpiroot=0):
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot
        self.set_seed(seed)

    def is_mpi_root(self):
        return self.mpicomm.rank == self.mpiroot

    def set_seed(self, seed=None):
        self.seed = seed
        if seed is None:
            self.seed = self.mpicomm.bcast(np.random.randint(0,high=0xffffffff) if self.is_mpi_root() else None,root=self.mpiroot)

    def sample(self, redshift_to_distance, size, factor=3, exact=True, distance_to_redshift=None, seed=None):
        drange = redshift_to_distance(self.zrange)
        rng = mpi.MPIRandomState(size,seed=self.seed)
        gsize = self.mpicomm.allreduce(size)
        distance = rng.uniform(drange[0],drange[1])
        if distance_to_redshift is not None:
            redshift = distance_to_redshift(distance)
        else:
            redshift = DistanceToRedshift(distance=redshift_to_distance,zmax=zrange[1]+1,nz=4096)(distance)
        prob = (distance/drange[1])**2
        assert (prob <= 1.).all()
        mask_redshift = (prob >= rng.uniform(0.,1.)) & redshift_density(redshift)
        nmask_redshift = mpi.sum_array(mask_redshift,mpicomm=self.mpicomm)
        if size <= nmask_redshift:
            std = 1./np.sqrt(nmask_redshift)
            updated_factor = factor*gsize*1./nmask_redshift*(1. + 3.*std)
            self.logger.info('Restarting with higher safe factor {:.4f}'.format(updated_factor))
            return self.sample(redshift_to_distance=redshift_to_distance,size=size,factor=updated_factor,exact=exact,distance_to_redshift=distance_to_redshift,seed=seed)
        redshift = redshift[mask_redshift]
        if exact: redshift[:size]
        return redshift

    def __call__(self, z):
        tmp = self.prob(z)
        rng = mpi.MPIRandomState(len(tmp),seed=self.seed)
        return tmp >= rng.uniform(low=0.,high=1.)


class UniformDensityMask(BaseRedshiftDensityMask):

    logger = logging.getLogger('UniformDensityMask')

    def __init__(self, nbar=1., **kwargs):
        self.nbar = np.array(nbar)
        super(UniformDensityMask,self).__init__(**kwargs)

    def prob(self, z):
        return np.clip(self.nbar,0.,1.)*np.ones(z.shape[-1],dtype='f8')


class RedshiftDensityMask(BaseRedshiftDensityMask):

    logger = logging.getLogger('RedshiftDensityMask')

    def __init__(self, z=None, nbar=None, zrange=None, filename=None, norm=None, **kwargs):
        super(RedshiftDensityMask,self).__init__(**kwargs)
        if filename is not None:
            self.logger.info('Loading density file: {}.'.format(filename))
            self.z, self.nbar = np.loadtxt(filename,unpack=True,**kwargs)
        else:
            self.z, self.nbar = z, nbar
        assert (self.nbar >= 0.).all()
        self.zrange = zrange
        zmin, zmax = self.z.min(),self.z.max()
        if self.zrange is None: self.zrange = zmin, zmax
        if not ((zmin <= self.zrange[0]) & (zmax >= self.zrange[1])):
            raise ValueError('Redshift range is {:.2f} - {:.2f} but the limiting range is {:.2f} - {:.2f}.'.format(zmin,zmax,self.zrange[0],self.zrange[1]))
        self.prepare(norm=norm)

    def prepare(self, norm=None):
        if norm is None: norm = 1./self.nbar[self.zmask].max(axis=0)
        self.norm = norm
        self.set_interp()
        return self.norm

    @property
    def zmask(self):
        return (self.z >= self.zrange[0]) & (self.z <= self.zrange[-1])

    def set_interp(self):
        prob = np.clip(self.norm*self.nbar,0.,1.)
        self.interp = interpolate.Akima1DInterpolator(self.z,prob,axis=0)

    def prob(self, z):
        toret = self.interp(z)
        mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
        toret[~mask] = 0.
        return toret

    def flatten(self, norm=None):
        if norm is None: norm = self.nbar[self.zmask].sum()
        self.nbar = np.ones_like(self.nbar)
        self.prepare(norm=self.zmask.sum()/norm)

    def integral(self, z=None, w=None, npoints=1000, normalize=True):
        if z is None and w is None:
            return self.interp.integrate(self.zrange[0],self.zrange[1])
        z,w = self._get_zw_(z=z,w=w,npoints=npoints,normalize=normalize)
        return self._integral_(z,w)

    def _get_zw_(self, z=None, w=None, npoints=1000, normalize=True):
        if z is None:
            if w is not None and not callable(w):
                raise ValueError('Provide z when giving w')
            z = mpi.linspace_array(*self.zrange,num=npoints)
        if w is None:
            w = 1.
            if normalize:
                nz = self.mpicomm.allreduce(len(z))
                w = w/nz
            return z,w
        if callable(w):
            w = w(z)
        if normalize:
            w = w/mpi.sum_array(w,mpicomm=mpicomm)
        return z,w

    def _integral_(self, z, w):
        return mpi.sum_array(self.prob(z)*w)

    def normalize(self, factor=1., **kwargs):

        assert factor <= 1.
        z,w = self._get_zw_(normalize=True, **kwargs)

        def normalization(norm):
            self.norm = norm
            self.set_interp()
            return self._integral_(z,w)/factor - 1.

        min_ = self.nbar[self.nbar>0.].min()
        norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.

        self.prepare(norm=norm)
        self.logger.info('Norm is: {:.12g}.'.format(self.norm))
        self.logger.info('Expected error: {:.5g}.'.format(self._integral_(z,w)-factor))

        return norm

    def convert_to_cosmo(self, distance_self, distance_target, zedges=None):
        if zedges is None:
            zedges = (self.z[:-1] + self.z[1:])/2.
            zedges = np.concatenate([self.z[0]],zedges,[self.z[-1]])
        dedges = distance_self(zedges)
        volume_self = dedges[1:]**3-dedges[:-1]**3
        dedges = distance_target(zedges)
        volume_target = dedges[1:]**3-dedges[:-1]**3
        self.nbar = self.nbar*volume_self/volume_target
        self.prepare()


class RedshiftDensityMask2D(RedshiftDensityMask):

    logger = logging.getLogger('RedshiftDensityMask2D')

    def __init__(self, z=None, other=None, **kwargs):
        self.other = other
        super(RedshiftDensityMask2D,self).__init__(z=z,**kwargs)

    def set_interp(self):
        prob = np.clip(self.norm*self.nbar,0.,1.)
        ky = min(len(other)-1,3)
        self.interp = interpolate.RectBivariateSpline(self.z,self.other,prob,kx=3,ky=ky,s=0)

    def prob(self, z, grid=False):
        z, other = z
        toret = self.interp(z,other,grid=grid)
        mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
        toret[~mask] = 0.
        return toret

    def _get_zw_(self, zo=None, w=None, npoints=1000, orange=[0.,1.], onpoints=10, grid=False, normalize=True):
        if zo is None:
            zo = [mpi.linspace_array(*self.zrange,num=npoints),np.linspace(range_other[0],range_other[1],onpoints)]
            grid = True
        if w is None:
            w = 1.
            if normalize:
                if grid:
                    noz = self.mpicomm.allreduce(len(zo[0])*len(z[1]))
                    w = w/noz
                else:
                    noz = self.mpicomm.allreduce(len(z))
                    w = w/noz
            return z,w,grid
        if callable(w):
            w = w(z)
        if normalize:
            w = w/mpi.sum_array(w,mpicomm=mpicomm)
        return z,w,grid

    def _integral_(self, z, w, grid=False):
        return mpi.sum_array(self.prob(z,grid=grid)*w)

    def normalize(self, factor=1., **kwargs):

        assert factor <= 1.
        z,w,grid = self._get_zw_(normalize=True,**kwargs)

        def normalization(norm):
            self.norm = norm
            self.set_interp()
            return self._integral_(z,w,grid=grid)/factor - 1

        min_ = self.prob[self.prob>0.].min()
        norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.

        self.prepare(norm=norm)
        self.logger.info('Norm is: {:.12g}.'.format(self.norm))
        self.logger.info('Expected error: {:.5g}.'.format(self._integral_(z,w,grid=grid)-factor))

        return norm


class KernelDensityMask(RedshiftDensityMask):

    logger = logging.getLogger('KernelDensityMask')

    def __init__(self, position, weight=None, distance=None, zrange=None, norm=1., seed=None, mpicomm=None, mpiroot=0):

        from sklearn.neighbors import KernelDensity
        self.distance = distance
        self.zrange = zrange
        self.kernel = KernelDensity(**kwargs)
        self.kernel.fit(self.get_position(position),sample_weight=weight)
        self.norm = norm
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot
        self.set_seed(seed)

    def get_position(self, position):
        if self.distance is not None:
            z,ra,dec = position
            position = sky_to_cartesian(self.distance(z),ra,dec)
        return position

    def prob(self, position, clip=True):
        if self.distance is not None:
            z,ra,dec = position
        logpdf = self.kernel.score_samples(self.get_position(position))
        toret = self.norm*np.exp(logpdf)
        if clip: toret = np.clip(toret,0.,1.)
        if self.distance is not None and self.zrange is not None:
            mask = (z >= self.zrange[0]) & (z <= self.zrange[-1])
            toret[~mask] = 0.
        return toret

    def integral(self, position, normalize=True):
        toret = mpi.sum_array(self.prob(position),mpicomm=self.mpicomm)
        gsize = mpi.shape_array(position,mpicomm=self.mpicomm)[0]
        if normalize: toret /= gsize

    def normalize(self, position, factor=1.):

        assert factor <= 1.
        prob = self.prob(position, clip=False)

        def normalization(norm):
            return mpi.sum_array(np.clip(norm*prob,0.,1.),mpicomm=self.mpicomm)/(size*factor) - 1.

        min_ = mpi.min_array(prob[prob>0.],mpicomm=self.mpicomm)
        self.norm = optimize.brentq(normalization,0.,1/min_) # the lowest point of n(z) is limited by 1.

        self.logger.info('Norm is: {:.12g}.'.format(self.norm))
        self.logger.info('Expected error: {:.5g}.'.format(self.integral(position,normalize=True)-factor))

        return self.norm


class DensityMaskChunk(object):

    logger = logging.getLogger('DensityMaskChunk')

    def set_seed(self, seed=None):
        for chunkz,density in self.items():
            density.set_seed(seed=seed)

    def __call__(self, z, other):
        z = np.asarray(z)
        other = np.asarray(other)
        toret = np.ones(z.shape[-1],dtype=np.bool_)
        for ichunkz,density in self.items():
            mask = other == ichunkz
            if mask.any():
                toret[...,mask] = density(z[...,mask])
        return toret

    def __radd__(self,other):
        if other == 0: return self
        return self.__add__(other)

    def __add__(self, other):
        new = self.__class__.__new__(self.__class__)
        new.data = {}
        for ichunkz,density in self.items():
            new[ichunkz] = density
        for ichunkz,density in other.items():
            new[ichunkz] = density
        return new


class BaseAngularMask(object):

    logger = logging.getLogger('BaseAngularMask')

    @SetMPIComm
    def __init__(self, mask=None, seed=None, mpicomm=None, mpiroot=0):
        self.mask = mask
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot
        self.set_seed(seed)

    def is_mpi_root(self):
        return self.mpicomm.rank == self.mpiroot

    def set_seed(self, seed=None):
        self.seed = seed
        if seed is None:
            self.seed = self.mpicomm.bcast(np.random.randint(0,high=0xffffffff) if self.is_mpi_root() else None,root=self.mpiroot)

    def __call__(self, ra, dec):
        rng = mpi.MPIRandomState(len(ra),seed=self.seed)
        mask = self.prob(ra,dec) > rng.uniform(low=0.,high=1.)
        return mask


class MangleAngularMask(BaseAngularMask):

    logger = logging.getLogger('MangleAngularMask')

    def __init__(self, mask=None, filename=None, **kwargs):
        super(MangleAngularMask,self).__init__(mask=mask,**kwargs)
        if filename is not None:
            import pymangle
            if self.is_mpi_root():
                self.logger.info('Loading geometry file: {}.'.format(filename))
            self.mask = pymangle.Mangle(filename)

    def prob(self, ra, dec):
        ids,prob = self.mask.polyid_and_weight(ra,dec)
        mask = ids != -1
        prob[~mask] = 0.
        return prob


try:
    import healpy
    HAVE_HEALPY = True
except ImportError:
    HAVE_HEALPY = False

if HAVE_HEALPY:

    class HealpixAngularMask(BaseAngularMask):

        logger = logging.getLogger('HealpixAngularMask')

        def __init__(self, mask=None, filename=None, nest=False, seed=None, mpicomm=None, mpiroot=0, **kwargs):
            super(HealpixAngularMask,self).__init__(mask=mask,seed=seed,mpicomm=mpicomm,mpiroot=mpiroot)
            if filename:
                 self.mask = healpy.fitsfunc.read_map(filename,nest=nest,**kwargs)
            self.nside = healpy.npix2nside(self.mask.size)
            self.nest = nest

        def prob(self, ra, dec):
            theta,phi = (-dec+90.)*np.pi/180., ra*np.pi/180.
            prob = self.mask[healpy.ang2pix(self.nside,theta,phi,nest=self.nest,lonlat=False)]
            return prob

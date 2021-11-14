"""A few utilities."""

import os
import sys
import time
import logging
import traceback
import functools

import numpy as np

from .mpi import CurrentMPIComm


@CurrentMPIComm.enable
def exception_handler(exc_type, exc_value, exc_traceback, mpicomm=None):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '='*100
    #log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')
    mpicomm.Abort()


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname) # MPI...
    except OSError:
        return


def setup_logging(level=logging.INFO, stream=sys.stdout, filename=None, filemode='w', **kwargs):
    """
    Set up logging.

    Parameters
    ----------
    level : string, int, default=logging.INFO
        Logging level.

    stream : _io.TextIOWrapper, default=sys.stdout
        Where to stream.

    filename : string, default=None
        If not ``None`` stream to file name.

    filemode : string, default='w'
        Mode to open file, only used if filename is not ``None``.

    kwargs : dict
        Other arguments for :func:`logging.basicConfig`.
    """
    # Cannot provide stream and filename kwargs at the same time to logging.basicConfig, so handle different cases
    # Thanks to https://stackoverflow.com/questions/30861524/logging-basicconfig-not-creating-log-file-when-i-run-in-pycharm
    if isinstance(level,str):
        level = {'info':logging.INFO,'debug':logging.DEBUG,'warning':logging.WARNING}[level.lower()]
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)

    t0 = time.time()

    class MyFormatter(logging.Formatter):

        @CurrentMPIComm.enable
        def format(self, record, mpicomm=None):
            ranksize = '[{:{dig}d}/{:d}]'.format(mpicomm.rank,mpicomm.size,dig=len(str(mpicomm.size)))
            self._style._fmt = '[%09.2f] ' % (time.time() - t0) + ranksize + ' %(asctime)s %(name)-25s %(levelname)-8s %(message)s'
            return super(MyFormatter,self).format(record)

    fmt = MyFormatter(datefmt='%m-%d %H:%M ')
    if filename is not None:
        mkdir(os.path.dirname(filename))
        handler = logging.FileHandler(filename,mode=filemode)
    else:
        handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(fmt)
    logging.basicConfig(level=level,handlers=[handler],**kwargs)
    sys.excepthook = exception_handler


class BaseMetaClass(type):

    """Metaclass to add logging attributes to :class:`BaseClass` derived classes."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        cls.set_logger()
        return cls

    def set_logger(cls):
        """
        Add attributes for logging:

        - logger
        - methods log_debug, log_info, log_warning, log_error, log_critical
        """
        cls.logger = logging.getLogger(cls.__name__)

        def make_logger(level):

            @classmethod
            @CurrentMPIComm.enable
            def logger(cls, *args, rank=None, mpicomm=None, **kwargs):
                if rank is None or mpicomm.rank == rank:
                    getattr(cls.logger,level)(*args,**kwargs)

            return logger

        for level in ['debug','info','warning','error','critical']:
            setattr(cls, 'log_{}'.format(level), make_logger(level))


class BaseClass(object,metaclass=BaseMetaClass):
    """
    Base class that implements :meth:`copy`.
    To be used throughout this package.
    """
    def __copy__(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__.update(self.__dict__)
        return new

    def copy(self, **kwargs):
        new = self.__copy__()
        new.__dict__.update(kwargs)
        return new

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def from_state(cls, state):
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        self.log_info('Saving {}.'.format(filename))
        np.save(filename, self.__getstate__(), allow_pickle=True)

    @classmethod
    def load(cls, filename):
        cls.log_info('Loading {}.'.format(filename))
        state = np.load(filename, allow_pickle=True)[()]
        new = cls.from_state(state)
        return new


def distance(position):
    """Return cartesian distance, taking coordinates along ``position`` last axis."""
    return np.sqrt((position**2).sum(axis=-1))


def wrap_angle(angle, degree=True):
    r"""
    Wrap angle in given range.

    Parameters
    ----------
    angle : array_like
        Angle.

    degree : bool, default=True
        Whether ``angle`` is in degrees (``True``) or radians (``False``).

    Returns
    -------
    angle : array
        Wrapped angle.
    """
    conversion = np.pi/180. if degree else 1.
    angle = np.array(angle)*conversion
    angle %= 2.*np.pi
    return angle/conversion


def cartesian_to_sky(position, wrap=True, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (N, 3)
        Position in cartesian coordinates.

    wrap : bool, default=True
        Whether to wrap RA in :math:`[0, 2 \pi]` radians.

    degree : bool, default=True
        Whether RA, Dec are in degrees (``True``) or radians (``False``).

    Returns
    -------
    dist : array
        Distance.

    ra : array
        Right ascension.

    dec : array
        Declination.
    """
    dist = distance(position)
    ra = np.arctan2(position[:,1], position[:,0])
    ra = wrap_angle(ra, degree=False)
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
    position : array of shape (N, 3)
        Position in cartesian coordinates.
    """
    conversion = np.pi/180. if degree else 1.
    position = [None]*3
    cos_dec = np.cos(dec*conversion)
    position[0] = cos_dec*np.cos(ra*conversion)
    position[1] = cos_dec*np.sin(ra*conversion)
    position[2] = np.sin(dec*conversion)
    return (dist*np.asarray(position,dtype=dtype)).T


def radecbox_area(rarange, decrange):
    """
    Return area of ra, dec box.

    Parameters
    ----------
    rarange : tuple
        Range (min, max) of right ascension (degree).

    decrange : tuple
        Range (min, max) of declination (degree).

    Returns
    -------
    area : float, ndarray.
        Area (degree^2).
    """
    decfrac = np.diff(np.rad2deg(np.sin(np.deg2rad(decrange))),axis=0)
    rafrac = np.diff(rarange,axis=0)
    area = decfrac*rafrac
    if np.isscalar(rarange[0]):
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
    direction = np.atleast_2d(direction)
    direction = direction / (direction ** 2).sum(axis=-1)[:, None] ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction

    return projection

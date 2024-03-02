"""A few utilities."""

import numpy as np
from scipy import stats

from mpytools.utils import mkdir, setup_logging, BaseMetaClass, BaseClass


def is_sequence(item):
    """Whether input item is a tuple or list."""
    return isinstance(item, (list, tuple))


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
    conversion = np.pi / 180. if degree else 1.
    angle = np.array(angle) * conversion
    angle %= 2. * np.pi
    return angle / conversion


def cartesian_to_sky(position, degree=True):
    r"""
    Transform cartesian coordinates into distance, RA, Dec.

    Parameters
    ----------
    position : array of shape (N, 3)
        Position in cartesian coordinates.

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
    ra = np.arctan2(position[..., 1], position[..., 0])
    dec = np.arcsin(position[..., 2] / dist)
    conversion = np.pi / 180. if degree else 1.
    return dist, ra / conversion, dec / conversion


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
    conversion = np.pi / 180. if degree else 1.
    dist, ra, dec = np.broadcast_arrays(dist, ra, dec)
    if dtype is None:
        dtype = np.result_type(dist, ra, dec)
    position = np.empty(dist.shape + (3,), dtype=dtype)
    cos_dec = np.cos(dec * conversion)
    position[..., 0] = cos_dec * np.cos(ra * conversion)
    position[..., 1] = cos_dec * np.sin(ra * conversion)
    position[..., 2] = np.sin(dec * conversion)

    return dist[..., None] * position


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
    decfrac = np.diff(np.rad2deg(np.sin(np.deg2rad(decrange))), axis=0)
    rafrac = np.diff(rarange, axis=0)
    area = decfrac * rafrac
    if np.ndim(rarange[0]) == 0:
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
    direction = direction / (direction ** 2).sum(axis=-1)[..., None] ** 0.5
    projection = (vector * direction).sum(axis=-1)

    return projection[..., None] * direction


def _rescaling_parser(func):
    from functools import wraps

    @wraps(func)
    def wrapper(a, b, loc=0., scale=1., **kwargs):
        # Here we do the rescaling
        a, b = (a - loc) / scale, (b - loc) / scale
        return func(a, b, loc=loc, scale=scale, **kwargs)
    return wrapper


class trunccauchy_gen(stats.rv_continuous):
    """
    A truncated Cauchy continuous random variable, where the range ``[a, b]`` is user-provided, and defined w.r.t. the (scaled) distribution.

    In order to have correct cfd and able to draw sample, we just need to redine correctly the pdf. This is simple done by truncating stats.cauchy.pdf
    and then divide by the integral of the pdf in the support (simply stats.cauchy.cdf(b) - stats.cauchy.cdf(a)). Implementing only the pdf is not super
    efficient, especially to draw samples with rvs.
    That is why we also implement ppf (used to draw) and cdf which is used to compute ppf doing the inversion via interpolation.

    Note
    ----
    For proper implementation, one should use logpdf as in truncnorm to avoid division by zero when the truncation is done far from the core of the distribution.

    Example
    -------
    >>> rv = trunccauchy(a=-1, b=1, loc=0., scale=0.1)
    >>> samples = rv.rvs(size=1000)

    References
    ----------
    * https://docs.scipy.org/doc/scipy/tutorial/stats.html#making-a-continuous-distribution-i-e-subclassing-rv-continuous
    * truncated normal function already implemented (see also source code): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
    """
    def _attach_argparser_methods(self):
        # Redefine arg parsers, scaling a and b to the standard distribution
        super()._attach_argparser_methods()

        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, _rescaling_parser(getattr(self, name)))

    def _argcheck(self, a, b):
        return a < b

    def _get_support(self, a, b):
        return a, b

    def _pdf(self, x, a, b):
        # x, a, b are standardized, a < x < b
        rv = stats.cauchy(loc=0., scale=1.)
        return rv.pdf(x) / (rv.cdf(b) - rv.cdf(a))

    def _cdf(self, x, a, b):
        """Also implement cdf to compute ppf efficiently."""
        # x, a, b are standardized, a < x < b
        rv = stats.cauchy(loc=0., scale=1.)
        x, a, b = np.broadcast_arrays(x, a, b)
        return (rv.cdf(x) - rv.cdf(a)) / (rv.cdf(b) - rv.cdf(a))

    def _ppf(self, q, a, b):
        """Implement ppf to quickly draw samples with rvs."""
        rv = stats.cauchy(loc=0., scale=1.)
        q, a, b = np.broadcast_arrays(q, a, b)
        q = q * (rv.cdf(b) - rv.cdf(a)) + rv.cdf(a)
        return rv.ppf(q)


trunccauchy = trunccauchy_gen(name='cauchy')


from scipy.stats._continuous_distns import truncnorm_gen as _truncnorm_gen


class truncnorm_gen(_truncnorm_gen):
    r"""A truncated normal continuous random variable.

    Notes
    -----
    Contrary to :class:`scipy.stats.truncnorm`, input ``a`` and ``b`` are *not* defined
    w.r.t. the *standard* normal, but to the (scaled) distribution.
    (Not sure this is an excellent idea to use a different convention from scipy.)
    """
    def _attach_argparser_methods(self):
        # Redefine arg parsers, scaling a and b to the standard distribution
        super()._attach_argparser_methods()

        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, _rescaling_parser(getattr(self, name)))


truncnorm = truncnorm_gen(name='truncnorm', momtype=1)

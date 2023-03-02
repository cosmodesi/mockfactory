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
    ra = np.arctan2(position[..., 1], position[..., 0])
    ra = wrap_angle(ra, degree=False)
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
    direction = direction / (direction ** 2).sum(axis=-1)[:, None] ** 0.5
    projection = (vector * direction).sum(axis=-1)
    projection = projection[:, None] * direction

    return projection


class trunccauchy(stats.rv_continuous):
    """
    A truncated cauchy continuous random variable, where the range ``[a, b]`` is user-provided

    In order to have correct cfd and able to draw sample, we just need to redine correctly the pdf. This is simple done by truncated the stats.cauchy.pdf
    and then divided by the integral of the pdf in the restriced area (simply stats.cauchy.cdf(b) - stats.cauchy.cdf(a)). Implement only the pdf is not super
    efficient, especially to draw samples with .rvs(). That is why we also implement ppf (used to draw) and cdf which is used to compute ppf doing the inversion via interpolation.

    Remark: For proper implementation, once should use logpdf as in truncnorm to avoid division by zero when the truncation is done far from the core of the distribution.

    Warning: loc and scale are built-in keywords. One cannot use them in _pdf ! Use lo and sc instead.

    Example:
        '''
            e = trunccauchy(a=-1, b=1, shapes='lo, sc')
            e = e.freeze(lo=0, sc=0.1)  # to freeze the parameter lo and sc
            samples = e.rvs(size=1000)
        '''

    References:
        * https://docs.scipy.org/doc/scipy/tutorial/stats.html#making-a-continuous-distribution-i-e-subclassing-rv-continuous
        * truncated normal function already implemented (see also source code): https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html

    """

    def _argcheck(*args):
        """ by default _argcheck return true only for args > 0, this is not our case since we use loc for scipy.cauchy which could be negative..."""
        return True

    def _pdf(self, x, lo, sc):
        """ Without any optimzation, pdf is the only function that we need to define a prbability law. """
        return stats.cauchy.pdf(x, loc=lo, scale=sc) / (stats.cauchy.cdf(self.b, loc=lo, scale=sc) - stats.cauchy.cdf(self.a, loc=lo, scale=sc))

    def _cdf(self, x, lo, sc):
        """ Need to implement cdf and not only the pdf to compute ppf efficiently ! """
        cdf = stats.cauchy.cdf(x, loc=lo, scale=sc) - stats.cauchy.cdf(self.a, loc=lo, scale=sc)
        cdf[x < self.a] = 0
        cdf[x > self.b] = stats.cauchy.cdf(self.b, loc=lo, scale=sc) - stats.cauchy.cdf(self.a, loc=lo, scale=sc)
        cdf /= (stats.cauchy.cdf(self.b, loc=lo, scale=sc) - stats.cauchy.cdf(self.a, loc=lo, scale=sc))
        return cdf

    def _ppf(self, q, lo, sc):
        """ Need to implement ppf and not only the pdf adn cdf if you want to draw quickly sample with rvs(). To speed up the inversion, we use interpolation.
            Direct computation may be more efficient. """
        from scipy.interpolate import interp1d
        x_interp = np.linspace(self.a, self.b, 1000)
        cdf = self._cdf(x_interp, lo=lo, sc=sc)
        return interp1d(cdf, x_interp, kind='cubic')(q)


class truncnorm(stats.rv_continuous):
    """
    A truncated normal continuous random variable, where the range ``[a, b]`` is user-provided.

    Similar remark than `utils.trunccauchy`.

    Note: I implemented a new truncnorm function to have exactly the same behaviour than `utils.trunccauchy` instead of used `scipy.stats.truncnorm`
    """

    def _argcheck(*args):
        """ by default _argcheck return true only for args > 0, this is not our case since we use loc for scipy.norm which could be negative..."""
        return True

    def _pdf(self, x, lo, sc):
        """ Without any optimzation, pdf is the only function that we need to define a prbability law. """
        return stats.norm.pdf(x, loc=lo, scale=sc) / (stats.norm.cdf(self.b, loc=lo, scale=sc) - stats.norm.cdf(self.a, loc=lo, scale=sc))

    def _cdf(self, x, lo, sc):
        """ Need to implement cdf and not only the pdf to compute ppf efficiently ! """
        cdf = stats.norm.cdf(x, loc=lo, scale=sc) - stats.norm.cdf(self.a, loc=lo, scale=sc)
        cdf[x < self.a] = 0
        cdf[x > self.b] = stats.norm.cdf(self.b, loc=lo, scale=sc) - stats.norm.cdf(self.a, loc=lo, scale=sc)
        cdf /= (stats.norm.cdf(self.b, loc=lo, scale=sc) - stats.norm.cdf(self.a, loc=lo, scale=sc))
        return cdf

    def _ppf(self, q, lo, sc):
        """ Need to implement ppf and not only the pdf adn cdf if you want to draw quickly sample with rvs(). To speed up the inversion, we use interpolation. """
        from scipy.interpolate import interp1d
        x_interp = np.linspace(self.a, self.b, 1000)
        cdf = self._cdf(x_interp, lo=lo, sc=sc)
        return interp1d(cdf, x_interp, kind='cubic')(q)

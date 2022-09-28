"""A few utilities."""

import numpy as np

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

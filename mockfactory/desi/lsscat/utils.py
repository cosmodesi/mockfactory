"""Tables, joins, grouping and region selection shared by the catalog stages."""

import logging

import numpy as np

# The table helpers are shared with altmtl; imported here so the stages keep one import.
from ..tables import NULL, as_table, set_column, encode_keys, match, join_left  # noqa: F401


logger = logging.getLogger('lsscat')


def last_of_each(key, sort=None, tie=None):
    """
    Return the index of one row per distinct value of ``key``: the one with the largest
    ``sort``, and among those the one with the smallest ``tie``.

    Ties are common and they matter. The rows a target can be kept at are ranked by what they
    say about it, and at a full survey about two in five targets have several rows at the top
    of that ranking, equally good by every criterion the ranking uses. Which one is kept still
    decides the fiber location the target is charged to, and so the completeness its
    neighbours are weighted by. The survey pipeline leaves the choice to an unstable sort,
    which makes its catalogs irreproducible at that level; here it is settled by ``tie``, so
    that the same inputs give the same catalog however the rows were assembled.

    The indices come back sorted, so the result is in the order of the input rather than in
    the order of the sort key.
    """
    size = len(key)
    if size == 0:
        return np.zeros(0, dtype='i8')
    position = np.arange(size)
    last = position if tie is None else -np.asarray(tie)
    order = np.lexsort((last,) + ((position,) if sort is None else (sort,)) + (key,))
    sorted_key = np.asarray(key)[order]
    is_last = np.empty(size, dtype='?')
    is_last[-1] = True
    is_last[:-1] = sorted_key[1:] != sorted_key[:-1]
    return np.sort(order[is_last])


def group_fraction(key, weights):
    """
    Return, for each row, the mean of ``weights`` over the rows sharing its ``key``.

    The survey pipeline builds this as a dictionary in a Python loop over the distinct keys and
    then reads it back in a second loop over the rows; at the tens of millions of fiber
    locations of a full survey that is most of the cost of the stage.
    """
    _, dense, counts = np.unique(key, return_inverse=True, return_counts=True)
    return np.bincount(dense, weights=np.asarray(weights, dtype='f8'),
                       minlength=len(counts))[dense] / counts[dense]


def get_photsys(ra, dec):
    """
    Return the photometric system, ``'N'`` for the BASS/MzLS imaging and ``'S'`` for DECaLS.

    North is the part of the North Galactic Cap above declination 32.375, the rest is South.
    """
    ra, dec = np.asarray(ra), np.asarray(dec)
    ngc = (ra > 100. - dec) & (ra < 280. + dec)
    return np.where(ngc & (dec > 32.375), 'N', 'S').astype('U1')


def get_galactic_cap(ra, dec):
    """Return ``True`` for the targets in the North Galactic Cap."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    coord = SkyCoord(ra=np.asarray(ra) * u.degree, dec=np.asarray(dec) * u.degree, frame='icrs')
    return np.asarray(coord.galactic.b.value) > 0.

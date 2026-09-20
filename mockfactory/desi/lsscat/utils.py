"""Joins, grouping and region selection shared by the catalog stages."""

import logging

import numpy as np


logger = logging.getLogger('lsscat')


#: Value the survey pipeline uses for a column that a left join left empty.
NULL = 999999


def encode_keys(*keys):
    """
    Turn several key columns into one integer label per distinct combination.

    The catalog stages join and group on keys that are either wide (``TARGETID``) or compound
    (``TARGETID``, ``LOCATION``, ``TILEID``), and packing them into a single integer by
    arithmetic overflows or collides. Each column is replaced by a dense code instead, and the
    codes are combined in a width that is known to fit.
    """
    keys = [np.asarray(key) for key in keys]
    code = np.zeros(len(keys[0]), dtype='i8')
    for key in keys:
        _, dense = np.unique(key, return_inverse=True)
        code = code * (dense.max() + 1 if len(dense) else 1) + dense
    return code


def match(left, right):
    """
    Return, for each entry of ``left``, the index of the entry of ``right`` holding the same
    key, and -1 where there is none. ``right`` must hold each key at most once.
    """
    left, right = np.asarray(left), np.asarray(right)
    if not len(right):
        return np.full(len(left), -1, dtype='i8')
    order = np.argsort(right, kind='stable')
    position = np.searchsorted(right[order], left)
    index = order[np.clip(position, 0, len(order) - 1)]
    return np.where(right[index] == left, index, -1)


def join_left(left, right, keys, columns=None, fill=None, rename=None):
    """
    Add ``columns`` of ``right`` to ``left``, matched on ``keys``, filling the rows of ``left``
    that ``right`` has no entry for.

    This stands in for :func:`astropy.table.join` with ``join_type='left'``, for the case every
    join of the pipeline is in: the right-hand side holds each key once, so the result has
    exactly the rows of ``left``, in their order. It returns plain arrays rather than the masked
    columns astropy produces, so a filled value has to be given for each added column.

    Parameters
    ----------
    left, right : array
        Structured arrays. ``right`` must hold each key combination at most once.
    keys : str, list
        Name, or names, of the columns to match on.
    columns : list, default=None
        Columns of ``right`` to add. Defaults to all of them but the keys.
    fill : dict, default=None
        Value to give a column where the key is absent from ``right``. Defaults to ``nan`` for
        a floating column, :data:`NULL` for an integer one, and to the type's zero otherwise,
        which is what the survey pipeline's masked columns come out as when written.
    rename : dict, default=None
        New name for a column of ``right``, for the cases where it would clash.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    if columns is None:
        columns = [name for name in right.dtype.names if name not in keys]
    fill, rename = dict(fill or {}), dict(rename or {})
    if len(keys) == 1:
        index = match(left[keys[0]], right[keys[0]])
    else:
        # Encoding the two sides apart would give them unrelated labels; encode them together.
        size = len(left)
        code = encode_keys(*[np.concatenate([left[key], right[key]]) for key in keys])
        index = match(code[:size], code[size:])
    absent = index < 0
    logger.info('joined on {}: {:d} of {:d} rows unmatched'.format(keys, absent.sum(), len(left)))
    toret = append_fields(left, [(rename.get(name, name), right[name].dtype, right[name].shape[1:])
                                 for name in columns])
    for name in columns:
        column = right[name][np.where(absent, 0, index)]
        if absent.any():
            value = fill.get(name, None)
            if value is None:
                value = np.nan if column.dtype.kind == 'f' \
                    else NULL if column.dtype.kind in 'iu' else column.dtype.type()
            column = np.where(absent.reshape((-1,) + (1,) * (column.ndim - 1)), value, column)
        toret[rename.get(name, name)] = column
    return toret


def append_fields(array, fields):
    """Return ``array`` with ``fields``, a list of ``(name, dtype, shape)``, added to it."""
    fields = [field if len(field) == 3 else tuple(field) + ((),) for field in fields]
    dtype = [(name, array.dtype[name].base, array.dtype[name].shape) for name in array.dtype.names]
    dtype += [(name, dt, shape) for name, dt, shape in fields if name not in array.dtype.names]
    toret = np.empty(len(array), dtype=dtype)
    for name in array.dtype.names:
        toret[name] = array[name]
    return toret


def drop_fields(array, names):
    """Return ``array`` without the columns ``names``."""
    names = [names] if isinstance(names, str) else list(names)
    keep = [name for name in array.dtype.names if name not in names]
    toret = np.empty(len(array), dtype=[(name, array.dtype[name].base, array.dtype[name].shape)
                                        for name in keep])
    for name in keep:
        toret[name] = array[name]
    return toret


def select_fields(array, names):
    """
    Return ``array`` cut to ``names``, as a new array rather than a view.

    Indexing a structured array with a list of names gives back a view that keeps the original
    itemsize and the original field offsets, so its fields are out of order and it cannot be
    written to a fits file. Copying is what makes the result a catalog rather than a window on
    one.
    """
    names = [name for name in names if name in array.dtype.names]
    toret = np.empty(len(array), dtype=[(name, array.dtype[name].base, array.dtype[name].shape)
                                        for name in names])
    for name in names:
        toret[name] = array[name]
    return toret


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

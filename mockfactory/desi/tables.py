"""
Tables: the one in-memory type the single-process stages of :mod:`mockfactory.desi` share.

:mod:`~mockfactory.desi.altmtl` and :mod:`~mockfactory.desi.lsscat` take and return
:class:`astropy.table.Table`, the type desitarget and fiberassign already speak. The stages that
run over MPI, :mod:`~mockfactory.desi.base` and the building of target catalogs, keep
:class:`mpytools.Catalog`; :func:`as_table` is where the two meet, and it wraps a catalog's
columns without copying them.

Two habits keep a table as cheap as the structured arrays it replaced, which copied every
column each time one was added:

- a column is added with :func:`set_column`, which does not copy it, rather than by plain
  assignment, which does;
- columns are dropped or kept with ``remove_columns`` or ``keep_columns`` on
  ``table.copy(copy_data=False)``, since ``table[names]`` copies every column it keeps.

And one for speed: arithmetic on a :class:`~astropy.table.Column` is about 1.8 times slower
than on the array behind it, so a hot expression reads ``table[name].value``.
"""

import logging

import numpy as np
from astropy.table import Table


logger = logging.getLogger('tables')


#: Value the survey pipeline uses for a column that a left join left empty.
NULL = 999999


def as_table(array):
    """
    Return ``array`` as an :class:`astropy.table.Table`, the table every stage takes and returns.

    A table stores its columns apart, so adding one costs that column and nothing else; the
    structured arrays this replaces copied every column each time one was added, and a random
    catalog gains about fifteen on its way through the stages.

    A :class:`~astropy.table.Table` is returned as is. An ``mpytools`` catalog, as the rest of
    ``mockfactory`` produces, is wrapped without copying its columns. A structured array, as
    :func:`fitsio.read` gives, has each column copied out, so that the table holds contiguous
    columns and does not keep the whole record array alive.
    """
    import mpytools
    if isinstance(array, Table):
        return array
    if isinstance(array, mpytools.Catalog):
        return Table({name: array.get(name, return_type=None) for name in array.columns()},
                     copy=False)
    array = np.asarray(array)
    return Table({name: np.ascontiguousarray(array[name]) for name in array.dtype.names},
                 copy=False)


def set_column(table, name, value, dtype=None):
    """
    Set column ``name`` of ``table`` to ``value``, typed as a structured array field would be.

    The column keeps its own type if it exists, and takes ``dtype`` if it does not; ``dtype``
    of None keeps the type of ``value``. A scalar fills every row. ``value`` is not copied,
    which plain assignment to a table would do.
    """
    if name in table.colnames:
        dtype = table[name].dtype
    value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype, copy=False)
    if not value.ndim:
        value = np.full(len(table), value, dtype=value.dtype)
    if name in table.colnames:
        table.replace_column(name, value, copy=False)
    else:
        table.add_column(value, name=name, copy=False)
    return table


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
    # The queries are sorted before they are probed, and the answers scattered back. A binary
    # search over a sorted array of tens of millions is a cache miss at nearly every level, and
    # probing it in a random order pays that for every query; probing it in order walks the same
    # memory the array is laid out in. Measured on a join of 89 million rows against 34 million,
    # 93 s against 17 s, for the same indices.
    argsort = np.argsort(left, kind='stable')
    position = np.empty(len(left), dtype='i8')
    position[argsort] = np.searchsorted(right[order], left[argsort])
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
    left, right : Table, array
        Tables, ``mpytools`` catalogs or structured arrays. ``right`` must hold each key
        combination at most once.
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

    Returns
    -------
    table : Table
        A new table; the columns of ``left`` are shared with it, not copied.
    """
    left, right = as_table(left), as_table(right)
    keys = [keys] if isinstance(keys, str) else list(keys)
    if columns is None:
        columns = [name for name in right.colnames if name not in keys]
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
    toret = left.copy(copy_data=False)
    for name in columns:
        column = right[name].value[np.where(absent, 0, index)]
        if absent.any():
            value = fill.get(name, None)
            if value is None:
                value = np.nan if column.dtype.kind == 'f' \
                    else NULL if column.dtype.kind in 'iu' else column.dtype.type()
            # Back to the type of ``right``: the fill value may have promoted it.
            column = np.where(absent.reshape((-1,) + (1,) * (column.ndim - 1)), value,
                              column).astype(right[name].dtype, copy=False)
        # A column ``left`` already has keeps its type, as a structured array's field would.
        set_column(toret, rename.get(name, name), column)
    return toret

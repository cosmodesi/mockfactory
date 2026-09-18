"""
Compatibility shim for building ledgers under numpy 2.4 and above.

:func:`desitarget.io.write_mtl` works out the data release of a ledger with::

    dr = np.unique(release // 1000)
    ...
    drint = int(dr)

``int()`` on a one-element array was deprecated in numpy 1.25 and raises from numpy 2.4 on.
A catalog with a single data release, which is every catalog, therefore fails, and the
``TypeError`` is caught and re-raised as "Multiple data releases in MTL ([0])" -- a message
that says the opposite of what happened. The code is identical in every desitarget released
so far, so this is not something a newer version fixes.

Rather than pin numpy or shadow the installed desitarget, :func:`patch_write_mtl` makes that
one ``np.unique`` return an array that ``int()`` still accepts, for the duration of the call.
It is a no-op on a numpy that does not need it.
"""

import contextlib
import logging

import numpy as np


logger = logging.getLogger('altmtl.compat')


class _IntableArray(np.ndarray):
    """
    An array that ``int()`` accepts while it holds a single element.

    Only ``__int__`` differs from a plain array, so it stays a one-dimensional array for the
    ``len(dr) == 0`` test that desitarget makes just before converting it.
    """
    def __int__(self):
        flat = self.reshape(-1)
        if flat.size != 1:
            raise TypeError('cannot convert an array of {:d} elements to an integer'.format(flat.size))
        return int(flat[0])


class _NumpyProxy(object):
    """Stands in for the numpy module, returning intable arrays from :func:`numpy.unique`."""
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    def unique(self, *args, **kwargs):
        result = self._module.unique(*args, **kwargs)
        # With return_index and friends, unique returns a tuple, which is left alone.
        if isinstance(result, self._module.ndarray) and result.ndim == 1:
            return result.view(_IntableArray)
        return result


def numpy_converts_size_one_arrays():
    """Return whether this numpy still lets ``int()`` convert a one-element array."""
    try:
        int(np.array([0]))
    except TypeError:
        return False
    return True


@contextlib.contextmanager
def patch_write_mtl():
    """
    Make :func:`desitarget.io.write_mtl` survive numpy 2.4, for the duration of the block.

    Ledgers are built by forked worker processes, which inherit the patch, so this has to be
    entered before the pool is started, as :func:`make_initial_ledgers` does.
    """
    import desitarget.io

    if numpy_converts_size_one_arrays():
        yield False
        return

    original = desitarget.io.np
    desitarget.io.np = _NumpyProxy(original)
    logger.info('Patched desitarget.io for numpy {}: int() no longer converts one-element '
                'arrays, which desitarget.io.write_mtl relies on.'.format(np.__version__))
    try:
        yield True
    finally:
        desitarget.io.np = original

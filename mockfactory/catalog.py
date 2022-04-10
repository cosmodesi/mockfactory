"""Base classes to handle catalog of objects."""

import os
import re
import functools
import math

import numpy as np

from . import mpi, utils
from .mpi import MPI, CurrentMPIComm
from .utils import BaseClass


class Slice(BaseClass):

    def __init__(self, *args, size=None, copy=False):
        if len(args) > 1:
            sl = slice(*args)
        else:
            sl = args[0]
        if isinstance(sl, self.__class__):
            self.__dict__.update(sl.__dict__)
            if self.is_array: self.idx = np.array(self.idx, copy=copy)
            return
        elif isinstance(sl, slice):
            if size is not None:
                start, stop, step = sl.indices(size)
            else:
                start, stop, step = sl.start, sl.stop, sl.step
                if step is None: step = 1
                if step == 0: raise ValueError('Zero step')
                if start is None:
                    if step > 0: start = 0
                    else: raise ValueError('Input slice must be bounded, or provide "size"')
                if stop is None:
                    if step < 0: stop = -1
                    else: raise ValueError('Input slice must be bounded, or provide "size"')
                if start < 0: start = 0
                if step < 0 and stop > start or step > 0 and stop < start: stop = start
            stop = (stop - start + (-1) ** (step > 0)) // step * step + start + (-1) ** (step < 0)
            if stop < 0: stop = None
            sl = slice(start, stop, step)
        else:
            sl = np.array(sl, copy=copy)
            if not (np.issubdtype(sl.dtype, np.integer) or sl.dtype == '?'):
                raise ValueError('If array, must be of integer or boolean type')
            if sl.dtype == '?':
                sl = np.flatnonzero(sl)
        self.idx = sl

    @property
    def is_array(self):
        return not isinstance(self.idx, slice)

    def to_array(self, copy=False):
        if self.is_array:
            return np.array(self.idx, copy=copy)
        return np.arange(self.idx.start, self.idx.stop, self.idx.step)

    def to_slices(self):
        from itertools import groupby
        from operator import itemgetter
        from collections import deque
        if self.is_array:
            diff = np.diff(self.idx)
            diff = np.insert(diff, 0, diff[0])
            # This is a bit inefficient in the sense that for [0, 1, 2, 3, 5, 6, 8, 10, 12]
            # we will obtain (0, 4, 1), (5, 6, 1), (6, 7, 1), (8, 13, 2)
            for k, g in groupby(zip(self.idx, diff), lambda x: x[1]):
                ind = map(itemgetter(0), g)
                start = stop = next(ind)
                try:
                    second = stop = next(ind)
                except StopIteration:
                    yield slice(start, start + 1, 1)
                    continue
                step = second - start
                try:
                    stop = deque(ind, maxlen=1).pop()
                except IndexError:
                    pass
                stop = stop + (-1) ** (step < 0)
                if stop < 0: stop = None
                yield slice(start, stop, step)
        else:
            yield self.idx

    def split(self, nsplits=1):
        if self.is_array:
            idxs = np.array_split(self.idx, nsplits)
        else:
            idxs = []
            for isplit in range(nsplits):
                start = isplit * self.size // nsplits
                stop = (isplit + 1) * self.size // nsplits
                if self.idx.step < 0:
                    start, stop = -start, -stop
                start += self.idx.start
                stop += self.idx.start
                if stop < 0: stop = None
                idxs.append(slice(start, stop, self.idx.step))
        return [Slice(idx, copy=False) for idx in idxs]

    def find(self, *args, return_index=False):
        sl2 = self.__class__(*args)
        if self.is_array or sl2.is_array:
            if return_index:
                idx, idx2 = utils.match1d_to(self.to_array(), sl2.to_array(), return_index=True)
            else:
                idx = utils.match1d_to(self.to_array(), sl2.to_array(), return_index=False)
        else:
            step1, step2, delta = self.idx.step, sl2.idx.step, sl2.idx.start - self.idx.start
            gcd = math.gcd(abs(step1), abs(step2))  # gcd always positive
            if delta % gcd != 0:
                idx = idx2 = slice(0, 0, 1)
            else:
                # Search solution
                a, b, c = abs(step1 // gcd), abs(step2 // gcd), delta // gcd
                if c == 0:
                    x0 = 0
                else:
                    for x0 in range(0, b):
                        if (a * x0) % b == 1: break
                    x0 *= c
                step = step2 // gcd
                if step1 < 0:
                    x0 *= -1
                    step *= -1
                # Positivity of ii1 & ii2
                stepa = step1 * step
                imin = (max(self.min, sl2.min) - step1 * x0 - self.idx.start) / stepa
                imax = (min(self.max, sl2.max) - step1 * x0 - self.idx.start) / stepa

                if stepa < 0:
                    imin, imax = imax, imin
                istart = math.ceil(imin)
                istop = math.floor(imax)
                if istop < istart:
                    idx = idx2 = slice(0, 0, 1)
                else:
                    start = step * istart + x0
                    stop = step * istop + x0 + (-1) ** (step < 0)
                    if step < 0 and stop < 0: stop = None
                    idx = slice(start, stop, step)
                    # indices in sl2
                    start = (step1 * (x0 + step * imin) - delta) // step2
                    stop = (step1 * (x0 + step * imax) - delta) // step2 + 1
                    step = step1 * step // step2  # always positive
                    idx2 = slice(start, stop, step)

        idx = self.__class__(idx, copy=False)
        if return_index:
            return idx, self.__class__(idx2, copy=False)
        return idx

    @classmethod
    def empty(cls):
        return cls(slice(0, 0, 1))

    def slice(self, *args):
        sl2 = self.__class__(*args)
        if self.is_array:
            idx = self.idx[sl2.idx]
        elif sl2.is_array:
            idx = sl2.idx[self.idx]
        else:
            # I = a i + b
            # I' = a' I + b' = a' a i + a' b + b'
            x0 = sl2.idx.step * self.idx.start + sl2.idx.start
            step = sl2.idx.step * self.idx.step
            min2 = self.idx.step * sl2.min + self.idx.start
            max2 = self.idx.step * sl2.max + self.idx.start
            if self.idx.step < 0: min2, max2 = max2, min2
            imin = (max(self.min, min2) - x0) / step
            imax = (min(self.max, max2) - x0) / step
            if step < 0:
                imin, imax = imax, imin
            istart = math.ceil(imin)
            istop = math.floor(imax)
            if istop < istart:
                return self.empty()
            start = step * istart + x0
            stop = step * istop + x0 + (-1) ** (step < 0)
            if step < 0 and stop < 0: stop = None
            idx = slice(start, stop, step)
        return self.__class__(idx, copy=False)

    def shift(self, offset=0, size=None):
        if self.is_array:
            idx = self.idx + offset
            idx = idx[idx >= 0]
            if size is not None:
                idx = idx[idx < size]
        else:
            start = max(self.idx.start + offset, 0)
            stop = self.idx.stop + offset
            if size is not None:
                start = min(start, size)
                stop = min(stop, size)
            step = self.idx.step
            if step < 0 and stop < 0: stop = None
            idx = slice(start, stop, step)
        return self.__class__(idx, copy=False)

    def __len__(self):
        if self.is_array:
            return self.idx.size
        return ((-1 if self.idx.stop is None else self.idx.stop) - self.idx.start + (-1)**(self.idx.step > 0)) // self.idx.step + 1

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def min(self):
        if self.is_array:
            return self.idx.min()
        if self.idx.step < 0:
            return self.idx.step * (self.size - 1) + self.idx.start
        return self.idx.start

    @property
    def max(self):
        if self.is_array:
            return self.idx.max()
        if self.idx.step > 0:
            return self.idx.step * (self.size - 1) + self.idx.start
        return self.idx.start

    @classmethod
    def snap(cls, *others):
        others = [Slice(other) for other in others]
        if any(other.is_array for other in others):
            return [Slice(np.concatenate([other.to_array() for other in others], axis=0))]
        if not others:
            return [Slice(0, 0, 0)]
        slices = [others[0].idx]
        for other in others:
            if other.idx.step == slices[-1].idx.step and other.idx.start == slices[-1].idx.stop:
                slices[-1] = slice(slices[-1].idx.start, other.idx.stop, slices[-1].idx.step)
        return [Slice(sl) for sl in slices]

    @CurrentMPIComm.enable
    def mpi_send(self, dest, tag=0, blocking=True, mpicomm=None):
        if blocking: send = mpicomm.send
        else: send = mpicomm.isend
        send(self.is_array, dest=dest, tag=tag + 1)
        if self.is_array:
            mpi.send_array(self.idx, dest=dest, tag=tag, blocking=blocking, mpicomm=mpicomm)
        else:
            send(self.idx, dest=dest, tag=tag)

    @classmethod
    @CurrentMPIComm.enable
    def mpi_recv(cls, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
        if mpicomm.recv(source=source, tag=tag + 1):  # is_array
            idx = mpi.recv_array(source=source, tag=tag, mpicomm=mpicomm)
        else:
            idx = mpicomm.recv(source=source, tag=tag)
        return cls(idx)


class MPIScatteredSource(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, *slices, csize=None, mpicomm=None):
        # let's restrict to disjoint slices...
        self.mpicomm = mpicomm
        self.slices = [Slice(sl, size=csize) for sl in slices]
        if any(sl.is_array for sl in self.slices):
            raise NotImplementedError('Only slices supported so far')
        if csize is not None and csize != self.csize:
            raise ValueError('Input slices do not have collective size "csize"')

    @property
    def size(self):
        if getattr(self, '_size', None) is None:
            self._size = sum(sl.size for sl in self.slices)
        return self._size

    @property
    def csize(self):
        return self.mpicomm.allreduce(self.size)

    def get(self, arrays, *args):
        # Here, slice in global coordinates
        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        if len(arrays) != len(self.slices):
            raise ValueError('Expected list of arrays of length {:d}, found {:d}'.format(len(self.slices), len(arrays)))
        size = sum(map(len, arrays))
        if size != self.size:
            raise ValueError('Expected list of arrays of total length {:d}, found {:d}'.format(self.size, size))
        if not args:
            args = (slice(self.mpicomm.rank * self.csize // self.mpicomm.size, (self.mpicomm.rank + 1) * self.csize // self.mpicomm.size, 1), )

        all_slices = self.mpicomm.allgather(self.slices)
        nslices = max(map(len, all_slices))
        toret = []

        for sli in args:
            sli = Slice(sli, size=self.csize)
            idx, tmp = [None] * self.mpicomm.size, [None] * self.mpicomm.size
            for irank in range(self.mpicomm.size):
                self_slice_in_irank = [sl.find(sli, return_index=True) for sl in all_slices[irank]]
                idx[irank] = [sl[1].idx for sl in self_slice_in_irank]
                if irank == self.mpicomm.rank:
                    tmp[irank] = [array[sl[0].idx] for iarray, (array, sl) in enumerate(zip(arrays, self_slice_in_irank))]
                else:
                    for isl, sl in enumerate(self_slice_in_irank): sl[0].mpi_send(dest=irank, tag=isl)
                    self_slice_in_irank = [Slice.mpi_recv(source=irank, tag=isl) for isl in range(len(self.slices))]
                    for iarray, (array, sl) in enumerate(zip(arrays, self_slice_in_irank)):
                        mpi.send_array(array[sl.idx], dest=irank, tag=nslices + iarray, mpicomm=self.mpicomm)
                    tmp[irank] = [mpi.recv_array(source=irank, tag=nslices + iarray, mpicomm=self.mpicomm) for iarray in range(len(self_slice_in_irank))]
            idx, tmp = _add_list(idx), _add_list(tmp)
            if sli.is_array:
                toret.append(np.concatenate(tmp)[np.argsort(np.concatenate(idx, axis=0))], axis=0)
            else:
                toret += [tmp[ii] for ii in np.argsort([iidx.start for iidx in idx])]
        return np.concatenate(toret)

    @classmethod
    def concatenate(cls, *others):
        slices, cumsize = [], 0
        for other in others:
            slices += [sl.shift(cumsize) for sl in other.slices]
            cumsize += other.csize
        return cls(*slices)

    def extend(self, other, **kwargs):
        new = self.concatenate(self, other, **kwargs)
        self.__dict__.update(new.__dict__)


def _multiple_columns(column):
    return isinstance(column, (list, tuple))


def vectorize_columns(func):
    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not _multiple_columns(column):
            return func(self, column, **kwargs)
        toret = [func(self, col, **kwargs) for col in column]
        if all(t is None for t in toret):  # in case not broadcast to all ranks
            return None
        return np.asarray(toret)
    return wrapper


def _select_columns(columns, include=None, exclude=None):

    def toregex(name):
        return name.replace('.', '\.').replace('*', '(.*)')

    if not _multiple_columns(columns):
        columns = [columns]

    toret = columns

    if include is not None:
        if not _multiple_columns(include):
            include = [include]
        toret = []
        for inc in include:
            inc = toregex(inc)
            for column in columns:
                if re.match(inc, str(column)):
                    toret.append(column)
        columns = toret

    if exclude is not None:
        if not _multiple_columns(exclude):
            exclude = [exclude]
        toret = []
        for exc in exclude:
            exc = toregex(exc)
            for column in columns:
                if re.match(exc, str(column)) is None:
                    toret.append(column)

    return toret


def _get_shape(size, itemshape):
    # join size and itemshape to get total shape
    if np.ndim(itemshape) == 0:
        return (size, itemshape)
    return (size,) + tuple(itemshape)


def _dict_to_array(data, struct=True):
    """
    Return dict as numpy array.

    Parameters
    ----------
    data : dict
        Data dictionary of name: array.

    struct : bool, default=True
        Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
        If ``False``, numpy will attempt to cast types of different columns.

    Returns
    -------
    array : array
    """
    array = [(name, data[name]) for name in data]
    if struct:
        array = np.empty(array[0][1].shape[0], dtype=[(name, col.dtype, col.shape[1:]) for name, col in array])
        for name in data: array[name] = data[name]
    else:
        array = np.array([col for _, col in array])
    return array


def _add_list(li):
    toret = []
    for el in li:
        if isinstance(el, (tuple, list)):
            toret += list(el)
        else:
            toret.append(el)
    return toret


class FileStack(BaseClass):

    @CurrentMPIComm.enable
    def __init__(self, *files, filetype=None, mpicomm=None, **kwargs):
        """
        Initialize :class:`FileStack`.

        Parameters
        ----------
        filename : string, list of strings
            File name(s).

        attrs : dict, default=None
            File attributes. Will be complemented by those read from disk.
            These will eventually be written to disk.

        mode : string, default=''
            If 'r', read file header (necessary for further reading of file columns).

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.files = []
        self.mpicomm = mpicomm
        for file in _add_list(files):
            if isinstance(file, BaseFile):
                self.files.append(file)
            else:
                FT = get_filetype(filetype=filetype, filename=file)
                self.files.append(FT(file, mpicomm=self.mpicomm, **kwargs))
        for file in self.files:
            if file.mpicomm is not self.mpicomm:
                raise ValueError('Input files with different mpicomm')
        self.mpiroot = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    @property
    def filesizes(self):
        return [file.csize for file in self.files]

    @property
    def cfilesize(self):
        return sum(self.filesizes)

    @property
    def slices(self):
        if getattr(self, '_slices', None) is None:
            self._slices = [Slice(self.mpicomm.rank * self.cfilesize // self.mpicomm.size, (self.mpicomm.rank + 1) * self.cfilesize // self.mpicomm.size, 1)]
        return self._slices

    @property
    def columns(self):
        if getattr(self, '_columns', None) is None:
            if not self.files:
                return []
            self._columns = [column for column in self.files[0].columns if all(column in file.columns for file in self.files[1:])]
        return self._columns

    @property
    def header(self):
        if getattr(self, '_header', None) is None:
            self._header = {}
            for file in self.files:
                self._header.update(file.header)
        return self._header

    @property
    def fileslices(self):
        # catalog slices
        cumsizes = np.cumsum([0] + self.filesizes)
        for sli in self.slices:
            yield [Slice(start, stop, 1).find(sli) for start, stop in zip(cumsizes[:-1], cumsizes[1:])]

    @property
    def size(self):
        return sum(sl.size for sl in self.slices)

    @property
    def csize(self):
        return self.mpicomm.allreduce(self.size)

    @property
    def _is_slice_array(self):
        return any(_add_list(self.mpicomm.allgather([sl.is_array for sl in self.slices])))

    def slice(self, *args):
        new = self.copy()
        sl = Slice(*args)
        new._slices = [sli.slice(sl) for sli in self.slices]
        return new

    def loadbalance(self):
        new_slice = Slice(self.mpicomm.rank * self.csize // self.mpicomm.csize, (self.mpicomm.rank + 1) * self.csize // self.mpicomm.csize, 1)
        if self._is_slice_array:
            cumsizes = np.cumsum([sum(self.mpicomm.allgather(self.size)[:self.mpicomm.rank])] + [sl.size for sl in self.slices])
            slices = [slice(size1, size2, 1) for size1, size2 in zip(cumsizes[:-1], cumsizes[1:])]
            source = MPIScatteredSource(*slices)
            self._slices = [source.get([sl.to_array() for sl in self._slices], new_slice)]
        else:
            all_slices = _add_list(self.mpicomm.allgather(self.slices))
            slices = []
            cumsize = 0
            for sli in all_slices:
                sl = sli.slice(new_slice.shift(-cumsize))
                if sl: slices.append(sl)
                cumsize += sl.size
            self._slices = Slice.snap(slices)

    def cslice(self, *args, loadbalance=True):
        new = self.copy()
        global_slice = Slice(*args, csize=self.csize)
        cumsizes = np.cumsum([sum(self.mpicomm.allgather(self.size)[:self.mpicomm.rank])] + [sl.size for sl in self.slices])
        new._slices = [sli.slice(global_slice.shift(-cumsizes[isli])) for isli, sli in enumerate(self.slices)]
        if loadbalance:
            new.loadbalance()
        return new

    def concatenate(cls, *others):
        new = cls(*_add_list(*(other.files for other in others)))
        if any(getattr(other, '_slices', None) is not None for other in others):
            if any(other._is_slice_array for other in others):
                csize = sum(other.csize for other in others)
                new_slice = Slice(new.mpicomm.rank * csize // new.mpicomm.csize, (new.mpicomm.rank + 1) * csize // new.mpicomm.csize, 1)
                source = []
                for other in others:
                    cumsizes = np.cumsum([sum(new.mpicomm.allgather(other.size)[:self.mpicomm.rank])] + [sl.size for sl in other.slices])
                    slices = [slice(size1, size2, 1) for size1, size2 in zip(cumsizes[:-1], cumsizes[1:])]
                    source.append(MPIScatteredSource(*slices))
                source = MPIScatteredSource.concatenate(*source)
                new._slices = [source.get(_add_list([other._slices for other in others]), new_slice)]
            else:
                slices, cumsize = [], 0
                for other in others:
                    slices += _add_list(new.mpicomm.allgather([sl.shift(cumsize) for sl in other.slices]))
                    cumsize += other.csize
                new._slices = slices if new.mpicomm.rank == 0 else []
            new.loadbalance()
        return new

    def extend(self, other, **kwargs):
        new = self.concatenate(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    def read(self, column):
        """Read column of name ``column``."""
        toret = []
        for islice, slices in enumerate(self.fileslices):
            for ifile, rows in enumerate(slices):
                if rows: toret.append(self.files[ifile].read(column, rows=rows))
        return np.concatenate(toret, axis=0)

    def write(self, data, mpiroot=None):
        isdict = None
        if self.mpicomm.rank == mpiroot or mpiroot is None:
            isdict = isinstance(data, dict)
        if mpiroot is not None:
            isdict = self.mpicomm.bcast(isdict, root=mpiroot)
            if isdict:
                columns = self.mpicomm.bcast(list(data.keys()) if self.mpicomm.rank == mpiroot else None, root=mpiroot)
                data = {name: mpi.scatter_array(data[name] if self.mpicomm.rank == mpiroot else None, mpicomm=self.mpicomm, root=self.mpiroot) for name in columns}
            else:
                data = mpi.scatter_array(data, mpicomm=self.mpicomm, root=self.mpiroot)
        if isdict:
            for name in data: size = len(data[name]); break
        else:
            size = len(data)

        csize = self.mpicomm.allreduce(size)
        nfiles = len(self.files)
        for ifile, file in enumerate(self.files):
            file._csize = (ifile + 1) * csize // nfiles - ifile * csize // nfiles
        self._slices = None
        fcumsizes = np.cumsum([0] + self.filesizes)
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
        for islice, slices in enumerate(self.fileslices):
            for ifile, rows in enumerate(slices):
                rows = rows.shift(fcumsizes[ifile] - cumsizes[self.mpicomm.rank])
                if isdict:
                    self.files[ifile].write({name: data[name][rows.idx] for name in data})
                else:
                    self.files[ifile].write(data[rows.idx])


def get_filetype(filetype=None, filename=None):

    if filetype is None:
        if filename is not None:
            ext = os.path.splitext(filename)[1][1:]
            for filetype in RegisteredFile._registry.values():
                if ext in filetype.extensions:
                    return filetype
            raise IOError('Extension {} is unknown'.format(ext))
    if isinstance(filetype, str):
        filetype = RegisteredFile._registry[filetype.lower()]

    return filetype


class RegisteredFile(type(BaseClass)):

    _registry = {}

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        meta._registry[cls.name] = cls
        return cls


class BaseFile(BaseClass, metaclass=RegisteredFile):
    """
    Base class to read/write a file from/to disk.
    File handlers should extend this class, by (at least) implementing :meth:`read`, :meth:`get` and :meth:`write`.
    """
    name = 'base'
    extensions = []
    _type_read_rows = ['slice', 'index']
    _type_write_data = ['dict', 'array']

    @CurrentMPIComm.enable
    def __init__(self, filename, mpicomm=None):
        """
        Initialize :class:`BaseFile`.

        Parameters
        ----------
        filename : string
            File name.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.filename = filename
        self.mpicomm = mpicomm
        self.mpiroot = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    def _read_header(self):
        if self.is_mpi_root():
            self.log_info('Reading {}.'.format(self.filename))
            state = self._read_header_root()
            state['_csize'] = int(state.pop('csize'))
            state['_columns'] = list(state.pop('columns'))
            state['_header'] = dict(state.pop('header', {}))
        state = self.mpicomm.bcast(state if self.is_mpi_root() else None, root=self.mpiroot)
        self.__dict__.update(state)
        return state['_header']

    @property
    def csize(self):
        if getattr(self, '_csize', None) is None:
            self._read_header()
        return self._csize

    @property
    def columns(self):
        if getattr(self, '_columns', None) is None:
            self._read_header()
        return self._columns

    @property
    def header(self):
        if getattr(self, '_header', None) is None:
            self._read_header()
        return self._header

    def read(self, column, rows=slice(None)):
        """Read column of name ``column``."""
        sl = Slice(rows, size=self.csize)
        rows = [sl.idx]
        if sl.is_array:
            if 'index' not in self._type_read_rows:
                rows = sl.to_slices()
        else:
            if 'slice' not in self._type_read_rows:
                rows = [sl.to_array()]
        return np.concatenate([self._read_rows(column, rows=row) for row in rows], axis=0)

    def write(self, data, header=None):
        """
        Write input data to file(s).

        Parameters
        ----------
        data : array, dict
            Data to write.
        """
        if self.is_mpi_root():
            self.log_info('Writing {}.'.format(self.filename))
        isdict = isinstance(data, dict)
        if isdict:
            if 'dict' not in self._type_write_data:
                data = _dict_to_array(data)
        else:
            data = np.asarray(data)
            if 'array' not in self._type_write_data:
                data = {name: data[name] for name in data.dtype.names}
        self._write_data(data, header=header or {})

    def _read_header_root(self):
        raise NotImplementedError

    def _read_rows(self, column, rows):
        raise NotImplementedError

    def _write_data(self, data, header):
        raise NotImplementedError


try: import fitsio
except ImportError: fitsio = None


class FITSFile(BaseFile):
    """
    Class to read/write a FITS file from/to disk.

    Note
    ----
    In some circumstances (e.g. catalog has just been written), :meth:`get` fails with a file not found error.
    We have tried making sure processes read the file one after the other, but that does not solve the issue.
    A similar issue happens with nbodykit - though at a lower frequency.
    """
    name = 'fits'
    extensions = ['fits']
    _type_read_rows = ['index']
    _type_write_data = ['array']

    def __init__(self, filename, ext=None, **kwargs):
        """
        Initialize :class:`FitsFile`.

        Parameters
        ----------
        filename : string
            File name.

        ext : int, default=None
            FITS extension. Defaults to first extension with data.

        kwargs : dict
            Arguments for :class:`BaseFile`.
        """
        if fitsio is None:
            raise ImportError('Install fitsio')
        self.ext = ext
        super(FITSFile, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        # Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/fits.py
        with fitsio.FITS(self.filename) as file:
            if getattr(self, 'ext') is None:
                for i, hdu in enumerate(file):
                    if hdu.has_data():
                        self.ext = i
                        break
                if self.ext is None:
                    raise IOError('{} has no binary table to read'.format(self.filename))
            else:
                if isinstance(self.ext, str):
                    if self.ext not in file:
                        raise IOError('{} does not contain extension with name {}'.format(self.filename, self.ext))
                elif self.ext >= len(file):
                    raise IOError('{} extension {} is not valid'.format(self.filename, self.ext))
            file = file[self.ext]
            # make sure we crash if data is wrong or missing
            if not file.has_data() or file.get_exttype() == 'IMAGE_HDU':
                raise IOError('{} extension {} is not a readable binary table'.format(self.filename, self.ext))
            return {'csize': file.get_nrows(), 'columns': file.get_rec_dtype()[0].names, 'attrs': dict(file.read_header()), 'ext': self.ext}

    def _read_rows(self, column, rows):
        return fitsio.read(self.filename, ext=self.ext, columns=column, rows=rows)

    def _write_data(self, data, header):
        data = mpi.gather_array(data, mpicomm=self.mpicomm, root=self.mpiroot)
        if self.is_mpi_root():
            fitsio.write(self.filename, data, header=header, clobber=True)


try: import h5py
except ImportError: h5py = None


class HDF5File(BaseFile):
    """
    Class to read/write a HDF5 file from/to disk.

    Note
    ----
    In some circumstances (e.g. catalog has just been written), :meth:`get` fails with a file not found error.
    We have tried making sure processes read the file one after the other, but that does not solve the issue.
    A similar issue happens with nbodykit - though at a lower frequency.
    """
    name = 'hdf5'
    extensions = ['hdf', 'h4', 'hdf4', 'he2', 'h5', 'hdf5', 'he5', 'h5py']
    _type_read_rows = ['slice', 'index']
    _type_write_data = ['dict']

    def __init__(self, filename, group='/', **kwargs):
        """
        Initialize :class:`HDF5File`.

        Parameters
        ----------
        filename : string
            File name.

        group : string, default='/'
            HDF5 group where columns are located.

        kwargs : dict
            Arguments for :class:`BaseFile`.
        """
        if h5py is None:
            raise ImportError('Install h5py')
        self.group = group
        if not group or group == '/' * len(group):
            self.group = '/'
        super(HDF5File, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        with h5py.File(self.filename, 'r') as file:
            grp = file[self.group]
            columns = list(grp.keys())
            size = grp[columns[0]].shape[0]
            for name in columns:
                if grp[name].shape[0] != size:
                    raise IOError('Column {} has different length (expected {:d}, found {:d})'.format(name, size, grp[name].shape[0]))
            return {'csize': size, 'columns': columns, 'attrs': dict(grp.attrs)}

    def _read_rows(self, column, rows):
        with h5py.File(self.filename, 'r') as file:
            grp = file[self.group]
            return grp[column][rows]

    def _write_data(self, data, header):
        driver = 'mpio'
        kwargs = {'comm': self.mpicomm}
        import h5py
        try:
            h5py.File(self.filename, 'w', driver=driver, **kwargs)
        except ValueError:
            driver = None
            kwargs = {}
        if driver == 'mpio':
            for name in data: size = len(data[name]); break
            with h5py.File(self.filename, 'w', driver=driver, **kwargs) as file:
                cumsizes = np.cumsum([0] + self.mpicomm.allgather(size))
                start, stop = cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1]
                csize = cumsizes[-1]
                grp = file
                if self.group != '/':
                    grp = file.create_group(self.group)
                grp.attrs.update(self.attrs)
                for name in data:
                    dset = grp.create_dataset(name, shape=(csize,) + data[name].shape[1:], dtype=data[name].dtype)
                    dset[start:stop] = data[name]
        else:
            if self.is_mpi_root():
                h5py.File(self.filename, 'w', driver=driver, **kwargs)
            first = True
            for name in data:
                array = mpi.gather_array(data[name], mpicomm=self.mpicomm, root=self.mpiroot)
                if self.is_mpi_root():
                    with h5py.File(self.filename, 'a', driver=driver, **kwargs) as file:
                        grp = file
                        if first:
                            if self.group != '/':
                                grp = file.create_group(self.group)
                            grp.attrs.update(header)
                        dset = grp.create_dataset(name, data=array)
                first = False


from numpy.lib.format import open_memmap


class BinaryFile(BaseFile):
    """Class to read/write a binary file from/to disk."""
    name = 'bin'
    extensions = ['npy']
    _type_read_rows = ['slice', 'index']
    _type_write_data = ['array']

    def _read_header_root(self):
        array = open_memmap(self.filename, mode='r')
        return {'csize': len(array), 'columns': array.dtype.names, 'attrs': {}}

    def _read_rows(self, column, rows):
        return open_memmap(self.filename, mode='r')[rows][column]

    def _write_data(self, data, header):
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(len(data)))
        if self.is_mpi_root():
            fp = open_memmap(self.filename, mode='w+', dtype=data.dtype, shape=(cumsizes[-1],))
        self.mpicomm.Barrier()
        start, stop = cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1]
        fp = open_memmap(self.filename, mode='r+')
        fp[start:stop] = data
        fp.flush()



import json
try: import bigfile
except ImportError: bigfile = None


class BigFile(BaseFile):
    """Class to read/write a BigFile from/to disk."""
    name = 'bigfile'
    extensions = ['bigfile']
    _type_read_rows = ['slice']
    _type_write_data = ['dict']

    def __init__(self, filename, group='/', **kwargs):
        """
        Initialize :class:`BigFile`.

        Parameters
        ----------
        filename : string
            File name.

        group : string, default='/'
            BigFile group where columns are located.

        kwargs : dict
            Arguments for :class:`BaseFile`.
        """
        if bigfile is None:
            raise ImportError('Install bigfile')
        self.group = group
        if not group or group == '/' * len(group):
            self.group = '/'
        if not self.group.endswith('/'): self.group = self.group + '/'
        super(BigFile, self).__init__(filename=filename, **kwargs)

    def _read_header_root(self):
        with bigfile.File(filename=self.filename) as file:
            columns = [block for block in file[self.group].blocks]
            header = self.header
            if header is None: header = ['Header', 'header', '.']
            if not isinstance(header, (tuple, list)): header = [header]
            headers = []
            for h in header:
                if h in file.blocks and h not in headers: headers.append(h)
            # Append the dataset itself
            headers.append(self.group.strip('/') + '/.')

            exclude = self.exclude
            if exclude is None:
                # By default exclude header only.
                exclude = headers

            columns = _select_columns(columns, exclude=exclude)
            csize = bigfile.Dataset(file[self.group], columns).csize

            attrs = {}
            for header in headers:
                # copy over the attrs
                for key, value in file[header].attrs.items():
                    # load a JSON representation if str starts with json:://
                    if isinstance(value, str) and value.startswith('json://'):
                        attrs[key] = json.loads(value[7:])  # , cls=JSONDecoder)
                    # copy over an array
                    else:
                        attrs[key] = np.array(value, copy=True)
            return {'csize': csize, 'columns': columns, 'attrs': attrs}

    def _read_rows(self, column, rows):
        with bigfile.File(filename=self.filename)[self.dataset] as file:
            return bigfile.Dataset(file, column)[rows.start:rows.stop][::rows.step]

    def _write_data(self, data, header):
        # trim out any default columns; these do not need to be saved as
        # they are automatically available to every Catalog
        columns = list(data.keys())

        # FIXME: merge this logic into bigfile
        # the slice writing support in bigfile 0.1.47 does not
        # support tuple indices.
        class _ColumnWrapper:

            def __init__(self, bb):
                self.bb = bb

            def __setitem__(self, sl, value):
                assert len(sl) <= 2  # no array shall be of higher dimension.
                # use regions argument to pick the offset.
                start, stop, step = sl[0].indices(self.bb.size)
                assert step == 1
                if len(sl) > 1:
                    start1, stop1, step1 = sl[1].indices(value.shape[1])
                    assert step1 == 1
                    assert start1 == 0
                    assert stop1 == value.shape[1]
                self.bb.write(start, value)

        with bigfile.FileMPI(comm=self.mpicomm, filename=self.filename, create=True) as file:

            sources, targets, regions = [], [], []

            # save meta data and create blocks, prepare for the write.
            for column in columns:
                array = data[column]
                column = self.group + column
                # ensure data is only chunked in the first dimension
                size = self.comm.allreduce(len(array))
                offset = np.sum(self.comm.allgather(len(array))[:self.comm.rank], dtype='i8')

                # sane value -- 32 million items per physical file
                size_per_file = 32 * 1024 * 1024

                nfiles = (size + size_per_file - 1) // size_per_file

                dtype = np.dtype((array.dtype, array.shape[1:]))

                # save column attrs too
                # first create the block on disk
                with file.create(column, dtype, size, nfiles) as bb:
                    pass

                # first then open it for writing
                bb = file.open(column)

                targets.append(_ColumnWrapper(bb))
                sources.append(array)
                regions.append((slice(offset, offset + len(array)),))

            # writer header afterwards, such that header can be a block that saves
            # data.
            if header is not None:
                try:
                    bb = file.open('header')
                except:
                    bb = file.create('header')
                with bb:
                    for key in self.attrs:
                        try:
                            bb.attrs[key] = self.attrs[key]
                        except ValueError:
                            try:
                                json_str = 'json://' + json.dumps(self.attrs[key])
                                bb.attrs[key] = json_str
                            except:
                                raise ValueError('Cannot save {} key in attrs dictionary'.format(key))

            # lock=False to avoid dask from pickling the lock with the object.
            # write blocks one by one
            for column, source, target, region in zip(columns, sources, targets, regions):
                if self.is_mpi_root():
                    self.log_info('Started writing column {}'.format(column))
                source.store(target, regions=region)
                target.bb.close()
                if self.is_mpi_root():
                    self.log_info('Finished writing column {}'.format(column))


class BaseCatalog(BaseClass):

    _attrs = ['attrs']

    """Base class that represents a catalog, as a dictionary of columns stored as arrays."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, attrs=None, mpicomm=None):
        """
        Initialize :class:`BaseCatalog`.

        Parameters
        ----------
        data : dict, BaseCatalog
            Dictionary of {name: array}.

        columns : list, default=None
            List of column names.
            Defaults to ``data.keys()``.

        attrs : dict, default=None
            Dictionary of other attributes.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.
        """
        self.data = {}
        if columns is None:
            columns = list((data or {}).keys())
        if data is not None:
            for name in columns:
                self[name] = data[name]
        self.attrs = attrs or {}
        self.mpicomm = mpicomm
        self.mpiroot = 0

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    @classmethod
    def from_nbodykit(cls, catalog, columns=None):
        """
        Build new catalog from **nbodykit**.

        Parameters
        ----------
        catalog : nbodykit.base.catalog.CatalogSource
            **nbodykit** catalog.

        columns : list, default=None
            Columns to import. Defaults to all columns.

        Returns
        -------
        catalog : BaseCatalog
        """
        if columns is None: columns = catalog.columns
        data = {col: catalog[col].compute() for col in columns}
        return cls(data, mpicomm=catalog.comm, attrs=catalog.attrs)

    def to_nbodykit(self, columns=None):
        """
        Return catalog in **nbodykit** format.

        Parameters
        ----------
        columns : list, default=None
            Columns to export. Defaults to all columns.

        Returns
        -------
        catalog : nbodykit.source.catalog.ArrayCatalog
        """
        if columns is None: columns = self.columns()
        source = {col: self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        attrs = {key: value for key, value in self.attrs.items() if key != 'fitshdr'}
        return ArrayCatalog(source, **attrs)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        keys = list(self.data.keys())
        if not keys:
            if self.has_source is not None:
                return self._source.size
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def csize(self):
        """Return catalog global size, i.e. sum of size in each process."""
        return self.mpicomm.allreduce(len(self))

    def columns(self, include=None, exclude=None):
        """
        Return catalog column names, after optional selections.

        Parameters
        ----------
        include : list, string, default=None
            Single or list of *regex* patterns to select column names to include.
            Defaults to all columns.

        exclude : list, string, default=None
            Single or list of *regex* patterns to select column names to exclude.
            Defaults to no columns.

        Returns
        -------
        columns : list
            Return catalog column names, after optional selections.
        """
        toret = None

        if self.is_mpi_root():
            columns = list(self.data.keys())
            source = getattr(self, '_source', None)
            if source is not None:
                columns += [column for column in source.columns if column not in columns]
            toret = _select_columns(columns, include=include, exclude=exclude)

        return self.mpicomm.bcast(toret, root=self.mpiroot)

    def __contains__(self, column):
        """Whether catalog contains column name ``column``."""
        return column in self.data or (self.has_source and column in self._source.columns)

    def __iter__(self):
        """Iterate on catalog columns."""
        return iter(self.data)

    def cindices(self):
        """Row numbers in the global catalog."""
        sizes = self.mpicomm.allgather(len(self))
        sizes = np.cumsum([0] + sizes)
        return sizes[self.mpicomm.rank] + np.arange(len(self))

    def zeros(self, itemshape=(), dtype='f8'):
        """Return array of size :attr:`size` filled with zero."""
        return np.zeros(_get_shape(len(self), itemshape), dtype=dtype)

    def ones(self, itemshape=(), dtype='f8'):
        """Return array of size :attr:`size` filled with one."""
        return np.ones(_get_shape(len(self), itemshape), dtype=dtype)

    def full(self, fill_value, itemshape=(), dtype='f8'):
        """Return array of size :attr:`size` filled with ``fill_value``."""
        return np.full(_get_shape(len(self), itemshape), fill_value, dtype=dtype)

    def falses(self, itemshape=()):
        """Return array of size :attr:`size` filled with ``False``."""
        return self.zeros(itemshape=itemshape, dtype=np.bool_)

    def trues(self, itemshape=()):
        """Return array of size :attr:`size` filled with ``True``."""
        return self.ones(itemshape=itemshape, dtype=np.bool_)

    def nans(self, itemshape=()):
        """Return array of size :attr:`size` filled with :attr:`numpy.nan`."""
        return self.ones(itemshape=itemshape) * np.nan

    @property
    def has_source(self):
        return getattr(self, '_source', None) is not None

    def get(self, column, *args, **kwargs):
        """Return catalog (local) column ``column`` if exists, else return provided default."""
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
        if column in self.data:
            return self.data[column]
        # if not in data, try in _source
        if self.has_source and column in self._source.columns:
            self.data[column] = self._source.read(column)
            return self.data[column]
        if has_default:
            return default
        raise KeyError('Column {} does not exist'.format(column))

    def set(self, column, item):
        """Set column of name ``column``."""
        self.data[column] = item

    def cget(self, column, mpiroot=None):
        """
        Return on process rank ``root`` catalog global column ``column`` if exists, else return provided default.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` return result on all processes.
        """
        if mpiroot is None: mpiroot = Ellipsis
        return mpi.gather_array(self[column], mpicomm=self.mpicomm, root=mpiroot)

    def cslice(self, *args, loadbalance=True):
        """
        Perform global slicing of catalog,
        e.g. ``catalog.cslice(0, 100, 1)`` will return a new catalog of global size ``100``.
        Same reference to :attr:`attrs`.
        """
        new = self.copy()
        cumsizes = np.cumsum([0] + self.mpicomm.allgather(self.size))
        local_slice = slice(cumsizes[self.mpicomm.rank], cumsizes[self.mpicomm.rank + 1], 1)
        global_slice = Slice(slice(*args))
        if loadbalance:
            source = MPIScatteredSource(local_slice)
            sl = global_slice.split(self.mpicomm.size)[self.mpicomm.rank]
            for column in self.columns():
                if column in self.data:
                    new[column] = source.get(self[column], sl)
        else:
            sl = local_slice.find(global_slice)
            for column in self.columns():
                if column in self.data:
                    new[column] = self[column][sl]
        if self.has_source:
            self._source = self._source.cslice(global_slice, loadbalance=loadbalance)
        return new

    @classmethod
    def concatenate(cls, *others, keep_order=False):
        """
        Concatenate catalogs together.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        keep_order : bool, default=False
            Whether to keep row order, which requires costly MPI-gather/scatter operations.
            If ``False``, rows on each MPI process will be added to those of the same MPI process.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        if not others:
            raise ValueError('Provide at least one {} instance.'.format(cls.__name__))
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs
        new_columns = new.columns()

        for other in others:
            other_columns = other.columns()
            if other.mpicomm is not new.mpicomm:
                raise ValueError('Input catalogs with different mpicomm')
            if new_columns and other_columns and set(other_columns) != set(new_columns):
                raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns, new_columns))

        in_data = {column: any(column in other.data for other in others) for column in new_columns}
        if keep_order and any(in_data.values()):
            source = []
            for other in others:
                cumsizes = np.cumsum([0] + other.mpicomm.allgather(other.size))
                source.append(MPIScatteredSource(slice(cumsizes[other.mpicomm.rank], cumsizes[other.mpicomm.rank + 1], 1)))
            source = MPIScatteredSource.concatenate(*source)

        for column in new_columns:
            if in_data[column]:
                if keep_order:
                    new[column] = source.get([other[column] for other in others])
                else:
                    new[column] = np.concatenate([other[column] for other in others])

        source = [other._source for other in others if other.has_source]
        if source:
            source = FileStack.concatenate(*source, keep_order=keep_order)
            new._source = source

        return new

    def extend(self, other, **kwargs):
        """Extend catalog with ``other``."""
        new = self.concatenate(self, other, **kwargs)
        self.__dict__.update(new.__dict__)

    def to_array(self, columns=None, struct=True):
        """
        Return catalog as *numpy* array.

        Parameters
        ----------
        columns : list, default=None
            Columns to use. Defaults to all catalog columns.

        struct : bool, default=True
            Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
            If ``False``, *numpy* will attempt to cast types of different columns.

        Returns
        -------
        array : array
        """
        if columns is None:
            columns = self.columns()
        data = {col: self[col] for col in columns}
        return _dict_to_array(data, struct=struct)

    @classmethod
    @CurrentMPIComm.enable
    def from_array(cls, array, columns=None, mpicomm=None, mpiroot=None, **kwargs):
        """
        Build :class:`BaseCatalog` from input ``array``.

        Parameters
        ----------
        array : array, dict
            Input array to turn into catalog.

        columns : list, default=None
            List of columns to read from array.
            If ``None``, inferred from ``array``.

        mpicomm : MPI communicator, default=None
            MPI communicator.

        mpiroot : int, default=None
            If ``None``, input array is assumed to be scattered across all ranks.
            Else the MPI rank where input array is gathered.

        kwargs : dict
            Other arguments for :meth:`__init__`.

        Returns
        -------
        catalog : BaseCatalog
        """
        isstruct = None
        if mpicomm.rank == mpiroot or mpiroot is None:
            isstruct = isdict = not hasattr(array, 'dtype')
            if isdict:
                if columns is None: columns = list(array.keys())
            else:
                isstruct = array.dtype.names is not None
                if isstruct and columns is None: columns = array.dtype.names
        if mpiroot is not None:
            isstruct = mpicomm.bcast(isstruct, root=mpiroot)
            columns = mpicomm.bcast(columns, root=mpiroot)
        columns = list(columns)
        new = cls(data=dict.fromkeys(columns), mpicomm=mpicomm, **kwargs)

        def get(column):
            value = None
            if mpicomm.rank == mpiroot or mpiroot is None:
                if isstruct:
                    value = array[column]
                else:
                    value = columns.index(column)
            if mpiroot is not None:
                return mpi.scatter_array(value, mpicomm=mpicomm, root=mpiroot)
            return value

        new.data = {column: get(column) for column in columns}
        return new

    def copy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
        new = super(BaseCatalog, self).__copy__()
        if columns is None: columns = self.columns()
        new.data = {col: self[col] if col in self else None for col in columns}
        if new.has_source: new._source = self._source.copy()
        import copy
        for name in new._attrs:
            if hasattr(self, name):
                tmp = copy.copy(getattr(self, name))
                setattr(new, name, tmp)
        return new

    def deepcopy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
        import copy
        new = self.copy(columns=columns)
        for name in self._attrs:
            if hasattr(self, name):
                setattr(new, name, copy.deepcopy(getattr(self, name)))
        new.data = {col: self[col].copy() for col in new}
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name): col for name, col in self.data.items()}
        state = {'data': data}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.__dict__.update(state)

    @classmethod
    @CurrentMPIComm.enable
    def from_state(cls, state, mpicomm=None):
        """Create class from state."""
        new = cls.__new__(cls)
        new.__setstate__(state)
        new.mpicomm = mpicomm
        new.mpiroot = 0
        return new

    def __getitem__(self, name):
        """Get catalog column ``name`` if string, else return copy with local slice."""
        if isinstance(name, str):
            return self.get(name)
        new = self.copy()
        if isinstance(name, slice):
            new.data = {col: self[col][name] for col in self.data}
            if self.has_source:
                new._source = self._source.slice(name)
        else:
            new.data = {col: self[col][name] for col in self.columns()}
            if self.has_source: del new._source
        return new

    def __setitem__(self, name, item):
        """Set catalog column ``name`` if string, else set slice ``name`` of all columns to ``item``."""
        if isinstance(name, str):
            return self.set(name, item)
        for col in self.columns():
            self[col][name] = item

    def __delitem__(self, name):
        """Delete column ``name``."""
        try:
            del self.data[name]
        except KeyError as exc:
            if self.has_source is not None:
                self._source.columns.remove(name)
            else:
                raise KeyError('Column {} not found'.format(name)) from exc

    def __repr__(self):
        """Return string representation of catalog, including global size and columns."""
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__, self.csize, self.columns())

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and columns? (ignoring :attr:`attrs`)"""
        if not isinstance(other, self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        assert self.mpicomm == other.mpicomm
        self, other = self.cslice(0, None), other.cslice(0, None)
        for col in self_columns:
            self_value = self.get(col)
            other_value = other.get(col)
            if not all(self.mpicomm.allgather(np.all(self_value == other_value))):
                return False

    @classmethod
    def read(cls, *args, **kwargs):
        source = FileStack(*args, **kwargs)
        new = cls(attrs={'header': source.header}, mpicomm=source.mpicomm)
        new._source = source
        return new

    def write(self, *args, **kwargs):
        """Save catalog to ``filename``."""
        source = FileStack(*args, **kwargs)
        source.write({name: self[name] for name in self.columns()})

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, filename, mpicomm=None):
        """
        Load catalog in *npy* binary format from disk.

        Parameters
        ----------
        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        mpiroot = 0
        if mpicomm.rank == mpiroot:
            cls.log_info('Loading {}.'.format(filename))
            state = np.load(filename, allow_pickle=True)[()]
            data = state.pop('data')
            columns = list(data.keys())
        else:
            state = None
            columns = None
        state = mpicomm.bcast(state, root=mpiroot)
        columns = mpicomm.bcast(columns, root=mpiroot)
        state['data'] = {}
        for name in columns:
            state['data'][name] = mpi.scatter_array(data[name] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, root=mpiroot)
        return cls.from_state(state, mpicomm=mpicomm)

    def save(self, filename):
        """Save catalog to ``filename`` as *npy* file."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        state['data'] = {name: self.cget(name, mpiroot=self.mpiroot) for name in self.columns()}
        if self.is_mpi_root():
            np.save(filename, state, allow_pickle=True)

    @vectorize_columns
    def csum(self, column, axis=0):
        """Return global sum of column(s) ``column``."""
        return mpi.sum_array(self[column], axis=axis, mpicomm=self.mpicomm)

    @vectorize_columns
    def caverage(self, column, weights=None, axis=0):
        """Return global average of column(s) ``column``, with weights ``weights`` (defaults to ``1``)."""
        return mpi.average_array(self[column], weights=weights, axis=axis, mpicomm=self.mpicomm)

    @vectorize_columns
    def cmean(self, column, axis=0):
        """Return global mean of column(s) ``column``."""
        return self.caverage(column, axis=axis)

    @vectorize_columns
    def cmin(self, column, axis=0):
        """Return global minimum of column(s) ``column``."""
        return mpi.min_array(self[column], axis=axis, mpicomm=self.mpicomm)

    @vectorize_columns
    def cmax(self, column, axis=0):
        """Return global maximum of column(s) ``column``."""
        return mpi.max_array(self[column], axis=axis, mpicomm=self.mpicomm)


class Catalog(BaseCatalog):

    """A simple catalog."""

"""Base classes to handle catalog of objects."""

import os
import logging
import functools

import numpy as np

from . import mpi, utils
from .mpi import CurrentMPIComm
from .utils import BaseClass


def _multiple_columns(column):
    return isinstance(column, (list,tuple))


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


def _get_shape(size, itemshape):
    # join size and itemshape to get total shape
    if np.ndim(itemshape) == 0:
        return (size, itemshape)
    return (size,) + tuple(itemshape)


def _dict_to_array(data, struct=True):
    """
    Return dict as *numpy* array.

    Parameters
    ----------
    data : dict
        Data dictionary of name: array.

    struct : bool, default=True
        Whether to return structured array, with columns accessible through e.g. ``array['Position']``.
        If ``False``, *numpy* will attempt to cast types of different columns.

    Returns
    -------
    array : array
    """
    array = [(name,data[name]) for name in data]
    if struct:
        array = np.empty(array[0][1].shape[0], dtype=[(name, col.dtype, col.shape[1:]) for name,col in array])
        for name in data: array[name] = data[name]
    else:
        array = np.array([col for _,col in array])
    return array


class BaseFile(BaseClass):
    """
    Base class to read/write a file from/to disk.
    File handlers should extend this class, by (at least) implementing :meth:`read`, :meth:`get` and :meth:`write`.
    """
    @CurrentMPIComm.enable
    def __init__(self, filename, attrs=None, mpicomm=None, mpiroot=0):
        """
        Initialize :class:`BaseFile`.

        Parameters
        ----------
        filename : string
            File name.

        attrs : dict, default=None
            File attributes. Will be complemented by those read from disk.
            These will eventually be written to disk.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
        self.filename = filename
        self.attrs = attrs or {}
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot

    @property
    def start(self):
        """Start of the data local chunk."""
        return self.mpicomm.rank * self.gsize // self.mpicomm.size

    @property
    def stop(self):
        """End of the data local chunk."""
        return (self.mpicomm.rank + 1) * self.gsize // self.mpicomm.size

    @property
    def size(self):
        """Size of local data chunk."""
        return self.stop - self.start

    def is_mpi_root(self):
        """Whether current rank is root."""
        return self.mpicomm.rank == self.mpiroot

    def read(self):
        """
        Set :attr:`gsize`, :attr:`columns` and update attr:`attrs`.
        To be implemented in your file handler.
        """
        raise NotImplementedError('Implement method "read" in your "{}"-inherited file handler'.format(self.__class__.___name__))

    def get(self, column):
        """
        Read column from file.
        To be implemented in your file handler.
        """
        raise NotImplementedError('Implement method "get" in your "{}"-inherited file handler'.format(self.__class__.___name__))

    def write(self, data):
         """
         Write ``data`` (structured array or dict) to file.
         To be implemented in your file handler.
         """
         raise NotImplementedError('Implement method "write" in your "{}"-inherited file handler'.format(self.__class__.___name__))


try: import fitsio
except ImportError: fitsio = None


class FitsFile(BaseFile):
    """
    Class to read/write a FITS file from/to disk.

    Note
    ----
    In some circumstances (e.g. catalog has just been written), :meth:`get` fails with a file not found error.
    We have tried making sure processes read the file one after the other, but that does not solve the issue.
    A similar issue happens with nbodykit - though at a lower frequency.
    """
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
        super(FitsFile, self).__init__(filename=filename, **kwargs)

    def read(self):
        # Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/io/fits.py
        if self.is_mpi_root():
            self.log_info('Loading {}.'.format(self.filename))
            msg = 'Input FITS file {}'.format(self.filename)
            with fitsio.FITS(self.filename) as file:
                if self.ext is None:
                    for i, hdu in enumerate(file):
                        if hdu.has_data():
                            self.ext = i
                            break
                    if self.ext is None:
                        raise IOError('{} has no binary table to read'.format(msg))
                else:
                    if isinstance(self.ext, str):
                        if self.ext not in file:
                            raise IOError('{} does not contain extension with name {}'.format(msg, self.ext))
                    elif self.ext >= len(file):
                        raise IOError('{} extension {} is not valid'.format(msg, self.ext))
                file = file[self.ext]
                # make sure we crash if data is wrong or missing
                if not file.has_data() or file.get_exttype() == 'IMAGE_HDU':
                    raise ValueError('{} extension {} is not a readable binary table'.format(msg, self.ext))
                self.gsize = file.get_nrows()
                self.columns = file.get_rec_dtype()[0].names
                header = file.read_header()
                self.attrs.update(dict(header))
                header.clean()
                state = {name: getattr(self, name) for name in ['filename','gsize','columns','ext','attrs']}
        self.__dict__.update(self.mpicomm.bcast(state if self.is_mpi_root() else None, root=self.mpiroot))
        #self.mpicomm.Barrier() # necessary to avoid blocking due to file not found

    def get(self, column):
        return fitsio.read(self.filename, ext=self.ext, columns=column, rows=range(self.start,self.stop))
        #self.mpicomm.Barrier() # necessary to avoid blocking due to file not found
        #if not self.is_mpi_root():
        #    do = self.mpicomm.recv(source=self.mpicomm.rank-1, tag=42)
        #toret = fitsio.read(self.filename, ext=self.ext, columns=column, rows=range(self.start,self.stop))
        #if self.mpicomm.rank < self.mpicomm.size -1:
        #    self.mpicomm.send(True, dest=self.mpicomm.rank+1, tag=42)
        #return toret

    def write(self, data):
        """Possible to change fitsio to write by chunks?."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(self.filename))
            utils.mkdir(os.path.dirname(self.filename))
        if not isinstance(data, np.ndarray):
            data = _dict_to_array(data)
        array = mpi.gather_array(data, mpicomm=self.mpicomm, root=self.mpiroot)
        if self.is_mpi_root():
            fitsio.write(self.filename, array, header=self.attrs.get('fitshdr',None), clobber=True)


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
        if not group or group == '/'*len(group):
            self.group = '/'
        super(HDF5File, self).__init__(filename=filename, **kwargs)

    def read(self):
        if self.is_mpi_root():
            self.log_info('Loading {}.'.format(self.filename))
            with h5py.File(self.filename, 'r') as file:
                grp = file[self.group]
                self.attrs.update(dict(grp.attrs))
                self.columns = list(grp.keys())
                self.gsize = grp[self.columns[0]].shape[0]
                for name in self.columns:
                    if grp[name].shape[0] != self.gsize:
                        raise ValueError('Column {} has different length (expected {:d}, found {:d})'.format(name, self.gsize, grp[name].shape[0]))
                state = {name: getattr(self, name) for name in ['filename','gsize','columns','attrs']}
        self.__dict__.update(self.mpicomm.bcast(state if self.is_mpi_root() else None, root=self.mpiroot))
        #self.mpicomm.Barrier() # necessary to avoid blocking due to file not found

    def get(self, column):
        #self.mpicomm.Barrier() # necessary to avoid blocking due to file not found
        with h5py.File(self.filename, 'r') as file:
            grp = file[self.group]
            toret = grp[column][self.start:self.stop]
        return toret

    def write(self, data):
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(self.filename))
            utils.mkdir(os.path.dirname(self.filename))
        if isinstance(data, np.ndarray):
            data = {name: data[name] for name in data.dtype.names}
        driver = 'mpio'
        kwargs = {'comm': self.mpicomm}
        import h5py
        try:
            h5py.File(self.filename, 'w', driver=driver, **kwargs)
        except ValueError:
            driver = None
            kwargs = {}
        if driver == 'mpio':
            with h5py.File(self.filename, 'w', driver=driver, **kwargs) as file:
                csizes = np.cumsum([0] + self.mpicomm.allgather(self.size))
                start, stop = csizes[self.mpicomm.rank], csizes[self.mpicomm.rank+1]
                gsize = csizes[-1]
                grp = file
                if self.group != '/':
                    grp = file.create_group(self.group)
                grp.attrs.update(self.attrs)
                for name in data:
                    dset = grp.create_dataset(name, shape=(gsize,)+data[name].shape[1:], dtype=data[name].dtype)
                    dset[start:stop] = self[name]
        else:
            first = True
            for name in data:
                array = mpi.gather_array(data[name], mpicomm=self.mpicomm, root=self.mpiroot)
                if self.is_mpi_root():
                    with h5py.File(self.filename, 'w', driver=driver, **kwargs) as file:
                        grp = file
                        if first:
                            if self.group != '/':
                                grp = file.create_group(self.group)
                            grp.attrs.update(self.attrs)
                        dset = grp.create_dataset(name, data=array)
                first = False


class BaseCatalog(BaseClass):

    _attrs = ['attrs']

    """Base class that represents a catalog, as a dictionary of columns stored as arrays."""

    @CurrentMPIComm.enable
    def __init__(self, data=None, columns=None, attrs=None, mpicomm=None, mpiroot=0):
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
            Other attributes.

        mpicomm : MPI communicator, default=None
            The current MPI communicator.

        mpiroot : int, default=0
            The rank number to use as master.
        """
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
        return cls(data, mpicomm=catalog.comm, mpiroot=0, attrs=catalog.attrs)

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
        source = {col:self[col] for col in columns}
        from nbodykit.lab import ArrayCatalog
        attrs = {key:value for key,value in self.attrs.items() if key != 'fitshdr'}
        return ArrayCatalog(source, **attrs)

    def __len__(self):
        """Return catalog (local) length (``0`` if no column)."""
        keys = list(self.data.keys())
        if not keys:
            source = getattr(self, '_source', None)
            if source is not None:
                return source.size
            return 0
        return len(self[keys[0]])

    @property
    def size(self):
        """Equivalent for :meth:`__len__`."""
        return len(self)

    @property
    def gsize(self):
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
            allcols = set(self.data.keys())
            source = getattr(self, '_source', None)
            if source is not None:
                allcols |= set(source.columns)
            toret = allcols = list(allcols)

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
        """Whether catalog contains column name ``column``."""
        return column in self.data

    def __iter__(self):
        """Iterate on catalog columns."""
        return iter(self.data)

    def gindices(self):
        """Row numbers in the global catalog."""
        sizes = self.mpicomm.allgather(len(self))
        sizes = [0] + np.cumsum(sizes[:1]).tolist()
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
        return self.ones(itemshape=itemshape)*np.nan

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
        if getattr(self, '_source', None) is not None and column in self._source.columns:
            self.data[column] = self._source.get(column)
            return self.data[column]
        if has_default:
            return default
        raise KeyError('Column {} does not exist'.format(column))

    def set(self, column, item):
        """Set column of name ``column``."""
        self.data[column] = item

    def gget(self, column, mpiroot=None):
        """
        Return on process rank ``root`` catalog global column ``column`` if exists, else return provided default.
        If ``mpiroot`` is ``None`` or ``Ellipsis`` return result on all processes.
        """
        if mpiroot is None: mpiroot = Ellipsis
        return mpi.gather_array(self[column], mpicomm=self.mpicomm, root=mpiroot)

    def gslice(self, *args):
        """
        Perform global slicing of catalog,
        e.g. ``catalog.gslice(0,100,1)`` will return a new catalog of global size ``100``.
        Same reference to :attr:`attrs`.
        """
        sl = slice(*args)
        new = self.copy()
        for col in self.columns():
            self_value = self.gget(col,mpiroot=self.mpiroot)
            new[col] = mpi.scatter_array(self_value if self.is_mpi_root() else None,mpiroot=self.mpiroot,mpicomm=self.mpicomm)
        return new

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
    def from_array(cls, array, columns=None, mpicomm=None, mpiroot=0, **kwargs):
        """
        Build :class:`BaseCatalog` from input ``array``.

        Parameters
        ----------
        columns : list
            List of columns to read from array.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpistate : string, mpi.CurrentMPIState
            MPI state of the input array: 'scattered', 'gathered', 'broadcast'?

        mpicomm : MPI communicator, default=None
            MPI communicator.

        kwargs : dict
            Other arguments for :meth:`__init__`.

        Returns
        -------
        catalog : BaseCatalog
        """
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

    def copy(self, columns=None):
        """Return copy, including column names ``columns`` (defaults to all columns)."""
        new = super(BaseCatalog,self).__copy__()
        if columns is None: columns = self.columns()
        new.data = {col:self[col] if col in self else None for col in columns}
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
        new.data = {col:self[col].copy() for col in new}
        return new

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        state = {'data':data}
        for name in self._attrs:
            if hasattr(self, name):
                state[name] = getattr(self, name)
        return state

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.__dict__.update(state)

    @classmethod
    @CurrentMPIComm.enable
    def from_state(cls, state, mpiroot=0, mpicomm=None):
        """Create class from state."""
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def __getitem__(self, name):
        """Get catalog column ``name`` if string, else return copy with local slice."""
        if isinstance(name,str):
            return self.get(name)
        new = self.copy()
        new.data = {col:self[col][name] for col in self.data}
        return new

    def __setitem__(self, name, item):
        """Set catalog column ``name`` if string, else set slice ``name`` of all columns to ``item``."""
        if isinstance(name,str):
            return self.set(name, item)
        for col in self.data:
            self[col][name] = item

    def __delitem__(self, name):
        """Delete column ``name``."""
        try:
            del self.data[name]
        except KeyError as exc:
            source = getattr(self, '_source', None)
            if source is not None:
                source.columns.remove(name)
            else:
                raise KeyError('Column {} not found') from exc

    def __repr__(self):
        """Return string representation of catalog, including global size and columns."""
        return '{}(size={:d}, columns={})'.format(self.__class__.__name__,self.gsize,self.columns())

    @classmethod
    def concatenate(cls, *others):
        """
        Concatenate catalogs together.

        Parameters
        ----------
        others : list
            List of :class:`BaseCatalog` instances.

        Returns
        -------
        new : BaseCatalog

        Warning
        -------
        :attr:`attrs` of returned catalog contains, for each key, the last value found in ``others`` :attr:`attrs` dictionaries.
        """
        attrs = {}
        for other in others: attrs.update(other.attrs)
        others = [other for other in others if other.columns()]

        new = others[0].copy()
        new.attrs = attrs
        new_columns = new.columns()

        for other in others:
            other_columns = other.columns()
            assert new.mpicomm is other.mpicomm
            if new_columns and other_columns and set(other_columns) != set(new_columns):
                raise ValueError('Cannot extend samples as columns do not match: {} != {}.'.format(other_columns,new_columns))

        for column in new_columns:
            columns = [other.gget(column,root=new.mpiroot) for other in others]
            if new.is_mpi_root():
                new[column] = np.concatenate(columns,axis=0)
            new[column] = mpi.scatter_array(new[column] if new.is_mpi_root() else None, root=new.mpiroot, mpicomm=new.mpicomm)
        return new

    def extend(self, other):
        """Extend catalog with ``other``."""
        new = self.concatenate(self,other)
        self.__dict__.update(new.__dict__)

    def __eq__(self, other):
        """Is ``self`` equal to ``other``, i.e. same type and columns? (ignoring :attr:`attrs`)"""
        if not isinstance(other,self.__class__):
            return False
        self_columns = self.columns()
        other_columns = other.columns()
        if set(other_columns) != set(self_columns):
            return False
        assert self.mpicomm == other.mpicomm
        toret = True
        for col in self_columns:
            self_value = self.gget(col, mpiroot=self.mpiroot)
            other_value = other.gget(col, mpiroot=self.mpiroot)
            if self.is_mpi_root():
                if not np.all(self_value == other_value):
                    toret = False
                    break
        return self.mpicomm.bcast(toret, root=self.mpiroot)

    @classmethod
    @CurrentMPIComm.enable
    def load_fits(cls, filename, ext=None, mpiroot=0, mpicomm=None):
        """
        Load catalog in FITS binary format from disk.

        Parameters
        ----------
        filename : string
            File name to load catalog from.

        ext : int, default=None
            FITS extension. Defaults to first extension with data.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        source = FitsFile(filename, ext=ext, mpiroot=mpiroot, mpicomm=mpicomm)
        source.read()
        new = cls(attrs={'fitshdr': source.attrs})
        new._source = source
        return new

    def save_fits(self, filename):
        """Save catalog to ``filename`` as *fits* file."""
        source = FitsFile(filename, ext=1, mpiroot=self.mpiroot, mpicomm=self.mpicomm)
        source.write({name: self[name] for name in self.columns()})

    @classmethod
    @CurrentMPIComm.enable
    def load_hdf5(cls, filename, group='/', mpiroot=0, mpicomm=None):
        """
        Load catalog in HDF5 binary format from disk.

        Parameters
        ----------
        filename : string
            File name to load catalog from.

        group : string, default='/'
            HDF5 group where columns are located.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        source = HDF5File(filename, group=group, mpiroot=mpiroot, mpicomm=mpicomm)
        source.read()
        new = cls(attrs=source.attrs)
        new._source = source
        return new

    def save_hdf5(self, filename, group='/'):
        """
        Save catalog to disk in *hdf5* binary format.

        Parameters
        ----------
        filename : string
            File name where to save catalog.

        group : string, default='/'
            HDF5 group where columns are located.
        """
        source = HDF5File(filename, group=group, mpiroot=self.mpiroot, mpicomm=self.mpicomm)
        source.write({name: self[name] for name in self.columns()})

    @classmethod
    @CurrentMPIComm.enable
    def load(cls, filename, columns=None, mpiroot=0, mpicomm=None):
        """
        Load catalog in *npy* binary format from disk.

        Parameters
        ----------
        columns : list, default=None
            List of column names to read. Defaults to all columns.

        mpiroot : int, default=0
            Rank of process where input array lives.

        mpicomm : MPI communicator, default=None
            The MPI communicator.

        Returns
        -------
        catalog : BaseCatalog
        """
        if mpicomm.rank == mpiroot:
            cls.log_info('Loading {}.'.format(filename))
            state = np.load(filename, allow_pickle=True)[()]
            data = state.pop('data')
            if columns is None: columns = list(data.keys())
        else:
            state = None
        state = mpicomm.bcast(state, root=mpiroot)
        columns = mpicomm.bcast(columns, root=mpiroot)
        state['data'] = {}
        for name in columns:
            state['data'][name] = mpi.scatter_array(data[name] if mpicomm.rank == mpiroot else None, mpicomm=mpicomm, root=mpiroot)
        return cls.from_state(state, mpicomm=mpicomm, mpiroot=mpiroot)

    def save(self, filename):
        """Save catalog to ``filename`` as *npy* file."""
        if self.is_mpi_root():
            self.log_info('Saving to {}.'.format(filename))
            utils.mkdir(os.path.dirname(filename))
        state = self.__getstate__()
        state['data'] = {name: self.gget(name) for name in self.columns()}
        if self.is_mpi_root():
            np.save(filename, state, allow_pickle=True)

    @vectorize_columns
    def sum(self, column, axis=0):
        """Return global sum of column(s) ``column``."""
        return mpi.sum_array(self[column],axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def average(self, column, weights=None, axis=0):
        """Return global average of column(s) ``column``, with weights ``weights`` (defaults to ``1``)."""
        return mpi.average_array(self[column],weights=weights,axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def mean(self, column, axis=0):
        """Return global mean of column(s) ``column``."""
        return self.average(column,axis=axis)

    @vectorize_columns
    def minimum(self, column, axis=0):
        """Return global minimum of column(s) ``column``."""
        return mpi.min_array(self[column],axis=axis,mpicomm=self.mpicomm)

    @vectorize_columns
    def maximum(self, column, axis=0):
        """Return global maximum of column(s) ``column``."""
        return mpi.max_array(self[column],axis=axis,mpicomm=self.mpicomm)


class Catalog(BaseCatalog):

    """A simple catalog."""

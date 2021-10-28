"""
MPI routines, many taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py
and https://github.com/bccp/nbodykit/blob/master/nbodykit/batch.py.
"""

import functools

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from mpi4py import MPI


_default_mpi_comm = MPI.COMM_WORLD


def SetMPIComm(func):
    """
    Decorator to attach the current MPI communicator to the input
    keyword arguments of ``func``, via the ``mpicomm`` keyword.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs.setdefault('mpicomm', None)
        if kwargs['mpicomm'] is None:
            kwargs['mpicomm'] = _default_mpi_comm
        return func(*args, **kwargs)

    return wrapper


@SetMPIComm
def linspace_array(start, stop, num=50, mpicomm=None):
    step = (stop - start)/(num - 1.)
    istart = mpicomm.rank * num // mpicomm.size
    istop = (mpicomm.rank + 1) * num // mpicomm.size
    return start + step*np.arange(istart,istop)


@SetMPIComm
def front_pad_array(array, front, mpicomm=None):
    """
    Padding an array in the front with items before this rank.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    """
    N = np.array(mpicomm.allgather(len(array)), dtype='intp')
    offsets = np.cumsum(np.concatenate([[0], N], axis=0))
    mystart = offsets[mpicomm.rank] - front
    torecv = (offsets[:-1] + N) - mystart

    torecv[torecv < 0] = 0 # before mystart
    torecv[torecv > front] = 0 # no more than needed
    torecv[torecv > N] = N[torecv > N] # fully enclosed

    if mpicomm.allreduce(torecv.sum() != front, MPI.LOR):
        raise ValueError("cannot work out a plan to padd items. Some front values are too large. %d %d"
            % (torecv.sum(), front))

    tosend = mpicomm.alltoall(torecv)
    sendbuf = [ array[-items:] if items > 0 else array[0:0] for i, items in enumerate(tosend)]
    recvbuf = mpicomm.alltoall(sendbuf)
    return np.concatenate(list(recvbuf) + [array], axis=0)


class MPIRandomState(object):
    """
    A Random number generator that is invariant against number of ranks,
    when the total size of random number requested is kept the same.
    The algorithm here assumes the random number generator from numpy
    produces uncorrelated results when the seeds are sampled from a single
    RNG.
    The sampler methods are collective calls; multiple calls will return
    uncorrerlated results.
    The result is only invariant under diif mpicomm.size when allreduce(size)
    and chunksize are kept invariant.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/mpirng.py
    """
    @SetMPIComm
    def __init__(self, size, seed=None, chunksize=100000, mpicomm=None):
        self.mpicomm = mpicomm
        self.seed = seed
        self.chunksize = chunksize

        self.size = size
        self.csize = np.sum(mpicomm.allgather(size), dtype='intp')

        self._start = np.sum(mpicomm.allgather(size)[:mpicomm.rank], dtype='intp')
        self._end = self._start + self.size

        self._first_ichunk = self._start // chunksize

        self._skip = self._start - self._first_ichunk * chunksize

        nchunks = (mpicomm.allreduce(np.array(size, dtype='intp')) + chunksize - 1) // chunksize
        self.nchunks = nchunks

        self._serial_rng = np.random.RandomState(seed)

    def _prepare_args_and_result(self, args, itemshape, dtype):
        """
        pad every item in args with values from previous ranks,
        and create an array for holding the result with the same length.
        Returns
        -------
        padded_r, padded_args
        """
        r = np.zeros((self.size,) + tuple(itemshape), dtype=dtype)

        r_and_args = (r,) + tuple(args)
        r_and_args_b = np.broadcast_arrays(*r_and_args)

        padded = []

        # we don't need to pad scalars,
        # loop over broadcasted and non broadcast version to figure this out)
        for a, a_b in zip(r_and_args, r_and_args_b):
            if np.ndim(a) == 0:
                # use the scalar, no need to pad.
                padded.append(a)
            else:
                # not a scalar, pad
                padded.append(front_pad_array(a_b, self._skip, mpicomm=self.mpicomm))

        return padded[0], padded[1:]

    def poisson(self, lam, itemshape=(), dtype='f8'):
        """ Produce `self.size` poissons, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            lam, = args
            return rng.poisson(lam=lam, size=size)
        return self._call_rngmethod(sampler, (lam,), itemshape, dtype)

    def normal(self, loc=0, scale=1, itemshape=(), dtype='f8'):
        """ Produce `self.size` normals, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            loc, scale = args
            return rng.normal(loc=loc, scale=scale, size=size)
        return self._call_rngmethod(sampler, (loc, scale), itemshape, dtype)

    def uniform(self, low=0., high=1.0, itemshape=(), dtype='f8'):
        """ Produce `self.size` uniforms, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            low, high = args
            return rng.uniform(low=low, high=high,size=size)
        return self._call_rngmethod(sampler, (low, high), itemshape, dtype)

    def _call_rngmethod(self, sampler, args, itemshape, dtype='f8'):
        """
            Loop over the seed table, and call sampler(rng, args, size)
            on each rng, with matched input args and size.
            the args are padded in the front such that the rng is invariant
            no matter how self.size is distributed.
            truncate the return value at the front to match the requested `self.size`.
        """

        seeds = self._serial_rng.randint(0, high=0xffffffff, size=self.nchunks)

        padded_r, running_args = self._prepare_args_and_result(args, itemshape, dtype)

        running_r = padded_r
        ichunk = self._first_ichunk

        while len(running_r) > 0:
            # at most get a full chunk, or the remaining items
            nreq = min(len(running_r), self.chunksize)

            seed = seeds[ichunk]
            rng = np.random.RandomState(seed)
            args = tuple([a if np.ndim(a) == 0 else a[:nreq] for a in running_args])

            # generate nreq random items from the sampler
            chunk = sampler(rng, args=args,
                size=(nreq,) + tuple(itemshape))

            running_r[:nreq] = chunk

            # update running arrays, since we have finished nreq items
            running_r = running_r[nreq:]
            running_args = tuple([a if np.ndim(a) == 0 else a[nreq:] for a in running_args])

            ichunk = ichunk + 1

        return padded_r[self._skip:]


@SetMPIComm
def gather_array(data, mpiroot=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Gather the input data array from all ranks to the specified ``mpiroot``.
    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like
        the data on each rank to gather
    mpicomm : MPI communicator
        the MPI communicator
    mpiroot : int, or Ellipsis
        the rank number to gather the data to. If mpiroot is Ellipsis or None,
        broadcast the result to all ranks.

    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on mpiroot, and `None` otherwise
    """
    if mpiroot is None: mpiroot = Ellipsis

    if np.isscalar(data):
        if mpiroot == Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=mpiroot)
        if mpicomm.rank == mpiroot:
            return np.array(gathered)
        return None

    if not isinstance(data, np.ndarray):
        raise ValueError('`data` must be numpy array in gather_array')

    # need C-contiguous order
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    local_length = data.shape[0]

    # check dtypes and shapes
    shapes = mpicomm.allgather(data.shape)
    dtypes = mpicomm.allgather(data.dtype)

    # check for structured data
    if dtypes[0].char == 'V':

        # check for structured data mismatch
        names = set(dtypes[0].names)
        if any(set(dt.names) != names for dt in dtypes[1:]):
            raise ValueError('mismatch between data type fields in structured data')

        # check for 'O' data types
        if any(dtypes[0][name] == 'O' for name in dtypes[0].names):
            raise ValueError('object data types ("O") not allowed in structured data in gather_array')

        # compute the new shape for each rank
        newlength = mpicomm.allreduce(local_length)
        newshape = list(data.shape)
        newshape[0] = newlength

        # the return array
        if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = gather_array(data[name], mpiroot=mpiroot, mpicomm=mpicomm)
            if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in structured data in gather_array')

    # check for bad dtypes and bad shapes
    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape = None; bad_dtype = None

    if mpiroot is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype),root=mpiroot)

    if bad_shape:
        raise ValueError('mismatch between shape[1:] across ranks in gather_array')
    if bad_dtype:
        raise ValueError('mismatch between dtypes across ranks in gather_array')

    shape = data.shape
    dtype = data.dtype

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newlength = mpicomm.allreduce(local_length)
    newshape = list(shape)
    newshape[0] = newlength

    # the return array
    if mpiroot is Ellipsis or mpicomm.rank == mpiroot:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = mpicomm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to mpiroot
    if mpiroot is Ellipsis:
        mpicomm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        mpicomm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=mpiroot)

    dt.Free()

    return recvbuffer


@SetMPIComm
def broadcast_array(data, mpiroot=0, mpicomm=None):
    """
    Broadcast the input data array across all ranks, assuming `data` is
    initially only on `mpiroot` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        on `mpiroot`, this gives the data to split and scatter
    mpicomm : MPI communicator
        the MPI communicator
    mpiroot : int
        the rank number that initially has the data
    counts : list of int
        list of the lengths of data to send to each rank
    Returns
    -------
    recvbuffer : array_like
        the chunk of `data` that each rank gets
    """

    # check for bad input
    if mpicomm.rank == mpiroot:
        isscalar = np.isscalar(data)
    else:
        isscalar = None
    isscalar = mpicomm.bcast(isscalar, root=mpiroot)

    if isscalar:
        return mpicomm.bcast(data, root=mpiroot)

    if mpicomm.rank == mpiroot:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input, root=mpiroot)
    if bad_input:
        raise ValueError('`data` must by numpy array on mpiroot in broadcast_array')

    if mpicomm.rank == mpiroot:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=mpiroot)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in broadcast_array; please specify specific data type')

    # initialize empty data on non-mpiroot ranks
    if mpicomm.rank != mpiroot:
        np_dtype = np.dtype((dtype, shape))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape, 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # the return array
    recvbuffer = np.empty(shape, dtype=dtype, order='C')

    # the send offsets
    counts = np.ones(mpicomm.size, dtype='i', order='C')
    offsets = np.zeros_like(counts, order='C')

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=mpiroot)
    dt.Free()
    return recvbuffer



@SetMPIComm
def scatter_array(data, counts=None, mpiroot=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `mpiroot` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        on `mpiroot`, this gives the data to split and scatter
    mpicomm : MPI communicator
        the MPI communicator
    mpiroot : int
        the rank number that initially has the data
    counts : list of int
        list of the lengths of data to send to each rank

    Returns
    -------
    recvbuffer : array_like
        the chunk of `data` that each rank gets
    """
    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != mpicomm.size:
            raise ValueError('counts array has wrong length!')

    # check for bad input
    if mpicomm.rank == mpiroot:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input, root=mpiroot)
    if bad_input:
        raise ValueError('`data` must by numpy array on mpiroot in scatter_array')

    if mpicomm.rank == mpiroot:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=mpiroot)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in scatter_array; please specify specific data type')

    # initialize empty data on non-mpiroot ranks
    if mpicomm.rank != mpiroot:
        np_dtype = np.dtype((dtype, shape[1:]))
        data = np.empty(0, dtype=np_dtype)

    # setup the custom dtype
    duplicity = np.product(np.array(shape[1:], 'intp'))
    itemsize = duplicity * dtype.itemsize
    dt = MPI.BYTE.Create_contiguous(itemsize)
    dt.Commit()

    # compute the new shape for each rank
    newshape = list(shape)

    if counts is None:
        newlength = shape[0] // mpicomm.size
        if mpicomm.rank < shape[0] % mpicomm.size:
            newlength += 1
        newshape[0] = newlength
    else:
        if counts.sum() != shape[0]:
            raise ValueError('the sum of the `counts` array needs to be equal to data length')
        newshape[0] = counts[mpicomm.rank]

    # the return array
    recvbuffer = np.empty(newshape, dtype=dtype, order='C')

    # the send counts, if not provided
    if counts is None:
        counts = mpicomm.allgather(newlength)
        counts = np.array(counts, order='C')

    # the send offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # do the scatter
    mpicomm.Barrier()
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=mpiroot)
    dt.Free()
    return recvbuffer


@SetMPIComm
def send_array(data, dest, tag=0, mpicomm=None):
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    shape_and_dtype = (data.shape, data.dtype)
    mpicomm.send(shape_and_dtype,dest=dest,tag=tag)
    mpicomm.Send(data,dest=dest,tag=tag)


@SetMPIComm
def recv_array(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
    shape, dtype = mpicomm.recv(source=source,tag=tag)
    data = np.empty(shape, dtype=dtype)
    mpicomm.Recv(data,source=source,tag=tag)
    return data


def local_size(size, mpicomm=None):
    localsize = size // mpicomm.size
    if mpicomm.rank < size % mpicomm.size: localsize += 1
    return localsize


def _reduce_array(data, npop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    toret = npop(data,*args,axis=axis,**kwargs)
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 in axis:
        if np.isscalar(toret):
            return mpicomm.allreduce(toret,op=mpiop)
        total = np.empty_like(toret)
        mpicomm.Allreduce(toret,total,op=mpiop)
        return total
    return toret


@SetMPIComm
def size_array(data, mpicomm=None):
    return mpicomm.allreduce(data.size,op=MPI.SUM)


@SetMPIComm
def shape_array(data, mpicomm=None):
    shapes = mpicomm.allgather(data.shape)
    shape0 = sum(s[0] for s in shapes)
    return (shape0,) + shapes[0][1:]


@SetMPIComm
def sum_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.sum,MPI.SUM,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@SetMPIComm
def mean_array(data, *args, mpicomm=None, axis=-1, **kwargs):
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 not in axis:
        toret = np.mean(data,*args,axis=axis,**kwargs)
    else:
        toret = sum_array(data,*args,mpicomm=mpicomm,axis=axis,**kwargs)
        N = size_array(data,mpicomm=mpicomm)/(1. if np.isscalar(toret) else toret.size)
        toret /= N
    return toret


@SetMPIComm
def prod_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.prod,MPI.PROD,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@SetMPIComm
def min_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.min,MPI.MIN,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@SetMPIComm
def max_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_array(data,np.max,MPI.MAX,*args,mpicomm=mpicomm,axis=axis,**kwargs)


def _reduce_arg_array(data, npop, mpiargop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    arg = npop(data,*args,axis=axis,**kwargs)
    if axis is None:
        val = data[np.unravel_index(arg,data.shape)]
    else:
        val = np.take_along_axis(data,np.expand_dims(arg,axis=axis),axis=axis)[0]
    # could not find out how to do mpicomm.Allreduce([tmp,MPI.INT_INT],[total,MPI.INT_INT],op=MPI.MINLOC) for e.g. (double,int)...
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 in axis:
        if np.isscalar(arg):
            rank = mpicomm.allreduce((val,mpicomm.rank),op=mpiargop)[1]
            argmin = mpicomm.bcast(arg,root=rank)
            return arg,rank
        #raise NotImplementedError('MPI argmin/argmax with non-scalar output is not implemented.')
        total = np.empty_like(val)
        # first decide from which rank we get the solution
        mpicomm.Allreduce(val,total,op=mpiop)
        mask = val == total
        rank = np.ones_like(arg) + mpicomm.size
        rank[mask] = mpicomm.rank
        totalrank = np.empty_like(rank)
        mpicomm.Allreduce(rank,totalrank,op=MPI.MIN)
        # f.. then fill in argmin
        mask = totalrank == mpicomm.rank
        tmparg = np.zeros_like(arg)
        tmparg[mask] = arg[mask]
        #print(mpicomm.rank,arg,mask)
        totalarg = np.empty_like(tmparg)
        mpicomm.Allreduce(tmparg,totalarg,op=MPI.SUM)
        return totalarg,totalrank

    return arg,None


@SetMPIComm
def argmin_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_arg_array(data,np.argmin,MPI.MINLOC,MPI.MIN,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@SetMPIComm
def argmax_array(data, *args, mpicomm=None, axis=None, **kwargs):
    return _reduce_arg_array(data,np.argmax,MPI.MAXLOC,MPI.MAX,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@SetMPIComm
def sort_array(data, axis=-1, kind=None, mpicomm=None):
    toret = np.sort(data,axis=axis,kind=kind)
    if mpicomm.size == 1:
        return toret
    if axis is None:
        data = data.flat
        axis = 0
    else:
        axis = normalize_axis_tuple(axis,data.ndim)[0]
    if axis != 0:
        return toret

    gathered = gather_array(toret,mpiroot=0,mpicomm=mpicomm)
    toret = None
    if mpicomm.rank == 0:
        toret = np.sort(gathered,axis=axis,kind=kind)
    return scatter_array(toret,mpiroot=0,mpicomm=mpicomm)


@SetMPIComm
def quantile_array(data, q, axis=None, overwrite_input=False, interpolation='linear', keepdims=False, mpicomm=None):
    if axis is None or 0 in normalize_axis_tuple(axis,data.ndim):
        gathered = gather_array(data,mpiroot=0,mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == 0:
            toret = np.quantile(gathered,q,axis=axis,overwrite_input=overwrite_input,keepdims=keepdims)
        return broadcast_array(toret,mpiroot=0,mpicomm=mpicomm)
    return np.quantile(data,q,axis=axis,overwrite_input=overwrite_input,keepdims=keepdims)


@SetMPIComm
def dot_array(a, b, mpicomm=None):
    # scatter axis is b first axis
    if b.ndim == 1:
        return sum_array(a*b,mpicomm=mpicomm)
    if a.ndim == b.ndim == 2:
        return sum_array(np.dot(a,b)[None,...],axis=0,mpicomm=mpicomm)
    raise NotImplementedError


@SetMPIComm
def average_array(a, axis=None, weights=None, returned=False, mpicomm=None):
    # TODO: allow several axes
    if axis is None: axis = tuple(range(a.ndim))
    else: axis = normalize_axis_tuple(axis,a.ndim)
    if 0 not in axis:
        return np.average(a,axis=axis,weights=weights,returned=returned)
    axis = axis[0]

    a = np.asanyarray(a)

    if weights is None:
        avg = mean_array(a, axis=axis, mpicomm=mpicomm)
        scl = avg.dtype.type(size_array(a)/avg.size)
    else:
        wgt = np.asanyarray(weights)

        if issubclass(a.dtype.type, (np.integer, np.bool_)):
            result_dtype = np.result_type(a.dtype, wgt.dtype, 'f8')
        else:
            result_dtype = np.result_type(a.dtype, wgt.dtype)

        # Sanity checks
        if a.shape != wgt.shape:
            if axis is None:
                raise TypeError(
                    "Axis must be specified when shapes of a and weights "
                    "differ.")
            if wgt.ndim != 1:
                raise TypeError(
                    "1D weights expected when shapes of a and weights differ.")
            if wgt.shape[0] != a.shape[axis]:
                raise ValueError(
                    "Length of weights not compatible with specified axis.")

            # setup wgt to broadcast along axis
            wgt = np.broadcast_to(wgt, (a.ndim-1)*(1,) + wgt.shape)
            wgt = wgt.swapaxes(-1, axis)

        scl = sum_array(wgt, axis=axis, dtype=result_dtype)
        if np.any(scl == 0.0):
            raise ZeroDivisionError(
                "Weights sum to zero, can't be normalized")

        avg = sum_array(np.multiply(a, wgt, dtype=result_dtype), axis=axis)/scl

    if returned:
        if scl.shape != avg.shape:
            scl = np.broadcast_to(scl, avg.shape).copy()
        return avg, scl
    else:
        return avg


@SetMPIComm
def var_array(a, axis=-1, fweights=None, aweights=None, ddof=0, mpicomm=None):
    X = np.array(a)
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[axis]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[axis]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average_array(X, axis=axis, weights=w, returned=True, mpicomm=mpicomm)

    # Determine the normalization
    if w is None:
        fact = shape_array(a)[axis] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum_array(w*aweights, axis=axis, mpicomm=mpicomm)/w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0

    X = np.apply_along_axis(lambda x: x-avg,axis,X)
    if w is None:
        X_T = X
    else:
        X_T = (X*w)
    c = sum_array(X*X.conj(), axis=axis, mpicomm=mpicomm)
    c *= np.true_divide(1, fact)
    return c.squeeze()


@SetMPIComm
def std_array(*args, **kwargs):
    return np.sqrt(var_array(*args,**kwargs))


@SetMPIComm
def cov_array(m, y=None, ddof=1, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    # scatter axis is data second axis
    # data (nobs, ndim)
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError(
                "fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError(
                "fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError(
                "cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError(
                "incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError(
                "aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = average_array(X.T, axis=0, weights=w, returned=True, mpicomm=mpicomm)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = shape_array(X.T)[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof*sum_array(w*aweights, mpicomm=mpicomm)/w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice",
                      RuntimeWarning, stacklevel=3)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X*w).T
    c = dot_array(X, X_T.conj(), mpicomm=mpicomm)
    c *= np.true_divide(1, fact)
    return c.squeeze()


@SetMPIComm
def corrcoef_array(x, y=None, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    c = cov_array(x, y, rowvar, fweights=None, aweights=None, dtype=dtype, mpicomm=mpicomm)
    try:
        d = np.diag(c)
    except ValueError:
        # scalar covariance
        # nan if incorrect value (nan, inf, 0), 1 otherwise
        return c / c
    stddev = np.sqrt(d.real)
    c /= stddev[:, None]
    c /= stddev[None, :]

    # Clip real and imaginary parts to [-1, 1].  This does not guarantee
    # abs(a[i,j]) <= 1 for complex arrays, but is the best we can do without
    # excessive work.
    np.clip(c.real, -1, 1, out=c.real)
    if np.iscomplexobj(c):
        np.clip(c.imag, -1, 1, out=c.imag)

    return c

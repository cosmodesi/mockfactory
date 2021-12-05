"""
MPI routines, many taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/__init__.py
and https://github.com/bccp/nbodykit/blob/master/nbodykit/batch.py.
"""
import logging
import functools
from contextlib import contextmanager

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from mpi4py import MPI


class CurrentMPIComm(object):
    """Class to faciliate getting and setting the current MPI communicator."""
    logger = logging.getLogger('CurrentMPIComm')

    _stack = [MPI.COMM_WORLD]

    @staticmethod
    def enable(func):
        """
        Decorator to attach the current MPI communicator to the input
        keyword arguments of ``func``, via the ``mpicomm`` keyword.
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            kwargs.setdefault('mpicomm', None)
            if kwargs['mpicomm'] is None:
                kwargs['mpicomm'] = CurrentMPIComm.get()
            return func(*args, **kwargs)

        return wrapper

    @classmethod
    @contextmanager
    def enter(cls, mpicomm):
        """
        Enter a context where the current default MPI communicator is modified to the
        argument `comm`. After leaving the context manager the communicator is restored.

        Example:

        .. code:: python

            with CurrentMPIComm.enter(comm):
                cat = UniformCatalog(...)

        is identical to

        .. code:: python

            cat = UniformCatalog(..., comm=comm)

        """
        cls.push(mpicomm)

        yield

        cls.pop()

    @classmethod
    def push(cls, mpicomm):
        """Switch to a new current default MPI communicator."""
        cls._stack.append(mpicomm)
        if mpicomm.rank == 0:
            cls.logger.info('Entering a current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()

    @classmethod
    def pop(cls):
        """Restore to the previous current default MPI communicator."""
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Leaving current communicator of size {:d}'.format(mpicomm.size))
        cls._stack[-1].barrier()
        cls._stack.pop()
        mpicomm = cls._stack[-1]
        if mpicomm.rank == 0:
            cls.logger.info('Restored current communicator to size {:d}'.format(mpicomm.size))

    @classmethod
    def get(cls):
        """Get the default current MPI communicator. The initial value is ``MPI.COMM_WORLD``."""
        return cls._stack[-1]


@CurrentMPIComm.enable
def gather_array(data, root=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Gather the input data array from all ranks to the specified ``root``.
    This uses `Gatherv`, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like
        The data on each rank to gather.

    root : int, Ellipsis, default=0
        The rank number to gather the data to. If root is Ellipsis or None,
        broadcast the result to all ranks.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like, None
        the gathered data on root, and `None` otherwise
    """
    if root is None: root = Ellipsis

    if np.isscalar(data):
        if root == Ellipsis:
            return np.array(mpicomm.allgather(data))
        gathered = mpicomm.gather(data, root=root)
        if mpicomm.rank == root:
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
        if root is Ellipsis or mpicomm.rank == root:
            recvbuffer = np.empty(newshape, dtype=dtypes[0], order='C')
        else:
            recvbuffer = None

        for name in dtypes[0].names:
            d = gather_array(data[name], root=root, mpicomm=mpicomm)
            if root is Ellipsis or mpicomm.rank == root:
                recvbuffer[name] = d

        return recvbuffer

    # check for 'O' data types
    if dtypes[0] == 'O':
        raise ValueError('object data types ("O") not allowed in structured data in gather_array')

    # check for bad dtypes and bad shapes
    if root is Ellipsis or mpicomm.rank == root:
        bad_shape = any(s[1:] != shapes[0][1:] for s in shapes[1:])
        bad_dtype = any(dt != dtypes[0] for dt in dtypes[1:])
    else:
        bad_shape = None; bad_dtype = None

    if root is not Ellipsis:
        bad_shape, bad_dtype = mpicomm.bcast((bad_shape, bad_dtype),root=root)

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
    if root is Ellipsis or mpicomm.rank == root:
        recvbuffer = np.empty(newshape, dtype=dtype, order='C')
    else:
        recvbuffer = None

    # the recv counts
    counts = mpicomm.allgather(local_length)
    counts = np.array(counts, order='C')

    # the recv offsets
    offsets = np.zeros_like(counts, order='C')
    offsets[1:] = counts.cumsum()[:-1]

    # gather to root
    if root is Ellipsis:
        mpicomm.Allgatherv([data, dt], [recvbuffer, (counts, offsets), dt])
    else:
        mpicomm.Gatherv([data, dt], [recvbuffer, (counts, offsets), dt], root=root)

    dt.Free()

    return recvbuffer


@CurrentMPIComm.enable
def broadcast_array(data, root=0, mpicomm=None):
    """
    Broadcast the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype.

    Parameters
    ----------
    data : array_like or None
        On `root`, this gives the data to broadcast.

    root : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        The chunk of `data` that each rank gets.
    """

    # check for bad input
    if mpicomm.rank == root:
        isscalar = np.isscalar(data)
    else:
        isscalar = None
    isscalar = mpicomm.bcast(isscalar, root=root)

    if isscalar:
        return mpicomm.bcast(data, root=root)

    if mpicomm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input,root=root)
    if bad_input:
        raise ValueError('`data` must by numpy array on root in broadcast_array')

    if mpicomm.rank == root:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=root)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in broadcast_array; please specify specific data type')

    # initialize empty data on non-root ranks
    if mpicomm.rank != root:
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
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=root)
    dt.Free()
    return recvbuffer


@CurrentMPIComm.enable
def scatter_array(data, counts=None, root=0, mpicomm=None):
    """
    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/utils.py
    Scatter the input data array across all ranks, assuming `data` is
    initially only on `root` (and `None` on other ranks).
    This uses ``Scatterv``, which avoids mpi4py pickling, and also
    avoids the 2 GB mpi4py limit for bytes using a custom datatype

    Parameters
    ----------
    data : array_like or None
        On `root`, this gives the data to split and scatter.

    counts : list of int
        List of the lengths of data to send to each rank.

    root : int, default=0
        The rank number that initially has the data.

    mpicomm : MPI communicator, default=None
        The MPI communicator.

    Returns
    -------
    recvbuffer : array_like
        The chunk of `data` that each rank gets.
    """
    if counts is not None:
        counts = np.asarray(counts, order='C')
        if len(counts) != mpicomm.size:
            raise ValueError('counts array has wrong length!')

    # check for bad input
    if mpicomm.rank == root:
        bad_input = not isinstance(data, np.ndarray)
    else:
        bad_input = None
    bad_input = mpicomm.bcast(bad_input, root=root)
    if bad_input:
        raise ValueError('`data` must by numpy array on root in scatter_array')

    if mpicomm.rank == root:
        # need C-contiguous order
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        shape_and_dtype = (data.shape, data.dtype)
    else:
        shape_and_dtype = None

    # each rank needs shape/dtype of input data
    shape, dtype = mpicomm.bcast(shape_and_dtype, root=root)

    # object dtype is not supported
    fail = False
    if dtype.char == 'V':
        fail = any(dtype[name] == 'O' for name in dtype.names)
    else:
        fail = dtype == 'O'
    if fail:
        raise ValueError('"object" data type not supported in scatter_array; please specify specific data type')

    # initialize empty data on non-root ranks
    if mpicomm.rank != root:
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
        newshape[0] = newlength = local_size(shape[0], mpicomm=mpicomm)
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
    mpicomm.Scatterv([data, (counts, offsets), dt], [recvbuffer, dt], root=root)
    dt.Free()
    return recvbuffer


@CurrentMPIComm.enable
def set_common_seed(seed=None, mpicomm=None):
    """
    Set same global :mod:`np.random` and :mod:`random` seed for all MPI processes.

    Parameters
    ----------
    seed : int, default=None
        Random seed to broadcast on all processes.
        If ``None``, draw random seed.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    Returns
    -------
    seed : int
        Seed used to initialize :mod:`np.random` and :mod:`random` global states.
    """
    if seed is None:
        if mpicomm.rank == 0:
            seed = np.random.randint(0,high=0xffffffff)
    seed = mpicomm.bcast(seed,root=0)
    np.random.seed(seed)
    random.seed(seed)
    return seed


@CurrentMPIComm.enable
def bcast_seed(seed=None, mpicomm=None, size=10000):
    """
    Generate array of seeds.

    Parameters
    ---------
    seed : int, default=None
        Random seed to use when generating seeds.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    size : int, default=10000
        Number of seeds to be generated.

    Returns
    -------
    seeds : array
        Array of seeds.
    """
    if mpicomm.rank == 0:
        seeds = np.random.RandomState(seed=seed).randint(0,high=0xffffffff,size=size)
    return broadcast_array(seeds if mpicomm.rank == 0 else None,root=0,mpicomm=mpicomm)


@CurrentMPIComm.enable
def set_independent_seed(seed=None, mpicomm=None, size=10000):
    """
    Set independent global :mod:`np.random` and :mod:`random` seed for all MPI processes.

    Parameters
    ---------
    seed : int, default=None
        Random seed to use when generating seeds.

    mpicomm : MPI communicator, default=None
        Communicator to use for broadcasting. Defaults to current communicator.

    size : int, default=10000
        Number of seeds to be generated.
        To ensure random draws are independent of the number of ranks,
        this should be larger than the total number of processes that will ever be used.

    Returns
    -------
    seed : int
        Seed used to initialize :mod:`np.random` and :mod:`random` global states.
    """
    seed = bcast_seed(seed=seed,mpicomm=mpicomm,size=size)[mpicomm.rank]
    np.random.seed(seed)
    random.seed(seed)
    return seed


@CurrentMPIComm.enable
def front_pad_array(array, front, mpicomm=None):
    """
    Pad an array in the front with items before this rank.

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
        raise ValueError('Cannot work out a plan to padd items. Some front values are too large. {:d} {:d}'.format(torecv.sum(), front))

    tosend = mpicomm.alltoall(torecv)
    sendbuf = [array[-items:] if items > 0 else array[0:0] for i, items in enumerate(tosend)]
    recvbuf = mpicomm.alltoall(sendbuf)
    return np.concatenate(list(recvbuf) + [array], axis=0)


class MPIRandomState(object):
    """
    A Random number generator that is invariant against number of ranks,
    when the total size of random number requested is kept the same.
    The algorithm here assumes the random number generator from numpy
    produces uncorrelated results when the seeds are sampled from a single
    random generator.
    The sampler methods are collective calls; multiple calls will return
    uncorrelated results.
    The result is only invariant under different ``mpicomm.size`` when ``mpicomm.allreduce(size)``
    and ``chunksize`` are kept invariant.

    Taken from https://github.com/bccp/nbodykit/blob/master/nbodykit/mpirng.py
    """
    @CurrentMPIComm.enable
    def __init__(self, size, seed=None, chunksize=100000, mpicomm=None):
        self.mpicomm = mpicomm
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
        Pad every item in args with values from previous ranks,
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
        """Produce :attr:`size` poissons, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            lam, = args
            return rng.poisson(lam=lam, size=size)
        return self._call_rngmethod(sampler, (lam,), itemshape, dtype)

    def normal(self, loc=0, scale=1, itemshape=(), dtype='f8'):
        """Produce :attr:`size` normals, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            loc, scale = args
            return rng.normal(loc=loc, scale=scale, size=size)
        return self._call_rngmethod(sampler, (loc, scale), itemshape, dtype)

    def uniform(self, low=0., high=1.0, itemshape=(), dtype='f8'):
        """Produce :attr:`size` uniforms, each of shape itemshape. This is a collective MPI call. """
        def sampler(rng, args, size):
            low, high = args
            return rng.uniform(low=low, high=high,size=size)
        return self._call_rngmethod(sampler, (low, high), itemshape, dtype)

    def _call_rngmethod(self, sampler, args, itemshape, dtype='f8'):
        """
        Loop over the seed table, and call ``sampler(rng, args, size)``
        on each rng, with matched input args and size.
        the args are padded in the front such that the rng is invariant
        no matter how :attr:`size` is distributed.
        Truncate the return value at the front to match the requested :attr:`size`.
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


class MPIPool(object):
    """
    A processing pool that distributes tasks using MPI.
    With this pool class, the master process distributes tasks to worker
    processes using an MPI communicator.

    This implementation is inspired by @juliohm in `this module
    <https://github.com/juliohm/HUM/blob/master/pyhum/utils.py#L24>`_
    and was adapted from schwimmbad.
    """
    logger = logging.getLogger('MPIPool')

    @CurrentMPIComm.enable
    def __init__(self, mpicomm=None, check_tasks=False):
        """
        initialize :class:`MPIPool`.

        Parameters
        ----------
        mpicomm : MPI communicator, default=None
            Communicator. Defaults to current communicator.

        check_tasks : bool, default=False
            Check that same tasks are input for all processes,
            if not, raises a :class:`ValueError`.
        """
        self.mpicomm = mpicomm

        self.master = 0
        self.rank = self.mpicomm.Get_rank()

        #atexit.register(lambda: MPIPool.close(self))

        #if not self.is_master():
        #    # workers branch here and wait for work
        #    self.wait()
        #    sys.exit(0)

        self.workers = set(range(self.mpicomm.size))
        self.workers.discard(self.master)
        self.size = self.mpicomm.Get_size() - 1
        self.check_tasks = check_tasks

        if self.size == 0:
            raise ValueError('Tried to create an MPI pool, but there '
                             'was only one MPI process available. '
                             'Need at least two.')

    def wait(self):
        """
        Tell the workers to wait and listen for the master process. This is
        called automatically when using :meth:`MPIPool.map` and doesn't need to
        be called by the user.
        """
        if self.is_master():
            return

        status = MPI.Status()
        while True:
            task = self.mpicomm.recv(source=self.master, tag=MPI.ANY_TAG, status=status)

            if task is None:
                # Worker told to quit work
                break

            result = self.function(task)
            # Worker is sending answer with tag
            self.mpicomm.ssend(result, self.master, status.tag)

    def map(self, function, tasks):
        """
        Evaluate a function or callable on each task in parallel using MPI.
        The callable, ``worker``, is called on each element of the ``tasks``
        iterable. The results are returned in the expected order.

        Parameters
        ----------
        function : callable
            A function or callable object that is executed on each element of
            the specified ``tasks`` iterable. This should accept a single positional
            argument and return a single object.

        tasks : iterable
            A list or iterable of tasks. Each task can be itself an iterable
            (e.g., tuple) of values or data to pass in to the worker function.

        Returns
        -------
        results : list
            A list of results from the output of each ``function`` call.
        """

        # If not the master just wait for instructions.
        self.function = function
        #if not self.is_master():
        #    self.wait()
        #    return
        results = None
        tasks = list(tasks)

        # check
        if self.check_tasks:
            alltasks = self.mpicomm.allgather(tasks)
            tasks = np.array(alltasks[self.master])
            for t in alltasks:
                if t is not None and not np.all(np.array(t) == tasks):
                    raise ValueError('Something fishy: not the same input tasks on all ranks')

        if self.is_master():

            workerset = self.workers.copy()
            tasklist = [(tid, arg) for tid, arg in enumerate(tasks)]
            pending = len(tasklist)
            results = [None]*len(tasklist)

            while pending:
                if workerset and tasklist:
                    worker = workerset.pop()
                    taskid, task = tasklist.pop()
                    # "Sent task %s to worker %s with tag %s"
                    self.mpicomm.send(task, dest=worker, tag=taskid)

                if tasklist:
                    flag = self.mpicomm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                    if not flag:
                        continue
                else:
                    self.mpicomm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

                status = MPI.Status()
                result = self.mpicomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                        status=status)
                worker = status.source
                taskid = status.tag

                # "Master received from worker %s with tag %s"

                workerset.add(worker)
                results[taskid] = result
                pending -= 1
            self.close()
        else:
            self.wait()

        self.mpicomm.Barrier()
        return self.mpicomm.bcast(results,root=self.master)

    def close(self):
        """Tell all the workers to quit."""
        if self.is_worker():
            return

        for worker in self.workers:
            self.mpicomm.send(None, worker, 0)

    def is_master(self):
        """
        Is the current process the master process?
        Master is responsible for distributing the tasks to the other available ranks.
        """
        return self.rank == self.master

    def is_worker(self):
        """
        Is the current process a valid worker?
        Workers wait for instructions from the master.
        """
        return self.rank != self.master

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit gracefully by closing and freeing the MPI-related variables."""

        if exc_value is not None:
            from . import utils
            utils.exception_handler(exc_type, exc_value, exc_traceback)
        # wait and exit
        self.logger.debug('Rank {:d} process finished'.format(self.rank))
        self.mpicomm.Barrier()

        if self.is_root():
            self.logger.debug('Master is finished; terminating')

        self.close()


@CurrentMPIComm.enable
def send_array(data, dest, tag=0, mpicomm=None):
    """
    Send input array ``data`` to process ``dest``.

    Parameters
    ----------
    data : array
        Array to send.

    dest : int
        Rank of process to send array to.

    tag : int, default=0
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.
    """
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)
    shape_and_dtype = (data.shape, data.dtype)
    mpicomm.send(shape_and_dtype,dest=dest,tag=tag)
    mpicomm.Send(data,dest=dest,tag=tag)


@CurrentMPIComm.enable
def recv_array(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, mpicomm=None):
    """
    Receive array from process ``source``.

    Parameters
    ----------
    source : int, default=MPI.ANY_SOURCE
        Rank of process to receive array from.

    tag : int, default=0
        Message identifier.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    data : array
    """
    shape, dtype = mpicomm.recv(source=source,tag=tag)
    data = np.empty(shape, dtype=dtype)
    mpicomm.Recv(data,source=source,tag=tag)
    return data


@CurrentMPIComm.enable
def local_size(size, mpicomm=None):
    """
    Divide global ``size`` into local (process) size.

    Parameters
    ----------
    size : int
        Global size.

    mpicomm : MPI communicator, default=None
        Communicator. Defaults to current communicator.

    Returns
    -------
    localsize : int
        Local size. Sum of local sizes over all processes equals global size.
    """
    start = mpicomm.rank * size // mpicomm.size
    stop = (mpicomm.rank + 1) * size // mpicomm.size
    localsize = stop - start
    #localsize = size // mpicomm.size
    #if mpicomm.rank < size % mpicomm.size: localsize += 1
    return localsize


def _reduce_array(data, npop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    """
    Apply operation ``npop`` on input array ``data`` and reduce the result
    with MPI operation ``mpiop``(e.g. sum).

    Parameters
    ----------
    data : array
        Input array to reduce with operations ``npop`` and ``mpiop``.

    npop : callable
        Function that takes ``data``, ``args``, ``axis`` and ``kwargs`` as argument,
        and keyword arguments and return (array) value.

    mpiop : MPI operation
        MPI operation to apply on ``npop`` result.

    mpicomm : MPI communicator
        Communicator. Defaults to current communicator.

    axis : int, list, default=None
        Array axis (axes) on which to apply operations.
        If ``0`` not in ``axis``, ``mpiop`` is not used.
        Defaults to all axes.

    Returns
    -------
    toret : scalar, array
        Result of reduce operations ``npop`` and ``mpiop``.
        If ``0`` in ``axis``, result is broadcast on all ranks.
        Else, result is local.
    """
    toret = npop(data,*args,axis=axis,**kwargs)
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 in axis:
        if np.ndim(toret) == 0:
            return mpicomm.allreduce(toret,op=mpiop)
        total = np.empty_like(toret)
        mpicomm.Allreduce(toret,total,op=mpiop)
        return total
    return toret


@CurrentMPIComm.enable
def size_array(data, mpicomm=None):
    """Return global size of ``data`` array."""
    return mpicomm.allreduce(data.size,op=MPI.SUM)


@CurrentMPIComm.enable
def shape_array(data, mpicomm=None):
    """Return global shape of ``data`` array (scattered along the first dimension)."""
    shapes = mpicomm.allgather(data.shape)
    shape0 = sum(s[0] for s in shapes)
    return (shape0,) + shapes[0][1:]


@CurrentMPIComm.enable
def sum_array(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return sum of input array ``data`` along ``axis``."""
    return _reduce_array(data,np.sum,MPI.SUM,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def mean_array(data, *args, mpicomm=None, axis=-1, **kwargs):
    """Return mean of array ``data`` along ``axis``."""
    if axis is None: axis = tuple(range(data.ndim))
    else: axis = normalize_axis_tuple(axis,data.ndim)
    if 0 not in axis:
        toret = np.mean(data,*args,axis=axis,**kwargs)
    else:
        toret = sum_array(data,*args,mpicomm=mpicomm,axis=axis,**kwargs)
        N = size_array(data,mpicomm=mpicomm)/(1. if np.isscalar(toret) else toret.size)
        toret /= N
    return toret


@CurrentMPIComm.enable
def prod_array(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return product of input array ``data`` along ``axis``."""
    return _reduce_array(data,np.prod,MPI.PROD,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def min_array(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return minimum of input array ``data`` along ``axis``."""
    return _reduce_array(data,np.min,MPI.MIN,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def max_array(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return maximum of input array ``data`` along ``axis``."""
    return _reduce_array(data,np.max,MPI.MAX,*args,mpicomm=mpicomm,axis=axis,**kwargs)


def _reduce_arg_array(data, npop, mpiargop, mpiop, *args, mpicomm=None, axis=None, **kwargs):
    """
    Apply operation ``npop`` on input array ``data`` and reduce the result
    with MPI operation ``mpiargop``.
    Contrary to :func:`_reduce_array`, ``npop`` is expected to return index in array.
    (e.g. index of minimum).

    Parameters
    ----------
    data : array
        Input array to reduce with operations ``npop`` and ``mpiop``.

    npop : callable
        Function that takes ``data``, ``args``, ``axis`` and ``kwargs`` as argument,
        and keyword arguments, and returns array index.

    mpiargop : MPI operation
        MPI operation to select index returned by ``npop`` among all processes
        (takes as input ``(value,rank)`` with ``value`` array value at index returned by ``npop``).

    mpiop : MPI operation
        MPI operation to apply on array value at index returned by ``npop``.

    mpicomm : MPI communicator
        Communicator. Defaults to current communicator.

    axis : int, list, default=None
        Array axis (axes) on which to apply operations.
        If ``0`` not in ``axis``, ``mpiop`` is not used.
        Defaults to all axes.

    Returns
    -------
    arg : scalar, array
        If ``0`` in ``axis``, index in global array; result is broadcast on all ranks.
        Else, result is local.

    rank : int, None
        If ``0`` in ``axis``, rank where index resides in.
        Else, ``None``.
    """
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
            arg = mpicomm.bcast(arg,root=rank)
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


@CurrentMPIComm.enable
def argmin_array(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return index of minimum in input array ``data`` along ``axis``."""
    return _reduce_arg_array(data,np.argmin,MPI.MINLOC,MPI.MIN,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def argmax_array(data, *args, mpicomm=None, axis=None, **kwargs):
    """Return index of maximum in input array ``data`` along ``axis``."""
    return _reduce_arg_array(data,np.argmax,MPI.MAXLOC,MPI.MAX,*args,mpicomm=mpicomm,axis=axis,**kwargs)


@CurrentMPIComm.enable
def sort_array(data, axis=-1, kind=None, mpicomm=None):
    """
    Sort input array ``data`` along ``axis``.
    Naive implementation: array is gathered, sorted, and scattered again.
    Faster than naive distributed sorts (bitonic, transposition)...

    Parameters
    ----------
    data : array
        Array to be sorted.

    axis : int, default=-1
        Sorting axis.

    kind : string, default=None
        Sorting algorithm. The default is ‘quicksort’.
        See :func:`numpy.sort`.

    mpicomm : MPI communicator
        Communicator. Defaults to current communicator.

    Returns
    -------
    toret : array
        Sorted array (scattered).
    """
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

    gathered = gather_array(toret,root=0,mpicomm=mpicomm)
    toret = None
    if mpicomm.rank == 0:
        toret = np.sort(gathered,axis=axis,kind=kind)
    return scatter_array(toret,root=0,mpicomm=mpicomm)


@CurrentMPIComm.enable
def quantile_array(data, q, axis=None, overwrite_input=False, interpolation='linear', keepdims=False, mpicomm=None):
    """
    Return array quantiles. See :func:`numpy.quantile`.
    Naive implementation: array is gathered before taking quantile.
    """
    if axis is None or 0 in normalize_axis_tuple(axis,data.ndim):
        gathered = gather_array(data,root=0,mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == 0:
            toret = np.quantile(gathered,q,axis=axis,overwrite_input=overwrite_input,keepdims=keepdims)
        return broadcast_array(toret,root=0,mpicomm=mpicomm)
    return np.quantile(data,q,axis=axis,overwrite_input=overwrite_input,keepdims=keepdims)


@CurrentMPIComm.enable
def weighted_quantile_array(data, q, weights=None, axis=None, interpolation='linear', mpicomm=None):
    """
    Return weighted array quantiles. See :func:`utils.weighted_quantile`.
    Naive implementation: array is gathered before taking quantile.
    """
    if axis is None or 0 in normalize_axis_tuple(axis,data.ndim):
        gathered = gather_array(data,root=0,mpicomm=mpicomm)
        isnoneweights = all(mpicomm.allgather(weights is None))
        if not isnoneweights: weights = gather_array(weights,root=0,mpicomm=mpicomm)
        toret = None
        if mpicomm.rank == 0:
            toret = utils.weighted_quantile(gathered,q,weights=weights,axis=axis,interpolation=interpolation)
        return broadcast_array(toret,root=0,mpicomm=mpicomm)
    return utils.weighted_quantile(data,q,weights=weights,axis=axis,interpolation=interpolation)


@CurrentMPIComm.enable
def dot_array(a, b, mpicomm=None):
    """
    Return dot product of input arrays ``a`` and ``b``.
    Currently accepts one-dimensional ``b`` or two-dimensional ``a`` and ``b``.
    ``b`` must be scattered along first axis, hence ``a`` scattered along last axis.
    """
    # scatter axis is b first axis
    if b.ndim == 1:
        return sum_array(a*b,mpicomm=mpicomm)
    if a.ndim == b.ndim == 2:
        return sum_array(np.dot(a,b)[None,...],axis=0,mpicomm=mpicomm)
    raise NotImplementedError


@CurrentMPIComm.enable
def average_array(a, axis=None, weights=None, returned=False, mpicomm=None):
    """
    Return weighted average of input array ``a`` along axis ``axis``.
    See :func:`numpy.average`.
    TODO: allow several axes.
    """
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


@CurrentMPIComm.enable
def var_array(a, axis=-1, fweights=None, aweights=None, ddof=1, mpicomm=None):
    """
    Estimate variance, given data and weights.
    See :func:`numpy.var`.
    TODO: allow several axes.

    Parameters
    ----------
    a : array
        Array containing numbers whose variance is desired.
        If a is not an array, a conversion is attempted.

    axis : int, default=-1
        Axis along which the variance is computed.

    fweights : array, int, default=None
        1D array of integer frequency weights; the number of times each
        observation vector should be repeated.

    aweights : array, default=None
        1D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    ddof : int, default=1
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        `fweights` and `aweights` are specified, and ``ddof=0`` will return
        the simple average.

    mpicomm : MPI communicator
        Current MPI communicator.

    Returns
    -------
    out : array
        The variance of the variables.
    """
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


@CurrentMPIComm.enable
def std_array(*args, **kwargs):
    """
    Return weighted standard deviation of input array along axis ``axis``.
    Simply take square root of :func:`var_array` result.
    TODO: allow for several axes.
    """
    return np.sqrt(var_array(*args,**kwargs))


@CurrentMPIComm.enable
def cov_array(m, y=None, ddof=1, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    """
    Estimate a covariance matrix, given data and weights.
    See :func:`numpy.cov`.

    Parameters
    ----------
    m : array
        A 1D or 2D array containing multiple variables and observations.
        Each row of ``m`` represents a variable, and each column a single
        observation of all those variables. Also see ``rowvar`` below.

    y : array, default=None
        An additional set of variables and observations. ``y`` has the same form
        as that of ``m``.

    rowvar : bool, default=True
        If ``rowvar`` is ``True`` (default), then each row represents a
        variable, with observations in the columns. Otherwise, the relationship
        is transposed: each column represents a variable, while the rows
        contain observations.

    fweights : array, int, default=None
        1D array of integer frequency weights; the number of times each
        observation vector should be repeated.

    aweights : array, default=None
        1D array of observation vector weights. These relative weights are
        typically large for observations considered "important" and smaller for
        observations considered less "important". If ``ddof=0`` the array of
        weights can be used to assign probabilities to observation vectors.

    ddof : int, default=1
        Number of degrees of freedom.
        Note that ``ddof=1`` will return the unbiased estimate, even if both
        ``fweights`` and `aweights` are specified, and ``ddof=0`` will return
        the simple average.

    dtype : data-type, default=None
        Data-type of the result. By default, the return data-type will have
        at least ``numpy.float64`` precision.

    mpicomm : MPI communicator
        Current MPI communicator.

    Returns
    -------
    out : array
        The covariance matrix of the variables.
    """
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


@CurrentMPIComm.enable
def corrcoef_array(x, y=None, rowvar=True, fweights=None, aweights=None, dtype=None, mpicomm=None):
    """
    Return weighted correlation matrix of input arrays ``m`` (``y``).
    See :func:`cov_array`.
    """
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

import os
import time
import tempfile

import numpy as np

from mockfactory import Catalog, mpi, setup_logging
from mockfactory.catalog import Slice, MPIScatteredSource


def test_slice():
    sl = Slice(0, None, size=100)
    assert np.allclose(sl.to_array(), np.arange(100))
    sl_array = Slice(sl.to_array())
    assert sl and sl_array
    assert not sl.is_array
    assert sl_array.is_array
    #assert sl_array.to_slices() == sl.to_slices() == [slice(0, 100, 1)]
    sl1, sl2 = sl.split(2)
    assert sl1.idx == slice(0, 50, 1) and sl2.idx == slice(50, 100, 1)
    sl1, sl2 = sl_array.split(2)
    assert np.allclose(sl1.idx, np.arange(50)) and np.allclose(sl2.idx, 50 + np.arange(50))

    assert sl.find(slice(10, 120, 2)).idx == slice(10, 99, 2)
    assert Slice(2, 81, 4).find(slice(10, 120, 2)).idx == slice(2, 20, 1)
    assert np.allclose(sl_array.find(slice(10, 120, 2)).idx, np.arange(10, 100, 2))
    assert np.allclose(sl.find([0, 1, 1, 1, 2, 3]).idx, [0, 1, 1, 1, 2, 3])
    assert np.allclose(Slice(2, None, size=100).find([0, 1, 1, 1, 2, 3]).idx, [0, 1])

    assert sl.slice(slice(10, 120, 2)).idx == slice(10, 99, 2)
    assert sl.slice(slice(10, None, -2)).idx == slice(10, None, -2)
    assert sl.slice(slice(120, 2, -2)).idx == slice(98, 3, -2)
    assert np.allclose(sl_array.slice(slice(10, 120, 2)).idx, np.arange(10, 100, 2))
    assert sl.shift(20).idx == slice(20, 120, 1)
    assert np.allclose(sl_array.shift(20).idx, 20 + np.arange(100))
    sl_array = Slice(np.array([0, 1, 2, 3, 5, 6, 8, 10, 12]))
    ref = [(0, 4, 1), (5, 6, 1), (6, 7, 1), (8, 13, 2)]
    for isl, sl in enumerate(sl_array.to_slices()):
        assert sl == slice(*ref[isl])


def test_scattered_source():

    mpicomm = mpi.COMM_WORLD

    carray = np.arange(100)
    sl = slice(mpicomm.rank * carray.size // mpicomm.size, (mpicomm.rank + 1) * carray.size // mpicomm.size, 1)
    local_array = carray[sl]
    source = MPIScatteredSource(sl, csize=carray.size, mpicomm=mpicomm)
    assert np.allclose(source.get(local_array), carray[sl])
    assert np.allclose(source.get(local_array, slice(20, 400)), carray[20:400])
    assert np.allclose(source.get(local_array, slice(400, 20, -1)), carray[400:20:-1])


def test_mpi():
    #array = np.arange(100)
    array = np.ones(100, dtype='f4, f8, i4')
    array[:] = np.arange(array.size)
    mpicomm = mpi.COMM_WORLD
    if mpicomm.rank == 0:
        mpi.send_array(array, dest=1, tag=43, mpicomm=mpicomm)
    if mpicomm.rank == 1:
        array2 = mpi.recv_array(source=0, tag=43, mpicomm=mpicomm)
        for name in array.dtype.names:
            assert np.allclose(array2[name], array[name])


def test_misc():

    csize = int(1e7)
    size = mpi.local_size(csize)
    rng = mpi.MPIRandomState(size, seed=42)
    local_slice = slice(rng.mpicomm.rank * csize // rng.mpicomm.size, (rng.mpicomm.rank + 1) * csize // rng.mpicomm.size)
    ref = Catalog(data={'RA': np.arange(csize)[local_slice]}) #, 'DEC': rng.uniform(0., 1.), 'Z': rng.uniform(0., 1.), 'Position': rng.uniform(0., 1., itemshape=3)})
    assert ref.csize == csize
    test = Catalog.concatenate(ref, ref, keep_order=False)
    assert test.csize == ref.csize * 2
    test = Catalog.concatenate(test, test)
    assert test.csize == ref.csize * 4
    assert np.allclose(test['RA'], np.concatenate([ref['RA']] * 4))
    test = ref.cslice(csize//2, None, -1)
    assert test.csize == csize//2 + 1
    assert np.allclose(test.cget('RA'), ref.cget('RA')[csize//2::-1])


def test_save():

    csize = 10
    size = mpi.local_size(csize)
    rng = mpi.MPIRandomState(size, seed=42)
    ref = Catalog(data={'RA': rng.uniform(0., 1.), 'DEC': rng.uniform(0., 1.), 'Z': rng.uniform(0., 1.), 'Position': rng.uniform(0., 1., itemshape=3)})
    mpicomm = ref.mpicomm
    assert ref.csize == csize

    for ext in ['fits', 'hdf5', 'npy', 'bigfile'][3:]:

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = '_tests'
            fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.{}'.format(ext)), root=0)
            ref.write(fn)
            for ii in range(1):
                test = Catalog.read(fn)
                assert set(test.columns()) == set(ref.columns())
                assert np.all(test['Position'] == ref['Position'])
            test['Position'] += 10
            assert np.allclose(test['Position'], ref['Position'] + 10)
            fns = [mpicomm.bcast(os.path.join(tmp_dir, 'tmp{:d}.{}'.format(i, ext)), root=0) for i in range(4)]
            ref.write(fns)
            test = Catalog.read(fns)
            assert np.all(test['Position'] == ref['Position'])


class MemoryMonitor(object):
    """
    Class that monitors memory usage and clock, useful to check for memory leaks.

    >>> with MemoryMonitor() as mem:
            '''do something'''
            mem()
            '''do something else'''
    """
    def __init__(self, pid=None):
        """
        Initalize :class:`MemoryMonitor` and register current memory usage.

        Parameters
        ----------
        pid : int, default=None
            Process identifier. If ``None``, use the identifier of the current process.
        """
        import psutil
        self.proc = psutil.Process(os.getpid() if pid is None else pid)
        self.mem = self.proc.memory_info().rss / 1e6
        self.time = time.time()
        msg = 'using {:.3f} [Mb]'.format(self.mem)
        print(msg, flush=True)

    def __enter__(self):
        """Enter context."""
        return self

    def __call__(self, log=None):
        """Update memory usage."""
        mem = self.proc.memory_info().rss / 1e6
        t = time.time()
        msg = 'using {:.3f} [Mb] (increase of {:.3f} [Mb]) after {:.3f} [s]'.format(mem, mem - self.mem, t - self.time)
        if log:
            msg = '[{}] {}'.format(log, msg)
        print(msg, flush=True)
        self.mem = mem
        self.time = t

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Exit context."""
        self()


def test_memory():

    with MemoryMonitor() as mem:

        size = mpi.local_size(int(1e7))
        rng = mpi.MPIRandomState(size, seed=42)
        catalog = Catalog(data={'Position': rng.uniform(0., 1., itemshape=3)})
        catalog['Position2'] = catalog['Position'].copy()
        mem('randoms')
        fn = os.path.join('_tests', 'tmp.fits')
        catalog.save_fits(fn)
        catalog.mpicomm.Barrier()
        mem('save')
        del catalog
        mem('free')
        catalog = Catalog.load_fits(fn)
        mem('load')
        catalog['Position']
        catalog['Position2']
        mem('load2')


if __name__ == '__main__':

    setup_logging()

    #test_slice()
    #test_scattered_source()
    #test_mpi()
    #test_misc()
    test_save()
    #test_memory()

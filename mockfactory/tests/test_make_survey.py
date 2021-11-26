import os
import tempfile
import numpy as np

from mockfactory.remap import Cuboid
from mockfactory.make_survey import (RandomBoxCatalog, RandomCutskyCatalog, ParticleCatalog, CutskyCatalog,
                                    EuclideanIsometry, DistanceToRedshift,
                                    rotation_matrix_from_vectors, cutsky_to_box, box_to_cutsky,
                                    MaskCollection, UniformRadialMask, TabulatedRadialMask, UniformAngularMask, HealpixAngularMask)

from mockfactory import utils, setup_logging


def test_remap():

    u = ((0, 1, 1), (1, 0, 1), (0, 1, 0))
    rng = np.random.RandomState(seed=42)

    size = 4
    position = np.array([rng.uniform(0., 1., size) for i in range(3)]).T
    cuboid = Cuboid(*u)
    test_transform = cuboid.transform(position)
    #print(position.tolist())
    ref = np.asarray([[1.242481120911554, 0.07927226867048176, 0.5366145978426384], [0.6109877044894946, 1.0016399037833623, 0.2301527456400675],
                     [0.05562675154380102, 0.5823615999835122, 0.4442670251479918], [1.298308859982872, 0.5311517137862527, 0.2857449536972063]])
    assert np.allclose(test_transform, ref)
    inverse = cuboid.inverse_transform(test_transform)
    assert np.allclose(position, inverse)

    maxint = 1
    boxsize = 3.14
    test_lattice = Cuboid.generate_lattice_vectors(maxint=maxint, maxcomb=1, sort=False, boxsize=boxsize)
    for v in test_lattice:
        assert np.allclose(np.prod(v), boxsize**3)
    test_lattice = Cuboid.generate_lattice_vectors(maxint=maxint, maxcomb=1, sort=True)
    assert len(test_lattice) == 7

    try:
        from cuboid_remap.remap import generate_lattice_vectors as ref_generate_lattice_vectors
        from cuboid_remap.cuboid import Cuboid as CuboidRef
        HAVE_REF = True
    except ImportError:
        HAVE_REF = False

    if HAVE_REF:
        ref = ref_generate_lattice_vectors(maxint)
        assert len(test_lattice) == len(ref)
        for key in ref:
            assert tuple(key) in test_lattice
        cuboidref = CuboidRef(*u)
        ref = np.array([cuboidref.Transform(*pos) for pos in position])
        assert np.allclose(test_transform, ref)

    boxsize = [1., 2., 3.]
    #boxsize = [2.]*3
    size = 10000
    position = np.array([rng.uniform(0., boxsize[i], size) for i in range(3)]).T
    cuboid = Cuboid(*u, boxsize=boxsize)
    test = cuboid.transform(position)
    assert np.all((test >= 0.) & (test <= cuboid.cuboidsize))
    inverse = cuboid.inverse_transform(test)
    assert np.allclose(position, inverse)


def test_randoms():

    catalog = RandomBoxCatalog(size=1000, boxsize=10., boxcenter=3., attrs={'name':'randoms'})

    position = catalog.position
    name = catalog.attrs['name']
    assert name == 'randoms'
    new = catalog.subbox((0.1, 0.5), boxsize_unit=True)
    assert np.allclose(new.boxsize, [4.]*3, rtol=1e-7, atol=1e-7)
    assert np.allclose(new.boxcenter, [1.]*3, rtol=1e-7, atol=1e-7)
    new.attrs['name'] = 'subrandoms'
    assert catalog.attrs['name'] == name
    new['Position'] -= 1.
    assert np.all(catalog.position == position)
    new['Position'] += 1.
    assert np.all((new.position >= -1.) & (new.position <= 3.))
    u = ((0, 1, 1), (1, 0, 1), (0, 1, 0))
    ref = catalog.remap(*u)
    test = catalog.remap(Cuboid(*u, boxsize=catalog.boxsize))
    for col in ref:
        assert np.allclose(test[col], ref[col])
    assert np.all(test.position >= test.boxcenter - test.boxsize/2.) & np.all(test.position <= test.boxcenter + test.boxsize/2.)

    rarange, decrange = [0., 30.], [-10., 10.]
    catalog = RandomCutskyCatalog(size=1000, rarange=rarange, decrange=decrange)
    assert np.all((catalog['RA'] >= rarange[0]) & (catalog['RA'] <= rarange[1]))
    assert np.all((catalog['DEC'] >= decrange[0]) & (catalog['DEC'] <= decrange[1]))
    assert np.all(catalog['Distance'] == 1.)
    assert catalog.gsize == 1000

    drange = [1000.,2000.]
    catalog = RandomCutskyCatalog(size=1000, rarange=rarange, decrange=decrange, drange=drange)
    assert np.all((catalog['RA'] >= rarange[0]) & (catalog['RA'] <= rarange[1]))
    assert np.all((catalog['DEC'] >= decrange[0]) & (catalog['DEC'] <= decrange[1]))
    assert np.all((catalog['Distance'] >= drange[0]) & (catalog['Distance'] <= drange[1]))
    assert catalog.gsize == 1000


def test_isometry():

    rng = np.random.RandomState(seed=42)
    size = 10
    position = np.array([rng.uniform(0., 1., size) for i in range(3)]).T

    isometry = EuclideanIsometry()
    isometry.translation(1.)
    assert np.all(isometry.transform(position) == position + 1.)
    isometry.reset_translate()
    isometry.rotation(angle=90., axis='x', degree=True)
    rposition = isometry.transform(position)
    isometry.reset_rotation()
    assert np.allclose(rposition, position[:,[0,2,1]]*np.array([1,-1,1]),rtol=1e-7,atol=1e-7)
    isometry.rotation(angle=90., axis='y', degree=True)
    rposition = isometry.transform(position)
    isometry.reset_rotation()
    assert np.allclose(rposition, position[:,[2,1,0]]*np.array([1,1,-1]),rtol=1e-7,atol=1e-7)
    isometry.rotation(angle=90., axis='z', degree=True)
    rposition = isometry.transform(position)
    isometry.reset_rotation()
    assert np.allclose(rposition, position[:,[1,0,2]]*np.array([-1,1,1]),rtol=1e-7,atol=1e-7)
    distance = utils.distance(position)
    isometry.rotation(axis=np.random.randint(0,3), angle=np.random.uniform(0.,360.))
    rposition = isometry.transform(position)
    assert np.allclose(utils.distance(rposition), distance, rtol=1e-7, atol=1e-7)

    isometry = EuclideanIsometry()
    size = 5
    angles = rng.uniform(-10., 10., 2)
    isometry.rotation(angle=angles[0], axis='y', degree=True)
    isometry.rotation(angle=angles[1], axis='y', degree=True)
    tmp = isometry.transform(position)
    isometry.reset_rotation()
    isometry.rotation(angle=sum(angles), axis='y', degree=True)
    assert np.allclose(isometry.transform(position), tmp)


def test_cutsky():

    drange = [2200.,2300.]; rarange = [0.,50.]; decrange = [-1.,5.]
    boxsize = cutsky_to_box(drange=drange,rarange=rarange,decrange=decrange,return_isometry=False)
    drange2, rarange2, decrange2 = box_to_cutsky(boxsize=boxsize,dmax=drange[-1])
    assert np.allclose(drange2,drange,rtol=1e-7,atol=1e-7)
    assert np.allclose(abs(rarange2[1]-rarange2[0]),abs(rarange[1]-rarange[0]),rtol=1e-7,atol=1e-7)
    assert np.allclose(abs(decrange2[1]-decrange2[0]),abs(decrange[1]-decrange[0]),rtol=1e-7,atol=1e-7)
    catalog = RandomBoxCatalog(boxsize=boxsize*1.1,size=10000,boxcenter=10000.,seed=42)
    cutsky = catalog.cutsky(drange=drange,rarange=rarange,decrange=decrange)
    assert cutsky.gsize
    rarange = utils.wrap_angle(rarange, degree=True)
    if rarange[1] < rarange[0]: rarange[0] -= 360.
    assert np.all((cutsky['Distance'] >= drange[0]) & (cutsky['Distance'] <= drange[1]) & \
                  (cutsky['RA'] >= rarange[0]) & (cutsky['RA'] < rarange[1]) & \
                  (cutsky['DEC'] >= decrange[0]) & (cutsky['DEC'] <= decrange[1]))
    dist, ra, dec = utils.cartesian_to_sky(cutsky['Position'], wrap=False)
    assert np.all((dist >= drange[0]) & (dist <= drange[1]) & (ra >= rarange[0]) & (ra < rarange[1]) & (dec >= decrange[0]) & (dec <= decrange[1]))

    catalog = RandomBoxCatalog(boxsize=boxsize*2.1,size=10000,boxcenter=10000.,seed=42)
    gsize = catalog.cutsky(drange=drange,rarange=rarange,decrange=decrange,noutput=1).gsize
    cutsky = catalog.cutsky(drange=drange,rarange=rarange,decrange=decrange,noutput=None)
    assert isinstance(cutsky, list)
    assert len(cutsky) == 8
    for catalog in cutsky:
        assert abs(catalog.gsize / gsize - 1) < 3./gsize**0.5 # 3 sigmas
    catalog['Velocity'] = catalog.zeros(3, dtype='f8')
    assert np.allclose(catalog.position, catalog.rsd_position(f=1., los=None))


def test_masks():
    zrange = (0.6,1.1)

    selection = UniformRadialMask(zrange=zrange, nbar=1.)
    z = np.linspace(0.5, 1.5, 100)
    mask = selection(z)
    assert np.all(mask == (z>=zrange[0]) & (z<=zrange[1]))
    z = selection.sample(10, distance=lambda z:z)
    assert np.all((z>=zrange[0]) & (z<=zrange[1]))
    #assert np.allclose(selection.integral(), 1.)

    z = np.linspace(0.5, 1.5, 100)
    nbar = np.ones_like(z)
    selection = TabulatedRadialMask(z=z, nbar=nbar, zrange=zrange)
    mask = selection(z)
    assert mask[(z>=zrange[0]) & (z<=zrange[1])].all()

    norm = 0.5
    selection.normalize(0.5)
    assert np.allclose(selection.integral(), norm)
    def distance_self(x): return x
    ref = selection.nbar.copy()
    selection.convert_to_cosmo(distance_self, distance_self, zedges=None)
    assert np.allclose(selection.nbar, ref)

    selections = MaskCollection()
    selections[0] = UniformRadialMask(zrange=(0.6,1.1))
    selections[1] = UniformRadialMask(zrange=(0.8,1.0))
    chunk = np.array([0, 0, 1, 1])
    z = np.array([0.4, 0.7, 0.7, 0.9])
    mask = selections(chunk, z)
    assert np.allclose(mask, [False, True, False, True])

    decrange = (-10, 5)
    for rarange in [(300, 20), (10, 20)]:
        selection = UniformAngularMask(rarange=rarange, decrange=decrange)
        size = 100
        ra, dec = selection.sample(size=size)
        assert ra.size == dec.size == size
        assert np.all((dec >= decrange[0]) & (dec <= decrange[1]))
        if rarange[0] <= rarange[1]:
            assert np.all((ra >= rarange[0]) & (ra <= rarange[1]))
        else:
            assert np.all((ra >= rarange[0]) | (ra <= rarange[1]))

    try:
        import healpy
        HAVE_HEALPY = True
    except ImportError:
        HAVE_HEALPY = False

    if HAVE_HEALPY:
        from mockfactory.make_survey import HealpixAngularMask
        nbar = np.zeros(healpy.nside2npix(256), dtype='f8')
        selection = HealpixAngularMask(nbar)
        ra, dec = np.random.uniform(0.,1.,100), np.random.uniform(0.,1.,100)
        assert np.all(selection.prob(ra, dec) == 0.)


def test_redshift_array():

    from cosmoprimo.fiducial import DESI
    cosmo = DESI()
    zmax = 10.
    distance = lambda z: cosmo.comoving_radial_distance(z)
    redshift = DistanceToRedshift(distance=distance, zmax=zmax, nz=4096)
    z = np.random.uniform(0., 2., 10000)
    assert np.allclose(redshift(distance(z)), z, atol=1e-6)


def test_save():

    size = 10
    ref = RandomBoxCatalog(boxsize=100., size=size, boxcenter=10000., seed=42)
    mpicomm = ref.mpicomm
    assert ref.gsize == size

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.npy'), root=0)
        ref.save(fn)
        test = RandomBoxCatalog.load(fn)
        assert np.all(test.boxsize == ref.boxsize)
        assert np.all(test.position == ref.position)

    from nbodykit.lab import FITSCatalog
    with tempfile.TemporaryDirectory() as tmp_dir:
        #tmp_dir = 'tmp'
        fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.fits'), root=0)
        ref.save_fits(fn)
        #catalog = FITSCatalog(fn)['Position'][:10].compute()
        for ii in range(10):
            test = ParticleCatalog.load_fits(fn)
            assert np.all(test.position == ref.position)
        test['Position'] += 10
        assert np.allclose(test['Position'], ref.position + 10)

    with tempfile.TemporaryDirectory() as tmp_dir:
        #tmp_dir = 'tmp'
        fn = mpicomm.bcast(os.path.join(tmp_dir, 'tmp.hdf5'))
        ref.save_hdf5(fn)
        for ii in range(10):
            test = ParticleCatalog.load_hdf5(fn)
            assert np.all(test.position == ref.position)


def test_rotation_matrix():

    def norm(v):
        return np.sqrt(np.dot(v,v))

    a = [1819.25599061, 340.48034526, 2.1809526]
    b = [0., 0., 1.]
    rot = rotation_matrix_from_vectors(a, b)
    rot = rotation_matrix_from_vectors(a, a)
    assert np.allclose(rot, np.eye(3))


if __name__ == '__main__':

    setup_logging()
    test_remap()
    test_isometry()
    test_randoms()
    test_save()
    test_cutsky()
    test_masks()
    test_redshift_array()
    test_rotation_matrix()

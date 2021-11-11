import os
import tempfile
import numpy as np

from mockfactory.remap import Cuboid
from mockfactory.make_survey import (RandomBoxCatalog, RandomCutskyCatalog, ParticleCatalog, CutskyCatalog,
                                    EuclideanIsometry, DistanceToRedshift,
                                    TabulatedRadialMask, rotation_matrix_from_vectors,
                                    cutsky_to_box, box_to_cutsky)
from mockfactory import utils


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


def test_catalog():

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
    drange2, rarange2,decrange2 = box_to_cutsky(boxsize=boxsize,dmax=drange[-1])
    assert np.allclose(drange2,drange,rtol=1e-7,atol=1e-7)
    assert np.allclose(abs(rarange2[1]-rarange2[0]),abs(rarange[1]-rarange[0]),rtol=1e-7,atol=1e-7)
    assert np.allclose(abs(decrange2[1]-decrange2[0]),abs(decrange[1]-decrange[0]),rtol=1e-7,atol=1e-7)
    catalog = RandomBoxCatalog(boxsize=boxsize*1.1,size=10000,boxcenter=10000.,seed=42)
    cutsky = catalog.cutsky(drange=drange,rarange=rarange,decrange=decrange)
    assert cutsky.gsize
    dist, ra, dec = utils.cartesian_to_sky(cutsky['Position'], wrap=False)
    assert np.all((dist >= drange[0]) & (dist <= drange[1]) & (ra >= rarange[0]) & (ra < rarange[1]) & (dec >= decrange[0]) & (dec <= decrange[1]))

    catalog = RandomBoxCatalog(boxsize=boxsize*2.1,size=10000,boxcenter=10000.,seed=42)
    cutsky = catalog.cutsky(drange=drange,rarange=rarange,decrange=decrange,noutput=None)
    assert type(cutsky) is not CutskyCatalog
    assert len(cutsky) == 8


def test_masks():
    n = 100; zrange = [0.6,1.1]
    z = np.linspace(0.5, 1.5, n)
    nbar = np.ones(n, dtype='f8')
    selection = TabulatedRadialMask(z=z, nbar=nbar, zrange=zrange)
    mask = selection(z)
    assert mask[(z>=zrange[0]) & (z<=zrange[1])].all()
    selection.normalize(0.5)
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

    ref = RandomBoxCatalog(boxsize=100.,size=10000,boxcenter=10000.,seed=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'tmp.npy')
        ref.save(fn)
        test = RandomBoxCatalog.load(fn)
        assert np.all(test.boxsize == ref.boxsize)
        assert np.all(test.position == ref.position)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'tmp.fits')
        ref.save_fits(fn)
        test = ParticleCatalog.load_fits(fn)
        assert np.all(test.position == ref.position)

    with tempfile.TemporaryDirectory() as tmp_dir:
        fn = os.path.join(tmp_dir, 'tmp.hdf5')
        ref.save_hdf5(fn)
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

    test_remap()
    test_isometry()
    test_catalog()
    test_save()
    test_cutsky()
    test_masks()
    test_redshift_array()
    test_rotation_matrix()

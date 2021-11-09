import numpy as np

from mockfactory.remap import Cuboid
from mockfactory.make_survey import RandomBoxCatalog, RandomCutskyCatalog, EuclideanIsometry, DistanceToRedshift, TabulatedRadialMask, rotation_matrix_from_vectors
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
    assert np.allclose(new.boxcenter, [3.]*3, rtol=1e-7, atol=1e-7)
    new.attrs['name'] = 'subrandoms'
    assert catalog.attrs['name'] == name
    new['Position'] -= 1.
    assert np.all(catalog.position == position)
    new['Position'] += 1.
    assert np.all((new.position >= -1.) & (new.position <= 3.))

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
    isometry = EuclideanIsometry()
    isometry.translational(1.)
    rng = np.random.RandomState(seed=42)
    position = np.array([rng.uniform(0., 1., size) for i in range(3)]).T
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
    rposition = isometry.rotation(axis=np.random.randint(0,3), angle=np.random.uniform(0.,360.))
    assert np.allclose(utils.distance(rposition), distance, rtol=1e-7, atol=1e-7)
    """
    position = new.position
    new.rotate_about_origin_axis(axis='x',angle=90.)
    assert np.allclose(new.Position,position[:,[0,2,1]]*np.array([1,-1,1]),rtol=1e-7,atol=1e-7)
    new.reset_rotate_about_origin()
    new.rotate_about_origin_axis(axis='y',angle=90.)
    assert np.allclose(new.Position,position[:,[2,1,0]]*np.array([1,1,-1]),rtol=1e-7,atol=1e-7)
    new.reset_rotate_about_origin()
    new.rotate_about_origin_axis(axis='z',angle=90.)
    assert np.allclose(new.Position,position[:,[1,0,2]]*np.array([-1,1,1]),rtol=1e-7,atol=1e-7)
    distance = new.distance()
    new.rotate_about_origin_axis(axis=np.random.randint(0,3),angle=np.random.uniform(0.,360.))
    assert np.allclose(new.distance(),distance,rtol=1e-7,atol=1e-7)
    """

"""
def test_cutsky():

    drange = [10.,20.]; rarange = np.array([0.,50.])-20.; decrange = [-5.,5.]
    boxsize,operations = cutsky_to_box(drange=drange,rarange=rarange,decrange=decrange)
    deltara,deltadec,dmin = box_to_cutsky(boxsize=boxsize,dmax=drange[-1])
    assert np.allclose(dmin,drange[0],rtol=1e-7,atol=1e-7)
    assert np.allclose(deltara,abs(rarange[1]-rarange[0]),rtol=1e-7,atol=1e-7)
    assert np.allclose(deltadec,abs(decrange[1]-decrange[0]),rtol=1e-7,atol=1e-7)
    catalog = RandomBoxCatalog(boxsize=boxsize,size=10000,boxcenter=0.)
    catalog.recenter()
    catalog.apply_operation(*operations)
    catalog['distance'],catalog['RA'],catalog['DEC'] = catalog.cartesian_to_sky(wrap=False)
    #for col in ['distance','RA','DEC']: print(col, catalog[col].min(), catalog[col].max())
"""

def test_redshift_array():
    from astropy import cosmology
    cosmo = cosmology.wCDM(H0=0.71, Om0=0.31, Ode0=0.69, w0=-1)
    zmax = 10.
    distance = lambda z: cosmo.comoving_distance(z).value*cosmo.h
    redshift = DistanceToRedshift(distance=distance, zmax=zmax, nz=4096)
    z = np.random.uniform(0., 2., 10000)
    assert np.allclose(redshift(distance(z)), z, atol=1e-6)


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
    test_catalog()
    test_redshift_array()
    test_masks()
    test_rotation_matrix()

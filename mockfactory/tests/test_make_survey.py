from mockfactory import mpi
from mockfactory.remap import Cuboid
from mockfactory.make_survey import *


def test_mpi():
    start,stop,num = 0.,1.,50
    ref = np.linspace(start,stop,num)
    test = mpi.gather_array(mpi.linspace_array(start,stop,num),mpiroot=Ellipsis)
    assert np.allclose(test,ref)


def test_remap():
    maxint = 1
    test = Cuboid.generate_lattice_vectors(maxint=maxint,maxcomb=1)
    from cuboid_remap.remap import generate_lattice_vectors as ref_generate_lattice_vectors
    ref = ref_generate_lattice_vectors(maxint)
    assert len(test) == len(ref)
    for key in ref:
        assert tuple(key) in test

    u = ((0, 1, 1), (1, 0, 1), (0, 1, 0))
    size = 10
    position = np.array([np.random.uniform(size=size) for i in range(3)]).T
    cuboid = Cuboid(*u)
    test = cuboid.transform(position)

    from cuboid_remap.cuboid import Cuboid as CuboidRef
    cuboid = CuboidRef(*u)
    ref = np.array([cuboid.Transform(*pos) for pos in position])
    assert np.allclose(test,ref)


def test_catalog():

    catalog = RandomCatalog(BoxSize=10.,size=1000,BoxCenter=3.,attrs={'name':'randoms'})

    position = catalog.Position
    name = catalog.attrs['name']
    new = catalog.subvolume([0.,4.])
    assert np.allclose(new.BoxSize,[4.,4.,4.],rtol=1e-7,atol=1e-7)
    assert np.allclose(new._boxcenter,[2.,2.,2.],rtol=1e-7,atol=1e-7)
    new.attrs['name'] = 'subrandoms'
    assert catalog.attrs['name'] == name
    new['Position'] -= 1.
    assert (catalog.Position == position).all()
    new['Position'] += 1.
    new.recenter()
    assert ((new.Position >= -2.) & (new.Position <= 2.)).all()
    position = new.Position
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


def test_cutsky():

    drange = [10.,20.]; rarange = np.array([0.,50.])-20.; decrange = [-5.,5.]
    boxsize,operations = cutsky_to_box(drange=drange,rarange=rarange,decrange=decrange)
    deltara,deltadec,dmin = box_to_cutsky(boxsize=boxsize,dmax=drange[-1])
    assert np.allclose(dmin,drange[0],rtol=1e-7,atol=1e-7)
    assert np.allclose(deltara,abs(rarange[1]-rarange[0]),rtol=1e-7,atol=1e-7)
    assert np.allclose(deltadec,abs(decrange[1]-decrange[0]),rtol=1e-7,atol=1e-7)
    catalog = RandomCatalog(BoxSize=boxsize,size=10000,BoxCenter=0.)
    catalog.recenter()
    catalog.apply_operation(*operations)
    catalog['distance'],catalog['RA'],catalog['DEC'] = catalog.cartesian_to_sky(wrap=False)
    #for col in ['distance','RA','DEC']: print(col, catalog[col].min(), catalog[col].max())


def test_redshift_array(nz=4096):
    from astropy import cosmology
    cosmo = cosmology.wCDM(H0=0.71,Om0=0.31,Ode0=0.69,w0=-1)
    zmax = 10.
    distance = lambda z: cosmo.comoving_distance(z).value*cosmo.h
    redshift = DistanceToRedshift(distance=distance,zmax=zmax,nz=nz)
    z = np.random.uniform(0.,2.,10000)
    assert np.allclose(redshift(distance(z)),z,atol=1e-6)


def test_density():
    n = 100; zrange = [0.6,1.1]
    z = np.linspace(0.5,1.5,n)
    nbar = np.ones(n,dtype=np.float64)
    density = RedshiftDensityMask(z=z,nbar=nbar,zrange=zrange)
    mask = density(z)
    assert mask[(z>=zrange[0]) & (z<=zrange[0])].all()
    density.normalize(0.5)


def test_rotation_matrix():

    def norm(v):
        return np.sqrt(np.dot(v,v))

    a = [1819.25599061,340.48034526,2.1809526]
    b = [0.,0.,1.]
    rot = rotation_matrix_from_vectors(a,b)
    rot = rotation_matrix_from_vectors(a,a)
    assert np.allclose(rot,np.eye(3))



if __name__ == '__main__':

    #test_remap()
    test_catalog()
    test_cutsky()
    test_redshift_array()
    test_density()
    test_rotation_matrix()

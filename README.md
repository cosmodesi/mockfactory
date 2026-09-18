# mockfactory

**mockfactory** is a MPI-parallel Python toolkit to apply cut-sky geometry to box galaxy mocks.
Its main purpose is to study geometry effects on the power spectrum.

![My image](https://github.com/echaussidon/mockfactory/blob/main/remap.png)

A typical run, applying some cutsky geometry to a box mock is
(pseudo-code, for an example with all variables defined see [this notebook](https://github.com/cosmodesi/mockfactory/blob/main/nb/basic_examples.ipynb)):
```
from mockfactory import BoxCatalog, utils, setup_logging

# Start from a box mock, read with MPI from disk
data = BoxCatalog.read(data_fn, filetype='fits', position='Position', velocity='Velocity',
                       boxsize=boxsize, boxcenter=boxcenter)
# To ensure that the box is centered: data.recenter()
# A BoxCatalog can also be built directly from a (dictionary of) numpy array(s) split over ranks:
# data = BoxCatalog(data=array, columns=['Position', 'Velocity'], position='Position', velocity='Velocity',
#                   boxsize=boxsize, boxcenter=boxcenter)

# We've got data, now turn to randoms
from mockfactory.make_survey import RandomBoxCatalog
randoms = RandomBoxCatalog(nbar=10. * nbar, boxsize=boxsize)

# Apply cutsky geometry
randoms = randoms.cutsky(drange=drange, rarange=rarange, decrange=decrange)
# For data, we want to apply RSD *before* selection function
isometry, mask_radial, mask_angular = data.isometry_for_cutsky(drange=drange, rarange=rarange, decrange=decrange)
# First move data to its final position
data = data.cutsky_from_isometry(isometry, dradec=None)
# Apply RSD
data['RSDPosition'] = data.rsd_position(f=f)
data['Distance'], data['RA'], data['DEC'] = utils.cartesian_to_sky(data['RSDPosition'])
# Apply selection function
mask = mask_radial(data['Distance']) & mask_angular(data['RA'], data['DEC'])
data = data[mask]

# Distance to redshift relation
from mockfactory.make_survey import DistanceToRedshift
distance_to_redshift = DistanceToRedshift(distance=cosmo.comoving_radial_distance)
for catalog in [data, randoms]:
    catalog['Distance'], catalog['RA'], catalog['DEC'] = utils.cartesian_to_sky(catalog.position)
    catalog['Z'] = distance_to_redshift(catalog['Distance'])

# Let us apply some redshift cuts
from mockfactory.make_survey import TabulatedRadialMask
mask_radial = TabulatedRadialMask(z=z, nbar=nbar)
data = data[mask_radial(data['Z'], seed=84)]
randoms = randoms[mask_radial(randoms['Z'], seed=85)]

# Save to disk
data.write(data_fn)
randoms.write(randoms_fn)
```

One can also apply Jordan Carlson and Martin White's remapping algorithm to any periodic mock, e.g. (pseudo-code, for an example with all variables defined see [this notebook](https://github.com/cosmodesi/mockfactory/blob/main/nb/remap_examples.ipynb)):
```
# We start from a random catalog, but can be anything with a periodic box geometry
from mockfactory.make_survey import RandomBoxCatalog
randoms = RandomBoxCatalog(nbar=nbar, boxsize=boxsize)
# Let's choose the 3 lattice vectors in available ones
from mockfactory.remap import Cuboid
lattice = Cuboid.generate_lattice_vectors(maxint=1, maxcomb=1, sort=False, boxsize=catalog.boxsize)
# lattice is a dictionary of cuboidsize: [basis]
# Choose the cuboid (final) size that best suits you and:
remapped_randoms = randoms.remap(*basis)
```

Example notebooks are provided in directory nb/.
Example scripts are provided in directory desi/.

## Requirements

Strict requirements are:

  - numpy
  - scipy
  - mpi4py
  - mpytools

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/cosmodesi/mockfactory
```

### git

First:
```
git clone https://github.com/cosmodesi/mockfactory.git
```
To install the code:
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately):
```
python setup.py develop --user
```

## License

**mockfactory** is free software distributed under a BSD3 license. For details see the [LICENSE](https://github.com/cosmodesi/mockfactory/blob/main/LICENSE).

## Credits

[nbodykit](https://github.com/bccp/nbodykit) for mpi helper functions.
[cuboid_remap](https://github.com/duncandc/cuboid_remap) by Duncan Campbell, based on [Jordan Carlson and Martin White's algorithm](https://arxiv.org/abs/1003.3178).
Edmond Chaussidon for box-to-cutsky debugging and DESI cutsky script.
Antoine Rocher for DESI SV3 cutsky script and debugging.
Marie Laplace and Christophe Yèche for implementation of DESI QSO redshift smearing.
Jiaxi Yu for implementation of DESI ELG, LRG and BGS redshift smearing.

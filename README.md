# mockfactory

**mockfactory** is a MPI-parallel Python toolkit to generate Gaussian and lognormal mocks and apply cut-sky geometry to box galaxy mocks.
Its main purpose is to study geometry effects on the power spectrum.

A typical run, generating lognormal mocks and applying some cutsky geometry is
(pseudo-code, for an example with all variables defined see [this notebook](https://github.com/adematti/mockfactory/blob/main/nb/basic_examples.ipynb)):
```
from mockfactory import LagrangianLinearMock, utils, setup_logging

# First generate mock in box
# power is the callable power spectrum as a function of k
mock = LagrangianLinearMock(power, nmesh=nmesh, boxsize=size, boxcenter=boxcenter, seed=42, unitary_amplitude=False)
# This is Lagrangian bias, Eulerian bias - 1
mock.set_real_delta_field(bias=bias-1)
mock.set_analytic_selection_function(nbar=nbar)
mock.poisson_sample(seed=43)
mock.set_rsd(f=f, los=None)
data = mock.to_catalog()

# We've got data, now turn to randoms
from mockfactory.make_survey import RandomBoxCatalog
randoms = RandomBoxCatalog(nbar=4.*nbar, boxsize=size, seed=44)

# Apply cutsky geometry
data = data.cutsky(drange=drange, rarange=rarange, decrange=decrange, noutput=None)
randoms = randoms.cutsky(drange=drange, rarange=rarange, decrange=decrange)

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
data.save_fits(data_fn)
randoms.save_fits(randoms_fn)
```

Example notebooks are provided in directory nb/.

# Requirements

Strict requirements are:

  - numpy
  - scipy
  - mpi4py
  - pmesh
  - mpsort

## Installation

### pip

Simply run:
```
python -m pip install git+https://github.com/adematti/mockfactory
```

### git

First:
```
git clone https://github.com/adematti/pycorr.git
```
To install the code::
```
python setup.py install --user
```
Or in development mode (any change to Python code will take place immediately)::
```
python setup.py develop --user
```

## License

**mockfactory** is free software distributed under a GPLv3 license. For details see the [LICENSE](https://github.com/adematti/mockfactory/blob/main/LICENSE).

## Credits

[nbodykit](https://github.com/bccp/nbodykit) for recipe for [lognormal mocks](https://github.com/bccp/nbodykit/blob/master/nbodykit/source/catalog/lognormal.py),
and mpi helper functions.
[cuboid_remap](https://github.com/duncandc/cuboid_remap) by Duncan Campbell, based on [Jordan Carlson and Martin White's algorithm](https://arxiv.org/abs/1003.3178).

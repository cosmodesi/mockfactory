from mpytools import Catalog

catalog = Catalog.read(['_tests/box-before-remap{:d}'.format(i) for i in range(5)], filetype='bigfile', group='y5-dark')
#catalog = Catalog.read('_tests/box-before-remap.fits')

from matplotlib import pyplot as plt

position = catalog['Position']
plt.hist2d(position[:, 0], position[:, 1], bins=100)
plt.savefig('test.png')
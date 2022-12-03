import os
import sys
from setuptools import setup, find_packages


package_basename = 'mockfactory'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), package_basename))
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='cosmodesi',
      author_email='',
      description='pmesh-based package to produce Gaussian mocks for validation of clustering pipelines',
      license='BSD3',
      url='http://github.com/cosmodesi/mockfactory',
      install_requires=['numpy', 'scipy', 'pmesh', 'mpytools @ git+https://github.com/adematti/mpytools'],
      package_data={package_basename + '.desi': ['data/*.ecsv']},
      packages=find_packages())

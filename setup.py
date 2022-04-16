from setuptools import setup


setup(name='mockfactory',
      version='0.0.1',
      author='cosmodesi',
      author_email='',
      description='pmesh-based package to produce Gaussian mocks for validation of clustering pipelines',
      license='BSD3',
      url='http://github.com/cosmodesi/mockfactory',
      install_requires=['numpy', 'scipy', 'pmesh', 'mpytools'],
      packages=['mockfactory'])

from setuptools import setup


setup(name='mockfactory',
      version='0.0.1',
      author='Arnaud de Mattia',
      author_email='',
      description='Small nbodykit-based extension to produce Gaussian mocks for validation of clustering pipelines',
      license='GPL3',
      url='http://github.com/adematti/mockfactory',
      install_requires=['numpy', 'scipy', 'pmesh', 'mpsort'],
      packages=['mockfactory']
)

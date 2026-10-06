from ._version import __version__
from .make_survey import (EuclideanIsometry, box_to_cutsky, cutsky_to_box, DistanceToRedshift, RedshiftDensityInterpolator,
                          Catalog, CutskyCatalog, BoxCatalog, RandomBoxCatalog, RandomCutskyCatalog,
                          MaskCollection, UniformRadialMask, TabulatedRadialMask,
                          UniformAngularMask, MangleAngularMask, HealpixAngularMask,
                          TabulatedPDF2DRedshiftSmearing, RVS2DRedshiftSmearing)
from .utils import cartesian_to_sky, sky_to_cartesian, setup_logging


#: Deprecated pmesh-based mocks, imported on first access so that ``import mockfactory`` neither
#: needs pmesh nor raises their deprecation warning.
_DEPRECATED = {'EulerianLinearMock': 'eulerian_mock', 'LagrangianLinearMock': 'lagrangian_mock'}


def __getattr__(name):
    if name in _DEPRECATED:
        import importlib
        return getattr(importlib.import_module('.' + _DEPRECATED[name], __name__), name)
    raise AttributeError('module {!r} has no attribute {!r}'.format(__name__, name))

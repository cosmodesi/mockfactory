import warnings

# FutureWarning, not DeprecationWarning: the latter is hidden by default when raised from a library.
warnings.warn('mockfactory.blinding is deprecated and will be removed; catalog-level blinding now lives in '
              'desiblind, which desi-clustering calls, and the measurements themselves in jaxpower.',
              FutureWarning, stacklevel=2)

from .catalog import CutskyCatalogBlinding, get_cosmo_blind

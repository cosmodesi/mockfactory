"""
The pieces a DESI mock is built from, before any survey replay.

Where a cubic box becomes a cutsky catalog that looks like DESI: the footprint it has to fall
in, the imaging quantities of the bricks it lands on, the redshift error the spectrograph would
have made, and a single pass of fiber assignment.

:mod:`mockfactory.desi.altmtl` and :mod:`mockfactory.desi.lsscat` take it from there, replaying
the survey's own observation history and building the clustering catalogs.
"""

from .brick_pixel_quantities import get_brick_pixel_quantities
from .footprint import is_in_desi_footprint
from .redshift_smearing import TracerRedshiftSmearing
from .fiber_assignment import (build_tiles_for_fa, read_sky_targets, apply_fiber_assignment,
                               compute_completeness_weight)

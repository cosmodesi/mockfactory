"""
Large scale structure catalogs from an alternative merged target list run.

:mod:`mockfactory.desi.altmtl` replays the survey against a mock and says which targets got a
fiber. This turns that into the catalogs a clustering measurement reads: one row per target,
weighted for the ones that lost their fiber to a neighbour, cut by the imaging vetoes, and
paired with randoms carrying the same selection.

The stages follow the survey pipeline's, and each one is a function taking and returning
arrays:

.. code-block:: python

    from mockfactory.desi.lsscat import (read_assignments, combine_data, read_good_tilelocid,
                                         make_full_data, apply_veto_data, make_clustering_data)

    assignments = read_assignments(altmtl_dir, tileids, numproc=32)
    data = combine_data(potential, assignments, targets=targets)
    good = read_good_tilelocid(spec_fn)

    full = make_full_data(data, assignments, 'LRG', targets=targets, good_tilelocid=good)
    full = apply_veto_data(full, 3200, bits=get_mask_bits('LRG'), maps_north=..., maps_south=...)
    clustering = make_clustering_data(full, 'LRG')

It is a self-contained implementation of the mock arm of ``desihub/LSS``, and does not import
``LSS``. Nothing is written between stages and nothing is read back: what the survey pipeline
keeps on disk in between, about 450 GB per realization for one tracer of the dark program and
a terabyte for its three, is held in memory and passed along instead. The only reads are the
survey's own inputs and the per tile assignments, and the only writes are the catalogs asked
for.
"""

from .utils import (get_photsys, get_galactic_cap, join_left, last_of_each,
                    group_fraction, select_fields)
from .combine import (read_assignments, combine_data, count_tiles, combine_randoms, read_good_tilelocid,
                      read_random_imaging)
from .full import make_full_data, make_full_randoms, select_tracer, get_max_priority
from .veto import (apply_veto_data, apply_veto_randoms, apply_imaging_veto, apply_map_veto,
                   get_mask_bits, get_frac_tlobs, add_frac_tlobs)
from .clustering import (make_clustering_data, make_clustering_randoms, select_good_redshift,
                         compute_iip_weight, get_redshift_range,
                         get_bgs_absmag_cut)
from .nz import compute_nz, write_nz, add_nz_weights, get_fkp_p0
from .pipeline import run_tracer, read_hpmaps, write_catalog, write_catalogs

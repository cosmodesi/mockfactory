"""
The catalog pipeline, from an alternative merged target list run to clustering catalogs.

One call per tracer takes the potential assignments and the fiber assignments of a mock and
returns everything a measurement reads. The stages are the survey pipeline's and each is
available on its own; what this adds is that nothing is written between them.

The random catalogs dominate the cost and are independent of one another, so they are built in
parallel. Building is all of the cost; what is left, giving a catalog the number density and
the weight that follows from it, needs a quantity measured on the data and on the first random
catalog, so it waits until every catalog is built and then takes seconds. Measuring it on the
first random catalog is what the survey pipeline does too.
"""

import logging
import os
from pathlib import Path

import numpy as np

from .clustering import get_redshift_range, make_clustering_data, make_clustering_randoms
from .combine import combine_data, count_tiles, read_random_imaging
from .full import get_max_priority, make_full_data, make_full_randoms
from .nz import (add_nz_weights, compute_completeness_per_ntile, compute_nz, get_fkp_p0,
                 write_nz)
from .utils import as_table, get_galactic_cap
from .veto import (add_frac_tlobs, apply_veto_data, apply_veto_randoms, get_frac_tlobs,
                   get_mask_bits)


logger = logging.getLogger('lsscat.pipeline')


#: Fraction of a mock kept so that its density matches the one the real survey found, per
#: tracer and survey. The survey pipeline draws this without a seed.
SUBSAMPLE = {('LRG', 'DA2'): (0.99, None), ('ELG', 'DA2'): ([0.91, 0.7], 1.5),
             ('QSO', 'DA2'): ([0.97, 1.], 2.1), ('LRG', 'Y1'): (0.976, None),
             ('ELG', 'Y1'): ([0.69, 0.54], 1.5), ('QSO', 'Y1'): (0.66, None)}


#: Everything a random catalog needs that does not depend on which one it is. It is put here,
#: rather than passed, so that a forked worker inherits it instead of having it pickled: the
#: good locations alone are a couple of hundred megabytes and the data catalog is read by every
#: worker.
_context = {}


def read_hpmaps(hpmap_dir, tracer, nside=256):
    """Return the northern and southern observing condition maps used for ``tracer``."""
    import fitsio
    name = 'ELG_LOPnotqso' if 'ELG' in tracer else ('BGS_BRIGHT' if 'BGS' in tracer else tracer)
    return tuple(fitsio.read(Path(hpmap_dir) / '{}_mapprops_healpix_nested_nside{:d}_{}.fits'.format(name, nside, region))
        for region in ('N', 'S'))


def _get_random(randoms, i):
    """Return random catalog ``i``, reading it if what was given is a path or a callable."""
    array = randoms[i]
    if isinstance(array, str):
        import fitsio
        array = fitsio.read(array)
    elif callable(array):
        array = array(i)
    return as_table(array)


def _make_clustering_randoms(i):
    """
    Build one clustering random catalog, from the context the parent set up.

    Returns the whole catalog and its two galactic caps, or, when the parent asked for them to
    be written, only what the caller needs to know about them.
    """
    context = _context
    array = _get_random(context['randoms'], i)
    imaging = read_random_imaging(i, tracer=context['tracer']) \
        if context['random_imaging'] is None else context['random_imaging'][i]
    tiles = None if context['random_tiles'] is None else context['random_tiles'][i]
    array = make_full_randoms(array, context['tracer'], notqso=context['notqso'],
                              good_tilelocid=context['good_tilelocid'], imaging=imaging,
                              tiles=tiles)
    array = apply_veto_randoms(array, context['maxp'], bits=context['bits'],
                               maps_north=context['maps_north'],
                               maps_south=context['maps_south'])
    array = add_frac_tlobs(array, context['frac_tlobs'], missing=context['missing_frac_tlobs'],
                           data=context['full'])
    array = make_clustering_randoms(array, context['clustering'], seed=i,
                                    tracer=context['tracer'],
                                    completeness=context['completeness'])
    logger.info('random {:d}: {:d} clustering randoms'.format(i, len(array)))

    if _context.get('caps', None) is None:
        return i, array, None
    array, split, writes = _finish_random(i, array)
    if writes:
        write_catalogs(writes, numproc=1)
        # Written here, so the catalog never has to travel back to the parent.
        return i, None, None
    return i, array, split


def _finish_random(i, array):
    """
    Give one clustering random catalog its density and weight and split it by galactic cap.

    The files it should become are returned rather than written, so that the writing of every
    catalog can be done at once. Writing a fits file is not limited by the disk but by turning
    the array into what a fits file holds, which is byte order conversion under the global
    interpreter lock: threads do not help and processes do.
    """
    context = _context
    split = {}
    ngc = get_galactic_cap(array['RA'], array['DEC'])
    for cap, select in [('NGC', ngc), ('SGC', ~ngc)]:
        nz, weight_ntile, completeness_ntile = context['caps'][cap]
        split[cap] = add_nz_weights(array[select], nz, context['zmin'], context['dz'],
                                    context['p0'], weight_ntile, completeness_ntile,
                                    randoms=True, completeness=context['completeness'])
    writes = []
    if context['output_dir'] is not None:
        name = context['name']
        writes.append((Path(context['output_dir']) / '{}_{:d}_clustering.ran.h5'.format(name, i), array))
        writes += [(Path(context['output_dir']) / '{}_{}_{:d}_clustering.ran.h5'.format(name, cap, i),
                    split[cap]) for cap in split]
    return array, split, writes


def run_tracer(data, randoms, assignments, tracer, notqso=False, targets=None,
               random_imaging=None, random_tiles=None, good_tilelocid=None, hpmaps=None, survey='DA2',
               completeness='fracz', nbits=128, missing_frac_tlobs=1., seed=0, zrange=None,
               subsample=None, data_selection=None, columns=(), name=None, output_dir=None,
               numproc=1,
               numproc_randoms=None,
               keep=True, bits=None):
    """
    Run every stage for one tracer, and return its clustering catalogs.

    Parameters
    ----------
    data : array, callable
        Combined potential assignments, from :func:`~mockfactory.desi.lsscat.combine.combine_data`.
    randoms : list
        The random catalogs, each carrying the mock's ``PRIORITY``. An entry may be an array,
        the path of a file to read, or a function of the index returning one; the last two keep
        the parent from holding every catalog at once, which at eighteen of them is well over a
        hundred gigabytes.
    assignments : array
        Fibers given, from :func:`~mockfactory.desi.lsscat.combine.read_assignments`.
    tracer : str
        Target class.
    notqso : bool, default=False
        Whether to reject targets that are also quasar targets.
    targets : array, default=None
        Target catalog, for the imaging columns.
    random_imaging : list, default=None
        Imaging columns of each parent random catalog, from
        :func:`~mockfactory.desi.lsscat.combine.read_random_imaging`. Read on demand when not
        given.
    random_tiles : list, default=None
        Tile counts of each random catalog, from
        :func:`~mockfactory.desi.lsscat.combine.count_tiles`. Counted on demand when not
        given. They depend on the random catalog and on which locations gave a usable
        spectrum, and on nothing the mock decides, so several mocks sharing a set of randoms
        can count once and hand the result to each.
    good_tilelocid : array, default=None
        Locations the real survey got a usable spectrum from, from
        :func:`~mockfactory.desi.lsscat.combine.read_good_tilelocid`.
    hpmaps : tuple, default=None
        Northern and southern observing condition maps, from :func:`read_hpmaps`. The map veto
        is skipped when not given.
    completeness : str, default='fracz'
        How to weight the targets that were not observed; see
        :func:`~mockfactory.desi.lsscat.clustering.make_clustering_data`.
    missing_frac_tlobs : float, str, default=1.
        What to give a random whose set of tiles the data does not have; see
        :func:`~mockfactory.desi.lsscat.veto.add_frac_tlobs`.
    seed : int, default=0
        Seed of the subsampling draw. Each random catalog is resampled with its own index, so
        the result does not depend on how many run at once or in what order.
    zrange : tuple, default=None
        Redshift range. Defaults to the tracer's own.
    subsample : float, list, default=None
        Density matching fraction. Defaults to the survey's value for the tracer.
    columns : tuple, default=()
        Extra columns to carry into the clustering catalogs, beyond the ones a measurement
        needs. A mock's own truth, such as ``TRUEZ``, comes through here.
    data_selection : callable, default=None
        Any further cut on the sample, handed the vetoed full catalog and returning a boolean
        array. The bright galaxy variants are an absolute magnitude cut,
        ``data_selection=absmag_selection(get_bgs_absmag_cut())``; see
        :func:`~mockfactory.desi.lsscat.clustering.make_clustering_data`.
    name : str, default=None
        Name the catalogs are written under. Defaults to the tracer, with ``notqso`` appended
        where it applies; a bright galaxy variant is usually named after its cut, as in
        ``BGS_BRIGHT-21.5``.
    output_dir : str, default=None
        Where to write the catalogs. Nothing is written when not given.
    numproc : int, default=1
        Number of random catalogs to process at once. They are independent, and at eighteen
        randoms they are about nineteen twentieths of the work. Each worker holds one random
        catalog and what it builds from it, so this trades memory for time.
    numproc_randoms : int, default=None
        Workers for the random loop alone, where the memory goes. Defaults to ``numproc``.
        Each worker holds one random catalog and everything built from it, so the peak scales
        with this and not with ``numproc``: one at a time is the cheapest the stage gets, and
        it leaves the writes parallel. With four randoms of a bright mock the peak is about
        37 GB at four workers, so five mocks fit a 512 GB node; at one it is a quarter of that.
    keep : bool, default=True
        Whether to return the random catalogs as well as writing them. Setting it to False,
        with ``output_dir``, keeps the parent from accumulating them -- each worker writes its
        own and hands back nothing, which is what makes ``numproc_randoms`` the whole of the
        random stage's memory.

    Returns
    -------
    clustering_data : dict
        The whole catalog under ``'ALL'`` and one entry per galactic cap.
    clustering_randoms : dict
        The same, each holding a list over the random catalogs.

    bits : list, str, default=None
        Imaging mask bits to veto, on the data and on the randoms alike. Defaults to the
        tracer's own, from :func:`~mockfactory.desi.lsscat.veto.get_mask_bits`. A mock whose
        targets were already cut on other bits upstream has to name them here: the randoms
        never saw that cut, and a mask applied to one side only is an angular selection the
        randoms cannot describe.
    """
    maxp = get_max_priority(tracer, notqso=notqso)
    if bits is None:
        bits = get_mask_bits(tracer)
    maps_north, maps_south = hpmaps if hpmaps is not None else (None, None)
    zmin, zmax = zrange if zrange is not None else get_redshift_range(tracer)
    p0, dz = get_fkp_p0(tracer)
    if name is None:
        name = tracer + ('notqso' if notqso else '')
    zsplit = None
    if subsample is None:
        subsample, zsplit = SUBSAMPLE.get((tracer[:3], survey), (None, None))
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    logger.info('--- {}: full data ---'.format(tracer))
    if callable(data):
        data = data()
    full = make_full_data(data, assignments, tracer, targets=targets, notqso=notqso,
                          good_tilelocid=good_tilelocid)
    # The combined potential assignments have done their work, and they are the largest thing
    # here: seven gigabytes against the four the full catalog keeps. Dropped before the random
    # workers fork, so that they do not inherit it either. Passing `data` as a callable is what
    # lets it go: an array passed in stays alive in the caller's frame for the whole run.
    del data
    logger.info('--- {}: vetoes ---'.format(tracer))
    full = apply_veto_data(full, maxp, bits=bits, maps_north=maps_north, maps_south=maps_south)
    frac_tlobs = get_frac_tlobs(full)

    logger.info('--- {}: clustering data ---'.format(tracer))
    clustering = make_clustering_data(full, tracer, zmin=zmin, zmax=zmax,
                                      completeness=completeness, nbits=nbits,
                                      subsample=subsample, zsplit=zsplit, seed=seed,
                                      columns=columns, data_selection=data_selection)

    _context.clear()
    _context.update(randoms=randoms, random_imaging=random_imaging, random_tiles=random_tiles,
                    tracer=tracer,
                    notqso=notqso, good_tilelocid=good_tilelocid, maxp=maxp, bits=bits,
                    maps_north=maps_north, maps_south=maps_south, frac_tlobs=frac_tlobs,
                    missing_frac_tlobs=missing_frac_tlobs, full=full, clustering=clustering,
                    completeness=completeness, zmin=zmin, dz=dz, p0=p0, name=name,
                    output_dir=output_dir, keep=True, caps=None)

    # The first random catalog is built here, because the number density and the completeness
    # per number of tiles come from it and every other catalog needs them. The rest are then
    # built and finished inside their own worker, which never sends the catalog back: at a few
    # gigabytes apiece, returning them through a pipe is what a pool is worst at.
    logger.info('--- {}: random 0 ---'.format(tracer))
    indices = list(range(len(randoms)))
    _, first, _ = _make_clustering_randoms(0)

    logger.info('--- {}: n(z) ---'.format(tracer))
    ngc_data = get_galactic_cap(clustering['RA'], clustering['DEC'])
    ngc_first = get_galactic_cap(first['RA'], first['DEC'])
    caps, out_data = {}, {'ALL': clustering}
    for cap, select_data, select_random in [('NGC', ngc_data, ngc_first),
                                            ('SGC', ~ngc_data, ~ngc_first)]:
        sub_data, sub_random = clustering[select_data], first[select_random]
        nz, area = compute_nz(sub_data, sub_random, zmin, zmax, dz=dz,
                              completeness=completeness)
        weight_ntile, completeness_ntile = compute_completeness_per_ntile(
            sub_data, randoms=sub_random, completeness=completeness)
        caps[cap] = (nz[3], weight_ntile, completeness_ntile)
        out_data[cap] = add_nz_weights(sub_data, nz[3], zmin, dz, p0, weight_ntile,
                                       completeness_ntile, completeness=completeness)
        if output_dir is not None:
            write_nz(Path(output_dir) / '{}_{}_nz.txt'.format(name, cap), nz,
                     area=len(sub_random) / 2500., effective_area=area)
    _context.update(caps=caps, keep=keep)

    if numproc_randoms is None: numproc_randoms = numproc
    logger.info('--- {}: randoms 1 to {:d}, numproc={:d} ---'
                .format(tracer, len(randoms) - 1, numproc_randoms))
    writes, results = [], {}
    array, split, todo = _finish_random(0, first)
    results[0], writes = (array, split), writes + todo
    del first
    rest = indices[1:]
    if numproc_randoms > 1 and len(rest) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # Not multiprocessing.Pool. It quietly starts a replacement for a worker the kernel
        # kills, and then waits forever for the result that worker was carrying, which from
        # the outside is indistinguishable from slow progress. The executor raises instead.
        context = multiprocessing.get_context('fork')
        with ProcessPoolExecutor(max_workers=min(numproc_randoms, len(rest)),
                                 mp_context=context) as pool:
            futures = [pool.submit(_make_clustering_randoms, i) for i in rest]
            for future in as_completed(futures):
                i, array, split = future.result()
                results[i] = (array, split)
    else:
        for i in rest:
            results[i] = _make_clustering_randoms(i)[1:]

    # A worker that wrote its own catalog hands back nothing, so those entries stay empty.
    out_randoms = {'ALL': [results[i][0] for i in indices]}
    for cap in caps:
        out_randoms[cap] = [results[i][1][cap] if results[i][1] is not None else None
                            for i in indices]

    if output_dir is not None:
        writes.append((Path(output_dir) / '{}_clustering.dat.h5'.format(name),
                       out_data['ALL']))
        writes += [(Path(output_dir) / '{}_{}_clustering.dat.h5'.format(name, cap),
                    out_data[cap]) for cap in caps]
        write_catalogs(writes, numproc=numproc)
    if not keep:
        out_randoms = {key: [None] * len(indices) for key in out_randoms}
    _context.clear()
    return out_data, out_randoms


#: Catalogs waiting to be written, so that a forked writer inherits them rather than being
#: sent them: a clustering random catalog is a couple of gigabytes.
_writes = []


def write_catalog(array, fn):
    """
    Write one catalog, under the single ``LSS`` group the survey catalogs come as an extension.

    HDF5, not FITS: a FITS file holds its numbers the other way round from the machine, so
    writing one is mostly byte order conversion, and measured on a clustering random of nine
    million rows that is 7.8 s against 1.2 s. HDF5 keeps them native, and one dataset per
    column is what a reader that wants three of twelve columns can take advantage of.
    """
    import h5py
    array = as_table(array)
    with h5py.File(fn, 'w') as file:
        group = file.create_group('LSS')
        for name in array.colnames:
            column = array[name].value
            # hdf5 has no unicode type, so a fixed width string goes in as bytes. That is what a
            # fits file holds as well, PHOTSYS coming back from one as '1A', so a reader gets the
            # same thing either way.
            if column.dtype.kind == 'U':
                column = column.astype('S{:d}'.format(column.dtype.itemsize // 4))
            group.create_dataset(name, data=column)
    logger.info('wrote {} ({:d} rows)'.format(fn, len(array)))


def _write_one(index):
    """Write the catalog at ``index`` of the pending writes."""
    fn, array = _writes[index]
    write_catalog(array, fn)
    return index


def write_catalogs(writes, numproc=1):
    """
    Write several catalogs, in parallel.

    Writing is not limited by the disk. A fits file holds its numbers the other way round from
    the machine, so writing one is mostly byte order conversion, and that happens under the
    global interpreter lock: measured on a compute node, six threads write six catalogs in
    exactly the time one thread takes, and six processes take a fifth of it. The pool is forked
    once the catalogs exist, so a worker inherits them instead of having them sent.
    """
    global _writes
    _writes = list(writes)
    try:
        if numproc > 1 and len(_writes) > 1:
            import multiprocessing
            with multiprocessing.get_context('fork').Pool(min(numproc, len(_writes))) as pool:
                for _ in pool.imap_unordered(_write_one, range(len(_writes))):
                    pass
        else:
            for index in range(len(_writes)):
                _write_one(index)
    finally:
        _writes = []

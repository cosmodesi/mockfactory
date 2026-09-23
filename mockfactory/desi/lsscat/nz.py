"""
Radial selection: the number density of each tracer, and the weight it implies.

A clustering measurement needs to know how many objects per unit volume the survey found at
each redshift, both to normalize the pair counts and to weight each object by how much
information it carries. The density is measured from the data, over the volume the randoms say
was covered, and turned into the weight of Feldman, Kaiser and Peacock, which balances shot
noise against sample variance.

The density is corrected for completeness before the weight is formed: an object standing in
for neighbours that were never observed should not also be treated as if the volume around it
were emptier than it is.
"""

import logging
import math

import numpy as np

from .utils import as_table, set_column


logger = logging.getLogger('lsscat.nz')


#: Power spectrum amplitude each tracer's weight is tuned at, and the redshift bin width its
#: density is measured in.
FKP_P0 = {'LRG': (10000., 0.01), 'ELG': (4000., 0.01), 'QSO': (6000., 0.02), 'BGS': (7000., 0.01)}

#: Density of the imaging randoms the catalogs are drawn from, per square degree. It is what
#: turns a number of randoms into an area.
RANDOM_DENSITY = 2500.


def get_fkp_p0(tracer):
    """Return the amplitude and redshift bin width used for ``tracer``."""
    return FKP_P0.get(tracer[:3], (6000., 0.02))


def get_cosmology():
    """
    Return the fiducial cosmology the survey measures volumes in.

    The tabulated one, not the Boltzmann solve: the catalogs are built against a fixed table
    of distances, and solving the same cosmology again moves the volume of a redshift bin by
    about one part in a hundred thousand.
    """
    from cosmoprimo.fiducial import TabulatedDESI
    return TabulatedDESI()


def compute_nz(data, randoms, zmin, zmax, dz=0.01, completeness='fracz', cosmology=None):
    """
    Return the number density of the data per redshift bin.

    The area comes from the randoms, which were drawn at a known density per square degree; it
    is the sum of their tile completeness rather than their number, so that a region only
    partly observed counts for the fraction of it that was. The data is counted with the
    weights that make up for what the survey missed, but not with the weight that depends on
    the density itself.

    Parameters
    ----------
    data : array
        Clustering data catalog.
    randoms : array
        Clustering random catalog covering the same footprint.
    zmin, zmax : float
        Redshift range.
    dz : float, default=0.01
        Bin width.
    completeness : str, default='fracz'
        Weighting scheme, which decides whether the area is corrected by the tile completeness.
    cosmology : default=None
        Fiducial cosmology, for turning a solid angle and a redshift range into a volume.

    Returns
    -------
    z, zlow, zhigh, nz, counts, volume : array
        Bin centres and edges, the density, the weighted counts and the volume of each bin.
    area : float
        Effective area, in square degrees.
    """
    if cosmology is None:
        cosmology = get_cosmology()
    area = len(randoms) / RANDOM_DENSITY
    logger.info('area is {:.2f} square degrees'.format(area))
    if completeness != 'bitweights':
        # Summed exactly: np.sum rounds differently depending on how the column is laid out in
        # memory, and the area scales every density, so the last digit would otherwise depend on
        # the container rather than the numbers.
        area = math.fsum(randoms['FRAC_TLOBS_TILES']) / RANDOM_DENSITY
        logger.info('effective area is {:.2f} square degrees'.format(area))

    nbin = int((zmax - zmin) * (1. + dz / 10.) / dz)
    weights = data['WEIGHT_COMP'] * data['WEIGHT_SYS'] * data['WEIGHT_ZFAIL']
    counts, edges = np.histogram(data['Z'], bins=nbin, range=(zmin, zmax), weights=weights)
    distance = cosmology.comoving_radial_distance
    volume = area / (360.**2 / np.pi) * 4. * np.pi / 3. * np.diff(distance(edges)**3)
    return (0.5 * (edges[:-1] + edges[1:]), edges[:-1], edges[1:],
            counts / volume, counts, volume), area


def write_nz(fn, nz, area=None, effective_area=None):
    """Write the density in the plain text format the survey catalogs come with."""
    header = []
    if area is not None:
        header.append('#area is {}square degrees'.format(area))
    if effective_area is not None:
        header.append('#effective area is {}square degrees'.format(effective_area))
    header.append('#zmid zlow zhigh n(z) Nbin Vol_bin')
    np.savetxt(fn, np.column_stack(nz), fmt='%.18g', header='\n'.join(header), comments='')
    logger.info('wrote {}'.format(fn))


def compute_completeness_per_ntile(data, randoms=None, completeness='fracz'):
    """
    Return, per number of overlapping tiles, the mean completeness weight and the completeness
    it implies.

    How complete the survey is depends mostly on how many tiles covered a piece of sky, so the
    density each object sits in is corrected by the mean over the objects with the same tile
    coverage rather than object by object, which would just undo the weighting.

    Returns
    -------
    weight, completeness : array
        Indexed by the number of tiles minus one.
    """
    ntile = np.asarray(data['NTILE']) - 1
    # Both means are indexed by the number of tiles, and the randoms reach coverages the data
    # does not, so the two have to be built to a common length or they will not line up.
    size = int(ntile.max()) + 1
    if randoms is not None:
        size = max(size, int(np.asarray(randoms['NTILE']).max()))

    def _mean(index, values):
        counts = np.bincount(index, minlength=size)
        total = np.bincount(index, weights=values, minlength=size)
        # A number of tiles that nothing has is never looked up; leave it at one rather than
        # dividing by zero.
        return np.divide(total, counts, out=np.ones(len(counts)), where=counts > 0)

    weight = _mean(ntile, np.asarray(data['WEIGHT_COMP'], dtype='f8'))
    toret = 1. / weight
    if completeness != 'bitweights':
        # The data carries only the fiber location completeness, the tile completeness having
        # been put on the randoms; the density has to be corrected by both.
        toret = toret * _mean(np.asarray(randoms['NTILE']) - 1,
                              np.asarray(randoms['FRAC_TLOBS_TILES'], dtype='f8'))
    return weight, toret


def add_nz_weights(array, nz, zmin, dz, p0, weight_ntile, completeness_ntile,
                   randoms=False, completeness='fracz'):
    """
    Add the local density and the weight it implies to one catalog.

    ``NX`` is the density at the object's redshift, corrected for how complete the survey was
    where it sits, and ``WEIGHT_FKP`` is one over one plus that density times ``p0``. The
    completeness weight is at the same time divided out of ``WEIGHT`` per number of tiles, so
    that the weighting does not count the same incompleteness twice.

    The two per tile factors are passed in rather than measured here, both because they are
    properties of the data and of the first random catalog and because it lets every other
    random catalog be finished without them.

    Parameters
    ----------
    array : array
        Clustering catalog, data or randoms.
    nz : array
        Density per redshift bin, the fourth column :func:`compute_nz` returns.
    zmin, dz : float
        Lower edge and width of the bins.
    p0 : float
        Power spectrum amplitude the weight is tuned at.
    weight_ntile, completeness_ntile : array
        Per number of tiles, from :func:`compute_completeness_per_ntile`.
    randoms : bool, default=False
        Whether ``array`` is a random catalog, which carries the tile completeness and was
        already normalized region by region.
    completeness : str, default='fracz'
        Weighting scheme.
    """
    index = ((np.asarray(array['Z']) - zmin) / dz).astype('i8')
    density = np.zeros(len(array), dtype='f8')
    valid = (index >= 0) & (index < len(nz))
    density[valid] = np.asarray(nz)[index[valid]]

    # A copy, however shallow: columns are about to be set on it, and those are the caller's.
    toret = as_table(array).copy(copy_data=False)
    ntile = np.clip(np.asarray(toret['NTILE']) - 1, 0, len(completeness_ntile) - 1)
    set_column(toret, 'NX', density * completeness_ntile[ntile], dtype='f8')
    set_column(toret, 'WEIGHT_FKP', 1. / (1. + toret['NX'] * p0), dtype='f8')

    weight = toret['WEIGHT_COMP'] * toret['WEIGHT_SYS'] * toret['WEIGHT_ZFAIL']
    if randoms:
        if completeness != 'bitweights':
            weight = weight * toret['FRAC_TLOBS_TILES']
        # The randoms were rescaled region by region when they were drawn; keep that
        # normalization rather than overwriting it.
        factor = np.ones(len(toret))
        positive = weight > 0
        factor[positive] = toret['WEIGHT'][positive] / weight[positive]
        weight = factor * weight
    set_column(toret, 'WEIGHT', weight / weight_ntile[ntile], dtype='f8')
    return toret

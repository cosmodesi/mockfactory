"""
Clustering catalogs: the targets with a good redshift, weighted for the ones that are missing.

Everything a measurement needs is here and nothing else. The data is cut to the targets whose
redshift can be trusted and to the range the tracer is defined over, and carries the weight
that makes up for its neighbours: a target that shared a fiber location with others stands in
for them. The randoms are given a redshift and a weight drawn from the data, separately in
each imaging region, so that they follow the same radial and angular selection.

Two ways of weighting incompleteness are supported. The default divides by the fraction of
targets observed at the fiber location, which is what a single pass of the survey can measure.
The alternative uses the probability of having been observed across many alternative merged
target lists, which is what :mod:`mockfactory.desi.altmtl` produces bitweights for, and is
unbiased where the first is not.
"""

import logging

import numpy as np

from .utils import NULL, append_fields, get_photsys, select_fields


logger = logging.getLogger('lsscat.clustering')


#: Redshift range each tracer is defined over.
REDSHIFT_RANGE = {'LRG': (0.4, 1.1), 'ELG': (0.8, 1.6), 'QSO': (0.8, 2.1), 'BGS': (0.1, 0.5)}

#: Largest target identifier of each tracer in a second generation mock, which is how the
#: contaminants added to a sample are told apart from its own targets.
MOCK_TARGETID_MAX = {'ELG': 838860800000000, 'QSO': 419430400000000}


def get_redshift_range(tracer):
    """Return the redshift range ``tracer`` is measured over."""
    return REDSHIFT_RANGE.get(tracer[:3], (0., 6.))


def select_good_redshift(data, tracer, ismock=True):
    """
    Return the rows of ``data`` whose redshift can be used.

    What counts as usable differs by tracer, following how each one's redshift is measured. In
    a mock there is no fitting to fail, so what is left is the requirement that the target was
    observed at all, plus the ceiling that separates a sample from the contaminants mixed into
    it.

    Parameters
    ----------
    data : array
        Vetoed full data catalog.
    tracer : str
        Target class.
    ismock : bool, default=True
        Whether this is a mock. The redshift quality cuts of the real survey, which need
        quantities a mock does not have, are skipped.
    """
    zwarn = data['ZWARN']
    select = (zwarn != NULL) & (zwarn * 0 == 0)
    if tracer[:3] in ('LRG', 'LGE', 'BGS'):
        select &= zwarn == 0
    if tracer[:3] == 'ELG':
        select &= data['LOCATION_ASSIGNED']
    if tracer[:3] == 'QSO':
        select &= (data['Z'] * 0 == 0) & (data['Z'] != NULL) & (data['Z'] != 1.e20)
    if ismock:
        if tracer[:3] == 'LRG':
            # The mock's luminous red galaxies stop here; anything above is a contaminant.
            select &= data['Z'] < 1.5
        maximum = MOCK_TARGETID_MAX.get(tracer[:3], None)
        if maximum is not None:
            select &= data['TARGETID'] < maximum
    logger.info('{:d} of {:d} targets have a usable redshift'.format(int(select.sum()), len(data)))
    return select


def compute_iip_weight(prob_obs, nbits=128):
    """
    Return the inverse of the probability that a target was observed.

    Over ``nbits`` alternative realizations of the survey, a target was reached in some and
    not in others; one over that frequency is the weight it has to carry to stand in for the
    times it was missed. The counts are offset by one so that a target never reached still has
    a finite weight.

    Parameters
    ----------
    prob_obs : array
        Fraction of the realizations the target was observed in, as
        :mod:`mockfactory.desi.altmtl` writes it.
    nbits : int, default=128
        Number of realizations behind it.
    """
    return (nbits + 1.) / (nbits * np.asarray(prob_obs) + 1.)


#: Where the polynomials defining the second generation bright galaxy magnitude cut live.
BGS_ABSMAG_CUT_DIR = '/pscratch/sd/z/zxzhai/DESI_LSS'


def get_bgs_absmag_cut(coeff_dir=None, zsplit=0.3, offset=0.078):
    """
    Return the redshift dependent absolute magnitude threshold of the ``BGS_ANY-02`` sample.

    That sample is not a fixed cut: the threshold follows a cubic in redshift below ``zsplit``
    and is constant above it, with an offset. The coefficients are fitted externally and read
    from file. The result takes redshifts and returns the threshold for each, so it can be
    handed to :func:`make_clustering_data` as ``absmag_max``.

    The survey pipeline evaluates this with ``numpy.empty`` and two masks that between them
    miss the targets with no redshift, so those rows of its full catalogs are selected on
    uninitialised memory. They carry no usable redshift and the clustering cut drops them
    regardless, so the clustering catalogs are unaffected.
    """
    import os
    coeff_dir = BGS_ABSMAG_CUT_DIR if coeff_dir is None else coeff_dir
    low = np.poly1d(np.loadtxt(os.path.join(coeff_dir, 'BGS_ANY_zmagcut_a.dat')))
    high = float(np.loadtxt(os.path.join(coeff_dir, 'BGS_ANY_zmagcut_b.dat')))

    def absmag_max(z):
        z = np.asarray(z, dtype='f8')
        toret = np.full(z.shape, np.nan)
        below = z < zsplit
        toret[below] = low(z[below])
        toret[z >= zsplit] = high
        return toret + offset

    return absmag_max


def make_clustering_data(data, tracer, zmin=None, zmax=None, completeness='fracz',
                         nbits=128, ismock=True, columns=(), subsample=None, zsplit=None,
                         seed=None, absmag_max=None):
    """
    Return the clustering data catalog of one tracer.

    Parameters
    ----------
    data : array
        Vetoed full data catalog, from
        :func:`~mockfactory.desi.lsscat.veto.apply_veto_data`.
    tracer : str
        Target class.
    zmin, zmax : float, default=None
        Redshift range. Defaults to the tracer's own.
    completeness : str, default='fracz'
        How to weight the targets that were not observed. ``'fracz'`` divides by the fraction
        observed at the fiber location, ``'fracz_tiles'`` also by the fraction of the location
        that went to this tracer, and ``'bitweights'`` uses ``PROB_OBS`` over the alternative
        realizations.
    nbits : int, default=128
        Number of realizations behind ``PROB_OBS``.
    columns : tuple
        Extra columns to carry through, beyond the ones a measurement needs.
    subsample : float, list, default=None
        Fraction of the targets to keep, to bring a mock down to the density the real survey
        found. Two values, with ``zsplit``, give a different fraction on either side of a
        redshift. The survey pipeline draws this without a seed, which makes its catalogs
        irreproducible; pass ``seed`` to fix the draw.
    zsplit : float, default=None
        Redshift the two subsampling fractions apply on either side of.
    seed : int, default=None
        Seed of the subsampling draw.
    absmag_max : float, callable, default=None
        Keep only the targets brighter than this absolute magnitude, read from ``R_MAG_ABS``.
        A callable is given the redshifts and returns the threshold for each, which is how the
        second generation bright galaxy samples are defined; see
        :func:`~mockfactory.desi.lsscat.clustering.get_bgs_absmag_cut`.
        The bright galaxy sample is defined this way rather than by a targeting bit, so its
        catalogs come in variants named after the cut, such as ``BGS_BRIGHT-21.5``.
    """
    if zmin is None or zmax is None:
        default = get_redshift_range(tracer)
        zmin = default[0] if zmin is None else zmin
        zmax = default[1] if zmax is None else zmax

    select = select_good_redshift(data, tracer, ismock=ismock)
    if subsample is not None:
        rng = np.random.default_rng(seed=seed)
        fraction = np.empty(len(data), dtype='f8')
        if zsplit is None:
            fraction[...] = subsample
        else:
            below = data['Z'] < zsplit
            fraction[below], fraction[~below] = subsample[0], subsample[1]
        select &= rng.random(len(data)) < fraction
        logger.info('subsampling to {} keeps {:d} targets'.format(subsample, int(select.sum())))

    if absmag_max is not None:
        # A threshold may depend on redshift; the bright galaxy samples of the second
        # generation mocks are defined that way, by a polynomial in z rather than a number.
        limit = absmag_max(data['Z']) if callable(absmag_max) else absmag_max
        select &= data['R_MAG_ABS'] < limit
        logger.info('absolute magnitude cut keeps {:d} targets'.format(int(select.sum())))

    toret = data[select]
    select = (toret['Z'] > zmin) & (toret['Z'] < zmax)
    toret = toret[select]
    logger.info('{:d} targets in {:.3f} < z < {:.3f}'.format(len(toret), zmin, zmax))

    add = [(name, 'f8') for name in ('WEIGHT', 'WEIGHT_COMP', 'WEIGHT_ZFAIL', 'WEIGHT_SYS')
           if name not in toret.dtype.names]
    toret = append_fields(toret, add)
    for name in ('WEIGHT_ZFAIL', 'WEIGHT_SYS'):
        if (name, 'f8') in add:
            toret[name] = 1.
    bad = toret['WEIGHT_SYS'] * 0 != 0
    if bad.any():
        logger.info('{:d} targets with no imaging weight, set to one'.format(int(bad.sum())))
        toret['WEIGHT_SYS'][bad] = 1.

    if completeness == 'bitweights':
        toret['WEIGHT_COMP'] = compute_iip_weight(toret['PROB_OBS'], nbits=nbits)
    else:
        toret['WEIGHT_COMP'] = 1. / toret['FRACZ_TILELOCID']
        if completeness == 'fracz_tiles':
            toret['WEIGHT_COMP'] /= toret['FRAC_TLOBS_TILES']
    toret['WEIGHT'] = toret['WEIGHT_COMP'] * toret['WEIGHT_ZFAIL'] * toret['WEIGHT_SYS']
    logger.info('completeness weight between {:.3f} and {:.3f}, mean {:.4f}'
                .format(toret['WEIGHT_COMP'].min(), toret['WEIGHT_COMP'].max(),
                        toret['WEIGHT_COMP'].mean()))

    keep = ['TARGETID', 'RA', 'DEC', 'Z', 'NTILE', 'PHOTSYS', 'FRAC_TLOBS_TILES',
            'WEIGHT', 'WEIGHT_ZFAIL', 'WEIGHT_COMP', 'WEIGHT_SYS']
    keep += [name for name in ('BITWEIGHTS', 'PROB_OBS', 'WEIGHT_FKP', 'TILEID', 'R_MAG_ABS',
                               'R_MAG_APP', 'G_R_REST', 'G_R_OBS') + tuple(columns)
             if name in toret.dtype.names]
    return select_fields(toret, keep)


def get_regions(ra, dec, photsys, des=False):
    """
    Return the imaging regions the randoms are resampled within, as a list of masks.

    The two imaging surveys have their own target selection and so their own redshift
    distribution, which is why a random is only ever given the redshift of a target from the
    same one. Quasars need the DES footprint separated out of the south as well, its imaging
    being deeper.
    """
    north = np.asarray(photsys) == 'N'
    if not des:
        return [north, ~north]
    from regressis import footprint
    import healpy as hp
    foot = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True,
                                  mask_around_des=False, cut_desi=False)
    in_des = foot.get_imaging_surveys()[2][
        hp.ang2pix(256, np.radians(90. - np.asarray(dec)), np.radians(np.asarray(ra)), nest=True)]
    return [north, in_des, ~north & ~in_des]


def make_clustering_randoms(randoms, data, seed=0, tracer='', completeness='fracz',
                            columns=('Z', 'WEIGHT', 'WEIGHT_SYS', 'WEIGHT_COMP', 'WEIGHT_ZFAIL')):
    """
    Return one clustering random catalog, carrying a redshift and a weight drawn from the data.

    A random has no redshift of its own, so it is given one taken at random from the data of
    the same imaging region, along with the weights that came with it. The weights of each
    region are then rescaled so that every region has the same ratio of random to data weight,
    which is what keeps the regions from being weighted against each other by how complete they
    happen to be.

    Parameters
    ----------
    randoms : array
        Vetoed full random catalog, carrying ``FRAC_TLOBS_TILES``.
    data : array
        Clustering data catalog to draw from.
    seed : int, default=0
        Seed of the draw. The survey pipeline uses the index of the random file, so that each
        one is drawn differently but reproducibly.
    tracer : str, default=''
        Target class, which decides whether the DES footprint is resampled on its own.
    completeness : str, default='fracz'
        Must match what the data was weighted with. Under ``'fracz'`` the random carries the
        tile completeness, the data having been weighted only by its fiber location.
    columns : tuple
        Columns drawn from the data.
    """
    rng = np.random.default_rng(seed=seed)
    columns = [name for name in columns if name in data.dtype.names]

    toret = append_fields(randoms, [(name, data[name].dtype, data[name].shape[1:])
                                    for name in columns] + [('TARGETID_DATA', data['TARGETID'].dtype)])
    des = tracer.startswith('QSO')
    randoms_regions = get_regions(toret['RA'], toret['DEC'], toret['PHOTSYS'], des=des)
    data_regions = get_regions(data['RA'], data['DEC'], data['PHOTSYS'], des=des)

    for rsel, dsel in zip(randoms_regions, data_regions):
        if not dsel.any():
            continue
        index = rng.choice(int(dsel.sum()), int(rsel.sum()))
        drawn = data[dsel][index]
        for name in columns:
            toret[name][rsel] = drawn[name]
        toret['TARGETID_DATA'][rsel] = drawn['TARGETID']

    if completeness != 'bitweights':
        # The data was weighted only by its fiber location, so the tile completeness that the
        # rest of the weight is missing belongs on the random instead.
        toret['WEIGHT'] *= toret['FRAC_TLOBS_TILES']

    ratios = [np.sum(toret['WEIGHT'][rsel]) / np.sum(data['WEIGHT'][dsel])
              for rsel, dsel in zip(randoms_regions, data_regions)]
    for i in range(1, len(randoms_regions)):
        toret['WEIGHT'][randoms_regions[i]] *= ratios[0] / ratios[i]
        logger.info('region {:d} rescaled by {:.5f}'.format(i, ratios[0] / ratios[i]))

    keep = ['TARGETID', 'RA', 'DEC', 'NTILE', 'PHOTSYS', 'FRAC_TLOBS_TILES'] + columns \
        + ['TARGETID_DATA']
    return select_fields(toret, keep)

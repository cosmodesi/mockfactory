"""
Contaminants of the dark-time mock target catalogs: targets the real survey selected as emission line galaxies or quasars
and could not give a usable redshift, put into the mock so that they compete for fibers as the real ones did. This is the
survey pipeline's recipe for its mocks, re-implemented without importing ``LSS``:

- the contaminant positions are those of the real targets of the tracer that were observed and failed the tracer's
  redshift criterion (``LSS.common_tools.goodz_infull``), each repeated by its completeness weight
  ``1 / FRACZ_TILELOCID / FRAC_TLOBS_TILES`` made an integer, the copies scattered uniformly within 0.025 deg of the
  original (``scripts/mock_Y3/create_{quasar,elg}_contaminants_duplicated.py``);
- for the emission line galaxies, an isotropic component of 97 per deg2 is taken out first, per nside 32 pixel in
  galactic coordinates, the share of the failures the mock already accounts for;
- the contaminants replace mock targets, positions and ``TARGETID`` only, every other column kept: random rows of the
  quasars with ``RSDZ <= 2.1``, of the ELG_LOP targets for the emission line galaxies; their identifiers start above
  :data:`CONTAMINANT_TARGETID_OFFSET`, which is how the catalog stage tells them apart
  (``scripts/mock_tools/add_contaminants_to_mock_z1.py``).

Three departures from the survey pipeline:

- the weights are made integers without bias, the integer part plus one with the probability of the fractional part,
  so that the number of copies follows the weights everywhere on average. The survey pipeline rounds each weight, then
  adds the missing units to the weights rounded up the most, and the quasar recipe does so separately for stars,
  galaxies and the unclassified: the total is kept, but where the copies go depends on how the weights are distributed,
  which differs between regions. On the DA2 quasar failures the north gets 312135 copies (311711 from grouped rounding,
  308727 from rounding all together) where its weights sum to 317560, 1.7% short;
- the copies are scattered with ``cos`` of the declination in radians, and the right ascension wrapped to [0, 360). The
  survey pipeline takes ``cos`` of the declination in degrees, so its right ascension offsets are wrong and at times
  enormous, and never wrapped: its files hold right ascensions from -54328 to 35891 (ELG) and -853 to 375 (QSO);
- every draw takes a seed, where the survey pipeline draws without one, both which contaminant realisation and which
  rows.
"""

import logging

import numpy as np


logger = logging.getLogger('altmtl.contaminants')

#: Identifiers of the contaminants start above these: ``1e8 * 2**22`` (QSO) and ``1e8 * 2**23`` (ELG), the mock targets
#: themselves being numbered from ``1e8`` times their targeting bit.
CONTAMINANT_TARGETID_OFFSET = {'QSO': 419430400000000, 'ELG': 838860800000000}
#: Density of the isotropic component of the emission line galaxy failures, per deg2, taken out before duplication.
ELG_ISOTROPIC_DENSITY = 97.
#: Largest distance of a copy from the original target, in degrees.
MAX_OFFSET = 0.025


def _tracer(tracer):
    tracer = tracer[:3].upper()
    if tracer not in CONTAMINANT_TARGETID_OFFSET:
        raise ValueError('contaminants are defined for ELG and QSO, not {}'.format(tracer))
    return tracer


def select_failed_redshifts(full, tracer):
    """
    Return the rows of a real full catalog (``{tracer}_full_noveto``) that were observed and failed the redshift criterion:
    ``o2c > 0.9`` for the emission line galaxies, a finite ``Z`` other than 999999 and 1e20 for the quasars.
    """
    tracer = _tracer(tracer)
    zwarn = np.asarray(full['ZWARN'])
    observed = (zwarn != 999999) & np.isfinite(zwarn)
    if tracer == 'ELG':
        good = np.asarray(full['o2c']) > 0.9
    else:
        z = np.asarray(full['Z'])
        good = np.isfinite(z) & (z != 999999) & (z != 1.e20)
    return observed & ~good


def integer_weights(weights, rng):
    """Integer weights without bias: the integer part, plus one with the probability of the fractional part."""
    weights = np.asarray(weights, dtype='f8')
    floor = np.floor(weights)
    return (floor + (rng.uniform(size=len(weights)) < weights - floor)).astype('i8')


def duplicate_positions(ra, dec, weights, max_offset=MAX_OFFSET, seed=None):
    """
    Return the contaminant positions: each target repeated by its integer weight (:func:`integer_weights`), at least once,
    the copies after the first scattered uniformly in distance up to ``max_offset`` degrees and in angle, the right
    ascension wrapped to [0, 360).
    """
    rng = np.random.default_rng(seed)
    ra, dec, weights = (np.asarray(array, dtype='f8') for array in (ra, dec, weights))
    weights = np.where(np.isinf(weights), 0., weights)
    counts = np.maximum(integer_weights(weights, rng), 1)
    index = np.repeat(np.arange(len(counts)), counts)
    first = np.zeros(len(index), dtype='?')
    first[np.concatenate([[0], np.cumsum(counts)[:-1]])] = True
    offset = np.where(first, 0., rng.uniform(0., max_offset, size=len(index)))
    phase = np.where(first, 0., rng.uniform(0., 2. * np.pi, size=len(index)))
    toret_ra = (ra[index] + offset * np.cos(phase) / np.cos(np.radians(dec[index]))) % 360.
    toret_dec = dec[index] + offset * np.sin(phase)
    return toret_ra, toret_dec


def thin_isotropic(ra, dec, weights, randoms_ra, randoms_dec, density=ELG_ISOTROPIC_DENSITY, nrandoms=18, randoms_density=2500.,
                   nside=32, seed=None):
    """
    Return the selection taking the isotropic component of ``density`` per deg2 out of the failures: in each nside
    ``nside`` pixel of galactic coordinates, a failure is kept with probability ``1 - density / n``, ``n`` the weighted
    failure density over the area the randoms cover (``nrandoms`` catalogs of ``randoms_density`` per deg2).
    """
    import healpy as hp
    # healpy's equatorial-to-galactic rotation, vectorised: astropy's SkyCoord takes tens of minutes on the ~600 million
    # randoms, and the two frames agree to well below the nside 32 pixel size (~1.8 deg)
    rotator = hp.Rotator(coord=['C', 'G'])

    def galactic_pixels(ra, dec):
        theta, phi = rotator(np.radians(90. - np.asarray(dec, dtype='f8')), np.radians(np.asarray(ra, dtype='f8')))
        return hp.ang2pix(nside, theta, phi)

    npix = hp.nside2npix(nside)
    pixel_area = 41253. / npix
    pix = galactic_pixels(ra, dec)
    weighted = np.bincount(pix, weights=np.where(np.isinf(weights), 0., weights), minlength=npix)
    covered = np.bincount(galactic_pixels(randoms_ra, randoms_dec), minlength=npix) / (randoms_density * nrandoms * pixel_area)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = density / (weighted / (covered * pixel_area))
    return np.random.default_rng(seed).uniform(size=len(pix)) > fraction[pix]


def get_frac_tlobs_tiles(full):
    """
    Return ``FRAC_TLOBS_TILES`` of a full catalog that lacks it, as the survey pipeline's contaminant scripts compute it:
    the fraction of the rows sharing a set of tiles ('TILES') at a location where a target of the tracer was assigned
    ('TILELOCID_ASSIGNED').
    """
    from ..lsscat.utils import group_fraction
    return group_fraction(np.asarray(full['TILES']), np.asarray(full['TILELOCID_ASSIGNED']))


def make_contaminants(full, tracer, randoms=None, seed=None):
    """
    Return the contaminant positions (ra, dec) from a real full catalog of the tracer (``{tracer}_full_noveto``, with
    'RA', 'DEC', 'ZWARN', 'FRACZ_TILELOCID', and 'o2c' or 'Z'; 'FRAC_TLOBS_TILES', else 'TILES' and 'TILELOCID_ASSIGNED' to
    compute it, see :func:`get_frac_tlobs_tiles`).

    The emission line galaxies need ``randoms``, the tracer's clustering randoms (18 catalogs, 'RA', 'DEC'), for the
    isotropic component.
    """
    tracer = _tracer(tracer)
    rng = np.random.default_rng(seed)
    failed = select_failed_redshifts(full, tracer)
    ra, dec = np.asarray(full['RA'])[failed], np.asarray(full['DEC'])[failed]
    frac_tlobs = full['FRAC_TLOBS_TILES'] if 'FRAC_TLOBS_TILES' in full.dtype.names else get_frac_tlobs_tiles(full)
    weights = 1. / np.asarray(full['FRACZ_TILELOCID'], dtype='f8')[failed] / np.asarray(frac_tlobs, dtype='f8')[failed]
    if tracer == 'ELG':
        if randoms is None:
            raise ValueError('the emission line galaxies need randoms for the isotropic component')
        keep = thin_isotropic(ra, dec, weights, randoms['RA'], randoms['DEC'], seed=rng.integers(2**32))
        ra, dec, weights = ra[keep], dec[keep], weights[keep]
    toret = duplicate_positions(ra, dec, weights, seed=rng.integers(2**32))
    logger.info('{}: {:d} failed redshifts give {:d} contaminants.'.format(tracer, len(ra), len(toret[0])))
    return toret


def get_replaceable(targets, tracer):
    """
    Return the rows of ``targets`` contaminants may replace: quasars with ``RSDZ <= 2.1``, ELG_LOP targets for the emission
    line galaxies. ``targets`` holds the one tracer.
    """
    tracer = _tracer(tracer)
    if tracer == 'QSO':
        return np.asarray(targets['RSDZ']) <= 2.1
    from desitarget.targetmask import desi_mask
    return (np.asarray(targets['DESI_TARGET']) & desi_mask['ELG_LOP']) > 0


def add_contaminants(targets, tracer, ra, dec, seed=None):
    """
    Replace random rows of ``targets`` (one tracer, a table with 'RA', 'DEC', 'TARGETID', and 'RSDZ' or 'DESI_TARGET', see
    :func:`get_replaceable`) by the contaminants at ``ra``, ``dec``, in place: new positions and identifiers above
    :data:`CONTAMINANT_TARGETID_OFFSET`, every other column kept, so a contaminant keeps the redshift, priority and
    subpriority of the target it replaced. The imaging columns are those of the replaced target: read them again at the
    new positions before any veto.

    Returns
    -------
    index : array
        Rows replaced, in the order of the contaminants.
    """
    tracer = _tracer(tracer)
    candidates = np.flatnonzero(get_replaceable(targets, tracer))
    if len(ra) > len(candidates):
        raise ValueError('{:d} contaminants for {:d} replaceable {} targets'.format(len(ra), len(candidates), tracer))
    index = np.random.default_rng(seed).choice(candidates, size=len(ra), replace=False)
    targetid = np.array(targets['TARGETID'])
    if targetid.max() >= CONTAMINANT_TARGETID_OFFSET[tracer]:
        raise ValueError('{} targets already have identifiers above the contaminant offset'.format(tracer))
    for name, value in [('RA', ra), ('DEC', dec)]:
        column = np.array(targets[name])
        column[index] = value
        targets[name] = column
    targetid[index] = CONTAMINANT_TARGETID_OFFSET[tracer] + 1 + np.arange(len(ra))
    targets['TARGETID'] = targetid
    logger.info('{}: {:d} of {:d} targets replaced by contaminants ({:.1f}%).'.format(tracer, len(ra), len(targetid), 100 * len(ra) / len(targetid)))
    return index


def get_zfix(targets, decimals=None):
    """
    Return the quasar redshifts the replay should use, ``(TARGETID, RSDZ)`` sorted by TARGETID, contaminants included;
    pass it as ``zfix`` to :func:`~mockfactory.desi.altmtl.run_mocks`. The replay uses them to decide whether an observed
    quasar is followed up (z > 2.1) or done.

    The survey pipeline goes through a text file, ``qsos/qso{i}.txt`` written with ``%.3f``, so its replay sees the
    redshifts rounded to three decimals; that is a side effect of the format, which only matters within 5e-4 of
    z = 2.1. ``decimals=3`` reproduces it; by default the redshifts are kept as they are.
    """
    targetid, z = np.asarray(targets['TARGETID']), np.asarray(targets['RSDZ'], dtype='f8')
    if decimals is not None: z = np.round(z, decimals)
    order = np.argsort(targetid)
    return targetid[order], z[order]

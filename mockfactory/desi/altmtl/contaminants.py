"""
Contaminants of the dark-time mock target catalogs: targets the real survey selected as emission line galaxies or quasars
and could not give a usable redshift, put into the mock so that they compete for fibers as the real ones did. This is the
survey pipeline's recipe for its mocks, re-implemented without importing ``LSS``:

- the contaminants come from the real survey's redshift failures (:func:`redshift_failures`): the targets of the tracer
  that were observed and failed the tracer's redshift criterion (``LSS.common_tools.goodz_infull``). Each is repeated by
  its completeness weight ``1 / FRACZ_TILELOCID / FRAC_TLOBS_TILES`` made an integer, the copies scattered uniformly within
  0.025 deg of the original (``scripts/mock_Y3/create_{quasar,elg}_contaminants_duplicated.py``);
- for the emission line galaxies, an isotropic component of 97 per deg2 is taken out first, per nside 32 pixel in
  galactic coordinates, the share of the failures the mock already accounts for;
- the contaminants replace mock targets, positions and ``TARGETID`` only, every other column kept: random rows of the
  quasars with ``RSDZ <= 2.1``, of the ELG_LOP targets for the emission line galaxies; their identifiers start above
  :data:`CONTAMINANT_TARGETID_OFFSET`, which is how the catalog stage tells them apart
  (``scripts/mock_tools/add_contaminants_to_mock_z1.py``).

Typically: ``ra, dec, weights = read_redshift_failures(fn, tracer)``, then for each mock
``draw_contaminants(ra, dec, weights, tracer, random_counts=..., seed=...)`` and :func:`add_contaminants`.

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

import os
import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger('altmtl.contaminants')

#: Identifiers of the contaminants start above these: ``1e8 * 2**22`` (QSO) and ``1e8 * 2**23`` (ELG), the mock targets
#: themselves being numbered from ``1e8`` times their targeting bit.
CONTAMINANT_TARGETID_OFFSET = {'QSO': 419430400000000, 'ELG': 838860800000000}
#: Density of the isotropic component of the emission line galaxy failures, per deg2, taken out before duplication.
ELG_ISOTROPIC_DENSITY = 97.
#: Largest distance of a copy from the original target, in degrees.
MAX_OFFSET = 0.025
#: The columns of a real full catalog (``{tracer}_full_noveto``) :func:`redshift_failures` reads, per tracer.
REDSHIFT_FAILURE_COLUMNS = {'QSO': ['RA', 'DEC', 'ZWARN', 'Z', 'FRACZ_TILELOCID', 'TILES', 'TILELOCID_ASSIGNED'],
                            'ELG': ['RA', 'DEC', 'ZWARN', 'o2c', 'FRACZ_TILELOCID', 'TILES', 'TILELOCID_ASSIGNED']}


def _tracer(tracer):
    """'ELG' or 'QSO' from a tracer name ('ELG_LOPnotqso', 'QSO'...): the classes contaminants are defined for."""
    tracer = tracer[:3].upper()
    if tracer not in CONTAMINANT_TARGETID_OFFSET:
        raise ValueError('contaminants are defined for ELG and QSO, not {}'.format(tracer))
    return tracer


def select_failed_redshifts(full, tracer):
    """
    Return the rows of a real full catalog that were observed and failed the tracer's redshift criterion.

    Parameters
    ----------
    full : array
        Real full catalog of the tracer (``{tracer}_full_noveto``), with 'ZWARN', and 'o2c' (ELG) or 'Z' (QSO).
    tracer : str
        'ELG' or 'QSO' (or a name starting so).

    Returns
    -------
    mask : array
        Observed (ZWARN neither 999999 nor NaN) and failed: ``o2c <= 0.9`` for the emission line galaxies, no finite ``Z``
        (other than 999999 and 1e20) for the quasars.
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
    """
    Return integer weights without bias: the integer part, plus one with the probability of the fractional part, so that
    their mean is ``weights``.

    Parameters
    ----------
    weights : array
        Weights, non-negative.
    rng : numpy.random.Generator
        Random generator.

    Returns
    -------
    weights : array
        Integer weights, 'i8'.
    """
    weights = np.asarray(weights, dtype='f8')
    floor = np.floor(weights)
    return (floor + (rng.uniform(size=len(weights)) < weights - floor)).astype('i8')


def duplicate_positions(ra, dec, weights, max_offset=MAX_OFFSET, seed=None, scatter_first=False):
    """
    Return the contaminant positions: each redshift failure repeated by its integer weight (:func:`integer_weights`), at
    least once, the copies scattered uniformly in distance up to ``max_offset`` degrees and in angle around it, the right
    ascension wrapped to [0, 360).

    Parameters
    ----------
    ra, dec : array
        Positions of the redshift failures, in degrees.
    weights : array
        Their completeness weights (infinite ones count as 0, giving one copy).
    max_offset : float, default=MAX_OFFSET
        Largest distance of a copy from its failure, in degrees.
    seed : int, default=None
        Random seed.
    scatter_first : bool, default=False
        ``False``: the first copy of each failure stays at its position, as the survey pipeline does. ``True``: every copy
        is scattered, the first included, so that no contaminant sits at the exact position where the survey gave the failure
        a fiber. The random draws are the same either way.

    Returns
    -------
    ra, dec : array
        Contaminant positions, in degrees.
    """
    rng = np.random.default_rng(seed)
    ra, dec, weights = (np.asarray(array, dtype='f8') for array in (ra, dec, weights))
    weights = np.where(np.isinf(weights), 0., weights)
    counts = np.maximum(integer_weights(weights, rng), 1)
    index = np.repeat(np.arange(len(counts)), counts)
    first = np.zeros(len(index), dtype='?')
    if not scatter_first: first[np.concatenate([[0], np.cumsum(counts)[:-1]])] = True
    offset = np.where(first, 0., rng.uniform(0., max_offset, size=len(index)))
    phase = np.where(first, 0., rng.uniform(0., 2. * np.pi, size=len(index)))
    toret_ra = (ra[index] + offset * np.cos(phase) / np.cos(np.radians(dec[index]))) % 360.
    toret_dec = dec[index] + offset * np.sin(phase)
    return toret_ra, toret_dec


def galactic_pixels(ra, dec, nside=32):
    """
    Return the HEALPix pixels (ring ordering) of positions in galactic coordinates.

    healpy's equatorial-to-galactic rotation, vectorised: astropy's SkyCoord takes tens of minutes on the ~600 million
    randoms of the emission line galaxies, and the two frames agree to well below the nside 32 pixel size (~1.8 deg).

    Parameters
    ----------
    ra, dec : array
        Equatorial positions, in degrees.
    nside : int, default=32
        HEALPix resolution.

    Returns
    -------
    pixels : array
        Pixel of each position.
    """
    import healpy as hp
    theta, phi = hp.Rotator(coord=['C', 'G'])(np.radians(90. - np.asarray(dec, dtype='f8')), np.radians(np.asarray(ra, dtype='f8')))
    return hp.ang2pix(nside, theta, phi)


def random_pixel_counts(randoms_ra, randoms_dec, nside=32):
    """
    Return the number of randoms in each pixel of galactic coordinates (:func:`galactic_pixels`): all :func:`thin_isotropic`
    needs of the randoms. The same for every mock, so computed once (5 min for the ~600 million randoms of the emission line
    galaxies, :func:`read_random_pixel_counts` keeps it).

    Parameters
    ----------
    randoms_ra, randoms_dec : array
        Positions of the randoms, in degrees: the tracer's clustering randoms.
    nside : int, default=32
        HEALPix resolution.

    Returns
    -------
    counts : array
        Number of randoms per pixel, of size ``12 * nside**2``.
    """
    import healpy as hp
    return np.bincount(galactic_pixels(randoms_ra, randoms_dec, nside=nside), minlength=hp.nside2npix(nside))


def thin_isotropic(ra, dec, weights, random_counts, density=ELG_ISOTROPIC_DENSITY, nrandoms=18, randoms_density=2500., nside=32, seed=None):
    """
    Return the selection of redshift failures that takes their isotropic component out: in each pixel of galactic coordinates,
    a failure is kept with probability ``1 - density / n``, ``n`` the weighted failure density there over the area the randoms
    cover.

    Parameters
    ----------
    ra, dec : array
        Positions of the redshift failures, in degrees.
    weights : array
        Their completeness weights.
    random_counts : array
        Number of randoms per pixel (:func:`random_pixel_counts`), of ``nrandoms`` catalogs of ``randoms_density`` per deg2.
    density : float, default=ELG_ISOTROPIC_DENSITY
        Density of the isotropic component, per deg2.
    nrandoms : int, default=18
        Number of random catalogs counted in ``random_counts``.
    randoms_density : float, default=2500.
        Density of each random catalog, per deg2.
    nside : int, default=32
        HEALPix resolution of ``random_counts``.
    seed : int, default=None
        Random seed.

    Returns
    -------
    mask : array
        The failures kept.
    """
    import healpy as hp
    npix = hp.nside2npix(nside)
    pixel_area = 41253. / npix
    pix = galactic_pixels(ra, dec, nside=nside)
    weighted = np.bincount(pix, weights=np.where(np.isinf(weights), 0., weights), minlength=npix)
    covered = np.asarray(random_counts) / (randoms_density * nrandoms * pixel_area)
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = density / (weighted / (covered * pixel_area))
    return np.random.default_rng(seed).uniform(size=len(pix)) > fraction[pix]


def get_frac_tlobs_tiles(full):
    """
    Return ``FRAC_TLOBS_TILES`` of a full catalog that lacks it, as the survey pipeline's contaminant scripts compute it:
    the fraction of the rows sharing a set of tiles ('TILES') at a location where a target of the tracer was assigned
    ('TILELOCID_ASSIGNED').

    Parameters
    ----------
    full : array
        Real full catalog, with 'TILES' and 'TILELOCID_ASSIGNED'.

    Returns
    -------
    frac_tlobs_tiles : array
        One value per row.
    """
    from ..lsscat.utils import group_fraction
    return group_fraction(np.asarray(full['TILES']), np.asarray(full['TILELOCID_ASSIGNED']))


def redshift_failures(full, tracer):
    """
    Return the redshift failures of a real full catalog: the positions of the targets that were observed and failed the
    redshift criterion (:func:`select_failed_redshifts`), and their completeness weights, ``1 / FRACZ_TILELOCID /
    FRAC_TLOBS_TILES``. This is what the contaminants are drawn from (:func:`draw_contaminants`), the same for every mock.

    Parameters
    ----------
    full : array
        Real full catalog of the tracer (``{tracer}_full_noveto``), with the columns :data:`REDSHIFT_FAILURE_COLUMNS`
        ('FRAC_TLOBS_TILES' instead of 'TILES' and 'TILELOCID_ASSIGNED' if it has it, see :func:`get_frac_tlobs_tiles`).
    tracer : str
        'ELG' or 'QSO'.

    Returns
    -------
    ra, dec, weights : array
        Positions (degrees) and completeness weights of the failures.
    """
    tracer = _tracer(tracer)
    failed = select_failed_redshifts(full, tracer)
    ra, dec = np.asarray(full['RA'])[failed], np.asarray(full['DEC'])[failed]
    frac_tlobs = full['FRAC_TLOBS_TILES'] if 'FRAC_TLOBS_TILES' in full.dtype.names else get_frac_tlobs_tiles(full)
    weights = 1. / np.asarray(full['FRACZ_TILELOCID'], dtype='f8')[failed] / np.asarray(frac_tlobs, dtype='f8')[failed]
    return ra, dec, weights


def draw_contaminants(ra, dec, weights, tracer, random_counts=None, seed=None, scatter_first=False):
    """
    Return the contaminant positions of one mock, drawn from the redshift failures (:func:`redshift_failures`): for the
    emission line galaxies, their isotropic component taken out first (:func:`thin_isotropic`), then each failure duplicated
    according to its weight (:func:`duplicate_positions`).

    Parameters
    ----------
    ra, dec, weights : array
        Redshift failures, from :func:`redshift_failures` or :func:`read_redshift_failures`.
    tracer : str
        'ELG' or 'QSO'.
    random_counts : array, default=None
        Emission line galaxies only (required): number of their clustering randoms per pixel (:func:`random_pixel_counts`).
    seed : int, default=None
        Random seed, of the mock.
    scatter_first : bool, default=False
        Whether the first copy of each failure is scattered too (see :func:`duplicate_positions`).

    Returns
    -------
    ra, dec : array
        Contaminant positions, in degrees.
    """
    tracer = _tracer(tracer)
    rng = np.random.default_rng(seed)
    if tracer == 'ELG':
        if random_counts is None:
            raise ValueError('the emission line galaxies need the count of their randoms per pixel for the isotropic component')
        keep = thin_isotropic(ra, dec, weights, random_counts, seed=rng.integers(2**32))
        ra, dec, weights = ra[keep], dec[keep], weights[keep]
    toret = duplicate_positions(ra, dec, weights, seed=rng.integers(2**32), scatter_first=scatter_first)
    logger.info('{}: {:d} failed redshifts give {:d} contaminants.'.format(tracer, len(ra), len(toret[0])))
    return toret


def make_contaminants(full, tracer, random_counts=None, seed=None, scatter_first=False):
    """
    Return the contaminant positions of one mock from a real full catalog: :func:`redshift_failures`, then
    :func:`draw_contaminants`. To draw several mocks, call :func:`redshift_failures` (or :func:`read_redshift_failures`) once
    and :func:`draw_contaminants` per mock.

    Parameters
    ----------
    full : array
        Real full catalog of the tracer (``{tracer}_full_noveto``), see :func:`redshift_failures`.
    tracer : str
        'ELG' or 'QSO'.
    random_counts, seed, scatter_first
        See :func:`draw_contaminants`.

    Returns
    -------
    ra, dec : array
        Contaminant positions, in degrees.
    """
    ra, dec, weights = redshift_failures(full, tracer)
    return draw_contaminants(ra, dec, weights, tracer, random_counts=random_counts, seed=seed, scatter_first=scatter_first)


def cached_arrays(fn, compute):
    """
    Return arrays computed once and kept in a file: read from the HDF5 file ``fn`` if it exists, else ``compute()``, written
    there. The first process to get to it computes them under a lock (``fn`` + '.lock'), the others wait and read them; the
    file is written next to its destination then renamed, so a reader never sees it half-written.

    Parameters
    ----------
    fn : str, Path
        HDF5 file, one dataset per array.
    compute : callable
        Returns the arrays, a dict.

    Returns
    -------
    arrays : dict
        The arrays.
    """
    import fcntl
    import h5py
    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    with open(str(fn) + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not fn.exists():
            arrays = compute()
            tmp_fn = '{}.{:d}.tmp'.format(fn, os.getpid())
            with h5py.File(tmp_fn, 'w') as file:
                for name, array in arrays.items(): file.create_dataset(name, data=array)
            os.replace(tmp_fn, fn)
            logger.info('cached -> {}'.format(fn))
            return arrays
    with h5py.File(fn, 'r') as file:
        return {name: file[name][...] for name in file}


def read_redshift_failures(fn, tracer, cache_dir=None):
    """
    Return the redshift failures (:func:`redshift_failures`) of the real full catalog file ``fn``, optionally kept in
    ``cache_dir``: tens of MB, where the full catalogs are 3 (QSO) and 22 (ELG) GB, so that many mocks can draw at once.

    Parameters
    ----------
    fn : str, Path
        Real full catalog of the tracer (``{tracer}_full_noveto.dat.fits``); only :data:`REDSHIFT_FAILURE_COLUMNS` are read.
    tracer : str
        'ELG' or 'QSO'.
    cache_dir : str, Path, default=None
        If given, the failures are kept there (``{name}_redshift_failures.h5``, :func:`cached_arrays`), read back by later calls.

    Returns
    -------
    ra, dec, weights : array
        Positions (degrees) and completeness weights of the failures.
    """
    def compute():
        """Read the failures from the full catalog: a dict of RA, DEC, WEIGHT."""
        import fitsio
        ra, dec, weights = redshift_failures(fitsio.read(str(fn), columns=REDSHIFT_FAILURE_COLUMNS[_tracer(tracer)]), tracer)
        return {'RA': ra, 'DEC': dec, 'WEIGHT': weights}

    arrays = compute() if cache_dir is None else cached_arrays(Path(cache_dir) / Path(fn).name.replace('.dat.fits', '_redshift_failures.h5'), compute)
    return arrays['RA'], arrays['DEC'], arrays['WEIGHT']


def read_random_pixel_counts(fns, cache_fn=None, nside=32):
    """
    Return the number of randoms per pixel (:func:`random_pixel_counts`) of the random catalog files ``fns``, optionally kept in
    ``cache_fn``.

    Parameters
    ----------
    fns : list
        Random catalogs (FITS, 'RA', 'DEC'): the tracer's clustering randoms, e.g. the 18 catalogs of both caps.
    cache_fn : str, Path, default=None
        If given, the counts are kept in this HDF5 file (:func:`cached_arrays`), read back by later calls.
    nside : int, default=32
        HEALPix resolution.

    Returns
    -------
    counts : array
        Number of randoms per pixel.
    """
    def compute():
        """Count the randoms per pixel from the random catalogs: a dict with COUNTS."""
        import fitsio
        randoms = [fitsio.read(str(fn), columns=['RA', 'DEC']) for fn in fns]
        return {'COUNTS': random_pixel_counts(np.concatenate([r['RA'] for r in randoms]), np.concatenate([r['DEC'] for r in randoms]),
                                              nside=nside)}
    return (compute() if cache_fn is None else cached_arrays(cache_fn, compute))['COUNTS']


def get_replaceable(targets, tracer):
    """
    Return the rows of ``targets`` contaminants may replace: quasars with ``RSDZ <= 2.1``, ELG_LOP targets for the emission
    line galaxies.

    Parameters
    ----------
    targets : array
        Mock targets of the one tracer, with 'RSDZ' (QSO) or 'DESI_TARGET' (ELG).
    tracer : str
        'ELG' or 'QSO'.

    Returns
    -------
    mask : array
        The replaceable rows.
    """
    tracer = _tracer(tracer)
    if tracer == 'QSO':
        return np.asarray(targets['RSDZ']) <= 2.1
    from desitarget.targetmask import desi_mask
    return (np.asarray(targets['DESI_TARGET']) & desi_mask['ELG_LOP']) > 0


def add_contaminants(targets, tracer, ra, dec, seed=None):
    """
    Replace random rows of ``targets`` by the contaminants at ``ra``, ``dec``, in place: new positions and identifiers above
    :data:`CONTAMINANT_TARGETID_OFFSET` (the k-th contaminant gets offset + 1 + k), every other column kept, so a contaminant
    keeps the redshift, priority and subpriority of the target it replaced. The imaging columns are those of the replaced
    target: read them again at the new positions before any veto.

    Parameters
    ----------
    targets : table
        Mock targets of the one tracer, with 'RA', 'DEC', 'TARGETID', and 'RSDZ' or 'DESI_TARGET' (see :func:`get_replaceable`).
    tracer : str
        'ELG' or 'QSO'.
    ra, dec : array
        Contaminant positions (:func:`draw_contaminants`), every one injected.
    seed : int, default=None
        Random seed of the choice of the rows replaced.

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
    Return the quasar redshifts the replay should use, contaminants included: pass them as ``zfix`` to
    :func:`~mockfactory.desi.altmtl.run_mocks`. The replay uses them to decide whether an observed quasar is followed up
    (z > 2.1) or done.

    The survey pipeline goes through a text file, ``qsos/qso{i}.txt`` written with ``%.3f``, so its replay sees the
    redshifts rounded to three decimals; that is a side effect of the format, which only matters within 5e-4 of
    z = 2.1. ``decimals=3`` reproduces it; by default the redshifts are kept as they are.

    Parameters
    ----------
    targets : array
        Mock targets, with 'TARGETID' and 'RSDZ'.
    decimals : int, default=None
        Round the redshifts to this many decimals.

    Returns
    -------
    targetid, z : array
        Sorted by TARGETID.
    """
    targetid, z = np.asarray(targets['TARGETID']), np.asarray(targets['RSDZ'], dtype='f8')
    if decimals is not None: z = np.round(z, decimals)
    order = np.argsort(targetid)
    return targetid[order], z[order]

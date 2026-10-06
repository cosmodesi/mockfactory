"""
Vetoes: the cuts that take a full catalog down to the footprint a measurement can use.

Three things are removed. Locations the tracer could never have been assigned at, because a
higher priority target had already claimed them. Targets whose imaging is unreliable, either
from the legacy survey's mask bits or from the per-tracer masks built afterwards. And regions
where an observing condition, read off a healpix map, is bad enough that the target density
cannot be trusted.

The data and the randoms take the same imaging and map cuts, which is what makes their ratio
the selection function; only the priority cut differs, the randoms having no assignment of
their own to be judged by.
"""

import logging

from pathlib import Path

import numpy as np

from astropy.table import Table

from .utils import as_table, group_fraction, get_photsys, set_column


logger = logging.getLogger('lsscat.veto')


#: Extinction coefficient per band, for turning a depth map into an extinction corrected one.
EXTINCTION_COEFF = {'G': 3.214, 'R': 2.165, 'Z': 1.211, 'W1': 0.184, 'W2': 0.113}

#: Cuts on the observing condition maps, shared by every tracer. A depth has to be above its
#: value and everything else below it.
MAP_CUTS = {'EBV': 0.15, 'STARDENS': 4.4, 'PSFSIZE_G': 2.4, 'PSFSIZE_R': 2.3, 'PSFSIZE_Z': 2.,
            'GALDEPTH_G': 250., 'GALDEPTH_R': 80., 'GALDEPTH_Z': 30., 'PSFDEPTH_W1': 2.}

#: Legacy survey mask bits vetoed per tracer. The string stands for the separate mask the
#: luminous red galaxy selection uses, which comes as a column rather than as bits.
#: The bright galaxies get the bits their targets are cut on (1, 5, 6, 7, 10, 13) as well as
#: their own, 11: `LSS` vetoes on 11 alone because its randoms arrive already cut on the
#: others, and these randoms do not, so 11 alone leaves ~1% of them where no target can be.
MASK_BITS = {'LRG': 'lrg_mask', 'ELG': None, 'QSO': [8, 9, 11], 'BGS': [1, 5, 6, 7, 10, 11, 13]}


#: Custom masks of circles and rectangles per tracer (``LSS.globals``, ``reccircmasks``), applied to data and randoms
#: before the imaging bits, as ``LSS.common_tools.apply_veto`` does.
CUSTOM_MASK_DIR = '/dvs_ro/cfs/cdirs/desi/users/rongpu/desi_mask'
CUSTOM_MASKS = {'QSO': ['desi_custom_mask_v1.txt'], 'ELG': ['desi_custom_mask_v1.txt', 'elg_custom_mask_v1.1_draft.txt']}


def get_custom_masks(tracer, mask_dir=CUSTOM_MASK_DIR):
    """Return the custom mask files of ``tracer``, see :data:`CUSTOM_MASKS`."""
    return [Path(mask_dir) / fn for fn in CUSTOM_MASKS.get(tracer[:3], [])]


def read_custom_mask(fn):
    """
    Return the circles, (ra, dec, radius [arcsec]), and rectangles, (ramin, ramax, decmin, decmax), of a custom mask file:
    one per line, comma separated, '#' starting a comment (``LSS.common_tools.parse_circandrec_mask``).
    """
    circles, rectangles = [], []
    with open(fn, 'r') as file:
        for line in file:
            line = line.split('#')[0].strip()
            if not line: continue
            values = [float(value) for value in line.split(',')]
            if len(values) == 3: circles.append(values)
            elif len(values) == 4: rectangles.append(values)
            else: raise ValueError('cannot parse {} in {}'.format(line, fn))
    return np.array(circles, dtype='f8').reshape(-1, 3), np.array(rectangles, dtype='f8').reshape(-1, 4)


def in_custom_mask(ra, dec, fns, chunk=1000000):
    """
    Return whether each position falls in one of the circles or rectangles of the custom mask files ``fns``, the bounds
    excluded as in ``LSS.common_tools.maskcircandrec``.
    """
    ra, dec = np.asarray(ra, dtype='f8'), np.asarray(dec, dtype='f8')
    toret = np.zeros(ra.size, dtype='?')
    for fn in fns:
        circles, rectangles = read_custom_mask(fn)
        for ramin, ramax, decmin, decmax in rectangles:
            toret |= (ra > ramin) & (ra < ramax) & (dec > decmin) & (dec < decmax)
        if not len(circles): continue
        def unit(ra, dec):
            ra, dec = np.radians(ra), np.radians(dec)
            return np.column_stack([np.cos(ra) * np.cos(dec), np.sin(ra) * np.cos(dec), np.sin(dec)])
        centers, cosradius = unit(circles[:, 0], circles[:, 1]), np.cos(np.radians(circles[:, 2] / 3600.))
        # in chunks: the survey pipeline forms the whole (objects x circles) matrix at once
        for start in range(0, ra.size, chunk):
            sl = slice(start, start + chunk)
            toret[sl] |= np.any(unit(ra[sl], dec[sl]) @ centers.T > cosradius, axis=1)
    return toret


def get_mask_bits(tracer):
    """Return the imaging mask bits vetoed for ``tracer``."""
    return MASK_BITS.get(tracer[:3], None)


def apply_imaging_veto(array, bits=None):
    """
    Return the rows of ``array`` whose imaging is usable.

    A target needs coverage in all three optical bands, and none of the mask bits that flag a
    bright star, a large galaxy or a globular cluster over its position.

    Parameters
    ----------
    array : array
        Catalog carrying ``NOBS_G``, ``NOBS_R``, ``NOBS_Z`` and ``MASKBITS``.
    bits : list, str, default=None
        Mask bits to veto, or the name of a mask column to use instead.
    """
    array = as_table(array)
    size = len(array)
    keep = (array['NOBS_G'] > 0) & (array['NOBS_R'] > 0) & (array['NOBS_Z'] > 0)
    if isinstance(bits, str):
        keep &= array[bits] == 0
    elif bits is not None:
        for bit in bits:
            keep &= (array['MASKBITS'] & 2**bit) == 0
    logger.info('imaging veto keeps {:d} of {:d} rows'.format(int(keep.sum()), size))
    return array[keep]


def apply_map_veto(array, maps_north, maps_south, cuts=None, nside=256):
    """
    Return the rows of ``array`` sitting where the imaging was good enough.

    Each condition is read off a healpix map at the target's position, from the northern or
    the southern map according to which survey imaged it. A depth is corrected for extinction
    first, and has to be above its cut; everything else has to be below.

    Parameters
    ----------
    array : array
        Catalog carrying ``RA``, ``DEC`` and, if it has it, ``PHOTSYS``.
    maps_north, maps_south : array
        Maps of the observing conditions, one value per healpix, for the two imaging surveys.
    cuts : dict, default=None
        Threshold per map. Defaults to :data:`MAP_CUTS`.
    nside : int, default=256
        Resolution the maps are given at, in the nested scheme.
    """
    import healpy as hp
    if cuts is None:
        cuts = MAP_CUTS
    array = as_table(array)
    photsys = array['PHOTSYS'] if 'PHOTSYS' in array.colnames \
        else get_photsys(array['RA'], array['DEC'])
    north = photsys == 'N'
    pixel = hp.ang2pix(nside, np.radians(90. - array['DEC']), np.radians(array['RA']), nest=True)

    size = len(array)
    keep = np.ones(size, dtype='?')
    for name, cut in cuts.items():
        values = np.empty(size, dtype='f8')
        for select, maps in [(north, maps_north), (~north, maps_south)]:
            values[select] = maps[name][pixel[select]]
            if 'DEPTH' in name:
                band = name.split('_')[-1]
                values[select] *= 10**(-0.4 * EXTINCTION_COEFF[band] * maps['EBV'][pixel[select]])
        if name == 'STARDENS':
            values = np.log10(values)
        keep &= values > cut if 'DEPTH' in name else values < cut
        logger.info('  {:<14s} keeps {:.4f}'.format(name, keep.mean()))
    logger.info('map veto keeps {:d} of {:d} rows'.format(int(keep.sum()), size))
    return array[keep]


def apply_veto_data(data, max_priority, bits=None, maps_north=None, maps_south=None,
                    cuts=None, nside=256, custom_masks=None):
    """
    Return the vetoed full data catalog, with its completeness recomputed over what is left.

    Both completeness columns are fractions over the targets sharing a set of tiles, so both
    have to be measured again once the vetoes have removed part of the sample: ``COMP_TILE``
    is how often such a target was observed, ``FRAC_TLOBS_TILES`` how often its fiber location
    went to something of the same tracer, which is what the clustering stage divides by.

    Parameters
    ----------
    data : array
        Full data catalog, from :func:`~mockfactory.desi.lsscat.full.make_full_data`.
    max_priority : int
        Priority above which the location was already claimed; see
        :func:`~mockfactory.desi.lsscat.full.get_max_priority`.
    bits : list, str, default=None
        Imaging mask bits, from :func:`get_mask_bits`.
    maps_north, maps_south : array, default=None
        Observing condition maps. The map veto is skipped when not given.
    custom_masks : list, default=None
        Custom mask files, from :func:`get_custom_masks`, applied before the imaging bits.
    """
    data = as_table(data)
    size = len(data)
    keep = data['GOODHARDLOC'] & (data['PRIORITY_ASSIGNED'] <= max_priority)
    logger.info('priority and hardware keep {:d} of {:d} rows'.format(int(keep.sum()), size))
    if custom_masks:
        masked = in_custom_mask(data['RA'], data['DEC'], custom_masks)
        logger.info('custom masks remove {:d} of {:d} rows'.format(int((keep & masked).sum()), int(keep.sum())))
        keep &= ~masked
    toret = apply_imaging_veto(data[keep], bits=bits)

    # Measured before the map veto, and deliberately so: the maps remove whole patches of sky
    # rather than individual targets, and a set of tiles straddling the edge of one would
    # otherwise be judged on whichever part of it happened to survive.
    set_column(toret, 'COMP_TILE', group_fraction(toret['TILES'], toret['LOCATION_ASSIGNED']))
    set_column(toret, 'FRAC_TLOBS_TILES', group_fraction(toret['TILES'],
                                                         toret['TILELOCID_ASSIGNED']), dtype='f8')
    logger.info('assignment completeness is {:.4f}'.format(toret['LOCATION_ASSIGNED'].mean()))

    if maps_north is not None:
        toret = apply_map_veto(toret, maps_north, maps_south, cuts=cuts, nside=nside)
    return toret


def apply_veto_randoms(randoms, max_priority, bits=None, maps_north=None, maps_south=None,
                       cuts=None, nside=256, custom_masks=None):
    """
    Return the vetoed full random catalog.

    The randoms take the same imaging and map cuts as the data. What differs is the priority
    they are judged by: a random has no assignment of its own, so the cut is on the priority
    that ruled at the location, which is what says whether the tracer could have been put
    there at all.
    """
    randoms = as_table(randoms)
    size = len(randoms)
    keep = randoms['GOODHARDLOC'] & (randoms['PRIORITY'] <= max_priority)
    logger.info('priority and hardware keep {:d} of {:d} rows'.format(int(keep.sum()), size))
    if custom_masks:
        keep &= ~in_custom_mask(randoms['RA'], randoms['DEC'], custom_masks)
    toret = apply_imaging_veto(randoms[keep], bits=bits)
    if maps_north is not None:
        toret = apply_map_veto(toret, maps_north, maps_south, cuts=cuts, nside=nside)
    return toret


def get_frac_tlobs(data):
    """
    Return, per distinct set of tiles, the fraction of its locations that went to this tracer:
    the table the survey pipeline writes per tracer, and which the randoms are given so that
    they carry the same completeness as the data they will be divided by.
    """
    data = as_table(data)
    tiles, index = np.unique(data['TILES'].value, return_index=True)
    frac = data['FRAC_TLOBS_TILES'].value[index].astype('f8', copy=False)
    return Table({'TILES': tiles.astype(data['TILES'].dtype, copy=False),
                  'FRAC_TLOBS_TILES': frac}, copy=False)


def add_frac_tlobs(randoms, frac_tlobs, missing=1., data=None):
    """
    Add the tile completeness to the randoms, looked up by their set of tiles.

    A few per cent of randoms carry a set of tiles that no target of the tracer has, so the
    data measured no completeness for it. This is not empty sky: a target drops from its set
    the tiles whose fiber location was unusable, and which those are depends on the location,
    so the same patch of sky yields slightly different sets for the data and for the randoms.

    What to give them is a real choice. The survey pipeline makes it twice and differently:
    the full randoms get zero, which throws them away, and the clustering randoms get one,
    which grants them full completeness. For the DA2 luminous red galaxies that is 2.35% of
    the randoms and 3.44% of the random weight.

    Parameters
    ----------
    randoms : array
        Randoms carrying ``TILES``.
    frac_tlobs : array
        Completeness per set of tiles, from :func:`get_frac_tlobs`.
    missing : float, str, default=1.
        Value for a set of tiles the data does not have. A number is used as is, one being
        what the survey pipeline's clustering catalogs carry. ``'ntile'`` instead uses the
        mean completeness of the data over the targets covered by as many tiles, which is
        neither of the two extremes and needs ``data``.
    data : array, default=None
        Data catalog, for ``missing='ntile'``.
    """
    from .utils import join_left
    randoms, frac_tlobs = as_table(randoms), as_table(frac_tlobs)
    # The two sides have to name a set of tiles the same way, and they are grouped in separate
    # calls: randoms built against a cached count from an older run carry the set written out
    # as text while the data carries the code for it. That join matches nothing, and nothing
    # about the result says so, so it is refused here instead.
    if randoms['TILES'].dtype.kind != frac_tlobs['TILES'].dtype.kind:
        raise ValueError('randoms name their tiles as {} and the data as {}: one of the two '
                         'was built against a stale tile count'
                         .format(randoms['TILES'].dtype, frac_tlobs['TILES'].dtype))
    fill = 0. if missing == 'ntile' else missing
    toret = join_left(randoms, frac_tlobs, 'TILES', columns=['FRAC_TLOBS_TILES'],
                      fill={'FRAC_TLOBS_TILES': fill})
    absent = ~np.isin(toret['TILES'], frac_tlobs['TILES'])
    if absent.any():
        logger.info('{:d} randoms ({:.4%}) sit on a set of tiles the data does not have'
                    .format(int(absent.sum()), absent.mean()))
    if missing == 'ntile':
        ntile = np.asarray(data['NTILE'])
        counts = np.bincount(ntile)
        mean = np.zeros(len(counts), dtype='f8')
        nonzero = counts > 0
        mean[nonzero] = (np.bincount(ntile, weights=data['FRAC_TLOBS_TILES'])[nonzero]
                         / counts[nonzero])
        at = np.clip(np.asarray(toret['NTILE'][absent]), 0, len(mean) - 1)
        toret['FRAC_TLOBS_TILES'][absent] = mean[at]
    return toret

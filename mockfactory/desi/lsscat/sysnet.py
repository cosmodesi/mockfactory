"""
Imaging systematic weights from SYSNet, the neural network regression the survey pipeline uses for the emission line
galaxies (Rezaie et al. 2020, `sysnet <https://github.com/mehdirezaie/sysnetdev>`_), re-implemented around the ``sysnet``
package without importing ``LSS``. It follows ``LSS`` as its mocks run it (``scripts/mock_tools/mkCat_amtl.py``
``--prep4sysnet`` / ``--addsysnet``, ``LSS.imaging.sysnet_tools.prep4sysnet``, ``scripts/sysnetELG_LOPnotqso_zbins.sh``):

- the fit is done in each photometric region ('N', 'S', by ``PHOTSYS``) and redshift bin on its own;
- the input is a healpix table, nside 256 in ring ordering: per pixel, the weighted data count ('label'), the fraction of
  the pixel the randoms cover ('fracgood': unweighted random count over the all-sky random count of the same pixel), and
  the template maps ('features', the depths corrected for Galactic extinction); pixels with no randoms or an unseen
  template are dropped. The data are counted with their completeness weights only, ``WEIGHT_COMP / FRAC_TLOBS_TILES *
  WEIGHT_ZFAIL`` (``1 / FRACZ_TILELOCID`` in place of ``WEIGHT_COMP`` where present), in ``zmin < Z < zmax``; the
  randoms are counted at all redshifts;
- the network is the survey pipeline's: ``-ax all --model dnnp --loss pnll --eta_min 1e-5 -k``, 5 folds by 5 chains,
  100 epochs, 3 layers of 10 units in 'N' (learning rate 0.009, batches of 256), 4 of 20 in 'S' (0.007, 1024);
- the weight of a pixel is one over the chain-averaged predicted density, normalised to a mean of one over the fitted
  pixels and clipped to [0.5, 2]; each data object in ``zmin < Z <= zmax`` (sic, the upper bound included, as there)
  takes the weight of its pixel, one where its pixel was not fitted.

The training is deterministic (seed 85 for the folds, a seed per chain): the same table gives bit-identical weights.
It is also chaotic: data counts perturbed by 1e-3 move the weights by 0.9-1.4% rms per pixel, correlation 0.82-0.91 in
``w - 1``. Rebuilt from the official AbacusHF DR2v2 ELG_LOPnotqso catalogs of mock 0, whose weight columns are stored in
float16, the weights match the official ``WEIGHT_SN`` to 0.8-1.2% rms per object, correlation 0.83-0.94, mean ratio 1 to
2e-4: the reproduction is at the level of that sensitivity, and bit-identity cannot be expected from rounded inputs.

The survey pipeline fits its dark-time mocks against the quasar maps, whatever the tracer (``tpmap = 'QSO'`` in
``mkCat_amtl.py``), and the data against the tracer's own: pass the maps accordingly.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np

from .imaging import get_template_maps


logger = logging.getLogger('lsscat.sysnet')

#: The survey pipeline's template maps for the emission line galaxies (``LSS.globals``, ``main('ELG').fit_maps``).
SYSNET_FIT_MAPS = {'ELG': ['STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
                           'EBV_DIFF_GR', 'EBV_DIFF_RZ', 'HI']}
#: All-sky random counts per ring pixel at nside 256, 18 random catalogs, the 'fracgood' denominator.
ALLSKY_RANDOMS_FN = '/dvs_ro/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/allsky_rpix_{region}_nran18_nside256_ring.fits'
#: Training flags common to both regions, then per region (``scripts/sysnetELG_LOPnotqso_zbins.sh``).
SYSNET_TRAIN_FLAGS = ['-ax', 'all', '--model', 'dnnp', '--loss', 'pnll', '--eta_min', '0.00001', '-k']
SYSNET_REGION_FLAGS = {'N': ['-lr', '0.009', '-bs', '256', '--nn_structure', '3', '10', '-ne', '100', '-nc', '5'],
                       'S': ['-lr', '0.007', '-bs', '1024', '--nn_structure', '4', '20', '-ne', '100', '-nc', '5']}


def read_allsky_randoms(region, fn=ALLSKY_RANDOMS_FN):
    """Return the all-sky random count map of photometric region ``region`` ('N' or 'S'), ring ordering, nside 256."""
    import fitsio
    return np.asarray(fitsio.read(str(fn).format(region=region), columns=['RANDS_HPIX'])['RANDS_HPIX'], dtype='f8')


def get_sysnet_data_weights(data, completeness='fracz'):
    """
    Return the weights the data are counted with, as ``LSS.imaging.sysnet_tools.prep4sysnet``: with ``wtmd='fracz'``,
    ``1 / FRACZ_TILELOCID`` (else ``WEIGHT_COMP``) over ``FRAC_TLOBS_TILES``; for the nearest neighbour catalogs
    (``completeness='nn'``, ``wtmd='fraczNN'``), ``WEIGHT_COMP``; times ``WEIGHT_ZFAIL`` where present.
    """
    names = data.dtype.names
    if completeness == 'nn':
        weights = np.array(data['WEIGHT_COMP'], dtype='f8')
    else:
        weights = 1. / np.asarray(data['FRACZ_TILELOCID'], dtype='f8') if 'FRACZ_TILELOCID' in names else np.array(data['WEIGHT_COMP'], dtype='f8')
        weights = weights / np.asarray(data['FRAC_TLOBS_TILES'], dtype='f8')
    if 'WEIGHT_ZFAIL' in names: weights = weights * np.asarray(data['WEIGHT_ZFAIL'], dtype='f8')
    return weights


def _ring_pixels(ra, dec, nside):
    import healpy as hp
    return hp.ang2pix(nside, np.radians(90. - np.asarray(dec, dtype='f8')), np.radians(np.asarray(ra, dtype='f8')))


def prepare_sysnet_table(data, randoms, maps, fit_maps, zrange, allsky_randoms, data_weights=None, ebv_diff=None, nside=256):
    """
    Return the SYSNet input table of one photometric region and redshift bin, as ``prep4sysnet`` builds it.

    Parameters
    ----------
    data, randoms : tables
        Data and randoms of the region, with 'RA', 'DEC'; the data also 'Z' and, unless ``data_weights`` is given, the
        completeness columns of :func:`get_sysnet_data_weights`.
    maps : structured array
        Observing condition maps of the region's imaging survey, nested ordering (:func:`~mockfactory.desi.lsscat.read_hpmaps`).
    fit_maps : list
        Template names, the order of the 'features' columns.
    zrange : tuple
        (zmin, zmax), both excluded.
    allsky_randoms : array
        All-sky random counts per ring pixel, see :func:`read_allsky_randoms`.

    Returns
    -------
    table : structured array
        'features' (npix, len(fit_maps)), 'label', 'fracgood', 'hpix' (ring), in increasing 'hpix' order.
    """
    import healpy as hp
    if data_weights is None: data_weights = get_sysnet_data_weights(data)
    z = np.asarray(data['Z'])
    sel = (z > zrange[0]) & (z < zrange[1])
    npix = hp.nside2npix(nside)
    label = np.bincount(_ring_pixels(data['RA'][sel], data['DEC'][sel], nside), weights=np.asarray(data_weights)[sel], minlength=npix)
    randoms_counts = np.bincount(_ring_pixels(randoms['RA'], randoms['DEC'], nside), minlength=npix).astype('f8')
    with np.errstate(divide='ignore', invalid='ignore'):
        fracgood = randoms_counts / np.asarray(allsky_randoms, dtype='f8')
    templates = get_template_maps(maps, fit_maps, ebv_diff=ebv_diff)
    features = np.column_stack([hp.reorder(templates[name], n2r=True) for name in fit_maps])
    mask = (fracgood > 0.) & np.all(features != hp.UNSEEN, axis=1)
    if not np.all(np.isfinite(fracgood[mask])):
        # randoms in a pixel the all-sky map has none in: the survey pipeline keeps them, with an infinite fraction
        logger.warning('{:d} pixels with randoms but no all-sky randoms; kept, as the survey pipeline does.'.format(np.sum(~np.isfinite(fracgood[mask]))))
    hpix = np.flatnonzero(mask)
    table = np.zeros(hpix.size, dtype=[('features', 'f8', (len(fit_maps),)), ('label', 'f8'), ('fracgood', 'f8'), ('hpix', 'i8')])
    table['features'] = features[hpix]
    table['label'] = label[hpix]
    table['fracgood'] = fracgood[hpix]
    table['hpix'] = hpix
    return table


def _sysnet_config(argv):
    # sysnet takes its configuration from the command line only (argparse, sys.argv): give it the survey pipeline's
    # flags there, so that every default is the one the survey pipeline runs with.
    from sysnet import parse_cmd_arguments
    saved = sys.argv
    try:
        sys.argv = ['sysnet'] + [str(arg) for arg in argv]
        return parse_cmd_arguments()
    finally:
        sys.argv = saved


def run_sysnet(table, output_dir, region, flags=None, mpicomm=None):
    """
    Train SYSNet on ``table`` (:func:`prepare_sysnet_table`) and return the fitted pixels and their predictions.

    The input table, the trained models and 'nn-weights.fits' are written to ``output_dir``. With ``mpicomm`` of more than
    one rank, the 25 fold-and-chain trainings are spread over the ranks (``SYSNetMPI``, which works on the world
    communicator), as the survey pipeline runs it with ``srun -n 25``; otherwise they run one after the other, with the same
    seeds and the same result.

    Returns
    -------
    hpix : array
        Fitted ring pixels.
    prediction : array
        Predicted density per unit 'fracgood', shape (npix, nchains).
    """
    import fitsio
    output_dir = Path(output_dir)
    rank = 0 if mpicomm is None else mpicomm.rank
    input_fn = output_dir / 'prep.fits'
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        fitsio.write(str(input_fn), table, clobber=True)
    if mpicomm is not None: mpicomm.Barrier()
    flags = list(SYSNET_TRAIN_FLAGS) + list(SYSNET_REGION_FLAGS[region] if flags is None else flags)
    config = _sysnet_config(flags + ['-i', str(input_fn), '-o', str(output_dir)])
    from sysnet import SYSNet, SYSNetMPI
    if mpicomm is not None and mpicomm.size > 1:
        SYSNetMPI(config).run()
    else:
        SYSNet(config).run()
    if mpicomm is not None: mpicomm.Barrier()
    weights = fitsio.read(str(output_dir / 'nn-weights.fits'))
    return np.asarray(weights['hpix']), np.asarray(weights['weight'], dtype='f8').reshape(len(weights), -1)


def get_sysnet_pixel_weights(hpix, prediction, nside=256, clip=(0.5, 2.)):
    """
    Return the healpix map (ring) of SYSNet weights: one over the chain-averaged prediction, normalised to a mean of one
    over the fitted pixels and clipped to ``clip``; one in the pixels not fitted.
    """
    import healpy as hp
    weights = 1. / np.mean(prediction, axis=1)
    weights = np.clip(weights / weights.mean(), *clip)
    toret = np.ones(hp.nside2npix(nside), dtype='f8')
    toret[hpix] = weights
    return toret


def compute_sysnet_weights(data, randoms, maps_north, maps_south, fit_maps, zranges, output_dir, regions=('N', 'S'),
                           data_weights=None, allsky_randoms=None, ebv_diff=None, flags=None, nside=256, mpicomm=None):
    """
    Return the SYSNet imaging weights of the data, fitted in each photometric region and redshift bin, and the pixel
    weight maps.

    Parameters
    ----------
    data : table
        Clustering data of both caps, with 'RA', 'DEC', 'Z', 'PHOTSYS' and the completeness columns of
        :func:`get_sysnet_data_weights`, unless ``data_weights`` is given.
    randoms : table
        Clustering randoms of both caps, all random catalogs, with 'RA', 'DEC', 'PHOTSYS'.
    maps_north, maps_south : structured arrays
        Observing condition maps (:func:`~mockfactory.desi.lsscat.read_hpmaps`); the survey pipeline's mocks use the
        quasar maps.
    fit_maps : list
        Template names, e.g. :data:`SYSNET_FIT_MAPS` ['ELG'].
    zranges : list
        Redshift bins, each fitted on its own; the survey pipeline's emission line galaxies take (0.8, 1.1), (1.1, 1.6).
    output_dir : str, Path
        Where each region and bin writes its input table, models and predictions, in '{region}_{zmin}_{zmax}'.
    regions : tuple, default=('N', 'S')
        Photometric regions, by 'PHOTSYS'.
    allsky_randoms : dict, default=None
        All-sky random count maps per region; read with :func:`read_allsky_randoms` when not given.
    flags : dict, default=None
        Per-region training flags replacing :data:`SYSNET_REGION_FLAGS`.

    mpicomm : MPI communicator, default=None
        With more than one rank, the trainings are spread over the ranks (run under ``srun -n 25`` as the survey
        pipeline does); only rank 0 needs ``data``, ``randoms`` and the maps, the others may pass ``None``, and only
        rank 0 returns weights.

    Returns
    -------
    weights : array
        Imaging weights of the data, one outside the regions and bins; ``None`` on ranks other than 0.
    pixel_weights : dict
        Healpix weight maps (ring), ``{(region, zrange): map}``.
    """
    root = mpicomm is None or mpicomm.rank == 0
    if allsky_randoms is None: allsky_randoms = {}
    if root:
        if data_weights is None: data_weights = get_sysnet_data_weights(data)
        data_photsys, randoms_photsys = np.asarray(data['PHOTSYS']).astype('U1'), np.asarray(randoms['PHOTSYS']).astype('U1')
        data_pix = _ring_pixels(data['RA'], data['DEC'], nside)
        z = np.asarray(data['Z'])
        weights = np.ones(len(data), dtype='f8')
    pixel_weights = {}
    for zrange in zranges:
        zrange = tuple(zrange)
        for region in regions:
            table = None
            if root:
                if region not in allsky_randoms: allsky_randoms[region] = read_allsky_randoms(region)
                in_data, in_randoms = data_photsys == region, randoms_photsys == region
                table = prepare_sysnet_table(data[in_data], randoms[in_randoms], maps_north if region == 'N' else maps_south,
                                             fit_maps, zrange, allsky_randoms[region], data_weights=np.asarray(data_weights)[in_data],
                                             ebv_diff=ebv_diff, nside=nside)
                logger.info('SYSNet in region {}, {} < z < {}: {:d} pixels, {:.0f} weighted data.'.format(region, *zrange, table.size, table['label'].sum()))
            hpix, prediction = run_sysnet(table, Path(output_dir) / '{}_{}_{}'.format(region, *zrange), region,
                                          flags=None if flags is None else flags.get(region), mpicomm=mpicomm)
            pixel_weights[region, zrange] = hpmap = get_sysnet_pixel_weights(hpix, prediction, nside=nside)
            if root:
                sel = in_data & (z > zrange[0]) & (z <= zrange[1])
                weights[sel] = hpmap[data_pix[sel]]
    return (weights if root else None), pixel_weights

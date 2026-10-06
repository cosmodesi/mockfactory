"""
Imaging systematic weights, from a linear regression of the data density against maps of the imaging conditions.

This is the regression of ``desihub/LSS`` (``LSS.imaging.systematics_linear_regression``, itself taken from
`alaeboss <https://github.com/domichbt/alaeboss>`_, "a la eBOSS"), re-implemented without importing ``LSS``:

- the value of each template map is read at the position of every data and random object, the map of the imaging
  survey the object belongs to (north or south);
- per template, the objects in the extreme ``tail`` percent of the data distribution are set aside, and the data range
  that is left is cut into ``nbins`` bins;
- in each bin of each template, the weighted data count over the weighted random count, normalised to one on average,
  is the density contrast the imaging conditions imprint;
- the data are given the weight ``1 / (1 + c0 + sum_i c_i x_i)``, with ``x_i`` the template value mapped onto [0, 1]
  by the bin edges, and the coefficients minimise the chi2 of the weighted density contrast against one, over all bins
  of all templates. The error of each bin is fixed by the unweighted counts.

The survey pipeline minimises the chi2 with jax; here it is numpy, with the analytic gradient, and the data grouped by
the combination of their bins in all templates, so that each evaluation is one bincount over the data. Same coefficients
to 1e-10, the fit 3x faster than the jit-compiled jax version on 1.8M data and 4M randoms.

The fit is done separately in each photometric region and redshift bin, with the coefficients of that fit, and the
weights are then given to every data object of that region and redshift bin, including the ones set aside as outliers.

The survey pipeline runs it on the clustering catalogs with the weights ``WEIGHT * WEIGHT_FKP / WEIGHT_SYS``, so that it
can be re-run on catalogs that already carry imaging weights, the randoms also divided by ``WEIGHT_ZFAIL`` (which they
take from their data donor; ``LSS.imaging.systematics_linear_regression.produce_imweights``):
:func:`compute_imaging_weights` does the same by default.
"""

import logging

import numpy as np

from .utils import get_photsys


logger = logging.getLogger('lsscat.imaging')

#: Extinction coefficients A / E(B-V) of the Legacy Surveys bands, used to correct the depths for Galactic extinction.
EXT_COEFF = {'G': 3.214, 'R': 2.165, 'Z': 1.211, 'W1': 0.184, 'W2': 0.113}

#: The DESI DR2 stellar reddening map, whose difference to SFD gives the 'EBV_DIFF_GR' and 'EBV_DIFF_RZ' templates.
EBV_FN = '/dvs_ro/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars_y3/v0.1/final_maps/lss/desi_ebv_lss_256.fits'

#: Templates fitted for each tracer by the survey pipeline on the mocks (``LSS.globals.main(tracer).fit_maps``;
#: the LRG take ``fit_maps_allebv``).
FIT_MAPS = {'BGS': ['STARDENS', 'GALDEPTH_R', 'HI'],
            'LRG': ['STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'HI',
                    'PSFDEPTH_W1', 'EBV_DIFF_GR', 'EBV_DIFF_RZ'],
            'ELG': ['STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z',
                    'EBV_DIFF_GR', 'EBV_DIFF_RZ', 'HI'],
            'QSO': ['PSFDEPTH_W1', 'PSFDEPTH_W2', 'STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'PSFDEPTH_G',
                    'PSFDEPTH_R', 'PSFDEPTH_Z', 'EBV_DIFF_GR', 'EBV_DIFF_RZ', 'HI']}


def read_ebv_diff(fn=EBV_FN):
    """
    Return the 'EBV_DIFF_GR' and 'EBV_DIFF_RZ' maps, DESI minus SFD reddening, in nested healpix ordering
    (the file is in ring ordering).
    """
    import fitsio
    import healpy as hp
    maps = fitsio.read(str(fn))
    return {'EBV_DIFF_' + color: hp.reorder(maps['EBV_DESI_' + color] - maps['EBV_SFD_' + color], r2n=True)
            for color in ['GR', 'RZ']}


def get_template_maps(maps, names, ebv_diff=None):
    """
    Return a dictionary of the healpix maps ``names``, taken from ``maps``, one imaging survey's observing condition maps
    (as returned by :func:`~mockfactory.desi.lsscat.read_hpmaps`).

    Depths ('DEPTH' in the name, the band last) are corrected for Galactic extinction with the 'EBV' map, as the survey
    pipeline does before fitting: ``depth * 10**(-0.4 * A_band / E(B-V) * EBV)``. 'EBV_DIFF_GR' and 'EBV_DIFF_RZ' come
    from ``ebv_diff`` (:func:`read_ebv_diff`), which is read when needed and not given.
    """
    toret = {}
    for name in names:
        if name.startswith('EBV_DIFF_'):
            if ebv_diff is None: ebv_diff = read_ebv_diff()
            toret[name] = np.asarray(ebv_diff[name], dtype='f8')
        elif 'DEPTH' in name:
            band = name.split('_')[-1]
            toret[name] = np.asarray(maps[name], dtype='f8') * 10**(-0.4 * EXT_COEFF[band] * np.asarray(maps['EBV'], dtype='f8'))
        else:
            toret[name] = np.asarray(maps[name], dtype='f8')
    return toret


def get_template_values(ra, dec, maps, names, nside=256, nest=True):
    """
    Return the values of the templates ``names`` at positions ``ra``, ``dec`` (degrees), an array of shape
    (len(names), len(ra)). ``maps`` is a dictionary of healpix maps, see :func:`get_template_maps`.
    """
    import healpy as hp
    pix = hp.ang2pix(nside, np.radians(90. - np.asarray(dec)), np.radians(np.asarray(ra)), nest=nest)
    return np.vstack([maps[name][pix] for name in names])


def select_photometric_region(ra, dec, region, photsys=None):
    """
    Return the mask of positions in the photometric ``region``: 'N' (BASS/MzLS), 'S' (DECaLS, DES included), and for the
    quasars 'DES' and 'SnotDES', the DES footprint being that of ``regressis``. ``photsys`` defaults to
    :func:`~mockfactory.desi.lsscat.get_photsys` of the positions.
    """
    if photsys is None: photsys = get_photsys(ra, dec)
    photsys = np.asarray(photsys)
    if photsys.dtype.kind in 'SU' and photsys.dtype.itemsize > (1 if photsys.dtype.kind == 'S' else 4): photsys = photsys.astype(photsys.dtype.kind + '1')
    # bytes, as read from a fits or hdf5 catalog, compared as such: casting a few 1e8 of them to str takes tens of seconds
    north, south = (b'N', b'S') if photsys.dtype.kind == 'S' else ('N', 'S')
    if region in ('N', 'S'):
        return photsys == (north if region == 'N' else south)
    if region in ('DES', 'SnotDES'):
        import healpy as hp
        from regressis import footprint
        des = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False).get_imaging_surveys()[2]
        is_des = des[hp.ang2pix(256, np.radians(90. - np.asarray(dec)), np.radians(np.asarray(ra)), nest=True)]
        return is_des if region == 'DES' else (photsys == south) & ~is_des
    raise ValueError('unknown photometric region {}'.format(region))


class LinearRegression(object):
    """
    Linear regression of the data density against imaging templates, see the module documentation.

    >>> regression = LinearRegression(data_values, data_weights, randoms_values, randoms_weights, nbins=10, tail=0.5)
    >>> coefficients = regression.fit()
    >>> weights = regression.weights(data_values)
    """

    bin_margin = 1e-7

    def __init__(self, data_values, data_weights, randoms_values, randoms_weights, nbins=10, tail=0.5, names=None):
        """
        Set up the regression.

        Parameters
        ----------
        data_values : array of shape (ntemplates, ndata)
            Template values for the data.
        data_weights : array of shape (ndata,)
            Data weights, other than the imaging ones.
        randoms_values : array of shape (ntemplates, nrandoms)
            Template values for the randoms.
        randoms_weights : array of shape (nrandoms,)
            Random weights.
        nbins : int, default=10
            Number of bins per template.
        tail : float, default=0.5
            Percentage of the data distribution, split equally between the two tails, set aside for each template.
            Data and randoms with any template value beyond these bounds are left out of the fit.
        names : list, default=None
            Template names, for the coefficient dictionary :meth:`fit` returns.
        """
        data_values, randoms_values = np.atleast_2d(data_values).astype('f8'), np.atleast_2d(randoms_values).astype('f8')
        data_weights, randoms_weights = np.asarray(data_weights, dtype='f8'), np.asarray(randoms_weights, dtype='f8')
        self.names = list(names) if names is not None else ['template{:d}'.format(i) for i in range(len(data_values))]
        self.nbins = int(nbins)
        # objects with an undefined template value
        good_data = np.all(np.isfinite(data_values), axis=0)
        good_randoms = np.all(np.isfinite(randoms_values), axis=0)
        data_values, data_weights = data_values[:, good_data], data_weights[good_data]
        randoms_values, randoms_weights = randoms_values[:, good_randoms], randoms_weights[good_randoms]
        # outliers: bounds from the data distribution, applied to the data and the randoms
        low = np.percentile(data_values, tail / 2., axis=1)
        high = np.percentile(data_values, 100. - tail / 2., axis=1)
        inside = lambda values: np.all((values >= low[:, None]) & (values <= high[:, None]), axis=0)
        keep_data, keep_randoms = inside(data_values), inside(randoms_values)
        data_values, data_weights = data_values[:, keep_data], data_weights[keep_data]
        randoms_values, randoms_weights = randoms_values[:, keep_randoms], randoms_weights[keep_randoms]
        logger.info('{:d} data and {:d} randoms in the fit, out of {:d} and {:d}.'.format(
                    data_weights.size, randoms_weights.size, good_data.size, good_randoms.size))
        self.normalization = randoms_weights.sum() / data_weights.sum()
        # bins, spanning the data range left
        self.edges = np.linspace(data_values.min(axis=1) - self.bin_margin, data_values.max(axis=1) + self.bin_margin, self.nbins + 1, axis=1)
        ntemplates, nb = len(data_values), self.nbins + 2

        def digitize(values):
            # bin index, 1 to nbins inside the edges, 0 and nbins + 1 outside, offset by nbins + 2 per template so that all
            # templates share one bincount; a value on the last edge goes to the last bin
            index = np.floor(self.normalize(values) * self.nbins).astype('i8') + 1 - (values == self.edges[:, -1:])
            return np.clip(index, 0, nb - 1) + nb * np.arange(ntemplates)[:, None]

        def bincount(index, weights):
            # weighted count in each bin of each template, flattened, the bins outside the edges dropped
            return np.bincount(index.ravel(), weights=np.tile(weights, len(index)), minlength=ntemplates * nb).reshape(ntemplates, nb)[:, 1:-1].ravel()

        self._randoms_binned = bincount(digitize(randoms_values), randoms_weights)
        # the data, all inside the edges, are grouped by cell, the combination of their bins in all templates: one bincount
        # over the data gives the weighted count per cell, and a small one over the cells the count per bin
        data_bin = np.floor(self.normalize(data_values) * self.nbins).astype('i8')  # 0 to nbins - 1
        cells, self._data_cell = np.unique(np.dot(self.nbins**np.arange(ntemplates), data_bin), return_inverse=True)
        # bin of each cell in each template, in the flattened layout
        self._cell_bin = cells // self.nbins**np.arange(ntemplates)[:, None] % self.nbins + self.nbins * np.arange(ntemplates)[:, None]
        self._data_normalized = self.normalize(data_values)
        self._data_weights = data_weights
        data_binned = self._binned_data(1.)
        # a bin with no randoms has no density contrast: it is given the error of an empty bin and a random count of one, so
        # that it weighs nothing instead of making the chi2 0 / 0
        empty = (data_binned == 0) | (self._randoms_binned == 0)
        self._randoms_binned = np.where(self._randoms_binned == 0, 1., self._randoms_binned)
        # the error of each bin is fixed by the unweighted counts
        error = self.normalization * np.sqrt(data_binned / self._randoms_binned**2 + data_binned**2 / self._randoms_binned**3)
        self._error = np.where(empty, 1e10, error)
        self._last = None
        self.coefficients = None

    def normalize(self, values):
        """Return the template ``values`` mapped onto [0, 1] by the bin edges."""
        return (np.atleast_2d(values) - self.edges[:, :1]) / (self.edges[:, -1:] - self.edges[:, :1])

    def _binned_data(self, weights):
        cells = np.bincount(self._data_cell, weights=self._data_weights * weights, minlength=self._cell_bin.shape[1])
        return np.bincount(self._cell_bin.ravel(), weights=np.tile(cells, len(self._cell_bin)), minlength=self._randoms_binned.size)

    def _residual(self, coefficients):
        # imaging weight of each data object, and density contrast minus one in each bin; kept for the next call, as migrad
        # asks for chi2 and grad at the same coefficients
        coefficients = np.array(coefficients, dtype='f8')
        if self._last is None or not np.array_equal(self._last[0], coefficients):
            model = 1. / (1. + coefficients[0] + coefficients[1:].dot(self._data_normalized))
            self._last = (coefficients, model, self.normalization * self._binned_data(model) / self._randoms_binned - 1.)
        return self._last[1:]

    def chi2(self, coefficients):
        """chi2 of the weighted density contrast against one, over all bins of all templates."""
        residual = self._residual(coefficients)[1]
        return np.sum(residual**2 / self._error**2)

    def grad(self, coefficients):
        """
        Gradient of :meth:`chi2`. With ``m = 1 / (1 + c0 + c.x)`` the weight of a data object, ``dchi2 / dm`` is its data
        weight times the sum, over the bins it falls in, of ``2 residual normalization / (randoms count * error**2)``,
        and ``dm / dc = -m**2 (1, x)``.
        """
        model, residual = self._residual(coefficients)
        dbin = 2. * residual * self.normalization / (self._randoms_binned * self._error**2)
        dmodel = -dbin[self._cell_bin].sum(axis=0)[self._data_cell] * self._data_weights * model**2
        return np.concatenate([[dmodel.sum()], self._data_normalized.dot(dmodel)])

    def density(self, coefficients=None):
        """Return the bin centers and the density contrast, weighted with ``coefficients`` (``None`` for unweighted), and its error."""
        weights = 1. if coefficients is None else self._residual(coefficients)[0]
        shape = (len(self.names), self.nbins)
        centers = (self.edges[:, :-1] + self.edges[:, 1:]) / 2.
        return centers, (self.normalization * self._binned_data(weights) / self._randoms_binned).reshape(shape), self._error.reshape(shape)

    def fit(self, guess=None):
        """
        Minimise :meth:`chi2` with ``iminuit``'s migrad and the analytic gradient :meth:`grad`, from ``guess`` (zeros by
        default), and return the coefficients as a dictionary, the constant under 'constant'.
        """
        from iminuit import Minuit
        if guess is None: guess = np.zeros(len(self.names) + 1, dtype='f8')
        minuit = Minuit(self.chi2, np.asarray(guess, dtype='f8'), grad=self.grad)
        minuit.errordef = Minuit.LEAST_SQUARES
        minuit.errors = [0.1] * len(guess)
        minuit.migrad()
        if not minuit.valid:
            raise RuntimeError('the imaging regression did not converge:\n{}'.format(minuit.fmin))
        self.coefficients = np.array(minuit.values)
        logger.info('chi2 {:.2f} -> {:.2f} over {:d} bins.'.format(self.chi2(np.zeros_like(guess)), minuit.fval, self.nbins * len(self.names)))
        return dict(zip(['constant'] + self.names, self.coefficients.tolist()))

    def weights(self, values, coefficients=None):
        """
        Return the imaging weights for template ``values`` (any objects, outliers included), with the fitted coefficients
        by default; one for objects with an undefined template value.
        """
        coefficients = self.coefficients if coefficients is None else np.asarray(coefficients)
        values = np.atleast_2d(values)
        good = np.all(np.isfinite(values), axis=0)
        toret = np.ones(values.shape[1], dtype='f8')
        toret[good] = 1. / (1. + coefficients[0] + np.dot(coefficients[1:], self.normalize(values[:, good])))
        return toret


def compute_imaging_weights(data, randoms, maps_north, maps_south, fit_maps, zranges, regions=('S', 'N'), nbins=10, tail=0.5,
                            data_weights=None, randoms_weights=None, ebv_diff=None, nside=256, nest=True):
    """
    Return the imaging weights of the data, fitted in each photometric region and redshift bin, and the fitted coefficients.

    Parameters
    ----------
    data : table
        Data, with 'RA', 'DEC', 'Z', and, unless ``data_weights`` is given, 'WEIGHT' (and if present 'WEIGHT_FKP',
        'WEIGHT_SYS'). 'PHOTSYS' is used if present, else computed from the positions.
    randoms : table
        Randoms, with the same columns.
    maps_north, maps_south : structured arrays
        Observing condition maps of the two imaging surveys, as returned by :func:`~mockfactory.desi.lsscat.read_hpmaps`.
    fit_maps : list
        Template names; see :data:`FIT_MAPS` for the survey pipeline's.
    zranges : list
        Redshift bins (zmin, zmax), each fitted on its own; the bounds are excluded, as in the survey pipeline.
    regions : tuple, default=('S', 'N')
        Photometric regions, see :func:`select_photometric_region`; the quasars take ('DES', 'SnotDES', 'N').
    nbins, tail : see :class:`LinearRegression`.
    data_weights, randoms_weights : arrays, default=None
        Weights for the fit; by default ``WEIGHT * WEIGHT_FKP / WEIGHT_SYS``, the randoms also divided by ``WEIGHT_ZFAIL``
        if they carry it, as the survey pipeline does with the clustering catalogs.
    ebv_diff : dict, default=None
        'EBV_DIFF_GR' and 'EBV_DIFF_RZ' maps, see :func:`read_ebv_diff`; read if needed and not given.

    Returns
    -------
    weights : array
        Imaging weights, one for the data in none of the regions and redshift bins.
    coefficients : dict
        Fitted coefficients, ``{(region, zrange): {name: value}}``.
    """
    weights = []
    for kind, catalog, catalog_weights in [('data', data, data_weights), ('randoms', randoms, randoms_weights)]:
        if catalog_weights is None:
            # WEIGHT * WEIGHT_FKP / WEIGHT_SYS, the weights the survey pipeline regresses the clustering catalogs with; it
            # divides the data by WEIGHT_ZFAIL and multiplies it back, and the randoms only divides
            catalog_weights = np.asarray(catalog['WEIGHT'], dtype='f8')
            if 'WEIGHT_FKP' in catalog.dtype.names: catalog_weights = catalog_weights * catalog['WEIGHT_FKP']
            if 'WEIGHT_SYS' in catalog.dtype.names: catalog_weights = catalog_weights / catalog['WEIGHT_SYS']
            if kind == 'randoms' and 'WEIGHT_ZFAIL' in catalog.dtype.names: catalog_weights = catalog_weights / catalog['WEIGHT_ZFAIL']
        weights.append(np.asarray(catalog_weights, dtype='f8'))
    data_weights, randoms_weights = weights
    if any(name.startswith('EBV_DIFF_') for name in fit_maps) and ebv_diff is None: ebv_diff = read_ebv_diff()
    photsys = {name: (np.asarray(catalog['PHOTSYS']) if 'PHOTSYS' in catalog.dtype.names else None)
               for name, catalog in [('data', data), ('randoms', randoms)]}
    import healpy as hp
    pix = {name: hp.ang2pix(nside, np.radians(90. - np.asarray(catalog['DEC'])), np.radians(np.asarray(catalog['RA'])), nest=nest)
           for name, catalog in [('data', data), ('randoms', randoms)]}
    weights, coefficients = np.ones(len(data), dtype='f8'), {}
    for region in regions:
        maps = get_template_maps(maps_north if region == 'N' else maps_south, fit_maps, ebv_diff=ebv_diff)
        in_data = select_photometric_region(data['RA'], data['DEC'], region, photsys=photsys['data'])
        in_randoms = select_photometric_region(randoms['RA'], randoms['DEC'], region, photsys=photsys['randoms'])
        for zrange in zranges:
            zrange = tuple(zrange)
            sel_data = in_data & (data['Z'] > zrange[0]) & (data['Z'] < zrange[1])
            sel_randoms = in_randoms & (randoms['Z'] > zrange[0]) & (randoms['Z'] < zrange[1])
            data_values = np.vstack([maps[name][pix['data'][sel_data]] for name in fit_maps])
            # the randoms of a pixel share its template values: they enter the regression as their weighted count per pixel
            randoms_counts = np.bincount(pix['randoms'][sel_randoms], weights=randoms_weights[sel_randoms], minlength=hp.nside2npix(nside))
            pixels = np.flatnonzero(randoms_counts)
            logger.info('Fitting region {} and {} < z < {}, {:d} randoms in {:d} pixels.'.format(region, *zrange, sel_randoms.sum(), pixels.size))
            regression = LinearRegression(data_values, data_weights[sel_data], np.vstack([maps[name][pixels] for name in fit_maps]),
                                          randoms_counts[pixels], nbins=nbins, tail=tail, names=fit_maps)
            coefficients[region, zrange] = regression.fit()
            weights[sel_data] = regression.weights(data_values)
    return weights, coefficients

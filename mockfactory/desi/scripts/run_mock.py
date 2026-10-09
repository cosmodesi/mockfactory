"""
One mock from a cubic box to clustering catalogs, every stage in one place.

    cubic box -> cutsky -> targets -> altmtl -> pota -> lsscat -> imsys

The stages are separate because they fail separately and because the middle ones cost hours, so
a run is normally restarted somewhere in the middle. Each writes its output and the next reads
it, so ``--stages`` picks up wherever the last one stopped.

    salloc -N 1 -C cpu -q interactive -t 04:00:00 -A desi
    srun -n 64 python run_mock.py --stages cutsky,targets --imocks 0 --output-dir $SCRATCH/mock
    srun -n 1  python run_mock.py --stages altmtl,pota,lsscat,imsys --imocks 0 --output-dir $SCRATCH/mock

The first two run over MPI, since their cost is reading bricks and positions; `altmtl` and
`lsscat` run in one process with forked pools and will refuse more than one rank, as do `pota`
and `imsys`, the imaging weights fitted on the clustering catalogs and written back into them.

`--tracer` is the only thing a run needs to change between programs: it picks the boxes, the
snapshots, the redshift range, the measured n(z), the tile file, the observing conditions and
the imaging bits out of `PROGRAMS`. `BGS_BRIGHT` is bright, `LRG`, `ELG_LOP` and `QSO` are dark.

A light cone is stitched from as many snapshots as the tracer has, each covering the shell out
to the midpoint between it and its neighbours: one for the bright galaxies over 0.1 < z < 0.4,
three for the luminous red galaxies over 0.4 < z < 1.1, five for the quasars.
"""

import argparse
import logging
import time
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger('run_mock')

DESI_DIR = Path('/dvs_ro/cfs/cdirs/desi')
LSS_DIR = DESI_DIR / 'survey/catalogs/DA2/LSS'
#: AbacusSummit high fidelity v2.0 cubic boxes.
BOX_DIR = DESI_DIR / 'mocks/cai/abacus_HF/DR2_v2.0'
#: The HOD of the boxes, unless the tracer names its own in PROGRAMS: the quasar boxes come in 'base' only.
BOX_COSMO, BOX_HOD = '000', 'base_B'
#: The measured redshift distributions the cutsky is downsampled to, {tracer}_{region}_nz.txt.
NZ_DIR = LSS_DIR / 'loa-v1/LSScats/v2/nonKP'

#: What each tracer needs, and nothing a tracer does not. ``box`` names the box directory and
#: ``snapshots`` the redshift of each, which the light cone is stitched from: a snapshot covers
#: the shell out to the midpoint between it and its neighbours, cut to ``zrange``. Bright has
#: one snapshot and so one shell; the dark tracers have three or more. ``nz`` is the tracer the
#: measured n(z) is read for, which is not always the one being built. ``imsys_zranges`` are the
#: redshift bins the imaging weights are fitted in, each on its own, and ``imsys_regions`` the
#: photometric regions: the survey pipeline's for its mocks (``mkCat_amtl.py``, as its DA2 mock
#: drivers call it), 'split' bins for the quasars, 'fine' (0.1 wide) for the luminous red
#: galaxies, the bright galaxies over their clustering range. The emission line galaxies are
#: weighted with SYSNet instead (``imsys_method``), against the quasar maps (``imsys_maps``), as
#: the survey pipeline does for its dark-time mocks.
PROGRAMS = {
    'BGS_BRIGHT': dict(box='BGS-21.35', snapshots=(0.300,), zrange=(0.1, 0.4), obscon='bright',
                       nz='BGS_BRIGHT-21.35', imsys_zranges=((0.1, 0.4),), imsys_regions=('S', 'N')),
    'LRG': dict(box='LRG', snapshots=(0.500, 0.725, 0.950), zrange=(0.4, 1.1), obscon='dark',
                nz='LRG', imsys_zranges=((0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1)),
                imsys_regions=('S', 'N')),
    'ELG_LOP': dict(box='ELG', snapshots=(0.950, 1.175, 1.475), zrange=(0.8, 1.6), obscon='dark',
                    nz='ELG_LOPnotqso', imsys_zranges=((0.8, 1.1), (1.1, 1.6)), imsys_regions=('N', 'S'),
                    imsys_method='sysnet', imsys_maps='QSO'),
    'QSO': dict(box='QSO', snapshots=(0.950, 1.250, 1.400, 1.550, 1.850), zrange=(0.8, 2.1),
                obscon='dark', nz='QSO', imsys_zranges=((0.8, 1.3), (1.3, 2.1), (2.1, 3.5)),
                imsys_regions=('DES', 'SnotDES', 'N'), hod='base'),
}
#: Observing condition maps the imaging weights are fitted against, the survey's own.
HPMAP_DIR = LSS_DIR / 'loa-v1/LSScats/v2/hpmaps'
#: DR9 brick pixel maskbits, filled in per region and brick by get_brick_pixel_quantities.
MASKBITS_FN = ('/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/{region}/coadd/{brickname:.3s}/'
               '{brickname}/legacysurvey-{brickname}-maskbits.fits.fz')
#: Exposure count per band, which the clustering veto reads as NOBS_G, NOBS_R, NOBS_Z.
NEXP_FN = ('/dvs_ro/cfs/cdirs/cosmo/data/legacysurvey/dr9/{{region}}/coadd/{{brickname:.3s}}/'
           '{{brickname}}/legacysurvey-{{brickname}}-nexp-{band}.fits.fz')
#: Imaging bits the targets are vetoed on, and which the randoms must then be vetoed on too.
#: Bit 11 is the bright galaxy sample's own; the dark tracers carry 12 and 13 instead.
VETO_BITS = {'bright': (1, 5, 6, 7, 10, 11, 13), 'dark': (1, 5, 6, 7, 10, 12, 13)}
#: The randoms' potential assignments (rancomb_{i}{program}wdupspec_zdone.fits), and where the survey
#: got a usable spectrum (datcomb_{program}_spec_zdone.fits). Both are survey products: only PRIORITY
#: comes from the mock, which combine_randoms puts there.
LOA_DIR = LSS_DIR / 'loa-v1'
#: Only the columns the catalog stage reads: at 89 million rows a full read is 7.2 GB a catalog.
RANDOM_COLUMNS = ('TARGETID', 'LOCATION', 'FIBER', 'TILEID', 'RA', 'DEC', 'PRIORITY')
STAGES = ('cutsky', 'targets', 'altmtl', 'pota', 'lsscat', 'imsys')


def parse_imocks(text):
    """``'0'``, ``'0,3'`` or ``'0-11'``, and any comma-separated mixture of the three."""
    out = []
    for part in text.split(','):
        if '-' in part:
            first, last = part.split('-')
            out += list(range(int(first), int(last) + 1))
        else:
            out.append(int(part))
    return out


def box_fn(imock, box, zsnap, hod=BOX_HOD):
    name = 'abacus_HF_{t}_{z}_DR2_v2.0_AbacusSummit_base_c{c}_ph{i:03d}_{h}_clustering.dat.h5'
    return Path(BOX_DIR) / 'AbacusSummit_base_c{c}_ph{i:03d}'.format(c=BOX_COSMO, i=imock) / 'Boxes' / box / name.format(t=box, z='{:.3f}'.format(zsnap).replace('.', 'p'),
                                    c=BOX_COSMO, i=imock, h=hod)


def get_shells(snapshots, zrange):
    """
    Redshift range each snapshot covers, the boundaries falling midway between them.

    One snapshot covers the whole range; several split it, so that every galaxy comes from the
    snapshot nearest in redshift. Returns ``[(zsnap, (zmin, zmax)), ...]``.
    """
    snapshots = sorted(snapshots)
    edges = [zrange[0]] + [0.5 * (a + b) for a, b in zip(snapshots[:-1], snapshots[1:])] + [zrange[1]]
    return [(zsnap, (edges[i], edges[i + 1])) for i, zsnap in enumerate(snapshots)
            if edges[i + 1] > edges[i]]


def cutsky_fn(output_dir, imock):
    return Path(output_dir) / 'cutsky{:d}.fits'.format(imock)


def targets_fn(output_dir, imock):
    # .h5: write_targets writes HDF5, so that every rank puts its own slice in rather than
    # gathering the catalog onto one rank, and read_targets dispatches on the extension.
    return Path(output_dir) / 'forFA{:d}.h5'.format(imock)


def altmtl_dir(output_dir, imock):
    return Path(output_dir) / 'altmtl{:d}'.format(imock) / 'Univ000'


def pota_fn(output_dir, imock):
    return Path(output_dir) / 'altmtl{:d}'.format(imock) / 'pota.fits'


def read_box(imock, box, zsnap, mpicomm, hod=BOX_HOD):
    """Read one cubic box as a :class:`BoxCatalog`, velocities already in position units."""
    from mockfactory import BoxCatalog, Catalog
    fn = box_fn(imock, box, zsnap, hod=hod)
    logger.info('Reading {}.'.format(fn))
    catalog = Catalog.read(fn, filetype='hdf5', group='/', mpicomm=mpicomm)
    attrs = dict(catalog.header)
    boxsize = float(attrs['BOXSIZE'])
    # dx [Mpc/h] = v [km/s] / (100 a E(z)); vsmear is the mock redshift error, also radial.
    scale = float(attrs['VELZ2KMS'])
    columns = {'Position': np.column_stack([catalog[name] for name in ('X', 'Y', 'Z')]),
               'Velocity': np.column_stack([catalog[name] for name in ('VX', 'VY', 'VZ')]) / scale}
    columns['VSmear'] = (catalog['VSMEAR'] / scale if 'VSMEAR' in catalog.columns()
                         else np.zeros(len(columns['Position'])))
    box = BoxCatalog(columns, position='Position', velocity='Velocity', boxsize=boxsize,
                     boxcenter=0., mpicomm=mpicomm)
    box.attrs.update(attrs)
    logger.info('{:d} galaxies, boxsize {:.0f} Mpc/h, nbar {:.3e} (Mpc/h)^-3.'
                .format(box.csize, boxsize, box.csize / boxsize**3))
    return box


def get_radial_mask(tracer, region, nbar_box, zrange):
    """Downsampling probability per redshift, taking the box density to the data's n(z)."""
    from mockfactory import TabulatedRadialMask
    z, nbar = np.loadtxt(NZ_DIR / '{}_{}_nz.txt'.format(tracer, region), usecols=(0, 3), unpack=True)
    keep = (z > zrange[0]) & (z < zrange[1])
    if nbar[keep].max() > nbar_box:
        raise ValueError('the box (nbar {:.3e}) is too sparse for the {} {} n(z) (max {:.3e})'
                         .format(nbar_box, tracer, region, nbar[keep].max()))
    # The table is binned, so its centres stop short of the requested range on both sides. Let
    # it define its own limits rather than claiming coverage it does not have; the redshift cut
    # has already been applied, and the draw is zero where the table does not reach.
    # norm = 1 / nbar_box, not the default: that one rescales the selection to a maximum of one,
    # which keeps the shape of n(z) but puts its peak at the box density (the quasar boxes are
    # a hundred times denser than the sample). With it, prob = n(z) / nbar_box and prob / norm,
    # the NZ column, is n(z).
    return TabulatedRadialMask(z=z[keep], nbar=nbar[keep], norm=1. / nbar_box, interp_order=1)


def run_cutsky(args, mpicomm):
    """Pad each snapshot with periodic copies, apply radial RSD, and stitch the shells."""
    from mockfactory import Catalog, DistanceToRedshift, utils
    from mockfactory.desi.base import is_in_desi_footprint
    from mockfactory.desi.lsscat.utils import get_galactic_cap
    from cosmoprimo.fiducial import DESI

    cosmo = DESI()
    d2z = DistanceToRedshift(cosmo.comoving_radial_distance)
    shells = get_shells(args.program['snapshots'], args.zrange)
    logger.info('{:d} shell(s): {}'.format(
        len(shells), ', '.join('z={:.3f} over {:.2f}-{:.2f}'.format(zs, *zr) for zs, zr in shells)))

    for imock in args.imocks:
        pieces = []
        for zsnap, zrange in shells:
            box = read_box(imock, args.program['box'], zsnap, mpicomm, hod=args.program.get('hod', BOX_HOD))
            nbar_box = box.csize / box.boxsize.prod()
            drange = cosmo.comoving_radial_distance(np.array(zrange))
            # Enough copies to reach the far edge of this shell: (n + 1/2) L >= dmax.
            factor = 2 * np.ceil(drange[1] / box.boxsize - 0.5) + 1
            logger.info('shell z={:.3f}: padding by {} to cover {:.0f} Mpc/h.'
                        .format(zsnap, factor.astype('i4'), drange[1]))
            box = box.pad(factor=factor)

            position = box['Position']
            truedistance = utils.distance(position)
            los = position / truedistance[:, None]
            # Radial RSD, and the redshift error the spectrograph would have made, also radial.
            position = position + (np.sum(box['Velocity'] * los, axis=-1)
                                   + box['VSmear'])[:, None] * los
            distance, ra, dec = utils.cartesian_to_sky(position)
            del box, position, los

            select = (distance >= drange[0]) & (distance <= drange[1])
            # Both redshifts: Z is what a survey measures, TRUEZ is the same galaxy without the
            # displacement, which is the truth a closure test needs. The displacement is radial,
            # so the two share RA and DEC.
            piece = {'RA': ra[select], 'DEC': dec[select], 'Z': d2z(distance[select]),
                     'TRUEZ': d2z(truedistance[select])}

            # The n(z) is measured per galactic cap, so the draw is too.
            isngc = get_galactic_cap(piece['RA'], piece['DEC'])
            rng = np.random.default_rng(seed=args.seed + 1000 * imock + int(1000 * zsnap))
            prob, nz = np.zeros(len(isngc)), np.zeros(len(isngc))
            for region, inregion in zip(('NGC', 'SGC'), (isngc, ~isngc)):
                mask = get_radial_mask(args.program['nz'], region, nbar_box, zrange)
                prob[inregion] = mask.prob(piece['Z'][inregion])
                nz[inregion] = prob[inregion] / mask.norm
            select = prob >= rng.uniform(0., 1., size=prob.size)
            piece = {name: value[select] for name, value in piece.items()}
            piece['NZ'] = nz[select]
            logger.info('shell z={:.3f}: {:d} galaxies after the n(z) draw.'
                        .format(zsnap, mpicomm.allreduce(len(piece['Z']))))
            pieces.append(piece)

        catalog = {name: np.concatenate([piece[name] for piece in pieces])
                   for name in pieces[0]}
        del pieces
        logger.info('{:d} galaxies over {:d} shell(s), {:.2f} < z < {:.2f}.'
                    .format(mpicomm.allreduce(len(catalog['Z'])), len(shells), *args.zrange))

        # The DA2 tiles of this program as they are, no filtering on survey or program:
        # release=None takes tiles_fn at face value, which is what to cut the light cone to.
        select = is_in_desi_footprint(catalog['RA'], catalog['DEC'], release=None,
                                      tiles_fn=args.tiles_fn)
        catalog = {name: value[select] for name, value in catalog.items()}
        logger.info('footprint keeps {:d} galaxies.'.format(mpicomm.allreduce(len(catalog['Z']))))

        fn = cutsky_fn(args.output_dir, imock)
        Catalog(catalog, mpicomm=mpicomm).write(fn)
        logger.info('mock {:d}: cutsky -> {}'.format(imock, fn))


def run_targets(args, mpicomm):
    """Veto on the imaging, then build the target catalog the ledgers are made from."""
    from mockfactory import Catalog
    from mockfactory.desi.base import get_brick_pixel_quantities
    from mockfactory.desi.altmtl import make_targets, write_targets

    for imock in args.imocks:
        catalog = Catalog.read(cutsky_fn(args.output_dir, imock), mpicomm=mpicomm)
        # A format template, so a string: get_brick_pixel_quantities fills in the region and
        # the brick, and only then is it a path.
        columns = {'MASKBITS': {'fn': MASKBITS_FN, 'dtype': 'i2', 'default': 0}}
        for band in 'GRZ':
            columns['NOBS_' + band] = {'fn': NEXP_FN.format(band=band.lower()), 'dtype': 'i2',
                                       'default': 0}
        # With a cache the four quantities share one shard per brick prefix, against four
        # compressed files a brick from the legacy survey: 2070 s becomes minutes. Build it once
        # with scripts/convert_bricks_to_h5.py; without it the bricks are read where they live.
        quantities = get_brick_pixel_quantities(catalog['RA'], catalog['DEC'], columns,
                                                cache_dir=args.brick_cache_dir, mpicomm=mpicomm)
        for name, value in quantities.items():
            catalog[name] = value
        # The same cut the production target files were made with: the mask bits, and coverage
        # in all three bands. The clustering vetoes read these columns again for the randoms,
        # so they are carried into the target catalog rather than dropped here.
        select = np.ones(len(catalog['RA']), dtype='?')
        for bit in args.bits:
            select &= (catalog['MASKBITS'] & 2**bit) == 0
        for band in 'GRZ':
            select &= catalog['NOBS_' + band] > 0
        catalog = catalog[select]
        logger.info('mock {:d}: imaging veto keeps {:d} of {:d} targets.'
                    .format(imock, catalog.csize, mpicomm.allreduce(len(select))))

        targets = make_targets({args.tracer: catalog}, obscon=args.obscon,
                               seed=args.seed + imock, z='Z', mpicomm=mpicomm,
                               columns=('TRUEZ', 'MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z'))
        fn = targets_fn(args.output_dir, imock)
        write_targets(targets, fn, obscon=args.obscon, mpicomm=mpicomm)
        logger.info('mock {:d}: targets -> {}'.format(imock, fn))


def run_altmtl(args):
    """Replay the survey against the targets."""
    from mockfactory.desi.altmtl import run_mocks

    mocks = [(targets_fn(args.output_dir, imock), altmtl_dir(args.output_dir, imock), 0, {})
             for imock in args.imocks]
    results = run_mocks(mocks, args.end_date, obscon=args.obscon, numproc=args.numproc,
                        nummocks=args.nummocks)
    failed = [result['altmtl_dir'] for result in results if 'error' in result]
    for result in results:
        logger.info('{altmtl_dir}: {nactions:d} actions in {seconds:.0f} s'.format(**result)
                    if 'error' not in result else '{altmtl_dir}: FAILED {error}'.format(**result))
    if failed:
        raise SystemExit('{:d} mock(s) failed: {}'.format(len(failed), failed))


def run_pota(args):
    """Which targets each fiber could have reached, the denominator the assignment is measured against."""
    from mockfactory.desi.altmtl import compute_potential_assignments

    for imock in args.imocks:
        fn = pota_fn(args.output_dir, imock)
        compute_potential_assignments(targets_fn(args.output_dir, imock), fn, args.tiles_fn,
                                      program=args.obscon.upper(), numproc=args.numproc)
        logger.info('mock {:d}: potential assignments -> {}'.format(imock, fn))


def run_lsscat(args):
    """
    The clustering catalogs, data and randoms.

    The randoms follow the survey, not the mock. `LSS` does the same for its mocks
    (`mocktools.createrancomb_wdupspec`, called from `mkCat_SecondGen_amtl.py`): it reads the
    survey's own `rancomb_<n><program>wdupspec_zdone.fits`, keeps `LOCATION`, `FIBER`,
    `TARGETID`, `RA`, `DEC` and `TILEID` from it, and joins on only the mock's `PRIORITY`.
    Which tile and fiber a random falls on is decided by the randoms and the tiles, so the step
    from the imaging randoms to those files is survey level and serves every mock unchanged.
    The imaging randoms do enter directly, through `read_random_imaging`, which supplies the
    `MASKBITS` and `NOBS` the veto needs -- the same cut the targets were made with.
    """
    import fitsio
    import numpy as np
    from mockfactory.desi.altmtl.targets import read_targets
    from mockfactory.desi.tables import as_table, set_column
    from mockfactory.desi.lsscat import (read_assignments, combine_data, combine_randoms,
                                         count_tiles, read_good_tilelocid, read_random_imaging,
                                         run_tracer)

    # True, not the default None: None skips these cuts, and the survey applies all three.
    good_tilelocid = read_good_tilelocid(args.spec_fn, program=args.obscon, bad_fibers=True,
                                         bad_petal_nights=True, bad_fibers_time=True)
    tileids = np.unique(fitsio.read(args.tiles_fn, columns=['TILEID'])['TILEID'])

    # Counted once for the whole run, not once a mock: which tiles could reach a random depends
    # on the randoms and on where the survey got a usable spectrum, and on nothing the mock
    # decides. Left to run_tracer it would be recomputed for every mock.
    imaging, random_tiles = [], []
    for i in range(args.nrandom):
        start = time.time()
        imaging.append(read_random_imaging(i, tracer=args.tracer))
        array = as_table(fitsio.read(args.rancomb_fns[i], columns=RANDOM_COLUMNS))
        set_column(array, 'TILELOCID', 10000 * array['TILEID'] + array['LOCATION'], dtype='i8')
        random_tiles.append(count_tiles(array[np.isin(array['TILELOCID'], good_tilelocid)]))
        del array
        logger.info('random {:d}: imaging and tile counts in {:.0f} s.'.format(i, time.time() - start))

    for imock in args.imocks:
        targets = read_targets(targets_fn(args.output_dir, imock))
        assignments = read_assignments(altmtl_dir(args.output_dir, imock), tileids,
                                       obscon=args.obscon, numproc=args.numproc)

        # Callables, not arrays: run_tracer builds each in its own frame and lets it go, where
        # holding them here would keep them alive for every forked worker.
        def _data(imock=imock, assignments=assignments, targets=targets):
            return combine_data(fitsio.read(pota_fn(args.output_dir, imock)), assignments,
                                targets=targets, good_tilelocid=good_tilelocid)

        def _random(i, assignments=assignments):
            # combine_randoms replaces the survey's PRIORITY with the one this mock implies.
            return combine_randoms(fitsio.read(args.rancomb_fns[i], columns=RANDOM_COLUMNS),
                                   assignments)

        randoms = [(lambda j: (lambda i: _random(j)))(i) for i in range(args.nrandom)]
        run_tracer(_data, randoms, assignments, args.tracer, targets=targets,
                   random_imaging=imaging, random_tiles=random_tiles,
                   good_tilelocid=good_tilelocid, hpmaps=None,
                   bits=list(args.bits), columns=('TRUEZ',), name=args.name or args.tracer,
                   numproc=args.numproc, numproc_randoms=args.numproc_randoms,
                   output_dir=Path(args.output_dir) / 'mock{:d}'.format(imock), keep=False)
        logger.info('mock {:d}: clustering catalogs written.'.format(imock))


def run_imsys(args):
    """
    Imaging weights, fitted on the clustering catalogs and written back into them.

    The survey pipeline does it so for its mocks (`mkCat_amtl.py --doimlin y --replace_syscol`),
    once the catalogs are split by galactic cap and given their n(z): the fit takes the data and
    every random catalog of both caps, weighted ``WEIGHT * WEIGHT_FKP / WEIGHT_SYS`` (the randoms
    also divided by ``WEIGHT_ZFAIL``), so that it can be re-run on catalogs that already carry
    imaging weights. The weights then go in as
    ``WEIGHT_IMLIN`` and replace ``WEIGHT_SYS``, ``WEIGHT`` being rescaled to match, in the data
    and in the randoms, each random taking the weight of the data its redshift was drawn from
    (``TARGETID_DATA``). It is the survey pipeline's last step on the catalogs, and as there the
    n(z) is not recomputed. Every file of the mock is rewritten, the uncut ones too.

    The emission line galaxies are weighted with SYSNet, a neural network, as the survey pipeline
    does (`mkCat_amtl.py --prep4sysnet y --addsysnet y --replace_syscol`), see
    :mod:`mockfactory.desi.lsscat.sysnet`: the data are counted with their completeness weights
    only, against the quasar maps, and the weights go in as ``WEIGHT_SN``. Its inputs, models
    and predictions are kept in ``mock{i}/sysnet``.
    """
    import h5py
    from astropy.table import Table, vstack
    from mockfactory.desi.lsscat import compute_imaging_weights, compute_sysnet_weights, read_hpmaps, write_catalogs, FIT_MAPS, SYSNET_FIT_MAPS

    sysnet = args.program.get('imsys_method', 'linear') == 'sysnet'
    maps_north, maps_south = read_hpmaps(args.hpmap_dir, args.program.get('imsys_maps', args.tracer))
    fit_maps = (SYSNET_FIT_MAPS if sysnet else FIT_MAPS)[args.tracer[:3]]
    name, caps = args.name or args.tracer, ('NGC', 'SGC')
    syscol = 'WEIGHT_SN' if sysnet else 'WEIGHT_IMLIN'
    columns = ['RA', 'DEC', 'Z', 'PHOTSYS', 'WEIGHT', 'WEIGHT_FKP', 'WEIGHT_SYS', 'WEIGHT_ZFAIL']  # the randoms are also divided by WEIGHT_ZFAIL, as in LSS
    data_columns = ['TARGETID'] + columns + (['WEIGHT_COMP', 'FRAC_TLOBS_TILES'] if sysnet else [])
    randoms_columns = ['RA', 'DEC', 'PHOTSYS'] if sysnet else columns

    def read(fn, columns=None):
        with h5py.File(fn, 'r') as file:
            group = file['LSS']
            return Table({column: group[column][...] for column in (columns or list(group))}, copy=False)

    for imock in args.imocks:
        mock_dir = Path(args.output_dir) / 'mock{:d}'.format(imock)

        def fn(kind, cap=None, i=None):
            parts = [name] + ([cap] if cap else []) + (['{:d}'.format(i)] if i is not None else [])
            return mock_dir / '{}_clustering.{}.h5'.format('_'.join(parts), kind)

        start = time.time()
        data = vstack([read(fn('dat', cap), data_columns) for cap in caps])
        randoms = vstack([read(fn('ran', cap, i), randoms_columns) for cap in caps for i in range(args.nrandom)])
        if sysnet:
            weights, _ = compute_sysnet_weights(data, randoms, maps_north, maps_south, fit_maps,
                                                args.program['imsys_zranges'], mock_dir / 'sysnet',
                                                regions=args.program['imsys_regions'])
        else:
            weights, coefficients = compute_imaging_weights(data, randoms, maps_north, maps_south, fit_maps,
                                                            args.program['imsys_zranges'],
                                                            regions=args.program['imsys_regions'])
            for key, value in coefficients.items():
                logger.info('mock {:d}, region {}, {} < z < {}: {}'.format(imock, key[0], *key[1], value))
        del randoms
        logger.info('mock {:d}: imaging weights fitted in {:.0f} s, between {:.3f} and {:.3f}.'
                    .format(imock, time.time() - start, weights.min(), weights.max()))

        order = np.argsort(data['TARGETID'])
        targetid, weights = np.asarray(data['TARGETID'])[order], weights[order]
        del data
        files = [(fn('dat', cap), 'TARGETID') for cap in (None,) + caps]
        files += [(fn('ran', cap, i), 'TARGETID_DATA') for i in range(args.nrandom) for cap in (None,) + caps]
        # Updated one by one and written numproc at a time: writing is most of the cost and it
        # parallelises over processes (see write_catalogs), but holding every catalog until the end
        # costs a mock's whole size, about 110 GB at eighteen randoms, which five mocks on a node
        # cannot afford.
        writes = []
        for path, key in files:
            if len(writes) >= args.numproc:
                write_catalogs(writes, numproc=args.numproc)
                writes = []
            catalog = read(path)
            index = np.clip(np.searchsorted(targetid, catalog[key]), 0, len(targetid) - 1)
            found = targetid[index] == catalog[key]
            # A random whose region had no data to draw from has no donor: its weights stay as they were.
            new = np.array(catalog['WEIGHT_SYS'], dtype='f8')
            new[found] = weights[index[found]]
            catalog['WEIGHT'] = catalog['WEIGHT'] / catalog['WEIGHT_SYS'] * new
            catalog['WEIGHT_SYS'] = new
            catalog[syscol] = new
            writes.append((path, catalog))
        write_catalogs(writes, numproc=args.numproc)
        del writes
        logger.info('mock {:d}: imaging weights written into {:d} catalogs.'.format(imock, len(files)))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', required=True, help='where every stage writes')
    parser.add_argument('--stages', default=','.join(STAGES),
                        help='comma separated, in order: ' + ', '.join(STAGES))
    parser.add_argument('--imocks', default='0', help='which mocks, e.g. 0, 0,3 or 0-11')
    parser.add_argument('--tracer', default='BGS_BRIGHT', choices=sorted(PROGRAMS),
                        help='which sample; it sets the boxes, snapshots, n(z) and program')
    parser.add_argument('--name', default=None, help='name the clustering catalogs take')
    parser.add_argument('--zrange', type=float, nargs=2, default=None,
                        help="defaults to the tracer's own")
    parser.add_argument('--end-date', type=int, default=20240418)
    parser.add_argument('--bits', type=int, nargs='*', default=None,
                        help="imaging bits to veto; defaults to the program's own")
    parser.add_argument('--tiles-fn', default=None,
                        help="tiles the mock is cut and assigned to; defaults to the program's")
    parser.add_argument('--brick-cache-dir', default=None,
                        help='converted DR9 bricks, from scripts/convert_bricks_to_h5.py; '
                             'without it the legacy survey files are read directly, which costs '
                             'four compressed reads a brick instead of one shard a prefix')
    parser.add_argument('--nrandom', type=int, default=4,
                        help='random catalogs the clustering catalogs are paired with')
    parser.add_argument('--hpmap-dir', default=HPMAP_DIR,
                        help='observing condition maps the imaging weights are fitted against')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--numproc', type=int, default=32)
    parser.add_argument('--numproc-randoms', type=int, default=None)
    parser.add_argument('--nummocks', type=int, default=1, help='mocks replayed side by side')
    args = parser.parse_args(args=args)
    args.imocks = parse_imocks(args.imocks)
    # Everything a tracer implies is read off one table, so a dark run needs only --tracer.
    args.program = PROGRAMS[args.tracer]
    args.obscon = args.program['obscon']
    if args.zrange is None:
        args.zrange = args.program['zrange']
    if args.bits is None:
        args.bits = list(VETO_BITS[args.obscon])
    if args.tiles_fn is None:
        args.tiles_fn = str(LSS_DIR / 'tiles-{}.fits'.format(args.obscon.upper()))
    args.rancomb_fns = [str(LOA_DIR / 'rancomb_{:d}{}wdupspec_zdone.fits'.format(i, args.obscon.lower()))
                        for i in range(args.nrandom)]
    args.spec_fn = str(LOA_DIR / 'datcomb_{}_spec_zdone.fits'.format(args.obscon.lower()))
    stages = args.stages.split(',')
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ValueError('unknown stage(s) {}, expected among {}'.format(unknown, list(STAGES)))

    os.environ['OMP_NUM_THREADS'] = '1'
    from mockfactory import setup_logging
    import mpytools as mpy
    setup_logging()
    mpicomm = mpy.COMM_WORLD
    os.makedirs(args.output_dir, exist_ok=True)

    for stage in STAGES:
        if stage not in stages:
            continue
        if stage in ('cutsky', 'targets'):
            {'cutsky': run_cutsky, 'targets': run_targets}[stage](args, mpicomm)
        else:
            if mpicomm.size > 1:
                raise ValueError('stage {} runs in one process with forked pools; '
                                 'give it one rank'.format(stage))
            {'altmtl': run_altmtl, 'pota': run_pota, 'lsscat': run_lsscat, 'imsys': run_imsys}[stage](args)
        logger.info('--- {} done ---'.format(stage))


if __name__ == '__main__':
    main()

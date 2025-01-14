import os
import logging

import numpy as np


logger = logging.getLogger('DESI footprint')


# check if the env variable DESI_SPECTRO_REDUX is defined, otherwise load default path:
try:
    redux_path = os.environ['DESI_SPECTRO_REDUX']
except KeyError:
    logger.warning("$DESI_SPECTRO_REDUX is not set in the current environment. No assurance for the existence of files. Default path will be used: /global/cfs/cdirs/desi/spectro/redux")
    redux_path = '/global/cfs/cdirs/desi/spectro/redux'

# check if the env variable DESI_SURVEYOPS is defined (needed for desimodel.io.load_tiles()):
    try:
        redux_path = os.environ['DESI_SURVEYOPS']
    except KeyError:
        # see: https://desisurvey.slack.com/archives/C025RHKPV8R/p1729735768040629?thread_ts=1729733422.550579&cid=C025RHKPV8R
        logger.warning("$DESI_SURVEYOPS is not set in the current environment. No assurance for the existence of files. Default path will be used: /global/cfs/cdirs/desi/survey/ops/surveyops/trunk")
        os.environ['DESI_SURVEYOPS'] = '/global/cfs/cdirs/desi/survey/ops/surveyops/trunk'

def is_in_desi_footprint(ra, dec, release='m3', npasses=None, program='dark', survey='main',
                         tiles_fn=os.path.join(redux_path, '{redux}/tiles-{redux}.csv'),
                         return_tile_index=False):
    """
    Return mask for the requested DESI footprint.

    Note
    ----
    Y1 is defined with the daily catalog, before the fire (20220613) and not from the file in its reduction directory which is equivalent than use redux='iron'.

    Parameters
    ----------
    ra : array
        Right ascension (degree).

    dec : array
        Declination (degree).

    release : string, default='m3'
        Name of the survey. Available: onepercent, m3, y1, y3, y5.

    npasses : int, default=None
        Number of passes; ``None`` for all passes.

    program : string, default='dark'
        Name of the program. Either 'dark' for LRG/ELG/QSO or 'bright' for BGS.

    survey : string, default='main'
        Type of the survey, 'main' for the standard DESI clustering analysis.

    tiles_fn : string, Path
        Template path to csv or fits file of tiles.

    return_tile_index : bool
        if true also the tile id for each entry (ra, dec). If npasses is None, it gives only one tile id, otherwise it gives all the potential tile id for each targets.

    Returns
    -------
    mask : array
        Boolean array of the same size than ra, dec, with ``True`` if in footprint, ``False`` otherwise.
    """
    import desimodel.footprint

    lastnight = None
    release = release.lower()
    if release in ['sv3', 'onepercent']:
        redux = 'fuji'
    elif release in ['da02', 'm3']:
        redux = 'guadalupe'
    elif release == 'y1':
        redux = 'daily'
        lastnight = 20220613
    elif release == 'y3':
        redux = 'loa'
    elif release == 'y5':
        logger.warning('On 20241010, 231 PASS=0-6 tiles in the DES region (Dec < -20) were added. The expected Y5 footprint generated before 20241010 is therefore not the same... (need some tricks to match it --like recovering with svn the old file.)')
        redux = None   
    else:
        raise ValueError('Unknown release {}'.format(release))
         
    if redux is None:
        import desimodel.io
        tiles = desimodel.io.load_tiles()
        tiles = tiles[tiles['PROGRAM'] == program.upper()]
    else:
        import pandas as pd
        tiles_fn = tiles_fn.format(redux=redux)
        tiles = pd.read_csv(tiles_fn)
        tiles = tiles[(tiles['SURVEY'] == survey) & (tiles['FAPRGRM'] == program)]
        if lastnight is not None:
            tiles = tiles[tiles['LASTNIGHT'] <= lastnight]
        tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']

    if npasses is not None:
        tile_id = desimodel.footprint.find_tiles_over_point(tiles, ra, dec)
        if return_tile_index:
            return np.array([len(tt) >= npasses for tt in tile_id], dtype='?'), tile_id
        else:
            return np.array([len(tt) >= npasses for tt in tile_id], dtype='?')

    return desimodel.footprint.is_point_in_desi(tiles, ra, dec, return_tile_index=return_tile_index)


if __name__ == '__main__':
    import time
    from mockfactory import RandomCutskyCatalog, setup_logging

    setup_logging()

    # Generate example cutsky catalog, scattered on all processes
    cutsky = RandomCutskyCatalog(rarange=(200, 250), decrange=(15, 45), csize=10000, seed=44)
    ra, dec = cutsky['RA'], cutsky['DEC']

    for release in ['SV3', 'DA02', 'Y1', 'Y3', 'Y5']:
        for npasses in [None, 3]:
            t0 = time.time()
            is_in_desi = is_in_desi_footprint(ra, dec, release=release, npasses=npasses, program='dark')
            logger.info(f'Mask build for {ra.size} objects in {time.time() - t0:2.2f}s')
            logger.info(f'There are {is_in_desi.sum() / is_in_desi.size:2.2%} objects in {release} with {npasses} passes')

    # You can also request the list of tiles:
    is_in_footprint, tile_id = is_in_desi_footprint(ra, dec, release='Y1', npasses=3, program='dark', return_tile_index=True)

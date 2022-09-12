import os
import logging

import numpy as np

import desimodel.io
import desimodel.footprint


logger = logging.getLogger('DESI footprint')


# check if the env variable DESI_SPECTRO_REDUX is defined, otherwise load default path
try:
    redux_path = os.environ['DESI_SPECTRO_REDUX']
except KeyError:
    logger.warning("$DESI_SPECTRO_REDUX is not set in the current environment. No assurance for the existence of files. Default path will be used: /global/cfs/cdirs/desi/spectro/redux")
    redux_path = '/global/cfs/cdirs/desi/spectro/redux'


def is_in_desi_footprint(ra, dec, release='m3', npasses=None, program='dark', survey='main', tiles_fn=os.path.join(redux_path, '{redux}/tiles-{redux}.csv')):
    """
    Return mask for the requested DESI footprint.

    Note
    ----
    Y1 is for the moment defined with the daily catalog, before the fire (20220613).

    Parameters
    ----------
    ra : array
        Right ascension (degree).

    dec : array
        Declination (degree).

    release : string, default='m3'
        Name of the survey. Available: onepercent, m3, y1, y5.

    npasses : int, default=None
        Number of passes; ``None`` for all passes.

    program : string, default='dark'
        Name of the program. Either 'dark' for LRG/ELG/QSO or 'bright' for BGS.

    survey : string, default='main'
        Type of the survey, 'main' for the standard DESI clustering analysis.

    tiles_fn : string
        Template path to csv file of tiles.

    Returns
    -------
    mask : array
        Boolean array of the same size than ra, dec, with ``True`` if in footprint, ``False`` otherwise.
    """
    lastnight = None
    release = release.lower()
    if release in ['sv3', 'onepercent']:
        redux = 'fuji'
    elif release in ['da02', 'm3']:
        redux = 'guadalupe'
    elif release == 'y1':
        redux = 'daily'
        lastnight = 20220613
    elif release == 'y5':
        redux = None
    else:
        raise ValueError('Unknown release {}'.format(release))

    if redux is None:
        tiles = desimodel.io.load_tiles()
    else:
        import pandas as pd
        tiles_fn = tiles_fn.format(redux=redux)
        tiles = pd.read_csv(tiles_fn)
        tiles = tiles[(tiles['SURVEY'] == survey) & (tiles['FAPRGRM'] == program)]
        if lastnight is not None:
            tiles = tiles[tiles['LASTNIGHT'] <= lastnight]
        tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']

    if npasses is not None:
        return np.array([len(tt) >= npasses for tt in desimodel.footprint.find_tiles_over_point(tiles, ra, dec)], dtype='?')

    return desimodel.footprint.is_point_in_desi(tiles, ra, dec)


if __name__ == '__main__':

    import time
    from mockfactory import RandomCutskyCatalog, setup_logging

    setup_logging()

    # Generate example cutsky catalog, scattered on all processes
    cutsky = RandomCutskyCatalog(rarange=(200, 250), decrange=(15, 45), size=10000, seed=44)
    ra, dec = cutsky['RA'], cutsky['DEC']

    for release in ['SV3', 'DA02', 'Y1', 'Y5']:
        for npasses in [None, 3]:
            t0 = time.time()
            is_in_desi = is_in_desi_footprint(ra, dec, release=release, npasses=npasses, program='dark')
            logger.info(f'Mask build for {ra.size} objects in {time.time() - t0:2.2f}s')
            logger.info(f'There are {is_in_desi.sum() / is_in_desi.size:2.2%} objects in {release} with {npasses} passes')

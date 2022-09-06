import numpy as np
import pandas as pd

import desimodel.io
import desimodel.footprint


def is_in_desi_footprint(ra, dec, release='da02', program='dark', survey='main'):
    """
        Return mask for objects in the considered DESI footprint.

        **Remark:** Y1 is for the moment define with the daily catalog. Y1 DR is expected to be everything before the fire.

        Parameters
        ----------
        ra : array
            Right ascension (degree).

        dec : array
            Declination (degree).

        release : str
            Name of the survey. Available now: sv1, sv2, sv3, da02, y1, y1-3pass, y5.

        program : str
            Name of the program. Either dark for LRG/ELG/QSO or bright for BGS.

        survey : str
            Type of the survey. Let main for standard DESI analysis.

        Returns
        -------
        bool array of the same size than ra.
    """
    if release in ['sv1', 'sv2', 'sv3']:
        tiles = pd.read_csv('/global/cfs/cdirs/desi/spectro/redux/fuji/tiles-fuji.csv')
        tiles = tiles[(tiles['SURVEY'] == survey) & (tiles['FAPRGRM'] == program)]
        tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']

    if release == 'da02':
        tiles = pd.read_csv('/global/cfs/cdirs/desi/spectro/redux/guadalupe/tiles-guadalupe.csv')
        tiles = tiles[(tiles['SURVEY'] == survey) & (tiles['FAPRGRM'] == program)]
        tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']

    if release == 'y1':
        tiles = pd.read_csv('/global/cfs/cdirs/desi/spectro/redux/daily/tiles-daily.csv')
        tiles = tiles[(tiles['SURVEY'] == survey) & (tiles['FAPRGRM'] == program)]
        # keep all the observation before the fire --> expect to be the definition of Y1
        tiles = tiles[tiles['LASTNIGHT'] <= 20220613]
        tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']

    if release == 'y1-3pass':
        tiles = pd.read_csv('/global/cfs/cdirs/desi/spectro/redux/daily/tiles-daily.csv')
        tiles = tiles[(tiles['SURVEY'] == survey) & (tiles['FAPRGRM'] == program)]
        # keep all the observation before the fire --> expect to be the definition of Y1
        tiles = tiles[tiles['LASTNIGHT'] <= 20220613]
        tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']

        return np.array([len(tt) >= 3 for tt in desimodel.footprint.find_tiles_over_point(tiles, ra, dec)])

    if release == 'y5':
        tiles = desimodel.io.load_tiles()

    return desimodel.footprint.is_point_in_desi(tiles, ra, dec)


if __name__ == '__main__':

    from mockfactory import RandomCutskyCatalog
    import time

    # Generate example cutsky catalog, scattered on all processes
    cutsky = RandomCutskyCatalog(rarange=(200, 250), decrange=(15, 45), size=10000, seed=44)
    ra, dec = cutsky['RA'], cutsky['DEC']

    t0 = time.time()
    is_in_desi = is_in_desi_footprint(ra, dec, release='y1-3pass', program='dark')
    print(f'Mask build for {ra.size} objects in {time.time() - t0:2.2f}s')

    print(f'There is {is_in_desi.sum() / is_in_desi.size:2.2%} objects in the desi footprint')

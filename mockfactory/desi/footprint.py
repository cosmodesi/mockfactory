import numpy as np
import pandas as pd

import desimodel.io
import desimodel.footprint


def is_in_footprint(ra, dec, release='da02', survey='main', program='dark'):
    """
        TODO

        **Remark:** Y1 is for the moment define with the daily catalog --> With the fire, the Y1 DR is expected to be everything before the fire.
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
        print('EST-ce que je veux faire quelque chose comme ca?')
        # tiles = pd.read_csv('/global/cfs/cdirs/desi/spectro/redux/daily/tiles-daily.csv')
        # tiles['RA'], tiles['DEC'] = tiles['TILERA'], tiles['TILEDEC']
        # pix_per_tiles = desimodel.footprint.tiles2pix(1024, tiles=tiles[(tiles['FAPRGRM'] == faprgrm)], per_tile=True)
        #
        # mask = np.zeros(hp.nside2npix(1024))
        # for i in range(len(pix_per_tiles)):
        #     mask[pix_per_tiles[i]] += 1
        #
        # mask = hp.ud_grade(mask, nside, order_in='NESTED', order_out='NESTED')
        #
        # desi_foot['y1' + '-' + faprgrm] = mask > 0
        # desi_foot['y1-3pass' + '-' + faprgrm] = mask > 3

    if release == 'y5':
        tiles = desimodel.io.load_tiles()

    return desimodel.footprint.is_point_in_desi(tiles, ra, dec)

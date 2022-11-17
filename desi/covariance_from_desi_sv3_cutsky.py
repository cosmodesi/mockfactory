"""Compute covariance from SV3 cutsky mocks."""

import os
import argparse

import numpy as np


def get_edges(corr_type='smu', bin_type='lin'):
    """Taken from https://github.com/desihub/LSS/blob/main/scripts/xirunpc.py"""
    if bin_type == 'log':
        sedges = np.geomspace(0.01, 100., 49)
    elif bin_type == 'lin':
        sedges = np.linspace(0., 200, 201)
    else:
        raise ValueError('bin_type must be one of ["log", "lin"]')
    if corr_type == 'smu':
        edges = (sedges, np.linspace(-1., 1., 201))  # s is input edges and mu evenly spaced between -1 and 1
    elif corr_type == 'rppi':
        if bin_type == 'lin':
            edges = (sedges, np.linspace(-200., 200, 401))  # transverse and radial separations are coded to be the same here
        else:
            edges = (sedges, np.linspace(-40., 40., 81))
    elif corr_type == 'theta':
        if bin_type == 'lin':
            edges = np.linspace(0., 4., 101)
        else:
            edges = np.geomspace(0.001, 4., 41)
    else:
        raise ValueError('corr_type must be one of ["smu", "rppi", "theta"]')
    return edges


if __name__ == '__main__':
    """
    Example of how to compute correlation function and covariance matrix from DESI SV3 cutsky mocks.

    Compute correlation functions:

        srun -n 1 python covariance_from_desi_sv3_cutsky.py --todo corr --start 0 --stop 1000

    Compute covariance matrix:

        srun -n 1 python covariance_from_desi_sv3_cutsky.py --todo cov --start 0 --stop 1000

    Plot:

        srun -n 1 python covariance_from_desi_sv3_cutsky.py --todo plot

    """
    from mockfactory import Catalog
    from pycorr import TwoPointCorrelationFunction, setup_logging, utils

    from from_box_to_desi_sv3_cutsky import read_rosettes, catalog_fn

    setup_logging()

    parser = argparse.ArgumentParser(description='Generate DESI SV3 cutsky mocks')
    parser.add_argument('--todo', type=str, required=False, default='corr', choices=['corr', 'cov', 'plot'], help='What to do?')
    parser.add_argument('--corr_type', type=str, required=False, default='smu', choices=['smu', 'rppi', 'theta'], help='Correlation type')
    parser.add_argument('--start', type=int, required=False, default=0, help='First mock to compute correlation function of')
    parser.add_argument('--stop', type=int, required=False, default=1, help='Last (exclusive) mock to compute correlation function of')
    args = parser.parse_args()

    all_rosettes = [(1, 2), (12, 13), (9, 10), (8, 17), (15, 18), (4, 16), (7,), (14,), (6,), (11,), (5,), (0,), (3,), (19,)]

    # Output directory
    outdir = '_tests'
    utils.mkdir(outdir)

    ells = (0, 2, 4)
    pimax = 40.
    bin_type = 'log'
    nran = 10  # how many random files

    def corr_fn(corr_type, rosettes, imock=0):
        return os.path.join(outdir, 'correlation_{}_rosettes-{}_{:d}.npy'.format(corr_type, '-'.join([str(rosette) for rosette in rosettes]), imock))

    def cov_fn(corr_type):
        return os.path.join(outdir, 'covariance_correlation_{}.npy'.format(corr_type))

    if args.todo == 'corr':
        for rosettes in all_rosettes:
            R1R2 = None
            for imock in range(args.start, args.stop):
                data = Catalog.read(catalog_fn('data', rosettes, imock=imock))
                edges = get_edges(corr_type=args.corr_type, bin_type=bin_type)

                corr = 0
                D1D2 = None
                for iran in range(nran):
                    randoms = Catalog.read(catalog_fn('randoms', rosettes, imock=iran))
                    corr += TwoPointCorrelationFunction(args.corr_type, data_positions1=data['RSDPosition'], randoms_positions1=randoms['Position'],
                                                        edges=edges, nthreads=64, position_type='pos', mpicomm=data.mpicomm, mpiroot=None, D1D2=D1D2, R1R2=R1R2)
                corr.save(corr_fn(args.corr_type, rosettes, imock=imock))
                R1R2 = corr.R1R2

    if args.todo == 'cov':
        region_rosettes = read_rosettes()[-1]
        regions = list(set(region_rosettes.values()))
        # Check rosettes are grouped in the same region
        rosettes_to_region = {rosettes: region_rosettes[rosettes[0]] for rosettes in all_rosettes}
        for rosettes, region in rosettes_to_region.items():
            if not all(region_rosettes[rosette] == region for rosette in rosettes):
                raise ValueError('Rosettes must be grouped by regions')

        all_D1D2_wnorm = {rosettes: TwoPointCorrelationFunction.load(corr_fn(args.corr_type, rosettes, imock=args.start)).D1D2.wnorm for rosettes in all_rosettes}
        region_D1D2_wnorm = {region: sum(all_D1D2_wnorm[rosettes] for rosettes in all_rosettes if rosettes_to_region[rosettes] == region) for region in regions}

        # We weight each rosette correlation function measurement by (R1R2 of this rosette) / (R1R2 summed over all rosettes)
        all_R1R2 = {rosettes: TwoPointCorrelationFunction.load(corr_fn(args.corr_type, rosettes, imock=args.start)).normalize().R1R2 for rosettes in all_rosettes}
        region_R1R2_wnorm = {region: sum(all_R1R2[rosettes].wnorm for rosettes in all_rosettes if rosettes_to_region[rosettes] == region) for region in regions}
        # Rescale random pairs by total number of data pairs in each region N and S
        rescale_region = {region: region_D1D2_wnorm[region] / region_R1R2_wnorm[region] for region in regions}
        all_R1R2 = {rosettes: R1R2.normalize(wnorm=R1R2.wnorm * rescale_region[rosettes_to_region[rosettes]]) for rosettes, R1R2 in all_R1R2.items()}
        R1R2 = sum(all_R1R2.values())

        all_mean, all_cov, attrs = {}, {}, {}
        for rosettes in all_rosettes:
            ratio = all_R1R2[rosettes].wcounts / R1R2.wcounts
            all_corr = []
            for imock in range(args.start, args.stop):
                result = TwoPointCorrelationFunction.load(corr_fn(args.corr_type, rosettes, imock=imock))
                for name in ['D1D2', 'D1S2', 'S1D2', 'S1S2']:  # these are in the numerator
                    try:
                        tmp = getattr(result, name).deepcopy()
                    except AttributeError:
                        continue
                    tmp.wcounts *= ratio
                    setattr(result, name, tmp)
                result.run()  # recompute corr
                if args.corr_type == 'smu':
                    corr = result.get_corr(ells=ells, return_cov=False)
                    attrs['ells'] = tuple(ells)
                elif args.corr_type == 'rppi' and pimax is not None:
                    corr = result.get_corr(pimax=pimax, return_cov=False)
                    attrs['pimax'] = pimax
                else:
                    corr = result.corr
                corr = np.asarray(corr)
                all_corr.append(np.ravel(corr))
            all_mean[rosettes] = np.mean(all_corr, axis=0)
            all_cov[rosettes] = np.cov(all_corr, rowvar=False, ddof=1)
        mean = sum(all_mean.values()).reshape(corr.shape)
        cov = sum(all_cov.values())
        result = {'mean': mean, 'cov': cov, 'edges': result.edges, 'corr_type': args.corr_type, **attrs}
        np.save(cov_fn(args.corr_type), result, allow_pickle=True)

    if args.todo == 'plot':
        from matplotlib import pyplot as plt
        from matplotlib import gridspec
        result = np.load(cov_fn(args.corr_type), allow_pickle=True)[()]
        sep = (result['edges'][0][:-1] + result['edges'][0][1:]) / 2.
        std = np.diag(result['cov'])**0.5
        xlim = (result['edges'][0].min(), result['edges'][0].max())
        fig = plt.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(ncols=2, nrows=1)
        n = 1
        ax = plt.subplot(gs[0])
        if 'ells' in result:
            n = len(result['ells'])
            std = np.array_split(std, n)
            for ill, ell in enumerate(result['ells']):
                ax.errorbar(sep, sep * result['mean'][ill], yerr=sep * std[ill], color='C{:d}'.format(ill), label=r'$\ell = {:d}$'.format(ell))
            ax.set_xlabel(r'$s$ [$\mathrm{Mpc}/h$]')
            ax.set_ylabel(r'$s \xi_{\ell}(s)$ [$\mathrm{Mpc}/h$]')
            ax.legend()
        elif 'pimax' in result:
            ax.errorbar(sep, sep * result['mean'], yerr=sep * std)
            ax.set_xlabel(r'$r_{p}$ [$\mathrm{Mpc}/h$]')
            ax.set_ylabel(r'$r_{p} w_{p}$ [$(\mathrm{{Mpc}}/h)^{{2}}$]')
        ax.set_xscale(bin_type)
        ax.set_xlim(xlim)
        corrcoef = utils.cov_to_corrcoef(result['cov'])
        ns = len(sep)
        gs = gridspec.GridSpecFromSubplotSpec(n, n, subplot_spec=gs[1], wspace=0.1, hspace=0.1)
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=np.nanmin(corrcoef), vmax=np.nanmax(corrcoef))
        for i in range(n):
            for j in range(n):
                ax = plt.subplot(gs[n - 1 - i, j])
                mesh = ax.pcolor(sep, sep, corrcoef[i * ns: (i + 1) * ns, j * ns: (j + 1) * ns].T, norm=norm, cmap=plt.get_cmap('jet_r'))
                if i > 0: ax.xaxis.set_visible(False)
                if j > 0: ax.yaxis.set_visible(False)
                ax.set_xscale(bin_type)
                ax.set_yscale(bin_type)
                ax.set_xlim(xlim)
                ax.set_ylim(xlim)
        fig.savefig(os.path.splitext(cov_fn(args.corr_type))[0] + '.png')

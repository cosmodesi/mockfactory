"""
Replay the survey against a set of mock target catalogs, with :mod:`mockfactory.desi.altmtl`.

    srun -N1 -n1 -c 256 python run_altmtl.py --imocks 0 --numproc 32 --altmtl-dir $SCRATCH/altmtl
    srun -N1 -n1 -c 256 python run_altmtl.py --imocks 0-11 --numproc 9 --nummocks 12 \
         --altmtl-dir $SCRATCH/altmtl

Several mocks on one node at fewer workers each fill it better than one mock at many: the
assignment pool is not limited by per-tile CPU, so occupancy is what sets the cost. Twelve at
nine workers is the most a 512 GB node holds, the peak being the start, where every mock reads
its whole ``forFA`` file to build its state.

``OMP_NUM_THREADS`` is set to 1 here, and has to be: fiberassign is threaded, and without it
every worker spawns a thread per core and the pool runs slower than one mock alone.

Defaults describe the DA2 dark ``AbacusHF_DR2v2`` mocks -- 25 realizations, each with its own
quasar redshifts, which the updates need as ``zfix``. Point ``--forfa-dir`` elsewhere for
another set; ``--forfa`` and ``--zfix`` are formats taking the mock number.
"""

import argparse
import logging
import os
from pathlib import Path

logger = logging.getLogger('run_altmtl')

FORFA_DIR = '/dvs_ro/cfs/cdirs/desi/mocks/cai/LSS/DA2/mocks/AbacusHF_DR2v2'
#: The DA2 dark replay ends here: 6671 fa actions, the tiles of DA2 tiles-DARK.fits.
END_DATE = 20240418


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


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--altmtl-dir', required=True, help='where the replays are written')
    parser.add_argument('--forfa-dir', default=FORFA_DIR, help='where the mocks are read')
    parser.add_argument('--forfa', default='forFA{:d}.fits',
                        help='target catalog of one mock, relative to --forfa-dir')
    parser.add_argument('--zfix', default='qsos/qso{:d}.txt',
                        help='redshifts replacing the real ones on update, '
                             'relative to --forfa-dir; pass an empty string for none')
    parser.add_argument('--imocks', default='0', help='which mocks, e.g. 0, 0,3 or 0-11')
    parser.add_argument('--obscon', default='dark')
    parser.add_argument('--end-date', type=int, default=END_DATE)
    parser.add_argument('--numproc', type=int, default=32, help='workers per mock')
    parser.add_argument('--nummocks', type=int, default=1, help='mocks replayed side by side')
    args = parser.parse_args(args=args)

    os.environ['OMP_NUM_THREADS'] = '1'
    from mockfactory import setup_logging
    from mockfactory.desi.altmtl import run_mocks

    setup_logging()
    imocks = parse_imocks(args.imocks)
    mocks = []
    for imock in imocks:
        kwargs = {}
        if args.zfix:
            kwargs['zfix'] = Path(args.forfa_dir) / args.zfix.format(imock)
        mocks.append((Path(args.forfa_dir) / args.forfa.format(imock),
                      Path(args.altmtl_dir) / 'altmtl{:d}'.format(imock) / 'Univ000',
                      0, kwargs))
    logger.info('Replaying mock(s) {} to {:d}, {:d} at a time, {:d} worker(s) each.'
                .format(args.imocks, args.end_date, args.nummocks, args.numproc))
    results = run_mocks(mocks, args.end_date, obscon=args.obscon, numproc=args.numproc,
                        nummocks=args.nummocks)
    failed = [result['altmtl_dir'] for result in results if 'error' in result]
    total = 0.
    for result in results:
        if 'error' in result:
            logger.info('{altmtl_dir}: FAILED {error}'.format(**result))
        else:
            total += result['seconds']
            logger.info('{altmtl_dir}: {nactions:d} actions in {seconds:.0f} s '
                        '(setup {setup_seconds:.0f} s, state {state_seconds:.0f} s)'
                        .format(**result))
    if len(results) > len(failed):
        logger.info('{:d} mock(s) replayed, {:.2f} h of mock time in total.'
                    .format(len(results) - len(failed), total / 3600.))
    if failed:
        # run_mocks logs a failed mock and carries on; a job that lost mocks must not look done.
        raise SystemExit('{:d} mock(s) failed: {}'.format(len(failed), failed))
    logger.info('Done.')


if __name__ == '__main__':
    main()

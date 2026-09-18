"""Paths and small helpers shared by the alternative MTL modules."""

import os
import logging


logger = logging.getLogger('altmtl')


# Perlmutter serves a read-only, aggressively cached view of cfs; it is much faster for the
# many small survey files read by the loop, and it also makes accidental writes impossible.
DESI_ROOT = os.environ.get('DESI_ROOT_READONLY', '/dvs_ro/cfs/cdirs/desi')

# Per-tile fiberassign products of the real survey: rundate, field rotation, hour angle and
# the real fiber -> target assignment that the alternative assignment is matched against.
FIBERASSIGN_DIR = os.path.join(DESI_ROOT, 'target', 'fiberassign', 'tiles', 'trunk')
# Per-tile support files of the real survey: footprint, sky, secondary, gfa and too.
FIBERASSIGN_INPUT_DIR = os.path.join(DESI_ROOT, 'survey', 'fiberassign')
# Survey operations: which tiles were observed, when they were fiber-assigned and when the
# real ledgers were updated. Together these define the action list.
SURVEYOPS_DIR = os.path.join(DESI_ROOT, 'survey', 'ops', 'surveyops', 'trunk')
TILES_SPECSTATUS_FN = os.path.join(SURVEYOPS_DIR, 'ops', 'tiles-specstatus.ecsv')
MTL_DONE_TILES_FN = os.path.join(SURVEYOPS_DIR, 'mtl', 'mtl-done-tiles.ecsv')
MTL_DONE_VETOES_FN = os.path.join(SURVEYOPS_DIR, 'mtl', 'mtl-done-vetoes.ecsv')
# Redshift catalogs the real observations are taken from.
ZCAT_DIR = os.path.join(DESI_ROOT, 'spectro', 'redux', 'daily')


def mkdir(dirname):
    """Create directory ``dirname``, without complaining if it already exists."""
    os.makedirs(dirname, exist_ok=True)


def tile_string(tileid):
    """Return the 6-digit, zero-padded string that names all per-tile survey files."""
    return '{:06d}'.format(int(tileid))


def get_fiberassign_fn(tileid, fiberassign_dir=None):
    """Return the path to the real survey fiberassign file for ``tileid``."""
    if fiberassign_dir is None:
        fiberassign_dir = FIBERASSIGN_DIR
    ts = tile_string(tileid)
    return os.path.join(fiberassign_dir, ts[:3], 'fiberassign-{}.fits.gz'.format(ts))


def get_fiberassign_input_dir(tileid, survey='main', fiberassign_input_dir=None):
    """
    Return the directory holding the real survey per-tile inputs (footprint, sky, secondary,
    gfa, too) for ``tileid``. These are reused as is, so that the alternative assignment sees
    the same sky positions and the same focal plane state as the real one.
    """
    if fiberassign_input_dir is None:
        fiberassign_input_dir = FIBERASSIGN_INPUT_DIR
    ts = tile_string(tileid)
    return os.path.join(fiberassign_input_dir, survey.lower(), ts[:3])


def get_universe_dir(altmtl_dir, realization=0):
    """Return the directory holding one alternative realization of the ledgers."""
    return os.path.join(altmtl_dir, 'Univ{:03d}'.format(realization))


def iso_to_night(isodate):
    """Turn an ISO timestamp, or a yyyymmdd string, into the integer night yyyymmdd."""
    isodate = str(isodate)
    if 'T' in isodate:
        isodate = isodate.split('T')[0]
    return int(isodate.replace('-', ''))

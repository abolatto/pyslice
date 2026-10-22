# Created: 2026-06-01 00:00:00
# Last Modified: 2026-10-22 00:00:00
# Description: Python equivalent of Nepeaks.m (matlab_slice_06242026).
#   Extracts and fits spectra at seven pointings in M82 using the v2 MIRI MRS
#   channel cubes, produces [NeV], H I 7-6 and [NeIII] maps with make_map
#   (14-plane FITS cubes), and runs the ch2 line-list-restricted catalogs
#   (cat_peaks linelist + keep_lines + line_id).
#   Run from the project root or set DATA_DIR and OUT_DIR below.
# Last Edit: Sync with Nepeaks.m 06242026: v2 cubes, new pointings, makeMap parameters, ch2 linelist runs

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from slice_lib import read_fits, cat_peaks, make_map, keep_lines, line_id

# ---------------------------------------------------------------------------
# Configuration – adjust paths as needed
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'JWST')
OUT_DIR = DATA_DIR          # output files written alongside the input data

# MIRI MRS channel cubes (v2 reduction)
CUBES = {ch: os.path.join(DATA_DIR, f'M82_sky_v2_{ch}-shortmediumlong_s3d.fits')
         for ch in ('ch1', 'ch2', 'ch3', 'ch4')}

# Seven pointings (pka in Nepeaks.m): RA, Dec sexagesimal
PKA = [
    ('09:55:52.74',  '69:40:45.89'),
    ('09:55:51.276', '69:40:44.09'),
    ('09:55:54.461', '69:40:51.448'),
    ('09:55:54.610', '69:40:48.85'),
    ('09:55:51.58',  '69:40:45.87'),
    ('09:55:52.97',  '69:40:47.61'),
    ('09:55:53.95',  '69:40:48.48'),
]

# Output names and SNR thresholds used for the first block of Nepeaks.m
# (note the Matlab names skip Nepk5)
PK_NAMES = ['Nepk1', 'Nepk2', 'Nepk3', 'Nepk4', 'Nepk6', 'Nepk7', 'Nepk8']
SNRS = [5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]

# ch2 lines of interest ([NeVI], [ArIII], [SIV], [NiII], [NiIII], ...)
LINELIST = np.array([7.6527, 8.0258, 8.9921, 9.6659, 10.5110, 10.6833,
                     9.3948, 11.006, 8.763, 7.7836])


def run(ch='ch3'):
    """
    Extract and fit spectra for all seven pointings using the specified channel.

    Parameters
    ----------
    ch : str – one of 'ch1', 'ch2', 'ch3', 'ch4'
    """
    cube_file = CUBES[ch]
    if not os.path.exists(cube_file):
        raise FileNotFoundError(f'Cube not found: {cube_file}')

    print(f'Loading {os.path.basename(cube_file)} ...')
    cube = read_fits(cube_file, 'xten1')

    results = {}
    for (ra, dec), pk_name, snr in zip(PKA, PK_NAMES, SNRS):
        out_base = os.path.join(OUT_DIR, pk_name)
        print(f'\n=== {pk_name}  ({ra}, {dec}, r=0.2") ===')
        results[pk_name] = cat_peaks(cube, out_base, pixco=(ra, dec, 0.2), snr=snr)

    return results


def make_nev_map(flag='reject4'):
    """
    [NeV] 14.32 µm map as in Nepeaks.m / mkNeVmap.m.

    Writes JWST/NeV14.3-M82v2.fits — a 14-plane cube with all maps.
    """
    out_base = os.path.join(OUT_DIR, 'NeV14.3-M82v2')
    print('Making [NeV] 14.32 µm map ...')
    maps = make_map(CUBES['ch3'], out_base,
                    baseline=[14.27, 14.317, 14.3435, 14.355],
                    order=3,
                    intrange=[14.3185, 14.3440],
                    flag=flag)
    print(f'Map saved to {out_base}.fits')
    return maps


def make_hi76_map(flag=''):
    """
    H I 7-6 12.36 µm map.  Writes JWST/H7-6-M82v19.fits (name as in Nepeaks.m).
    """
    out_base = os.path.join(OUT_DIR, 'H7-6-M82v19')
    print('Making H I 7-6 12.36 µm map ...')
    maps = make_map(CUBES['ch3'], out_base,
                    baseline=[12.325, 12.3689, 12.4038, 12.47],
                    order=3,
                    intrange=[12.3687, 12.3914],
                    flag=flag)
    print(f'Map saved to {out_base}.fits')
    return maps


def make_neiii_map(flag=''):
    """
    [NeIII] 15.56 µm map.  Writes JWST/NeIII-M82v19.fits (name as in Nepeaks.m).
    """
    out_base = os.path.join(OUT_DIR, 'NeIII-M82v19')
    print('Making [NeIII] 15.56 µm map ...')
    maps = make_map(CUBES['ch3'], out_base,
                    baseline=[15.30, 15.5, 15.65, 15.85],
                    order=3,
                    intrange=[15.51, 15.62],
                    flag=flag)
    print(f'Map saved to {out_base}.fits')
    return maps


# Per-pointing ch2 settings from the last cells of Nepeaks.m:
# (snr, linelist scale factor, keep_lines tolerance or None, line_id vlsr)
CH2_RUNS = {
    1: (3,   1.0005, None,  200),
    2: (2,   1.0,    0.005, 100),
    3: (1.5, 1.001,  None,  300),
    4: (2,   1.0,    0.005, 100),
    5: (2,   1.0,    0.005, 100),
    6: (3,   1.001,  0.005, 300),
    7: (3,   1.001,  0.005, 300),
}


def run_ch2_linelist(sources=None):
    """
    ch2 catalogs restricted to LINELIST, then keep_lines and line_id,
    as in the last cells of Nepeaks.m.  Outputs Nepk{n}-ch2*.

    Parameters
    ----------
    sources : iterable of 1-based pointing numbers (default all seven)
    """
    cube = read_fits(CUBES['ch2'], 'xten1')
    for n in (sources or sorted(CH2_RUNS)):
        snr, fac, tol, vlsr = CH2_RUNS[n]
        ra, dec = PKA[n - 1]
        name = os.path.join(OUT_DIR, f'Nepk{n}-ch2')
        print(f'\n=== Nepk{n}-ch2 ===')
        cat_peaks(cube, name, pixco=(ra, dec, 0.2), snr=snr, linelist=LINELIST * fac)
        if tol is not None:
            keep_lines(name, LINELIST * fac, tol)
        line_id(name, vlsr)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract and fit MIRI MRS spectra at seven M82 pointings.')
    parser.add_argument('--channel', default='ch3',
                        choices=['ch1', 'ch2', 'ch3', 'ch4'],
                        help='MRS channel to process (default: ch3)')
    parser.add_argument('--maps', action='store_true',
                        help='Also generate [NeV], H I 7-6 and [NeIII] maps')
    parser.add_argument('--ch2lines', action='store_true',
                        help='Also run the ch2 line-list-restricted catalogs')
    parser.add_argument('--flag', default='',
                        help='Flag string for make_map (e.g. "reject4")')
    args = parser.parse_args()

    results = run(ch=args.channel)
    if args.maps:
        make_nev_map(flag=args.flag if args.flag else 'reject4')
        make_hi76_map(flag=args.flag)
        make_neiii_map(flag=args.flag)
    if args.ch2lines:
        run_ch2_linelist()

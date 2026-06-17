# Created: 2026-06-01 00:00:00
# Last Modified: 2026-06-03 00:00:00
# Description: Python equivalent of JWST/Nepeaks.m.
#   Extracts and fits spectra at seven pointings in M82 using all four MIRI
#   MRS channel cubes.  Produces peaklist, spectrum, continuum, and model
#   text files plus a PNG for each pointing/channel combination.
#   make_int_map now writes a single 12-plane cube FITS file per map.
#   Run from the project root or set DATA_DIR and OUT_DIR below.
# Last Edit: Sync with matlab_slice_06032026: single cube FITS output; default flag='reject4' for NeV map

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slice_lib import read_fits, cat_peaks, make_int_map

# ---------------------------------------------------------------------------
# Configuration – adjust paths as needed
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'JWST')
OUT_DIR = DATA_DIR          # output files written alongside the input data

# MIRI MRS channel cubes
CUBES = {
    'ch1': os.path.join(DATA_DIR, 'M82v1_19_1_sky_ch1-shortmediumlong_s3d.fits'),
    'ch2': os.path.join(DATA_DIR, 'M82v1_19_1_sky_ch2-shortmediumlong_s3d.fits'),
    'ch3': os.path.join(DATA_DIR, 'M82v1_19_1_sky_ch3-shortmediumlong_s3d.fits'),
    'ch4': os.path.join(DATA_DIR, 'M82v1_19_1_sky_ch4-shortmediumlong_s3d.fits'),
}

# Seven pointings used in Nepeaks.m (RA, Dec sexagesimal; aperture radius arcsec)
POINTINGS = [
    ('09:55:52.74',  '69:40:45.89', 0.2),
    ('09:55:51.276', '69:40:44.09', 0.2),
    ('09:55:54.461', '69:40:51.448', 0.2),
    ('09:55:54.610', '69:40:48.85', 0.2),
    ('09:55:51.276', '69:40:44.0',  0.2),
    ('09:55:51.58',  '69:40:45.87', 0.2),
    ('09:55:52.97',  '69:40:47.61', 0.2),
]

# SNR thresholds (default 5 for pk1; 1.5 for the rest, matching Matlab call)
SNRS = [5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]


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
    for idx, (ra, dec, rad) in enumerate(POINTINGS):
        pk_name = f'Nepk{idx+1}'
        out_base = os.path.join(OUT_DIR, pk_name)
        print(f'\n=== {pk_name}  ({ra}, {dec}, r={rad}\") ===')
        q = cat_peaks(
            cube,
            out_base,
            pixco=(ra, dec, rad),
            snr=SNRS[idx],
        )
        results[pk_name] = q

    return results


def make_nev_map(flag='reject4'):
    """
    Reproduce the [NeV] 14.33 µm integrated intensity map from Nepeaks.m.

    Writes JWST/NeV14.3-M82v19.fits — a 12-plane cube with all maps.
    Plane 1: integrated intensity (W/m²/sr)
    Plane 3: peak intensity (MJy/sr, baseline-subtracted)
    Plane 4: baseline RMS (MJy/sr)  ← new in 06032026

    Parameters
    ----------
    flag : str – passed to make_int_map.  Default 'reject4' matches Matlab
                 Nepeaks.m (sigma-clips baseline at 4σ).
    """
    cube_file = CUBES['ch3']
    out_base = os.path.join(OUT_DIR, 'NeV14.3-M82v19')
    print(f'Making [NeV] 14.33 µm map ...')
    maps = make_int_map(
        cube_file,
        out_base,
        baseline=[14.27, 14.3265, 14.3435, 14.355],
        order=1,
        intrange=[14.3210, 14.3440],
        flag=flag,
    )
    print(f'Map saved to {out_base}.fits')
    return maps


def make_hi76_map(flag=''):
    """
    Reproduce the H I 7-6 12.36 µm integrated intensity map.

    Writes JWST/H7-6-M82v19.fits — a 12-plane cube with all maps.

    Parameters
    ----------
    flag : str – passed to make_int_map (default '' = no sigma-clipping,
                 reject threshold stays at 5σ).  Matches Matlab Nepeaks.m.
    """
    cube_file = CUBES['ch3']
    out_base = os.path.join(OUT_DIR, 'H7-6-M82v19')
    print(f'Making H I 7-6 12.36 µm map ...')
    maps = make_int_map(
        cube_file,
        out_base,
        baseline=[12.325, 12.3689, 12.4038, 12.47],
        order=3,
        intrange=[12.3687, 12.3914],
        flag=flag,
    )
    print(f'Map saved to {out_base}.fits')
    return maps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract and fit MIRI MRS spectra at seven M82 pointings.')
    parser.add_argument('--channel', default='ch3',
                        choices=['ch1', 'ch2', 'ch3', 'ch4'],
                        help='MRS channel to process (default: ch3)')
    parser.add_argument('--maps', action='store_true',
                        help='Also generate [NeV] and H I 7-6 intensity maps')
    parser.add_argument('--flag', default='',
                        help='Flag string for make_int_map (e.g. "reject4")')
    args = parser.parse_args()

    results = run(ch=args.channel)
    if args.maps:
        make_nev_map(flag=args.flag if args.flag else 'reject4')
        make_hi76_map(flag=args.flag)

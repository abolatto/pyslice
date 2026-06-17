# Created: 2026-06-01 00:00:00
# Last Modified: 2026-06-03 00:00:00
# Description: Python translation of JWST MRS spectral analysis library,
#   updated to reflect matlab_slice_06032026 changes: make_int_map now adds
#   rmsmap (per-pixel baseline RMS) and writes a single 12-plane FITS cube
#   instead of separate intensity/uncertainty 2D maps.
#   Mirrors the Matlab library API as closely as possible.
# Last Edit: Sync with matlab_slice_06032026: rmsmap field, 12-plane cube FITS output

import os
import sys
import warnings
import numpy as np
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
from scipy import signal, optimize, interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Default catalog directory – catalog files remain in matlab_slice_06012026/
_CATDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'matlab_slice_06012026')


# ---------------------------------------------------------------------------
# Coordinate conversion utilities  (new in 06022026)
# ---------------------------------------------------------------------------

def sex2dec(a):
    """
    Sexagesimal to decimal conversion.

    Parameters
    ----------
    a : str or list of str
        Sexagesimal string(s).  Supported formats:
          '12:34:56.78'   colon-separated (degrees or hours — no conversion)
          '12h34m56.78s'  h/m/s notation
          '12d34m56.78s'  d/m/s notation
          '-12:34:56.78'  negative values
        The function mirrors Matlab sex2dec: it returns the raw decimal
        value with no unit conversion.  For RA (hours → degrees), multiply
        by 15 after calling.

    Returns
    -------
    float (scalar input) or ndarray (list input)
    """
    scalar = isinstance(a, str)
    if scalar:
        a = [a]

    results = []
    for s in a:
        s = s.strip()
        sgn = -1 if s.startswith('-') else 1
        s = s.lstrip('+-').strip()

        if ':' in s:
            parts = [float(x) for x in s.split(':')]
        elif 'h' in s.lower():
            s2 = s.lower().replace('h', ' ').replace('m', ' ').replace('s', ' ')
            parts = [float(x) for x in s2.split()]
        elif 'd' in s.lower():
            s2 = s.lower().replace('d', ' ').replace('m', ' ').replace('s', ' ')
            parts = [float(x) for x in s2.split()]
        else:
            raise ValueError(f'Unrecognised sexagesimal format: {s!r}')

        divisors = [1.0, 60.0, 3600.0]
        val = sum(p / divisors[i] for i, p in enumerate(parts[:3]))
        results.append(sgn * val)

    return results[0] if scalar else np.array(results)


def dec2sex(n, fmt=':', return_parts=False):
    """
    Decimal to sexagesimal conversion.

    Parameters
    ----------
    n : float or array-like
        Input decimal value(s).
    fmt : str
        Output format specifier:
          ':'  (default) → 'DD:MM:SS.sss'
          'h'            → 'DDh MMm SS.sssS'
          'd'            → 'DDd MMm SS.sssS'
          Append '+' to force explicit positive sign (e.g. ':+').
    return_parts : bool
        If True, return (h, m, s_int, s_frac, sign) instead of string(s).

    Returns
    -------
    str (scalar) or list of str (array), or tuple/list of tuples if
    return_parts is True.
    """
    force_plus = len(fmt) > 1 and fmt[1] == '+'
    sep = fmt[0]

    scalar = np.isscalar(n)
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    results = []

    for ni in n_arr:
        sg = -1 if ni < 0 else 1
        nabs = abs(ni)
        h_val = int(nabs)
        m_val = int(60 * (nabs - h_val) + 1e-4)
        s_val = 3600 * (nabs - h_val - m_val / 60.0)
        s_int = int(s_val + 1e-4)
        s_frac = int(round(1000 * (s_val - int(s_val + 1e-4))))

        if return_parts:
            results.append((sg * h_val, m_val, s_int, s_frac / 1000.0, sg))
            continue

        plus = '+' if (force_plus and sg > 0) else ''
        sign = '-' if sg < 0 else plus
        habs = abs(h_val)

        if sep == ':':
            result = f'{sign}{habs:02d}:{m_val:02d}:{s_int:02d}.{s_frac:03d}'
        elif sep == 'h':
            result = f'{sign}{habs:02d}h {m_val:02d}m {s_int:02d}.{s_frac:03d}s'
        elif sep == 'd':
            result = f'{sign}{habs:02d}d {m_val:02d}m {s_int:02d}.{s_frac:03d}s'
        else:
            result = f'{sign}{habs:02d} {m_val:02d} {s_int:02d}.{s_frac:03d}'
        results.append(result)

    return results[0] if scalar else results


# ---------------------------------------------------------------------------
# Private coordinate helpers (used by extract_aperture)
# ---------------------------------------------------------------------------

def _sex2dec_ra(s):
    """Sexagesimal RA string 'HH:MM:SS.s' to decimal degrees."""
    return Angle(s.replace(':', 'h', 1).replace(':', 'm', 1) + 's').deg


def _sex2dec_dec(s):
    """Sexagesimal Dec string 'DD:MM:SS.s' to decimal degrees."""
    return Angle(s.replace(':', 'd', 1).replace(':', 'm', 1) + 's').deg


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def read_fits(fname, options=''):
    """
    Read FITS file, returning a dict matching the Matlab rfits structure.

    Parameters
    ----------
    fname : str or dict
        FITS filename, or a pre-loaded rfits dict (returned as-is).
    options : str
        Space-separated flags.  Key options:
          'head'    – return header info only, skip data read
          'xtenN'  – read HDU extension N (default 0 = PRIMARY; use 1 for SCI)
          'nowcs'  – skip WCS array computation (default: compute if string passed)
          'noabs'  – produce relative pixel coordinates, not absolute world coords
          'verbose' – print progress messages

    Returns
    -------
    dict with keys:
        data    – ndarray (nx, ny, nz) for 3-D cubes (matches Matlab ordering)
        numpt   – [nx, ny, nz]
        crval   – reference values per axis
        crpix   – reference pixels per axis (1-based, FITS convention)
        cdelt   – pixel scale per axis
        ctype   – list of axis type strings
        cunit   – list of axis unit strings
        bunit   – brightness unit string
        x       – list of coordinate arrays: x[2] is wavelength (1-D, absolute)
                  x[0]/x[1] are 2-D spatial grids, shape (ny, nx)
        wcs1    – 2-D RA grid (ny, nx), degrees  (computed unless 'nowcs')
        wcs2    – 2-D Dec grid (ny, nx), degrees (computed unless 'nowcs')
        header  – dict copy of the FITS header
    """
    if isinstance(fname, dict):
        return fname

    head_only = 'head' in options
    # Matlab default when a string option is passed: compute WCS unless 'nowcs'
    if options:
        do_wcs = 'nowcs' not in options
    else:
        do_wcs = False
    no_abs = 'noabs' in options
    verbose = 'verbose' in options

    xten = 0
    if 'xten' in options:
        idx = options.find('xten')
        xten = int(options[idx + 4])

    with fits.open(fname) as hdulist:
        hdu = hdulist[xten]
        hdr = hdu.header
        naxis = hdr['NAXIS']

        r = {}
        r['naxis'] = naxis
        r['bitpix'] = hdr.get('BITPIX', -32)
        r['numpt'] = np.array([hdr.get(f'NAXIS{i+1}', 0) for i in range(naxis)],
                               dtype=int)
        r['crval'] = np.array([hdr.get(f'CRVAL{i+1}', 0.0) for i in range(naxis)])
        r['crpix'] = np.array([hdr.get(f'CRPIX{i+1}', 1.0) for i in range(naxis)])
        r['cdelt'] = np.array([hdr.get(f'CDELT{i+1}', 1.0) for i in range(naxis)])
        r['crota'] = np.array([hdr.get(f'CROTA{i+1}', 0.0) for i in range(naxis)])
        r['ctype'] = [hdr.get(f'CTYPE{i+1}', '') for i in range(naxis)]
        r['cunit'] = [hdr.get(f'CUNIT{i+1}', '') for i in range(naxis)]
        r['bunit'] = hdr.get('BUNIT', '')
        r['bscale'] = hdr.get('BSCALE', 1.0)
        r['bzero'] = hdr.get('BZERO', 0.0)
        r['header'] = dict(hdr)

        for key in ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2'):
            if key in hdr:
                r[key.lower()] = float(hdr[key])

        has_cd = 'cd1_1' in r
        has_pc = 'pc1_1' in r

        nx = int(r['numpt'][0])
        ny = int(r['numpt'][1]) if naxis > 1 else 1
        nz = int(r['numpt'][2]) if naxis > 2 else 1

        # 1-based pixel offsets, matching Matlab (1:N) - crpix
        xv1 = np.arange(1, nx + 1) - r['crpix'][0]
        xv2 = np.arange(1, ny + 1) - r['crpix'][1] if naxis > 1 else np.array([0.0])

        r['x'] = [None] * naxis
        dcp = False

        if (has_cd or has_pc) and nx > 0 and ny > 0:
            # meshgrid(xv1, xv2) – same convention as Matlab: shape (ny, nx)
            xx, yy = np.meshgrid(xv1, xv2)
            if has_cd:
                r['x'][0] = r['cd1_1'] * xx + r.get('cd1_2', 0.0) * yy
                r['x'][1] = r.get('cd2_1', 0.0) * xx + r.get('cd2_2', 0.0) * yy
            else:
                r['x'][0] = r['cdelt'][0] * (r['pc1_1'] * xx + r.get('pc1_2', 0.0) * yy)
                r['x'][1] = r['cdelt'][1] * (r.get('pc2_1', 0.0) * xx + r['pc2_2'] * yy)
            if naxis > 2:
                r['x'][2] = (np.arange(1, nz + 1) - r['crpix'][2]) * r['cdelt'][2]
            dcp = True
        else:
            for i in range(naxis):
                ni = int(r['numpt'][i])
                r['x'][i] = (np.arange(1, ni + 1) - r['crpix'][i]) * r['cdelt'][i]

        # Absolute world coordinates (add CRVAL)
        if not no_abs:
            for i in range(naxis):
                r['x'][i] = r['x'][i] + r['crval'][i]
            r['coormode'] = 'ABSCOOR'
        else:
            for i in range(2, naxis):
                r['x'][i] = r['x'][i] + r['crval'][i]
            r['coormode'] = 'RELCOOR'

        # Spherical WCS (TAN / SIN projection)
        if do_wcs and r['ctype']:
            ctype_str = r['ctype'][0] if r['ctype'] else ''
            if len(ctype_str) >= 8:
                proj = ctype_str[5:8]
                # Offsets from reference in degrees
                if dcp:
                    dx = r['x'][0] - r['crval'][0]
                    dy = r['x'][1] - r['crval'][1]
                else:
                    dy, dx = np.meshgrid(r['x'][1] - r['crval'][1],
                                         r['x'][0] - r['crval'][0])
                wcs_ok = False
                if proj == 'TAN':
                    phi = np.arctan2(dy, dx) + np.pi / 2
                    theta = np.arctan(180.0 / (np.pi * np.sqrt(dx**2 + dy**2)))
                    wcs_ok = True
                elif proj == 'SIN':
                    phi = np.arctan2(dy, dx) + np.pi / 2
                    theta = np.arccos(np.clip(np.pi / 180 * np.sqrt(dx**2 + dy**2),
                                              -1, 1))
                    wcs_ok = True
                if wcs_ok:
                    dc = r['crval'][1] * np.pi / 180
                    ac = r['crval'][0] * np.pi / 180
                    sd = (np.sin(theta) * np.sin(dc)
                          - np.cos(theta) * np.cos(phi) * np.cos(dc))
                    csina = np.cos(theta) * np.sin(phi)
                    ccosa = (np.sin(theta) * np.cos(dc)
                             + np.cos(theta) * np.cos(phi) * np.sin(dc))
                    r['wcs1'] = 180 / np.pi * (ac + np.arctan2(csina, ccosa))
                    neg = r['wcs1'] < 0
                    r['wcs1'][neg] += 360
                    r['wcs2'] = 180 / np.pi * np.arctan2(
                        sd, np.sqrt(csina**2 + ccosa**2))

        if head_only:
            return r

        if hdu.data is not None:
            if verbose:
                print(f'read_fits: reading data from extension {xten}')
            raw = np.array(hdu.data, dtype=np.float64)
            # astropy returns (nz, ny, nx) – transpose to (nx, ny, nz) like Matlab
            r['data'] = np.transpose(raw)
        r['_fname'] = os.path.basename(fname)

    return r


def write_fits(r, fname, options=''):
    """
    Write a dict (from read_fits) to a FITS file.

    Only the data and basic WCS keywords are written.  Extra fields stored
    in r (besides 'data', 'x', 'numpt', 'header', 'wcs1', 'wcs2',
    'coormode') are written as additional header keywords.
    """
    avoid = {'data', 'x', 'numpt', 'header', 'wcs1', 'wcs2', 'coormode',
             'bscale', 'bzero', 'bitpix', 'naxis'}

    data = r['data']
    # Transpose back to (nz, ny, nx) for writing
    if data.ndim == 3:
        out_data = np.transpose(data).astype(np.float32)
    else:
        out_data = data.astype(np.float32)

    hdr = fits.Header()
    naxis = r.get('naxis', data.ndim)
    numpt = r.get('numpt', np.array(out_data.shape[::-1]))

    for i in range(naxis):
        hdr[f'CTYPE{i+1}'] = r['ctype'][i] if i < len(r.get('ctype', [])) else ''
        hdr[f'CRPIX{i+1}'] = r['crpix'][i] if i < len(r.get('crpix', [])) else 1.0
        hdr[f'CRVAL{i+1}'] = r['crval'][i] if i < len(r.get('crval', [])) else 0.0
        hdr[f'CDELT{i+1}'] = r['cdelt'][i] if i < len(r.get('cdelt', [])) else 1.0
        if r.get('cunit') and i < len(r['cunit']) and r['cunit'][i]:
            hdr[f'CUNIT{i+1}'] = r['cunit'][i]

    for key in ('cd1_1', 'cd1_2', 'cd2_1', 'cd2_2',
                'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2'):
        if key in r:
            hdr[key.upper()] = r[key]

    if r.get('bunit'):
        hdr['BUNIT'] = r['bunit']

    # Write extra scalar fields
    for k, v in r.items():
        if k in avoid:
            continue
        if np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0):
            kup = k.upper()[:8]
            if kup not in hdr:
                hdr[kup] = v

    hdu = fits.PrimaryHDU(data=out_data, header=hdr)
    hdu.writeto(fname, overwrite=True)


# ---------------------------------------------------------------------------
# Circular aperture weight map (partial pixel overlap)
# ---------------------------------------------------------------------------

def aperture_weights(x, y, xc, yc, psize, rad):
    """
    Fractional contribution of each square pixel (side psize) to a circular
    aperture of radius rad centred on (xc, yc).

    x, y : 2-D coordinate arrays of pixel centres (same shape)
    Returns weight array of same shape as x.
    """
    weight = np.zeros_like(x, dtype=float)
    c1 = np.sqrt((x + psize/2 - xc)**2 + (y + psize/2 - yc)**2)
    c2 = np.sqrt((x - psize/2 - xc)**2 + (y + psize/2 - yc)**2)
    c3 = np.sqrt((x - psize/2 - xc)**2 + (y - psize/2 - yc)**2)
    c4 = np.sqrt((x + psize/2 - xc)**2 + (y - psize/2 - yc)**2)
    nc = (c1 <= rad).astype(int) + (c2 <= rad) + (c3 <= rad) + (c4 <= rad)
    weight[nc == 4] = 1.0
    partial = np.where((nc > 0) & (nc < 4))
    if len(partial[0]) > 0:
        nsamp = 10000
        rng = np.random.default_rng(seed=42)
        xy_rand = psize * (rng.random((nsamp, 2)) - 0.5)
        for idx in zip(*partial):
            xi, yi = x[idx], y[idx]
            dist2 = ((xi + xy_rand[:, 0] - xc)**2
                     + (yi + xy_rand[:, 1] - yc)**2)
            weight[idx] = np.sum(dist2 <= rad**2) / nsamp
    return weight


# ---------------------------------------------------------------------------
# Aperture spectrum extraction
# ---------------------------------------------------------------------------

def extract_aperture(r, pixco):
    """
    Extract a spectrum from a FITS cube dict over a specified region.

    Parameters
    ----------
    r : dict
        FITS cube dict from read_fits.
    pixco : various
        Extraction region specification (matching Matlab catPeaks convention):
          [row, col]           – single pixel (0-based integer indices)
          [[row,...],[col,...]]– list of pixels to median-average
          scalar               – average entire cube
          {'RA', 'Dec', rad}   – circular aperture; RA/Dec sexagesimal strings
                                 or decimal degrees; rad in arcsec
          (float, float, float)– circular aperture with decimal RA, Dec, rad

    Returns
    -------
    s      : 1-D spectrum (nz,)
    region : string describing the extraction region
    mask   : 2-D weight/mask array (nx, ny)
    flag   : 0 = OK; -1 = all NaN; -2 = >5% NaN channels
    """
    data = r['data']  # (nx, ny, nz)
    nx, ny, nz = data.shape
    mask = np.zeros((nx, ny))
    flag = 0

    # --- single pixel ---
    if (isinstance(pixco, (list, np.ndarray))
            and np.ndim(pixco) == 1
            and len(pixco) == 2
            and not isinstance(pixco[0], str)):
        row, col = int(pixco[0]), int(pixco[1])
        s = data[row, col, :]
        mask[row, col] = 1
        region = 'pixel'

    # --- list of pixels ---
    elif (isinstance(pixco, np.ndarray) and pixco.ndim == 2
          and pixco.shape[0] == 2 and pixco.shape[1] > 1):
        rows = pixco[0, :].astype(int)
        cols = pixco[1, :].astype(int)
        s = np.nanmedian(data[rows, cols, :], axis=0)
        mask[rows, cols] = 1
        region = 'list'

    # --- average entire cube ---
    elif np.isscalar(pixco) or (isinstance(pixco, np.ndarray)
                                 and pixco.size == 1):
        s = np.nanmedian(data.reshape(-1, nz), axis=0)
        mask[:] = 1
        region = 'all'

    # --- circular aperture ---
    elif (isinstance(pixco, (list, tuple)) and len(pixco) == 3):
        p0, p1, rad = pixco[0], pixco[1], float(pixco[2])

        if isinstance(p0, str):
            rac = _sex2dec_ra(p0)
            dec = _sex2dec_dec(p1)
        else:
            rac, dec = float(p0), float(p1)

        if 'wcs1' not in r or 'wcs2' not in r:
            raise ValueError('wcs1/wcs2 not in cube dict – load with read_fits '
                             'using a string option (e.g. "xten1") to trigger WCS')

        # Offsets in arcsec; wcs1/wcs2 have shape (ny, nx) → transpose to (nx, ny)
        cos_dec = np.cos(np.deg2rad(r['crval'][1]))
        xx = ((r['wcs1'] - rac) * cos_dec * 3600).T   # (nx, ny)
        yy = ((r['wcs2'] - dec) * 3600).T              # (nx, ny)
        psize_arcsec = abs(r['cdelt'][1]) * 3600       # pixel size in arcsec

        weight = aperture_weights(xx, yy, 0.0, 0.0, psize_arcsec, rad)
        sw = weight / np.nansum(weight)

        s = np.zeros(nz)
        ck = np.zeros(nz)
        for k in range(nz):
            plane = data[:, :, k]          # (nx, ny)
            pw = plane * sw
            valid_sw = sw.copy()
            valid_sw[np.isnan(plane)] = 0.0
            ck[k] = np.nansum(valid_sw)
            s[k] = np.nansum(pw)
            if s[k] == 0:
                s[k] = np.nan
            if ck[k] < 0.25:
                s[k] = np.nan
            else:
                s[k] /= ck[k]

        if np.all(np.isnan(s)):
            print('Aperture is empty (probably outside the cube)')
            return s, 'circle', weight, -1
        if np.sum(np.isnan(s)) / len(s) > 0.05:
            print('A lot of NaNs in the spectrum (close to an edge)')
            return s, 'circle', weight, -2

        mask = weight
        region = 'circle'
    else:
        raise ValueError('Unsupported pixco format')

    if np.all(np.isnan(s)):
        flag = -1
    elif np.sum(np.isnan(s)) / len(s) > 0.05:
        flag = -2

    return s, region, mask, flag


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def sig_chans(xx, yy, noise, snr, thr):
    """
    Detect significant peaks in a 1-D spectrum.

    Parameters
    ----------
    xx    : wavelength array (same length as yy)
    yy    : cross-correlation amplitude array
    noise : noise estimate (e.g. mean absolute deviation of yy)
    snr   : S/N threshold multiplier (threshold = snr * noise)
    thr   : minimum channel separation between independent peaks

    Returns
    -------
    pp : (N, 2) array of [wavelength, amplitude] for each detected peak
    """
    results = []

    ix = np.where(yy > snr * noise)[0]
    if len(ix) == 0:
        return np.zeros((0, 2))

    dix = np.diff(ix)
    jump_pos = np.where(dix > thr)[0]
    # ul[j] = last index in ix that belongs to group j
    ul = np.concatenate([jump_pos, [len(ix) - 1]])

    for j in range(len(ul)):
        if j == 0:
            group_ix = ix[:ul[0] + 1]
        else:
            group_ix = ix[ul[j-1] + 1:ul[j] + 1]

        first_ch = group_ix[0]
        last_ch = group_ix[-1]
        st = first_ch - 1 if first_ch > 0 else first_ch
        en = last_ch + 1 if last_ch < len(xx) - 1 else last_ch

        xl = xx[st:en + 1]
        yl = yy[st:en + 1]

        if len(xl) == 2:
            k = int(np.argmax(yl))
            xc = xl[k]
            ixc_arr = np.where(xx == xc)[0]
            if len(ixc_arr) == 0 or ixc_arr[0] <= 1 or ixc_arr[0] >= len(xx) - 2:
                results.append((xc, float(yl[k])))
                continue
            ixc = ixc_arr[0]
            xp = xx[ixc - 2:ixc + 3]
            yp = yy[ixc - 2:ixc + 3]
            pcoef = np.polyfit(xp, yp, 2)
            if pcoef[0] > 0:
                results.append((float(xl[k]), float(yl[k])))
            else:
                x_pk = -pcoef[1] / (2 * pcoef[0])
                y_pk = pcoef[2] - pcoef[1]**2 / (4 * pcoef[0])
                results.append((float(x_pk), float(y_pk)))
        else:
            pks, _ = signal.find_peaks(yl, prominence=noise / 2)
            if len(pks) == 0:
                pks = [int(np.argmax(yl))]

            for k in pks:
                xc = xl[k]
                ixc_arr = np.where(xx == xc)[0]
                if (len(ixc_arr) == 0
                        or ixc_arr[0] == 0
                        or ixc_arr[0] == len(xx) - 1):
                    results.append((float(xc), float(yl[k])))
                    continue
                ixc = ixc_arr[0]
                lo, hi = max(0, ixc - 1), min(len(xx) - 1, ixc + 1)
                xp = xx[lo:hi + 1]
                yp = yy[lo:hi + 1]
                if len(xp) < 3:
                    results.append((float(xc), float(yl[k])))
                    continue
                pcoef = np.polyfit(xp, yp, 2)
                if pcoef[0] > 0:
                    results.append((float(xl[k]), float(yl[k])))
                else:
                    x_pk = -pcoef[1] / (2 * pcoef[0])
                    y_pk = pcoef[2] - pcoef[1]**2 / (4 * pcoef[0])
                    results.append((float(x_pk), float(y_pk)))

    if not results:
        return np.zeros((0, 2))
    return np.array(results)


# ---------------------------------------------------------------------------
# Gaussian fitting
# ---------------------------------------------------------------------------

def gaussmfit(x, y, K, S, X0, tol=1e-7, graphout=0):
    """
    Multi-component Gaussian fit using Levenberg-Marquardt (scipy.optimize).

    Parameters
    ----------
    x, y : 1-D arrays – independent variable and data
    K, S, X0 : initial guesses for amplitude, sigma, and centre of each component
    tol : convergence tolerance for curve_fit
    graphout : unused (kept for API compatibility)

    Returns
    -------
    Ko, So, X0o : fitted parameters (1-D arrays, length = n_components)
    chi2   : reduced chi-squared
    ymodel : best-fit model evaluated on x
    err    : 0 = OK; 1 = did not converge; 4 = ill-conditioned covariance
    epar   : parameter uncertainties, length 3*n_components, order [K,S,X0 per comp]
    """
    K = np.atleast_1d(np.asarray(K, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    X0 = np.atleast_1d(np.asarray(X0, dtype=float))
    n_comp = len(K)

    p0 = np.empty(3 * n_comp)
    for i in range(n_comp):
        p0[3*i], p0[3*i+1], p0[3*i+2] = K[i], S[i], X0[i]

    def multi_gauss(xv, *params):
        result = np.zeros_like(xv, dtype=float)
        for ci in range(n_comp):
            Ki, Si, X0i = params[3*ci], params[3*ci+1], params[3*ci+2]
            if Si != 0:
                result += Ki * np.exp(-((xv - X0i)**2) / (2 * Si**2))
        return result

    err = 0
    epar = np.zeros(3 * n_comp)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, pcov = optimize.curve_fit(
                multi_gauss, x, y, p0=p0,
                maxfev=50000, ftol=tol, xtol=tol
            )
        diag = np.diag(pcov)
        if np.any(~np.isfinite(diag)):
            err = 4
            popt = p0.copy()
        else:
            # pcov from curve_fit already contains chi2_red scaling (sigma=None case)
            epar = np.sqrt(np.abs(diag))
    except (RuntimeError, optimize.OptimizeWarning):
        err = 1
        popt = p0.copy()

    Ko = popt[0::3].copy()
    So = np.abs(popt[1::3]).copy()
    X0o = popt[2::3].copy()
    ymodel = multi_gauss(x, *popt)

    residuals = y - ymodel
    dof = max(len(x) - 3 * n_comp, 1)
    chi2 = np.sum(residuals**2) / dof

    if graphout and n_comp > 0:
        plt.figure()
        plt.plot(x, y, 'o', ms=3)
        plt.plot(x, ymodel, '-')
        plt.show()

    return Ko, So, X0o, chi2, ymodel, err, epar


def gaussfit(x, y, K=None, S=None, X0=None, tol=1e-4,
             graphout=0, mask=None, ey=None):
    """
    Single-component Gaussian fit (wrapper around gaussmfit for API parity).
    Returns Ko, So, X0o, chi2, ymodel, error, epar (each scalar or length-1).
    """
    if K is None:
        K = float(np.max(y))
        idx = int(np.argmax(y))
        X0 = float(x[idx])
        S = float(np.sqrt(np.sum(np.abs(y) * (x - X0)**2) / np.sum(np.abs(y))))
        if S > (np.max(x) - np.min(x)) / 10:
            S = (np.max(x) - np.min(x)) / 10

    Ko, So, X0o, chi2, ymodel, err, epar = gaussmfit(
        x, y, [K], [S], [X0], tol=tol, graphout=graphout
    )
    return Ko[0], So[0], X0o[0], chi2, ymodel, err, epar


# ---------------------------------------------------------------------------
# Line fitting
# ---------------------------------------------------------------------------

def fit_lines(wv, spec, peaks):
    """
    Fit Gaussians to detected peaks after continuum subtraction.

    Parameters
    ----------
    wv    : 1-D wavelength array (microns)
    spec  : 1-D spectrum array (MJy/sr)
    peaks : (N, 2) array of [wavelength, amplitude] for each peak

    Returns
    -------
    q      : dict with arrays K, S, X0, area, eK, eS, eX0, earea, flag (length N)
    ymodel : Gaussian model array (same length as wv)
    smo    : filtered pseudo-continuum array (same length as wv)
    """
    c = 3e8            # m/s
    peaklist = peaks[:, 0]
    peakamp = peaks[:, 1]
    nl = len(peaklist)

    q = {
        'K':     np.full(nl, np.nan),
        'S':     np.full(nl, np.nan),
        'X0':    np.full(nl, np.nan),
        'area':  np.full(nl, np.nan),
        'eK':    np.full(nl, np.nan),
        'eS':    np.full(nl, np.nan),
        'eX0':   np.full(nl, np.nan),
        'earea': np.full(nl, np.nan),
        'flag':  np.full(nl, -1, dtype=int),
    }

    # Adaptive mask width around each peak (wider for brighter lines)
    maskw = (3 + np.round(np.abs(np.log10(np.abs(peakamp) + 1e-30)) - 1) * 2
             ).astype(int)
    maskw[maskw <= 0] = 3

    # Build mask and cont (cell-of-line-indices per channel)
    mask = np.zeros(len(wv), dtype=int)
    cont = [[] for _ in range(len(wv))]

    for i in range(nl):
        idx = int(np.argmin(np.abs(peaklist[i] - wv)))
        mw = int(maskw[i])
        st = max(idx - mw, 0)
        en = min(idx + mw, len(wv) - 1)
        mask[st:en+1] += 1
        for j in range(st, en+1):
            cont[j].append(i)

    include = np.where(mask == 0)[0]
    # Linear interpolation through masked regions
    f_interp = interpolate.interp1d(
        wv[include], spec[include], kind='linear',
        bounds_error=False, fill_value=np.nan
    )
    base = f_interp(wv)
    nan_mask = np.isnan(base)
    base[nan_mask] = spec[nan_mask]

    # Low-pass FIR filter for pseudo-continuum
    forder = 20
    if len(base) <= 60:
        forder = max(int(len(base) / 3.5), 3)
    cutoff = 0.005 if np.min(wv) > 4 else 0.25
    B = signal.firwin(forder + 1, cutoff)
    try:
        smo = signal.filtfilt(B, 1.0, base)
    except Exception:
        print('filtfilt failed in fit_lines')
        smo = base.copy()

    lines = spec - smo                         # continuum-subtracted spectrum
    rms = np.std(lines[mask == 0])

    fitreg = 15                                # fit window half-width in channels
    ymodel = np.zeros(len(wv))

    for i in range(nl):
        idx = int(np.argmin(np.abs(peaklist[i] - wv)))
        st = max(idx - fitreg, 0)
        en = min(idx + fitreg, len(wv) - 1)

        wvreg = wv[st:en+1]
        spreg = lines[st:en+1]

        # All lines that contribute in this window
        contrib_set = set()
        for j in range(st, en+1):
            contrib_set.update(cont[j])
        contrib = sorted(contrib_set)
        if not contrib:
            contrib = [i]

        X0_init = peaklist[list(np.array(contrib))]
        S_init = X0_init / 1000 / 2.35
        if len(contrib) > 1:
            S_init /= 2
        K_init = np.array([
            spreg[np.argmin(np.abs(wvreg - peaklist[ci]))]
            for ci in contrib
        ], dtype=float)

        Ko, So, X0o, chi2, yz, err, epar = gaussmfit(
            wvreg, spreg, K_init, S_init, X0_init, tol=1e-7
        )

        # Extract this peak's component from the multi-component fit
        try:
            ix = list(contrib).index(i)
        except ValueError:
            ix = 0

        q['K'][i] = Ko[ix]
        q['eK'][i] = epar[ix * 3]
        q['S'][i] = So[ix]
        q['eS'][i] = epar[ix * 3 + 1]
        q['X0'][i] = X0o[ix]
        q['eX0'][i] = epar[ix * 3 + 2]

        # Integrated flux in W/m²/sr
        nu = c / (1e-6 * q['X0'][i])          # Hz
        dnu = nu * q['S'][i] / q['X0'][i]
        ednu = ((nu / q['X0'][i])
                * np.sqrt(q['eS'][i]**2
                          + 4 * q['S'][i]**2 / q['X0'][i]**2 * q['eX0'][i]**2))
        q['area'][i] = 1e-20 * np.sqrt(2 * np.pi) * q['K'][i] * dnu
        q['earea'][i] = (1e-20 * np.sqrt(2 * np.pi)
                         * np.sqrt(dnu**2 * q['eK'][i]**2
                                   + q['K'][i]**2 * ednu**2))

        q['flag'][i] = err

        # Additional flag bits (matching Matlab convention)
        chan_spacing = np.mean(np.diff(wvreg)) if len(wvreg) > 1 else 1
        if np.abs(X0o[ix] - X0_init[ix]) > 2 * chan_spacing:
            q['flag'][i] += 8
        if q['K'][i] < 0:
            q['flag'][i] += 16
        if np.sqrt(chi2) > 3 * rms:
            q['flag'][i] += 32
        if q['X0'][i] > np.max(wvreg) or q['X0'][i] < np.min(wvreg):
            q['flag'][i] += 64

        if np.isfinite(q['S'][i]) and q['S'][i] != 0:
            ymodel += q['K'][i] * np.exp(
                -((wv - q['X0'][i]) / (np.sqrt(2) * q['S'][i]))**2
            )

    return q, ymodel, smo


# ---------------------------------------------------------------------------
# Main cataloguing function
# ---------------------------------------------------------------------------

def cat_peaks(fname, outname, pixco=0, snr=5, thr=2, nchunk=1):
    """
    Create a catalog of significant Gaussian peaks for a spectrum extracted
    from a JWST datacube.

    Mirrors Matlab catPeaks.m.  Writes four text files:
      {outname}-peaklist.txt  – Gaussian fit results
      {outname}_spec.txt      – raw spectrum
      {outname}_cont.txt      – pseudo-continuum
      {outname}_model.txt     – Gaussian model (continuum excluded)
    Also saves {outname}.png (or chunk PNGs) of the spectrum.

    Parameters
    ----------
    fname   : str or dict – FITS filename or pre-loaded rfits dict
    outname : str – base name for output files (may include path)
    pixco   : extraction region (see extract_aperture); default 0 = whole cube
    snr     : minimum S/N for peak detection (default 5)
    thr     : minimum channel separation between peaks (default 2)
    nchunk  : number of chunks, or [start_um, end_um] wavelength range

    Returns
    -------
    q : dict with Gaussian fit results (same fields as fit_lines output)
        q['flag'] is set to the extraction flag if extraction failed.
    """
    if isinstance(fname, str):
        r = read_fits(fname, 'xten1')
    else:
        r = fname
        fname = r.get('_fname', 'datacube')

    if not isinstance(nchunk, (list, np.ndarray)):
        nchunk_val = int(nchunk)
    else:
        nchunk_val = 1

    # Threshold scale: divide by 1.5 to convert Gaussian sigma to MAD
    snr_scaled = snr / 1.5

    s, region, mask, flag = extract_aperture(r, pixco)
    if flag != 0:
        return {'flag': flag}

    wv_all = r['x'][2]
    valid = ~np.isnan(s)
    x = wv_all[valid]
    y = s[valid]

    # Determine wavelength range / chunk boundaries
    stw, enw = None, None
    if isinstance(nchunk, (list, np.ndarray)) and len(nchunk) == 2:
        stw, enw = float(nchunk[0]), float(nchunk[1])
        ix1 = int(np.argmin(np.abs(stw - x)))
        ix2 = int(np.argmin(np.abs(enw - x)))
        nchunk_val = 1
    elif nchunk_val > 1:
        chunk = len(y) // nchunk_val

    mf = np.array([1, 1, 0, -1, -1], dtype=float)   # matched filter
    pp_all = []
    q_combined = None
    ymo_combined = None
    cont_combined = None

    for ci in range(nchunk_val):
        if stw is not None:
            st, en = ix1, ix2
        elif nchunk_val > 1:
            st = ci * chunk
            en = min((ci + 1) * chunk, len(y)) - 1
        else:
            st, en = 0, len(y) - 1

        xx = x[st:en + 1]
        yy = y[st:en + 1]

        dy = np.diff(yy)
        dx = (xx[1:] + xx[:-1]) / 2
        cc = np.convolve(dy, -mf, mode='same')
        # Mean absolute deviation of cc
        scc = np.mean(np.abs(cc - np.mean(cc)))

        pp = sig_chans(dx, cc, scc, snr_scaled / 1.5, thr)
        nr = len(pp)
        pp_all.append(pp)

        q_chunk, ymo, cont = fit_lines(xx, yy, pp)

        if q_combined is None:
            q_combined = q_chunk
            ymo_combined = ymo
            cont_combined = cont
        else:
            for fld in ('K', 'S', 'X0', 'area', 'eK', 'eS', 'eX0', 'earea', 'flag'):
                q_combined[fld] = np.concatenate([q_combined[fld], q_chunk[fld]])
            ymo_combined = np.concatenate([ymo_combined, ymo])
            cont_combined = np.concatenate([cont_combined, cont])

        # --- Plot spectrum with peak markers ---
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(xx, yy)
        for j in range(nr):
            idx_peak = int(np.argmin(np.abs(pp[j, 0] - xx)))
            ty_idx = min(idx_peak + 5, len(yy) - 1)
            ty = yy[ty_idx] * 0.99
            ax.axvline(pp[j, 0], color='r', linestyle='-.')
            ax.text(pp[j, 0], ty, str(len(pp_all) - nr + j + 1),
                    ha='center', va='top', fontsize=12)
        ax.set_xlabel('Wavelength (um)')
        ax.set_ylabel(f'Intensity ({r["bunit"]})')
        ax.set_xlim(xx[0], xx[-1])
        ax.tick_params(labelsize=12)
        if nchunk_val > 1:
            fig.savefig(f'{outname}-chunk{ci+1}.png', dpi=100, bbox_inches='tight')
        else:
            fig.savefig(f'{outname}.png', dpi=100, bbox_inches='tight')
        plt.close(fig)

        # Print detected peaks to terminal
        for j in range(nr):
            peak_no = sum(len(p) for p in pp_all[:-1]) + j + 1
            print(f'{peak_no:3d}   {pp[j, 0]:.4f} um  {pp[j, 1]:8.4f} {r["bunit"]}')

    if q_combined is None:
        q_combined = {'K': np.array([]), 'S': np.array([]),
                      'X0': np.array([]), 'area': np.array([]),
                      'eK': np.array([]), 'eS': np.array([]),
                      'eX0': np.array([]), 'earea': np.array([]),
                      'flag': np.array([], dtype=int)}
        ymo_combined = np.array([])
        cont_combined = np.array([])

    pp_total = np.vstack(pp_all) if pp_all else np.zeros((0, 2))
    np_total = len(pp_total)

    # Build region string for header
    if isinstance(pixco, (list, tuple)) and len(pixco) == 3 and isinstance(pixco[0], str):
        region_hdr = f'circle {pixco[0]},{pixco[1]},{float(pixco[2]):f}'
    elif region == 'pixel':
        region_hdr = f'pixel {int(pixco[0])}, {int(pixco[1])}'
    elif region == 'all':
        region_hdr = 'all'
    else:
        region_hdr = region

    # Write peaklist
    s2f = np.sqrt(np.log(256))      # sigma-to-FWHM factor
    with open(f'{outname}-peaklist.txt', 'w') as fp:
        fp.write(f'% Cube: {os.path.basename(str(fname))}\n')
        fp.write(f'% Region: {region_hdr}\n')
        fp.write('% num wave(um) peak(MJy/sr) GaussK(MJy/sr) GaussFWHM(um) '
                 'GaussX(um) Flux(W/m2/sr) eGaussK eGaussS eGaussX eFlux flag\n')
        for i in range(np_total):
            q_combined['S'][i] = abs(q_combined['S'][i])
            fp.write(
                f'{i+1:3d} {pp_total[i,0]:8.4f} {pp_total[i,1]:12.4f} '
                f'{q_combined["K"][i]:12.4f} {q_combined["S"][i]*s2f:8.4f} '
                f'{q_combined["X0"][i]:8.4f} {q_combined["area"][i]:8.4g} '
                f'{q_combined["eK"][i]:8.4g} {q_combined["eS"][i]*s2f:8.4g} '
                f'{q_combined["eX0"][i]:8.4g} {q_combined["earea"][i]:8.4g} '
                f'{q_combined["flag"][i]:2d}\n'
            )
            status = '*' if q_combined['flag'][i] > 0 else ' '
            R_val = q_combined['X0'][i] / (q_combined['S'][i] * s2f + 1e-30)
            print(f'{status}{i+1:3d} {pp_total[i,0]:8.4f} '
                  f'{pp_total[i,1]:12.4f} {q_combined["K"][i]:12.4f} '
                  f'{q_combined["S"][i]*s2f:8.4f} {R_val:8.0f} '
                  f'{q_combined["X0"][i]:8.4f} {q_combined["area"][i]:8.4g} '
                  f'{q_combined["flag"][i]:2d}')

    # Write spectrum
    with open(f'{outname}_spec.txt', 'w') as fp:
        fp.write(f'% {os.path.basename(str(fname))}\n')
        if isinstance(pixco, (list, tuple)) and len(pixco) == 3:
            fp.write(f"% {{'{pixco[0]}','{pixco[1]}',{float(pixco[2]):e}}}\n")
        for wvi, yi in zip(x, y):
            fp.write(f'{wvi:8.5f}  {yi:8.5e}\n')

    # Write continuum
    with open(f'{outname}_cont.txt', 'w') as fp:
        fp.write(f'% {os.path.basename(str(fname))}\n')
        if isinstance(pixco, (list, tuple)) and len(pixco) == 3:
            fp.write(f"% {{'{pixco[0]}','{pixco[1]}',{float(pixco[2]):e}}}\n")
        for wvi, ci in zip(x, cont_combined):
            fp.write(f'{wvi:8.5f}  {ci:8.5e}\n')

    # Write model
    with open(f'{outname}_model.txt', 'w') as fp:
        fp.write(f'% {os.path.basename(str(fname))}\n')
        if isinstance(pixco, (list, tuple)) and len(pixco) == 3:
            fp.write(f"% {{'{pixco[0]}','{pixco[1]}',{float(pixco[2]):e}}}\n")
        for wvi, mi in zip(x, ymo_combined):
            fp.write(f'{wvi:8.5f}  {mi:8.5e}\n')

    return q_combined


# ---------------------------------------------------------------------------
# Fit lines at specified wavelengths
# ---------------------------------------------------------------------------

def do_lines(fname, pixco, lwave):
    """
    Fit spectral lines at (or near) the wavelengths lwave.

    Parameters
    ----------
    fname  : str or dict
    pixco  : extraction region (see extract_aperture)
    lwave  : list/array of expected line centre wavelengths (microns)

    Returns
    -------
    q, wv, spec, model, smo
    """
    swin = 5
    lwave = np.asarray(lwave)

    if isinstance(fname, str):
        r = read_fits(fname, 'xten1')
    else:
        r = fname

    s, region, mask, flag = extract_aperture(r, pixco)
    if flag != 0:
        return {'flag': flag}, None, None, None, None

    wv_all = r['x'][2]
    valid = ~np.isnan(s)
    wv = wv_all[valid]
    spec = s[valid]

    pp = np.zeros((len(lwave), 2))
    for i, lw in enumerate(lwave):
        idx = int(np.argmin(np.abs(wv - lw)))
        st = max(idx - swin, 0)
        en = min(idx + swin, len(wv) - 1)
        idx_max = st + int(np.argmax(spec[st:en+1]))
        pp[i, 0] = wv[idx_max]
        pp[i, 1] = spec[idx_max]

    q, model, smo = fit_lines(wv, spec, pp)
    return q, wv, spec, model, smo


# ---------------------------------------------------------------------------
# Parabolic centroid helper  (new in 06022026, used by make_int_map)
# ---------------------------------------------------------------------------

def _parcenter(xp, yp):
    """
    Parabolic centroid of 3 points.

    Fits a quadratic to (xp, yp).  Returns (x_peak, y_peak) at the vertex.
    If the parabola opens upward (minimum, not maximum), falls back to
    the highest of the 3 input points.
    """
    p = np.polyfit(xp, yp, 2)
    if p[0] > 0:  # parabola opens up – pick the highest data point
        ix = int(np.argmax(yp))
        return float(xp[ix]), float(yp[ix])
    x_cen = -p[1] / (2.0 * p[0])
    y_cen = p[2] - p[1]**2 / (4.0 * p[0])
    return float(x_cen), float(y_cen)


# ---------------------------------------------------------------------------
# Multi-plane FITS cube writer  (new in 06032026, used by make_int_map)
# ---------------------------------------------------------------------------

def _write_int_map_cube(r, mapname, planes, baseline, order, intrange, reject):
    """
    Write the 12-plane output FITS cube from make_int_map.

    Planes (each nx×ny) in order:
      1  integrated intensity     W/m2/sr
      2  error in int. intensity  W/m2/sr
      3  peak intensity           MJy/sr (baseline-subtracted)
      4  baseline rms             MJy/sr
      5  wavelength of peak       µm
      6  sigma-clip outlier flag  (0/1)
      7  Gaussian fit area        W/m2/sr
      8  Gaussian fit area error  W/m2/sr
      9  Gaussian fit center      µm
     10  Gaussian fit center err  µm
     11  Gaussian fit sigma       µm
     12  Gaussian fit sigma err   µm

    The spatial WCS is copied from the input cube dict r.  All 12 planes are
    written as float32.  This matches the Matlab mkintmap output (NAXIS3=12).
    """
    nx, ny = planes[0].shape
    nplanes = len(planes)   # = 12

    # Stack (nx, ny, nplanes) → transpose to (nplanes, ny, nx) for FITS storage
    cube = np.stack(planes, axis=2).astype(np.float32)
    out_data = np.transpose(cube)   # (nplanes, ny, nx)

    hdr = fits.Header()

    # Spatial axis 1 (RA)
    hdr['CTYPE1'] = r['ctype'][0] if r.get('ctype') else ''
    hdr['CRPIX1'] = float(r['crpix'][0])
    hdr['CRVAL1'] = float(r['crval'][0])
    hdr['CDELT1'] = float(r['cdelt'][0])
    hdr['CROTA1'] = float(r['crota'][0])
    if r.get('cunit') and r['cunit'][0]:
        hdr['CUNIT1'] = r['cunit'][0]

    # Spatial axis 2 (Dec)
    hdr['CTYPE2'] = r['ctype'][1] if len(r.get('ctype', [])) > 1 else ''
    hdr['CRPIX2'] = float(r['crpix'][1])
    hdr['CRVAL2'] = float(r['crval'][1])
    hdr['CDELT2'] = float(r['cdelt'][1])
    hdr['CROTA2'] = float(r['crota'][1])
    if r.get('cunit') and len(r['cunit']) > 1 and r['cunit'][1]:
        hdr['CUNIT2'] = r['cunit'][1]

    # CD/PC rotation matrix (copy from input cube if present)
    for key in ('cd1_1', 'cd1_2', 'cd2_1', 'cd2_2',
                'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2'):
        if key in r:
            hdr[key.upper()] = r[key]

    # Axis 3: plane index
    hdr['CTYPE3'] = 'PLANE'
    hdr['CRPIX3'] = 1.0
    hdr['CRVAL3'] = 1.0
    hdr['CDELT3'] = 1.0
    hdr['CROTA3'] = 0.0
    hdr['CUNIT3'] = 'PLANE'

    hdr['BUNIT'] = 'see plane'

    # Copy selected metadata keywords from original cube header
    _skip = {
        '', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'NAXIS3',
        'EXTEND', 'XTENSION', 'PCOUNT', 'GCOUNT',
        'CTYPE1', 'CRPIX1', 'CRVAL1', 'CDELT1', 'CROTA1', 'CUNIT1',
        'CTYPE2', 'CRPIX2', 'CRVAL2', 'CDELT2', 'CROTA2', 'CUNIT2',
        'CTYPE3', 'CRPIX3', 'CRVAL3', 'CDELT3', 'CROTA3', 'CUNIT3',
        'BUNIT', 'COMMENT', 'HISTORY',
        'WCSAXES', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
        'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
        'PC1_3', 'PC2_3', 'PC3_1', 'PC3_2', 'PC3_3',
        'BASELINE', 'BASEORD', 'INTRNG', 'CLIPLVL', 'COORMODE',
    }
    orig_hdr = r.get('header', {})
    for key, val in orig_hdr.items():
        if key in _skip or val is None:
            continue
        try:
            hdr[key] = val
        except Exception:
            pass

    # Analysis parameters
    hdr['BASELINE'] = ' '.join(f'{v:.4f}' for v in baseline)
    hdr['BASEORD'] = order
    hdr['INTRNG'] = ' '.join(f'{v:.4f}' for v in intrange)
    hdr['CLIPLVL'] = reject

    # COMMENT lines describing each plane (13 lines, matching Matlab)
    for cmnt in [
        'PLANE 1 integrated intensity W/m2/sr',
        'PLANE 2 error in integrated intensity W/m2/sr',
        'PLANE 3 peak intensity W/m2/sr',
        'PLANE 4 baseline rms W/m2/sr',
        'PLANE 5 wave of peak intensity um',
        'PLANE 6 pixels with clipped spectral data in sigma-clip',
        'PLANE 7 Gaussian fit area W/m2/sr',
        'PLANE 8 Gaussian fit area error W/m2/sr',
        'PLANE 9 Gaussian fit center um',
        'PLANE 10 Gaussian fit center error um',
        'PLANE 11 Gaussian fit sigma um',
        'PLANE 12 Gaussian fit sigma error um',
        'PLANE 13 Gaussian fit flag from fitLines',
    ]:
        hdr['COMMENT'] = cmnt

    hdu = fits.PrimaryHDU(data=out_data, header=hdr)
    hdu.writeto(f'{mapname}.fits', overwrite=True)


# ---------------------------------------------------------------------------
# Integrated intensity map  (updated in 06032026)
# ---------------------------------------------------------------------------

def make_int_map(fitsfile, mapname, baseline, order, intrange, flag=''):
    """
    Compute an integrated intensity map and its uncertainty.

    Parameters
    ----------
    fitsfile  : str – input FITS cube filename
    mapname   : str – base name for output FITS file
    baseline  : list of wavelength pairs [l1, l2, l3, l4, ...] defining
                baseline regions (microns)
    order     : polynomial order for baseline fit
    intrange  : [l_start, l_end] integration range (microns)
    flag      : str – optional behavior flags (default ''):
                'debug'       – print column progress and flag suspicious spectra
                'rejectX.X'  – sigma-clip baseline outliers at X.X sigma
                               (default reject=5 when 'reject' not in flag)

    Writes {mapname}.fits — a 12-plane FITS cube containing all maps.
    Plane order: int, eint, peak, rms, wpeak, outliers, gm0, gem0, gm1, gem1,
                 gm2, gem2 (see COMMENT keywords in output FITS header).

    Returns
    -------
    maps : dict with fields:
        int       – integrated intensity map           (W/m²/sr)
        eint      – uncertainty map                    (W/m²/sr)
        peak      – per-pixel peak amplitude           (MJy/sr, baseline-subtracted)
        rms       – per-pixel baseline RMS             (MJy/sr)
        wpeak     – per-pixel peak wavelength          (µm)
        outliers  – per-pixel outlier flag (1 if any baseline sigma-clipping occurred)
        gm0       – per-pixel Gaussian integrated flux (W/m²/sr)
        gem0      – uncertainty on gm0
        gm1       – per-pixel Gaussian centroid        (µm)
        gem1      – uncertainty on gm1
        gm2       – per-pixel Gaussian sigma           (µm)
        gem2      – uncertainty on gm2
        err       – per-pixel Gaussian fit error flag
    """
    # Parse flag string
    debug = 'debug' in flag
    reject = 5.0
    if 'reject' in flag:
        ix_f = flag.find('reject')
        remainder = flag[ix_f + 6:]
        num_str = remainder.split()[0] if remainder.split() else ''
        try:
            reject = float(num_str)
        except ValueError:
            pass

    r = read_fits(fitsfile, 'xten1')
    wv = r['x'][2]
    c = 3e14  # µm/s

    # Baseline channel indices – reset per pixel to allow sigma-clipping
    ibs_orig = []
    for k in range(len(baseline) // 2):
        ibs_orig.extend(
            np.where((wv >= baseline[2*k]) & (wv <= baseline[2*k + 1]))[0])
    ibs_orig = np.array(ibs_orig)

    # Integration channel indices
    iint = np.where((wv >= intrange[0]) & (wv <= intrange[1]))[0]
    dnu = c * np.mean(np.diff(wv[iint])) / np.mean(wv[iint])**2  # Hz/channel

    nx, ny, nz = r['data'].shape
    intensity_map = np.zeros((nx, ny))
    error_map     = np.zeros((nx, ny))
    pmap          = np.zeros((nx, ny))   # outlier flag map
    peak_map      = np.zeros((nx, ny))
    rmsmap        = np.zeros((nx, ny))   # per-pixel baseline RMS (new in 06032026)
    wpeak_map     = np.zeros((nx, ny))
    m0map         = np.zeros((nx, ny))
    em0map        = np.zeros((nx, ny))
    m1map         = np.zeros((nx, ny))
    em1map        = np.zeros((nx, ny))
    m2map         = np.zeros((nx, ny))
    em2map        = np.zeros((nx, ny))
    errmap        = np.zeros((nx, ny))
    rmsa          = np.full(ny, np.nan)  # per-row rms, used for debug reporting

    for i in range(nx):
        if debug:
            print(f'Column {i + 1} of {nx}')

        for j in range(ny):
            ibs = ibs_orig.copy()
            sp = r['data'][i, j, :]

            if np.any(np.isnan(sp)):
                rmsa[j] = np.nan
                continue

            # Iterative sigma-clipping: remove one outlier at a time from
            # the baseline until all residuals are within reject*rms
            while True:
                basew = wv[ibs]
                pcoef = np.polyfit(basew, sp[ibs], order)
                spb = sp - np.polyval(pcoef, wv)
                rms = np.std(spb[ibs])
                if len(ibs) < order + 1:
                    break
                if np.max(np.abs(spb[ibs])) > reject * rms:
                    ix_out = int(np.argmax(np.abs(spb[ibs])))
                    ibs = np.setdiff1d(ibs, [ibs[ix_out]])
                    pmap[i, j] = 1
                else:
                    break

            rmsmap[i, j] = rms   # store per-pixel baseline RMS

            # Moment-0 map (sum over integration range × dnu → W/m²/sr)
            intensity_map[i, j] = np.sum(spb[iint]) * dnu * 1e6 * 1e-26
            error_map[i, j]     = rms * np.sqrt(len(iint)) * dnu * 1e6 * 1e-26

            # Parabolic peak centroid in the combined baseline + line window
            v_idx = int(np.argmax(spb[iint]))
            v_amp = float(spb[iint[v_idx]])
            uu = np.unique(np.concatenate([ibs, iint]))
            ixu_arr = np.where(uu == iint[v_idx])[0]
            if len(ixu_arr) > 0:
                ixu = int(ixu_arr[0])
                if 0 < ixu < len(uu) - 1:
                    wp, pk = _parcenter(wv[uu[ixu - 1:ixu + 2]],
                                        spb[uu[ixu - 1:ixu + 2]])
                else:
                    wp, pk = float(wv[iint[v_idx]]), v_amp
            else:
                wp, pk = float(wv[iint[v_idx]]), v_amp
            peak_map[i, j]  = pk
            wpeak_map[i, j] = wp

            # Single-Gaussian fit over the combined window
            try:
                q_pix, _, _ = fit_lines(
                    wv[uu], spb[uu],
                    np.array([[wv[iint[v_idx]], v_amp]])
                )
                m0map[i, j]  = q_pix['area'][0]
                em0map[i, j] = q_pix['earea'][0]
                m1map[i, j]  = q_pix['X0'][0]
                em1map[i, j] = q_pix['eX0'][0]
                m2map[i, j]  = q_pix['S'][0]
                em2map[i, j] = q_pix['eS'][0]
                errmap[i, j] = float(q_pix['flag'][0])
            except Exception:
                errmap[i, j] = 1.0

            if debug:
                rmsa[j] = rms
                spb_uu = spb[uu]
                if np.abs(np.min(spb_uu)) > 6 * rms:
                    med_rms = float(np.nanmedian(rmsa[:j])) if j > 0 else np.nan
                    print(f'  Row {j + 1} of {ny}: '
                          f'Min={np.min(spb_uu):.3g}, rms={rms:.3g}, '
                          f'typ.rms={med_rms:.3g}')

        if debug:
            print(f'  Typ. RMS={np.nanmedian(rmsa):.3g}')

    # Write 12-plane FITS cube (matches Matlab mkintmap output, NAXIS3=12)
    _write_int_map_cube(
        r, mapname,
        planes=[intensity_map, error_map, peak_map, rmsmap, wpeak_map, pmap,
                m0map, em0map, m1map, em1map, m2map, em2map],
        baseline=baseline, order=order, intrange=intrange, reject=reject,
    )

    return {
        'int':      intensity_map,
        'eint':     error_map,
        'peak':     peak_map,
        'rms':      rmsmap,
        'wpeak':    wpeak_map,
        'outliers': pmap,
        'gm0':      m0map,
        'gem0':     em0map,
        'gm1':      m1map,
        'gem1':     em1map,
        'gm2':      m2map,
        'gem2':     em2map,
        'err':      errmap,
    }


# ---------------------------------------------------------------------------
# Peaklist editing
# ---------------------------------------------------------------------------

def _read_peaklist(fname):
    """Read a peaklist text file; return header lines and data array."""
    headers = []
    rows = []
    with open(fname, 'r') as fp:
        for ln in fp:
            if ln.startswith('%'):
                headers.append(ln.rstrip('\n'))
            else:
                vals = ln.split()
                if not vals:
                    continue
                rows.append([float(v) for v in vals])
    return headers, np.array(rows) if rows else np.zeros((0, 3))


def _write_peaklist(fname, headers, data):
    with open(fname, 'w') as fp:
        for h in headers:
            fp.write(h + '\n')
        if data.shape[1] >= 12:
            fmt = ('{:3d} {:8.4f} {:12.4f} {:12.4f} {:8.4f} {:8.4f} '
                   '{:8.4g} {:8.4g} {:8.4g} {:8.4g} {:8.4g} {:2d}\n')
            for row in data:
                fp.write(fmt.format(
                    int(row[0]), row[1], row[2], row[3], row[4], row[5],
                    row[6], row[7], row[8], row[9], row[10], int(row[11])
                ))
        else:
            for row in data:
                fp.write(f'{int(row[0]):3d} {row[1]:8.4f} {row[2]:12.4f}\n')


def recompute_lines(file):
    """
    Re-fit all lines in a peaklist file using the saved spectrum.

    Reads {file}-peaklist(-ed).txt and {file}_spec.txt, re-runs fit_lines,
    and overwrites the peaklist, _cont.txt, and _model.txt files.
    """
    ed_fname = f'{file}-peaklist-ed.txt'
    fname = ed_fname if os.path.exists(ed_fname) else f'{file}-peaklist.txt'

    headers, data = _read_peaklist(fname)

    spec_data = np.loadtxt(f'{file}_spec.txt', comments='%')
    wv = spec_data[:, 0]
    spec = spec_data[:, 1]

    if data.size == 0:
        return

    peaks = data[:, 1:3] if data.shape[1] >= 3 else data[:, 1:3]
    q, ymo, cont = fit_lines(wv, spec, peaks)

    s2f = np.sqrt(np.log(256))
    n = len(q['K'])
    new_data = np.zeros((n, 12))
    new_data[:, 0] = data[:n, 0]
    new_data[:, 1] = data[:n, 1]
    new_data[:, 2] = data[:n, 2]
    new_data[:, 3] = q['K']
    new_data[:, 4] = q['S'] * s2f
    new_data[:, 5] = q['X0']
    new_data[:, 6] = q['area']
    new_data[:, 7] = q['eK']
    new_data[:, 8] = q['eS'] * s2f
    new_data[:, 9] = q['eX0']
    new_data[:, 10] = q['earea']
    new_data[:, 11] = q['flag']

    _write_peaklist(fname, headers, new_data)

    # Update _cont.txt
    _update_derived_file(f'{file}_cont.txt', wv, cont)
    # Update _model.txt
    _update_derived_file(f'{file}_model.txt', wv, ymo)


def _update_derived_file(fname, wv, values):
    """Re-write a _cont.txt or _model.txt file, preserving comment header."""
    hdrs = []
    if os.path.exists(fname):
        with open(fname, 'r') as fp:
            for ln in fp:
                if ln.startswith('%'):
                    hdrs.append(ln.rstrip('\n'))
                else:
                    break
    with open(fname, 'w') as fp:
        for h in hdrs:
            fp.write(h + '\n')
        for wvi, vi in zip(wv, values):
            fp.write(f'{wvi:8.5f}  {vi:8.5e}\n')


def cull_lines(file, lst):
    """
    Remove lines in lst (1-based indices) from the peaklist and refit.

    Parameters
    ----------
    file : str – base name (without extension)
    lst  : list/array of 1-based line numbers to remove
    """
    fname = f'{file}-peaklist.txt'
    headers, data = _read_peaklist(fname)
    if data.size == 0:
        return

    keep_mask = ~np.isin(data[:, 0].astype(int), lst)
    kept = data[keep_mask].copy()
    kept[:, 0] = np.arange(1, len(kept) + 1, dtype=float)  # renumber

    ed_fname = f'{file}-peaklist-ed.txt'
    short_data = kept[:, :3] if data.shape[1] < 12 else kept[:, :3]
    _write_peaklist(ed_fname, headers, short_data)

    recompute_lines(file)


def keep_lines(file, lst, tol=0.05):
    """
    Keep only lines within tol um of the wavelengths in lst; refit.

    Parameters
    ----------
    file : str – base name
    lst  : array of target wavelengths (microns)
    tol  : matching tolerance in microns (scalar or array, default 0.05)
    """
    lst = np.asarray(lst)
    if np.isscalar(tol):
        tol = tol * np.ones(len(lst))
    tol = np.asarray(tol)

    fname = f'{file}-peaklist.txt'
    headers, data = _read_peaklist(fname)
    if data.size == 0:
        return

    keep_idx = []
    for i, (lw, toli) in enumerate(zip(lst, tol)):
        hits = np.where((data[:, 1] >= lw - toli) & (data[:, 1] <= lw + toli))[0]
        keep_idx.extend(hits)
    keep_idx = sorted(set(keep_idx))

    kept = data[keep_idx].copy() if keep_idx else np.zeros((0, data.shape[1]))
    if len(kept) > 0:
        kept[:, 0] = np.arange(1, len(kept) + 1, dtype=float)

    ed_fname = f'{file}-peaklist-ed.txt'
    _write_peaklist(ed_fname, headers, kept)
    print(kept)

    recompute_lines(file)


# ---------------------------------------------------------------------------
# Line identification
# ---------------------------------------------------------------------------

def _parse_csv_catalog(fname):
    """Read a simple 2-column LineName, RestValue CSV catalog."""
    df = pd.read_csv(fname)
    cat = {}
    # First column = line name, second = rest wavelength
    cols = df.columns.tolist()
    cat['LineName'] = df[cols[0]].astype(str).tolist()
    cat['RestValue'] = df[cols[1]].astype(float).values
    return cat


def _parse_pdr4all(fname):
    """Read fixed-width PDR4all line list (.lst format)."""
    colspecs = [(0, 6), (6, 14), (14, 24), (24, 38), (38, 51), (51, 92),
                (92, 102), (102, 114), (114, 134)]
    names = ['Model', 'Species', 'M-lambda', 'RestValue', 'LineName',
             'transition', 'A', 'Eupper', 'Comments']
    df = pd.read_fwf(fname, colspecs=colspecs, names=names, skiprows=1)
    cat = {}
    cat['LineName'] = df['LineName'].astype(str).str.strip().tolist()
    cat['RestValue'] = pd.to_numeric(df['RestValue'], errors='coerce').values
    cat['transition'] = df['transition'].astype(str).str.strip().tolist()
    ok = np.isfinite(cat['RestValue'])
    cat['LineName'] = [cat['LineName'][i] for i in range(len(ok)) if ok[i]]
    cat['transition'] = [cat['transition'][i] for i in range(len(ok)) if ok[i]]
    cat['RestValue'] = cat['RestValue'][ok]
    return cat


def _parse_iso(fname):
    """Read ISO line list (space-separated, 4 header lines)."""
    rows = []
    names = []
    transitions = []
    with open(fname, 'r') as fp:
        for _ in range(4):
            fp.readline()
        for ln in fp:
            ln = ln.strip()
            if not ln or ln.startswith('#') or ln.startswith('-'):
                continue
            parts = ln.split()
            if len(parts) >= 3:
                names.append(parts[0])
                transitions.append(parts[1] if len(parts) > 1 else '')
                try:
                    rows.append(float(parts[2]))
                except ValueError:
                    rows.append(np.nan)
    cat = {'LineName': names, 'transition': transitions,
           'RestValue': np.array(rows)}
    return cat


def _parse_nist(fname):
    """Read pipe-delimited NIST ASD lines file (skipping 5 header rows)."""
    names = []
    rest_vals = []
    transitions = []
    with open(fname, 'r') as fp:
        for _ in range(5):
            fp.readline()
        for ln in fp:
            if not ln.strip() or ln.startswith('-'):
                continue
            parts = [p.strip() for p in ln.split('|')]
            if len(parts) < 11:
                continue
            try:
                wv = float(parts[1])
            except ValueError:
                continue
            try:
                aki = float(parts[2])
            except ValueError:
                aki = 1.0
            species = parts[0].replace(' ', '')
            if aki < 1.0:
                species = f'[{species}]'
            lower_term = parts[6].strip() if len(parts) > 6 else ''
            lower_j = parts[7].strip() if len(parts) > 7 else ''
            upper_term = parts[9].strip() if len(parts) > 9 else ''
            upper_j = parts[10].strip() if len(parts) > 10 else ''
            trans = f'{upper_term}{upper_j} - {lower_term}{lower_j}'
            names.append(species)
            rest_vals.append(wv)
            transitions.append(trans)
    cat = {'LineName': names, 'transition': transitions,
           'RestValue': np.array(rest_vals)}
    return cat


def line_id(linefile, vlsr, R=250, outname=None, catdir=None):
    """
    Identify spectral lines by matching detected peaks to line catalogs.

    Parameters
    ----------
    linefile : str or array
        Base name of peaklist file (reads {linefile}-peaklist.txt) or
        array of wavelengths to match.
    vlsr     : float
        Radial velocity in km/s (if > 20) or redshift z (if <= 20).
    R        : float
        Spectral resolving power λ/Δλ used as matching window (default 250).
    outname  : str, optional
        Output filename (default: {linefile}-lineID.txt).
    catdir   : str, optional
        Directory containing catalog files (default: matlab_slice_06012026/).
    """
    clight = 3e5   # km/s

    if catdir is None:
        catdir = _CATDIR

    if isinstance(linefile, str):
        ed_fname = f'{linefile}-peaklist-ed.txt'
        pk_fname = (ed_fname if os.path.exists(ed_fname)
                    else f'{linefile}-peaklist.txt')
        if outname is None:
            outname = f'{linefile}-lineID.txt'
        headers, data = _read_peaklist(pk_fname)
        if data.size == 0:
            print('Empty peaklist')
            return
        wv = data[:, 1]
        lineno = data[:, 0].astype(int)
    else:
        wv = np.asarray(linefile, dtype=float)
        lineno = np.arange(1, len(wv) + 1)
        if outname is None:
            outname = 'default-lineID.txt'

    # Rest-frame wavelengths
    if vlsr > 20:
        wvr = wv / (1 + vlsr / clight)
        print(f'Matching lines assuming vlsr={vlsr:.0f} km/s')
    else:
        wvr = wv / (1 + vlsr)
        print(f'Matching lines assuming redshift z={vlsr:.5f}')

    # Load catalogs
    catnames = [
        ('Atomic-Ionic_FineStructure.csv', 'csv'),
        ('Atomic-Ionic.csv', 'csv'),
        ('H-He.csv', 'csv'),
        ('H2.csv', 'csv'),
        ('HeI-HeII.csv', 'csv'),
        ('CO.csv', 'csv'),
        ('new-list-20231010.lst', 'lst'),
        ('ISOLineList.txt', 'iso'),
        ('NIST-ASDLines.html', 'nist'),
    ]
    cats = []
    cat_fnames = []
    for cname, ctype in catnames:
        cpath = os.path.join(catdir, cname)
        if not os.path.exists(cpath):
            continue
        try:
            if ctype == 'csv':
                cats.append(_parse_csv_catalog(cpath))
            elif ctype == 'lst':
                cats.append(_parse_pdr4all(cpath))
            elif ctype == 'iso':
                cats.append(_parse_iso(cpath))
            elif ctype == 'nist':
                cats.append(_parse_nist(cpath))
            cat_fnames.append(cname)
        except Exception as e:
            print(f'Warning: could not load {cname}: {e}')

    nm = 0
    with open(outname, 'w') as fp:
        if vlsr > 20:
            fp.write(f'Matching lines in {linefile} assuming vlsr={vlsr:.0f} km/s\n\n')
        else:
            fp.write(f'Matching lines in {linefile} assuming redshift z={vlsr:.5f}\n\n')

        for i, (wvi, wvri) in enumerate(zip(wv, wvr)):
            dw = wvi / R
            line_str = (f'#{lineno[i]:3d} {wvi:8.4f}+/-{dw:.4f} um '
                        f'(rest={wvri:8.4f} um):\n       ')
            print(line_str, end='')
            fp.write(line_str)

            matches = []
            for ci, cat in enumerate(cats):
                rv = np.asarray(cat['RestValue'])
                dist = np.abs(wvri - rv)
                hit_idx = np.where(dist <= dw)[0]
                for k in hit_idx:
                    if 'transition' in cat:
                        name = (cat['LineName'][k].replace(' ', '')
                                + ' ' + cat['transition'][k])
                    else:
                        name = cat['LineName'][k]
                    matches.append({
                        'name': name,
                        'species': cat['LineName'][k].replace(' ', ''),
                        'rv': float(rv[k]),
                        'dd': float(wvri - rv[k]),
                        'cat': cat_fnames[ci],
                    })

            if matches:
                # Prefer non-H2 vibrational transitions at top
                matches_sorted = sorted(matches, key=lambda m: abs(m['dd']))
                h2vib = [m for m in matches_sorted if 'H2 v=' in m['name']]
                other = [m for m in matches_sorted if 'H2 v=' not in m['name']]
                ordered = other + h2vib

                for j, m in enumerate(ordered):
                    vel = m['dd'] / m['rv'] * clight
                    if j == 0:
                        fp.write(f"| {m['name']} {vel:+.0f} | ")
                        print(f"| {m['name']} {vel:+.0f} | ", end='')
                    else:
                        fp.write(f"{m['name']} {vel:+.0f} | ")
                        print(f"{m['name']} {vel:+.0f} | ", end='')

                best = ordered[0]
                fp.write(f"\n       BEST: DLambda={best['dd']:.4f} um, "
                         f"Catalog={best['cat']}, Restwv={best['rv']:.4f}\n"
                         f"       {best['name']}\n")
                print(f"\n       BEST: DLambda={best['dd']:.4f} um, "
                      f"Catalog={best['cat']}, Restwv={best['rv']:.4f}\n"
                      f"       {best['name']}")

                if len(ordered) > 1:
                    k2 = 1
                    while (k2 < len(ordered)
                           and abs(ordered[0]['dd'] - ordered[k2]['dd']) < 1e-4):
                        k2 += 1
                    if k2 < len(ordered):
                        nxt = ordered[k2]
                        fp.write(f"       NEXT: DLambda={nxt['dd']:.4f} um, "
                                 f"Catalog={nxt['cat']}, Restwv={nxt['rv']:.4f}\n"
                                 f"       {nxt['name']}\n")
                        print(f"       NEXT: DLambda={nxt['dd']:.4f} um, "
                              f"Catalog={nxt['cat']}, Restwv={nxt['rv']:.4f}\n"
                              f"       {nxt['name']}")
            else:
                fp.write(f'       NOMATCH within +/-{dw:.4f} um\n')
                print(f'       NOMATCH within +/-{dw:.4f} um')
                nm += 1

    print(f'Attempted IDs for {len(wv)} lines, found no matches for {nm} lines')


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_peaks(file, idfile=None, yrange=None, color=None):
    """
    Plot spectrum with detected peaks and optional line IDs.

    Reads {file}_spec.txt, {file}_model.txt, {file}_cont.txt, and
    {file}-peaklist.txt (or -ed version).

    Parameters
    ----------
    file    : str – base name
    idfile  : str or False, optional – lineID file; auto-detected if None
    yrange  : [ymin, ymax] for vertical peak markers
    color   : matplotlib colour string for the spectrum

    Returns
    -------
    hh : matplotlib Line2D handle
    spx, spy : spectrum wavelength and intensity arrays
    """
    if idfile is None and os.path.exists(f'{file}-lineID.txt'):
        idfile = f'{file}-lineID.txt'

    spec_data = np.loadtxt(f'{file}_spec.txt', comments='%')
    spx, spy = spec_data[:, 0], spec_data[:, 1]

    fig, ax = plt.subplots(figsize=(12, 5))
    if color:
        hh, = ax.semilogy(spx, spy, linewidth=2, color=color)
    else:
        hh, = ax.semilogy(spx, spy, linewidth=2)

    # Overlay total model if available
    model_fname = f'{file}_model.txt'
    cont_fname = f'{file}_cont.txt'
    if os.path.exists(model_fname) and os.path.exists(cont_fname):
        ymo1 = np.loadtxt(model_fname, comments='%')
        ymo2 = np.loadtxt(cont_fname, comments='%')
        total = ymo1[:, 1] + ymo2[:, 1]
        total[spy == 0] = 0
        ax.plot(ymo1[:, 0], total, '-k')

    # Load peaklist
    ed_fname = f'{file}-peaklist-ed.txt'
    pk_fname = ed_fname if os.path.exists(ed_fname) else f'{file}-peaklist.txt'
    headers, pk_data = _read_peaklist(pk_fname)

    # Load line IDs if available
    ids = []
    if isinstance(idfile, str) and os.path.exists(idfile):
        with open(idfile, 'r') as fp:
            for ln in fp:
                if 'BEST:' in ln:
                    ids.append(fp.readline().strip())
                elif 'NOMATCH' in ln:
                    ids.append('?')

    do_range = yrange is not None

    for i, row in enumerate(pk_data):
        line_no = int(row[0])
        wvi = row[1]
        flux = row[6] if pk_data.shape[1] >= 7 else row[2]

        if ids and i < len(ids):
            print(f'{line_no:2d}: {wvi:.6f} um, {flux:.4e} W/m2/sr ---- {ids[i]}')
        else:
            print(f'{line_no:2d}: {wvi:.6f} um, {flux:.4e} W/m2/sr')

        # Vertical marker
        if do_range:
            ax.axvline(wvi, ymin=0, ymax=1, color='r', linestyle='-.')
        else:
            ylo = 10**int(np.log10(max(np.min(spy[spy > 0]), 1e-30)))
            ax.plot([wvi, wvi], [ylo, np.max(spy)], 'r-.')

        idx_near = int(np.argmin(np.abs(spx - wvi)))
        ty_idx = min(idx_near + 8, len(spy) - 1)
        ty = spy[ty_idx] * 0.99
        ax.text(wvi, ty, str(line_no), ha='center', va='top', fontsize=14)

        if ids and i < len(ids) and ids[i] != '?':
            species = ids[i].split()[0]
            ax.text(wvi, ty * 0.7, species, ha='center', va='top', fontsize=14)

    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Specific Intensity (MJy/sr)')
    ax.tick_params(labelsize=14)
    ax.autoscale(axis='x', tight=True)

    return hh, spx, spy


def plot_all_ch(name):
    """
    Plot all four MIRI MRS channels together on one log-scale axis.

    Looks for {name}_ch{1-4}_spec.txt and {name}_ch{1-4}-lineID.txt.
    """
    colors = ['b', 'm', 'g', 'c']
    fig, ax = plt.subplots(figsize=(16, 6))

    maxv, minv = -1e20, 1e20
    maxx, minx = -1000.0, 1000.0
    handles = []
    have_ch = []

    for i in range(1, 5):
        spec_f = f'{name}_ch{i}_spec.txt'
        if not os.path.exists(spec_f):
            have_ch.append(False)
            handles.append(None)
            continue
        have_ch.append(True)
        u = np.loadtxt(spec_f, comments='%')
        maxx = max(maxx, np.max(u[:, 0]))
        maxv = max(maxv, np.max(u[:, 1]))
        minx = min(minx, np.min(u[:, 0]))
        minv = min(minv, np.min(u[:, 1]))
        handles.append(u)

    if minv < 0:
        minv = 0.1
    minv = 10 ** np.floor(np.log10(minv))
    maxv = 10 ** np.ceil(np.log10(maxv))

    hh_list = []
    for i in range(1, 5):
        if not have_ch[i-1]:
            hh_list.append(None)
            continue
        id_f = f'{name}_ch{i}-lineID.txt'
        h, _, _ = plot_peaks(f'{name}_ch{i}',
                              idfile=id_f if os.path.exists(id_f) else False,
                              yrange=[minv, maxv], color=colors[i-1])
        hh_list.append(h)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(minv, maxv)
    ax.set_xlabel('Wavelength (um)')
    ax.set_ylabel('Spectral density (MJy/sr)')
    ax.tick_params(labelsize=14)
    ax.set_xscale('log')
    valid_handles = [(h, f'ch {i+1}') for i, h in enumerate(hh_list) if h is not None]
    if valid_handles:
        ax.legend([v[0] for v in valid_handles],
                  [v[1] for v in valid_handles], loc='lower right')
    ax.set_title(name)
    return fig, ax


def read_model(file):
    """Read a model spectrum file (e.g. CLOUDY/PDR model output).

    Returns a list of arrays matching the columns.
    """
    cols = [[], [], [], [], [], [], [], [], []]
    with open(file, 'r') as fp:
        for _ in range(28):
            fp.readline()
        for ln in fp:
            parts = ln.split()
            if len(parts) < 2:
                continue
            for j in range(min(len(parts), 9)):
                try:
                    cols[j].append(float(parts[j]))
                except ValueError:
                    cols[j].append(np.nan)
    return [np.array(c) for c in cols if c]

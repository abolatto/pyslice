# Created: 2026-10-22 00:00:00
# Last Modified: 2026-10-22 00:00:00
# Description: Python equivalent of mkNeVmap.m (matlab_slice_10222026).
#   [NeV] 14.32 um peak map of M82 with the seven sources, X-ray image with the
#   MRS footprints, differential (aperture minus annulus) spectra of sources
#   1-4 across the four MIRI channels, Gaussian fits to broadened lines in
#   source 3, and the ULX X-1 ch4 spectrum.  Each Matlab %% cell is a function;
#   run all or a subset from the command line.
# Last Edit: Initial port of mkNeVmap.m

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus, Circle
from matplotlib.ticker import NullFormatter

from slice_lib import (read_fits, sex2dec, make_map, diff_spec, cat_peaks,
                       gaussfit)

# ---------------------------------------------------------------------------
# Configuration – adjust paths as needed
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'JWST')
# Matlab wrote into JWST/ itself; a separate directory keeps the Matlab products
# (e.g. the NeV14.3-M82v2.fits reference map) from being overwritten
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mk_nev_map_output')

CUBE = os.path.join(DATA_DIR, 'M82_sky_v2_ch?-shortmediumlong_s3d.fits')
XRAY = os.path.join(DATA_DIR, 'broad_norm.fits')
FENECH = os.path.join(DATA_DIR, 'Fenech.reg')

PKO = [('09:55:52.74', '69:40:45.89'), ('09:55:51.276', '69:40:44.09'),
       ('09:55:54.461', '69:40:51.448'), ('09:55:54.55', '69:40:49.1'),
       ('09:55:51.54', '69:40:45.87'), ('09:55:52.95', '69:40:47.51'),
       ('09:55:53.91', '69:40:48.48')]
ARROW0 = np.array([[-2, -2], [-1.5, -3.5], [3, -3], [1, -1.5], [-2, 4], [2, 5], [1, -2]])

# Fixing the order so it's in increasing RA order
_ix = np.argsort(sex2dec([p[0] for p in PKO]), kind='stable')
PKA = [PKO[i] for i in _ix]
ARROW = ARROW0[_ix]

# Reference position: Lester et al 1990 2um peak (J2000)
REFRA = sex2dec('09:55:52.3') * 15
REFDE = sex2dec('69:40:46.1')
DIST = 3.6e6   # pc

# Radii (arcsec) of central aperture, inner and outer annulus, per channel
R = np.array([[0.2, 0.2, 0.3, 0.4],
              [0.6, 0.6, 0.7, 0.85],
              [0.4, 0.4, 0.5, 0.6]])

LINELIST = np.array([5.057, 5.130, 5.342, 5.513, 5.611, 5.910, 6.109, 6.113, 6.294,
                     6.638, 6.912, 6.988, 7.355, 7.462, 7.505, 7.5104, 7.6527, 8.0258,
                     8.9921, 9.6659, 10.5110, 10.6833, 9.3948, 11.006, 8.763, 7.7836,
                     12.288, 12.382, 12.824, 14.324, 14.332, 14.378, 15.561, 17.048,
                     17.900, 17.950, 18.7277, 22.94, 24.3364, 25.907, 26.0092])
# source 4 and ULX: adds H7beta, H7alpha and [ArIII] 21.8
LINELIST4 = np.array([5.057, 5.130, 5.342, 5.513, 5.611, 5.910, 6.109, 6.113, 6.294,
                      6.638, 6.912, 6.988, 7.355, 7.462, 7.505, 7.5104, 7.6527, 8.0258,
                      8.9921, 9.6659, 10.5110, 10.6833, 9.3948, 11.006, 8.763, 7.7836,
                      11.3171, 12.288, 12.382, 12.824, 14.324, 14.332, 14.378, 15.561,
                      17.048, 17.900, 17.950, 18.7277, 19.069, 21.840, 22.94, 24.3364,
                      25.907, 26.0092])

# Rest wavelengths and labels for the multi-channel figures
LINES1 = [5.0528, 5.1283, 5.3398, 5.5108, 5.6091, 5.9075, 6.1079, 6.6356, 6.9088,
          6.9848, 7.3518, 7.4596, 7.5021]
LAB1 = ['S(8)', r'Hu$\delta$', '[FeII]', 'S(7)', '[MgV]', r'Hu$\gamma$', 'S(6)',
        '[NiII]', 'S(5)', '[ArII]', '[NiIII]', r'Pf$\alpha$', r'Hu$\beta$']
LINES2 = [7.6515, 8.0251, 8.7597, 8.9914, 9.6640, 10.510, 10.6817, 11.0020]
LAB2 = ['[NeVI]', 'S(4)', r'H7$\gamma$', '[ArIII]', 'S(3)', '[SIV]', '[NiII]', '[NiIII]']
LINES3 = [12.2788, 12.3715, 12.8139, 14.3207, 14.3661, 15.5561, 17.0352, 17.8850, 17.9359]
LAB3 = ['S(2)', r'Hu$\alpha$', '[NeII]', '[NeV]', '[ClII]', '[NeIII]', 'S(1)',
        '[PIII]', '[FeII]']
LINES4 = [17.8858, 17.9359, 18.7123, 22.9283, 24.3178, 25.8872, 25.9878]
LAB4 = ['[PIII]', '[FeII]', '[SIII]', '[FeIII]', '[NeV]', '[OIV]', 'FeII']

# Per-source settings of the differential spectra and multi-channel figures.
# panels: (lines, labels, dir, ylim, spoff, dfoff, vfraclab) per channel
SOURCES = {
    3: dict(vlsr=230, fac=1.0004, linelist=LINELIST, chans=[1, 2, 3, 4], panels=[
        (LINES1, LAB1, [1, 0, 0, 1, 0, 0, -1, 1, -1, 1, 0, 0, 0], [4000, 5e4], 5000, 4800, 0.4),
        (LINES2, LAB2, [1, 1, 0, 1, 1, 1, 0, 0], [6000, 20000], 5000, 6400, 0.4),
        (LINES3, LAB3, [-1, 0, 1, -1, 0, 1, 1, 0, -1], [8e3, 2e5], 5000, 0.95e4, 0.5),
        (LINES4, LAB4, [0, 1, 0, 0, 1, 1, 0], [1.8e4, 1e5], 5000, 2e4, 0.4)]),
    1: dict(vlsr=110, fac=1.0, linelist=LINELIST, chans=[1, 2, 3, 4], panels=[
        (LINES1, LAB1, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [4000, 2e5], 5000, 5500, 0.29),
        (LINES2, LAB2, [1, 0, 0, 0, 0, 0, 0, 0], [6000, 40000], 5000, 7400, 0.4),
        (LINES3, LAB3, [0, 0, 0, -1, 0, 0, 0, 0, 0], [1.2e4, 7e5], 5000, 1.6e4, 0.4),
        (LINES4, LAB4, [0, 0, 0, 0, 1, 0, 0], [3.5e4, 3e5], 5000, 3.9e4, 0.4)],
        # insets: (axes position, x range, y range) of the differential spectrum
        insets=[([0.303, 0.13, 0.06, 0.14], [5.58, 5.64], [0, 1000]),
                ([0.15, 0.36, 0.06, 0.14], [7.58, 7.72], [600, 2300]),
                ([0.34, 0.58, 0.06, 0.14], [14.2, 14.45], [1000, 2000]),
                ([0.55, 0.83, 0.06, 0.14], [24.15, 24.5], [6000, 7200])]),
    2: dict(vlsr=150, fac=1.0, linelist=LINELIST, chans=[1, 2, 3, 4], panels=[
        (LINES1, LAB1, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [4400, 3e4], 5000, 5500, 0.4),
        (LINES2, LAB2, [0, 0, 0, 1, 0, 1, 0, 0], [5000, 20000], 5000, 6000, 0.4),
        (LINES3, LAB3, [0, 0, 0, -1, 0, 0, 0, 0, 0], [0.7e4, 1e5], 5000, 1e4, 0.4),
        (LINES4, LAB4, [0, 0, 0, 0, 0, 1, 0], [1e4, 1e5], 5000, 2e4, 0.4)]),
    # Matlab computes only ch4 for source 4 (for ch=4:4)
    4: dict(vlsr=250, fac=1.0004, linelist=LINELIST4, chans=[4], panels=[
        (LINES1, LAB1, [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 1], [5000, 0.5e5], 5000, 5500, 0.29),
        (LINES2 + [11.3087], LAB2 + [r'H7$\beta$'], [0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0.7e4, 3e4], 5000, 8000, 0.4),
        (LINES3, LAB3, [0, 1, 0, -1, 0, 0, 0, -1, 0], [1.2e4, 7e5], 5000, 1.6e4, 0.4),
        ([17.8858, 17.9359, 18.7123, 19.0625, 21.8314, 22.9283, 24.3178, 25.8872, 25.9878],
         ['[PIII]', '[FeII]', '[SIII]', r'H7$\alpha$', '[ArIII]', '[FeIII]', '[NeV]',
          '[OIV]', 'FeII'], [1, 0, 1, 1, 1, 0, 0, 1, 0], [3.5e4, 3e5], 5000, 3.9e4, 0.4)]),
}


def _mround(v):
    """Matlab round (half away from zero) for positive values, as an int."""
    return int(np.floor(v + 0.5))


def _offsets(r):
    """RA/Dec offset axes (arcsec) from the reference position, as in mkNeVmap."""
    nx, ny = int(r['numpt'][0]), int(r['numpt'][1])
    x = (r['wcs1'] - REFRA) * 3600 * np.cos(np.radians(REFDE))
    x = x[_mround(nx / 2) - 1, :]
    y = (r['wcs2'] - REFDE) * 3600
    y = y[:, _mround(ny / 2) - 1]
    return x, y


def _xy(ra, dec):
    """Offsets (arcsec) of a sexagesimal position from the reference position."""
    return ((sex2dec(ra) * 15 - REFRA) * np.cos(np.radians(REFDE)) * 3600,
            (sex2dec(dec) - REFDE) * 3600)


def _out(name):
    return os.path.join(OUT_DIR, name)


# ---------------------------------------------------------------------------
# %% Figure for NeV map
# ---------------------------------------------------------------------------

def nev_map_figure():
    """[NeV] peak intensity map with SNR contours and sources.  NeV-peakSNR.png"""
    cube = CUBE.replace('?', '3')
    maps = make_map(cube, _out('NeV14.3-M82v2'), [14.27, 14.317, 14.3435, 14.355], 3,
                    [14.3185, 14.3440], 'reject4')
    cc = read_fits(cube, 'head xten1')
    x, y = _offsets(cc)
    rah = REFRA / 15
    print(f'Offsets from RA={int(rah):02d}:{int(rah*60 % 60):02d}:{rah*3600 % 60:06.3f}, '
          f'DE={int(REFDE):+02d}:{int(REFDE*60 % 60):02d}:{REFDE*3600 % 60:06.3f}')
    maps['peak'][56, 66] = np.nan
    kk = maps['peak'].copy()
    kk[kk == 0] = np.nan

    fig = plt.figure(figsize=(10, 5.2))
    ax = fig.add_axes([0.03, 0.09, 0.9, 0.9])
    ax.set_facecolor([0.4] * 3)
    im = ax.pcolormesh(x, y, kk.T, shading='nearest', vmin=-50, vmax=2000)
    ax.contour(x, y, (maps['peak'] / maps['rms']).T, [6, 10, 15], colors='w')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label('Peak Intensity (MJy/sr)', fontsize=18)
    ax.set_xlabel('R.A. Offset (arcsec)', fontsize=16)
    ax.set_ylabel('Dec. Offset (arcsec)', fontsize=16)
    ax.tick_params(labelsize=16, width=2)

    for i, (ra, dec) in enumerate(PKA):
        xx, yy = _xy(ra, dec)
        axx, ayy = ARROW[i]
        ax.plot(xx + np.array([0, axx]), yy + np.array([0, ayy]), '-w')
        ax.text(xx + axx + np.sign(axx) * 0.4, yy + ayy + np.sign(ayy) * 0.4, str(i + 1),
                color='w', ha='center', va='center', fontsize=20)

    if os.path.exists(FENECH):
        import re
        with open(FENECH) as fp:
            regs = re.findall(r'circle\(([^,]+),([^,]+),', fp.read())
        for i, (ra, dec) in enumerate(regs):
            xx, yy = _xy(ra.strip(), dec.strip())
            if abs(xx) < 16 and abs(yy) < 9:
                ax.text(xx, yy, str(i + 1), color='m', ha='center', va='center', fontsize=12)
    else:
        print(f'{FENECH} not found: skipping Fenech SNR labels')

    ax.set_xlim(16, -16)
    ax.set_ylim(-9, 9)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_linewidth(2)

    ax.plot(*_xy('09:55:52.3', '69:40:46.1'), '+w', markersize=16, mew=1.5)
    L = 200 / DIST * 180 * 3600 / np.pi
    ax.plot([15, 15 - L], [-8, -8], '-w', linewidth=3)
    ax.text(15 - L / 2, -7.9, '200 pc', ha='center', va='bottom', color='w', fontsize=16)
    # aperture and annulus used for the ch3 differential spectra
    ax.add_patch(Circle((-12.5, 6.5), R[0, 2], color='w', ec='none'))
    ax.add_patch(Annulus((-12.5, 6.5), R[1, 2], R[1, 2] - R[2, 2], color='w', ec='none'))
    bm = 1.2 * 14.3e-6 / 6.5 * 180 * 3600 / np.pi / 2
    ax.add_patch(Circle((14.5, -6), bm, color='w', ec='none'))
    ax.text(14, -6, 'Resolution', color='w', va='center', fontsize=16)
    fig.savefig(_out('NeV-peakSNR.png'), dpi=100, facecolor='w', bbox_inches='tight')
    plt.close(fig)
    return maps


# ---------------------------------------------------------------------------
# %% X-ray image
# ---------------------------------------------------------------------------

def xray_figure():
    """X-ray image with ch3/ch4 footprints, sources and ULX boxes.  Xray-image.png"""
    if not os.path.exists(XRAY):
        print(f'{XRAY} not found: skipping X-ray figure')
        return
    xray = read_fits(XRAY, 'wcs')
    nx, ny = int(xray['numpt'][0]), int(xray['numpt'][1])
    xxr = (xray['wcs1'][:, _mround(nx / 2) - 1] - REFRA) * 3600 * np.cos(np.radians(REFDE))
    yxr = (xray['wcs2'][_mround(ny / 2) - 1, :] - REFDE) * 3600
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.sqrt(xray['data']).T, origin='lower', vmin=0, vmax=60,
              extent=[xxr[0], xxr[-1], yxr[0], yxr[-1]])
    for ch, col in ((3, 'r'), (4, 'm')):
        ifu = read_fits(CUBE.replace('?', str(ch)), 'xten1')
        x, y = _offsets(ifu)
        i1 = np.all(np.isfinite(ifu['data'][:, :, :-15]), axis=2).astype(float)
        ax.contour(x, y - 0.5, i1.T, [0.5], colors=col, linewidths=2)
    th = np.radians(np.arange(0, 361, 10))
    for ra, dec in PKA:
        xx, yy = _xy(ra, dec)
        ax.plot(xx + np.cos(th), yy + np.sin(th), 'w-', linewidth=2)
    ax.plot(0, 0, '+w', markersize=20)
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]).T
    a = np.radians(25)
    sqr = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ sq
    xu, yu = _xy('09:55:50.11', '69:40:46.76')
    ax.plot(xu + 2.75 * sqr[0], yu + 2.75 * sqr[1], 'w-', linewidth=2)
    ax.plot(xu + 1.5 * sqr[0], yu + 1.5 * sqr[1], '-', linewidth=2, color=[1, 0.647, 0])
    ax.set_xlim(20, -20)
    ax.set_ylim(-20, 20)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=16)
    ax.set_xlabel('R.A. Offset (")', fontsize=16)
    ax.set_ylabel('Dec. Offset (")', fontsize=16)
    fig.savefig(_out('Xray-image.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# %% Differential spectra and multi-channel figures
# ---------------------------------------------------------------------------

def diff_spectra(src):
    """Differential spectra of a source in its channels; returns {ch: dss}."""
    cfg = SOURCES[src]
    return {ch: diff_spec(CUBE, PKA, ch, src, R, cfg['linelist'] * cfg['fac'],
                          cfg['vlsr'], outdir=OUT_DIR)
            for ch in cfg['chans']}


def plot_spec(fig, nplt, sp, lines, linelab, dirs, ylim, spoff, dfoff, vfraclab):
    """Port of the plotSpec local function: one channel panel of the figure."""
    ha = {-1: 'right', 1: 'left'}
    color = [0, 0.447, 0.741]
    afs, lfs = 14, 16          # axis and line label font sizes
    offy, offx, spcy, sizx, topy = 0.04, 0.05, 0.005, 0.945, 0.015
    sizy = (1 - (offy + 3 * spcy + topy)) / 4

    ax = fig.add_axes([offx, offy + (nplt - 1) * (sizy + spcy), sizx, sizy])
    ax.semilogy(sp[:, 0], spoff + sp[:, 2], color=color, linewidth=1)
    ax.semilogy(sp[:, 0], dfoff + sp[:, 1], color='k', linewidth=1)
    ax.set_xlim(np.min(sp[:, 0]), np.max(sp[:, 0]))
    if ylim is not None:
        ax.set_ylim(min(ylim), max(ylim))
    y0, y1 = ax.get_ylim()
    spwi = np.max(sp[:, 0]) - np.min(sp[:, 0])
    spht = np.log10(y1) - np.log10(y0)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.patch.set_alpha(0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=afs)
    for i, d in enumerate(dirs):
        if d == 0:
            continue
        wvix = np.abs(sp[:, 0] - lines[i]) < spwi / 50
        vst = np.nanmedian(sp[wvix, 1]) + dfoff
        ven = y0 * 10**(spht * vfraclab)
        ax.plot(lines[i] + d * np.array([0.002, 0.015]) * spwi, [vst, ven], 'k-')
        ax.text(lines[i] + d * 0.0155 * spwi, ven, linelab[i], ha=ha[d],
                va='baseline', fontsize=lfs)
    return ax


def multi_channel_figure(src, dss):
    """Multi-channel spectrum figure of a source.  diffSpecSrc{src}.png"""
    cfg = SOURCES[src]
    vfac = 1 + cfg['vlsr'] / 3e5
    fig = plt.figure(figsize=(10, 6.7), facecolor='w')
    for ch, (lines, lab, dirs, ylim, spoff, dfoff, vfrac) in enumerate(cfg['panels'], 1):
        if ch not in dss:
            print(f'Source {src}: no ch{ch} differential spectrum, panel skipped')
            continue
        ax = plot_spec(fig, ch, dss[ch], np.array(lines) * vfac, lab, dirs, ylim,
                       spoff, dfoff, vfrac)
        if ch == 1:
            ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=14)
    fig.text(0.005, 0.5, r'Intensity (MJy sr$^{-1}$)', rotation=90, va='center',
             fontsize=14)
    for ch, (pos, xr, yr) in enumerate(cfg.get('insets', []), 1):
        if ch not in dss:
            continue
        ia = fig.add_axes(pos, facecolor='w')
        ia.plot(dss[ch][:, 0], dss[ch][:, 1], '-k', linewidth=2)
        ia.set_xlim(xr)
        ia.set_ylim(yr)
        ia.set_xticks([])
        ia.set_yticks([])
        for sp in ia.spines.values():
            sp.set_linewidth(2)
    fig.savefig(_out(f'diffSpecSrc{src}.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)


def stitched_spectrum(dss):
    """Full stitched spectrum (Matlab notes it is impractical).  Figure only."""
    color = [0, 0.447, 0.741]
    fig, ax = plt.subplots(figsize=(12, 5))
    ix = dss[1][:, 0] < 7.35
    ax.plot(dss[1][ix, 0], 100 + dss[1][ix, 1], color=color, linewidth=1)
    ax.plot(dss[2][:, 0], dss[2][:, 1], color=color, linewidth=1)
    ax.plot(dss[3][:, 0], dss[3][:, 1], color=color, linewidth=1)
    ix = dss[4][:, 0] < 27
    ax.plot(dss[4][ix, 0], 900 + dss[4][ix, 1], color=color, linewidth=1)
    ax.axis([4.9, 27, -600, 22000])
    fig.savefig(_out('stitchedSpecSrc3.png'), dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# %% Gaussian fits to broadened lines in the brightest source (src 3)
# ---------------------------------------------------------------------------

# (channel, baseline channels (Matlab 1-based ranges), baseline order, last
#  channel of fit region or None, ey overrides (1-based ranges, value),
#  initial K, S, X0, line name)
BROAD_FITS = [
    (1, [(2530, 2590), (2640, 2720)], 5, None, [((82, 90), 1000), ((86, 86), 1e4)],
     6000, 0.2, 6.99, '[ArII]'),
    (1, [(2100, 2150), (2200, 2240)], 3, None, [], 1500, 0.05, 6.64, '[NiII]'),
    (3, [(1540, 1590), (1630, 1680)], 3, None, [((64, 72), 20000)], 3000, 0.05, 15.57, '[NeIII]'),
    (3, [(400, 490), (530, 650)], 3, None, [((106, 114), 30000)], 10000, 0.05, 12.82, '[NeII]'),
    (3, [(2500, 2540)], 1, 2572, [], 300, 0.05, 17.95, '[FeII]'),
]


def broad_line_fits(dss):
    """Gaussian fits (weighted by ey) to broadened lines; prints FWHM in km/s."""
    results = {}
    for ch, bs, order, last, eyset, K, S, X0, name in BROAD_FITS:
        wv, sp = dss[ch][:, 0], dss[ch][:, 1]
        bsix = np.concatenate([np.arange(a - 1, b) for a, b in bs])
        p = np.polyfit(wv[bsix], sp[bsix], order)
        spix = np.arange(bsix.min(), (last if last else bsix.max() + 1))
        wvp = wv[spix]
        spp = sp[spix] - np.polyval(p, wvp)
        noise = np.std(sp[bsix] - np.polyval(p, wv[bsix]), ddof=1)
        ey = np.ones_like(spp) * noise
        for (a, b), v in eyset:
            ey[a - 1:b] = v
        Ko, So, X0o, chi2, ymodel, err, epar = gaussfit(wvp, spp, K, S, X0, 1e-5, 0,
                                                        [1, 1, 1], ey)
        fwhm = So * np.sqrt(np.log(256))
        fwhmv = fwhm / X0o * 3e5
        print(f'FWHM of {name}, {fwhmv:f} km/s')
        results[name] = (Ko, So, X0o, fwhmv, err, epar)
    return results


# ---------------------------------------------------------------------------
# %% ULX X-1 ch4 spectrum
# ---------------------------------------------------------------------------

def ulx_spectrum():
    """ch4 spectrum of ULX X-1 and [OIV] brightness/flux/luminosity.  ULX-1-spec.png"""
    cube = read_fits(CUBE.replace('?', '4'), 'xten1')
    q = cat_peaks(cube, _out('ULX-1'), pixco=('09:55:50.102', '69:40:46.9', 0.4),
                  snr=1, linelist=LINELIST4)
    print(f"Brightness of [OIV]: {q['area'][6]:0.2g} +/- {q['earea'][6]:0.2g} W/m2/sr ")
    b2f = np.pi * (R[0, 3] / 3600 * np.pi / 180)**2
    print(f"Flux for [OIV] (no ap corr): {q['area'][6]*b2f:0.2g} +/- "
          f"{q['earea'][6]*b2f:0.2g} W/m2 ")
    f2l = 4 * np.pi * (3.6e6 * 3.09e16)**2 / 3.83e26
    print(f"Luminosity for [OIV] (no ap corr): {q['area'][6]*b2f*f2l:0.2g} +/- "
          f"{q['earea'][6]*b2f*f2l:0.2g} Lsun ")
    cat_peaks(cube, _out('ULX-1-bg'), pixco=('09:55:50.43', '69:40:47.646', 0.4),
              snr=1, linelist=LINELIST4)
    sp = np.loadtxt(_out('ULX-1_spec.txt'), comments='%')

    fig = plt.figure(figsize=(3.6, 3.6))
    ax = fig.add_axes([0.12, 0.14, 0.87, 0.78])
    ax.plot(sp[:, 0], sp[:, 1], 'k', linewidth=2)
    ax.set_xlim(np.min(sp[:, 0]), np.max(sp[:, 0]))
    ax.text(23, 6e4, '[FeIII]', fontsize=16)
    ax.text(18.9, 13e4, '[SIII]', fontsize=16)
    ax.text(19.1, 3e4, r'H7$\alpha$', fontsize=16)
    ax.set_xlabel(r'Wavelength ($\mu$m)', fontsize=16)
    ax.set_ylabel('Intensity (MJy/sr)', fontsize=16)
    ax.set_yscale('log')
    ax.tick_params(labelsize=16, width=2)
    ia = fig.add_axes([0.65, 0.55, 0.25, 0.4])
    ia.plot(sp[:, 0], sp[:, 1], '-k', linewidth=1)
    ia.axis([25.7, 26.2, 4e4, 5.5e4])
    ia.text(25.71, 5e4, '[OIV]', fontsize=15)
    ia.text(26.02, 5.1e4, '[FeII]', fontsize=15)
    ia.set_xticks([])
    ia.set_yticks([])
    fig.savefig(_out('ULX-1-spec.png'), dpi=100, bbox_inches='tight')
    plt.close(fig)
    return q


SECTIONS = ['nev', 'xray', 'src3', 'broad', 'stitched', 'src1', 'src2', 'src4', 'ulx']


def run(sections=SECTIONS):
    """Run the selected mkNeVmap cells in Matlab order; returns their results."""
    res = {}
    if 'nev' in sections:
        res['map'] = nev_map_figure()
    if 'xray' in sections:
        xray_figure()
    for src in (3, 1, 2, 4):
        if f'src{src}' in sections or (src == 3 and {'broad', 'stitched'} & set(sections)):
            dss = diff_spectra(src)
            res[f'dss{src}'] = dss
            if f'src{src}' in sections:
                multi_channel_figure(src, dss)
            if src == 3 and 'broad' in sections:
                res['broad'] = broad_line_fits(dss)
            if src == 3 and 'stitched' in sections:
                stitched_spectrum(dss)
    if 'ulx' in sections:
        res['ulx'] = ulx_spectrum()
    return res


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Port of mkNeVmap.m')
    parser.add_argument('sections', nargs='*', default=SECTIONS,
                        help=f'cells to run (default all): {" ".join(SECTIONS)}')
    parser.add_argument('--outdir', default=None,
                        help='output directory (default mk_nev_map_output/)')
    args = parser.parse_args()
    if args.outdir:
        OUT_DIR = args.outdir
    os.makedirs(OUT_DIR, exist_ok=True)
    run(args.sections)

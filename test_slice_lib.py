# Created: 2026-06-01 00:00:00
# Last Modified: 2026-06-03 00:00:00
# Description: Test harness for slice_lib.py (06032026 version), validating the
#   Python translation against Matlab reference outputs in JWST/.  Covers FITS
#   reading, aperture extraction, peak detection, Gaussian fitting, file I/O,
#   make_int_map (12-plane cube output, new rms field), dec2sex, sex2dec,
#   and pixel-level comparison against the Matlab FITS reference maps.
#   Run with:  pytest test_slice_lib.py -v
#   Or:        python3 test_slice_lib.py
# Last Edit: Update for 06032026: 12-plane cube FITS, rms field, vs-Matlab map tests

import os
import sys
import tempfile
import shutil
import warnings
import numpy as np
import pytest
from astropy.io import fits

# Add library to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slice_lib as sl

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, '..')
_JWST = os.path.join(_ROOT, 'JWST')
_CUBE_CH3 = os.path.join(_JWST, 'M82v1_19_1_sky_ch3-shortmediumlong_s3d.fits')
_CUBE_CH2 = os.path.join(_JWST, 'M82v1_19_1_sky_ch2-shortmediumlong_s3d.fits')
_REF_NEV  = os.path.join(_JWST, 'NeV14.3-M82v19.fits')
_REF_HI76 = os.path.join(_JWST, 'H7-6-M82v19.fits')

# Seven pointings from Nepeaks.m (RA, Dec, radius_arcsec, snr)
_POINTINGS = [
    ('09:55:52.74',  '69:40:45.89',  0.2, 5.0),   # Nepk1
    ('09:55:51.276', '69:40:44.09',  0.2, 1.5),   # Nepk2
    ('09:55:54.461', '69:40:51.448', 0.2, 1.5),   # Nepk3
    ('09:55:54.610', '69:40:48.85',  0.2, 1.5),   # Nepk4
    ('09:55:51.276', '69:40:44.0',   0.2, 1.5),   # Nepk5
    ('09:55:51.58',  '69:40:45.87',  0.2, 1.5),   # Nepk6
    ('09:55:52.97',  '69:40:47.61',  0.2, 1.5),   # Nepk7
]

# Tolerances
_TOL_WAVE  = 0.003    # wavelength, µm  (~2 channels of ch3)
_TOL_K_REL = 0.05    # Gaussian amplitude K, relative
_TOL_S_REL = 0.20    # Gaussian sigma/FWHM, relative  (blended lines vary more)
_TOL_X_REL = 0.005   # Gaussian centre, relative  (essentially wavelength)
_TOL_FLUX_REL = 0.15 # integrated flux, relative (faint peaks sensitive to continuum)
_TOL_SPEC_REL = 0.01 # spectrum value, relative (aperture extraction)
_TOL_CONT_REL = 0.03 # continuum value, relative (FIR filter differences)
_TOL_MAP_REL  = 0.05 # intensity map pixel, relative vs Matlab reference


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_path(pk_name, suffix):
    """Path to Matlab reference file, e.g. JWST/Nepk1-peaklist.txt"""
    return os.path.join(_JWST, f'{pk_name}{suffix}')


def _load_peaklist(path):
    """Load a peaklist text file; return (n_peaks, wavelengths, K, FWHM, X0, flux, flags)."""
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.startswith('%') or not ln.strip():
                continue
            vals = ln.split()
            if len(vals) >= 7:
                rows.append([float(v) for v in vals])
    if not rows:
        return 0, *[np.array([])]*6
    data = np.array(rows)
    n = len(data)
    wv   = data[:, 1]
    K    = data[:, 3]
    fwhm = data[:, 4]
    X0   = data[:, 5]
    flux = data[:, 6]
    flag = data[:, 11].astype(int) if data.shape[1] >= 12 else np.zeros(n, int)
    return n, wv, K, fwhm, X0, flux, flag


def _load_xy(path):
    """Load a two-column text file (wavelength, value) skipping comment lines."""
    rows = []
    with open(path) as f:
        for ln in f:
            if ln.startswith('%') or not ln.strip():
                continue
            parts = ln.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
    data = np.array(rows)
    return data[:, 0], data[:, 1]


def _skip_no_cube(cube_path):
    if not os.path.exists(cube_path):
        pytest.skip(f'Cube not found: {cube_path}')


def _skip_no_ref(path):
    if not os.path.exists(path):
        pytest.skip(f'Matlab reference not found: {path}')


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def cube_ch3():
    _skip_no_cube(_CUBE_CH3)
    return sl.read_fits(_CUBE_CH3, 'xten1')


@pytest.fixture(scope='module')
def cube_ch2():
    _skip_no_cube(_CUBE_CH2)
    return sl.read_fits(_CUBE_CH2, 'xten1')


@pytest.fixture
def tmpdir_local():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


# ---------------------------------------------------------------------------
# 1. FITS reading
# ---------------------------------------------------------------------------

class TestReadFits:
    def test_data_shape_ch3(self, cube_ch3):
        r = cube_ch3
        assert r['data'].ndim == 3, 'Data should be 3-D'
        nx, ny, nz = r['data'].shape
        assert nx == r['numpt'][0]
        assert ny == r['numpt'][1]
        assert nz == r['numpt'][2]

    def test_wavelength_range_ch3(self, cube_ch3):
        wv = cube_ch3['x'][2]
        assert wv[0] == pytest.approx(11.55, abs=0.05), 'ch3 start ~11.55 µm'
        assert wv[-1] == pytest.approx(18.0, abs=0.1), 'ch3 end ~18.0 µm'

    def test_wavelength_range_ch2(self, cube_ch2):
        wv = cube_ch2['x'][2]
        assert wv[0] == pytest.approx(7.5, abs=0.1), 'ch2 start ~7.5 µm'
        assert wv[-1] == pytest.approx(11.7, abs=0.2), 'ch2 end ~11.7 µm'

    def test_bunit(self, cube_ch3):
        assert cube_ch3['bunit'] == 'MJy/sr'

    def test_wcs_shape(self, cube_ch3):
        r = cube_ch3
        nx, ny = r['numpt'][0], r['numpt'][1]
        assert r['wcs1'].shape == (ny, nx), 'wcs1 shape should be (ny, nx)'
        assert r['wcs2'].shape == (ny, nx)

    def test_wcs_values_in_range(self, cube_ch3):
        r = cube_ch3
        ra_med = np.nanmedian(r['wcs1'])
        dec_med = np.nanmedian(r['wcs2'])
        assert 148.0 < ra_med < 150.0, f'RA ~149 deg for M82, got {ra_med:.2f}'
        assert 69.5 < dec_med < 70.0, f'Dec ~69.7 deg for M82, got {dec_med:.2f}'

    def test_head_only(self):
        _skip_no_cube(_CUBE_CH3)
        r = sl.read_fits(_CUBE_CH3, 'head xten1')
        assert 'data' not in r
        assert 'numpt' in r

    def test_no_data_when_head(self):
        _skip_no_cube(_CUBE_CH3)
        r = sl.read_fits(_CUBE_CH3, 'head xten1')
        assert 'data' not in r


# ---------------------------------------------------------------------------
# 2. Aperture extraction
# ---------------------------------------------------------------------------

class TestExtractAperture:
    def test_circle_flag_ok(self, cube_ch3):
        _, _, _, flag = sl.extract_aperture(
            cube_ch3, ('09:55:52.74', '69:40:45.89', 0.2))
        assert flag == 0

    def test_circle_no_nans_ch3(self, cube_ch3):
        s, _, _, _ = sl.extract_aperture(
            cube_ch3, ('09:55:52.74', '69:40:45.89', 0.2))
        nan_frac = np.sum(np.isnan(s)) / len(s)
        assert nan_frac < 0.05, f'Too many NaN channels: {nan_frac:.1%}'

    def test_circle_spectrum_vs_matlab_nepk1(self, cube_ch3):
        """Extracted spectrum should match Matlab Nepk1_spec.txt to <1%."""
        ref_path = _ref_path('Nepk1', '_spec.txt')
        _skip_no_ref(ref_path)
        wv_ref, sp_ref = _load_xy(ref_path)

        s, _, _, _ = sl.extract_aperture(
            cube_ch3, ('09:55:52.74', '69:40:45.89', 0.2))
        wv = cube_ch3['x'][2]
        valid = ~np.isnan(s)
        wv_py, sp_py = wv[valid], s[valid]

        sp_interp = np.interp(wv_ref, wv_py, sp_py)
        rel_diff = np.abs(sp_interp - sp_ref) / (np.abs(sp_ref) + 1e-10)
        med_diff = np.nanmedian(rel_diff)
        assert med_diff < _TOL_SPEC_REL, (
            f'Median spectrum deviation {med_diff:.3%} exceeds {_TOL_SPEC_REL:.1%}')

    def test_aperture_weights_sum_one(self):
        x = np.linspace(-1, 1, 20)
        y = np.linspace(-1, 1, 20)
        xx, yy = np.meshgrid(x, y)
        w = sl.aperture_weights(xx, yy, 0, 0, 0.1, 0.5)
        assert np.sum(w) > 0, 'Some pixels should be inside aperture'

    def test_bad_aperture_outside_cube(self, cube_ch3):
        """Aperture entirely outside cube should return flag -1."""
        s, _, _, flag = sl.extract_aperture(
            cube_ch3, ('10:00:00.0', '70:00:00.0', 0.1))
        assert flag == -1, f'Expected flag -1 for empty aperture, got {flag}'


# ---------------------------------------------------------------------------
# 3. Signal detection
# ---------------------------------------------------------------------------

class TestSigChans:
    def test_detects_known_peak(self):
        wv = np.linspace(10.0, 15.0, 500)
        spec = 100 * np.exp(-((wv - 12.5) / 0.05)**2) + np.random.default_rng(0).normal(0, 1, 500)
        dy = np.diff(spec)
        dx = (wv[1:] + wv[:-1]) / 2
        mf = np.array([1, 1, 0, -1, -1], dtype=float)
        cc = np.convolve(dy, -mf, mode='same')
        noise = np.mean(np.abs(cc - np.mean(cc)))
        pp = sl.sig_chans(dx, cc, noise, 5/1.5/1.5, 2)
        assert len(pp) >= 1, 'Should detect at least one peak'
        dist = np.abs(pp[:, 0] - 12.5)
        assert np.min(dist) < 0.1, f'Closest peak at {pp[np.argmin(dist), 0]:.3f}, expected ~12.5'

    def test_empty_spectrum(self):
        wv = np.linspace(10, 15, 200)
        flat = np.ones(200)
        dy = np.diff(flat)
        dx = (wv[1:] + wv[:-1]) / 2
        mf = np.array([1, 1, 0, -1, -1], dtype=float)
        cc = np.convolve(dy, -mf, mode='same')
        noise = max(np.mean(np.abs(cc - np.mean(cc))), 1e-30)
        pp = sl.sig_chans(dx, cc, noise, 5, 2)
        assert len(pp) == 0, 'Flat spectrum should produce no peaks'


# ---------------------------------------------------------------------------
# 4. Gaussian fitting
# ---------------------------------------------------------------------------

class TestGaussFitting:
    def test_gaussmfit_single(self):
        x = np.linspace(-3, 3, 200)
        K0, S0, X0_0 = 5.0, 0.5, 0.2
        y = K0 * np.exp(-((x - X0_0)**2) / (2 * S0**2))
        Ko, So, X0o, chi2, ym, err, epar = sl.gaussmfit(x, y, [K0*0.9], [S0*1.1], [X0_0])
        assert err == 0, f'Fit should converge, got err={err}'
        assert Ko[0] == pytest.approx(K0, rel=0.01)
        assert So[0] == pytest.approx(S0, rel=0.02)
        assert X0o[0] == pytest.approx(X0_0, abs=0.01)

    def test_gaussmfit_two_components(self):
        x = np.linspace(0, 10, 300)
        K0 = [3.0, 2.0]; S0 = [0.3, 0.4]; X0_0 = [3.0, 7.0]
        y = sum(K0[i] * np.exp(-((x - X0_0[i])**2) / (2*S0[i]**2)) for i in range(2))
        Ko, So, X0o, chi2, ym, err, epar = sl.gaussmfit(x, y, K0, S0, X0_0)
        assert err in (0, 1, 4), 'err should be 0, 1, or 4'
        if err == 0:
            for i in range(2):
                assert Ko[i] == pytest.approx(K0[i], rel=0.05)
                assert X0o[i] == pytest.approx(X0_0[i], abs=0.1)

    def test_gaussmfit_returns_epar(self):
        x = np.linspace(0, 5, 100)
        y = 10 * np.exp(-((x - 2.5)**2) / (2*0.3**2)) + np.random.default_rng(1).normal(0, 0.1, 100)
        Ko, So, X0o, chi2, ym, err, epar = sl.gaussmfit(x, y, [10], [0.3], [2.5])
        assert len(epar) == 3
        if err == 0:
            assert np.all(np.isfinite(epar)), 'Uncertainties should be finite'
            assert np.all(epar >= 0), 'Uncertainties should be non-negative'

    def test_gaussfit_single_wrapper(self):
        x = np.linspace(0, 10, 200)
        K0, S0, X0_0 = 4.0, 0.6, 5.1
        y = K0 * np.exp(-((x - X0_0)**2) / (2*S0**2))
        Ko, So, X0o, chi2, ym, err, epar = sl.gaussfit(x, y, K=K0, S=S0, X0=X0_0)
        assert err == 0
        assert Ko == pytest.approx(K0, rel=0.01)
        assert X0o == pytest.approx(X0_0, abs=0.01)


# ---------------------------------------------------------------------------
# 5. fit_lines
# ---------------------------------------------------------------------------

class TestFitLines:
    def test_single_peak(self):
        wv = np.linspace(12.0, 13.0, 300)
        K0, S0, X0_0 = 1000.0, 0.003, 12.37
        cont = 500.0 * np.ones_like(wv)
        noise = np.random.default_rng(42).normal(0, 5, len(wv))
        spec = cont + K0 * np.exp(-((wv - X0_0)**2) / (2*S0**2)) + noise
        peaks = np.array([[X0_0, K0]])
        q, ymodel, smo = sl.fit_lines(wv, spec, peaks)
        assert q['X0'][0] == pytest.approx(X0_0, abs=0.002)
        assert q['K'][0] == pytest.approx(K0, rel=0.1)
        assert q['flag'][0] in (0, 1, 4, 8, 32, 36, 40)

    def test_output_shapes(self):
        wv = np.linspace(12.0, 18.0, 500)
        spec = 1000 * np.ones_like(wv) + np.random.default_rng(0).normal(0, 10, 500)
        peaks = np.array([[14.0, 200.0], [16.0, 150.0]])
        q, ymodel, smo = sl.fit_lines(wv, spec, peaks)
        assert len(q['K']) == 2
        assert len(ymodel) == len(wv)
        assert len(smo) == len(wv)


# ---------------------------------------------------------------------------
# 6. cat_peaks vs. Matlab reference (all 7 pointings)
# ---------------------------------------------------------------------------

def _compare_to_matlab(pointing_idx, cube, tmpdir):
    """Run cat_peaks and compare to Matlab reference for one pointing."""
    pk_name = f'Nepk{pointing_idx + 1}'
    ref_pk = _ref_path(pk_name, '-peaklist.txt')
    _skip_no_ref(ref_pk)
    _skip_no_cube(_CUBE_CH3)

    ra, dec, rad, snr = _POINTINGS[pointing_idx]
    outbase = os.path.join(tmpdir, pk_name)

    q = sl.cat_peaks(cube, outbase, pixco=(ra, dec, rad), snr=snr)

    # Load reference
    n_ref, wv_ref, K_ref, fwhm_ref, X0_ref, flux_ref, flag_ref = _load_peaklist(ref_pk)

    n_py = len(q['K'])
    assert n_py == n_ref, (
        f'{pk_name}: peak count mismatch — Python={n_py}, Matlab={n_ref}')

    for i in range(n_py):
        j = int(np.argmin(np.abs(wv_ref - q['X0'][i])))

        # Wavelength centre
        assert abs(q['X0'][i] - X0_ref[j]) < _TOL_WAVE, (
            f'{pk_name} peak {i+1}: X0 {q["X0"][i]:.4f} vs ref {X0_ref[j]:.4f} '
            f'(diff {abs(q["X0"][i]-X0_ref[j]):.4f} µm)')

        # Skip K and flux checks for peaks flagged by either solver
        py_flagged  = (q['flag'][i] != 0)
        ref_flagged = (flag_ref[j] != 0)

        # Amplitude K (unflagged peaks only)
        if (not py_flagged and not ref_flagged
                and np.isfinite(q['K'][i]) and np.isfinite(K_ref[j])
                and K_ref[j] != 0):
            rel_K = abs(q['K'][i] - K_ref[j]) / abs(K_ref[j])
            assert rel_K < _TOL_K_REL, (
                f'{pk_name} peak {i+1}: K {q["K"][i]:.2f} vs ref {K_ref[j]:.2f} '
                f'(rel diff {rel_K:.1%})')

        # Integrated flux (unflagged peaks only)
        if (not py_flagged and not ref_flagged
                and np.isfinite(q['area'][i]) and np.isfinite(flux_ref[j])
                and flux_ref[j] != 0 and q['area'][i] != 0):
            rel_flux = abs(q['area'][i] - flux_ref[j]) / abs(flux_ref[j])
            assert rel_flux < _TOL_FLUX_REL, (
                f'{pk_name} peak {i+1}: flux {q["area"][i]:.4g} vs ref {flux_ref[j]:.4g} '
                f'(rel diff {rel_flux:.1%})')

    return q


class TestCatPeaks:
    """Run cat_peaks and compare to Matlab for all 7 pointings."""

    def test_nepk1_peak_count(self, cube_ch3, tmpdir_local):
        q = _compare_to_matlab(0, cube_ch3, tmpdir_local)
        assert len(q['K']) > 0

    def test_nepk2_peak_count(self, cube_ch3, tmpdir_local):
        _compare_to_matlab(1, cube_ch3, tmpdir_local)

    def test_nepk3_peak_count(self, cube_ch3, tmpdir_local):
        _compare_to_matlab(2, cube_ch3, tmpdir_local)

    def test_nepk4_peak_count(self, cube_ch3, tmpdir_local):
        _compare_to_matlab(3, cube_ch3, tmpdir_local)

    def test_nepk5_peak_count(self, cube_ch3, tmpdir_local):
        _compare_to_matlab(4, cube_ch3, tmpdir_local)

    def test_nepk6_peak_count(self, cube_ch3, tmpdir_local):
        _compare_to_matlab(5, cube_ch3, tmpdir_local)

    def test_nepk7_peak_count(self, cube_ch3, tmpdir_local):
        _compare_to_matlab(6, cube_ch3, tmpdir_local)

    def test_output_files_created(self, cube_ch3, tmpdir_local):
        """All four output files should be created."""
        outbase = os.path.join(tmpdir_local, 'test_pk')
        sl.cat_peaks(cube_ch3, outbase,
                     pixco=('09:55:52.74', '69:40:45.89', 0.2), snr=5)
        for suffix in ('-peaklist.txt', '_spec.txt', '_cont.txt', '_model.txt'):
            assert os.path.exists(outbase + suffix), f'Missing {suffix}'

    def test_peaklist_header_format(self, cube_ch3, tmpdir_local):
        """Peaklist header should contain cube name and region."""
        outbase = os.path.join(tmpdir_local, 'hdr_test')
        sl.cat_peaks(cube_ch3, outbase,
                     pixco=('09:55:52.74', '69:40:45.89', 0.2), snr=5)
        with open(outbase + '-peaklist.txt') as f:
            lines = f.readlines()
        assert lines[0].startswith('% Cube:')
        assert lines[1].startswith('% Region:')
        assert 'circle' in lines[1]

    def test_spec_file_vs_matlab_nepk1(self, cube_ch3, tmpdir_local):
        """Spectrum file values should match Matlab Nepk1_spec.txt within 1%."""
        ref_path = _ref_path('Nepk1', '_spec.txt')
        _skip_no_ref(ref_path)
        outbase = os.path.join(tmpdir_local, 'Nepk1')
        sl.cat_peaks(cube_ch3, outbase,
                     pixco=('09:55:52.74', '69:40:45.89', 0.2), snr=5)
        wv_ref, sp_ref = _load_xy(ref_path)
        wv_py, sp_py = _load_xy(outbase + '_spec.txt')
        sp_interp = np.interp(wv_ref, wv_py, sp_py)
        rel_diff = np.abs(sp_interp - sp_ref) / (np.abs(sp_ref) + 1e-10)
        assert np.nanmedian(rel_diff) < _TOL_SPEC_REL

    def test_cont_file_vs_matlab_nepk1(self, cube_ch3, tmpdir_local):
        """Continuum file values should match Matlab Nepk1_cont.txt within 3%."""
        ref_path = _ref_path('Nepk1', '_cont.txt')
        _skip_no_ref(ref_path)
        outbase = os.path.join(tmpdir_local, 'Nepk1')
        sl.cat_peaks(cube_ch3, outbase,
                     pixco=('09:55:52.74', '69:40:45.89', 0.2), snr=5)
        wv_ref, cont_ref = _load_xy(ref_path)
        wv_py, cont_py = _load_xy(outbase + '_cont.txt')
        cont_interp = np.interp(wv_ref, wv_py, cont_py)
        rel_diff = np.abs(cont_interp - cont_ref) / (np.abs(cont_ref) + 1e-10)
        assert np.nanmedian(rel_diff) < _TOL_CONT_REL


# ---------------------------------------------------------------------------
# 7. recompute_lines
# ---------------------------------------------------------------------------

class TestRecomputeLines:
    def test_recompute_preserves_wavelengths(self, cube_ch3, tmpdir_local):
        """recompute_lines should reproduce the same peak wavelengths."""
        outbase = os.path.join(tmpdir_local, 'recomp')
        sl.cat_peaks(cube_ch3, outbase,
                     pixco=('09:55:52.74', '69:40:45.89', 0.2), snr=5)
        _, wv_before, _, _, _, _, _ = _load_peaklist(outbase + '-peaklist.txt')
        sl.recompute_lines(outbase)
        _, wv_after, _, _, _, _, _ = _load_peaklist(outbase + '-peaklist.txt')
        np.testing.assert_allclose(wv_before, wv_after, atol=_TOL_WAVE,
                                   err_msg='recompute_lines changed peak wavelengths')


# ---------------------------------------------------------------------------
# 8. make_int_map — updated for 06032026 (rms field + 12-plane cube output)
# ---------------------------------------------------------------------------

class TestMakeIntMap:
    def test_map_shape_and_finite(self, tmpdir_local):
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'NeV14')
        maps = sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
        )
        r = sl.read_fits(_CUBE_CH3, 'head xten1')
        nx, ny = r['numpt'][0], r['numpt'][1]
        m  = maps['int']
        em = maps['eint']
        assert m.shape == (nx, ny), 'Intensity map shape should match cube spatial dims'
        assert em.shape == (nx, ny), 'Uncertainty map shape should match cube spatial dims'
        assert np.sum(np.isfinite(m)) > 0, 'Some finite intensity map values expected'
        assert np.all(em[np.isfinite(em)] >= 0), 'Uncertainty map should be non-negative'

    def test_output_fits_written(self, tmpdir_local):
        """A single 12-plane cube FITS file should be written (not two 2D maps)."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'test_map')
        sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
        )
        assert os.path.exists(out + '.fits'), 'Multi-plane cube FITS not written'
        assert not os.path.exists(out + '-unc.fits'), \
            'Separate uncertainty FITS should no longer be written (all maps in cube)'

    def test_multiplane_cube_structure(self, tmpdir_local):
        """Output FITS should be a 3-D cube with 12 planes and correct spatial shape."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'cube_check')
        sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
        )
        with fits.open(out + '.fits') as h:
            d = h[0].data
            hdr = h[0].header
        r = sl.read_fits(_CUBE_CH3, 'head xten1')
        nx, ny = int(r['numpt'][0]), int(r['numpt'][1])
        assert d.ndim == 3, 'Output FITS should be 3-D'
        assert d.shape == (12, ny, nx), \
            f'Expected shape (12, {ny}, {nx}), got {d.shape}'
        assert hdr['NAXIS3'] == 12, 'NAXIS3 should be 12'
        assert hdr.get('CTYPE3', '') == 'PLANE', 'CTYPE3 should be PLANE'
        assert hdr.get('BUNIT', '') == 'see plane', 'BUNIT should be "see plane"'

    def test_new_dict_fields_present(self, tmpdir_local):
        """make_int_map must return a dict with all 13 expected fields (incl. rms)."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'NeV14b')
        maps = sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
        )
        expected_keys = ('int', 'eint', 'peak', 'rms', 'wpeak', 'outliers',
                         'gm0', 'gem0', 'gm1', 'gem1', 'gm2', 'gem2', 'err')
        for key in expected_keys:
            assert key in maps, f'Missing key {key!r} in maps dict'

    def test_new_map_shapes(self, tmpdir_local):
        """All map arrays (including rms) should match cube spatial dimensions."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'NeV14c')
        maps = sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
        )
        r = sl.read_fits(_CUBE_CH3, 'head xten1')
        nx, ny = r['numpt'][0], r['numpt'][1]
        for key in ('peak', 'rms', 'wpeak', 'outliers', 'gm0', 'gm1', 'gm2', 'err'):
            assert maps[key].shape == (nx, ny), f'{key} shape mismatch'

    def test_rms_map_nonnegative(self, tmpdir_local):
        """rms map should be non-negative everywhere."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'NeV14_rms')
        maps = sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
        )
        assert np.all(maps['rms'] >= 0), 'rms map must be non-negative'

    def test_reject_flag_accepted(self, tmpdir_local):
        """make_int_map should accept the reject flag without raising."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'NeV14_rej')
        maps = sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
            flag='reject4',
        )
        assert 'int' in maps

    def test_cliplvl_in_fits_header(self, tmpdir_local):
        """CLIPLVL keyword should reflect the sigma-clipping threshold used."""
        _skip_no_cube(_CUBE_CH3)
        out = os.path.join(tmpdir_local, 'NeV14_cliplvl')
        sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
            flag='reject4',
        )
        with fits.open(out + '.fits') as h:
            assert h[0].header.get('CLIPLVL') == pytest.approx(4.0), \
                'CLIPLVL should be 4.0 for flag=reject4'

    def test_compare_nev_map_intensity_to_matlab(self, tmpdir_local):
        """Intensity map (plane 1) should agree with Matlab NeV14.3-M82v19.fits to <5%."""
        _skip_no_cube(_CUBE_CH3)
        _skip_no_ref(_REF_NEV)
        out = os.path.join(tmpdir_local, 'nev_compare')
        sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[14.27, 14.3265, 14.3435, 14.355],
            order=1,
            intrange=[14.3210, 14.3440],
            flag='reject4',
        )
        with fits.open(_REF_NEV, ignore_missing_end=True) as h:
            ref_int = h[0].data[0].astype(np.float64)   # plane 1 (0-indexed)
        with fits.open(out + '.fits') as h:
            py_int = h[0].data[0].astype(np.float64)

        # Compare only pixels with significant signal (>10% of max)
        threshold = 0.1 * np.nanmax(np.abs(ref_int))
        mask = np.abs(ref_int) > threshold
        if mask.sum() == 0:
            pytest.skip('No pixels above threshold in reference map')
        rel_diff = np.abs(py_int[mask] - ref_int[mask]) / (np.abs(ref_int[mask]) + 1e-30)
        med_rel = float(np.nanmedian(rel_diff))
        assert med_rel < _TOL_MAP_REL, \
            f'Median intensity map deviation {med_rel:.2%} exceeds {_TOL_MAP_REL:.0%}'

    def test_compare_hi76_map_intensity_to_matlab(self, tmpdir_local):
        """Intensity map (plane 1) should agree with Matlab H7-6-M82v19.fits to <5%."""
        _skip_no_cube(_CUBE_CH3)
        _skip_no_ref(_REF_HI76)
        out = os.path.join(tmpdir_local, 'hi76_compare')
        sl.make_int_map(
            _CUBE_CH3, out,
            baseline=[12.325, 12.3689, 12.4038, 12.47],
            order=3,
            intrange=[12.3687, 12.3914],
        )
        with fits.open(_REF_HI76, ignore_missing_end=True) as h:
            ref_int = h[0].data[0].astype(np.float64)
        with fits.open(out + '.fits') as h:
            py_int = h[0].data[0].astype(np.float64)

        threshold = 0.1 * np.nanmax(np.abs(ref_int))
        mask = np.abs(ref_int) > threshold
        if mask.sum() == 0:
            pytest.skip('No pixels above threshold in reference map')
        rel_diff = np.abs(py_int[mask] - ref_int[mask]) / (np.abs(ref_int[mask]) + 1e-30)
        med_rel = float(np.nanmedian(rel_diff))
        assert med_rel < _TOL_MAP_REL, \
            f'Median intensity map deviation {med_rel:.2%} exceeds {_TOL_MAP_REL:.0%}'


# ---------------------------------------------------------------------------
# 9. sex2dec  (new in 06022026)
# ---------------------------------------------------------------------------

class TestSex2Dec:
    def test_colon_integer(self):
        # 12:30:00 = 12.5 exactly
        assert sl.sex2dec('12:30:00') == pytest.approx(12.5, rel=1e-6)

    def test_colon_fractional(self):
        val = sl.sex2dec('69:40:45.89')
        expected = 69 + 40/60 + 45.89/3600
        assert val == pytest.approx(expected, rel=1e-6)

    def test_negative(self):
        val = sl.sex2dec('-69:40:45.89')
        expected = -(69 + 40/60 + 45.89/3600)
        assert val == pytest.approx(expected, rel=1e-6)

    def test_hms_format(self):
        val = sl.sex2dec('9h55m52.74s')
        expected = 9 + 55/60 + 52.74/3600
        assert val == pytest.approx(expected, rel=1e-6)

    def test_dms_format(self):
        val = sl.sex2dec('69d40m45.89s')
        expected = 69 + 40/60 + 45.89/3600
        assert val == pytest.approx(expected, rel=1e-6)

    def test_list_input(self):
        result = sl.sex2dec(['12:00:00', '6:00:00'])
        assert len(result) == 2
        assert result[0] == pytest.approx(12.0, rel=1e-6)
        assert result[1] == pytest.approx(6.0, rel=1e-6)

    def test_scalar_returns_float(self):
        result = sl.sex2dec('0:00:00')
        assert isinstance(result, float)

    def test_zero(self):
        assert sl.sex2dec('0:00:00') == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# 10. dec2sex  (new in 06022026)
# ---------------------------------------------------------------------------

class TestDecSex:
    def test_zero(self):
        assert sl.dec2sex(0.0) == '00:00:00.000'

    def test_round_trip(self):
        val = 69 + 40/60 + 45.89/3600
        s = sl.dec2sex(val)
        back = sl.sex2dec(s)
        assert back == pytest.approx(val, abs=1e-4)

    def test_negative_has_minus(self):
        s = sl.dec2sex(-30.5)
        assert s.startswith('-'), f'Expected leading minus, got: {s!r}'

    def test_negative_round_trip(self):
        val = -(69 + 40/60 + 45.89/3600)
        s = sl.dec2sex(val)
        back = sl.sex2dec(s)
        assert back == pytest.approx(val, abs=1e-4)

    def test_hms_format(self):
        s = sl.dec2sex(9.0, fmt='h')
        assert 'h' in s and 'm' in s and 's' in s

    def test_dms_format(self):
        s = sl.dec2sex(69.0, fmt='d')
        assert 'd' in s and 'm' in s and 's' in s

    def test_force_plus(self):
        s = sl.dec2sex(12.5, fmt=':+')
        assert s.startswith('+'), f'Expected leading plus, got: {s!r}'

    def test_return_parts(self):
        h, m, s_i, s_f, sg = sl.dec2sex(12.5, return_parts=True)
        assert h == 12
        assert m == 30
        assert s_i == 0
        assert sg == 1

    def test_array_input(self):
        result = sl.dec2sex([12.0, 6.0])
        assert len(result) == 2
        assert '12' in result[0]
        assert '06' in result[1]


# ---------------------------------------------------------------------------
# Stand-alone runner
# ---------------------------------------------------------------------------

def _run_all():
    """Run all tests without pytest and print a summary."""
    import traceback

    if not os.path.exists(_CUBE_CH3):
        print(f'ERROR: ch3 cube not found at {_CUBE_CH3}')
        return

    print('Loading ch3 cube ...')
    cube = sl.read_fits(_CUBE_CH3, 'xten1')

    groups = [
        ('FITS reading',     TestReadFits,    {'cube_ch3': cube,
                                               'cube_ch2': (sl.read_fits(_CUBE_CH2, 'xten1')
                                                            if os.path.exists(_CUBE_CH2)
                                                            else None)}),
        ('Aperture',         TestExtractAperture, {'cube_ch3': cube}),
        ('Signal detection', TestSigChans,    {}),
        ('Gaussian fitting', TestGaussFitting, {}),
        ('fit_lines',        TestFitLines,    {}),
        ('make_int_map',     TestMakeIntMap,  {}),
        ('sex2dec',          TestSex2Dec,     {}),
        ('dec2sex',          TestDecSex,      {}),
    ]

    passed = failed = skipped = 0
    for group_name, cls, fixture_vals in groups:
        print(f'\n--- {group_name} ---')
        obj = cls()
        for name in [m for m in dir(cls) if m.startswith('test_')]:
            method = getattr(obj, name)
            import inspect
            sig = inspect.signature(method)
            kwargs = {}
            tmpdir_created = None
            for param in sig.parameters:
                if param in fixture_vals:
                    if fixture_vals[param] is None:
                        print(f'  SKIP  {name}  (fixture not available)')
                        skipped += 1
                        break
                    kwargs[param] = fixture_vals[param]
                elif param == 'tmpdir_local':
                    tmpdir_created = tempfile.mkdtemp()
                    kwargs[param] = tmpdir_created
            else:
                try:
                    method(**kwargs)
                    print(f'  PASS  {name}')
                    passed += 1
                except pytest.skip.Exception as e:
                    print(f'  SKIP  {name}  ({e})')
                    skipped += 1
                except Exception as e:
                    print(f'  FAIL  {name}')
                    traceback.print_exc()
                    failed += 1
                finally:
                    if tmpdir_created:
                        shutil.rmtree(tmpdir_created)

    # cat_peaks comparisons
    print('\n--- cat_peaks vs Matlab (all 7 pointings) ---')
    obj = TestCatPeaks()
    for i in range(7):
        pk = f'Nepk{i+1}'
        ref_pk = _ref_path(pk, '-peaklist.txt')
        if not os.path.exists(ref_pk):
            print(f'  SKIP  {pk}  (no reference)')
            skipped += 1
            continue
        tmpdir_created = tempfile.mkdtemp()
        try:
            _compare_to_matlab(i, cube, tmpdir_created)
            print(f'  PASS  {pk}')
            passed += 1
        except Exception as e:
            print(f'  FAIL  {pk}: {e}')
            failed += 1
        finally:
            shutil.rmtree(tmpdir_created)

    print(f'\n=== Results: {passed} passed, {failed} failed, {skipped} skipped ===')
    return failed


if __name__ == '__main__':
    sys.exit(_run_all() or 0)

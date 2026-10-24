# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale (University of Arizona),
# Gerhard Heinzel (Albert Einstein Institute).

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import signal

from speckit import SpectrumAnalyzer, compute_spectrum
from speckit.spectral import integral_rms

# --------- Utilities / Fixtures ---------------------------------------------


@pytest.fixture
def rand_seed():
    np.random.seed(1234)


def _white_noise(N, sigma=1.0):
    return sigma * np.random.randn(N)


def _two_channel_linear_system(N=200000, fs=100.0, tau=0.5, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(N)
    t = np.arange(N) / fs
    # First-order lowpass H(s) = 1/(tau s + 1)
    b = [1.0]
    a = [tau, 1.0]
    tout, y, _ = signal.lsim(signal.TransferFunction(b, a), U=x, T=t)
    if noise:
        y = y + noise * rng.standard_normal(N)
    return fs, x, y


# --------- 1) NumPy vs Numba parity (functional) ----------------------------


_NP_KERNELS = (
    "_stats_win_only_auto_np",
    "_stats_win_only_csd_np",
    "_stats_detrend0_auto_np",
    "_stats_detrend0_csd_np",
    "_stats_poly_auto_np",
    "_stats_poly_csd_np",
)


def _force_numpy_fallback(monkeypatch):
    """Select the `_np` kernels through `speckit.core` and count their calls."""
    from speckit import core

    calls = {"n": 0}
    for name in _NP_KERNELS:
        fn = getattr(core, name)

        def spy(*args, _fn=fn, **kwargs):
            calls["n"] += 1
            return _fn(*args, **kwargs)

        monkeypatch.setattr(core, name, spy)
    monkeypatch.setattr(core, "_NUMBA_ENABLED", False, raising=True)
    return calls


@pytest.mark.parametrize("order", [-1, 0, 1, 2])
def test_numpy_vs_numba_parity(order, rand_seed, monkeypatch):
    """Ensure the NumPy fallback and Numba path produce close spectra."""
    N = 50000
    fs = 200.0
    x = _white_noise(N)

    # Run with Numba enabled first (default)
    res_nb = compute_spectrum(x, fs=fs, order=order, Jdes=200, Kdes=40, win="hann")

    calls = _force_numpy_fallback(monkeypatch)
    res_np = compute_spectrum(x, fs=fs, order=order, Jdes=200, Kdes=40, win="hann")
    assert calls["n"] == len(res_np.f)

    # Same frequency grid and close values
    assert_allclose(res_nb.f, res_np.f, rtol=0, atol=0)
    assert_allclose(res_nb.Gxx, res_np.Gxx, rtol=5e-3, atol=5e-12)

    # If cross not computed, csd is None
    assert res_nb.csd is None and res_np.csd is None


@pytest.mark.parametrize("order", [-1, 0, 1, 2])
def test_numpy_vs_numba_cross_parity(order, monkeypatch):
    """
    The `_np` fallbacks return the Goertzel cross-spectrum, phase included.

    A conjugated fallback flips Gxy.imag, an O(|Gxy|) error. The two kernel
    families agree to ~1e-9 of sqrt(Gxx * Gyy) (Goertzel roundoff), so 1e-6 of
    that scale separates them cleanly.
    """
    rng = np.random.default_rng(0)
    N, fs = 30_000, 200.0
    x = rng.standard_normal(N)
    y = np.roll(x, 5) + 0.5 * rng.standard_normal(N)
    data = np.vstack([x, y])
    kw = {
        "win": "hann",
        "olap": 0.5,
        "Jdes": 40,
        "Kdes": 20,
        "order": order,
        "scheduler": "ltf",
    }

    res_nb = compute_spectrum(data, fs=fs, **kw)
    calls = _force_numpy_fallback(monkeypatch)
    res_np = compute_spectrum(data, fs=fs, **kw)
    assert calls["n"] == len(res_np.f)

    assert_allclose(res_nb.f, res_np.f, rtol=0, atol=0)
    assert_allclose(res_nb.Gxx, res_np.Gxx, rtol=5e-3, atol=5e-12)
    assert_allclose(res_nb.Gyy, res_np.Gyy, rtol=5e-3, atol=5e-12)
    scale = np.sqrt(res_nb.Gxx * res_nb.Gyy)
    assert np.all(np.abs(res_np.Gxy.real - res_nb.Gxy.real) <= 1e-6 * scale)
    assert np.all(np.abs(res_np.Gxy.imag - res_nb.Gxy.imag) <= 1e-6 * scale)

    # 5-sample delay: Hxy phase is -2 pi f 5 / fs on the NumPy path too.
    mask = res_np.coh > 0.5
    expected = np.exp(-2j * np.pi * res_np.f * 5 / fs)
    dphi = np.angle(res_np.Hxy[mask] / expected[mask])
    assert np.median(np.abs(dphi)) < np.deg2rad(5.0)

    # compute_single_bin dispatches through the same flag as compute().
    i = len(res_np.f) // 2
    ana = SpectrumAnalyzer(data, fs, **kw)
    one = ana.compute_single_bin(float(res_np.f[i]), L=int(res_np.L[i]))
    assert calls["n"] == len(res_np.f) + 1
    assert_allclose(one.Gxy[0], res_np.Gxy[i], rtol=1e-12)


@pytest.mark.parametrize("order", [-1, 0, 1])
def test_numpy_csd_kernel_sign_on_delay(order):
    """Kernel-level: `_np` CSD equals the Goertzel CSD for a 3-sample delay."""
    from speckit import core

    rng = np.random.default_rng(0)
    N, L, fs = 8000, 512, 200.0
    x = rng.standard_normal(N)
    y = np.roll(x, 3)
    w = np.hanning(L)
    starts = np.arange(0, N - L + 1, L // 2, dtype=np.int64)
    omega = 2 * np.pi * 40.0 / fs
    Q = core._build_Q(L, order) if order == 1 else None
    nb_fn, np_fn = {
        -1: (core._stats_win_only_csd, core._stats_win_only_csd_np),
        0: (core._stats_detrend0_csd, core._stats_detrend0_csd_np),
        1: (core._stats_poly_csd, core._stats_poly_csd_np),
    }[order]
    args = (x, y, starts, L, w, omega) + ((Q,) if Q is not None else ())
    nb = nb_fn(*args)
    npk = np_fn(*args)
    Znb = nb[2] + 1j * nb[3]
    Znp = npk[2] + 1j * npk[3]
    assert abs(Znp - Znb) <= 1e-12 * abs(Znb)
    assert abs(np.angle(Znb)) > np.deg2rad(10.0)  # the check is phase-sensitive


# --------- 2) CSD sign/phase matches SciPy ----------------------------------


def test_csd_phase_matches_scipy(rand_seed):
    fs, x, y = _two_channel_linear_system(N=200000, fs=200.0, tau=0.7, seed=42)
    # SciPy reference (Welch/CSD style)
    f_ref, Pxy = signal.csd(x, y, fs=fs, nperseg=4096, noverlap=2048, window="hann")
    f_ref, Pxx = signal.csd(x, x, fs=fs, nperseg=4096, noverlap=2048, window="hann")
    H_ref = Pxy / Pxx  # should follow X * conj(Y) convention

    # Our estimator
    res = compute_spectrum(
        np.vstack([x, y]), fs=fs, win="hann", Jdes=250, Kdes=50, order=0
    )

    # Compare phase where coherence is decent (to avoid noise-dominated bins)
    mask = res.coh > 0.2
    # Interpolate SciPy onto our grid for fair comparison
    H_ref_phase = np.interp(res.f, f_ref, np.angle(H_ref, deg=True))
    dphi = np.angle(res.Hxy, deg=True) - H_ref_phase
    # Allow small bias; unwrap reduces ±360 issues
    dphi = np.unwrap(np.deg2rad(dphi))
    assert np.median(np.abs(dphi[mask])) < np.deg2rad(5.0)  # < 5 degrees median error


# --------- 2b) Conditioned spectra ------------------------------------------


@pytest.mark.parametrize("delay", [7, 0])
def test_gyysx_is_coherent_residual(delay):
    """
    GyySx and GyyRx are both Gyy - |Gxy|^2 / Gxx.

    The identity Gyy (1 - coh) = Gyy - |Gxy|^2 / Gxx is exact, so the
    tolerance is float error. The 7-sample delay gives bins with
    |phase(Gxy)| > 60 deg, where the old expansion was off by O(10);
    delay 0 is the real-Gxy control.
    """
    rng = np.random.default_rng(1)
    N, fs = 80_000, 200.0
    x = rng.standard_normal(N)
    y = 0.8 * np.roll(x, delay) + 0.3 * rng.standard_normal(N)
    res = compute_spectrum(
        np.vstack([x, y]),
        fs=fs,
        win="hann",
        olap=0.5,
        Jdes=100,
        Kdes=30,
        order=-1,
        scheduler="ltf",
    )
    closed = res.Gyy - np.abs(res.Gxy) ** 2 / res.Gxx
    assert np.all(res.GyySx >= 0.0) and np.all(res.GyyRx >= 0.0)
    assert_allclose(res.GyySx, res.GyyRx, rtol=0, atol=0)
    assert np.all(np.abs(res.GyySx - closed) <= 1e-12 * res.Gyy)

    mid = (res.coh > 0.2) & (res.coh < 0.9) & (res.f > 1)
    assert mid.sum() > 10
    assert np.median(np.abs(res.GyySx[mid] / closed[mid] - 1)) < 1e-12
    if delay:
        hard = (np.abs(np.angle(res.Gxy)) > np.deg2rad(60)) & (res.coh > 0.5)
        assert hard.sum() > 5
        assert_allclose(res.GyySx[hard], closed[hard], rtol=1e-12)


# --------- 3) compute_single_bin vs pipeline consistency --------------------


def test_single_bin_matches_multi(rand_seed):
    fs = 500.0
    x = _white_noise(200000)
    ana = SpectrumAnalyzer(x, fs, win="hann", Jdes=64, Kdes=64, order=0)
    plan = ana.plan()

    # pick a middle bin
    idx = plan["nf"] // 2
    freq = float(plan["f"][idx])
    res_full = ana.compute()
    # compute just that bin via single-bin path
    res_one = ana.compute_single_bin(freq, L=int(plan["L"][idx]))

    # Compare XX, S2-scaled Gxx
    assert_allclose(res_full.XX[idx], res_one.XX[0], rtol=1e-10, atol=1e-12)
    assert_allclose(res_full.S2[idx], res_one.S2[0], rtol=1e-10, atol=1e-12)
    assert_allclose(res_full.Gxx[idx], res_one.Gxx[0], rtol=1e-10, atol=1e-12)


# --------- 4) Band filtering in plan() --------------------------------------


def test_band_filtering_in_plan(rand_seed):
    fs = 1000.0
    x = _white_noise(200000)
    fmin, fmax = 10.0, 50.0
    res_full = compute_spectrum(x, fs=fs, win="hann", Jdes=300, Kdes=40)
    res_band = compute_spectrum(
        x, fs=fs, win="hann", Jdes=300, Kdes=40, band=(fmin, fmax)
    )

    assert res_band.f.min() >= fmin - 1e-9
    assert res_band.f.max() <= fmax + 1e-9
    assert len(res_band.f) < len(res_full.f)
    # ensure arrays align lengths
    for key in ("Gxx", "ENBW"):
        assert len(getattr(res_band, key)) == len(res_band.f)


# --------- 5) Polynomial detrend effectiveness ------------------------------


@pytest.mark.parametrize("order", [-1, 0, 1, 2])
def test_polynomial_detrend_reduces_low_freq(rand_seed, order):
    """
    Build a signal with quadratic trend + white noise and check that
    low-frequency ASD is progressively reduced from -1/0 to 1/2.
    """
    N, fs = 200000, 200.0
    t = np.arange(N) / fs
    trend = 0.5 + 0.2 * t + 0.8 * t**2
    x = trend + 0.5 * np.random.randn(N)

    res = compute_spectrum(x, fs=fs, win="hann", Jdes=256, Kdes=64, order=order)
    # average the lowest ~5% of bins as a proxy for low frequency power
    k = max(3, int(0.05 * len(res.f)))
    low_pow = np.mean(res.asd[:k]) if res.asd is not None else np.nan
    assert np.isfinite(low_pow)

    # record result for ordering check in a separate test via caching on function attribute
    if not hasattr(test_polynomial_detrend_reduces_low_freq, "_vals"):
        test_polynomial_detrend_reduces_low_freq._vals = {}
    test_polynomial_detrend_reduces_low_freq._vals[order] = low_pow


def test_polynomial_detrend_ordering():
    vals = getattr(test_polynomial_detrend_reduces_low_freq, "_vals", None)
    # Only assert if previous test executed in same session
    if vals is None or any(k not in vals for k in (-1, 0, 1, 2)):
        pytest.skip("Ordering cache not populated.")
    # Expect: order 1 < order 0 ≤ order -1, and order 2 ≤ order 1 (roughly)
    assert vals[1] <= 1.10 * vals[0]
    assert vals[0] <= 1.10 * vals[-1]
    assert vals[2] <= 1.10 * vals[1]


# --------- 6) Empirical stats exposure & sanity -----------------------------


def test_empirical_stats_exposed_and_finite(rand_seed):
    fs = 200.0
    x = _white_noise(100000)
    y = signal.lfilter([1.0], [0.1, 1.0], x)  # simple low-pass
    res = compute_spectrum([x, y], fs=fs, win="hann", Jdes=128, Kdes=30, order=0)

    for attr in ("XX_mean", "YY_mean", "XY_M2", "XY_emp_var", "XY_emp_dev"):
        v = getattr(res, attr)
        assert isinstance(v, np.ndarray)
        assert np.all(np.isfinite(v))

    # Cross empirical dev should correlate with parametric error magnitude
    # (loose check; just ensure finite and not exploding)
    assert np.all(res.Gxy_emp_dev[np.isfinite(res.Gxy_emp_dev)] >= 0.0)


# --------- 7) Input validation & plan caching --------------------------------


def test_invalid_input_shapes():
    fs = 100.0
    with pytest.raises(ValueError):
        SpectrumAnalyzer(np.zeros((3, 3)), fs=fs)  # 3x3 ambiguous


def test_plan_caching(rand_seed):
    fs = 100.0
    x = _white_noise(20000)
    ana = SpectrumAnalyzer(x, fs, win="hann", Jdes=64, Kdes=16)
    p1 = ana.plan()
    p2 = ana.plan()
    # same object back -> cached
    assert p1 is p2


# --------- 8) Interpolation & DataFrame smoke tests --------------------------


def test_get_measurement_and_to_dataframe(rand_seed):
    fs = 200.0
    x = _white_noise(40000)
    res = compute_spectrum(x, fs=fs, win="hann", Jdes=128, Kdes=32)
    # interpolation
    f_query = np.array([res.f[10] * 1.01, res.f[20] * 0.99])
    y_interp = res.get_measurement(f_query, "asd")
    assert y_interp.shape == (2,)
    # dataframe export
    df = res.to_dataframe()
    assert "psd" in df.columns or "Gxx" in df.columns
    assert df.index.is_monotonic_increasing
    assert res.get_rms() == pytest.approx(integral_rms(res.f, res.asd))


def test_cf_db_is_twenty_log10(rand_seed):
    fs, x, y = _two_channel_linear_system(N=20000, fs=100.0, tau=0.2, seed=3)
    res = compute_spectrum(np.vstack([x, y]), fs=fs, win="hann", Jdes=64, Kdes=16)
    assert np.all(res.Gxx > 0)
    expected = 20.0 * np.log10(np.abs(res.Hxy))
    assert_allclose(res.cf_db, expected)


def test_import_speckit_skips_optional_modules():
    script = """
import sys
import speckit
import speckit.dsp
import speckit.systems
from speckit import compute_spectrum
banned = ("pandas", "control", "py7zr", "sympy", "tqdm")
loaded = [name for name in banned if name in sys.modules]
if loaded:
    raise SystemExit("imported " + ", ".join(loaded))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


# --------- 9) ENBW sanity ----------------------------------------------------


def test_enbw_positive_and_reasonable(rand_seed):
    fs = 1000.0
    x = _white_noise(200000)
    for win in ("hann", "kaiser"):
        res = compute_spectrum(x, fs=fs, win=win, Jdes=128, Kdes=64)
        assert np.all(res.ENBW > 0)
        # ENBW ~ fs * S2 / S1^2; check it’s not absurd (> Nyquist by orders)
        assert np.max(res.ENBW) < fs * 10.0

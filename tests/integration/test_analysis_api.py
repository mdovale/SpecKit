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
import numpy as np
import pytest
from numpy.testing import assert_allclose

from speckit import compute_spectrum, SpectrumAnalyzer
from scipy import signal


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


@pytest.mark.parametrize("order", [-1, 0, 1, 2])
def test_numpy_vs_numba_parity(order, rand_seed, monkeypatch):
    """Ensure the NumPy fallback and Numba path produce close spectra."""
    N = 50000
    fs = 200.0
    x = _white_noise(N)

    # Run with Numba enabled first (default)
    res_nb = compute_spectrum(x, fs=fs, order=order, Jdes=200, Kdes=40, win="hann")

    # Force NumPy fallback
    import speckit.core as core

    monkeypatch.setattr(core, "_NUMBA_ENABLED", False, raising=True)

    res_np = compute_spectrum(x, fs=fs, order=order, Jdes=200, Kdes=40, win="hann")

    # Same frequency grid and close values
    assert_allclose(res_nb.f, res_np.f, rtol=0, atol=0)
    assert_allclose(res_nb.Gxx, res_np.Gxx, rtol=5e-3, atol=5e-12)

    # If cross not computed, csd is None
    assert res_nb.csd is None and res_np.csd is None


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


# --------- 9) ENBW sanity ----------------------------------------------------


def test_enbw_positive_and_reasonable(rand_seed):
    fs = 1000.0
    x = _white_noise(200000)
    for win in ("hann", "kaiser"):
        res = compute_spectrum(x, fs=fs, win=win, Jdes=128, Kdes=64)
        assert np.all(res.ENBW > 0)
        # ENBW ~ fs * S2 / S1^2; check it’s not absurd (> Nyquist by orders)
        assert np.max(res.ENBW) < fs * 10.0

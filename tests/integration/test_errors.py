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
from tqdm import trange

from speckit import SpectrumAnalyzer, compute_spectrum


def test_error_formulas_against_monte_carlo():
    """
    Validates the analytical error formulas for PSD and coherence by comparing
    them to the standard deviation measured from a Monte Carlo simulation.
    """
    # --- Simulation Configuration ---
    num_realizations = 200
    fs = 100.0
    n_samples = 20000
    target_freq = 20.0
    freq_res = target_freq * 0.5

    rng = np.random.default_rng(seed=123)

    # --- Storage for estimates from each realization ---
    estimates = {"Gxx": [], "coh": [], "Hxy_mag": [], "Gxy_mag": []}
    # --- Storage for analytical predictions from the first realization ---
    analytical_predictions = {}

    for i in trange(num_realizations, desc="Running Monte Carlo Error Validation"):
        # --- 1. Generate new signal realizations ---
        t = np.arange(n_samples) / fs
        common_signal = 0.8 * np.sin(2 * np.pi * target_freq * t)
        noise1 = rng.normal(loc=0, scale=1.0, size=n_samples)
        noise2 = rng.normal(loc=0, scale=1.0, size=n_samples)
        x = common_signal + noise1
        y = common_signal + noise2

        # --- 2. Compute spectral estimates for this realization ---
        # Use the SpectrumAnalyzer directly for single-bin computation
        analyzer = SpectrumAnalyzer(data=[x, y], fs=fs, win="hann", olap=0.5)
        result = analyzer.compute_single_bin(freq=target_freq, fres=freq_res)

        # --- 3. Store results from this realization ---
        estimates["Gxx"].append(result.Gxx[0])
        estimates["coh"].append(result.coh[0])
        estimates["Hxy_mag"].append(np.abs(result.Hxy[0]))
        estimates["Gxy_mag"].append(np.abs(result.Gxy[0]))

        if i == 0:
            # Store the analytical prediction from the first run
            analytical_predictions["Gxx_dev"] = result.Gxx_dev[0]
            analytical_predictions["coh_dev"] = result.coh_dev[0]
            analytical_predictions["Hxy_dev"] = result.Hxy_dev[0]
            analytical_predictions["Gxy_dev"] = result.Gxy_dev[0]
            analytical_predictions["coh_error"] = result.coh_error[0]
            analytical_predictions["Hxy_mag_error"] = result.Hxy_mag_error[0]

    # --- 4. Analyze the distribution of estimates ---
    measured_devs = {key: np.std(val) for key, val in estimates.items()}
    measured_errors = {
        key: np.std(val) / np.mean(val) for key, val in estimates.items()
    }

    # --- 5. Assert that analytical and measured deviations are close ---
    # Standard Deviations (_dev)
    assert measured_devs["Gxx"] == pytest.approx(
        analytical_predictions["Gxx_dev"], rel=0.2
    )
    assert measured_devs["coh"] == pytest.approx(
        analytical_predictions["coh_dev"], rel=0.2
    )
    assert measured_devs["Hxy_mag"] == pytest.approx(
        analytical_predictions["Hxy_dev"], rel=0.2
    )
    assert measured_devs["Gxy_mag"] == pytest.approx(
        analytical_predictions["Gxy_dev"], rel=0.5
    )
    # Normalized Random Errors (_error)
    assert measured_errors["coh"] == pytest.approx(
        analytical_predictions["coh_error"], rel=0.2
    )
    assert measured_errors["Hxy_mag"] == pytest.approx(
        analytical_predictions["Hxy_mag_error"], rel=0.2
    )


def _delay_spectrum(olap):
    rng = np.random.default_rng(1)
    N, fs = 80_000, 200.0
    x = rng.standard_normal(N)
    y = 0.8 * np.roll(x, 7) + 0.3 * rng.standard_normal(N)
    return compute_spectrum(
        np.vstack([x, y]),
        fs=fs,
        win="hann",
        olap=olap,
        Jdes=100,
        Kdes=30,
        order=-1,
        scheduler="ltf",
    )


def test_phase_error_is_bendat_closed_form():
    """
    Hxy_rad_error = sqrt(1 - coh) / sqrt(2 coh n_d), Bendat & Piersol's phase
    standard deviation in radians, with n_d = nu / 2 (= navg at zero overlap).

    The closed form is checked to float tolerance. A coh ~ 0.2 bin separates
    it from the former arcsin(sqrt(1 - coh)) numerator by a factor ~1.24.
    """
    res = _delay_spectrum(olap=0.0)
    coh, navg = res.coh, res.navg.astype(float)
    sel = (coh > 0.2) & (coh < 0.99) & (navg > 1)
    assert sel.sum() > 10
    # Rounded starts can overlap by a sample or two; window tails make that
    # a ~1e-8 correction, not a change of regime.
    assert_allclose(res.nu[sel], 2 * navg[sel], rtol=1e-6)
    n_d = 0.5 * res.nu[sel]

    bendat = np.sqrt(1 - coh[sel]) / np.sqrt(coh[sel] * 2 * n_d)
    assert_allclose(res.Hxy_rad_error[sel], bendat, rtol=1e-12)
    assert_allclose(res.Hxy_deg_error[sel], np.rad2deg(bendat), rtol=1e-12)
    assert_allclose(res.Hxy_mag_error[sel], bendat, rtol=1e-12)

    old = np.arcsin(np.sqrt(1 - coh[sel])) / np.sqrt(coh[sel] * 2 * n_d)
    low = coh[sel] < 0.3
    assert low.any()
    assert np.all(old[low] / res.Hxy_rad_error[sel][low] > 1.15)

    # With overlap the same formula uses nu / 2.
    res = _delay_spectrum(olap=0.5)
    coh, n_d = res.coh, 0.5 * res.nu
    sel = (coh > 0.2) & (coh < 0.99) & (res.navg > 1)
    expected = np.sqrt(1 - coh[sel]) / np.sqrt(coh[sel] * 2 * n_d[sel])
    assert_allclose(res.Hxy_rad_error[sel], expected, rtol=1e-12)


@pytest.mark.parametrize("win", ["hann", "kaiser"])
def test_zero_overlap_errors_are_independent_segment_limit(win):
    """Without overlap nu == 2 navg and every error keeps its navg form."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(40_000)
    ana = SpectrumAnalyzer(x, 200.0, win=win, olap=0.0, order=-1)
    r = ana.compute_single_bin(20.0, L=512)
    D = r.D[0]
    assert np.all(np.diff(D) >= 512)
    navg = r.navg.astype(float)
    assert_allclose(r.nu, 2 * navg, rtol=0, atol=0)
    assert_allclose(r.Gxx_dev, r.Gxx / np.sqrt(navg), rtol=1e-15)
    assert_allclose(r.Gxx_error, 1 / np.sqrt(navg), rtol=1e-15)
    scale = 2.0 / (200.0 * r.S2)
    assert_allclose(r.Gxx_emp_dev, scale * np.sqrt(r.M2 / navg), rtol=1e-15)


def test_overlap_corrected_psd_error_matches_monte_carlo():
    """
    Hann at 75% overlap: Gxx_dev = Gxx sqrt(2/nu) predicts the PSD scatter.

    Setup: white unit-variance noise, fs = 200, N = 40000, 20 Hz,
    L = 512, order -1, 400 realizations, numpy Generator seed 4. Measured
    std uses ddof=1; the prediction is averaged over realizations because
    Gxx_dev scales with each realization's Gxx. The standard error of a
    std estimate at n = 400 is ~3.5%, so +/-0.12 is ~3.4 sigma.

    Percival & Walden give nu / (2K) ~ 0.52 for Hann at 75% overlap, so the
    independent-segment prediction is low by sqrt(1/0.52) ~ 1.39. The empirical
    M2-based error had the same bias and is checked the same way.
    """
    fs, N, freq, L = 200.0, 40_000, 20.0, 512
    rng = np.random.default_rng(4)
    g, dev, emp, old = [], [], [], []
    for _ in range(400):
        x = rng.standard_normal(N)
        ana = SpectrumAnalyzer(x, fs, win="hann", olap=0.75, order=-1)
        r = ana.compute_single_bin(freq, L=L)
        g.append(r.Gxx[0])
        dev.append(r.Gxx_dev[0])
        emp.append(r.Gxx_emp_dev[0])
        old.append(r.Gxx[0] / np.sqrt(r.navg[0]))
    std = np.std(g, ddof=1)
    nu_ratio = r.nu[0] / (2 * r.navg[0])
    assert 0.50 < nu_ratio < 0.54
    assert std / np.mean(dev) == pytest.approx(1.0, abs=0.12)
    assert std / np.mean(emp) == pytest.approx(1.0, abs=0.12)
    assert std / np.mean(old) > 1.25

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
from scipy.signal import welch
from scipy.stats import linregress
from speckit import noise


@pytest.fixture
def n_samples_large() -> int:
    """Provides a large number of samples (2^18) for robust statistical testing."""
    return 2**18


def test_white_noise_reproducibility():
    """
    Tests that two white_noise generators initialized with the same seed
    produce identical output sequences, ensuring reproducibility.
    """
    gen1 = noise.white_noise(f_sample=1000.0, psd=1.0, seed=123)
    gen2 = noise.white_noise(f_sample=1000.0, psd=1.0, seed=123)

    series1 = gen1.get_series(100)
    series2 = gen2.get_series(100)

    np.testing.assert_array_equal(series1, series2)


def test_white_noise_statistics(n_samples_large):
    """
    Tests that the generated white noise has the correct mean (approx. 0) and
    Root Mean Square (RMS) value, validating its statistical properties.
    """
    f_samp = 200000.0
    psd = 1e-10  # Example two-sided PSD in V^2 / Hz

    gen = noise.white_noise(f_sample=f_samp, psd=psd, seed=42)
    series = gen.get_series(n_samples_large)

    # The theoretical RMS of a real signal generated from a two-sided PSD
    # is sqrt(PSD * f_sample).
    expected_rms = np.sqrt(psd * f_samp)

    # Assert that the mean is close to zero.
    assert np.mean(series) == pytest.approx(0.0, abs=1e-3)
    # Assert that the measured RMS is close to the theoretical value.
    # A relative tolerance is used as this is a statistical quantity.
    assert np.std(series) == pytest.approx(expected_rms, rel=1e-2)


def test_alpha_noise_reproducibility():
    """
    Tests that the alpha_noise generator is reproducible with a given seed.
    """
    gen1 = noise.alpha_noise(
        f_sample=1000.0, f_min=10.0, f_max=400.0, alpha=1.0, seed=456, init_filter=False
    )
    gen2 = noise.alpha_noise(
        f_sample=1000.0, f_min=10.0, f_max=400.0, alpha=1.0, seed=456, init_filter=False
    )

    series1 = gen1.get_series(1000)
    series2 = gen2.get_series(1000)

    np.testing.assert_array_equal(series1, series2)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5, 2.0])
def test_alpha_noise_spectral_slope(alpha, n_samples_large):
    """
    Validates the core feature of alpha_noise: that the generated noise has
    the correct spectral slope in the frequency domain.

    This test generates a long noise series, computes its Power Spectral
    Density (PSD), and performs a linear regression on the log-log plot of
    the PSD to verify that its slope is approximately -alpha.

    Parameters
    ----------
    alpha : float
        The spectral exponent to test.
    n_samples_large : int
        The number of samples to generate, provided by the fixture.
    """
    f_samp = 10000.0
    f_min = 20.0
    f_max = 4000.0

    # Generate a long series for good frequency resolution and statistics
    gen = noise.alpha_noise(f_samp, f_min, f_max, alpha=alpha, seed=int(alpha * 100))
    series = gen.get_series(n_samples_large)

    # Calculate the Power Spectral Density using Welch's method
    freqs, psd = welch(series, fs=f_samp, nperseg=n_samples_large // 16)

    # We only fit the slope in the region where it should be 1/f^alpha,
    # avoiding the filter transition bands at the edges.
    fit_mask = (freqs > f_min * 2) & (freqs < f_max / 2) & (psd > 0)

    # Perform a linear regression on the log10-transformed data
    log_freqs = np.log10(freqs[fit_mask])
    log_psd = np.log10(psd[fit_mask])

    # The slope of the line in log-log space is the spectral index
    result = linregress(log_freqs, log_psd)

    # For a PSD proportional to 1/f^alpha, the log-log slope is -alpha
    expected_slope = -alpha

    # Assert that the measured slope is close to the expected slope.
    # A generous tolerance is used because this is a statistical test on a
    # random process.
    assert result.slope == pytest.approx(expected_slope, abs=0.15)


def test_red_noise_spectral_slope(n_samples_large):
    """
    Tests that the specialized red_noise generator produces a spectrum with
    a slope of approximately -2 (i.e., 1/f^2).
    """
    f_samp = 10000.0
    f_min = 20.0

    gen = noise.red_noise(f_samp, f_min, seed=222)
    series = gen.get_series(n_samples_large)

    freqs, psd = welch(series, fs=f_samp, nperseg=n_samples_large // 16)

    # Fit the slope above the corner frequency, in the roll-off region
    fit_mask = (freqs > f_min * 2) & (freqs < f_samp / 4) & (psd > 0)

    log_freqs = np.log10(freqs[fit_mask])
    log_psd = np.log10(psd[fit_mask])

    result = linregress(log_freqs, log_psd)

    # Red noise has a 1/f^2 spectrum, so the slope should be -2
    expected_slope = -2.0
    # The new implementation should be more accurate, so we can tighten the tolerance.
    assert result.slope == pytest.approx(expected_slope, abs=0.1)

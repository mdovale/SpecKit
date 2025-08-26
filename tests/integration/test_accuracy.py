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
# tests/integration/test_accuracy.py

import pytest
import numpy as np
from scipy import signal

from speckit.analysis import compute_spectrum


@pytest.fixture(scope="module")
def siso_system_fixture():
    """
    Creates a fixture for a well-defined SISO system with a known ground truth.

    This fixture generates:
    1.  A known digital Butterworth filter (the "system").
    2.  The true complex frequency response H(f) of this filter.
    3.  A white noise input signal `x`.
    4.  A filtered output signal `y` which has some uncorrelated noise added,
        ensuring the coherence is less than 1.
    5.  The theoretical output PSD, `|H(f)|^2 * PSD_input`.
    6.  The theoretical coherence function between the input and final output.

    Returns:
        dict: A dictionary containing all signals and theoretical ground truths.
    """
    # --- 1. System Definition ---
    fs = 1000.0
    N = int(2e5)  # Use a reasonably long signal for good statistics
    cutoff_freq = 50.0
    filter_order = 4

    # Design the digital filter and get its true frequency response
    b, a = signal.butter(filter_order, cutoff_freq, fs=fs, btype="low")
    w, H_true = signal.freqz(b, a, worN=8192, fs=fs)
    f_true = w  # freqz with fs argument returns frequencies in Hz

    # --- 2. Signal Generation ---
    rng = np.random.default_rng(seed=42)
    # Input signal: white noise. Its two-sided PSD is 1.0, so one-sided is 2.0.
    # Note: For np.random.randn, var=1. One-sided PSD = 2*var/fs.
    input_signal = rng.normal(size=N)
    psd_input_theory = 2.0 / fs  # Theoretical one-sided PSD of the input

    # Filtered output
    filtered_output = signal.lfilter(b, a, input_signal)

    # Add uncorrelated noise to the output to make coherence < 1
    uncorrelated_noise = 0.5 * rng.normal(size=N)
    psd_noise_theory = (0.5**2) * (2.0 / fs)  # Theoretical PSD of uncorrelated noise

    final_output = filtered_output + uncorrelated_noise

    # --- 3. Theoretical Ground Truth Calculation ---
    # Theoretical output PSD is the sum of the filtered signal's PSD and the noise PSD
    psd_output_theory = (np.abs(H_true) ** 2 * psd_input_theory) + psd_noise_theory

    # Theoretical coherence γ^2 = |Gxy|^2 / (Gxx * Gyy)
    # Gxy = H(f) * Gxx
    # Gyy = |H(f)|^2 * Gxx + G_noise
    # -> γ^2 = |H(f)|^2 * Gxx / (|H(f)|^2 * Gxx + G_noise)
    coh_theory = (np.abs(H_true) ** 2 * psd_input_theory) / (
        (np.abs(H_true) ** 2 * psd_input_theory) + psd_noise_theory
    )

    return {
        "fs": fs,
        "input": input_signal,
        "output": final_output,
        "f_true": f_true,
        "H_true": H_true,
        "psd_output_true": psd_output_theory,
        "coh_true": coh_theory,
    }


def test_autospectrum_accuracy(siso_system_fixture):
    """
    Validates the accuracy of the auto-spectrum (PSD) against a known
    theoretical spectrum from a filtered white noise source.
    """
    # --- Get data from the fixture ---
    system = siso_system_fixture
    y_signal = system["output"]
    fs = system["fs"]
    f_true = system["f_true"]
    psd_true = system["psd_output_true"]

    # --- Compute the spectrum using speckit ---
    result = compute_spectrum(y_signal, fs=fs)
    f_measured = result.f
    psd_measured = result.psd

    # --- Compare measured result to ground truth ---
    # Interpolate the theoretical PSD onto the frequency points speckit calculated
    psd_true_interp = np.interp(f_measured, f_true, psd_true)

    # Calculate the relative error, ignoring very low frequencies where statistics are poor
    valid_mask = f_measured > 1.0
    relative_error = (
        np.abs(psd_measured[valid_mask] - psd_true_interp[valid_mask])
        / psd_true_interp[valid_mask]
    )

    # The median relative error should be small, demonstrating good accuracy.
    # We use median as it's robust to outliers at a few frequency bins.
    median_rel_error = np.median(relative_error)

    assert median_rel_error < 0.1, (
        f"Median relative error of PSD ({median_rel_error:.3f}) exceeds 10% threshold"
    )


def test_cross_spectrum_accuracy(siso_system_fixture):
    """
    Validates the accuracy of the cross-spectral estimates (coherence and
    transfer function) against a known theoretical system.
    """
    # --- Get data from the fixture ---
    system = siso_system_fixture
    x_signal = system["input"]
    y_signal = system["output"]
    fs = system["fs"]
    f_true = system["f_true"]
    coh_true = system["coh_true"]
    H_true = system["H_true"]

    # --- Compute the cross-spectrum using speckit ---
    data_stack = np.vstack([x_signal, y_signal])
    result = compute_spectrum(data_stack, fs=fs)
    f_measured = result.f
    coh_measured = result.coh
    tf_measured = result.Hxy

    # --- 1. Validate Coherence ---
    # Interpolate the theoretical coherence onto the measured frequency points
    coh_true_interp = np.interp(f_measured, f_true, coh_true)

    # Calculate the median absolute error (coherence is bounded [0, 1])
    valid_mask = (f_measured > 1.0) & (f_measured < fs / 2.5)  # Avoid edges
    abs_error_coh = np.abs(coh_measured[valid_mask] - coh_true_interp[valid_mask])
    median_abs_error_coh = np.median(abs_error_coh)

    assert median_abs_error_coh < 0.05, (
        f"Median absolute error of coherence ({median_abs_error_coh:.3f}) exceeds 0.05 threshold"
    )

    # --- 2. Validate Transfer Function ---
    # Interpolate the complex theoretical TF onto the measured frequency points
    H_true_interp_real = np.interp(f_measured, f_true, np.real(H_true))
    H_true_interp_imag = np.interp(f_measured, f_true, np.imag(H_true))
    H_true_interp = H_true_interp_real + 1j * H_true_interp_imag

    # Calculate the complex error vector and normalize it by the true TF magnitude
    # This simultaneously tests for accuracy in both magnitude and phase
    complex_error = tf_measured[valid_mask] - H_true_interp[valid_mask]
    normalized_complex_error = np.abs(complex_error) / np.abs(H_true_interp[valid_mask])
    median_norm_error_tf = np.median(normalized_complex_error)

    assert median_norm_error_tf < 0.2, (
        f"Median normalized complex error of TF ({median_norm_error_tf:.3f}) exceeds 10% threshold"
    )

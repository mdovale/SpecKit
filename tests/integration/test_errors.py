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
import pytest
import numpy as np
from tqdm import trange

from speckit import SpectrumAnalyzer


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

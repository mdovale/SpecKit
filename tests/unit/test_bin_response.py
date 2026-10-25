# BSD 3-Clause License
#
# Copyright (c) 2025, Miguel Dovale (University of Arizona),
# Gerhard Heinzel (Albert Einstein Institute).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""One bin's direct-DFT response against the oracle and ENBW."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from speckit.response import BinResponse
from speckit.utils import kaiser_alpha

_ULPS = 8.0
_EPS = float(np.finfo(np.float64).eps)
_ORACLE_PATH = Path(__file__).resolve().parents[1] / "direct_dft_oracle.py"


def _oracle():
    spec = importlib.util.spec_from_file_location("direct_dft_oracle", _ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _windows(length: int):
    alpha = kaiser_alpha(200.0)
    kaiser = np.kaiser(length + 1, alpha * np.pi)[:-1]
    return {"hann": np.hanning(length), "kaiser": kaiser}


@pytest.mark.parametrize("order", [-1, 0, 1, 2])
@pytest.mark.parametrize("window_name", ["hann", "kaiser"])
def test_center_response_matches_the_direct_dft(order, window_name):
    """Allowance is 8 * L * eps * sum(|y_windowed|), the oracle's bound."""
    oracle = _oracle()
    length = 256
    fs = 256.0
    freq = 32.0
    window = _windows(length)[window_name]
    response = BinResponse(window, fs, freq, order=order)
    resolution = fs / length
    tones = (
        freq,
        freq + 0.5 * resolution,
        freq + 3.0 * resolution,
        -freq,
    )
    omega = 2.0 * np.pi * freq / fs
    n = np.arange(length, dtype=np.float64)
    for tone_freq in tones:
        phase = 2.0 * np.pi * tone_freq / fs * n
        # The oracle casts its segment to float64. The complex exponential
        # is the real operator applied to cos and to sin.
        ref_real, windowed_real = oracle.coefficient(
            np.cos(phase), window, omega, order
        )
        ref_imag, windowed_imag = oracle.coefficient(
            np.sin(phase), window, omega, order
        )
        reference = ref_real + 1j * ref_imag
        atol = (
            _ULPS
            * length
            * _EPS
            * (
                float(np.sum(np.abs(windowed_real)))
                + float(np.sum(np.abs(windowed_imag)))
            )
        )
        got = response.coefficient(tone_freq)
        assert abs(got - reference) <= atol


def test_order_minus_one_enbw_matches_s1_s2_and_the_dft_integral():
    length = 256
    fs = 256.0
    window = np.hanning(length)
    response = BinResponse(window, fs, freq=32.0, order=-1)
    formula = fs * response.S2 / response.S1**2
    assert response.window_enbw == pytest.approx(formula, rel=0.0, abs=0.0)
    assert response.enbw == pytest.approx(formula, rel=1e-12)
    assert response.enbw_numerical == pytest.approx(formula, rel=1e-12)
    assert response.coherent_gain == pytest.approx(response.S1)


def test_detrend_enbw_is_the_projected_tap_not_the_raw_window():
    window = np.hanning(128)
    raw = BinResponse(window, fs=128.0, freq=16.0, order=-1)
    detrended = BinResponse(window, fs=128.0, freq=16.0, order=1)
    assert detrended.window_enbw == pytest.approx(raw.window_enbw)
    assert detrended.enbw != pytest.approx(raw.enbw, rel=1e-6)
    assert detrended.enbw > 0.0

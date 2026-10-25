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
"""Seeded synthetic corpus for the one-sided WOSA estimator.

Generator ``psd`` values are two-sided. Reported ``Gxx`` is one-sided, so a
flat generator level ``psd`` has target ``2 * psd``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from speckit import compute_single_bin, compute_spectrum
from speckit.noise import pink_noise, white_noise
from speckit.response import BinResponse

_SEED = 20261025


def test_white_generator_one_sided_level_is_two_psd():
    """Unit-variance white noise. Hann, order=-1, no overlap.

    ``white_noise(..., psd=1/fs)`` has variance 1. The one-sided target is
    ``2 * psd = 2/fs``. One interior bin, K = 200, so ``nu = 400`` and
    ``Gxx_dev / Gxx`` is ``sqrt(2/nu)``. The realization must sit inside
    four of those standard deviations. That is one seeded draw, not a
    coverage rate.
    """
    fs = 256.0
    length = 256
    n = length * 200
    psd = 1.0 / fs
    target = 2.0 * psd
    series = white_noise(fs, psd=psd, seed=_SEED).get_series(n)
    result = compute_single_bin(
        series,
        fs,
        freq=32.0,
        L=length,
        order=-1,
        win="hann",
        olap=0.0,
    )
    gxx = float(result.Gxx[0])
    dev = float(result.Gxx_dev[0])
    assert abs(gxx - target) <= 4.0 * dev


def test_pink_one_sided_target_is_twice_the_two_sided_power_law():
    """Two-sided generator level ``f**(-1)``. One-sided target ``2 / f``.

    The band is a decade above ``f_min`` and a decade below ``f_max``,
    where the generator documents its transfer function within 1% of the
    power law. The median ratio here checks that factor of two. It is not
    a leakage calibration.
    """
    fs = 256.0
    n = 2**16
    f_min = 0.2
    f_max = 80.0
    series = pink_noise(f_sample=fs, f_min=f_min, f_max=f_max, seed=_SEED).get_series(n)
    result = compute_spectrum(
        series,
        fs,
        Jdes=48,
        Kdes=40,
        order=-1,
        win="hann",
        olap=0.5,
        bmin=1.0,
    )
    band = (result.f >= 2.0) & (result.f <= 8.0)
    ratio = result.Gxx[band] * result.f[band] / 2.0
    median = float(np.median(ratio))
    assert 0.75 < median < 1.25


def test_half_bin_tone_matches_the_predicted_coefficient():
    """On-bin and half-bin cosines. The neighbor level is the response,
    including the negative-frequency image. Rectangular window, one segment.
    """
    length = 64
    fs = 64.0
    freq = 8.0
    window = np.ones(length)
    response = BinResponse(window, fs, freq, order=-1)
    n = np.arange(length, dtype=np.float64)
    resolution = fs / length
    for tone_freq in (freq, freq + 0.5 * resolution):
        series = np.cos(2.0 * np.pi * tone_freq / fs * n)
        result = compute_single_bin(
            series,
            fs,
            freq=freq,
            L=length,
            order=-1,
            win=np.ones,
            olap=0.0,
        )
        predicted = response.cosine_coefficient(tone_freq)
        assert float(result.XX[0]) == pytest.approx(abs(predicted) ** 2, rel=1e-12)


def test_integer_delay_phase_follows_x_conj_y():
    """Rectangular window, integer bin. A delay of channel 1 multiplies
    ``Y`` by ``exp(-i ω delay)``. ``Gxy = X * conj(Y)`` then has phase
    ``+ω delay``. ``Hxy = conj(Gxy) / Gxx`` has phase ``-ω delay``.
    """
    length = 64
    fs = 64.0
    freq = 8.0
    delay = 3
    omega = 2.0 * np.pi * freq / fs
    n = np.arange(length, dtype=np.float64)
    x = np.cos(omega * n)
    y = np.cos(omega * (n - delay))
    result = compute_single_bin(
        np.vstack([x, y]),
        fs,
        freq=freq,
        L=length,
        order=-1,
        win=np.ones,
        olap=0.0,
    )
    phase = float(np.angle(result.XY[0]))
    assert phase == pytest.approx(omega * delay, abs=1e-10)
    assert float(np.angle(result.Hxy[0])) == pytest.approx(-omega * delay, abs=1e-10)


def test_one_pole_transfer_function_phase():
    """Stable one-pole ``H(z) = 0.2 / (1 - 0.8 z^{-1})``.

    Hann, order=-1, no overlap, 80 segments. The expected value is the
    filter's frequency response. The comparison is the phase and a
    relative magnitude bound that rejects a swapped CSD sign or a missing
    factor of two.
    """
    fs = 256.0
    length = 256
    n = length * 80
    freq = 32.0
    rng = np.random.default_rng(_SEED)
    x = rng.standard_normal(n)
    y = signal.lfilter([0.2], [1.0, -0.8], x)
    result = compute_single_bin(
        np.vstack([x, y]),
        fs,
        freq=freq,
        L=length,
        order=-1,
        win="hann",
        olap=0.0,
    )
    omega = 2.0 * np.pi * freq / fs
    expected = 0.2 / (1.0 - 0.8 * np.exp(-1j * omega))
    got = complex(result.Hxy[0])
    assert abs(got - expected) / abs(expected) < 0.05

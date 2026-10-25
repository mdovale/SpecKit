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
"""Compare one segment length with the default frequency plan.

The default is ``vectorized_ltf_plan``. The alternative uses the same
Hann length at every frequency. It has to touch about the same number
of samples, within 5%. A worse result is recorded as a worse result.
"""

from __future__ import annotations

import numpy as np
import pytest

from speckit.plan_risk import (
    constant_length_score,
    leaked_density,
    score_band,
    search_constant_length,
    search_leakage,
)
from speckit.schedulers import vectorized_ltf_plan

_SCHEDULER = {
    "olap": 0.5,
    "bmin": 1.0,
    "Lmin": 1,
    "Jdes": 48,
    "Kdes": 40,
}


def _flat(level: float):
    def spectrum(freq: np.ndarray) -> np.ndarray:
        return np.full(np.shape(freq), level, dtype=np.float64)

    return spectrum


def _pink(freq: np.ndarray) -> np.ndarray:
    """One-sided pink spectrum, ``2/f``.

    Below 0.2 Hz the level stays at ``2/0.2``, so the value at 0 Hz
    is finite.
    """
    floor = np.maximum(np.asarray(freq, dtype=np.float64), 0.2)
    return 2.0 / floor


def _matched(samples: int, budget: int) -> bool:
    return abs(int(samples) - int(budget)) <= 0.05 * float(budget)


def test_white_reference_and_cost_matched_search():
    """White noise from 8 to 64 Hz, ``fs = 256``, ``N = 256 * 200``.

    The reference plan is Hann, length 256, with no overlap. Its risk
    equals the white-noise variance ``2 G**2 / nu`` at ``nu = 400``.
    The default plan uses ``Jdes = 48``, ``Kdes = 40``, overlap 0.5.
    On this flat spectrum the single-length search has the lower risk.
    The two sample totals differ by at most 5%.
    """
    fs = 256.0
    n = 256 * 200
    level = 2.0 / fs
    spectrum = _flat(level)
    plan = vectorized_ltf_plan(N=n, fs=fs, **_SCHEDULER)
    baseline = score_band(plan, fs, spectrum, 8.0, 64.0)
    reference = constant_length_score(n, fs, 256, 0.0, 8.0, 64.0, spectrum)
    prototype = search_constant_length(
        n, fs, 8.0, 64.0, spectrum, baseline.segment_samples
    )

    assert reference.n_bins == 57
    assert reference.length == 256
    assert reference.segment_samples == 57 * 200 * 256
    assert reference.mean_risk == pytest.approx(2.0 * level**2 / 400.0, rel=1e-12)
    assert prototype.mean_risk < baseline.mean_risk
    assert _matched(prototype.segment_samples, baseline.segment_samples)


def test_pink_constant_length_loses_to_vectorized_ltf():
    """Pink noise from 2 to 8 Hz. ``fs = 256``, ``N = 2**16``.

    The true spectrum is ``2/f``. Same plan settings as the white case.
    The single length has the higher average risk, which is why
    ``SpectrumAnalyzer`` keeps ``vectorized_ltf_plan``.
    """
    fs = 256.0
    n = 2**16
    plan = vectorized_ltf_plan(N=n, fs=fs, **_SCHEDULER)
    baseline = score_band(plan, fs, _pink, 2.0, 8.0)
    prototype = search_constant_length(n, fs, 2.0, 8.0, _pink, baseline.segment_samples)
    assert prototype.mean_risk > baseline.mean_risk
    assert _matched(prototype.segment_samples, baseline.segment_samples)


def test_half_bin_leakage_at_matched_cost():
    """A cosine halfway between two bins. ``fs = 64``, ``N = 64 * 32``, Hann.

    The default plan uses ``Jdes = 32``, ``Kdes = 20``, and overlap 0.5.
    We take its frequency nearest 8 Hz. The next bin is one resolution
    higher, ``f + fs/L``, and uses that same length. The cosine sits
    halfway between them. The allowed cost is two copies of that row.
    The search aims its first bin at 8 Hz. Its squared leakage is lower,
    and its sample total stays within 5%.
    """
    fs = 64.0
    n = 64 * 32
    plan = vectorized_ltf_plan(
        N=n,
        fs=fs,
        olap=0.5,
        bmin=1.0,
        Lmin=1,
        Jdes=32,
        Kdes=20,
    )
    freqs = np.asarray(plan["f"], dtype=np.float64)
    lengths = np.asarray(plan["L"], dtype=np.int64)
    index = int(np.argmin(np.abs(freqs - 8.0)))
    analysis = float(freqs[index])
    length = int(lengths[index])
    resolution = fs / length
    assert float(freqs[index + 1]) == pytest.approx(analysis + resolution, rel=1e-9)
    window = np.hanning(length)
    leakage = leaked_density(
        window,
        fs,
        analysis + resolution,
        analysis + 0.5 * resolution,
    )
    starts = np.asarray(plan["D"][index])
    budget = 2 * int(starts.size) * length
    prototype = search_leakage(n, fs, 8.0, budget)
    assert prototype.mean_risk < leakage**2
    assert _matched(prototype.segment_samples, budget)
    assert prototype.n_bins == 2

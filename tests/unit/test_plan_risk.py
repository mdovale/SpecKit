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
"""White-noise identity of the predicted row risk."""

from __future__ import annotations

import numpy as np
import pytest

import speckit
from speckit import compute_single_bin
from speckit.analysis import SpectrumAnalyzer
from speckit.plan_risk import (
    _count_window,
    _feasible_counts,
    _overlap_for_count,
    band_frequencies,
    constant_length_score,
    kernel_power,
    leaked_density,
    row_risk,
    search_constant_length,
    segment_starts,
)
from speckit.response import BinResponse


def _flat(level: float):
    def spectrum(freq: np.ndarray) -> np.ndarray:
        return np.full(np.shape(freq), level, dtype=np.float64)

    return spectrum


def test_flat_spectrum_mean_is_g_on_the_dft_grid():
    """Hann window, no detrend, one bin away from 0 and from ``fs/2``.

    The energy in the DFT of the windowed tone equals ``L * S2``.
    Dividing by ``L * S2`` therefore returns a flat one-sided ``G``
    with no extra scale factor. For ``L = 256`` the FFT residual is
    near ``1e-16``. The test uses ``1e-12``, which is looser than that
    residual and still tight enough to catch a missing or extra factor
    of 2.
    """
    length = 256
    fs = 256.0
    level = 2.0 / fs
    window = np.hanning(length)
    starts = segment_starts(length * 200, length, 0.0)
    for freq in (32.0, 32.5):
        row = row_risk(window, fs, freq, starts, _flat(level))
        assert row.mean == pytest.approx(level, rel=1e-12)
        assert row.truth == pytest.approx(level)
        assert row.nu == pytest.approx(400.0)
        assert row.var == pytest.approx(2.0 * level**2 / row.nu, rel=1e-12)
        assert row.risk == pytest.approx(row.var, rel=1e-12)


def test_kernel_power_matches_bin_response_coefficient():
    """Power of the bin response at one DFT frequency and at the matching
    negative frequency.
    """
    length = 256
    fs = 256.0
    freq = 32.0
    window = np.hanning(length)
    response = BinResponse(window, fs, freq, order=-1)
    power = kernel_power(window, fs, np.array([freq]))[0]
    for bin_index in (0, 10, length - 1):
        tone = bin_index * fs / length
        if bin_index > length // 2:
            tone -= fs
        href = abs(response.coefficient(tone)) ** 2
        assert power[bin_index] == pytest.approx(href, rel=1e-10)


def test_curved_spectrum_is_biased_for_a_short_window():
    """A curved spectrum, ``G = 2 / max(f, 1)``.

    A short Hann window averages ``G`` over a wide band, so the expected
    level moves away from ``G(f)``. A long window averages over a narrow
    band and stays closer. The shift comes from that width.
    """
    fs = 256.0
    freq = 16.0

    def spectrum(values: np.ndarray) -> np.ndarray:
        return 2.0 / np.maximum(np.asarray(values, dtype=np.float64), 1.0)

    wide = row_risk(np.hanning(64), fs, freq, np.array([0]), spectrum)
    narrow = row_risk(np.hanning(4096), fs, freq, np.array([0]), spectrum)
    wide_bias = abs(wide.mean - wide.truth) / wide.truth
    narrow_bias = abs(narrow.mean - narrow.truth) / narrow.truth
    assert wide_bias > 1e-3
    assert narrow_bias < wide_bias


def test_overlap_uses_the_starts_in_nu():
    """Separate segments have ``nu = 2 K``.

    Segments with 50% overlap share samples, so they have fewer degrees
    of freedom.
    """
    length = 256
    fs = 256.0
    window = np.hanning(length)
    level = 2.0 / fs
    separate = segment_starts(length * 10, length, 0.0)
    overlapped = segment_starts(length * 10, length, 0.5)
    plain = row_risk(window, fs, 32.0, separate, _flat(level))
    shared = row_risk(window, fs, 32.0, overlapped, _flat(level))
    assert plain.nu == pytest.approx(2.0 * separate.size)
    assert shared.nu < 2.0 * overlapped.size
    assert shared.nu != pytest.approx(plain.nu)
    assert shared.var != pytest.approx(plain.var)


def test_moved_start_changes_the_variance():
    length = 64
    window = np.hanning(length)
    starts = segment_starts(length * 8, length, 0.5)
    moved = starts.copy()
    moved[-1] = moved[-1] - 1
    level = 1.0
    original = row_risk(window, 64.0, 8.0, starts, _flat(level))
    shifted = row_risk(window, 64.0, 8.0, moved, _flat(level))
    assert shifted.nu != pytest.approx(original.nu)
    assert shifted.var != pytest.approx(original.var)


def test_detrend_order_is_rejected():
    window = np.hanning(32)
    with pytest.raises(ValueError, match="order"):
        row_risk(window, 32.0, 4.0, np.array([0]), _flat(1.0), order=0)


def test_leakage_matches_single_bin_gxx():
    """One cosine, Hann window, one segment. ``Gxx = 2 |X|**2 / (fs * S2)``."""
    length = 64
    fs = 64.0
    analysis = 8.0
    resolution = fs / length
    tone = analysis + 0.5 * resolution
    neighbor = analysis + resolution
    window = np.hanning(length)
    predicted = leaked_density(window, fs, neighbor, tone)
    series = np.cos(2.0 * np.pi * tone / fs * np.arange(length))
    result = compute_single_bin(
        series,
        fs,
        freq=neighbor,
        L=length,
        order=-1,
        win="hann",
        olap=0.0,
    )
    assert float(result.Gxx[0]) == pytest.approx(predicted, rel=1e-12)


def test_search_matches_an_exhaustive_cost_band():
    """On a small grid, the search keeps the lowest risk inside the 5% cost window.

    Every allowed segment count is scored directly, including its real
    degrees of freedom. The search has to pick that same minimum.
    """
    n = 600
    fs = 60.0
    level = 2.0 / fs
    budget = n * 8
    lengths = (20, 30, 40)

    def spectrum(freq: np.ndarray) -> np.ndarray:
        return np.full(np.shape(freq), level, dtype=np.float64)

    found = search_constant_length(
        n,
        fs,
        6.0,
        18.0,
        spectrum,
        budget,
        lengths=lengths,
    )
    best = None
    for length in lengths:
        resolution = fs / length
        n_bins = int(band_frequencies(6.0, 18.0, resolution).size)
        window = _count_window(budget, n_bins, length)
        if window is None:
            continue
        for count in _feasible_counts(n, length, n_bins, budget, window[0], window[1]):
            olap = _overlap_for_count(n, length, count)
            assert olap is not None
            score = constant_length_score(n, fs, length, olap, 6.0, 18.0, spectrum)
            if best is None or score.mean_risk < best:
                best = score.mean_risk
    assert best is not None
    assert found.mean_risk == pytest.approx(best, rel=1e-12)
    assert abs(found.segment_samples - budget) <= 0.05 * budget


def test_response_risk_is_not_a_scheduler_name():
    assert not hasattr(speckit, "search_constant_length")
    assert not hasattr(speckit, "search_leakage")
    with pytest.raises(ValueError, match="not recognized"):
        SpectrumAnalyzer(np.zeros(32), fs=8.0, scheduler="response_risk")

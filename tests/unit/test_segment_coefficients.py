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
"""Segment coefficients from both kernel families against the direct DFT."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from speckit import core

_ULPS = 8.0
_EPS = float(np.finfo(np.float64).eps)
_ORACLE_PATH = Path(__file__).resolve().parents[1] / "direct_dft_oracle.py"


def _oracle():
    spec = importlib.util.spec_from_file_location("direct_dft_oracle", _ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_both_families_match_the_oracle_and_the_summary():
    """One auto bin and one cross bin. The coefficient allowance is the
    oracle bound 8 * L * eps * sum(|y_windowed|).
    """
    oracle = _oracle()
    length = 64
    hop = length // 2
    n = length + hop
    window = np.hanning(length)
    omega = 2.0 * np.pi * 4.5 / length
    starts = np.array([0, hop], dtype=np.int64)
    rng = np.random.default_rng(20261025)
    x1 = rng.standard_normal(n)
    x2 = np.roll(x1, 3)
    order = -1
    reference_x, _ = oracle.segment_coefficients(
        x1, starts, length, window, omega, order
    )
    reference_y, _ = oracle.segment_coefficients(
        x2, starts, length, window, omega, order
    )
    saved = core._NUMBA_ENABLED
    try:
        for use_numba in (True, False):
            core._NUMBA_ENABLED = use_numba
            got_x, got_y = core.segment_coefficients(
                x1, x2, starts, length, window, omega, order
            )
            assert got_y is not None
            for got, reference, series in (
                (got_x, reference_x, x1),
                (got_y, reference_y, x2),
            ):
                for value, ref, start in zip(got, reference, starts):
                    _, windowed = oracle.coefficient(
                        series[int(start) : int(start) + length],
                        window,
                        omega,
                        order,
                    )
                    atol = _ULPS * length * _EPS * float(np.sum(np.abs(windowed)))
                    assert abs(value - ref) <= atol
            summary = core._bin_stats(
                x1, x2, starts, length, window, omega, order, None
            )
            assert abs(float(np.mean(np.abs(got_x) ** 2)) - summary[0]) <= 1e-9
            cross = np.mean(got_x * np.conj(got_y))
            assert abs(cross - complex(summary[2], summary[3])) <= 1e-9
    finally:
        core._NUMBA_ENABLED = saved

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
"""Production bin statistics against an independent direct DFT."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from speckit import core

# A naive sum of L complex terms. Each term is one product of a windowed
# sample with exp(-i omega n), and each product carries a few ulps.
# 8 ulps per term is the allowance: 8 * L * eps * sum(|y_windowed|).
_ULPS = 8.0
_EPS = float(np.finfo(np.float64).eps)

_ORACLE_PATH = Path(__file__).resolve().parents[1] / "direct_dft_oracle.py"


def _oracle():
    spec = importlib.util.spec_from_file_location("direct_dft_oracle", _ORACLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coefficient_atol(windowed: np.ndarray) -> float:
    length = windowed.size
    return _ULPS * length * _EPS * float(np.sum(np.abs(windowed)))


def _production_windowed(
    segment: np.ndarray, window: np.ndarray, order: int
) -> np.ndarray:
    y = np.array(segment, dtype=np.float64, copy=True)
    if order == 0:
        y -= y.mean()
    elif order in (1, 2):
        q = core._build_Q(y.size, order)
        y -= q @ (q.T @ y)
    return y * np.asarray(window, dtype=np.float64)


def _goertzel_coefficient(windowed: np.ndarray, omega: float) -> complex:
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    real, imag = core._goertzel_real_imag(
        np.ascontiguousarray(windowed, dtype=np.float64),
        cosw,
        sinw,
        2.0 * cosw,
    )
    # The recurrence equals the direct sum times exp(i omega (L - 1)).
    length = windowed.size
    return complex(real, imag) * np.exp(-1j * omega * (length - 1))


def test_oracle_module_does_not_import_speckit():
    source = _ORACLE_PATH.read_text()
    assert "import speckit" not in source
    assert "from speckit" not in source
    assert "goertzel" not in source.lower()


def test_direct_dft_matches_production_coefficients():
    """Every detrend order, one window, short and long, integer and fractional.

    The near-zero tone is a fraction of a bin. The interior tone is many
    bins up from zero. Both kernel families are checked. The Goertzel
    coefficient is rotated by ``exp(-i omega (L-1))`` before it is
    compared with the direct sum.
    """
    oracle = _oracle()
    cases = (
        ("short_integer", 32, 4.0),
        ("short_fractional", 32, 4.5),
        ("short_near_zero", 32, 0.15),
        ("long_integer", 2048, 128.0),
        ("long_fractional", 2048, 128.5),
        ("long_near_zero", 2048, 0.2),
    )
    orders = (-1, 0, 1, 2)
    saved = core._NUMBA_ENABLED
    try:
        for name, length, cycles in cases:
            hop = length // 2
            n = length + hop
            window = np.hanning(length)
            omega = 2.0 * np.pi * cycles / length
            starts = np.array([0, hop], dtype=np.int64)
            rng = np.random.default_rng(20261025 + length + int(10 * cycles))
            t = np.arange(n)
            x1 = 0.1 * rng.standard_normal(n) + np.cos(2.0 * np.pi * 0.13 * t)
            x2 = 0.1 * rng.standard_normal(n) + np.cos(2.0 * np.pi * 0.13 * t - 0.4)
            for order in orders:
                q = core._build_Q(length, order) if order in (1, 2) else None
                x1_coef, _ = oracle.segment_coefficients(
                    x1, starts, length, window, omega, order
                )
                x2_coef, _ = oracle.segment_coefficients(
                    x2, starts, length, window, omega, order
                )
                for start, _direct_x, direct_y in zip(starts, x1_coef, x2_coef):
                    seg1 = x1[int(start) : int(start) + length]
                    seg2 = x2[int(start) : int(start) + length]
                    got1 = _goertzel_coefficient(
                        _production_windowed(seg1, window, order), omega
                    )
                    got2 = _goertzel_coefficient(
                        _production_windowed(seg2, window, order), omega
                    )
                    ref1, yw1 = oracle.coefficient(seg1, window, omega, order)
                    atol1 = _coefficient_atol(yw1)
                    atol2 = _coefficient_atol(
                        oracle.coefficient(seg2, window, omega, order)[1]
                    )
                    assert abs(got1 - ref1) <= atol1, (
                        name,
                        order,
                        "X",
                        abs(got1 - ref1),
                        atol1,
                    )
                    assert abs(got2 - direct_y) <= atol2, (name, order, "Y")
                    # The allowance has to be smaller than the coefficient,
                    # or the comparison would pass for a zero transform.
                    assert atol1 < 1e-6 * max(abs(ref1), 1e-30), (
                        name,
                        order,
                        atol1,
                        ref1,
                    )
                csd = np.mean(x1_coef * np.conj(x2_coef))
                mxx = float(np.mean(np.abs(x1_coef) ** 2))
                myy = float(np.mean(np.abs(x2_coef) ** 2))
                # |X|^2 error from a coefficient error atol is 2 |X| atol + atol^2.
                power_atol = 0.0
                csd_atol = 0.0
                for direct_x, direct_y, start in zip(x1_coef, x2_coef, starts):
                    seg1 = x1[int(start) : int(start) + length]
                    _, yw1 = oracle.coefficient(seg1, window, omega, order)
                    ax = _coefficient_atol(yw1)
                    seg2 = x2[int(start) : int(start) + length]
                    _, yw2 = oracle.coefficient(seg2, window, omega, order)
                    ay = _coefficient_atol(yw2)
                    power_atol += 2.0 * abs(direct_x) * ax + ax**2
                    csd_atol += abs(direct_x) * ay + abs(direct_y) * ax + ax * ay
                power_atol /= x1_coef.size
                csd_atol /= x1_coef.size
                for use_numba in (True, False):
                    core._NUMBA_ENABLED = use_numba
                    auto = core._bin_stats(
                        x1, None, starts, length, window, omega, order, q
                    )
                    cross = core._bin_stats(
                        x1, x2, starts, length, window, omega, order, q
                    )
                    assert abs(auto[0] - mxx) <= power_atol, (
                        name,
                        order,
                        use_numba,
                        "MXX",
                    )
                    assert abs(auto[2] - mxx) <= power_atol, (
                        name,
                        order,
                        use_numba,
                        "mu_r",
                    )
                    assert auto[3] == 0.0
                    assert abs(cross[0] - mxx) <= power_atol, (name, order, use_numba)
                    assert abs(cross[1] - myy) <= power_atol, (name, order, use_numba)
                    assert abs(cross[2] - csd.real) <= csd_atol, (
                        name,
                        order,
                        use_numba,
                        "mu_r",
                    )
                    assert abs(cross[3] - csd.imag) <= csd_atol, (
                        name,
                        order,
                        use_numba,
                        "mu_i",
                    )
    finally:
        core._NUMBA_ENABLED = saved


def test_kernel_does_not_replace_nonfinite_with_zero():
    x = np.ones(32, dtype=np.float64)
    x[3] = np.nan
    window = np.hanning(32)
    starts = np.array([0], dtype=np.int64)
    saved = core._NUMBA_ENABLED
    try:
        for use_numba in (True, False):
            core._NUMBA_ENABLED = use_numba
            stats = core._bin_stats(x, None, starts, 32, window, 0.2, -1, None)
            assert not np.isfinite(stats[0]), use_numba
    finally:
        core._NUMBA_ENABLED = saved


def test_detrend_projection_matches_build_q():
    oracle = _oracle()
    rng = np.random.default_rng(7)
    for length in (32, 2048):
        segment = rng.standard_normal(length)
        for order in (1, 2):
            got = oracle.detrend(segment, order)
            q = core._build_Q(length, order)
            ref = segment - q @ (q.T @ segment)
            # Reduced QR of the same Vandermonde. Each inner product of
            # length L contributes about L ulps.
            atol = _ULPS * length * _EPS * length * float(np.max(np.abs(segment)))
            assert np.max(np.abs(got - ref)) <= atol

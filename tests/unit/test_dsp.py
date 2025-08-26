# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

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
from pytest import approx
import itertools

import numpy as np
import pandas as pd

from speckit import dsp

# --- Tests for polynomial_detrend ---


def test_polynomial_detrend():
    """Tests that polynomial detrending correctly removes known trends."""
    t = np.linspace(0, 10, 1000)
    # Signal with linear trend + constant offset
    signal_linear = 5.0 * t + 3.0 + np.sin(t)
    # Signal with quadratic trend
    signal_quad = 2.0 * t**2 - 3.0 * t + 1.0 + np.cos(t)

    detrended_linear = dsp.polynomial_detrend(signal_linear, order=1)
    detrended_quad = dsp.polynomial_detrend(signal_quad, order=2)

    # After detrending, the mean should be close to zero
    assert np.mean(detrended_linear) == pytest.approx(0, abs=1e-12)
    assert np.mean(detrended_quad) == pytest.approx(0, abs=1e-12)

    # The standard deviation should be close to that of the original sine/cosine wave
    assert np.std(detrended_linear) == pytest.approx(np.std(np.sin(t)), rel=0.1)
    assert np.std(detrended_quad) == pytest.approx(np.std(np.cos(t)), rel=0.1)


# --- Tests for DataFrame utilities ---


@pytest.fixture
def sample_dataframe():
    """Creates a sample pandas DataFrame for testing."""
    return pd.DataFrame(
        {
            "time": np.linspace(0, 9.99, 1000),
            "signal1": np.random.randn(1000),
            "signal2": np.arange(1000),
        }
    )


def test_df_timeshift(sample_dataframe):
    """Tests the DataFrame timeshift wrapper."""
    df = sample_dataframe
    fs = 100.0
    delay_sec = 0.05  # 5 samples

    # Test out-of-place shift
    df_shifted = dsp.df_timeshift(df, fs, delay_sec, columns=["signal1"], inplace=False)
    assert "signal1_shifted" in df_shifted.columns
    assert "signal1" in df_shifted.columns
    assert not np.allclose(df_shifted["signal1"], df_shifted["signal1_shifted"])

    # Test in-place shift
    df_shifted_inplace = dsp.df_timeshift(
        df.copy(), fs, delay_sec, columns=["signal1"], inplace=True
    )
    assert "signal1_shifted" not in df_shifted_inplace.columns
    assert not np.allclose(df["signal1"], df_shifted_inplace["signal1"])


# --- Tests for optimal_linear_combination ---


def test_optimal_linear_combination_siso():
    """Tests OLC on a simple single-input, single-output system."""
    rng = np.random.default_rng(seed=1)
    x = rng.normal(size=1000)
    noise = 0.1 * rng.normal(size=1000)
    A = -3.5
    y = A * x + noise
    df = pd.DataFrame({"input": x, "output": y})

    res, residual = dsp.optimal_linear_combination(
        df, inputs=["input"], output="output"
    )

    # The recovered coefficient should be very close to the true one
    recovered_A = res.x[0]
    assert recovered_A == pytest.approx(-A, rel=0.01)

    # The residual noise should have a smaller RMS than the original output
    assert np.std(residual) < np.std(y)
    assert np.std(residual) == pytest.approx(np.std(noise), rel=0.1)


# BSD 3-Clause License
#
# Copyright (c) 2022, California Institute of Technology and
# Max Planck Institute for Gravitational Physics (Albert Einstein Institute)
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
def test_lagrange_taps_linear_interpolation():
    """
    Tests the lagrange_taps function for the simplest case: order 1 (halfp=1),
    which should be equivalent to linear interpolation.
    """
    # Test shifts at key points: 0, 0.25, 0.5, 1.0
    shifts = np.array([0, 0.25, 0.5, 1.0])
    taps = dsp.lagrange_taps(shifts, halfp=1)

    # Expected taps for linear interpolation: [1-d, d]
    expected_taps = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.0, 1.0]])

    # Use numpy's testing utilities for array comparison
    np.testing.assert_allclose(taps, expected_taps, atol=1e-9)


def test_constant_integer_timeshift():
    """Test `time_shift()` using constant integer time shifts."""
    data = np.random.normal(size=10)

    shifts = [-2, 2, 0, 10, 11]
    fss = [1, 2, 11]
    orders = [1, 3, 31, 111]

    for shift, fs, order in itertools.product(shifts, fss, orders):
        shifted = dsp.timeshift(data, shift * fs, order=order)
        print(shifted)
        if shift < 0:
            assert np.all(shifted[: -shift * fs] == data[0])
            assert np.all(shifted[-shift * fs :] == data[: shift * fs])
        elif shift > 0:
            assert np.all(shifted[-shift * fs :] == data[-1])
            assert np.all(shifted[: -shift * fs] == data[shift * fs :])
        else:
            assert np.all(shifted == data)


def test_constant_fractional_timeshift_first_order():
    """Test `time_shift()` at first order using a constant time shift."""
    size = 10

    slopes = [1.23]
    offsets = [4.56]
    fss = [1]

    for slope, offset, fs in itertools.product(slopes, offsets, fss):
        times = np.arange(size) / fs
        data = slope * times + offset
        shift = np.random.uniform(low=-size / fs, high=size / fs)
        shifted = dsp.timeshift(data, shift * fs, order=1)

        i = np.arange(size)
        k = np.floor(shift * fs).astype(int)
        valid_mask = np.logical_and(i + k > 0, i + k + 1 < size)

        assert np.all(
            shifted[valid_mask] == approx(slope * (times + shift)[valid_mask] + offset)
        )


def test_constant_fractional_timeshift():
    """Test `time_shift()` at higher order using a constant time shift."""
    size = 10

    funcs = [lambda time, fs: np.sin(2 * np.pi * fs / 4 * time)]
    fss = [1, 0.2]
    orders = [11, 31, 101]

    for func, fs, order in itertools.product(funcs, fss, orders):
        times = np.arange(size) / fs
        data = func(times, fs)
        shift = np.random.uniform(low=-size / fs, high=size / fs)
        shifted = dsp.timeshift(data, shift * fs, order=order)

        i = np.arange(size)
        k = np.floor(shift * fs).astype(int)
        p = (order + 1) // 2  # pylint: disable=invalid-name
        valid_mask = np.logical_and(i + k - (p - 1) > 0, i + k + p < size)

        assert np.all(
            shifted[valid_mask] == approx(func((times + shift)[valid_mask], fs))
        )


def test_variable_integer_timeshift():
    """Test `time_shift()` using variable integer time shifts."""
    size = 10

    data = np.random.normal(size=size)
    shifts = [
        np.arange(size),
        -2 * np.arange(size) + size // 2,
        -1 * np.ones(size, dtype=int),
    ]
    fss = [1, 2, 5]
    orders = [1, 3, 11, 31]

    for shift, fs, order in itertools.product(shifts, fss, orders):
        shifted = dsp.timeshift(data, shift * fs, order=order)
        indices = np.arange(size) + shift * fs
        zeros_mask = np.logical_or(indices >= size, indices < 0)
        non_zeros_mask = np.invert(zeros_mask)

        assert np.all(shifted[zeros_mask] == 0)
        assert np.all(shifted[non_zeros_mask] == data[indices[non_zeros_mask]])


def test_variable_fractional_timeshift_first_order():
    """Test `time_shift()` at first order using variable time shifts."""
    size = 10

    slopes = [1.23]
    offsets = [4.56]
    fss = [1]

    for slope, offset, fs in itertools.product(slopes, offsets, fss):
        times = np.arange(size) / fs
        data = slope * times + offset
        shifts = np.random.uniform(low=-size / fs, high=size / fs, size=size)
        shifted = dsp.timeshift(data, shifts * fs, order=1)

        i = np.arange(size)
        k = np.floor(shifts * fs).astype(int)
        valid_mask = np.logical_and(i + k > 0, i + k + 1 < size)

        assert np.all(
            shifted[valid_mask] == approx(slope * (times + shifts)[valid_mask] + offset)
        )


def test_variable_fractional_timeshift():
    """Test `time_shift()` at higher order using variable time shifts."""
    size = 10

    funcs = [lambda time, fs: np.sin(2 * np.pi * fs / 4 * time)]
    fss = [1, 0.2]
    orders = [11, 31, 101]

    for func, fs, order in itertools.product(funcs, fss, orders):
        times = np.arange(size) / fs
        data = func(times, fs)
        shifts = np.random.uniform(low=-size / fs, high=size / fs, size=size)
        shifted = dsp.timeshift(data, shifts * fs, order=order)

        i = np.arange(size)
        k = np.floor(shifts * fs).astype(int)
        p = (order + 1) // 2  # pylint: disable=invalid-name
        valid_mask = np.logical_and(i + k - (p - 1) > 0, i + k + p < size)

        assert np.all(
            shifted[valid_mask] == approx(func((times + shifts)[valid_mask], fs))
        )

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


# --- Tests for frequency2phase ---


def test_frequency2phase():
    """Tests frequency to phase integration."""
    fs = 100.0
    t = np.linspace(0, 1, 1000)
    dt = 1.0 / fs
    
    # Constant frequency with subtract_mean=False should give linear phase ramp
    f_const = 10.0 * np.ones(1000)
    phase = dsp.frequency2phase(f_const, fs, subtract_mean=False)
    expected_phase = 2 * np.pi * 10.0 * np.cumsum(np.ones(1000) * dt)
    assert phase == pytest.approx(expected_phase, abs=1e-10)

    # Constant frequency with subtract_mean=True (default) should give zero phase
    phase_zero = dsp.frequency2phase(f_const, fs, subtract_mean=True)
    assert np.allclose(phase_zero, 0.0, atol=1e-10)


def test_frequency2phase_errors():
    """Tests error handling in frequency2phase."""
    # Empty array
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.frequency2phase([], 100.0)

    # Invalid sampling frequency
    with pytest.raises(ValueError, match="must be positive"):
        dsp.frequency2phase([1, 2, 3], 0.0)
    with pytest.raises(ValueError, match="must be positive"):
        dsp.frequency2phase([1, 2, 3], -1.0)


# --- Tests for crop_data ---


def test_crop_data():
    """Tests data cropping functionality."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Crop to middle range
    x_crop, y_crop = dsp.crop_data(x, y, 2.0, 8.0)
    assert np.all(x_crop >= 2.0)
    assert np.all(x_crop <= 8.0)
    assert len(x_crop) == len(y_crop)

    # Crop to full range
    x_crop2, y_crop2 = dsp.crop_data(x, y, 0.0, 10.0)
    assert len(x_crop2) == len(x)

    # Crop to empty range
    x_crop3, y_crop3 = dsp.crop_data(x, y, 20.0, 30.0)
    assert len(x_crop3) == 0
    assert len(y_crop3) == 0


def test_crop_data_errors():
    """Tests error handling in crop_data."""
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        dsp.crop_data(x, y[:-1], 2.0, 8.0)

    # Invalid range
    with pytest.raises(ValueError, match="must be <="):
        dsp.crop_data(x, y, 8.0, 2.0)


# --- Tests for truncation ---


def test_truncation():
    """Tests array truncation functionality."""
    data = np.arange(10)

    # Truncate 2 from each end
    truncated = dsp.truncation(data, 2)
    assert len(truncated) == 6
    assert np.all(truncated == np.arange(2, 8))

    # No truncation
    truncated_zero = dsp.truncation(data, 0)
    assert np.array_equal(truncated_zero, data)

    # Truncate 1 from each end
    truncated_one = dsp.truncation(data, 1)
    assert len(truncated_one) == 8
    assert np.all(truncated_one == np.arange(1, 9))


def test_truncation_errors():
    """Tests error handling in truncation."""
    data = np.arange(10)

    # Invalid type
    with pytest.raises(ValueError, match="must be an integer"):
        dsp.truncation(data, 2.5)

    # Negative value
    with pytest.raises(ValueError, match="must be non-negative"):
        dsp.truncation(data, -1)

    # Too large
    with pytest.raises(ValueError, match="Cannot truncate"):
        dsp.truncation(data, 6)


# --- Tests for integral_rms ---


def test_integral_rms():
    """Tests RMS computation from ASD."""
    # Create a simple ASD: constant value
    freq = np.linspace(0, 100, 1000)
    asd = 1e-6 * np.ones_like(freq)  # Constant ASD

    # Integrate over full range
    rms_full = dsp.integral_rms(freq, asd)
    expected_rms = 1e-6 * np.sqrt(100.0)  # sqrt(integral of constant^2)
    assert rms_full == pytest.approx(expected_rms, rel=1e-3)

    # Integrate over partial range
    rms_partial = dsp.integral_rms(freq, asd, pass_band=(10.0, 50.0))
    expected_rms_partial = 1e-6 * np.sqrt(40.0)
    assert rms_partial == pytest.approx(expected_rms_partial, rel=1e-3)


def test_integral_rms_errors():
    """Tests error handling in integral_rms."""
    freq = np.linspace(0, 100, 1000)
    asd = 1e-6 * np.ones_like(freq)

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        dsp.integral_rms(freq, asd[:-1])

    # Empty arrays
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.integral_rms([], [])

    # Invalid pass_band
    with pytest.raises(ValueError, match="length 2"):
        dsp.integral_rms(freq, asd, pass_band=(10.0,))
    with pytest.raises(ValueError, match="must be <="):
        dsp.integral_rms(freq, asd, pass_band=(50.0, 10.0))


# --- Tests for peak_finder ---


def test_peak_finder_freq_band():
    """Tests peak finding with frequency band filtering."""
    freq = np.linspace(0, 100, 1000)
    measurement = np.ones_like(freq)
    measurement[500] = 100.0  # Peak at 50 Hz

    # Search in a band that includes the peak
    peak_freqs, _ = dsp.peak_finder(
        freq, measurement, cnr=10, freq_band=(40.0, 60.0)
    )
    assert len(peak_freqs) > 0

    # Search in a band that excludes the peak
    peak_freqs_empty, _ = dsp.peak_finder(
        freq, measurement, cnr=10, freq_band=(10.0, 20.0)
    )
    assert len(peak_freqs_empty) == 0


def test_peak_finder_errors():
    """Tests error handling in peak_finder."""
    freq = np.linspace(0, 100, 1000)
    measurement = np.ones_like(freq)

    # Mismatched lengths
    with pytest.raises(ValueError, match="same length"):
        dsp.peak_finder(freq, measurement[:-1])

    # Empty arrays
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.peak_finder([], [])

    # Invalid rtol
    with pytest.raises(ValueError, match="must be positive"):
        dsp.peak_finder(freq, measurement, rtol=-1.0)

    # Invalid freq_band
    with pytest.raises(ValueError, match="length 2"):
        dsp.peak_finder(freq, measurement, freq_band=(10.0,))
    with pytest.raises(ValueError, match="must be <="):
        dsp.peak_finder(freq, measurement, freq_band=(50.0, 10.0))


# --- Tests for df_detrend ---


def test_df_detrend(sample_dataframe):
    """Tests DataFrame detrending functionality."""
    df = sample_dataframe.copy()
    # Add a linear trend to signal1
    df["signal1"] = df["signal1"] + 0.1 * df["time"]

    # Detrend out-of-place
    df_detrended = dsp.df_detrend(df, columns=["signal1"], order=1, inplace=False)
    assert "signal1_detrended" in df_detrended.columns
    assert "signal1" in df_detrended.columns
    # Detrended signal should have mean close to zero
    assert np.abs(np.mean(df_detrended["signal1_detrended"])) < 1e-10

    # Detrend in-place
    df_detrended_inplace = dsp.df_detrend(
        df.copy(), columns=["signal1"], order=1, inplace=True
    )
    assert "signal1_detrended" not in df_detrended_inplace.columns
    assert np.abs(np.mean(df_detrended_inplace["signal1"])) < 1e-10


def test_df_detrend_all_columns(sample_dataframe):
    """Tests detrending all columns."""
    df = sample_dataframe.copy()
    df_detrended = dsp.df_detrend(df, order=1, inplace=False)
    # Should have detrended versions of numeric columns
    assert "signal1_detrended" in df_detrended.columns
    assert "signal2_detrended" in df_detrended.columns


def test_df_detrend_errors():
    """Tests error handling in df_detrend."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Not a DataFrame
    with pytest.raises(TypeError, match="pandas DataFrame"):
        dsp.df_detrend([1, 2, 3])

    # Empty DataFrame
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.df_detrend(pd.DataFrame())

    # Negative order
    with pytest.raises(ValueError, match="must be non-negative"):
        dsp.df_detrend(df, order=-1)

    # Missing columns
    with pytest.raises(ValueError, match="not found"):
        dsp.df_detrend(df, columns=["nonexistent"])


# --- Tests for df_timeshift edge cases ---


def test_df_timeshift_zero_shift(sample_dataframe):
    """Tests timeshift with zero shift."""
    df = sample_dataframe
    df_shifted = dsp.df_timeshift(df, fs=100.0, seconds=0.0, columns=["signal1"])
    assert df_shifted.equals(df)


def test_df_timeshift_truncate(sample_dataframe):
    """Tests timeshift with truncation."""
    df = sample_dataframe
    df_shifted = dsp.df_timeshift(
        df, fs=100.0, seconds=0.05, columns=["signal1"], truncate=True
    )
    # Should be shorter due to truncation
    assert len(df_shifted) < len(df)

    # Test with integer truncate
    df_shifted_int = dsp.df_timeshift(
        df, fs=100.0, seconds=0.05, columns=["signal1"], truncate=5
    )
    assert len(df_shifted_int) == len(df) - 10


def test_df_timeshift_errors():
    """Tests error handling in df_timeshift."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Not a DataFrame
    with pytest.raises(TypeError, match="pandas DataFrame"):
        dsp.df_timeshift([1, 2, 3], fs=1.0, seconds=0.1)

    # Empty DataFrame
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.df_timeshift(pd.DataFrame(), fs=1.0, seconds=0.1)

    # Invalid fs
    with pytest.raises(ValueError, match="must be positive"):
        dsp.df_timeshift(df, fs=0.0, seconds=0.1)
    with pytest.raises(ValueError, match="must be positive"):
        dsp.df_timeshift(df, fs=-1.0, seconds=0.1)

    # Missing columns
    with pytest.raises(ValueError, match="not found"):
        dsp.df_timeshift(df, fs=1.0, seconds=0.1, columns=["nonexistent"])

    # Invalid truncate type
    with pytest.raises(ValueError, match="bool, int, or None"):
        dsp.df_timeshift(df, fs=1.0, seconds=0.1, truncate="invalid")


# --- Tests for optimal_linear_combination edge cases ---


def test_optimal_linear_combination_mimo():
    """Tests OLC on a multiple-input, multiple-output system."""
    rng = np.random.default_rng(seed=2)
    x1 = rng.normal(size=1000)
    x2 = rng.normal(size=1000)
    noise = 0.1 * rng.normal(size=1000)
    A1, A2 = -2.0, 1.5
    y = A1 * x1 + A2 * x2 + noise
    df = pd.DataFrame({"input1": x1, "input2": x2, "output": y})

    res, residual = dsp.optimal_linear_combination(
        df, inputs=["input1", "input2"], output="output"
    )

    # The recovered coefficients should be close to the true ones
    recovered_A1 = res.x[0]
    recovered_A2 = res.x[1]
    assert recovered_A1 == pytest.approx(-A1, rel=0.05)
    assert recovered_A2 == pytest.approx(-A2, rel=0.05)

    # The residual noise should have a smaller RMS than the original output
    assert np.std(residual) < np.std(y)


def test_optimal_linear_combination_frequency_domain():
    """Tests OLC in frequency domain."""
    rng = np.random.default_rng(seed=3)
    x = rng.normal(size=1000)
    noise = 0.1 * rng.normal(size=1000)
    A = -3.5
    y = A * x + noise
    df = pd.DataFrame({"input": x, "output": y})

    res, residual = dsp.optimal_linear_combination(
        df, inputs=["input"], output="output", domain="frequency"
    )

    # Should still recover the coefficient reasonably well
    recovered_A = res.x[0]
    assert recovered_A == pytest.approx(-A, rel=0.1)


def test_optimal_linear_combination_errors():
    """Tests error handling in optimal_linear_combination."""
    df = pd.DataFrame({"input": [1, 2, 3], "output": [4, 5, 6]})

    # Not a DataFrame
    with pytest.raises(TypeError, match="pandas DataFrame"):
        dsp.optimal_linear_combination([1, 2, 3], inputs=["input"], output="output")

    # Empty DataFrame
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.optimal_linear_combination(
            pd.DataFrame(), inputs=["input"], output="output"
        )

    # Invalid inputs type
    with pytest.raises(TypeError, match="list or tuple"):
        dsp.optimal_linear_combination(df, inputs="input", output="output")

    # Empty inputs
    with pytest.raises(ValueError, match="must not be empty"):
        dsp.optimal_linear_combination(df, inputs=[], output="output")

    # Invalid domain
    with pytest.raises(ValueError, match="'time' or 'frequency'"):
        dsp.optimal_linear_combination(
            df, inputs=["input"], output="output", domain="invalid"
        )

    # Missing columns
    with pytest.raises(ValueError, match="not found"):
        dsp.optimal_linear_combination(
            df, inputs=["nonexistent"], output="output"
        )
    with pytest.raises(ValueError, match="not found"):
        dsp.optimal_linear_combination(df, inputs=["input"], output="nonexistent")

    # Invalid tol
    with pytest.raises(ValueError, match="must be positive"):
        dsp.optimal_linear_combination(
            df, inputs=["input"], output="output", tol=-1.0
        )

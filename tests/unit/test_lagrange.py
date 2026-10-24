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
import itertools

import numpy as np
from pytest import approx

from speckit.lagrange import lagrange_taps, timeshift


def test_lagrange_taps_linear_interpolation():
    """
    Tests the lagrange_taps function for the simplest case: order 1 (halfp=1),
    which should be equivalent to linear interpolation.
    """
    # Test shifts at key points: 0, 0.25, 0.5, 1.0
    shifts = np.array([0, 0.25, 0.5, 1.0])
    taps = lagrange_taps(shifts, halfp=1)

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
        shifted = timeshift(data, shift * fs, order=order)
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
        shifted = timeshift(data, shift * fs, order=1)

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
        shifted = timeshift(data, shift * fs, order=order)

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
        shifted = timeshift(data, shift * fs, order=order)
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
        shifted = timeshift(data, shifts * fs, order=1)

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
        shifted = timeshift(data, shifts * fs, order=order)

        i = np.arange(size)
        k = np.floor(shifts * fs).astype(int)
        p = (order + 1) // 2  # pylint: disable=invalid-name
        valid_mask = np.logical_and(i + k - (p - 1) > 0, i + k + p < size)

        assert np.all(
            shifted[valid_mask] == approx(func((times + shifts)[valid_mask], fs))
        )

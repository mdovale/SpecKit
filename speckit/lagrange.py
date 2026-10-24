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

# Adapted from PyTDI (doi:10.5281/zenodo.8429119), Staab, Bayle, and Hartwig.
# speckit.dsp re-exports lagrange_taps and timeshift.

import logging

import numpy as np

logger = logging.getLogger(__name__)


def lagrange_taps(shift_fracs, halfp):
    """Computes the coefficients for a Lagrange fractional delay filter.

    This function calculates the FIR filter tap coefficients for a centered
    Lagrange interpolating polynomial of order `2 * halfp - 1`. It uses an
    efficient, vectorized formulation to compute coefficients for multiple
    fractional delays simultaneously.

    This is a core utility function used by `timeshift`.

    Parameters
    ----------
    shift_fracs : np.ndarray
        An array of fractional time shifts, with each value typically in the
        range [0, 1). Each value corresponds to the sub-sample shift for
        which a set of filter coefficients is required.
    halfp : int
        The number of filter taps on each side of the interpolation center.
        This is related to the filter order `p` by `halfp = (p + 1) // 2`.
        The total number of taps will be `2 * halfp`.

    Returns
    -------
    np.ndarray
        A 2D array of the calculated Lagrange coefficients. The shape of the
        array is `(N, 2 * halfp)`, where `N` is the number of fractional
        shifts in `shift_fracs`. Each row contains the `2 * halfp` filter
        taps for the corresponding shift in `shift_fracs`.
    """
    num_taps = 2 * halfp
    taps = np.zeros((num_taps, shift_fracs.size), dtype=np.float64)

    # --- Case 1: Linear Interpolation (order=1, halfp=1) ---
    if halfp == 1:
        taps[0] = 1 - shift_fracs
        taps[1] = shift_fracs
        return taps.T

    # --- Case 2: Higher-Order Interpolation (halfp > 1) ---
    # The algorithm is structured to build parts of the formula and then apply
    # common factors to all taps at the end.

    # This running 'factor' is used to set the initial values for the outer taps.
    # These values are NOT final until the last two multiplications are applied.
    factor = np.ones(shift_fracs.size, dtype=np.float64)
    factor *= shift_fracs * (1 - shift_fracs)

    for j in range(1, halfp):
        # Iteratively build the product term for the outer taps
        factor *= (-1) * (1 - j / halfp) / (1 + j / halfp)
        taps[halfp - 1 - j] = factor / (j + shift_fracs)
        taps[halfp + j] = factor / (j + 1 - shift_fracs)

    # Set the initial values for the two central taps.
    taps[halfp - 1] = 1 - shift_fracs
    taps[halfp] = shift_fracs

    # Now, apply the remaining common factors to ALL taps to finalize them.
    # First common factor: product over (1 - (d/j)^2)
    for j in range(2, halfp):
        taps *= 1 - (shift_fracs / j) ** 2

    # Second common factor: final normalization term.
    taps *= (1 + shift_fracs) * (1 - shift_fracs / halfp)

    return taps.T


def timeshift(data, shifts, order=31):
    """Time-shift data using high-order Lagrange interpolation.

    This function applies a fractional time delay or advancement to a signal
    by convolving it with a time-varying finite-impulse response (FIR)
    Lagrange interpolation filter.

    Parameters
    ----------
    data : np.ndarray
        The 1D input signal to be shifted.
    shifts : Union[float, np.ndarray]
        The desired time shift(s) in units of samples. A positive value
        delays the signal (shifts it to the right), while a negative value
        advances it. Can be a single float for a constant shift or an array
        of the same size as `data` for a time-varying shift.
    order : int, optional
        The order of the Lagrange interpolator, which must be an odd integer.
        Higher orders provide better accuracy but have higher computational
        cost and longer filter ringing. Defaults to 31.

    Returns
    -------
    Union[float, np.ndarray]
        The time-shifted signal.

    Raises
    ------
    ValueError
        If `order` is not an odd integer or if `data` and `shifts` have
        mismatched shapes for a time-varying shift.

    Notes
    -----
    The implementation is fully vectorized using NumPy for high performance,
    avoiding slow Python loops even for time-varying shifts.
    """
    if order % 2 == 0:
        raise ValueError(f"`order` must be an odd integer (got {order})")

    data = np.asarray(data)
    shifts = np.asarray(shifts)

    # --- Handle trivial cases ---
    if data.size <= 1:
        logger.debug("Input data is scalar or empty, returning as is.")
        return data.item() if data.size == 1 else data
    if np.all(shifts == 0):
        logger.debug("Time shifts are all zero, returning original data.")
        return data

    logger.debug("Time shifting data samples (order=%d)", order)

    halfp = (order + 1) // 2
    # num_taps = 2 * halfp

    shift_ints = np.floor(shifts).astype(int)
    shift_fracs = shifts - shift_ints

    logger.debug("Computing Lagrange coefficients")
    taps = lagrange_taps(shift_fracs, halfp)

    # --- Constant Shift Path (Optimized for a single shift value) ---
    if shifts.size == 1:
        logger.debug("Constant shifts, using correlation method")
        shift_int = shift_ints.item()

        i_min = shift_int - (halfp - 1)
        i_max = shift_int + halfp + data.size

        if i_max - 1 < 0:
            return np.repeat(data[0], data.size)
        if i_min > data.size - 1:
            return np.repeat(data[-1], data.size)

        pad_left = max(0, -i_min)
        pad_right = max(0, i_max - data.size)
        logger.debug("Padding data (left=%d, right=%d)", pad_left, pad_right)
        data_trimmed = data[max(0, i_min) : min(data.size, i_max)]
        data_padded = np.pad(data_trimmed, (pad_left, pad_right), mode="edge")

        logger.debug("Computing correlation product")
        return np.correlate(data_padded, taps[0], mode="valid")

    # --- Time-Varying Shift Path ---
    if data.size != shifts.size:
        raise ValueError(
            f"`data` and `shift` must be of the same size (got {data.size}, {shifts.size})"
        )

    logger.debug("Time-varying shifts, using sliding window view")
    indices = np.clip(
        np.arange(data.size) + shift_ints, -(halfp + 1), data.size + (halfp - 1)
    )
    padded = np.pad(data, 2 * halfp)
    slices = np.lib.stride_tricks.sliding_window_view(padded, 2 * halfp)
    slices = slices[indices + 2 * halfp - (halfp - 1)]
    logger.debug("Computing matrix-vector product")
    return np.einsum("ij,ij->i", taps, slices)

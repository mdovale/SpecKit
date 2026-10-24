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
"""Array helpers on the spectral-estimator import path.

`integral_rms` is the one-sided band RMS used by `SpectrumResult.get_rms`:
the square root of a trapezoidal integral of ASD squared. `crop_data` is the
inclusive slice that integral uses. Neither function imports pandas.
"""

import logging

import numpy as np
from scipy.integrate import cumulative_trapezoid

logger = logging.getLogger(__name__)


def crop_data(x, y, xmin, xmax):
    """
    Crop data to a specified range in x.

    Parameters
    ----------
    x : array-like
        The x-axis data (e.g., frequency, time).
    y : array-like
        The y-axis data corresponding to x.
    xmin : float
        Lower bound for cropping (inclusive).
    xmax : float
        Upper bound for cropping (inclusive).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple (x_cropped, y_cropped) containing the cropped arrays.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths, or if `xmin > xmax`.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError(
            f"Arrays `x` and `y` must have the same length, got {len(x)} and {len(y)}."
        )
    if xmin > xmax:
        raise ValueError(f"`xmin` ({xmin}) must be <= `xmax` ({xmax}).")
    if x.size == 0:
        return x, y

    mask = (x >= xmin) & (x <= xmax)
    return x[mask], y[mask]


def integral_rms(fourier_freq, asd, pass_band=None):
    """
    Compute the RMS as integral of an Amplitude Spectral Density (ASD).

    The ASD is one-sided when it comes from `compute_spectrum`. The RMS is
    `sqrt` of the trapezoidal integral of ASD squared, with `initial=0`, and
    the returned value is the last sample of that cumulative integral.

    Parameters
    ----------
    fourier_freq : array-like
        Fourier frequency array in Hz.
    asd : array-like
        Amplitude spectral density from which RMS is computed.
    pass_band : tuple of float, optional
        Frequency band for integration as (fmin, fmax). If None, integrates
        over the entire frequency range. Default is None.

    Returns
    -------
    float
        The RMS value computed from the integral of the squared ASD.

    Raises
    ------
    ValueError
        If `fourier_freq` and `asd` have different lengths, or if `pass_band`
        is invalid.
    """
    fourier_freq = np.asarray(fourier_freq)
    asd = np.asarray(asd)

    if len(fourier_freq) != len(asd):
        raise ValueError(
            f"Arrays `fourier_freq` and `asd` must have the same length, "
            f"got {len(fourier_freq)} and {len(asd)}."
        )
    if fourier_freq.size == 0:
        raise ValueError("Input arrays must not be empty.")

    if pass_band is None:
        pass_band = [-np.inf, np.inf]
    else:
        if len(pass_band) != 2:
            raise ValueError(
                f"`pass_band` must be a tuple of length 2, got {len(pass_band)}."
            )
        if pass_band[0] > pass_band[1]:
            raise ValueError(
                f"`pass_band[0]` ({pass_band[0]}) must be <= "
                f"`pass_band[1]` ({pass_band[1]})."
            )

    integral_range_min = max(np.min(fourier_freq), pass_band[0])
    integral_range_max = min(np.max(fourier_freq), pass_band[1])

    if integral_range_min >= integral_range_max:
        logger.warning("No valid frequency range for integration. Returning 0.0.")
        return 0.0

    f_tmp, asd_tmp = crop_data(
        fourier_freq, asd, integral_range_min, integral_range_max
    )

    if len(f_tmp) == 0:
        logger.warning("No data points in the specified frequency band. Returning 0.0.")
        return 0.0

    integral_rms2 = cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return np.sqrt(integral_rms2[-1])

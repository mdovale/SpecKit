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
"""Flat-top windows and their recommended overlaps.

Each window is the cosine sum

    w(z) = a0 - a1 cos(2z) + a2 cos(4z) - ...

with z = pi * arange(M) / M. The coefficients are the values published by
Heinzel, Rüdiger, and Schilling (2002). This module is SpecKit code under
the project BSD-3-Clause license.

Implemented flat-top window functions:
SFT3F, 31.7 dB PSLL, differentiable, 0.0082 dB emax
SFT4F, 44.7 dB PSLL, 2nd differentiable, 0.0041 dB emax
SFT5F, 57.3 dB PSLL, 3rd differentiable, -0.0025 dB emax
SFT3M, 44.2 dB PSLL, -0.0115 dB emax
SFT4M, 66.5 dB PSLL, -0.0067 dB emax
SFT5M, 89.9 dB PSLL, 0.0039 dB emax
FTNI (National Instruments), 44.4 dB PSLL, 0.0169 dB emax
FTHP (old Hewlett Packard), 70.4 dB PSLL, 0.0096 dB emax
FTSRS (Stanford Research SR785), 76.6 dB PSLL, differentiable, -0.0156 dB emax
Matlab, 93.0 dB PSLL, 0.0097 dB emax
HFT70 (3-term cosine), 70.4 dB PSLL, -0.0065 dB emax
HFT95 (4-term cosine), 95.0 dB PSLL, 0.0044 dB emax
HFT90D (4-term cosine), 90.2 dB PSLL, differentiable, -0.0039 dB emax
HFT116D (5-term cosine), 116.8 dB PSLL, differentiable, -0.0028 dB emax
HFT144D (6-term cosine), 144.1 dB PSLL, differentiable, 0.0021 dB emax
HFT169D (7-term cosine), 169.5 dB PSLL, differentiable, 0.0017 dB emax
HFT196D (8-term cosine), 196.2 dB PSLL, differentiable, 0.0013 dB emax
HFT223D (9-term cosine), 223.0 dB PSLL, differentiable, -0.0011 dB emax
HFT248D (10-term cosine), 248.4 dB PSLL, differentiable, 0.0009 dB emax

In the above, "F" indicates "fast decaying" which enforces a high
degree of differentiability, "M" indicates minimum sidelobe, and
"HFT" indicates the window with lowest sidelobe level.

It also includes a dictionary in which the optimal overlap values are given.

Ref:
Spectrum and spectral density estimation by the Discrete Fourier transform
(DFT), including a comprehensive list of window functions and some new
flat-top windows.
G. Heinzel, A. Rüdiger, and R. Schilling,
Max-Planck-Institut für Gravitationsphysik
(Albert-Einstein-Institut)
Teilinstitut Hannover,
February 15, 2002.
"""

from typing import Sequence

import numpy as np


def _cosine_sum(M: int, coeffs: Sequence[float]) -> np.ndarray:
    """Return a0 - a1 cos(2z) + a2 cos(4z) - ... at z = pi * arange(M) / M.

    Parameters
    ----------
    M : int
        Window length in samples.
    coeffs : sequence of float
        Cosine-sum coefficients ``(a0, a1, ...)``.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    z = np.pi / M * np.arange(0, M)
    w = np.zeros(M, dtype=np.float64)
    for k, coeff in enumerate(coeffs):
        w = w + ((-1) ** k) * coeff * np.cos((2 * k) * z)
    return w


def SFT3F(M: int) -> np.ndarray:
    """Flat-top window function SFT3F.

    Differentiable, -31.7 dB, NBW 3.1681 bins, 0.0082 dB, first zero at +/- 3 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.26526, 0.5, 0.23474))


def SFT4F(M: int) -> np.ndarray:
    """Flat-top window function SFT4F.

    2nd differentiable, -44.7 dB, NBW 3.7970 bins, 0.0041 dB, first zero at
    +/- 4 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.21706, 0.42103, 0.28294, 0.07897))


def SFT5F(M: int) -> np.ndarray:
    """Flat-top window function SFT5F.

    3rd differentiable, -57.3 dB, NBW 4.3412 bins, -0.0025 dB, first zero at
    +/- 5 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.1881, 0.36923, 0.28702, 0.13077, 0.02488))


def SFT3M(M: int) -> np.ndarray:
    """Flat-top window function SFT3M.

    -44.2 dB, NBW 2.9452 bins, -0.0115 dB, first zero at +/- 3 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.28235, 0.52105, 0.19659))


def SFT4M(M: int) -> np.ndarray:
    """Flat-top window function SFT4M.

    -66.5 dB, NBW 3.3868 bins, -0.0067 dB, first zero at +/- 4 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.241906, 0.460841, 0.255381, 0.041872))


def SFT5M(M: int) -> np.ndarray:
    """Flat-top window function SFT5M.

    -89.9 dB, NBW 3.8852 bins, 0.0039 dB, first zero at +/- 5 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.209671, 0.407331, 0.281225, 0.092669, 0.0091036))


def FTNI(M: int) -> np.ndarray:
    """Flat-top window function FTNI (National Instruments).

    -44.4 dB, NBW 2.9656 bins, 0.0169 dB, first zero at +/- 3 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (0.2810639, 0.5208972, 0.1980399))


def FTHP(M: int) -> np.ndarray:
    """Flat-top window function FTHP (old Hewlett Packard).

    -70.4 dB, NBW 3.4279 bins, 0.0096 dB, first zero at +/- 4 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (1.0, 1.912510941, 1.079173272, 0.1832630879))


def FTSR(M: int) -> np.ndarray:
    """Flat-top window function FTSRS (Stanford Research SR785).

    Differentiable, -76.6 dB, NBW 3.7702 bins, -0.0156 dB, first zero at
    +/- 4.72 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (1.0, 1.93, 1.29, 0.388, 0.028))


def Matlab(M: int) -> np.ndarray:
    """Flat-top window function Matlab.

    -93.0 dB, NBW 3.774 bins, 0.0025 dB, first zero at +/- 5.01 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(
        M, (0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)
    )


def HFT70(M: int) -> np.ndarray:
    """Flat-top window function HFT70 (lowest sidelobe level with 3 cosine terms).

    -70.4 dB, NBW 3.4129 bins, -0.0065 dB, first zero at +/- 4 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (1.0, 1.90796, 1.07349, 0.18199))


def HFT95(M: int) -> np.ndarray:
    """Flat-top window function HFT95 (lowest sidelobe level with 4 cosine terms).

    -95.0 dB, NBW 3.8112 bins, 0.0044 dB, first zero at +/- 5 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (1.0, 1.9383379, 1.3045202, 0.4028270, 0.0350665))


def HFT90D(M: int) -> np.ndarray:
    """Flat-top window function HFT90D (lowest sidelobe level with 4 cosine terms).

    Differentiable, -90.2 dB, NBW 3.8832 bins, -0.0039 dB, first zero at
    +/- 5 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (1.0, 1.942604, 1.340318, 0.440811, 0.043097))


def HFT116D(M: int) -> np.ndarray:
    """Flat-top window function HFT116D (lowest sidelobe level with 5 cosine terms).

    Differentiable, -116.8 dB, NBW 4.2186 bins, -0.0028 dB, first zero at
    +/- 6 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(M, (1.0, 1.9575375, 1.4780705, 0.6367431, 0.1228389, 0.0066288))


def HFT144D(M: int) -> np.ndarray:
    """Flat-top window function HFT144D (lowest sidelobe level with 6 cosine terms).

    Differentiable, -144.1 dB, NBW 4.5386 bins, 0.0021 dB, first zero at
    +/- 7 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(
        M,
        (
            1.0,
            1.96760033,
            1.57983607,
            0.81123644,
            0.22583558,
            0.02773848,
            0.00090360,
        ),
    )


def HFT169D(M: int) -> np.ndarray:
    """Flat-top window function HFT169D (lowest sidelobe level with 7 cosine terms).

    Differentiable, -169.5 dB, NBW 4.8347 bins, 0.0017 dB, first zero at
    +/- 8 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(
        M,
        (
            1.0,
            1.97441842,
            1.65409888,
            0.95788186,
            0.33673420,
            0.06364621,
            0.00521942,
            0.00010599,
        ),
    )


def HFT196D(M: int) -> np.ndarray:
    """Flat-top window function HFT196D (lowest sidelobe level with 8 cosine terms).

    Differentiable, -196.2 dB, NBW 5.1134 bins, 0.0013 dB, first zero at
    +/- 9 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(
        M,
        (
            1.0,
            1.979280420,
            1.710288951,
            1.081629853,
            0.448734314,
            0.112376628,
            0.015122992,
            0.000871252,
            0.000011896,
        ),
    )


def HFT223D(M: int) -> np.ndarray:
    """Flat-top window function HFT223D (lowest sidelobe level with 9 cosine terms).

    Differentiable, -223.0 dB, NBW 5.3888 bins, -0.0011 dB, first zero at
    +/- 10 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(
        M,
        (
            1.0,
            1.98298997309,
            1.75556083063,
            1.19037717712,
            0.56155440797,
            0.17296769663,
            0.03233247087,
            0.00324954578,
            0.00013801040,
            0.00000132725,
        ),
    )


def HFT248D(M: int) -> np.ndarray:
    """Flat-top window function HFT248D (lowest sidelobe level with 10 cosine terms).

    Differentiable, -248.4 dB, NBW 5.6512 bins, 0.0009 dB, first zero at
    +/- 11 bins.

    Parameters
    ----------
    M : int
        Window length in samples.

    Returns
    -------
    w : np.ndarray
        Window samples of length ``M``.
    """
    return _cosine_sum(
        M,
        (
            1.0,
            1.985844164102,
            1.791176438506,
            1.282075284005,
            0.667777530266,
            0.240160796576,
            0.056656381764,
            0.008134974479,
            0.000624544650,
            0.000019808998,
            0.000000132974,
        ),
    )


olap_dict = {
    "Rectangular": 0.0,
    "Welch": 0.293,
    "Barlett": 0.500,
    "Hanning": 0.500,
    "Hamming": 0.500,
    "Nuttall3": 0.647,
    "Nuttall4": 0.705,
    "Nuttall3a": 0.612,
    "Kaiser3": 0.619,
    "Nuttall3b": 0.598,
    "Nuttall4a": 0.680,
    "BH92": 0.661,
    "Nuttall4b": 0.663,
    "Kaiser4": 0.670,
    "Nuttall4c": 0.656,
    "Kaiser5": 0.705,
    "SFT3F": 0.667,
    "SFT3M": 0.655,
    "FTNI": 0.656,
    "SFT4F": 0.750,
    "SFT5F": 0.785,
    "SFT4M": 0.721,
    "FTHP": 0.723,
    "HFT70": 0.722,
    "FTSRS": 0.754,
    "SFT5M": 0.760,
    "HFT90D": 0.760,
    "HFT95": 0.756,
    "HFT116D": 0.782,
    "HFT144D": 0.799,
    "HFT169D": 0.812,
    "HFT196D": 0.823,
    "HFT223D": 0.833,
    "HFT248D": 0.841,
    0: 0.0,
    1: 0.293,
    2: 0.500,
    3: 0.500,
    4: 0.500,
    5: 0.647,
    6: 0.705,
    7: 0.612,
    8: 0.598,
    9: 0.680,
    10: 0.661,
    11: 0.663,
    12: 0.656,
    13: 0.667,
    14: 0.655,
    15: 0.656,
    16: 0.750,
    17: 0.785,
    18: 0.721,
    19: 0.723,
    20: 0.722,
    21: 0.754,
    22: 0.760,
    23: 0.760,
    24: 0.756,
    25: 0.782,
    26: 0.799,
    27: 0.812,
    28: 0.823,
    29: 0.833,
    30: 0.841,
}

win_dict = {
    "SFT3F": SFT3F,
    "SFT3M": SFT3M,
    "FTNI": FTNI,
    "SFT4F": SFT4F,
    "SFT5F": SFT5F,
    "SFT4M": SFT4M,
    "FTHP": FTHP,
    "HFT70": HFT70,
    "FTSRS": FTSR,
    "SFT5M": SFT5M,
    "HFT90D": HFT90D,
    "HFT95": HFT95,
    "HFT116D": HFT116D,
    "HFT144D": HFT144D,
    "HFT169D": HFT169D,
    "HFT196D": HFT196D,
    "HFT223D": HFT223D,
    "HFT248D": HFT248D,
}

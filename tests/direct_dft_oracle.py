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
"""Direct-DFT reference for one analysis bin.

Does not import the production package. Each segment is detrended from the documented
projection, multiplied by the window it is given, and summed against
``exp(-i omega n)``. One-sided scaling is not applied.
"""

from __future__ import annotations

import numpy as np


def detrend(segment: np.ndarray, order: int) -> np.ndarray:
    """Return a copy of ``segment`` after the documented detrend.

    ``order=-1`` leaves the samples unchanged. ``order=0`` subtracts the
    mean. ``order=1`` and ``2`` subtract the projection onto polynomials
    in ``t`` from ``-1`` to ``1``, via a reduced QR of that Vandermonde.
    """
    y = np.array(segment, dtype=np.float64, copy=True)
    if order == -1:
        return y
    if order == 0:
        y -= y.mean()
        return y
    if order not in (1, 2):
        raise ValueError(f"order must be -1, 0, 1, or 2, got {order!r}.")
    length = y.size
    t = np.linspace(-1.0, 1.0, length, dtype=np.float64)
    if order == 1:
        basis = np.column_stack([np.ones(length, dtype=np.float64), t])
    else:
        basis = np.column_stack([np.ones(length, dtype=np.float64), t, t * t])
    q, _ = np.linalg.qr(basis, mode="reduced")
    y -= q @ (q.T @ y)
    return y


def coefficient(
    segment: np.ndarray, window: np.ndarray, omega: float, order: int
) -> tuple[complex, np.ndarray]:
    """Direct Fourier coefficient and the windowed, detrended segment."""
    y = detrend(segment, order)
    w = np.asarray(window, dtype=np.float64)
    windowed = y * w
    n = np.arange(windowed.size, dtype=np.float64)
    value = np.dot(windowed, np.exp(-1j * omega * n))
    return complex(value), windowed


def segment_coefficients(
    x: np.ndarray,
    starts: np.ndarray,
    length: int,
    window: np.ndarray,
    omega: float,
    order: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """One direct coefficient per start in ``starts``."""
    values = []
    windowed = []
    for start in np.asarray(starts, dtype=np.int64):
        value, yw = coefficient(
            x[int(start) : int(start) + length], window, omega, order
        )
        values.append(value)
        windowed.append(yw)
    return np.asarray(values, dtype=np.complex128), windowed

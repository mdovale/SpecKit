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
"""
core_cuda.py — CUDA-accelerated single-bin spectral kernels
-----------------------------------------------------------------------------
CUDA implementations of the spectral analysis kernels using Numba CUDA.
Each kernel processes segments in parallel on the GPU, with one thread per segment.

Design notes:
- Uses @cuda.jit for GPU kernel compilation
- One thread per segment (parallel over K segments)
- Window weights w and detrend basis Q are loaded into shared memory when beneficial
- Returns same 5-tuple format as CPU kernels: (MXX, MYY, mu_r, mu_i, M2)
- Host wrapper functions handle memory transfer and kernel launch
-----------------------------------------------------------------------------
"""

__all__ = [
    "_stats_win_only_auto_cuda",
    "_stats_win_only_csd_cuda",
    "_stats_detrend0_auto_cuda",
    "_stats_detrend0_csd_cuda",
    "_stats_poly_auto_cuda",
    "_stats_poly_csd_cuda",
]

import numpy as np
import math
from numba import cuda
from numba import types

from .core import _reduce_stats_nb

# Threads per block (good occupancy for most GPUs)
THREADS_PER_BLOCK = 256


# CUDA kernel: window-only auto-spectral
@cuda.jit
def _stats_win_only_auto_cuda_kernel(x, starts, L, w, omega, xx, yy, xyr, xyi):
    """CUDA kernel for window-only auto-spectral statistics."""
    j = cuda.grid(1)  # Thread index = segment index
    if j < starts.shape[0]:
        s = int(starts[j])
        cosw = math.cos(omega)
        sinw = math.sin(omega)
        coeff = 2.0 * cosw

        # Goertzel recurrence with streaming window application
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            v = x[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0

        r = s1 - s2 * cosw
        i = s2 * sinw
        p = r * r + i * i

        xx[j] = p
        yy[j] = p
        xyr[j] = p
        xyi[j] = 0.0


def _stats_win_only_auto_cuda(x, starts, L, w, omega):
    """
    CUDA host function for window-only auto-spectral statistics.

    Parameters
    ----------
    x : (N,) ndarray
        Input time series (float64).
    starts : (K,) ndarray
        Segment start indices.
    L : int
        Segment length.
    w : (L,) ndarray
        Window weights.
    omega : float
        Digital radian frequency (can be fractional).

    Returns
    -------
    MXX, MYY, mu_r, mu_i, M2 : floats
        Sufficient statistics; for auto: MYY==MXX, mu_i==0.
    """
    K = starts.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Transfer arrays to device
    x_d = cuda.to_device(np.ascontiguousarray(x, dtype=np.float64))
    starts_d = cuda.to_device(np.ascontiguousarray(starts, dtype=np.int64))
    w_d = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))

    # Allocate output arrays on device
    xx_d = cuda.device_array(K, dtype=np.float64)
    yy_d = cuda.device_array(K, dtype=np.float64)
    xyr_d = cuda.device_array(K, dtype=np.float64)
    xyi_d = cuda.device_array(K, dtype=np.float64)

    # Launch kernel
    blocks = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _stats_win_only_auto_cuda_kernel[blocks, THREADS_PER_BLOCK](
        x_d, starts_d, L, w_d, omega, xx_d, yy_d, xyr_d, xyi_d
    )

    # Copy results back
    xx = xx_d.copy_to_host()
    yy = yy_d.copy_to_host()
    xyr = xyr_d.copy_to_host()
    xyi = xyi_d.copy_to_host()

    # Reduce on CPU (minimal overhead)
    return _reduce_stats_nb(xx, yy, xyr, xyi)


# CUDA kernel: window-only cross-spectral
@cuda.jit
def _stats_win_only_csd_cuda_kernel(x1, x2, starts, L, w, omega, xx, yy, xyr, xyi):
    """CUDA kernel for window-only cross-spectral statistics."""
    j = cuda.grid(1)
    if j < starts.shape[0]:
        s = int(starts[j])
        cosw = math.cos(omega)
        sinw = math.sin(omega)
        coeff = 2.0 * cosw

        # X channel
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            v = x1[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        # Y channel
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            v = x2[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j] = r1 * r1 + i1 * i1
        yy[j] = r2 * r2 + i2 * i2
        xyr[j] = r1 * r2 + i1 * i2  # Re{X * conj(Y)}
        xyi[j] = i1 * r2 - r1 * i2  # Im{X * conj(Y)}


def _stats_win_only_csd_cuda(x1, x2, starts, L, w, omega):
    """
    CUDA host function for window-only cross-spectral statistics.

    Returns SciPy-compatible CSD sign: Pxy = X * conj(Y).
    """
    K = starts.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x1_d = cuda.to_device(np.ascontiguousarray(x1, dtype=np.float64))
    x2_d = cuda.to_device(np.ascontiguousarray(x2, dtype=np.float64))
    starts_d = cuda.to_device(np.ascontiguousarray(starts, dtype=np.int64))
    w_d = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))

    xx_d = cuda.device_array(K, dtype=np.float64)
    yy_d = cuda.device_array(K, dtype=np.float64)
    xyr_d = cuda.device_array(K, dtype=np.float64)
    xyi_d = cuda.device_array(K, dtype=np.float64)

    blocks = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _stats_win_only_csd_cuda_kernel[blocks, THREADS_PER_BLOCK](
        x1_d, x2_d, starts_d, L, w_d, omega, xx_d, yy_d, xyr_d, xyi_d
    )

    xx = xx_d.copy_to_host()
    yy = yy_d.copy_to_host()
    xyr = xyr_d.copy_to_host()
    xyi = xyi_d.copy_to_host()

    return _reduce_stats_nb(xx, yy, xyr, xyi)


# CUDA kernel: detrend0 (mean removal) auto-spectral
@cuda.jit
def _stats_detrend0_auto_cuda_kernel(x, starts, L, w, omega, xx, yy, xyr, xyi):
    """CUDA kernel for mean-removal auto-spectral statistics."""
    j = cuda.grid(1)
    if j < starts.shape[0]:
        s = int(starts[j])

        # Compute mean of segment
        m = 0.0
        for n in range(L):
            m += x[s + n]
        m = m / float(L)

        cosw = math.cos(omega)
        sinw = math.sin(omega)
        coeff = 2.0 * cosw

        # Goertzel on (x - m) * w
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            v = (x[s + n] - m) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0

        r = s1 - s2 * cosw
        i = s2 * sinw
        p = r * r + i * i

        xx[j] = p
        yy[j] = p
        xyr[j] = p
        xyi[j] = 0.0


def _stats_detrend0_auto_cuda(x, starts, L, w, omega):
    """
    CUDA host function for mean-removal auto-spectral statistics.
    """
    K = starts.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x_d = cuda.to_device(np.ascontiguousarray(x, dtype=np.float64))
    starts_d = cuda.to_device(np.ascontiguousarray(starts, dtype=np.int64))
    w_d = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))

    xx_d = cuda.device_array(K, dtype=np.float64)
    yy_d = cuda.device_array(K, dtype=np.float64)
    xyr_d = cuda.device_array(K, dtype=np.float64)
    xyi_d = cuda.device_array(K, dtype=np.float64)

    blocks = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _stats_detrend0_auto_cuda_kernel[blocks, THREADS_PER_BLOCK](
        x_d, starts_d, L, w_d, omega, xx_d, yy_d, xyr_d, xyi_d
    )

    xx = xx_d.copy_to_host()
    yy = yy_d.copy_to_host()
    xyr = xyr_d.copy_to_host()
    xyi = xyi_d.copy_to_host()

    return _reduce_stats_nb(xx, yy, xyr, xyi)


# CUDA kernel: detrend0 (mean removal) cross-spectral
@cuda.jit
def _stats_detrend0_csd_cuda_kernel(x1, x2, starts, L, w, omega, xx, yy, xyr, xyi):
    """CUDA kernel for mean-removal cross-spectral statistics."""
    j = cuda.grid(1)
    if j < starts.shape[0]:
        s = int(starts[j])

        # Compute means
        m1 = 0.0
        m2 = 0.0
        for n in range(L):
            m1 += x1[s + n]
            m2 += x2[s + n]
        m1 = m1 / float(L)
        m2 = m2 / float(L)

        cosw = math.cos(omega)
        sinw = math.sin(omega)
        coeff = 2.0 * cosw

        # X channel
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            v = (x1[s + n] - m1) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        # Y channel
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            v = (x2[s + n] - m2) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j] = r1 * r1 + i1 * i1
        yy[j] = r2 * r2 + i2 * i2
        xyr[j] = r1 * r2 + i1 * i2
        xyi[j] = i1 * r2 - r1 * i2


def _stats_detrend0_csd_cuda(x1, x2, starts, L, w, omega):
    """
    CUDA host function for mean-removal cross-spectral statistics.

    SciPy-compatible CSD sign.
    """
    K = starts.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x1_d = cuda.to_device(np.ascontiguousarray(x1, dtype=np.float64))
    x2_d = cuda.to_device(np.ascontiguousarray(x2, dtype=np.float64))
    starts_d = cuda.to_device(np.ascontiguousarray(starts, dtype=np.int64))
    w_d = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))

    xx_d = cuda.device_array(K, dtype=np.float64)
    yy_d = cuda.device_array(K, dtype=np.float64)
    xyr_d = cuda.device_array(K, dtype=np.float64)
    xyi_d = cuda.device_array(K, dtype=np.float64)

    blocks = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _stats_detrend0_csd_cuda_kernel[blocks, THREADS_PER_BLOCK](
        x1_d, x2_d, starts_d, L, w_d, omega, xx_d, yy_d, xyr_d, xyi_d
    )

    xx = xx_d.copy_to_host()
    yy = yy_d.copy_to_host()
    xyr = xyr_d.copy_to_host()
    xyi = xyi_d.copy_to_host()

    return _reduce_stats_nb(xx, yy, xyr, xyi)


# CUDA kernel: polynomial detrend auto-spectral
@cuda.jit
def _stats_poly_auto_cuda_kernel(x, starts, L, w, omega, Q, xx, yy, xyr, xyi):
    """CUDA kernel for polynomial detrend auto-spectral statistics."""
    j = cuda.grid(1)
    if j < starts.shape[0]:
        s = int(starts[j])

        # Compute alpha = Q.T @ x_seg
        p1 = Q.shape[1]
        alpha = cuda.local.array(3, dtype=types.float64)  # Max 3 for order=2
        for k in range(p1):
            acc = 0.0
            for n in range(L):
                acc += Q[n, k] * x[s + n]
            alpha[k] = acc

        cosw = math.cos(omega)
        sinw = math.sin(omega)
        coeff = 2.0 * cosw

        # Goertzel on (x - Q @ alpha) * w, streamed
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            # Compute Q[n, :] @ alpha
            qdot = 0.0
            for k in range(p1):
                qdot += Q[n, k] * alpha[k]
            v = (x[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0

        r = s1 - s2 * cosw
        i = s2 * sinw
        p = r * r + i * i

        xx[j] = p
        yy[j] = p
        xyr[j] = p
        xyi[j] = 0.0


def _stats_poly_auto_cuda(x, starts, L, w, omega, Q):
    """
    CUDA host function for polynomial detrend auto-spectral statistics.

    Q must be orthonormal with shape (L, p+1), p ∈ {1, 2}, from _build_Q.
    """
    K = starts.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x_d = cuda.to_device(np.ascontiguousarray(x, dtype=np.float64))
    starts_d = cuda.to_device(np.ascontiguousarray(starts, dtype=np.int64))
    w_d = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))
    Q_d = cuda.to_device(np.ascontiguousarray(Q, dtype=np.float64))

    xx_d = cuda.device_array(K, dtype=np.float64)
    yy_d = cuda.device_array(K, dtype=np.float64)
    xyr_d = cuda.device_array(K, dtype=np.float64)
    xyi_d = cuda.device_array(K, dtype=np.float64)

    blocks = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _stats_poly_auto_cuda_kernel[blocks, THREADS_PER_BLOCK](
        x_d, starts_d, L, w_d, omega, Q_d, xx_d, yy_d, xyr_d, xyi_d
    )

    xx = xx_d.copy_to_host()
    yy = yy_d.copy_to_host()
    xyr = xyr_d.copy_to_host()
    xyi = xyi_d.copy_to_host()

    return _reduce_stats_nb(xx, yy, xyr, xyi)


# CUDA kernel: polynomial detrend cross-spectral
@cuda.jit
def _stats_poly_csd_cuda_kernel(x1, x2, starts, L, w, omega, Q, xx, yy, xyr, xyi):
    """CUDA kernel for polynomial detrend cross-spectral statistics."""
    j = cuda.grid(1)
    if j < starts.shape[0]:
        s = int(starts[j])

        p1 = Q.shape[1]
        alpha1 = cuda.local.array(3, dtype=types.float64)
        alpha2 = cuda.local.array(3, dtype=types.float64)

        # Compute alpha1 = Q.T @ x1_seg
        for k in range(p1):
            acc = 0.0
            for n in range(L):
                acc += Q[n, k] * x1[s + n]
            alpha1[k] = acc

        # Compute alpha2 = Q.T @ x2_seg
        for k in range(p1):
            acc = 0.0
            for n in range(L):
                acc += Q[n, k] * x2[s + n]
            alpha2[k] = acc

        cosw = math.cos(omega)
        sinw = math.sin(omega)
        coeff = 2.0 * cosw

        # X channel
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            qdot = 0.0
            for k in range(p1):
                qdot += Q[n, k] * alpha1[k]
            v = (x1[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        # Y channel
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n in range(L):
            qdot = 0.0
            for k in range(p1):
                qdot += Q[n, k] * alpha2[k]
            v = (x2[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1
            s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j] = r1 * r1 + i1 * i1
        yy[j] = r2 * r2 + i2 * i2
        xyr[j] = r1 * r2 + i1 * i2
        xyi[j] = i1 * r2 - r1 * i2


def _stats_poly_csd_cuda(x1, x2, starts, L, w, omega, Q):
    """
    CUDA host function for polynomial detrend cross-spectral statistics.

    SciPy-compatible CSD sign; same Q is applied to both channels.
    """
    K = starts.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    x1_d = cuda.to_device(np.ascontiguousarray(x1, dtype=np.float64))
    x2_d = cuda.to_device(np.ascontiguousarray(x2, dtype=np.float64))
    starts_d = cuda.to_device(np.ascontiguousarray(starts, dtype=np.int64))
    w_d = cuda.to_device(np.ascontiguousarray(w, dtype=np.float64))
    Q_d = cuda.to_device(np.ascontiguousarray(Q, dtype=np.float64))

    xx_d = cuda.device_array(K, dtype=np.float64)
    yy_d = cuda.device_array(K, dtype=np.float64)
    xyr_d = cuda.device_array(K, dtype=np.float64)
    xyi_d = cuda.device_array(K, dtype=np.float64)

    blocks = (K + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    _stats_poly_csd_cuda_kernel[blocks, THREADS_PER_BLOCK](
        x1_d, x2_d, starts_d, L, w_d, omega, Q_d, xx_d, yy_d, xyr_d, xyi_d
    )

    xx = xx_d.copy_to_host()
    yy = yy_d.copy_to_host()
    xyr = xyr_d.copy_to_host()
    xyi = xyi_d.copy_to_host()

    return _reduce_stats_nb(xx, yy, xyr, xyi)

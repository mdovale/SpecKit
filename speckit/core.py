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
"""
core.py — high-performance single-bin spectral kernels with optional Numba
-----------------------------------------------------------------------------
Design notes
- SciPy-compatible CSD convention: Pxy = X * conj(Y)
  => Re{Pxy} = r1*r2 + i1*i2
     Im{Pxy} = i1*r2 - r1*i2
- Hot JIT kernels avoid per-segment temporaries by streaming windowing and
  detrending inside the Goertzel recurrence. Pure-NumPy fallbacks are provided.
- Returns follow the "sufficient statistics" 5-tuple:
    (MXX, MYY, mu_r, mu_i, M2)
  where M2 is the mean squared distance of XY from its mean across segments.
-----------------------------------------------------------------------------
"""
__all__ = [
    "_NUMBA_ENABLED",
    "BinStats",
    "_build_Q",
    # helpers
    "_goertzel_real_imag",
    "_reduce_stats",
    # jitted helpers
    "_reduce_stats_nb",
    "_apply_detrend0_inplace_nb_mean",
    "_apply_detrend0_inplace_nb_val",
    "_apply_poly_detrend_inplace_nb_alpha",
    "_apply_poly_detrend_inplace_nb_rowdot",
    # jitted kernels
    "_stats_win_only_auto",
    "_stats_win_only_csd",
    "_stats_detrend0_auto",
    "_stats_detrend0_csd",
    "_stats_poly_auto",
    "_stats_poly_csd",
    # numpy fallbacks
    "_stats_win_only_auto_np",
    "_stats_win_only_csd_np",
    "_stats_detrend0_auto_np",
    "_stats_detrend0_csd_np",
    "_stats_poly_auto_np",
    "_stats_poly_csd_np",
]

from typing import NamedTuple, Tuple
import numpy as np

# Optional Numba integration ---------------------------------------------------

try:
    import numba as _nb
    from numba import njit as _njit, prange as _prange
    _NUMBA_ENABLED = True
except Exception:  # pragma: no cover
    _NUMBA_ENABLED = False

    def _njit(*args, **kwargs):
        def _decorator(f):
            return f
        return _decorator

    def _prange(*args, **kwargs):
        return range(*args)


# Typed container for readability (not used inside nopython regions) ----------

class BinStats(NamedTuple):
    """
    Sufficient statistics for a single (frequency, set of segments).

    MXX : float
        Mean auto-power of X across segments.
    MYY : float
        Mean auto-power of Y across segments (for auto == MXX).
    mu_r : float
        Mean real{X * conj(Y)} across segments.
    mu_i : float
        Mean imag{X * conj(Y)} across segments.
    M2 : float
        Mean squared distance of XY from its mean across segments.
    """
    MXX: float
    MYY: float
    mu_r: float
    mu_i: float
    M2:  float


# Public helper: orthonormal polynomial detrend basis --------------------------

def _build_Q(L: int, order: int) -> np.ndarray:
    """
    Build an orthonormal polynomial detrend basis Q for segments of length L.

    Constructs a centered Vandermonde with t ∈ [-1, 1] and performs a reduced QR
    to obtain an orthonormal basis Q with p+1 columns, where order ∈ {1, 2}.

    Parameters
    ----------
    L : int
        Segment length.
    order : int
        Detrend order: 1 (constant+linear) or 2 (constant+linear+quadratic).

    Returns
    -------
    Q : (L, order+1) ndarray
        Orthonormal columns spanning the chosen polynomial subspace.

    Notes
    -----
    Use as: y_detr = y - Q @ (Q.T @ y).
    """
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2 for polynomial detrend")
    t = np.linspace(-1.0, 1.0, L, dtype=np.float64)
    if order == 1:
        V = np.stack([np.ones(L, dtype=np.float64), t], axis=1)
    else:
        V = np.stack([np.ones(L, dtype=np.float64), t, t * t], axis=1)
    # QR with economic mode gives orthonormal columns
    Q, _ = np.linalg.qr(V, mode="reduced")
    # ensure float64 contiguous
    return np.ascontiguousarray(Q, dtype=np.float64)


# Reference Goertzel (optional helper; not used in hot JIT code) ---------------

@_njit(cache=True, fastmath=True)
def _goertzel_real_imag(y: np.ndarray, cosw: float, sinw: float, coeff: float) -> Tuple[float, float]:
    """
    Goertzel recurrence for a single DFT bin on an in-memory segment y.

    Parameters
    ----------
    y : (L,) ndarray
        Segment samples (already windowed/detrended).
    cosw : float
        cos(omega).
    sinw : float
        sin(omega).
    coeff : float
        2*cos(omega).

    Returns
    -------
    r, i : float
        Real and imaginary parts of the bin value.
    """
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for n in range(y.shape[0]):
        s0 = y[n] + coeff * s1 - s2
        s2 = s1
        s1 = s0
    r = s1 - s2 * cosw
    i = s2 * sinw
    return r, i


# Pure-Python reducer (handy in fallbacks & tests) -----------------------------

def _reduce_stats(xx: np.ndarray, yy: np.ndarray, xyr: np.ndarray, xyi: np.ndarray) -> BinStats:
    """
    Reduce per-segment arrays into (MXX, MYY, mu_r, mu_i, M2).
    """
    K = xx.size
    if K == 0:
        return BinStats(0.0, 0.0, 0.0, 0.0, 0.0)
    MXX = float(xx.mean())
    MYY = float(yy.mean())
    mu_r = float(xyr.mean())
    mu_i = float(xyi.mean())
    if K >= 2:
        dr = xyr - mu_r
        di = xyi - mu_i
        M2 = float(np.mean(dr * dr + di * di))
    else:
        M2 = 0.0
    return BinStats(MXX, MYY, mu_r, mu_i, M2)


# Numba-compatible helpers -----------------------------------------------------

@_njit(cache=True, fastmath=True)
def _reduce_stats_nb(xx: np.ndarray, yy: np.ndarray, xyr: np.ndarray, xyi: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Numba version of _reduce_stats returning 5 floats (not a NamedTuple).
    """
    K = xx.shape[0]
    if K == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    MXX = np.mean(xx)
    MYY = np.mean(yy)
    mu_r = np.mean(xyr)
    mu_i = np.mean(xyi)
    if K >= 2:
        dr = xyr - mu_r
        di = xyi - mu_i
        M2 = np.mean(dr * dr + di * di)
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2


@_njit(cache=True, fastmath=True)
def _apply_detrend0_inplace_nb_mean(x: np.ndarray, s: int, L: int) -> float:
    """
    Compute per-segment mean for detrend0 (Numba-friendly).
    """
    m = 0.0
    for n in range(L):
        m += x[s + n]
    return m / float(L)


@_njit(cache=True, fastmath=True)
def _apply_detrend0_inplace_nb_val(xn: float, m: float) -> float:
    """
    Subtract mean value (scalar form for streaming).
    """
    return xn - m


@_njit(cache=True, fastmath=True)
def _apply_poly_detrend_inplace_nb_alpha(x: np.ndarray, Q: np.ndarray, s: int, L: int) -> np.ndarray:
    """
    Compute alpha = Q.T @ segment (Numba-friendly, tiny p ∈ {2,3}).
    """
    p1 = Q.shape[1]
    alpha = np.zeros(p1, np.float64)
    for k in range(p1):
        acc = 0.0
        for n in range(L):
            acc += Q[n, k] * x[s + n]
        alpha[k] = acc
    return alpha


@_njit(cache=True, fastmath=True)
def _apply_poly_detrend_inplace_nb_rowdot(Q: np.ndarray, n: int, alpha: np.ndarray) -> float:
    """
    Compute qdot = Q[n, :] @ alpha (scalar form for streaming).
    """
    p1 = Q.shape[1]
    acc = 0.0
    for k in range(p1):
        acc += Q[n, k] * alpha[k]
    return acc


# JIT kernels (streaming window/detrend; SciPy CSD sign) -----------------------

@_njit(parallel=True, fastmath=True, cache=True)
def _stats_win_only_auto(x, starts, L, w, omega):
    """
    Single-bin auto-spectral sufficient statistics using windowing only.

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
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment arrays kept (fast & simple); can switch to online if desired.
    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in _prange(K):
        s = int(starts[j])
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = x[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r = s1 - s2 * cosw
        i = s2 * sinw
        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    return _reduce_stats_nb(xx, yy, xyr, xyi)


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_win_only_csd(x1, x2, starts, L, w, omega):
    """
    Single-bin cross-spectral sufficient statistics using windowing only.

    Returns SciPy-compatible CSD sign: Pxy = X * conj(Y).
    """
    K = starts.shape[0]
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # X
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = x1[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        # Y
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = x2[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r1*r2 + i1*i2          # Re{X * conj(Y)}
        xyi[j] = i1*r2 - r1*i2          # Im{X * conj(Y)}

    return _reduce_stats_nb(xx, yy, xyr, xyi)


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_detrend0_auto(x, starts, L, w, omega):
    """
    Single-bin auto-spectral stats with mean removal (order=0 detrend).
    """
    K = starts.shape[0]
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # mean of segment
        m = _apply_detrend0_inplace_nb_mean(x, s, L)

        # Goertzel on (x - m) * w
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = _apply_detrend0_inplace_nb_val(x[s + n], m) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r = s1 - s2 * cosw
        i = s2 * sinw

        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    return _reduce_stats_nb(xx, yy, xyr, xyi)


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_detrend0_csd(x1, x2, starts, L, w, omega):
    """
    Single-bin cross-spectral stats with mean removal (order=0 detrend).

    SciPy-compatible CSD sign.
    """
    K = starts.shape[0]
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # means
        m1 = _apply_detrend0_inplace_nb_mean(x1, s, L)
        m2 = _apply_detrend0_inplace_nb_mean(x2, s, L)

        # X
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = _apply_detrend0_inplace_nb_val(x1[s + n], m1) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        # Y
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = _apply_detrend0_inplace_nb_val(x2[s + n], m2) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r1*r2 + i1*i2
        xyi[j] = i1*r2 - r1*i2

    return _reduce_stats_nb(xx, yy, xyr, xyi)


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_poly_auto(x, starts, L, w, omega, Q):
    """
    Single-bin auto-spectral stats with polynomial detrending via Q.

    Q must be orthonormal with shape (L, p+1), p ∈ {1, 2}, from _build_Q.
    """
    K = starts.shape[0]
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # alpha = Q.T @ x_seg  (tiny vector)
        alpha = _apply_poly_detrend_inplace_nb_alpha(x, Q, s, L)

        # Goertzel on (x - Q[:, :] @ alpha) * w, streamed
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            qdot = _apply_poly_detrend_inplace_nb_rowdot(Q, n, alpha)
            v = (x[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r = s1 - s2 * cosw
        i = s2 * sinw

        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    return _reduce_stats_nb(xx, yy, xyr, xyi)


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_poly_csd(x1, x2, starts, L, w, omega, Q):
    """
    Single-bin cross-spectral stats with polynomial detrending via Q.

    SciPy-compatible CSD sign; same Q is applied to both channels.
    """
    K = starts.shape[0]
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in _prange(K):
        s = int(starts[j])

        alpha1 = _apply_poly_detrend_inplace_nb_alpha(x1, Q, s, L)
        alpha2 = _apply_poly_detrend_inplace_nb_alpha(x2, Q, s, L)

        # X
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            qdot = _apply_poly_detrend_inplace_nb_rowdot(Q, n, alpha1)
            v = (x1[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        # Y
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            qdot = _apply_poly_detrend_inplace_nb_rowdot(Q, n, alpha2)
            v = (x2[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r1*r2 + i1*i2
        xyi[j] = i1*r2 - r1*i2

    return _reduce_stats_nb(xx, yy, xyr, xyi)


# Pure-NumPy fallbacks (readable, no Numba dependency) -------------------------

def _stats_win_only_auto_np(x, starts, L, w, omega):
    """
    NumPy fallback: window-only auto-spectral stats (matches JIT behavior).
    """
    K = starts.shape[0]
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in range(K):
        s = int(starts[j])
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = x[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r = s1 - s2 * cosw
        i = s2 * sinw
        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    bs = _reduce_stats(xx, yy, xyr, xyi)
    return bs.MXX, bs.MYY, bs.mu_r, bs.mu_i, bs.M2


def _stats_win_only_csd_np(x1, x2, starts, L, w, omega):
    """
    NumPy fallback: window-only cross-spectral stats (SciPy CSD sign).
    """
    K = starts.shape[0]
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in range(K):
        s = int(starts[j])

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = x1[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = x2[s + n] * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r1*r2 + i1*i2
        xyi[j] = i1*r2 - r1*i2

    bs = _reduce_stats(xx, yy, xyr, xyi)
    return bs.MXX, bs.MYY, bs.mu_r, bs.mu_i, bs.M2


def _stats_detrend0_auto_np(x, starts, L, w, omega):
    """
    NumPy fallback: detrend0 auto-spectral stats (matches JIT behavior).
    """
    K = starts.shape[0]
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in range(K):
        s = int(starts[j])
        m = float(np.mean(x[s:s+L]))

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = (x[s + n] - m) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r = s1 - s2 * cosw
        i = s2 * sinw

        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    bs = _reduce_stats(xx, yy, xyr, xyi)
    return bs.MXX, bs.MYY, bs.mu_r, bs.mu_i, bs.M2


def _stats_detrend0_csd_np(x1, x2, starts, L, w, omega):
    """
    NumPy fallback: detrend0 cross-spectral stats (SciPy CSD sign).
    """
    K = starts.shape[0]
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in range(K):
        s = int(starts[j])
        m1 = float(np.mean(x1[s:s+L]))
        m2 = float(np.mean(x2[s:s+L]))

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = (x1[s + n] - m1) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            v = (x2[s + n] - m2) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r1*r2 + i1*i2
        xyi[j] = i1*r2 - r1*i2

    bs = _reduce_stats(xx, yy, xyr, xyi)
    return bs.MXX, bs.MYY, bs.mu_r, bs.mu_i, bs.M2


def _stats_poly_auto_np(x, starts, L, w, omega, Q):
    """
    NumPy fallback: polynomial-detrended auto-spectral stats.
    """
    K = starts.shape[0]
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in range(K):
        s = int(starts[j])
        alpha = Q.T @ x[s:s+L]

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            qdot = float(Q[n, :].dot(alpha))
            v = (x[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r = s1 - s2 * cosw
        i = s2 * sinw

        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    bs = _reduce_stats(xx, yy, xyr, xyi)
    return bs.MXX, bs.MYY, bs.mu_r, bs.mu_i, bs.M2


def _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q):
    """
    NumPy fallback: polynomial-detrended cross-spectral stats (SciPy sign).
    """
    K = starts.shape[0]
    cosw = float(np.cos(omega))
    sinw = float(np.sin(omega))
    coeff = 2.0 * cosw

    xx  = np.empty(K, np.float64)
    yy  = np.empty(K, np.float64)
    xyr = np.empty(K, np.float64)
    xyi = np.empty(K, np.float64)

    for j in range(K):
        s = int(starts[j])
        alpha1 = Q.T @ x1[s:s+L]
        alpha2 = Q.T @ x2[s:s+L]

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            qdot = float(Q[n, :].dot(alpha1))
            v = (x1[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r1 = s1 - s2 * cosw
        i1 = s2 * sinw

        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            qdot = float(Q[n, :].dot(alpha2))
            v = (x2[s + n] - qdot) * w[n]
            s0 = v + coeff * s1 - s2
            s2 = s1; s1 = s0
        r2 = s1 - s2 * cosw
        i2 = s2 * sinw

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r1*r2 + i1*i2
        xyi[j] = i1*r2 - r1*i2

    bs = _reduce_stats(xx, yy, xyr, xyi)
    return bs.MXX, bs.MYY, bs.mu_r, bs.mu_i, bs.M2
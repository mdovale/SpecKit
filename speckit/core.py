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
from typing import NamedTuple
import numpy as np

# ---------- NUMBA SETUP ----------
try:
    import numba as _nb
    from numba import njit as _njit, prange as _prange
    _NUMBA_ENABLED = True
except Exception:  # pragma: no cover
    _NUMBA_ENABLED = False
    def _njit(*args, **kwargs):
        # Transparent pass-through when Numba isn't available
        def _decorator(f):
            return f
        return _decorator
    def _prange(*args, **kwargs):
        return range(*args)

__all__ = [
    "BinStats",
    "_NUMBA_ENABLED",
    "_njit",
    "_prange",
    "_apply_detrend0_inplace",
    "_apply_poly_detrend_inplace",
    "_goertzel_real_imag",
    "_reduce_stats",
]

class BinStats(NamedTuple):
    """
    Sufficient statistics for a single (frequency, segment-set) evaluation.

    Fields
    ------
    MXX : float
        Mean auto-power of channel X across segments.
    MYY : float
        Mean auto-power of channel Y across segments. (For auto kernels this
        equals MXX; it is kept for interface symmetry.)
    mu_r : float
        Mean real part of the cross term XY across segments.
    mu_i : float
        Mean imaginary part of the cross term XY across segments.
    M2 : float
        Mean squared magnitude deviation of XY about its mean:
        E[ (XY_r - mu_r)^2 + (XY_i - mu_i)^2 ].
    """
    MXX: float
    MYY: float
    mu_r: float
    mu_i: float
    M2:  float


def _apply_detrend0_inplace(y: np.ndarray) -> np.ndarray:
    """
    Subtract the per-segment mean (order=0 detrend) in-place.

    Parameters
    ----------
    y : ndarray, shape (L,), float64
        Segment samples. Modified in-place.

    Returns
    -------
    y : ndarray
        Same object, mean removed.
    """
    # In-place mean removal; keep float64 path
    y -= np.mean(y, dtype=np.float64)
    return y


@_njit(cache=True, fastmath=True)
def _apply_detrend0_inplace_nb(y):
    """
    Numba version of mean removal (order=0) in-place.
    """
    L = y.shape[0]
    m = 0.0
    for n in range(L):
        m += y[n]
    m /= L
    for n in range(L):
        y[n] -= m
    return y


def _apply_poly_detrend_inplace(y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """
    Project out polynomial components spanned by orthonormal Q (order=1/2).

    Uses y <- y - Q @ (Q.T @ y). Q is expected to be orthonormal (Q.T @ Q = I),
    typically produced by `_build_Q(L, order)`.

    Parameters
    ----------
    y : ndarray, shape (L,), float64
        Segment samples. Modified in-place.
    Q : ndarray, shape (L, p+1), float64
        Orthonormal basis for constant+linear (p=1) or +quadratic (p=2).

    Returns
    -------
    y : ndarray
        Same object, with polynomial trend removed.
    """
    # Compute coefficients in the orthonormal basis, then subtract projection
    alpha = Q.T @ y
    y -= Q @ alpha
    return y



@_njit(cache=True, fastmath=True)
def _apply_poly_detrend_inplace_nb(y, Q):
    """
    Numba version of polynomial detrend in-place using orthonormal Q.
    y <- y - Q @ (Q.T @ y)
    """
    # alpha = Q.T @ y  (shape: (p+1,))
    alpha = np.dot(Q.T, y)
    # y -= Q @ alpha
    y -= np.dot(Q, alpha)
    return y


@_njit(cache=True, fastmath=True)
def _goertzel_real_imag(y: np.ndarray,
                        cosw: float,
                        sinw: float,
                        coeff: float) -> tuple:
    """
    Goertzel recurrence for a single DFT bin; returns (real, imag).

    Parameters
    ----------
    y : ndarray, shape (L,), float64
        Windowed (and optionally detrended) segment samples.
    cosw : float
        cos(omega) for the target bin.
    sinw : float
        sin(omega) for the target bin.
    coeff : float
        2*cos(omega); precomputed to save multiplications.

    Returns
    -------
    r : float
        Real part of bin value.
    i : float
        Imaginary part of bin value.

    Notes
    -----
    The standard Goertzel recurrence:
        s0 = x[n] + coeff*s1 - s2
        s2 = s1
        s1 = s0
    Finalization:
        r = s1 - s2*cosw
        i = s2*sinw
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


def _reduce_stats(xx: np.ndarray,
                  yy: np.ndarray,
                  xyr: np.ndarray,
                  xyi: np.ndarray) -> BinStats:
    """
    Reduce per-segment arrays into the five sufficient statistics.

    Parameters
    ----------
    xx : ndarray, shape (K,), float64
        Per-segment |X|^2 values.
    yy : ndarray, shape (K,), float64
        Per-segment |Y|^2 values. For auto kernels this may equal `xx`.
    xyr : ndarray, shape (K,), float64
        Per-segment real{X * conj(Y)} values.
    xyi : ndarray, shape (K,), float64
        Per-segment imag{X * conj(Y)} values.

    Returns
    -------
    BinStats
        (MXX, MYY, mu_r, mu_i, M2)

    Notes
    -----
    M2 is the mean squared distance of (xyr, xyi) from its mean vector.
    For K < 2, M2 is set to 0.0 to avoid degenerate variance estimates.
    """
    K = float(xx.size)
    if K == 0.0:
        return BinStats(0.0, 0.0, 0.0, 0.0, 0.0)

    MXX = float(xx.mean())
    MYY = float(yy.mean())
    mu_r = float(xyr.mean())
    mu_i = float(xyi.mean())

    if xx.size >= 2:
        dr = xyr - mu_r
        di = xyi - mu_i
        M2 = float(np.mean(dr * dr + di * di))
    else:
        M2 = 0.0

    return BinStats(MXX, MYY, mu_r, mu_i, M2)


@_njit(cache=True, fastmath=True)
def _reduce_stats_nb(xx, yy, xyr, xyi):
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


# ---------- QR BASIS CACHE (for order=1,2 detrend) ----------
# We build a centered Vandermonde [1, t, t^2] with t in [-1,1] and QR-reduce to Q (orthonormal columns).
# Detrend: y <- y - Q @ (Q^T y). This is stable and O(L·order).

def _build_Q(L: int, order: int) -> np.ndarray:
    """
    Build an orthonormal polynomial detrend basis Q for segments of length L.

    Constructs a centered Vandermonde matrix with t ∈ [-1, 1] and performs a
    reduced QR to obtain an orthonormal basis Q with p+1 columns, where
    p == order ∈ {1, 2}. This basis can be used to project out constant/linear
    (order=1) or constant/linear/quadratic (order=2) trends via
    y_detr = y - Q @ (Q.T @ y).

    Parameters
    ----------
    L : int
        Segment length.
    order : int
        Polynomial detrend order. Supported values are 1 (constant+linear)
        or 2 (constant+linear+quadratic).

    Returns
    -------
    Q : ndarray of shape (L, order+1), dtype float64, C-contiguous
        Orthonormal columns spanning the chosen polynomial subspace.

    Notes
    -----
    - Use with: y_detr = y - Q @ (Q.T @ y).
    - Q has orthonormal columns (Q.T @ Q == I), improving numerical stability
      vs. fitting raw polynomials.
    """
    t = np.linspace(-1.0, 1.0, L, dtype=np.float64)
    if order == 1:
        V = np.stack([np.ones(L, dtype=np.float64), t], axis=1)              # (L,2)
    elif order == 2:
        V = np.stack([np.ones(L, dtype=np.float64), t, t*t], axis=1)         # (L,3)
    else:
        raise ValueError("Q requested for unsupported order")
    # Reduced QR; Q has orthonormal columns
    Q, _ = np.linalg.qr(V, mode="reduced")                                   # (L, p+1)
    return np.ascontiguousarray(Q, dtype=np.float64)


# ---------- IN-KERNEL STATS (AUTO / CSD) ----------
# All kernels compute: MXX, MYY, μ_XYr, μ_XYi, M2 across frames (population variance of complex XY).

@_njit(parallel=True, fastmath=True, cache=True)
def _stats_win_only_auto(x, starts, L, w, omega):
    """
    Single-bin auto-spectral sufficient statistics using windowing only.

    Parameters
    ----------
    x : ndarray, shape (N,), float64
        Input time series (single channel). Must index x[starts[j]:starts[j]+L].
    starts : ndarray, shape (K,), int64 or int32
        Start indices of each segment (non-negative, valid with L).
    L : int
        Segment length.
    w : ndarray, shape (L,), float64
        Window weights applied sample-wise inside each segment.
    omega : float
        Target digital radian frequency (radians/sample), possibly fractional.

    Returns
    -------
    MXX : float
        Mean auto-power of X across segments.
    MYY : float
        Identical to MXX for auto (returned for interface symmetry).
    mu_r : float
        Mean real part of XY (here X with itself), equals MXX for auto.
    mu_i : float
        Mean imaginary part of XY (zero for auto).
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.
    """
    K = starts.shape[0]

    # Precompute trig constants for the Goertzel tail
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment accumulators
    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # Window this segment (copy ensures contiguous scratch)
        y = x[s:s+L] * w

        # Single-bin complex value via Goertzel
        r, i = _goertzel_real_imag(y, cosw, sinw, coeff)

        # For auto, XY = X * conj(X) = |X|^2 (purely real)
        p = r*r + i*i
        xx[j]  = p              # |X|^2
        yy[j]  = p              # same as xx for symmetry
        xyr[j] = p              # Re{X * conj(X)} = |X|^2
        xyi[j] = 0.0            # Im{X * conj(X)} = 0

    MXX, MYY, mu_r, mu_i, M2 = _reduce_stats_nb(xx, yy, xyr, xyi)
    return MXX, MYY, mu_r, mu_i, M2


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_win_only_csd(x1, x2, starts, L, w, omega):
    """
    Single-bin cross-spectral sufficient statistics using windowing only.

    Parameters
    ----------
    x1 : ndarray, shape (N,), float64
        Channel-1 time series. Must index x1[starts[j]:starts[j]+L].
    x2 : ndarray, shape (N,), float64
        Channel-2 time series. Must index x2[starts[j]:starts[j]+L].
    starts : ndarray, shape (K,), int64 or int32
        Segment start indices (valid with L for both channels).
    L : int
        Segment length.
    w : ndarray, shape (L,), float64
        Window applied sample-wise to both channels.
    omega : float
        Target digital radian frequency (radians/sample), possibly fractional.

    Returns
    -------
    MXX : float
        Mean auto-power of X across segments.
    MYY : float
        Mean auto-power of Y across segments.
    mu_r : float
        Mean real part of the cross term XY across segments.
    mu_i : float
        Mean imaginary part of the cross term XY across segments.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.
    """
    K = starts.shape[0]

    # Precompute trig constants for the Goertzel tail
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment accumulators (shape-only arrays)
    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # Window each segment (copy into contiguous scratch to keep inner loop fast)
        y1 = x1[s:s+L] * w
        y2 = x2[s:s+L] * w

        # Single-bin complex values via Goertzel
        r1, i1 = _goertzel_real_imag(y1, cosw, sinw, coeff)
        r2, i2 = _goertzel_real_imag(y2, cosw, sinw, coeff)

        # Per-segment sufficient pieces
        xx[j]  = r1*r1 + i1*i1           # |X|^2
        yy[j]  = r2*r2 + i2*i2           # |Y|^2
        xyr[j] = r2*r1 + i2*i1           # Re{X * conj(Y)}
        xyi[j] = i2*r1 - r2*i1           # Im{X * conj(Y)}

    # Reduce to (MXX, MYY, mu_r, mu_i, M2)
    MXX, MYY, mu_r, mu_i, M2 = _reduce_stats_nb(xx, yy, xyr, xyi)
    return MXX, MYY, mu_r, mu_i, M2


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_detrend0_auto(x, starts, L, w, omega):
    """
    Single-bin auto-spectral stats with mean removal (order=0 detrend).

    Parameters
    ----------
    x : ndarray, shape (N,), float64
        Input time series (single channel).
    starts : ndarray, shape (K,), int64 or int32
        Segment start indices.
    L : int
        Segment length.
    w : ndarray, shape (L,), float64
        Window weights.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.

    Returns
    -------
    MXX : float
        Mean auto-power (returned also as MYY for interface symmetry).
    MYY : float
        Same as MXX.
    mu_r : float
        Mean real part of XY (here X with itself).
    mu_i : float
        Mean imaginary part of XY.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.
    """
    K = starts.shape[0]

    # Precompute trig constants for the Goertzel tail
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment accumulators
    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # Copy segment to scratch, detrend in-place, then window
        y = x[s:s+L].copy()
        _apply_detrend0_inplace_nb(y)
        y *= w

        # Single-bin complex value via Goertzel
        r, i = _goertzel_real_imag(y, cosw, sinw, coeff)

        # Auto case: XY = X * conj(X) = |X|^2 (purely real)
        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    MXX, MYY, mu_r, mu_i, M2 = _reduce_stats_nb(xx, yy, xyr, xyi)
    return MXX, MYY, mu_r, mu_i, M2


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_detrend0_csd(x1, x2, starts, L, w, omega):
    """
    Single-bin cross-spectral stats with mean removal (order=0 detrend).

    Parameters
    ----------
    x1 : ndarray, shape (N,), float64
        Channel-1 time series.
    x2 : ndarray, shape (N,), float64
        Channel-2 time series.
    starts : ndarray, shape (K,), int64 or int32
        Segment start indices.
    L : int
        Segment length.
    w : ndarray, shape (L,), float64
        Window weights applied to both channels.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.

    Returns
    -------
    MXX : float
        Mean auto-power of X across segments.
    MYY : float
        Mean auto-power of Y across segments.
    mu_r : float
        Mean real part of the cross term XY across segments.
    mu_i : float
        Mean imaginary part of the cross term XY across segments.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.
    """
    K = starts.shape[0]

    # Precompute trig constants for the Goertzel tail
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment accumulators
    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # Copy segments to scratch, detrend (mean removal), then window
        y1 = x1[s:s+L].copy()
        y2 = x2[s:s+L].copy()
        _apply_detrend0_inplace_nb(y1)
        _apply_detrend0_inplace_nb(y2)
        y1 *= w
        y2 *= w

        # Single-bin complex values via Goertzel
        r1, i1 = _goertzel_real_imag(y1, cosw, sinw, coeff)
        r2, i2 = _goertzel_real_imag(y2, cosw, sinw, coeff)

        # Per-segment sufficient pieces
        xx[j]  = r1*r1 + i1*i1           # |X|^2
        yy[j]  = r2*r2 + i2*i2           # |Y|^2
        xyr[j] = r2*r1 + i2*i1           # Re{X * conj(Y)}
        xyi[j] = i2*r1 - r2*i1           # Im{X * conj(Y)}

    MXX, MYY, mu_r, mu_i, M2 = _reduce_stats_nb(xx, yy, xyr, xyi)
    return MXX, MYY, mu_r, mu_i, M2


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_poly_auto(x, starts, L, w, omega, Q):
    """
    Single-bin auto-spectral stats with polynomial detrending via Q.

    Parameters
    ----------
    x : ndarray, shape (N,), float64
        Input time series (single channel).
    starts : ndarray, shape (K,), int64 or int32
        Segment start indices.
    L : int
        Segment length (must match Q.shape[0]).
    w : ndarray, shape (L,), float64
        Window weights.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.
    Q : ndarray, shape (L, p+1), float64
        Orthonormal polynomial basis (p ∈ {1, 2}) built by `_build_Q`.

    Returns
    -------
    MXX : float
        Mean auto-power of X across segments.
    MYY : float
        Same as MXX (returned for interface symmetry).
    mu_r : float
        Mean real part of XY (here X with itself).
    mu_i : float
        Mean imaginary part of XY (zero for auto).
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.
    """
    K = starts.shape[0]

    # Precompute trig constants for the Goertzel tail
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment accumulators
    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # Copy segment to scratch, polynomial-detrend in-place, then window
        y = x[s:s+L].copy()
        _apply_poly_detrend_inplace_nb(y, Q)
        y *= w

        # Single-bin complex value via Goertzel
        r, i = _goertzel_real_imag(y, cosw, sinw, coeff)

        # Auto case: XY = X * conj(X) = |X|^2 (purely real)
        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    MXX, MYY, mu_r, mu_i, M2 = _reduce_stats_nb(xx, yy, xyr, xyi)
    return MXX, MYY, mu_r, mu_i, M2


@_njit(parallel=True, fastmath=True, cache=True)
def _stats_poly_csd(x1, x2, starts, L, w, omega, Q):
    """
    Single-bin cross-spectral stats with polynomial detrending via Q.

    Parameters
    ----------
    x1 : ndarray, shape (N,), float64
        Channel-1 time series.
    x2 : ndarray, shape (N,), float64
        Channel-2 time series.
    starts : ndarray, shape (K,), int64 or int32
        Segment start indices.
    L : int
        Segment length (must match Q.shape[0]).
    w : ndarray, shape (L,), float64
        Window weights applied to both channels.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.
    Q : ndarray, shape (L, p+1), float64
        Orthonormal polynomial basis (p ∈ {1, 2}) built by `_build_Q`.

    Returns
    -------
    MXX : float
        Mean auto-power of X across segments.
    MYY : float
        Mean auto-power of Y across segments.
    mu_r : float
        Mean real part of the cross term XY across segments.
    mu_i : float
        Mean imaginary part of the cross term XY across segments.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.
    """
    K = starts.shape[0]

    # Precompute trig constants for the Goertzel tail
    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    # Per-segment accumulators
    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in _prange(K):
        s = int(starts[j])

        # Copy segments to scratch, polynomial-detrend, then window
        y1 = x1[s:s+L].copy()
        y2 = x2[s:s+L].copy()
        _apply_poly_detrend_inplace_nb(y1, Q)
        _apply_poly_detrend_inplace_nb(y2, Q)
        y1 *= w
        y2 *= w

        # Single-bin complex values via Goertzel
        r1, i1 = _goertzel_real_imag(y1, cosw, sinw, coeff)
        r2, i2 = _goertzel_real_imag(y2, cosw, sinw, coeff)

        # Per-segment sufficient pieces
        xx[j]  = r1*r1 + i1*i1           # |X|^2
        yy[j]  = r2*r2 + i2*i2           # |Y|^2
        xyr[j] = r2*r1 + i2*i1           # Re{X * conj(Y)}
        xyi[j] = i2*r1 - r2*i1           # Im{X * conj(Y)}

    MXX, MYY, mu_r, mu_i, M2 = _reduce_stats_nb(xx, yy, xyr, xyi)
    return MXX, MYY, mu_r, mu_i, M2


# ---------- NumPy fallbacks for poly paths (not used if Numba available) ----------
def _stats_poly_auto_np(x, starts, L, w, omega, Q):
    """
    NumPy fallback: auto-spectral stats with polynomial detrending via Q.
    Behavior matches _stats_poly_auto (Numba) exactly.
    """
    K = starts.shape[0]

    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in range(K):
        s = int(starts[j])
        y = x[s:s+L].copy()
        _apply_poly_detrend_inplace(y, Q)
        y *= w

        r, i = _goertzel_real_imag(y, cosw, sinw, coeff)

        p = r*r + i*i
        xx[j]  = p
        yy[j]  = p
        xyr[j] = p
        xyi[j] = 0.0

    stats = _reduce_stats(xx, yy, xyr, xyi)
    return stats.MXX, stats.MYY, stats.mu_r, stats.mu_i, stats.M2


def _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q):
    """
    NumPy fallback: cross-spectral stats with polynomial detrending via Q.
    Behavior matches _stats_poly_csd (Numba) exactly.
    """
    K = starts.shape[0]

    cosw = np.cos(omega)
    sinw = np.sin(omega)
    coeff = 2.0 * cosw

    xx  = np.empty(K, dtype=np.float64)
    yy  = np.empty(K, dtype=np.float64)
    xyr = np.empty(K, dtype=np.float64)
    xyi = np.empty(K, dtype=np.float64)

    for j in range(K):
        s = int(starts[j])

        y1 = x1[s:s+L].copy()
        y2 = x2[s:s+L].copy()
        _apply_poly_detrend_inplace(y1, Q)
        _apply_poly_detrend_inplace(y2, Q)
        y1 *= w
        y2 *= w

        r1, i1 = _goertzel_real_imag(y1, cosw, sinw, coeff)
        r2, i2 = _goertzel_real_imag(y2, cosw, sinw, coeff)

        xx[j]  = r1*r1 + i1*i1
        yy[j]  = r2*r2 + i2*i2
        xyr[j] = r2*r1 + i2*i1
        xyi[j] = i2*r1 - r2*i1

    stats = _reduce_stats(xx, yy, xyr, xyi)
    return stats.MXX, stats.MYY, stats.mu_r, stats.mu_i, stats.M2
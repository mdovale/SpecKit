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
import numpy as np

# ---------- NUMBA SETUP ----------
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        def deco(f): 
            return f
        return deco
    def prange(n): 
        return range(n)

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

@njit(parallel=True, fastmath=True, cache=True)
def _stats_win_only_auto(x, starts, L, w, omega):
    """
    Single-bin auto-spectral sufficient statistics using windowing only.

    Computes, for one target DFT bin (at digital radian frequency `omega`),
    the windowed Goertzel response per overlapping segment and returns
    sufficient statistics across all segments: mean auto-power (MXX == MYY),
    mean complex cross term (μ_XY == μ of X with itself), and second moment M2.

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
        Target digital radian frequency (radians/sample) for the single-bin
        Goertzel evaluation. Can be fractional (not restricted to 2π m/L).

    Returns
    -------
    MXX : float
        Mean auto-power of channel X across segments.
    MYY : float
        Identical to MXX for auto (returned for interface symmetry).
    mu_r : float
        Mean real part of the complex XY term (here X with itself).
    mu_i : float
        Mean imaginary part of the complex XY term.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments,
        i.e. E[ |XY - μ|^2 ].

    Notes
    -----
    - Uses Goertzel recurrence with precomputed cos/sin(omega).
    - Does not perform detrending; only windowing is applied.
    - Returns are sufficient statistics consumed by higher-level estimators.
    """
    navg = starts.size
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    sum_XX = 0.0; sum_YY = 0.0; sum_XYr = 0.0; sum_XYi = 0.0
    XYr_tmp = np.empty(navg, np.float64); XYi_tmp = np.empty(navg, np.float64)
    for j in prange(navg):
        base = starts[j]
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = x[base + n] * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        rx = s1 - cosw * s2; ix = sinw * s2
        ry = rx; iy = ix  # auto
        # XYr = ry*rx + iy*ix; XYi = iy*rx - ry*ix
        XYr = rx*ry + ix*iy; XYi = ix*ry - rx*iy
        XX = rx*rx + ix*ix; YY = ry*ry + iy*iy
        XYr_tmp[j] = XYr; XYi_tmp[j] = XYi
        sum_XYr += XYr; sum_XYi += XYi; sum_XX += XX; sum_YY += YY
    inv = 1.0 / navg
    mu_r = sum_XYr*inv; mu_i = sum_XYi*inv; MXX = sum_XX*inv; MYY = sum_YY*inv
    acc = 0.0
    if navg > 1:
        for j in prange(navg):
            dr = XYr_tmp[j]-mu_r; di = XYi_tmp[j]-mu_i
            acc += dr*dr + di*di
        M2 = acc*inv
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

@njit(parallel=True, fastmath=True, cache=True)
def _stats_win_only_csd(x1, x2, starts, L, w, omega):
    """
    Single-bin cross-spectral sufficient statistics using windowing only.

    For one target DFT bin at `omega`, computes Goertzel responses of two
    channels (X, Y) over overlapping segments and returns sufficient statistics:
    mean auto-powers (MXX, MYY), mean complex cross term (μ_XY = μ_r + i μ_i),
    and second moment M2 of XY across segments.

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

    Notes
    -----
    - Uses windowing only (no detrending).
    - XY is formed from the complex bin values of X and Y via Goertzel.
    """
    navg = starts.size
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    sum_XX = 0.0; sum_YY = 0.0; sum_XYr = 0.0; sum_XYi = 0.0
    XYr_tmp = np.empty(navg, np.float64); XYi_tmp = np.empty(navg, np.float64)
    for j in prange(navg):
        base = starts[j]
        # ch1
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = x1[base + n] * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        rx = s1 - cosw * s2; ix = sinw * s2
        # ch2
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = x2[base + n] * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        ry = s1 - cosw * s2; iy = sinw * s2
        # XYr = ry*rx + iy*ix; XYi = iy*rx - ry*ix
        XYr = rx*ry + ix*iy; XYi = ix*ry - rx*iy
        XX = rx*rx + ix*ix; YY = ry*ry + iy*iy
        XYr_tmp[j] = XYr; XYi_tmp[j] = XYi
        sum_XYr += XYr; sum_XYi += XYi; sum_XX += XX; sum_YY += YY
    inv = 1.0 / navg
    mu_r = sum_XYr*inv; mu_i = sum_XYi*inv; MXX = sum_XX*inv; MYY = sum_YY*inv
    acc = 0.0
    if navg > 1:
        for j in prange(navg):
            dr = XYr_tmp[j]-mu_r; di = XYi_tmp[j]-mu_i
            acc += dr*dr + di*di
        M2 = acc*inv
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

@njit(parallel=True, fastmath=True, cache=True)
def _stats_detrend0_auto(x, starts, L, w, omega):
    """
    Single-bin auto-spectral stats with mean removal (order=0 detrend).

    As `_stats_win_only_auto`, but first de-mean each segment before applying
    the window and Goertzel recurrence.

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

    Notes
    -----
    - Mean is computed per-segment as sum(y)/L, then removed.
    - Useful to suppress DC leakage prior to windowing.
    """
    navg = starts.size
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    sum_XX = 0.0; sum_YY = 0.0; sum_XYr = 0.0; sum_XYi = 0.0
    XYr_tmp = np.empty(navg, np.float64); XYi_tmp = np.empty(navg, np.float64)
    for j in prange(navg):
        base = starts[j]
        # mean
        s = 0.0
        for n in range(L): s += x[base+n]
        mu = s / L
        # Goertzel on demeaned * windowed
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = (x[base+n] - mu) * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        rx = s1 - cosw * s2; ix = sinw * s2
        ry = rx; iy = ix
        # XYr = ry*rx + iy*ix; XYi = iy*rx - ry*ix
        XYr = rx*ry + ix*iy; XYi = ix*ry - rx*iy
        XX = rx*rx + ix*ix; YY = ry*ry + iy*iy
        XYr_tmp[j] = XYr; XYi_tmp[j] = XYi
        sum_XYr += XYr; sum_XYi += XYi; sum_XX += XX; sum_YY += YY
    inv = 1.0 / navg
    mu_r = sum_XYr*inv; mu_i = sum_XYi*inv; MXX = sum_XX*inv; MYY = sum_YY*inv
    acc = 0.0
    if navg > 1:
        for j in prange(navg):
            dr = XYr_tmp[j]-mu_r; di = XYi_tmp[j]-mu_i
            acc += dr*dr + di*di
        M2 = acc*inv
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

@njit(parallel=True, fastmath=True, cache=True)
def _stats_detrend0_csd(x1, x2, starts, L, w, omega):
    """
    Single-bin cross-spectral stats with mean removal (order=0 detrend).

    As `_stats_win_only_csd`, but removes the per-segment mean from each
    channel independently before windowing and Goertzel.

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
        Window weights.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.

    Returns
    -------
    MXX : float
        Mean auto-power of X.
    MYY : float
        Mean auto-power of Y.
    mu_r : float
        Mean real part of the cross term XY.
    mu_i : float
        Mean imaginary part of the cross term XY.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.

    Notes
    -----
    - Mean removal is performed separately on x1 and x2 in each segment.
    - Helps reduce low-frequency leakage in cross-spectral estimates.
    """
    navg = starts.size
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    sum_XX = 0.0; sum_YY = 0.0; sum_XYr = 0.0; sum_XYi = 0.0
    XYr_tmp = np.empty(navg, np.float64); XYi_tmp = np.empty(navg, np.float64)
    for j in prange(navg):
        base = starts[j]
        # ch1 mean
        s = 0.0
        for n in range(L): s += x1[base+n]
        mu1 = s / L
        # ch2 mean
        s = 0.0
        for n in range(L): s += x2[base+n]
        mu2 = s / L
        # ch1 Goertzel
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = (x1[base+n] - mu1) * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        rx = s1 - cosw * s2; ix = sinw * s2
        # ch2 Goertzel
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = (x2[base+n] - mu2) * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        ry = s1 - cosw * s2; iy = sinw * s2
        # XYr = ry*rx + iy*ix; XYi = iy*rx - ry*ix
        XYr = rx*ry + ix*iy; XYi = ix*ry - rx*iy
        XX = rx*rx + ix*ix; YY = ry*ry + iy*iy
        XYr_tmp[j] = XYr; XYi_tmp[j] = XYi
        sum_XYr += XYr; sum_XYi += XYi; sum_XX += XX; sum_YY += YY
    inv = 1.0 / navg
    mu_r = sum_XYr*inv; mu_i = sum_XYi*inv; MXX = sum_XX*inv; MYY = sum_YY*inv
    acc = 0.0
    if navg > 1:
        for j in prange(navg):
            dr = XYr_tmp[j]-mu_r; di = XYi_tmp[j]-mu_i
            acc += dr*dr + di*di
        M2 = acc*inv
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

@njit(parallel=True, fastmath=True, cache=True)
def _stats_poly_auto(x, starts, L, w, omega, Q):
    """
    Single-bin auto-spectral stats with polynomial detrending via Q.

    Projects each segment onto the orthonormal basis Q (built by `_build_Q`)
    and subtracts the projection (constant+linear for Q with 2 cols, or
    constant+linear+quadratic for Q with 3 cols). Then applies windowing and
    Goertzel to compute sufficient statistics across segments.

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
        Orthonormal polynomial basis with p ∈ {1, 2}. Typically produced by
        `_build_Q(L, order)`.

    Returns
    -------
    MXX : float
        Mean auto-power of X (returned also as MYY for interface symmetry).
    MYY : float
        Same as MXX.
    mu_r : float
        Mean real part of XY (X with itself).
    mu_i : float
        Mean imaginary part of XY.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.

    Notes
    -----
    - Detrending uses y_detr = y - Q @ (Q.T @ y) with Q orthonormal.
    - Q.shape[1] determines detrend order: 2 → order=1, 3 → order=2.
    """
    navg = starts.size
    p1 = Q.shape[1]  # p+1
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    sum_XX = 0.0; sum_YY = 0.0; sum_XYr = 0.0; sum_XYi = 0.0
    XYr_tmp = np.empty(navg, np.float64); XYi_tmp = np.empty(navg, np.float64)
    for j in prange(navg):
        base = starts[j]
        # alpha = Q^T y
        alpha0 = 0.0; alpha1 = 0.0; alpha2 = 0.0
        # limited to p<=2 for speed; handle p1 ∈ {2,3}
        if p1 == 2:
            # columns Q[:,0], Q[:,1]
            for n in range(L):
                y = x[base+n]
                alpha0 += Q[n,0] * y
                alpha1 += Q[n,1] * y
        else:
            for n in range(L):
                y = x[base+n]
                alpha0 += Q[n,0] * y
                alpha1 += Q[n,1] * y
                alpha2 += Q[n,2] * y
        # y' = y - Q @ alpha ; then window + goertzel
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        if p1 == 2:
            for n in range(L):
                y = x[base+n] - (Q[n,0]*alpha0 + Q[n,1]*alpha1)
                xn = y * w[n]
                s0 = xn + coeff * s1 - s2
                s2 = s1; s1 = s0
        else:
            for n in range(L):
                y = x[base+n] - (Q[n,0]*alpha0 + Q[n,1]*alpha1 + Q[n,2]*alpha2)
                xn = y * w[n]
                s0 = xn + coeff * s1 - s2
                s2 = s1; s1 = s0
        rx = s1 - cosw * s2; ix = sinw * s2
        ry = rx; iy = ix
        # XYr = ry*rx + iy*ix; XYi = iy*rx - ry*ix
        XYr = rx*ry + ix*iy; XYi = ix*ry - rx*iy
        XX = rx*rx + ix*ix; YY = ry*ry + iy*iy
        XYr_tmp[j] = XYr; XYi_tmp[j] = XYi
        sum_XYr += XYr; sum_XYi += XYi; sum_XX += XX; sum_YY += YY
    inv = 1.0 / navg
    mu_r = sum_XYr*inv; mu_i = sum_XYi*inv; MXX = sum_XX*inv; MYY = sum_YY*inv
    acc = 0.0
    if navg > 1:
        for j in prange(navg):
            dr = XYr_tmp[j]-mu_r; di = XYi_tmp[j]-mu_i
            acc += dr*dr + di*di
        M2 = acc*inv
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

@njit(parallel=True, fastmath=True, cache=True)
def _stats_poly_csd(x1, x2, starts, L, w, omega, Q):
    """
    Single-bin cross-spectral stats with polynomial detrending via Q.

    For each segment and each channel, projects onto Q and subtracts the
    projection to remove constant/linear (or quadratic) trends. Then applies
    windowing and Goertzel to compute sufficient statistics for the cross term.

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
        Window weights.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.
    Q : ndarray, shape (L, p+1), float64
        Orthonormal polynomial basis (p ∈ {1, 2}) from `_build_Q`.

    Returns
    -------
    MXX : float
        Mean auto-power of X.
    MYY : float
        Mean auto-power of Y.
    mu_r : float
        Mean real part of the cross term XY.
    mu_i : float
        Mean imaginary part of the cross term XY.
    M2 : float
        Mean squared magnitude deviation of XY about its mean across segments.

    Notes
    -----
    - The same Q is applied to both channels (shape must match L).
    - Detrending is numerically stable due to orthonormal Q.
    """
    navg = starts.size
    p1 = Q.shape[1]
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    sum_XX = 0.0; sum_YY = 0.0; sum_XYr = 0.0; sum_XYi = 0.0
    XYr_tmp = np.empty(navg, np.float64); XYi_tmp = np.empty(navg, np.float64)
    for j in prange(navg):
        base = starts[j]
        # alpha for ch1
        a10 = 0.0; a11 = 0.0; a12 = 0.0
        # alpha for ch2
        a20 = 0.0; a21 = 0.0; a22 = 0.0
        if p1 == 2:
            for n in range(L):
                y1 = x1[base+n]; y2 = x2[base+n]
                a10 += Q[n,0]*y1; a11 += Q[n,1]*y1
                a20 += Q[n,0]*y2; a21 += Q[n,1]*y2
        else:
            for n in range(L):
                y1 = x1[base+n]; y2 = x2[base+n]
                a10 += Q[n,0]*y1; a11 += Q[n,1]*y1; a12 += Q[n,2]*y1
                a20 += Q[n,0]*y2; a21 += Q[n,1]*y2; a22 += Q[n,2]*y2
        # ch1 goertzel on (y - Q@alpha) * w
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        if p1 == 2:
            for n in range(L):
                y = x1[base+n] - (Q[n,0]*a10 + Q[n,1]*a11)
                xn = y * w[n]
                s0 = xn + coeff * s1 - s2
                s2 = s1; s1 = s0
        else:
            for n in range(L):
                y = x1[base+n] - (Q[n,0]*a10 + Q[n,1]*a11 + Q[n,2]*a12)
                xn = y * w[n]
                s0 = xn + coeff * s1 - s2
                s2 = s1; s1 = s0
        rx = s1 - cosw * s2; ix = sinw * s2
        # ch2
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        if p1 == 2:
            for n in range(L):
                y = x2[base+n] - (Q[n,0]*a20 + Q[n,1]*a21)
                xn = y * w[n]
                s0 = xn + coeff * s1 - s2
                s2 = s1; s1 = s0
        else:
            for n in range(L):
                y = x2[base+n] - (Q[n,0]*a20 + Q[n,1]*a21 + Q[n,2]*a22)
                xn = y * w[n]
                s0 = xn + coeff * s1 - s2
                s2 = s1; s1 = s0
        ry = s1 - cosw * s2; iy = sinw * s2
        # XYr = ry*rx + iy*ix; XYi = iy*rx - ry*ix
        XYr = rx*ry + ix*iy; XYi = ix*ry - rx*iy
        XX = rx*rx + ix*ix; YY = ry*ry + iy*iy
        XYr_tmp[j] = XYr; XYi_tmp[j] = XYi
        sum_XYr += XYr; sum_XYi += XYi; sum_XX += XX; sum_YY += YY
    inv = 1.0 / navg
    mu_r = sum_XYr*inv; mu_i = sum_XYi*inv; MXX = sum_XX*inv; MYY = sum_YY*inv
    acc = 0.0
    if navg > 1:
        for j in prange(navg):
            dr = XYr_tmp[j]-mu_r; di = XYi_tmp[j]-mu_i
            acc += dr*dr + di*di
        M2 = acc*inv
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

# ---------- NumPy fallbacks for poly paths (not used if Numba available) ----------
def _stats_poly_auto_np(x, starts, L, w, omega, Q):
    """
    NumPy fallback: auto-spectral stats with polynomial detrending via Q.

    Equivalent to `_stats_poly_auto`, implemented in pure NumPy without Numba.
    Useful when Numba is unavailable; API and returns are identical.

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
        Orthonormal polynomial basis.

    Returns
    -------
    MXX, MYY, mu_r, mu_i, M2 : floats
        Sufficient statistics across segments; see `_stats_poly_auto`.

    Notes
    -----
    - Uses matrix-vector products (Q.T @ y) and (Q @ alpha) per segment, then
      Goertzel recurrence for the target bin.
    """
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    navg = starts.size
    XYr = np.empty(navg); XYi = np.empty(navg); XX = np.empty(navg); YY = np.empty(navg)
    for j, base in enumerate(starts):
        y = x[base:base+L].astype(np.float64, copy=False)
        alpha = Q.T @ y
        y = y - Q @ alpha
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = y[n] * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        rx = s1 - cosw*s2; ix = sinw*s2
        ry = rx; iy = ix
        # XYr[j] = ry*rx + iy*ix
        # XYi[j] = iy*rx - ry*ix
        XYr[j] = rx*ry + ix*iy
        XYi[j] = ix*ry - rx*iy
        XX[j]  = rx*rx + ix*ix
        YY[j]  = ry*ry + iy*iy
    MXX = float(XX.mean()); MYY = float(YY.mean())
    mu_r = float(XYr.mean()); mu_i = float(XYi.mean())
    if navg > 1:
        M2 = float(np.mean((XYr-mu_r)**2 + (XYi-mu_i)**2))
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2

def _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q):
    """
    NumPy fallback: cross-spectral stats with polynomial detrending via Q.

    Equivalent to `_stats_poly_csd`, implemented in pure NumPy without Numba.
    Detrends both channels with the same Q, applies windowing, runs Goertzel,
    and returns sufficient statistics across segments.

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
        Window weights.
    omega : float
        Digital radian frequency (radians/sample), possibly fractional.
    Q : ndarray, shape (L, p+1), float64
        Orthonormal polynomial basis.

    Returns
    -------
    MXX, MYY, mu_r, mu_i, M2 : floats
        Sufficient statistics across segments; see `_stats_poly_csd`.

    Notes
    -----
    - Mirrors the Numba version's math and interface for drop-in use.
    """
    cosw = np.cos(omega); sinw = np.sin(omega); coeff = 2.0 * cosw
    navg = starts.size
    XYr = np.empty(navg); XYi = np.empty(navg); XX = np.empty(navg); YY = np.empty(navg)
    for j, base in enumerate(starts):
        y1 = x1[base:base+L].astype(np.float64, copy=False)
        y2 = x2[base:base+L].astype(np.float64, copy=False)
        a1 = Q.T @ y1; a2 = Q.T @ y2
        y1 = y1 - Q @ a1; y2 = y2 - Q @ a2
        # ch1
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = y1[n] * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        rx = s1 - cosw*s2; ix = sinw*s2
        # ch2
        s0 = 0.0; s1 = 0.0; s2 = 0.0
        for n in range(L):
            xn = y2[n] * w[n]
            s0 = xn + coeff * s1 - s2
            s2 = s1; s1 = s0
        ry = s1 - cosw*s2; iy = sinw*s2
        XYr[j] = ry*rx + iy*ix
        XYi[j] = iy*rx - ry*ix
        # XYr[j] = rx*ry + ix*iy
        # XYi[j] = ix*ry - rx*iy
        XX[j]  = rx*rx + ix*ix
        YY[j]  = ry*ry + iy*iy
    MXX = float(XX.mean()); MYY = float(YY.mean())
    mu_r = float(XYr.mean()); mu_i = float(XYi.mean())
    if navg > 1:
        M2 = float(np.mean((XYr-mu_r)**2 + (XYi-mu_i)**2))
    else:
        M2 = 0.0
    return MXX, MYY, mu_r, mu_i, M2
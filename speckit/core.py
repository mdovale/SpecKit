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
    # order in {1,2}; returns Q with shape (L, order+1)
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
    # order is inferred from Q.shape[1]-1
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
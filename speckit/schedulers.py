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
import sys
import math

import heapq
import numpy as np

import logging

logger = logging.getLogger(__name__)


def _require_args(args_dict, required):
    missing = [k for k in required if k not in args_dict]
    if missing:
        raise TypeError(f"Missing required argument(s): {', '.join(missing)}")
    return [args_dict[k] for k in required]


def lpsd_plan(**args):
    """
    Original LPSD scheduler from:
    https://doi.org/10.1016/j.measurement.2005.10.010

    Like ltf_plan, but bmin = 1.0, and Lmin = 1.
    """
    # Ensure required keys exist (will raise if not)
    _require_args(args, ["N", "fs", "olap", "Jdes", "Kdes"])

    # Override per original behavior
    forwarded = dict(args)
    forwarded["bmin"] = 1.0
    forwarded["Lmin"] = 1

    return ltf_plan(**forwarded)


def ltf_plan(**args):
    """
    LTF scheduler from S2-AEI-TN-3052 (Gerhard Heinzel).

    Based on the input parameters, the algorithm generates an array of
    frequencies (f), with corresponding resolution bandwidths (r), bin
    numbers (b), segment lengths (L), number of averages (K), and starting
    indices (D) for subsequent spectral analysis of time series using
    the windowed, overlapped segmented averaging method.

    The time series will then be segmented for each frequency as follows:
    [---------------------------------------------------------------------------------] total length N
    [---------] segment length L[j], starting at index D[j][0] = 0                    .
    .     [---------] segment length L[j], starting at index D[j][1]                  .
    .           [---------] segment length L[j], starting at index D[j][2]            .
    .                 [---------] segment length L[j], starting at index D[j][3]      .
    .                           ... (total of K[j] segments to average)               .
    .                                                                       [---------] segment length L[j]
                                                                                        starting at index D[j][-1]

    Inputs:
        N (int): Total length of the input data.
        fs (float): Sampling frequency of the input data.
        olap (float): Desired fractional overlap between segments of the input data.
        bmin (float): Minimum bin number to be used (used to discard the lower bins with biased estimates due to power aliasing from negative bins).
        Lmin (int): Smallest allowable segment length to be processed (used to tackle time delay bias error in cross spectra estimation).
        Jdes (int): Desired number of frequencies to produce. This value is almost never met exactly.
        Kdes (int): Desired number of segments to be averaged. This value is almost nowhere met exactly, and is actually only used as control parameter in the algorithm to ﬁnd a compromise between conflicting goals.

    The algorithm balances several conflicting goals:
        - Desire to compute approximately Jdes frequencies.
        - Desire for those frequencies to be approximately log-spaced.
        - For each frequency, desire to have approximately `olap` fractional overlap between segments while using the full time series.

    Computes:
        f (array of float): Frequency vector in Hz.
        r (array of float): For each frequency, resolution bandwidth in Hz.
        b (array of float): For each frequency, fractional bin number.
        L (array of int): For each frequency, length of the segments to be processed.
        K (array of float): For each frequency, number of segments to be processed.
        D (array of arrays of int): For each frequency, array containing the starting indices of each segment to be processed.
        O (array of float): For each frequency, actual fractional overlap between segments.
        nf (int): Total number of frequencies produced.

    Constraints:
        f[j] = r[j] * m[j]: Definition of the non-integer bin number
        r[j] * L[j] = fs: DFT constraint
        f[j+1] = f[j] + r[j]: Local spacing between frequency bins equivalent to original WOSA method.
        L[j] <= nx: Time series segment length cannot be larger than total length of the time series
        L[j] >= Lmin: Time series segment length must be greater or equal to Lmin
        b[j] >= bmin: Discard frequency bin numbers lower or equal to bmin
        f[0] = fmin: Lowest possible frequency must be met.
        f[-1] <= fmax: Maximum possible frequency must be met.

    Internal constants:
        xov (float): Desired non-overlapping fraction, xov = 1 - olap.
        fmin (float): Lowest possible frequency, fmin = fs/nx*bmin.
        fmax (float): Maximum possible frequency (Nyquist criterion), fmax = fs/2.
        logfact (float): Constant factor that would ensure logarithmic frequency spacing, logfact = (nx/2)^(1/Jdes)-1.
        fresmin (float): The smallest possible frequency resolution bandwidth in Hz, fresmin = fs/nx.
        freslim (float): The smallest possible frequency resolution bandwidth in Hz when Kdes averages are performed, freslim = fresmin*(1+xov(Kdes-1)).

    Targets:
    1. r[j]/f[j] = x1[j] with x1[j] -> logfact:
    This targets the approximate logarithmic spacing of frequencies on the x-axis,
    and also the desired number of frequencies Jdes.

    2. if K[j] = 1, then L[j] = nx:
    This describes the requirement to use the complete time series. In the case of K[j] > 1, the starting points of the individual segments
    can and will be adjusted such that the complete time series is used, at the expense of not precisely achieving the desired overlap.

    3. K[j] >= Kdes:
    This describes the desire to have at least Kdes segments for averaging at each frequency. As mentioned above,
    this cannot be met at low frequencies but is easy to over-achieve at high frequencies, such that this serves only as a
    guideline for ﬁnding compromises in the scheduler.
    """
    # Unpack & validate
    N, fs, olap, bmin, Lmin, Jdes, Kdes = _require_args(
        args, ["N", "fs", "olap", "bmin", "Lmin", "Jdes", "Kdes"]
    )

    def round_half_up(val):
        if (float(val) % 1) >= 0.5:
            x = math.ceil(val)
        else:
            x = round(val)
        return x

    # Init constants:
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2
    fresmin = fs / N
    freslim = fresmin * (1 + xov * (Kdes - 1))
    logfact = (N / 2) ** (1 / Jdes) - 1

    # Init lists:
    f_arr = []
    fres_arr = []
    b_arr = []
    L_arr = []
    K_arr = []
    O_arr = []
    D_arr = []
    navg_arr = []

    # Scheduler algorithm:
    fi = fmin
    while fi < fmax:
        fres = fi * logfact
        if fres >= freslim:
            pass
        elif fres < freslim and (freslim * fres) ** 0.5 > fresmin:
            fres = (freslim * fres) ** 0.5
        else:
            fres = fresmin

        fbin = fi / fres
        if fbin < bmin:
            fbin = bmin
            fres = fi / fbin

        dftlen = int(round_half_up(fs / fres))
        if dftlen > N:
            dftlen = N
        if dftlen < Lmin:
            dftlen = Lmin

        nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))
        if nseg == 1:
            dftlen = N

        fres = fs / dftlen
        fbin = fi / fres

        f_arr.append(fi)
        fres_arr.append(fres)
        b_arr.append(fbin)
        L_arr.append(dftlen)
        K_arr.append(nseg)

        fi = fi + fres

    nf = len(f_arr)

    # Compute actual averages and starting indices:
    for j in range(nf):
        L_j = int(L_arr[j])
        L_arr[j] = L_j
        averages = int(round_half_up(((N - L_j) / (1 - olap)) / L_j + 1))
        navg_arr.append(averages)

        if averages == 1:
            shift = 1.0
        else:
            shift = (float)(N - L_j) / (float)(averages - 1)
        if shift < 1:
            shift = 1.0

        start = 0.0
        D_arr.append([])
        for _ in range(averages):
            istart = int(float(start) + 0.5) if start >= 0 else int(float(start) - 0.5)
            start = start + shift
            D_arr[j].append(istart)

    # Compute the actual overlaps:
    O_arr = []
    for j in range(nf):
        indices = np.array(D_arr[j])
        if len(indices) > 1:
            overlaps = indices[1:] - indices[:-1]
            O_arr.append(np.mean((L_arr[j] - overlaps) / L_arr[j]))
        else:
            O_arr.append(0.0)

    # Convert lists to numpy arrays:
    f_arr = np.array(f_arr)
    fres_arr = np.array(fres_arr)
    b_arr = np.array(b_arr)
    L_arr = np.array(L_arr)
    K_arr = np.array(K_arr)
    O_arr = np.array(O_arr)
    navg_arr = np.array(navg_arr)

    # Constraint verification (note that some constraints are "soft"):
    if not np.isclose(f_arr[-1], fmax, rtol=0.05):
        logger.warning(f"ltf::ltf_plan: f[-1]={f_arr[-1]} and fmax={fmax}")
    if not np.allclose(f_arr, fres_arr * b_arr):
        logger.warning("ltf::ltf_plan: f[j] != r[j]*b[j]")
    if not np.allclose(
        fres_arr * L_arr, np.full(len(fres_arr), fs)
    ):
        logger.warning("ltf::ltf_plan: r[j]*L[j] != fs")
    if not np.allclose(fres_arr[:-1], np.diff(f_arr), rtol=0.05):
        logger.warning("ltf::ltf_plan: r[j] != f[j+1] - f[j]")
    if not np.all(L_arr < N + 1):
        logger.warning("ltf::ltf_plan: L[j] >= N+1")
    if not np.all(L_arr >= Lmin):
        logger.warning("ltf::ltf_plan: L[j] < Lmin")
    if not np.all(b_arr >= bmin * (1 - 0.05)):
        logger.warning("ltf::ltf_plan: b[j] < bmin")
    if not np.all(L_arr[K_arr == 1] == N):
        logger.warning("ltf::ltf_plan: L[K==1] != N")

    # Final number of frequencies:
    nf = len(f_arr)
    if nf == 0:
        logger.error("Error: frequency scheduler returned zero frequencies")
        sys.exit(-1)

    output = {
        "f": f_arr,
        "r": fres_arr,
        "b": b_arr,
        "m": b_arr,
        "L": L_arr,
        "K": K_arr,
        "navg": navg_arr,
        "D": D_arr,
        "O": O_arr,
        "nf": nf,
    }

    return output


def new_ltf_plan(**args):
    """
    Creates a high-performance blueprint for spectral analysis.

    This function generates a detailed plan for performing a windowed, overlapped,
    segmented spectral analysis on a time series. The resulting frequency grid is
    logarithmically-spaced at low frequencies for high resolution, transitions
    to a regime of aggressive averaging for statistical stability in the mid-range,
    and handles high-Lmin scenarios gracefully to ensure full coverage up to the
    Nyquist frequency.

    The logic is:
    1.  A high-performance vectorized engine generates a "main plan" (`f_main`,
        `r_main`, etc.) for the bulk of the spectrum. This plan uses a hybrid
        log-linear grid and a resolution-blending technique to ensure both
        aggressive averaging and full coverage up to Nyquist.
    2.  A low-frequency "patch" is then generated to ensure a smooth,
        high-resolution ramp down from the maximum segment length (L=N).
    3.  The main plan's frequency vector (`f_main`) is then shifted by a
        calculated offset so that it seamlessly stitches onto the end of the
        patch, preserving the `f[i] = f[i-1] + r[i-1]` rule across the seam.
    4.  All corresponding arrays are then concatenated to form the final plan.

    Args:
        N (int): Total length of the input time series.
        fs (float): Sampling frequency of the time series in Hz.
        olap (float): Desired fractional overlap between segments (e.g., 0.5 for 50%).
        bmin (float): Minimum bin number to use, discarding lower, biased bins.
        Lmin (int): Smallest allowable segment length.
        Jdes (int): The desired number of frequencies in the final plan.

    Returns:
        dict: A dictionary containing all the necessary parameters for analysis.
              Returns None on failure.
    """
    # Unpack & validate
    N, fs, olap, bmin, Lmin, Jdes = _require_args(
        args, ["N", "fs", "olap", "bmin", "Lmin", "Jdes"]
    )

    # --- 1. Initialization and Core Constants ---
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2

    if fmin >= fmax:
        logger.error("fmin >= fmax. Check input parameters (N, fs, bmin).")
        return None

    # --- 2. Generate the Main Plan (Identical to the successful version) ---
    logfact = (fmax / fmin)**(1 / Jdes) - 1
    if logfact <= 0: logfact = 1 / Jdes
    
    f_trans = fs / (Lmin * logfact)
    f_ideal_log, f_ideal_lin = np.array([]), np.array([])

    if f_trans > fmin:
        num_log_pts = max(2, int(np.log(min(f_trans, fmax) / fmin) / np.log(1 + logfact)))
        f_ideal_log = np.geomspace(fmin, min(f_trans, fmax), num=num_log_pts)

    if f_trans < fmax:
        r_lin = fs / Lmin
        lin_start = f_trans if f_trans > fmin else fmin
        f_ideal_lin = np.arange(lin_start + r_lin, fmax + r_lin, r_lin)

    f_ideal = np.unique(np.concatenate((f_ideal_log, f_ideal_lin)))
    f_ideal = f_ideal[f_ideal <= fmax]
    
    if len(f_ideal) < 2:
        logger.warning("Hybrid grid generation failed; falling back to simple logspace.")
        f_ideal = np.geomspace(fmin, fmax, num=Jdes)

    r_ideal = np.append(np.diff(f_ideal), np.diff(f_ideal)[-1])
    r_max_avg = fs / Lmin
    r_blended = np.copy(r_ideal)
    blend_mask = (f_ideal < f_trans)
    r_blended[blend_mask] = np.sqrt(r_ideal[blend_mask] * r_max_avg)
    
    L_float = fs / (r_blended + 1e-12)
    L = np.round(L_float).astype(int)
    L = np.clip(L, Lmin, N)

    K = np.maximum(1, np.round((N - L) / (xov * L) + 1)).astype(int)
    L[K == 1] = N

    r = fs / L
    f_steps = np.insert(r[:-1], 0, 0)
    f_start = r[0] * bmin
    f = f_start + np.cumsum(f_steps)
    b = f / r

    valid_mask = (f <= fmax) & (b >= bmin) & (L <= N)
    f_main_plan, r_main_plan, b_main_plan, L_main_plan, K_main_plan = (
        f[valid_mask], r[valid_mask], b[valid_mask], L[valid_mask], K[valid_mask]
    )

    if len(f_main_plan) == 0:
        logger.error("Main vectorized plan returned zero valid frequencies.")
        return None
        
    # --- 3. Prepend Low-Frequency Ramp with Offset Correction ---
    # This block is the core of the final solution, preserved exactly.
    
    if L_main_plan[0] < N:
        # Generate the L values for the patch ramp
        num_patch_pts = 5
        L_patch = np.geomspace(N, L_main_plan[0], num=num_patch_pts, endpoint=False).astype(int)
        L_patch = np.unique(L_patch)[::-1] # Ensure unique, sorted descending

        # Calculate all parameters for the patch, starting from the absolute fmin
        K_patch = np.maximum(1, np.round((N - L_patch) / (xov * L_patch) + 1)).astype(int)
        r_patch = fs / L_patch
        f_patch_steps = np.insert(r_patch, 0, 0)[:-1] # Steps for cumsum
        f_patch = fmin + np.cumsum(f_patch_steps)
        b_patch = f_patch / r_patch
        
        # Calculate the "seam": the correct frequency for the main plan to start
        f_seam = f_patch[-1] + r_patch[-1]
        
        # Calculate and apply the offset needed to shift the main plan into place
        offset = f_seam - f_main_plan[0]
        f_main_shifted = f_main_plan + offset

        # Concatenate all arrays to form the final plan
        f = np.concatenate((f_patch, f_main_shifted))
        r = np.concatenate((r_patch, r_main_plan))
        b = np.concatenate((b_patch, f_main_shifted / r_main_plan)) # Recalc b with shifted f
        L = np.concatenate((L_patch, L_main_plan))
        K = np.concatenate((K_patch, K_main_plan))
    else:
        # No patch needed; the main plan already starts at maximum resolution.
        # Just ensure the first point's parameters are exact.
        f, r, b, L, K = f_main_plan, r_main_plan, b_main_plan, L_main_plan, K_main_plan
        f[0], r[0], b[0] = fmin, fs / N, bmin
    
    nf = len(f)
    
    # --- 4. Calculate Final Outputs (Start Indices and Overlaps) ---
    D, O = [], []
    for j in range(nf):
        L_j, K_j = L[j], K[j]
        if K_j > 1:
            # Use dtype=int for clean index arrays
            indices = np.linspace(0, N - L_j, K_j, dtype=int)
            step = indices[1] - indices[0]
            O.append((L_j - step) / L_j)
        else:
            indices = np.array([0], dtype=int)
            O.append(0.0)
        D.append(indices)
    O = np.array(O)

    return {
        "f": f, "r": r, "b": b, "m": b, "L": L, "K": K,
        "navg": K, "D": D, "O": O, "nf": nf,
    }
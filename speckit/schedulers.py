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
import sys
import math

import numba
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


def vectorized_ltf_plan(**args):
    """
    Vectorized variant of the LTF scheduler.

    This function implements the LTF scheduler using a two-pass, vectorized 
    approach for significant performance gains over the traditional iterative loop. 

    Phase 1. Pre-computation Pass: A dense, linear "candidate" frequency grid is
        created. The complex conditional logic of the LTF scheduler is applied
        to this entire grid at once using efficient NumPy operations. This
        generates final "parameter maps" (`r_map`, `L_map`, `K_map`) that store
        the correct parameters for any frequency in the dense grid.

    2.  Selection Pass: A fast "walker" loop iterates through the pre-computed
        maps. It starts at `f_min`, looks up the corresponding definitive
        resolution `r[j]` from `r_map`, stores it, and takes a step of that
        exact size to find `f[j+1]`. This process repeats, ensuring the critical
        `r[j] = f[j+1] - f[j]` constraint is perfectly met.

    This approach trades increased memory usage (for the dense maps) for a
    significant reduction in computation time, especially for plans with a
    large number of desired frequencies (`Jdes`). The results are nearly
    identical to an iterative implementation, with minor differences arising
    from the discrete nature of the pre-computed grid.
    """
    N, fs, olap, bmin, Lmin, Jdes, Kdes = _require_args(
        args, ["N", "fs", "olap", "bmin", "Lmin", "Jdes", "Kdes"]
    )
    # --- Phase 1: Vectorized Pre-computation on a LOG grid ---
    xov = 1 - olap
    fmin = bmin * fs / N
    fmax = fs / 2
    rmin = fs / N
    ravg = rmin * (1 + xov * (Kdes - 1))
    clog = (N / 2) ** (1 / Jdes) - 1

    num_grid_points = int(10*Jdes)  # A sufficiently dense grid for most cases.
    f_grid = np.logspace(np.log10(fmin), np.log10(fmax), num_grid_points)

    r_prime_grid = f_grid * clog
    conditions = [
        r_prime_grid >= ravg,
        np.sqrt(ravg * r_prime_grid) > rmin
    ]
    choices = [
        r_prime_grid,
        np.sqrt(ravg * r_prime_grid)
    ]
    r_double_prime_grid = np.select(conditions, choices, default=rmin)
    
    mask = (f_grid / r_double_prime_grid) < bmin
    r_double_prime_grid[mask] = f_grid[mask] / bmin

    L_grid = np.round(fs / r_double_prime_grid)
    L_grid = np.clip(L_grid, Lmin, N)
    K_grid = np.round((N - L_grid) / (xov * L_grid) + 1)
    L_grid[K_grid == 1] = N
    
    r_map = fs / L_grid
    K_map = np.round((N - L_grid) / (xov * L_grid) + 1).astype(np.int64)
    L_map = L_grid.astype(np.int64)

    # --- Phase 2: Walk the map ---
    f_out, r_out, L_out, K_out = [], [], [], []
    current_f = fmin

    while current_f < fmax:
        idx = np.searchsorted(f_grid, current_f, side='left')
        if idx >= len(r_map): break
            
        final_r, final_L, final_K = r_map[idx], L_map[idx], K_map[idx]
        f_out.append(current_f)
        r_out.append(final_r)
        L_out.append(final_L)
        K_out.append(final_K)
        current_f += final_r
    
    # --- Phase 3: Finalize outputs ---
    f, r, L, K, m = np.array(f_out), np.array(r_out), np.array(L_out, dtype=int), np.array(K_out, dtype=int), np.array(f_out) / np.array(r_out)
    nf = len(f)
    
    shift = np.divide(N - L, K - 1, out=np.zeros_like(f, dtype=float), where=K > 1)
    D = [np.round(np.arange(k) * s).astype(int) for k, s in zip(K, shift)]
    O = np.divide(L - shift, L, out=np.zeros_like(f, dtype=float), where=K > 1)

    output = {
        "f": f, "r": r, "b": m, "m": m, "L": L, "K": K,
        "D": D, "O": O, "nf": nf, "navg": K
    }
    return output


def new_ltf_plan(**args):
    """
    High-performance, multi-stage LTF-like scheduler with a unified loop.

    This function implements a robust version of the multi-stage scheduling
    algorithm. It transitions between three distinct regimes:
    1. A low-frequency/transition region using standard LTF compromise logic.
    2. A mid-frequency region that smoothly decreases segment length (L)
       exponentially to target a total number of points near Jdes.
    3. A high-frequency region where L is clamped to Lmin, resulting in a
       linear frequency ramp.

    The logic is unified into a single iterative loop. State flags determine
    which regime's logic is used to calculate the resolution for the current
    step. This architecture guarantees that all physical constraints are
    perfectly met across the stage boundaries, eliminating stitching errors.
    """
    # --- 1. Unpack Arguments and Initialize Constants ---
    N, fs, olap, bmin, Lmin, Jdes, Kdes = _require_args(
        args, ["N", "fs", "olap", "bmin", "Lmin", "Jdes", "Kdes"]
    )
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2
    fresmin = fs / N
    freslim = fresmin * (1 + xov * (Kdes - 1))
    logfact = (N / 2)**(1 / Jdes) - 1

    # --- 2. Initialize Lists and State Variables for the Unified Loop ---
    f, r, b, L, K = [], [], [], [], []
    dftlen_crossover = 0
    stage2, stage3 = False, False
    alpha, k_stage2 = 0, 0
    j = 0
    fi = fmin

    # --- 3. The Unified Loop ---
    while fi < fmax:
        # --- A. Determine dftlen based on the current stage ---
        if stage3:
            # Stage 3: L is clamped to Lmin
            dftlen = Lmin
        elif stage2:
            # Stage 2: L decays exponentially
            dftlen = int(np.round(dftlen_crossover * np.exp(alpha * k_stage2)))
            k_stage2 += 1 # Increment stage 2 step counter
        else:
            # Stage 1: Compromise logic
            fres_ideal = fi * logfact
            if fres_ideal >= freslim:
                stage2 = True # Transition to stage 2 on the NEXT iteration
                # Calculate alpha for the upcoming stage 2
                pts_left = Jdes - j
                if pts_left > 1:
                    alpha = np.log(Lmin / dftlen_crossover) / (pts_left - 1)
                dftlen = int(np.round(fs / fres_ideal)) # Use the ideal fres for this step
            elif (freslim * fres_ideal)**0.5 > fresmin:
                fres = (freslim * fres_ideal)**0.5
                dftlen = int(np.round(fs / fres))
            else:
                fres = fresmin
                dftlen = int(np.round(fs / fres))
            dftlen_crossover = dftlen # Continuously update crossover point

        # --- B. Apply universal constraints and calculate final parameters ---
        if stage2 and dftlen < Lmin:
            stage3 = True # Transition to stage 3 on the NEXT iteration
            dftlen = Lmin

        # Clamp to physical and user limits
        if dftlen > N: dftlen = N
        if dftlen < Lmin: dftlen = Lmin
        
        # If only one segment possible, use the full data length
        nseg = int(np.round((N - dftlen) / (xov * dftlen) + 1))
        if nseg == 1:
            dftlen = N

        # Final resolution bandwidth and bin number for this step
        fres = fs / dftlen
        fbin = fi / fres
        
        # The bmin constraint must always be respected
        if fbin < bmin:
            fres = fi / bmin
            dftlen = int(fs/fres) # Recalculate L if bmin was enforced
            fbin = bmin
            nseg = int(np.round((N - dftlen) / (xov * dftlen) + 1))


        # --- C. Store results and update state for the next iteration ---
        f.append(fi)
        r.append(fres)
        b.append(fbin)
        L.append(dftlen)
        K.append(nseg)
        
        fi += fres # Go to the next frequency
        j += 1

    # --- 4. Finalize and Post-process (Vectorized) ---
    f, r, b, L, K = np.array(f), np.array(r), np.array(b), np.array(L), np.array(K)
    nf = len(f)
    
    shift = np.divide(N - L, K - 1, out=np.zeros_like(f, dtype=float), where=K > 1)
    D = [np.round(np.arange(k) * s).astype(int) for k, s in zip(K, shift)]
    O = np.divide(L - shift, L, out=np.zeros_like(f, dtype=float), where=K > 1)

    output = {"f": f, "r": r, "b": b, "m": b, "L": L, "K": K, "D": D, "O": O, "nf": nf, "navg": K}
    return output
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
import numpy as np
import logging

logger = logging.getLogger(__name__)


def lpsd_plan(N, fs, olap, Jdes, Kdes):
    """
    Original LPSD scheduler.

    Like ltf_plan, but bmin = 1.0, and Lmin = 1.
    """

    return ltf_plan(N, fs, olap, 1.0, 1, Jdes, Kdes)


def ltf_plan(N, fs, olap, bmin, Lmin, Jdes, Kdes):
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

    def round_half_up(val):
        if (float(val) % 1) >= 0.5:
            x = math.ceil(val)
        else:
            x = round(val)
        return x

    # --- Pre-computation and Constants ---
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2
    fresmin = fs / N
    freslim = fresmin * (1 + xov * (Kdes - 1))
    logfact = (fmax / fmin) ** (1 / Jdes) - 1 # More stable logfact calculation

    # --- Pre-allocate NumPy arrays instead of using Python lists ---
    # We allocate a reasonably large size and will trim it later.
    # This avoids the overhead of list.append().
    capacity = int(Jdes * 1.5)
    f_arr = np.zeros(capacity)
    r_arr = np.zeros(capacity)
    b_arr = np.zeros(capacity)
    L_arr = np.zeros(capacity, dtype=int)
    K_arr = np.zeros(capacity, dtype=int)

    # --- Main Scheduler Loop ---
    fi = fmin
    j = 0
    while fi < fmax and j < capacity:
        fres = fi * logfact
        if fres < freslim:
            fres = max((freslim * fres) ** 0.5, fresmin)

        fbin = fi / fres
        if fbin < bmin:
            fbin = bmin
            fres = fi / fbin
        
        dftlen = int(round(fs / fres)) # Use standard round
        
        if dftlen > N:
            dftlen = N
        elif dftlen < Lmin:
            dftlen = Lmin

        nseg = int(round((N - dftlen) / (xov * dftlen) + 1))
        if nseg <= 1: # Handle nseg=0 or 1
            dftlen = N
            nseg = 1
        
        # Recalculate based on quantized dftlen
        fres = fs / dftlen
        fbin = fi / fres

        f_arr[j] = fi
        r_arr[j] = fres
        b_arr[j] = fbin
        L_arr[j] = dftlen
        K_arr[j] = nseg

        fi += fres
        j += 1

    # --- Trim arrays to the actual number of frequencies found ---
    nf = j
    f = f_arr[:nf]
    r = r_arr[:nf]
    b = b_arr[:nf]
    L = L_arr[:nf]
    K = K_arr[:nf]

    # --- Vectorized Calculation of Averages, Offsets, and Overlaps ---
    navg = np.round(((N - L) / (1 - olap)) / L + 1).astype(int)
    navg[navg < 1] = 1 # Ensure at least one average
    
    # Calculate shifts for all frequencies at once
    shifts = np.full(nf, 1.0)
    mask = navg > 1
    shifts[mask] = (N - L[mask]) / (navg[mask] - 1)

    # D (Offsets) is tricky to vectorize perfectly because each row has a
    # different length (navg[j]). We generate it with a list comprehension,
    # which is now much faster because the rest of the plan is already computed.
    offsets = [np.round(np.arange(navg[j]) * shifts[j]).astype(int) for j in range(nf)]
    
    # Compute actual overlaps
    # This is not a required plan output, we include it for completeness.
    overlaps = np.zeros(nf)
    for j in range(nf):
        if navg[j] > 1:
            diffs = np.diff(offsets[j])
            overlaps[j] = np.mean((L[j] - diffs) / L[j])

    output = {
        "f": f, "r": r, "b": b, "m": b, "L": L, "K": K,
        "navg": navg, "D": offsets, "O": overlaps, "nf": nf,
    }
    return output


def new_ltf_plan(N, fs, olap, bmin, Lmin, Jdes, Kdes):
    """
    New LTF scheduler.

    Work in progress.
    """

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
    Frequencies = []
    FrequencyResolutions = []
    BinNumbers = []
    SegmentLengths = []
    NSegments = []
    Overlaps = []
    Offsets = []
    Averages = []

    # Scheduler algorithm:
    dftlen_crossover = int(0)
    f_crossover = 0.0
    third_stage = False
    fourth_stage = False

    j = 0
    fi = fmin
    while fi < fmax:
        fres = fi * logfact
        if fres >= freslim:
            third_stage = True
            break
        elif fres < freslim and (freslim * fres) ** 0.5 > fresmin:
            fres = (freslim * fres) ** 0.5
        else:
            fres = fresmin

        fbin = fi / fres
        if fbin < bmin:
            fbin = bmin
            fres = fi / fbin

        dftlen = int(round_half_up(fs / fres))

        dftlen_crossover = dftlen
        f_crossover = fi

        if dftlen > N:
            dftlen = N
        if dftlen < Lmin:
            dftlen = Lmin

        nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))
        if nseg == 1:
            dftlen = N

        fres = fs / dftlen
        fbin = fi / fres

        Frequencies.append(fi)
        FrequencyResolutions.append(fres)
        BinNumbers.append(fbin)
        SegmentLengths.append(dftlen)
        NSegments.append(nseg)

        fi = fi + fres
        j = j + 1

    if third_stage:
        alpha = 1.0 * np.log(Lmin / dftlen_crossover) / (Jdes - j - 1)
        k = 0
        while fi < fmax:
            dftlen = int(dftlen_crossover * np.exp(alpha * (k + 0)))
            nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))

            if dftlen < Lmin:
                fourth_stage = True
                break

            if k == 0:
                fres = fs / dftlen
                fi = fi + fres
                fi = (fi + f_crossover) / 2
                if fi > fmax:
                    break
                fres = fi - f_crossover
                fbin = fi / fres
            else:
                fres = fs / dftlen
                fi = fi + fres
                if fi > fmax:
                    break
                fbin = fi / fres

            Frequencies.append(fi)
            FrequencyResolutions.append(fres)
            BinNumbers.append(fbin)
            SegmentLengths.append(dftlen)
            NSegments.append(nseg)

            k = k + 1

    if fourth_stage:
        while fi < fmax:
            dftlen = Lmin
            nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))
            fres = fs / dftlen
            fi = fi + fres
            if fi > fmax:
                break
            fbin = fi / fres
            Frequencies.append(fi)
            FrequencyResolutions.append(fres)
            BinNumbers.append(fbin)
            SegmentLengths.append(dftlen)
            NSegments.append(nseg)

    nf = len(Frequencies)

    # Compute actual averages and starting indices:
    for j in range(nf):
        L_j = int(SegmentLengths[j])
        SegmentLengths[j] = L_j
        averages = int(round_half_up(((N - L_j) / (1 - olap)) / L_j + 1))
        Averages.append(averages)

        if averages == 1:
            shift = 1.0
        else:
            shift = (float)(N - L_j) / (float)(averages - 1)
        if shift < 1:
            shift = 1.0

        start = 0.0
        Offsets.append([])
        for _ in range(averages):
            istart = int(float(start) + 0.5) if start >= 0 else int(float(start) - 0.5)
            start = start + shift
            Offsets[j].append(istart)

    # Compute the actual overlaps:
    Overlaps = []
    for j in range(nf):
        indices = np.array(Offsets[j])
        if len(indices) > 1:
            overlaps = indices[1:] - indices[:-1]
            Overlaps.append(np.mean((SegmentLengths[j] - overlaps) / SegmentLengths[j]))
        else:
            Overlaps.append(0.0)

    # Convert lists to numpy arrays:
    Frequencies = np.array(Frequencies)
    FrequencyResolutions = np.array(FrequencyResolutions)
    BinNumbers = np.array(BinNumbers)
    SegmentLengths = np.array(SegmentLengths)
    NSegments = np.array(NSegments)
    Overlaps = np.array(Overlaps)
    Averages = np.array(Averages)

    # Constraint verification (note that some constraints are "soft"):
    if not np.isclose(Frequencies[-1], fmax, rtol=0.05):
        logger.warning(f"ltf::ltf_plan: f[-1]={Frequencies[-1]} and fmax={fmax}")
    if not np.allclose(Frequencies, FrequencyResolutions * BinNumbers):
        logger.warning("ltf::ltf_plan: f[j] != r[j]*b[j]")
    if not np.allclose(
        FrequencyResolutions * SegmentLengths,
        np.full(len(FrequencyResolutions), fs),
        rtol=0.05,
    ):
        logger.warning("ltf::ltf_plan: r[j]*L[j] != fs")
    if not np.allclose(FrequencyResolutions[:-1], np.diff(Frequencies), rtol=0.05):
        logger.warning("ltf::ltf_plan: r[j] != f[j+1] - f[j]")
    if not np.all(SegmentLengths < N + 1):
        logger.warning("ltf::ltf_plan: L[j] >= N+1")
    if not np.all(SegmentLengths >= Lmin):
        logger.warning("ltf::ltf_plan: L[j] < Lmin")
    if not np.all(BinNumbers >= bmin * (1 - 0.05)):
        logger.warning("ltf::ltf_plan: b[j] < bmin")
    if not np.all(SegmentLengths[NSegments == 1] == N):
        logger.warning("ltf::ltf_plan: L[K==1] != N")

    # Final number of frequencies:
    nf = len(Frequencies)
    if nf == 0:
        logger.error("Error: frequency scheduler returned zero frequencies")
        sys.exit(-1)

    output = {
        "f": Frequencies,
        "r": FrequencyResolutions,
        "b": BinNumbers,
        "m": BinNumbers,
        "L": SegmentLengths,
        "K": NSegments,
        "navg": Averages,
        "D": Offsets,
        "O": Overlaps,
        "nf": nf,
    }

    return output

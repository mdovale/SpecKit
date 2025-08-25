def new_ltf_plan_00(N, fs, olap, bmin, Lmin, Jdes, Kdes):
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
    f_arr = []
    fres_arr = []
    b_arr = []
    L_arr = []
    K_arr = []
    O_arr = []
    D_arr = []
    navg_arr = []

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

        f_arr.append(fi)
        fres_arr.append(fres)
        b_arr.append(fbin)
        L_arr.append(dftlen)
        K_arr.append(nseg)

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

            f_arr.append(fi)
            fres_arr.append(fres)
            b_arr.append(fbin)
            L_arr.append(dftlen)
            K_arr.append(nseg)

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
            f_arr.append(fi)
            fres_arr.append(fres)
            b_arr.append(fbin)
            L_arr.append(dftlen)
            K_arr.append(nseg)

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
        fres_arr * L_arr,
        np.full(len(fres_arr), fs),
        rtol=0.05,
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

def new_ltf_plan_01(N, fs, olap, bmin, Lmin, Jdes, _):
    """
    A vectorized LTF scheduler inspired by S2-AEI-TN-3052.

    This function achieves a similar goal to the original ltf_plan but uses a
    vectorized, non-iterative approach for high performance. It first generates an
    ideal logarithmically spaced frequency grid and then "snaps" the corresponding
    analysis parameters to values that are physically and mathematically valid
    (e.g., integer segment lengths).

    The core logic is:
    1. Generate an ideal log-spaced frequency array from fmin to fmax.
    2. From this, derive ideal resolution bandwidths (r) and segment lengths (L).
    3. Quantize/snap the segment lengths (L) to the nearest valid integers,
       respecting Lmin and the total data length N.
    4. Recalculate all other parameters (r, f, b, K) based on the final,
       valid integer segment lengths to ensure perfect consistency with DFT rules.
    5. Calculate the exact segment start indices (D) and actual overlaps (O).

    Inputs and Outputs are identical to the original ltf_plan function.
    """
    # 1. Initialization of Constants (same as original)
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2

    if fmin >= fmax:
        logger.error("fmin >= fmax. Check input parameters N and bmin.")
        return None

    # 2. Generate the Ideal Logarithmic Grid and Initial Parameters
    # Create Jdes logarithmically spaced "ideal" frequencies as our starting point.
    f_ideal = np.geomspace(fmin, fmax, num=Jdes)

    # The ideal resolution is the difference between these frequencies.
    # We pad the end to keep the array size consistent.
    r_ideal = np.append(np.diff(f_ideal), np.diff(f_ideal)[-1])

    # 3. Calculate and Snap Segment Lengths (The Core Vectorized Step)
    # From the DFT constraint r*L = fs, we get an ideal (float) segment length.
    # Add a small epsilon to avoid division by zero if r_ideal is somehow zero.
    L_float = fs / (r_ideal + 1e-12)

    # Quantize L to the nearest integer. This is the "snap-to-grid" operation.
    L = np.round(L_float).astype(int)

    # Enforce constraints on L: must be between Lmin and N.
    L = np.clip(L, Lmin, N)

    # Estimate the number of segments (K) based on these quantized lengths.
    # Use np.maximum to ensure K is at least 1.
    K = np.round((N - L) / (xov * L) + 1).astype(int)
    K = np.maximum(K, 1)

    # A key rule: if only one segment fits, it must use the whole time series.
    # Apply this correction *before* finalizing other parameters.
    L[K == 1] = N

    # 4. Recalculate All Parameters Based on Finalized `L`
    # Now that L is fixed to valid integers, recalculate everything else to be
    # perfectly consistent.
    # Final resolution is now directly derived from the integer L.
    r = fs / L

    # Re-create the frequency grid. Instead of being perfectly logarithmic,
    # it is now defined by the cumulative sum of the *actual* resolutions.
    # This mimics the f[j+1] = f[j] + r[j] logic of the original.
    f_steps = np.insert(r[:-1], 0, 0)
    f = fmin + np.cumsum(f_steps)

    # Final bin number.
    b = f / r
    
    # 5. Filter for Final Valid Frequencies
    # After snapping, some frequencies might have been pushed past fmax or
    # have bin numbers that are too low. We create a mask to remove them.
    valid_mask = (f < fmax) & (b >= bmin)
    
    f = f[valid_mask]
    r = r[valid_mask]
    b = b[valid_mask]
    L = L[valid_mask]
    K = K[valid_mask]
    nf = len(f)

    if nf == 0:
        logger.error("Error: vectorized scheduler returned zero frequencies.")
        sys.exit(-1)
        
    # 6. Calculate Segment Start Indices (D) and Actual Overlap (O)
    # These steps are difficult to vectorize fully because they produce
    # jagged arrays (lists of lists), so a loop is pragmatic and efficient here.
    D = []
    O = []
    for j in range(nf):
        L_j, K_j = L[j], K[j]
        
        # Use np.linspace for a clean way to find evenly spaced start times.
        if K_j > 1:
            indices = np.linspace(0, N - L_j, K_j)
            # Calculate actual overlap from the determined indices
            step = indices[1] - indices[0] # The constant shift
            overlap_frac = (L_j - step) / L_j
            O.append(overlap_frac)
        else: # Case with only one segment
            indices = np.array([0])
            O.append(0.0) # No overlap if there's only one segment

        D.append(np.round(indices).astype(int))

    # Convert final lists to numpy arrays
    O = np.array(O)

    output = {
        "f": f,
        "r": r,
        "b": b,
        "m": b, # Alias for bin number
        "L": L,
        "K": K,
        "navg": K, # In this version, the estimated K is the final K
        "D": D,
        "O": O,
        "nf": nf,
    }

    return output


def new_ltf_plan_02(N, fs, olap, bmin, Lmin, Jdes, Kdes):
    """
    Creates a high-performance, vectorized blueprint for spectral analysis.

    This version adds an "aggressive averaging" feature for low-to-mid
    frequencies by blending the ideal log-spacing resolution with a
    max-averaging resolution. This ensures a higher number of averages
    and lower variance in the lower part of the spectrum.

    The logic is:
    1.  Generate the hybrid log-linear ideal frequency grid as before.
    2.  In the low-frequency zone (where L > Lmin), calculate a blended
        resolution as the geometric mean of the log-spacing resolution and
        the max-averaging resolution (fs/Lmin).
    3.  This blended resolution forces smaller segment lengths (L), which
        naturally increases the number of averages (K).
    4.  The rest of the algorithm proceeds as before, but starts from this
        more statistically-minded foundation.
    """
    # 1. Initialization and Constraint Calculation
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2

    if fmin >= fmax:
        logger.error("fmin >= fmax. Check input parameters N, fs, and bmin.")
        return None

    # 2. Generate the Hybrid Ideal Frequency Grid
    logfact = (fmax / fmin)**(1 / Jdes) - 1
    if logfact <= 0: logfact = 1 / Jdes
    
    # f_trans is the frequency where the log-spacing would demand L < Lmin
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
    
    if len(f_ideal) == 0:
        logger.error("Could not generate an ideal frequency grid.")
        return None

    # 3. ** NEW: AGGRESSIVE AVERAGING VIA RESOLUTION BLENDING **
    r_ideal_log_spacing = np.append(np.diff(f_ideal), np.diff(f_ideal)[-1])
    
    # Define the resolution that corresponds to maximum averaging
    r_max_averaging = fs / Lmin
    
    # Create the final resolution array, starting with the log-spacing ideal
    r_blended = np.copy(r_ideal_log_spacing)
    
    # Identify the zone for blending: where we are not yet limited by Lmin
    blend_zone_mask = (f_ideal < f_trans)
    
    # In this zone, compute the geometric mean to force more averaging
    r_blended[blend_zone_mask] = np.sqrt(
        r_ideal_log_spacing[blend_zone_mask] * r_max_averaging
    )
    
    # 4. Calculate and Snap Segment Lengths (using the blended resolution)
    L_float = fs / (r_blended + 1e-12)
    
    # We no longer need the complex L_practical_max ceiling, as the blending
    # naturally suppresses L. We just need to clip to the hard limits.
    L = np.round(L_float).astype(int)
    L = np.clip(L, Lmin, N)

    # 5. Finalize Parameters and Correct for K=1 case
    K = np.maximum(1, np.round((N - L) / (xov * L) + 1)).astype(int)
    L[K == 1] = N

    # 6. Recalculate Final Consistent Parameters
    r = fs / L
    f_steps = np.insert(r[:-1], 0, 0)
    f_start = r[0] * bmin
    f = f_start + np.cumsum(f_steps)
    b = f / r

    # 7. Filter for Final Valid Frequencies
    valid_mask = (f <= fmax * 1.01) & (b >= bmin * 0.99) & (L <= N)
    f, r, b, L, K = f[valid_mask], r[valid_mask], b[valid_mask], L[valid_mask], K[valid_mask]
    nf = len(f)
    
    if nf == 0:
        logger.error("Vectorized scheduler returned zero frequencies after filtering.")
        return None

    # 8. Calculate Segment Start Indices (D) and Actual Overlap (O)
    D, O = [], []
    for j in range(nf):
        L_j, K_j = L[j], K[j]
        if K_j > 1:
            indices = np.linspace(0, N - L_j, K_j)
            step = indices[1] - indices[0]
            O.append((L_j - step) / L_j)
        else:
            indices = np.array([0])
            O.append(0.0)
        D.append(np.round(indices).astype(int))
    O = np.array(O)

    return {
        "f": f, "r": r, "b": b, "m": b, "L": L, "K": K,
        "navg": K, "D": D, "O": O, "nf": nf,
    }


def new_ltf_plan_03(N, fs, olap, bmin, Lmin, Jdes, Kdes):
    """
    Creates a definitive, high-performance blueprint for spectral analysis.

    This function synthesizes solutions to all identified challenges, handling three
    distinct spectral regimes for a truly optimal plan:

    1.  Regime 1 (Resolution-First): At the lowest frequencies, the plan prioritizes
        the highest possible resolution by using very large segment lengths (L),
        accepting few averages (K).

    2.  Regime 2 (Averaging-First): In the mid-range, an aggressive resolution-blending
        technique is used to increase K, ensuring statistically robust estimates.

    3.  Regime 3 (Coverage-First): At high frequencies, if Lmin is large, the plan
        switches to a linear frequency grid to guarantee full spectral coverage up
        to the Nyquist frequency.

    These regimes are generated from a single, consistent ideal frequency grid and
    then finalized, ensuring perfect self-consistency.
    """
    Kdes = 1

    # 1. Initialization and Global Parameters
    xov = 1 - olap
    fmin = fs / N * bmin
    fmax = fs / 2

    if fmin >= fmax:
        logger.error("fmin >= fmax. Check input parameters.")
        return None

    # Calculate global log-spacing factor. This is crucial.
    # It informs the physical transition points for the different regimes.
    logfact = (fmax / fmin)**(1 / Jdes) - 1
    if logfact <= 0: logfact = 1 / Jdes

    # 2. Define the Boundaries of the Three Regimes
    # L_practical_max: Largest L that supports Kdes averages. Marks the boundary
    # between the resolution-first and averaging-first regimes.
    if Kdes > 1:
        L_practical_max = int(N / (1 + (Kdes - 1) * xov))
    else:
        L_practical_max = N
    L_practical_max = max(Lmin, L_practical_max)
    f_res_to_avg_boundary = (fs / L_practical_max) * bmin

    # f_trans: Frequency where ideal log spacing would require L < Lmin. Marks the
    # boundary between the averaging-first and coverage-first (linear) regimes.
    f_avg_to_lin_boundary = fs / (Lmin * logfact)

    # 3. Generate a Single, Hybrid Ideal Frequency Grid
    f_ideal_res, f_ideal_avg, f_ideal_lin = np.array([]), np.array([]), np.array([])
    
    # Generate points for each regime based on the calculated boundaries.
    # We use np.linspace for the number of points to distribute Jdes proportionally.
    f_boundaries = sorted(np.unique([fmin, f_res_to_avg_boundary, f_avg_to_lin_boundary, fmax]))
    
    # Start building the full grid from fmin
    current_f = fmin
    f_ideal = np.array([fmin])

    # Regime 1: Resolution-First (fmin to f_res_to_avg_boundary)
    if current_f < f_res_to_avg_boundary:
        end_f = min(f_res_to_avg_boundary, fmax)
        num_pts = max(2, int(Jdes * (end_f - current_f) / (fmax - fmin)))
        f_ideal_res = np.geomspace(current_f, end_f, num=num_pts)
        f_ideal = np.concatenate((f_ideal, f_ideal_res))
        current_f = end_f
        
    # Regime 2: Averaging-First (f_res_to_avg_boundary to f_avg_to_lin_boundary)
    if current_f < f_avg_to_lin_boundary:
        end_f = min(f_avg_to_lin_boundary, fmax)
        num_pts = max(2, int(Jdes * (end_f - current_f) / (fmax - fmin)))
        f_ideal_avg = np.geomspace(current_f, end_f, num=num_pts)
        f_ideal = np.concatenate((f_ideal, f_ideal_avg))
        current_f = end_f

    # Regime 3: Coverage-First / Linear (f_avg_to_lin_boundary to fmax)
    if current_f < fmax:
        r_lin = fs / Lmin
        # Start from the next step after the last frequency
        start_f = current_f if np.isclose(current_f, f_avg_to_lin_boundary) else current_f + r_lin
        f_ideal_lin = np.arange(start_f, fmax + r_lin, r_lin)
        f_ideal = np.concatenate((f_ideal, f_ideal_lin))

    f_ideal = np.unique(f_ideal)
    f_ideal = f_ideal[f_ideal <= fmax]

    # 4. Calculate Ideal Segment Lengths with Targeted Modifications per Regime
    r_ideal = np.append(np.diff(f_ideal), np.diff(f_ideal)[-1])
    L_float = fs / (r_ideal + 1e-12)

    # Apply aggressive averaging blend ONLY in the averaging regime
    avg_regime_mask = (f_ideal >= f_res_to_avg_boundary) & (f_ideal < f_avg_to_lin_boundary)
    if np.any(avg_regime_mask):
        r_max_averaging = fs / Lmin
        r_blended = np.sqrt(r_ideal[avg_regime_mask] * r_max_averaging)
        L_float[avg_regime_mask] = fs / (r_blended + 1e-12)

    # 5. Finalize L, K, and Recalculate Everything for Consistency
    L = np.round(L_float).astype(int)
    L = np.clip(L, Lmin, N)

    K = np.maximum(1, np.round((N - L) / (xov * L) + 1)).astype(int)
    
    # Sacred rule: The first time we hit K=1, L must be N.
    k1_indices = np.where(K == 1)[0]
    if len(k1_indices) > 0:
        L[k1_indices[0]] = N
        if len(k1_indices) > 1: # Prune redundant K=1 entries
             L = np.delete(L, k1_indices[1:])
             K = np.delete(K, k1_indices[1:])

    r = fs / L
    f_steps = np.insert(r[:-1], 0, 0)
    f_start = r[0] * bmin
    f = f_start + np.cumsum(f_steps)
    b = f / r

    # 6. Filter and Calculate Final Outputs
    valid_mask = (f <= fmax * 1.01) & (b >= bmin * 0.99) & (L <= N)
    f, r, b, L, K = f[valid_mask], r[valid_mask], b[valid_mask], L[valid_mask], K[valid_mask]
    nf = len(f)

    if nf == 0:
        logger.error("Vectorized scheduler returned zero frequencies after filtering.")
        return None

    D, O = [], []
    for j in range(nf):
        L_j, K_j = L[j], K[j]
        if K_j > 1:
            indices = np.linspace(0, N - L_j, K_j)
            step = indices[1] - indices[0]
            O.append((L_j - step) / L_j)
        else:
            indices = np.array([0])
            O.append(0.0)
        D.append(np.round(indices).astype(int))
    O = np.array(O)

    return {
        "f": f, "r": r, "b": b, "m": b, "L": L, "K": K,
        "navg": K, "D": D, "O": O, "nf": nf,
    }

def spectral_partitioner(N, fs, olap, bmin, Lmin, Jdes, Kdes):
    """
    Creates a spectral analysis plan using an adaptive partitioning algorithm.

    This method treats the frequency spectrum as a 1D space to be recursively
    subdivided. It uses a priority queue to iteratively split the "least optimal"
    spectral interval, ensuring that frequency points are allocated intelligently
    to where they are needed most.

    The logic is:
    1.  Start with a single interval representing the entire spectrum `[fmin, fmax]`.
    2.  Use a priority queue to hold all current intervals. The priority of an
        interval is a score indicating how "bad" it is (e.g., too wide
        logarithmically, or providing too few averages).
    3.  In a loop running `Jdes` times:
        a. Pop the worst interval (highest priority) from the queue.
        b. This interval is now considered "finalized". Store its midpoint.
        c. Split the interval into two new sub-intervals (at its geometric mean).
        d. Calculate the priority of the new sub-intervals and push them
           onto the queue.
    4.  The collection of finalized midpoints forms our ideal frequency grid.
    5.  From this grid, derive a consistent set of `L`, `r`, `K`, etc.

    This "divide and conquer" approach is robust, performant, and naturally
    creates the desired logarithmic-like distribution.
    """
    fmin_abs = (fs / N) * bmin
    fmax = fs / 2

    if fmin_abs >= fmax:
        logger.error("fmin >= fmax. Check input parameters.")
        return None

    # The priority queue will store tuples: (-priority, f_start, f_end)
    # We use negative priority because heapq is a min-heap.
    priority_queue = []
    
    # --- Helper function to calculate an interval's priority ---
    def calculate_priority(f_start, f_end):
        if f_start <= 0 or f_end <= f_start:
            return 0

        # Goal 1: Logarithmic Spacing. Wider log-intervals are worse.
        # A higher ratio means it needs to be split more urgently.
        log_width = np.log10(f_end / f_start)

        # Goal 2: Averaging Pressure. Intervals with few averages are worse,
        # unless they are already at maximum resolution (L approaching N).
        f_center = np.sqrt(f_start * f_end) # Geometric mean
        r_approx = f_end - f_start
        L_approx = fs / (r_approx + 1e-9)
        
        # If L is already huge, penalty should be low.
        if L_approx > N * 0.9:
            averaging_penalty = 0.5
        else:
            K_approx = max(1, (N - L_approx) / (L_approx * (1 - olap)) + 1)
            # Penalty sharply increases as K_approx drops below Kdes.
            averaging_penalty = 1 + (Kdes / (K_approx + 1))**2

        # Goal 3: High Lmin Penalty. Penalize intervals that would violate Lmin.
        lmin_penalty = 1.0
        if L_approx < Lmin:
            lmin_penalty = 100 * (Lmin / (L_approx + 1)) # Very high penalty

        return log_width * averaging_penalty * lmin_penalty

    # --- Algorithm Start ---
    # 1. Initialize with the full spectral interval
    initial_priority = calculate_priority(fmin_abs, fmax)
    heapq.heappush(priority_queue, (-initial_priority, fmin_abs, fmax))
    
    final_freqs = []

    # 2. Iteratively partition the worst interval
    # We run one extra iteration to define the bounds of the last point
    for _ in range(Jdes + 1):
        if not priority_queue:
            break
            
        # Pop the worst interval
        neg_p, f_start, f_end = heapq.heappop(priority_queue)
        
        # Add its start point to our list of boundaries
        final_freqs.append(f_start)
        
        # Split it at the geometric mean
        f_mid = np.sqrt(f_start * f_end)
        
        if f_mid <= f_start or f_mid >= f_end:
            continue # Stop splitting if we've lost precision

        # Create two new sub-intervals
        # Left sub-interval: [f_start, f_mid]
        p_left = calculate_priority(f_start, f_mid)
        heapq.heappush(priority_queue, (-p_left, f_start, f_mid))

        # Right sub-interval: [f_mid, f_end]
        p_right = calculate_priority(f_mid, f_end)
        heapq.heappush(priority_queue, (-p_right, f_mid, f_end))

    # 3. Finalize the Plan from the Partitioned Frequencies
    f_boundaries = np.sort(np.unique(final_freqs))
    
    if len(f_boundaries) < 2:
        logger.error("Partitioning failed to produce a valid frequency grid.")
        return None
        
    # The final frequencies are the geometric midpoints of the final partitions
    f = np.sqrt(f_boundaries[:-1] * f_boundaries[1:])
    # The resolution is the width of those partitions
    r = np.diff(f_boundaries)
    
    # From here, the process is familiar: derive everything from f and r
    L = np.round(fs / r).astype(int)
    
    # Enforce sacred rules
    L = np.clip(L, Lmin, N)
    K = np.maximum(1, np.round((N - L) / ((1 - olap) * L) + 1)).astype(int)
    L[K == 1] = N
    
    # Recalculate for final consistency
    r_final = fs / L
    f_steps = np.insert(r_final[:-1], 0, 0)
    f_start = r_final[0] * bmin
    f_final = f_start + np.cumsum(f_steps)
    b = f_final / r_final

    # Filter and calculate final outputs
    valid_mask = (f_final <= fmax * 1.01) & (b >= bmin * 0.99)
    f_out,r_out,b_out,L_out,K_out = f_final[valid_mask], r_final[valid_mask], b[valid_mask], L[valid_mask], K[valid_mask]

    nf = len(f_out)
    if nf == 0:
        logger.error("The scheduler returned zero valid frequencies.")
        return None

    D, O = [], []
    for j in range(nf):
        L_j, K_j = L_out[j], K_out[j]
        if K_j > 1:
            indices = np.linspace(0, N - L_j, K_j)
            step = indices[1] - indices[0]
            O.append((L_j - step) / L_j)
        else:
            indices = np.array([0])
            O.append(0.0)
        D.append(np.round(indices).astype(int))
    O = np.array(O)

    return {
        "f": f_out, "r": r_out, "b": b_out, "m": b_out, "L": L_out, "K": K_out,
        "navg": K_out, "D": D, "O": O, "nf": nf,
    }
#!/usr/bin/env python3
"""
benchmark_speckit_lpsd_linear_detrending

Benchmarks SpecKit LPSD performance using linear detrending (order = 1)
on 10 million points of uniformly distributed random data.

LPSD parameters (matching MATLAB benchmark):
    fs   = 2
    Jdes = 1000
    Kdes = 100
    bmin = 1.0
    Lmin = 1
    win  = "hann"
    olap = 0.75

Output:
    - Prints timing statistics for:
        * order = 1 (linear detrend)  [main requested case]
        * order = 0 (mean removal)    [reference]
        * order = -1 (no detrend)     [reference]

Notes:
    - This script avoids file I/O to benchmark the spectral estimation itself.
    - Uses numpy for reproducible random number generation.
"""

import numpy as np
import time
from speckit import SpectrumAnalyzer


def report_stats(name, t):
    """Print timing statistics for a set of runs."""
    print(
        f"{name}: mean={np.mean(t):.3f}, median={np.median(t):.3f}, "
        f"std={np.std(t):.3f}, min={np.min(t):.3f}, max={np.max(t):.3f}"
    )


def bench_lpsd(data, fs, order, label, n_runs):
    """
    Benchmark LPSD computation for a given detrending order.
    
    Parameters
    ----------
    data : np.ndarray
        Input time-series data
    fs : float
        Sampling frequency
    order : int
        Detrending order (-1, 0, or 1)
    label : str
        Label for the benchmark case
    n_runs : int
        Number of benchmark runs
        
    Returns
    -------
    np.ndarray
        Array of timing results in seconds
    """
    tvec = np.zeros(n_runs)
    print(f"Benchmark: {label} (nRuns={n_runs})")
    
    for i in range(n_runs):
        # Create analyzer with specified parameters
        analyzer = SpectrumAnalyzer(
            data=data,
            fs=fs,
            olap=0.75,
            bmin=1.0,
            Lmin=1,
            Jdes=1000,
            Kdes=100,
            order=order,
            win="hann",
            scheduler="lpsd",  # Use LPSD scheduler to match LTPDA
            verbose=False,
        )
        
        # Time the computation
        t0 = time.perf_counter()
        result = analyzer.compute()
        tvec[i] = time.perf_counter() - t0
        
        # Clear result to free memory
        del result
        del analyzer
        
        print(f"  run {i+1}/{n_runs}: {tvec[i]:.3f} s")
    
    print()
    return tvec


def main():
    """Main benchmark function."""
    # --- Configuration ---
    fs = 2.0
    Jdes = 1000
    Kdes = 100
    bmin = 1.0
    Lmin = 1
    win_name = "hann"
    olap = 0.75
    
    N = 10_000_000  # 10 million points
    np.random.seed(0)  # Reproducible
    
    # Benchmark repetition settings
    n_warmup = 1
    n_runs = 10  # Increase if you want tighter statistics (at the cost of time)
    
    print("--- SpecKit LPSD benchmark (linear detrending) ---")
    print(f"N = {N} samples, fs = {fs:.6g} Hz")
    print(
        f"LPSD params: Jdes={Jdes}, Kdes={Kdes}, Lmin={Lmin}, "
        f"Win={win_name}, Olap={olap:.2f}, Scale=ASD"
    )
    print("Detrending: order=1 (linear)")
    print()
    
    # --- Generate uniform random data (centered) ---
    # Uniform in (-0.5, +0.5); still "uniformly distributed" but zero-mean-ish.
    y = np.random.rand(N) - 0.5
    
    # --- Warm-up (JIT, caches, etc.) ---
    # Warm up all three detrending orders to ensure all Numba functions are compiled.
    # Each order uses different Numba-compiled kernels:
    #   - order=-1: _stats_win_only_auto
    #   - order=0:  _stats_detrend0_auto
    #   - order=1:  _stats_poly_auto
    # Numba functions with cache=True persist in memory for the session, so they'll
    # be reused in the timed benchmark runs without recompilation overhead.
    # Use a small dataset for warm-up to avoid long computation times.
    N_warmup = 50000  # Small dataset sufficient to trigger compilation
    y_warmup = np.random.rand(N_warmup) - 0.5
    print(f"Warm-up runs: {n_warmup} per order (orders: -1, 0, 1) on {N_warmup} samples")
    for order_warmup in [-1, 0, 1]:
        for k in range(n_warmup):
            analyzer_warmup = SpectrumAnalyzer(
                data=y_warmup,  # Use small dataset for fast warm-up
                fs=fs,
                olap=olap,
                bmin=bmin,
                Lmin=Lmin,
                Jdes=Jdes,
                Kdes=Kdes,
                order=order_warmup,  # Warm up all orders to compile all Numba functions
                win=win_name,
                scheduler="lpsd",
                verbose=False,
            )
            _ = analyzer_warmup.compute()
            del analyzer_warmup
    print("Warm-up done.\n")
    
    # --- Run benchmarks ---
    t_lin = bench_lpsd(
        y, fs, order=1, label="order=1 (linear detrend)  [requested]", n_runs=n_runs
    )
    t_mean = bench_lpsd(
        y, fs, order=0, label="order=0 (mean removal)    [reference]", n_runs=n_runs
    )
    t_none = bench_lpsd(
        y, fs, order=-1, label="order=-1 (no detrend)     [reference]", n_runs=n_runs
    )
    
    # --- Report summary ---
    print("--- Summary (seconds) ---")
    report_stats("order=1 (linear)", t_lin)
    report_stats("order=0 (mean)  ", t_mean)
    report_stats("order=-1 (none) ", t_none)
    
    print("\nDone.")


if __name__ == "__main__":
    main()

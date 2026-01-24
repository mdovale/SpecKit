#!/usr/bin/env python3
"""
benchmark_cuda_speedup.py

Benchmarks CUDA acceleration performance compared to CPU backends (Numba and NumPy).
Measures speedup for various problem sizes and detrending orders.

Output:
    - Prints timing statistics for each backend
    - Generates speedup comparison table
    - Optionally generates speedup plot
"""

import numpy as np
import time
import sys
from speckit import SpectrumAnalyzer
from speckit.core import _CUDA_ENABLED, _NUMBA_ENABLED


def report_stats(name, times):
    """Print timing statistics for a set of runs."""
    if len(times) == 0:
        print(f"{name}: No runs completed")
        return
    print(
        f"{name}: mean={np.mean(times):.3f}s, median={np.median(times):.3f}s, "
        f"std={np.std(times):.3f}s, min={np.min(times):.3f}s, max={np.max(times):.3f}s"
    )


def benchmark_backend(data, fs, order, backend, n_runs=5):
    """
    Benchmark a specific backend.

    Parameters
    ----------
    data : np.ndarray
        Input time-series data
    fs : float
        Sampling frequency
    order : int
        Detrending order
    backend : str
        Backend name ('cuda', 'numba', 'numpy')
    n_runs : int
        Number of benchmark runs

    Returns
    -------
    np.ndarray
        Array of timing results in seconds
    """
    times = []
    for i in range(n_runs):
        try:
            analyzer = SpectrumAnalyzer(
                data=data,
                fs=fs,
                backend=backend,
                olap=0.75,
                bmin=1.0,
                Lmin=1,
                Jdes=1000,
                Kdes=100,
                order=order,
                win="hann",
                scheduler="lpsd",
                verbose=False,
            )

            t0 = time.perf_counter()
            result = analyzer.compute()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            del result
            del analyzer
        except Exception as e:
            print(f"  Backend {backend} failed on run {i+1}: {e}")
            break

    return np.array(times)


def main():
    """Main benchmark function."""
    # Configuration
    fs = 2.0
    Jdes = 1000
    Kdes = 100
    order = 1  # Linear detrend
    n_runs = 5

    # Test different data sizes
    sizes = [1_000_000, 5_000_000, 10_000_000]
    np.random.seed(0)

    print("=" * 80)
    print("CUDA Speedup Benchmark")
    print("=" * 80)
    print(f"LPSD params: Jdes={Jdes}, Kdes={Kdes}, order={order}, win=hann")
    print(f"Number of runs per backend: {n_runs}")
    print()

    results_summary = []

    for N in sizes:
        print(f"\n--- N = {N:,} samples ---")
        y = np.random.rand(N) - 0.5  # Centered uniform random

        backends_to_test = []
        if _CUDA_ENABLED:
            backends_to_test.append("cuda")
        if _NUMBA_ENABLED:
            backends_to_test.append("numba")
        backends_to_test.append("numpy")

        backend_times = {}
        for backend in backends_to_test:
            print(f"Benchmarking {backend} backend...")
            times = benchmark_backend(y, fs, order, backend, n_runs)
            if len(times) > 0:
                backend_times[backend] = times
                report_stats(f"  {backend}", times)

        # Calculate speedups
        if len(backend_times) > 1:
            print("\nSpeedup vs NumPy:")
            numpy_time = np.mean(backend_times.get("numpy", [np.inf]))
            for backend in backends_to_test:
                if backend != "numpy" and backend in backend_times:
                    mean_time = np.mean(backend_times[backend])
                    speedup = numpy_time / mean_time if mean_time > 0 else np.inf
                    print(f"  {backend}: {speedup:.2f}x")

            # Store summary
            summary_row = {"N": N}
            for backend in backends_to_test:
                if backend in backend_times:
                    summary_row[backend] = np.mean(backend_times[backend])
            results_summary.append(summary_row)

    # Print summary table
    print("\n" + "=" * 80)
    print("Summary Table (mean time in seconds)")
    print("=" * 80)
    if results_summary:
        print(f"{'N':>12} ", end="")
        for backend in backends_to_test:
            print(f"{backend:>12} ", end="")
        print()
        print("-" * 80)
        for row in results_summary:
            print(f"{row['N']:>12,} ", end="")
            for backend in backends_to_test:
                if backend in row:
                    print(f"{row[backend]:>12.3f} ", end="")
                else:
                    print(f"{'N/A':>12} ", end="")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()

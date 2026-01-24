#!/usr/bin/env python3
"""
benchmark_cuda_scaling.py

Analyzes CUDA performance scaling with problem size.
Tests different segment counts (K) and segment lengths (L) to identify
optimal problem sizes for GPU acceleration.

Output:
    - Scaling analysis table
    - Performance vs problem size metrics
"""

import numpy as np
import time
from speckit import SpectrumAnalyzer
from speckit.core import _CUDA_ENABLED, _NUMBA_ENABLED


def benchmark_scaling(N, fs, K_target, L_target, backend, order=1, n_runs=3):
    """
    Benchmark performance for a specific problem size.

    Parameters
    ----------
    N : int
        Total data length
    fs : float
        Sampling frequency
    K_target : int
        Target number of segments
    L_target : int
        Target segment length
    backend : str
        Backend to test
    order : int
        Detrending order
    n_runs : int
        Number of runs

    Returns
    -------
    float
        Mean computation time in seconds
    """
    np.random.seed(42)
    data = np.random.randn(N).astype(np.float64)

    times = []
    for _ in range(n_runs):
        try:
            # Adjust Jdes to get approximately K_target segments
            # Rough heuristic: Jdes ~ K_target for LPSD scheduler
            analyzer = SpectrumAnalyzer(
                data=data,
                fs=fs,
                backend=backend,
                olap=0.75,
                bmin=1.0,
                Lmin=L_target,
                Jdes=max(100, K_target // 2),  # Approximate K
                Kdes=K_target,
                order=order,
                win="hann",
                scheduler="lpsd",
                verbose=False,
            )

            t0 = time.perf_counter()
            result = analyzer.compute()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

            # Get actual K from result
            actual_K = result.K.mean() if hasattr(result, "K") else K_target
            del result
            del analyzer
        except Exception as e:
            print(f"  Error: {e}")
            return None

    return np.mean(times) if times else None


def main():
    """Main scaling benchmark function."""
    fs = 2.0
    order = 1
    n_runs = 3

    # Test different segment counts
    K_values = [100, 500, 1000, 5000, 10000]
    # Test different segment lengths (approximate)
    L_values = [100, 500, 1000, 5000]

    print("=" * 100)
    print("CUDA Scaling Benchmark")
    print("=" * 100)
    print(f"Testing scaling behavior with different K (segments) and L (segment length)")
    print(f"Backends available: CUDA={_CUDA_ENABLED}, Numba={_NUMBA_ENABLED}")
    print(f"Number of runs per configuration: {n_runs}")
    print()

    if not _CUDA_ENABLED:
        print("CUDA not available. Skipping CUDA benchmarks.")
        return

    # Fixed total data size
    N = 10_000_000

    print(f"Total data size: N = {N:,} samples")
    print()

    # Test CUDA scaling
    print("CUDA Backend Scaling:")
    print("-" * 100)
    print(f"{'K (segments)':>12} {'L (approx)':>12} {'Time (s)':>12} {'Throughput':>15}")
    print("-" * 100)

    cuda_results = []
    for K in K_values:
        for L in L_values:
            if K * L > N:
                continue  # Skip if segments don't fit

            time_cuda = benchmark_scaling(N, fs, K, L, "cuda", order, n_runs)
            if time_cuda is not None:
                throughput = (K * L) / time_cuda / 1e6  # Million samples/second
                print(f"{K:>12} {L:>12} {time_cuda:>12.3f} {throughput:>15.2f} MS/s")
                cuda_results.append({"K": K, "L": L, "time": time_cuda, "throughput": throughput})

    # Compare with Numba if available
    if _NUMBA_ENABLED:
        print("\n" + "=" * 100)
        print("Numba Backend Scaling (for comparison):")
        print("-" * 100)
        print(f"{'K (segments)':>12} {'L (approx)':>12} {'Time (s)':>12} {'Speedup':>12}")
        print("-" * 100)

        for K in K_values[:3]:  # Test subset for comparison
            for L in L_values[:2]:
                if K * L > N:
                    continue

                time_numba = benchmark_scaling(N, fs, K, L, "numba", order, n_runs)
                if time_numba is not None:
                    # Find matching CUDA result
                    cuda_time = None
                    for r in cuda_results:
                        if r["K"] == K and r["L"] == L:
                            cuda_time = r["time"]
                            break

                    if cuda_time is not None:
                        speedup = time_numba / cuda_time
                        print(
                            f"{K:>12} {L:>12} {time_numba:>12.3f} {speedup:>12.2f}x"
                        )

    # Summary
    print("\n" + "=" * 100)
    print("Summary:")
    print("=" * 100)
    if cuda_results:
        best_throughput = max(cuda_results, key=lambda x: x["throughput"])
        print(f"Best throughput: {best_throughput['throughput']:.2f} MS/s")
        print(f"  Configuration: K={best_throughput['K']}, L={best_throughput['L']}")

        worst_throughput = min(cuda_results, key=lambda x: x["throughput"])
        print(f"Worst throughput: {worst_throughput['throughput']:.2f} MS/s")
        print(f"  Configuration: K={worst_throughput['K']}, L={worst_throughput['L']}")

    print("\nDone.")


if __name__ == "__main__":
    main()

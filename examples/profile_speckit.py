# profile_speckit.py

import numpy as np
import cProfile
import pstats

# Import your main user-facing function
from speckit.analysis import compute_spectrum
# from speckit.lpsd import ltf as compute_spectrum


def main():
    """Sets up and runs the profiling task."""
    print("Setting up profiling workload...")

    # --- 1. Define a realistic, single-core workload ---
    # We profile the serial case first to understand the baseline performance.
    N = int(1e7)  # A reasonably long time series
    fs = 2.0
    data = np.random.randn(N)

    print(f"Profiling compute_spectrum on a time series of length {N}...")

    # --- 2. Run the function under cProfile ---
    # The command to execute is a single call to your main function.
    command = "compute_spectrum(data, fs=fs, win='hann', Jdes=1000, order=0, scheduler='new_ltf')"

    # Define the context for the profiler
    # It needs access to the variables used in `command`.
    profiler_context = {"compute_spectrum": compute_spectrum, "data": data, "fs": fs}

    # Run the profiler and save the stats to a file
    cProfile.runctx(
        command, globals=profiler_context, locals={}, filename="speckit_profile.prof"
    )

    print("Profiling complete. Stats saved to 'speckit_profile.prof'")

    # --- 3. (Optional) Print a simple summary to the console ---
    print("\n--- Top 10 Functions by Cumulative Time ---")
    stats = pstats.Stats("speckit_profile.prof")
    stats.sort_stats("cumulative").print_stats(10)


if __name__ == "__main__":
    main()

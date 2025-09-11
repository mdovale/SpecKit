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
    command = "compute_spectrum(data, fs=fs, win='hann', Jdes=1000, order=-1, scheduler='vectorized_ltf')"

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

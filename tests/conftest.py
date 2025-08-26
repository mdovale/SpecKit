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
import pytest
import numpy as np
import multiprocessing as mp

from speckit.noise import butter_lowpass_filter

# Set multiprocessing start method for consistency across platforms
mp.set_start_method("spawn", force=True)


@pytest.fixture(scope="session")
def reproducible_rng():
    """Provides a reproducible random number generator."""
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def short_white_noise_data(reproducible_rng):
    """
    A short, reproducible white noise time series for quick tests.
    Returns a dictionary with data, length N, and sampling frequency fs.
    """
    N = int(2e4)
    fs = 1000.0
    data = reproducible_rng.normal(size=N)
    return {"data": data, "N": N, "fs": fs}


@pytest.fixture(scope="session")
def long_white_noise_data(reproducible_rng):
    """
    A longer, reproducible white noise time series for more intensive tests.
    """
    N = int(5e5)
    fs = 2.0
    data = reproducible_rng.normal(size=N)
    return {"data": data, "N": N, "fs": fs}


@pytest.fixture(scope="session")
def siso_data(reproducible_rng):
    """
    Generates a simple SISO (Single-Input, Single-Output) system.
    Input is white noise, output is a low-pass filtered version of the input.
    """
    N = int(1e5)
    fs = 100.0
    cutoff = 10.0
    order = 4
    input_signal = reproducible_rng.normal(size=N)
    output_signal = butter_lowpass_filter(input_signal, cutoff, fs, order)
    # Add some independent noise to the output
    output_signal += 0.1 * reproducible_rng.normal(size=N)
    return {"input": input_signal, "output": output_signal, "N": N, "fs": fs}


@pytest.fixture(scope="session")
def multiprocessing_pool():
    """Provides a multiprocessing pool that is properly closed."""
    with mp.Pool() as pool:
        yield pool

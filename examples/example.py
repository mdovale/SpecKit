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
import matplotlib.pyplot as plt
from scipy import signal
from speckit import SpectrumAnalyzer
from speckit.noise import band_limited_noise

if __name__ == "__main__":
    N = int(1e6)  # Size of the example time series
    fs = 2.0  # Sampling frequency

    x1 = (
        1e-6 * band_limited_noise(1e-6, 1e0, N, fs)
        + 1e3 * band_limited_noise(1e-3, 10e-3, N, fs)
        + 1e3 * band_limited_noise(50e-3, 60e-3, N, fs)
    )
    fig, ax = plt.subplots(figsize=(16, 4), dpi=150)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Signal")
    ax.plot(x1, color="black")
    plt.show(block=False)

    f1, psd = signal.welch(
        x1,
        fs=fs,
        window=("kaiser", 30),
        nperseg=int(N),
        noverlap=int(N / 2),
        scaling="density",
        return_onesided=True,
    )

    # 1. Configure the analyzer
    print("Configuring SpectrumAnalyzer...")
    analyzer = SpectrumAnalyzer(
        data=x1,
        fs=fs,
        bmin=1.0,
        order=-1,
        win="Kaiser",
        Jdes=1000,
        psll=200,
        verbose=True,
    )

    # 2. (Optional) Inspect the plan before computation
    plan = analyzer.plan()
    print(f"\nGenerated a plan with {plan['nf']} frequency bins.")

    # 3. Create a multiprocessing pool and run the computation
    print("\nStarting computation...")
    result = analyzer.compute()

    print("\nComputation complete!")
    print(f"Result object type: {type(result)}")

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.set_xlabel("Fourier frequency (Hz)")
    ax.set_ylabel(r"Spectral density (units $/\sqrt{\rm Hz}$)")
    ax.loglog(f1, np.sqrt(psd), label="Welch", color="gray")
    result.plot(which="asd", ax=ax, label="LPSD")
    ax.legend()
    fig.tight_layout()
    plt.show()

    print("Done!")

# SpecKit: Python library for dynamic-resolution spectral analysis
[Miguel Dovale](https://orcid.org/0000-0002-6300-5226)

**SpecKit** is a high-performance Python toolkit for advanced spectral analysis, designed for scientists and engineers who require high-fidelity, high-dynamic-range spectral estimates.

The library's core feature is its **logarithmically-spaced frequency scheduling algorithm**. Unlike traditional methods such as Welch’s, which use a fixed segment length and resolution across all frequencies, SpecKit dynamically adjusts the frequency resolution. The result is that, for a given dataset, this method provides many more spectral estimates with exceptionally high resolution at low frequencies, while minimizing statistical uncertainty (via more averages) at high frequencies. This method is ideally suited for analyzing high dynamic range signals with features spanning many decades.

Built on rigorous, textbook-backed theory, SpecKit also provides **comprehensive error analysis** for all its estimates, making it a reliable tool for scientific research.

---

## Key Features

- **Logarithmic Frequency Scheduling**  
  Dynamically adapts segment length and averaging to deliver optimal resolution and stability across the entire frequency spectrum.

- **High-Performance Engine**  
  Leverages `numba` for JIT-compiling and parallelizing performance-critical code.

- **Comprehensive Spectral Quantities**  
  Calculates a full suite of auto-spectral and cross-spectral estimates, including:  
    - Power Spectral Density (PSD) and Amplitude Spectral Density (ASD)  
    - Cross Spectral Density (CSD)  
    - Transfer Functions
    - Coherence  
  - Conditioned spectra for optimal spectral analysis of MISO/SISO systems  

- **Rigorous Error Analysis**  
  Provides analytically derived standard deviations and normalized random errors for all key estimates, based on the work of Bendat & Piersol.

- **Advanced DSP Utilities**  
  Includes tools for multi-file data loading/resampling, high-order Lagrange interpolation for time shifting signals by variable fractional delays, optimal linear combination for noise subtraction, and specialized windowing functions.

- **High-Quality Noise Generation**  
  Ultra-fast, numerically stable generators for white, red (Brownian), pink, and general 1/f^α colored noise.

- **Publication-Ready Plotting**  
  Built-in, user-friendly plotting interface for creating clear and professional figures.

---

## Installation

Install SpecKit directly from PyPI:

```bash
pip install speckit
```

For development, clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/mdovale/speckit.git
cd speckit
pip install -e .[dev]
```

---

## Quickstart

The main user-facing function is `speckit.compute_spectrum`.  
Here’s a simple example of computing the Amplitude Spectral Density (ASD) of a noisy signal and comparing it to SciPy’s Welch method:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from speckit import compute_spectrum

# 1. Generate sample data
N = int(1e6)  # Length of the time series
fs = 2.0      # Sampling frequency
x = np.random.randn(N)

# 2. Calculate spectrum with SciPy’s Welch method for comparison
f_welch, psd_welch = welch(x, fs, window="hann", nperseg=N // 8)

# 3. Calculate spectrum with SpecKit
# This single function call configures, plans, and computes the spectrum.
result = compute_spectrum(
    x,
    fs=fs,
    Jdes=1000,      # Target ~1000 log-spaced frequency bins
    Kdes=100,       # Scheduler control parameter for averaging
    win="Kaiser",   # Use a Kaiser window
    psll=200,       # with 200 dB peak side-lobe suppression
)

# 4. Plot the results
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"ASD (units / $\sqrt{\rm Hz}$)")
ax.set_title("SpecKit vs. Welch’s Method")

# Welch’s result (convert PSD to ASD)
ax.loglog(f_welch, np.sqrt(psd_welch), label="Welch", color="gray", alpha=0.7)

# SpecKit result with 1σ error band
result.plot(which="asd", errors=True, sigma=1, ax=ax, label="SpecKit", color="crimson")

ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
ax.set_ylim(5e-4, 2)
fig.tight_layout()
plt.show()
```

---

## References and Acknowledgements

1. **General Spectral Analysis and Uncertainty**  
   - Bendat, Piersol. *Random Data: Analysis and Measurement Procedures*. John Wiley & Sons.  
   - Bendat, Piersol. *Engineering Applications of Correlation and Spectral Analysis*. John Wiley & Sons.  
   - Primary reference for spectral definitions, interrelationships, and formulas for calculating random errors.

2. **Logarithmic Frequency Scheduling**
   - Tröbs, M., Heinzel, G. *Improved spectrum estimation from digitized time series on a logarithmic frequency axis*. [doi:10.1016/j.measurement.2005.10.010](https://doi.org/10.1016/j.measurement.2005.10.010)
   - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). *Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows*. Max-Planck-Institut für Gravitationsphysik.  
   - Basis for SpecKit’s `lpsd_plan` and `ltf_plan` scheduler, inspired by algorithms developed for LISA Pathfinder data analysis.

3. **Colored Noise Generation**  
   - Plaszczynski, S. (2007). *Generating long streams of 1/f^α noise*. Fluctuation and Noise Letters. [doi:10.1142/S0219477507003635](https://doi.org/10.1142/S0219477507003635)  
   - Provides the stable cascade-of-filters method used in SpecKit.
   - Numba-optimized version of original code by [Jan Waldmann](https://github.com/janwaldmann/pyplnoise)

4. **High-order Lagrange Interpolation**
   - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8429119.svg)](https://doi.org/10.5281/zenodo.8429119)
   - `dsp.timeshift` and `dsp.lagrange_taps` adopted from the PyTDI project. Credit to: [Staab, Martin](https://orcid.org/0000-0001-5036-6586), [Bayle, Jean-Baptiste](https://orcid.org/0000-0001-7629-6555), [Hartwig, Olaf](https://orcid.org/0000-0003-2670-3815)

---

## License

This project is licensed under the **BSD 3-Clause License**.  
See the LICENSE file for details.

---

## Contributing

Contributions are welcome!  
If you have a bug report, feature request, or suggestion, please open an issue on the GitHub repository.

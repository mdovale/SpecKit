<p align="center">
  <img src="https://raw.githubusercontent.com/mdovale/spectools/main/docs/images/logo.png" alt="Speckit Logo" width="200"/>
</p>

<h1 align="center">SpecKit</h1>
<p align="center">A Python toolkit for dynamic-resolution spectral analysis</p>

<!-- # SpecKit: Python library for dynamic-resolution spectral analysis -->
[Miguel Dovale](https://orcid.org/0000-0002-6300-5226)  
[Gerhard Heinzel](https://orcid.org/0000-0003-1661-7868)

**SpecKit** is a high-performance Python toolkit for advanced spectral analysis, designed for scientists and engineers who require high-fidelity, high-dynamic-range spectral estimates.

The library's core feature is its **logarithmically-spaced frequency scheduling algorithm**. Unlike traditional methods such as Welch’s, which use a fixed segment length and resolution across all frequencies, SpecKit dynamically adjusts the frequency resolution. The result is that, for a given dataset, this method provides many more high-resolution spectral estimates at low frequencies, while minimizing statistical uncertainty (via more averages) at high frequencies. This method is ideally suited for analyzing signals that span many decades in frequency.

Built on rigorous, textbook-backed theory, SpecKit also provides **comprehensive error analysis** for all its estimates, making it a reliable tool for scientific research.

---

## Key Features

- **Logarithmic Frequency Scheduling**  
  Dynamically adapts segment length and averaging to deliver optimal resolution and stability across the entire frequency spectrum.

- **High-Performance Engine**  
  Leverages [`numba`](https://numba.pydata.org/) for JIT-compiling and parallelizing performance-critical code. Provides pure NumPy fallbacks for full portability.

- **Comprehensive Spectral Quantities**  
  Calculates a full suite of auto-spectral and cross-spectral estimates, including:  
    - Power Spectral Density (PSD) and Amplitude Spectral Density (ASD)  
    - Cross Spectral Density (CSD)  
    - Transfer Functions and Coherence  
    - Conditioned spectra for MISO/SISO systems  

- **Rigorous Error Analysis**  
  Provides analytically derived **standard deviations** and **normalized random errors** for all key estimates, following Bendat & Piersol.  
  Additionally, SpecKit computes **empirical errors** directly from segment scatter using Welford-style accumulation (see below).

- **Advanced DSP Utilities**  
  Includes tools for:  
  - Multi-file data loading/resampling  
  - High-order Lagrange interpolation for fractional time shifting  
  - Optimal linear combination for noise subtraction  
  - Specialized flat-top windows with known Equivalent Noise Bandwidth (ENBW)

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

For development:

```bash
git clone https://github.com/mdovale/speckit.git
cd speckit
pip install -e '.[dev]'
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

N = int(1e6)  # samples
fs = 2.0      # Hz
x = np.random.randn(N)

# Welch (fixed resolution)
f_welch, psd_welch = welch(x, fs, window="hann", nperseg=N // 8)

# SpecKit (dynamic resolution)
result = compute_spectrum(x, fs=fs, Jdes=1000, Kdes=100, win="Kaiser", psll=200)

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.loglog(f_welch, np.sqrt(psd_welch), label="Welch", color="tomato", alpha=0.7)
result.plot(which="asd", errors=True, sigma=3, ax=ax, label="SpecKit", color="royalblue")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel(r"ASD (units / $\sqrt{\rm Hz}$)")
ax.legend()
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
```

![Example spectrum](https://raw.githubusercontent.com/mdovale/spectools/main/docs/images/example.png)

---

## API Overview

### Top-level entry point
- **`speckit.compute_spectrum(data, fs, **kwargs)`**  
  Configures, plans, and executes a spectral analysis in one line.

### Core classes
- **`SpectrumAnalyzer`**  
  Configuration + computation object.  
  Methods:
  - `.plan()` → returns frequency/segmentation plan.
  - `.compute()` → full spectrum analysis.
  - `.compute_single_bin(freq, fres or L)` → analyze a single frequency bin.

- **`SpectrumResult`**  
  Immutable container with all results and helper methods.  
  Attributes are lazily computed and cached on demand:
  - `psd`, `asd`, `csd`, `tf`, `coh`, `ENBW`, …
  - Errors: `Gxx_dev`, `Hxy_dev`, `coh_error`, etc.
  - Empirical errors: `XY_emp_dev`, `Gxy_emp_dev`, …
  - Utilities: `.plot()`, `.to_dataframe()`, `.get_rms()`, `.get_measurement()`.

---

## Performance Strategies

SpecKit is designed for datasets with millions of samples. Performance comes from:

1. **Numba JIT Kernels**  
   - Core bin-by-bin statistics are computed with hand-optimized Goertzel recurrences.
   - Parallelized across segments using `numba.prange`.

2. **Selective Detrending**  
   - Order = `-1`: windowing only. 
   - Order = `0`: mean removal.
   - Order = `1,2`: polynomial detrend via orthonormal basis (`_build_Q`).
   - The detrend mode changes which JIT kernel is dispatched.

3. **Window/Polynomial Caching**  
   - Expensive objects like windows and polynomial bases are cached per segment length.

4. **Fallback to NumPy**  
   - If Numba is unavailable, pure-NumPy helpers vectorize segment processing.
   - NumPy path uses highly optimized `polyfit`/`polyval` for detrending.

5. **Single-Bin Evaluation**  
   - `.compute_single_bin()` avoids planning full spectra and is highly efficient for probing specific frequencies.

---

## Error Analysis and Welford’s Algorithm

### Textbook formulas
SpecKit implements Bendat & Piersol-style analytic formulas:
- Variances of PSD, CSD, and coherence estimates scale as `1/√K`, where `K` is the number of averages.
- Coherence errors scale with `(1 - γ²) / √K`.

### Empirical scatter (M2)
In addition, the JIT kernels compute:
- **`MXX`, `MYY`**: mean segment powers.
- **`M2`**: mean squared deviation of per-segment cross-spectra about the mean.

This is essentially Welford’s algorithm for streaming variance, adapted to complex quantities.  

From `M2`, SpecKit derives:
- **`XY_emp_var`**: variance of the averaged cross-spectrum.
- **`XY_emp_dev`**: empirical standard deviation.
- **`Gxx_emp_dev`**, **`Gxy_emp_dev`**: empirical deviations mapped into spectral density units.

These provide **non-parametric uncertainty estimates**, complementing the analytic formulas. They are particularly useful when:
- The Gaussian assumptions of textbook theory may not hold.
- Segments are correlated due to overlap.
- One wants to validate that statistical convergence is as expected.

### Why both?  
- **Analytic errors** are fast, interpretable, and match the literature.  
- **Empirical errors** reflect actual scatter in the data.  
Together, they provide a complete picture of uncertainty.

---

## References and Acknowledgements

1. **General Spectral Analysis and Uncertainty**  
   - Bendat, Piersol. *Random Data: Analysis and Measurement Procedures*. John Wiley & Sons.  
   - Bendat, Piersol. *Engineering Applications of Correlation and Spectral Analysis*. John Wiley & Sons.  

2. **Logarithmic Frequency Scheduling**
   - Tröbs, M., Heinzel, G. *Improved spectrum estimation from digitized time series on a logarithmic frequency axis*. [doi:10.1016/j.measurement.2005.10.010](https://doi.org/10.1016/j.measurement.2005.10.010)
   - Heinzel, G., Rüdiger, A., & Schilling, R. (2002). *Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows*. Max-Planck-Institut für Gravitationsphysik.  
   - Basis for SpecKit’s `schedulers.lpsd_plan` and `schedulers.ltf_plan`, inspired by algorithms developed for LISA Pathfinder data analysis.

3. **Colored Noise Generation**  
   - Plaszczynski, S. (2007). *Generating long streams of 1/f^α noise*. Fluctuation and Noise Letters. [doi:10.1142/S0219477507003635](https://doi.org/10.1142/S0219477507003635)  
   - Provides the stable cascade-of-filters method used in SpecKit.
   - Numba-optimized version of original code by [Jan Waldmann](https://github.com/janwaldmann/pyplnoise)

4. **High-order Lagrange Interpolation**
   - [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8429119.svg)](https://doi.org/10.5281/zenodo.8429119)
   - `dsp.timeshift` and `dsp.lagrange_taps` adopted from the PyTDI project. Credit to: [Staab, Martin](https://orcid.org/0000-0001-5036-6586), [Bayle, Jean-Baptiste](https://orcid.org/0000-0001-7629-6555), [Hartwig, Olaf](https://orcid.org/0000-0003-2670-3815)

---

## License

BSD 3-Clause License. See the LICENSE file.

---

## Contributing

Contributions are welcome!  
Please open issues or PRs for bug reports, feature requests, or ideas.

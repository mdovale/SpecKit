# Changelog

Notable changes to SpecKit. Versions follow semantic versioning. A change to
the numbers a seeded generator or estimator returns is a behavioral change
and gets at least a minor version bump.

## Unreleased

### Changed (behavioral)

- Non-finite input samples raise `ValueError` by default
  (`missing="raise"`). The message includes the count.
  `missing="zero"` is the 1.0.3 behavior: replace those samples with 0 and
  record the count on `SpectrumAnalyzer.n_replaced`. Bin outputs and the
  derived error properties are no longer passed through `numpy.nan_to_num`.
  On a finite series, `Gxx`, `Gxy`, `nu`, and the plan are unchanged.

### Added

- `SpectralPlan` is an immutable copy of a scheduler dict plus the window,
  detrend order, resolved overlap, missing-sample policy, and the one-sided
  `X*conj(Y)` convention. `SpectrumAnalyzer.plan()` still returns the dict.
  `SpectrumAnalyzer.spectral_plan()` returns the object. `digest()` is
  SHA-256 of a documented little-endian layout. `report()` states boundary
  counts and does not edit the plan.
- `BinResponse` evaluates one bin's direct-DFT response. For `order=-1`
  its ENBW is `fs * S2 / S1**2`. For a detrend, the reported ENBW is the
  projected analysis tap.
- `compute_single_bin(..., storage="replicates")` keeps per-segment direct
  coefficients. The default remains `storage="summary"`.
- `SpectrumResult.gxx_interval(conf=0.95)` returns the pointwise chi-square
  bounds on the one-sided PSD. `nu` is the unrounded overlap-corrected
  degrees of freedom. The interval assumes a locally white Gaussian
  process, ignores the detrend, and is not a simultaneous band. See
  `docs/reference/spectral-contract.md`.

## 1.0.3 — 2026-10-24

### Changed (behavioral)

- `speckit.noise.alpha_noise` and `pink_noise` now meet their documented
  two-sided level, `S2(f) = f**-alpha`, 1.0 at 1 Hz. The scale now comes from
  the filter cascade's own transfer function instead of `1 / f_max**(alpha/2)`.
  The old output was high by up to 13% in PSD (about 6% in amplitude): about
  13% at `alpha = 1`, 10% at `alpha = 0.5` and `1.5`, and correct at
  `alpha = 2`. For a given seed, realizations change by that factor. The
  spectral shape is unchanged.
- Analytic and empirical errors divide by the overlap-corrected segment
  count `n_d = nu / 2`. `SpectrumResult.nu` is Percival & Walden (1993),
  eq. 292b, from the window and the actual segment starts. With no overlap,
  `nu = 2 * navg` and these errors are unchanged. With overlap the old
  errors, which divided by `navg`, were small: about 1.39× for a Hann window
  at 75% overlap, and about 1.06× for Kaiser `psll=200` at overlap 0.766.
  `plot(..., errors=True)` uses `Gxx_dev`, so those bars move with it.
  `Gxx`, `Gxy`, `navg`, and `M2` are unchanged. The affected properties are
  `Gxx_dev`, `Gyy_dev`, `Gxy_dev`, `Hxy_dev`, `coh_dev`, the `*_error`
  properties, `XY_emp_var`, `XY_emp_dev`, `Gxx_emp_dev`, and `Gxy_emp_dev`.
- `Hxy_rad_error` is `sqrt(|1 - coh|) / sqrt(coh * 2 * n_d)`, the same value
  as `Hxy_mag_error`. `Hxy_deg_error` is that value in degrees. The previous
  numerator was `arcsin(sqrt(|1 - coh|))`, high by about 11% at coherence 0.5
  and 24% at 0.2, before the overlap correction above.
- `GyySx` is the coherent residual, the same array as `GyyRx`:
  `max(1 - coh, 0) * Gyy`, which is `Gyy - |Gxy|**2 / Gxx`. The old expansion
  matched that residual only at zero CSD phase. On a delayed two-channel
  series the median ratio on mid-coherence bins was about 7.5.
  `SISO_optimal_spectral_analysis` returns `sqrt(GyySx)`, so that ASD changes
  with it. The MISO residuals already used this quantity. The floor at 0 also
  removes a roundoff-negative `GyyRx` on single-segment bins.

### Documentation

- `white_noise`: the docstring RMS is now `sqrt(psd * f_sample)`, matching
  the code. It previously read `sqrt(psd * f_sample / 2)`.
- `speckit.noise` and `compute_spectrum` now state the PSD conventions. Generators
  are two-sided, `compute_spectrum` is one-sided, and `Gxx = 2 * S2` for a
  real signal. Both include a round-trip example.

### Fixed

- The NumPy bin kernels correlate with `exp(-i ω n)`, the same Fourier sign
  as the Goertzel kernels. Auto-spectra already agreed. On the fallback path
  the cross-spectrum was conjugated: a 3-sample delay at 40 Hz read
  `+144.181°` instead of `-144.181°`. `compute` and `compute_single_bin` both
  follow `speckit.core._NUMBA_ENABLED`. With Numba left on, `Gxy` is unchanged.

### Packaging

- Required dependencies are NumPy, SciPy, Matplotlib, and Numba. pandas and
  py7zr are the `io` extra. `to_dataframe`, the DataFrame helpers, and `.7z`
  loading import them when called and name that extra if they are missing.
  `python-control` is no longer required; `cf_db` is `20 * log10`, the same
  conversion. `tqdm` is not required.
  `MISO_analytic_optimal_spectral_analysis` calls the numeric solver. SymPy
  is not imported, and the residual matches the previous solver.
- Flat-top window samples are unchanged and are released under the project
  BSD-3-Clause license. The coefficients are the published values from
  Heinzel, Rüdiger, and Schilling (2002). `lagrange_taps` and `timeshift`
  keep the 2022 Caltech and AEI copyright and stay available from
  `speckit.dsp`. See `NOTICE`.

## 1.0.2

Last release before this changelog was started.

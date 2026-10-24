# Handoff: lean library reorganization for publication

## Scope

Reorganize SpecKit so the published library is the spectral estimator, with a
small required dependency set, while every working utility that notebooks and
tests already call remains importable.

This is a packaging and module-boundary change. It is not a new spectral
estimator, and it is not a license to retune a PSD scale.

Out of scope: `new_ltf_plan`, the noise-generator normalization handoff, and
the LTPDA feature-gap series. Do not fold those into this reorganization.

## Goal

`import speckit` and `compute_spectrum` should need only NumPy, SciPy, Numba,
and Matplotlib. Pandas, `python-control`, `py7zr`, `tqdm`, and SymPy must not
be imported on that path.

DataFrame helpers, archive loading, and the SISO/MISO optimal-subtraction
functions stay. Their numeric results stay the ones the current tests lock.

## Current status

Import split and the shared MISO solver are in the working tree. Not
committed. The manuscript architecture paragraph has not been edited.

`import speckit`, `import speckit.dsp`, and `import speckit.systems` do not
import pandas, `control`, `py7zr`, SymPy, or `tqdm`. Required dependencies in
`pyproject.toml` are NumPy, SciPy, Matplotlib, and Numba. Extra `io` is
pandas and `py7zr`. `tqdm` is in the `dev` extra. `control` is gone. SymPy
is not imported.

`MISO_analytic_optimal_spectral_analysis` calls
`MISO_numeric_optimal_spectral_analysis`.
`test_miso_analytic_vs_numeric_consistency` still passes at decimal 8 after
the first four bins.

SISO versus one-input MISO: explained and fixed on 24 Oct 2026. The old
`GyySx` expansion swapped `Hxy` and `Hyx` in its cross terms (see
`docs/handoffs/20261024_gyysx-residual.md`). `GyySx` is now
`Gyy - |Gxy|²/Gxx`, and `test_siso_vs_miso_single_input` is enabled and
passes at decimal 8. Neither ASD was scaled. The numbers under **What was
tried** record the pre-fix state. Bin 0 there was the `navg == 1` bin, where
the correct residual is 0.

`requests` is still listed in `requirements.txt` and is still not imported
by `speckit/`.

## What was tried

Moved `crop_data` and `integral_rms` to `speckit/spectral.py` and re-exported
them from `speckit.dsp`. `SpectrumResult.get_rms` imports `integral_rms`
from `spectral`, so the estimator no longer imports `dsp.py`. The trapezoid
(`cumulative_trapezoid`, `initial=0`, last sample, then `sqrt`) is unchanged.

`ct.mag2db` is `20 * log10` at `cf_db` and at the Bode error bars. Matplotlib
is imported inside `SpectrumResult.plot`. pandas is imported inside
`to_dataframe`, the DataFrame helpers, and `time_histogram`. `py7zr` is
imported only when a `.7z` file is opened. Missing pandas or `py7zr` raises
`ImportError` naming the `io` extra.

`optimal_linear_combination` validates the DataFrame and calls
`optimal_linear_combination_arrays`. The time-domain objective and the Welch
frequency-domain objective are the same calls as before.
`test_optimal_linear_combination_siso` checks that the DataFrame wrapper and
the array function return the same coefficients and residual.
`resample_to_common_grid` interpolates with `resample_series`, which is
`numpy.interp`.

The MISO input matrix no longer uses `np.any` on a spectrum to decide
whether a diagonal was filled. Each input index is filled once, from the
first pair that contains it (`Gxx` if it is the first channel, `Gyy` if it
is the second). The `q == 1` diagonal is still the two-channel spectrum of
the input with itself. The `1e12` condition cutoff and the `pinv` fallback
are unchanged.

SISO versus one-input MISO, `tests/unit/test_systems.py`, session-scoped
`reproducible_rng` seed 42, `N = 10000`, `fs = 100`, Butterworth lowpass
cutoff 10 Hz order 4, additive noise scale 0.1, `Jdes=100`, `win="hann"`:

- Frequencies match (50 bins).
- `assert_array_almost_equal(..., decimal=8)` fails on all 50 bins.
- Bin 0: SISO `0.03347403608516972`, MISO `6.594084684475506e-10`.
- Max absolute difference `0.26424979`.
- Max relative difference about `5.08e7`, driven by that near-zero MISO bin.

A fresh seed-42 series (not the session-scoped generator, so earlier tests
have not drawn from it) gives the same shape: frequencies match, bin 0 is
`0.03273097897987125` versus `1.8626992288909132e-09`, and the median
SISO/MISO ratio on bins `1:` is about `3.36`. This is not a factor of 2.
Stopped. No scale was applied.

`python -m pytest tests/unit/test_dsp.py` with pandas blocked
(`sys.modules["pandas"] = None` before pytest): 19 passed, 11 skipped.
The skips are `pytest.importorskip("pandas")` on the DataFrame tests.
With pandas installed, the same file passes, including those 11.

## Problem

The manuscript (`docs/paper/main.tex`, architecture section) describes four
pieces: `analysis.py`, `schedulers.py`, `core.py`, and a utility layer
(`noise.py`, `dsp.py`, `flattop.py`, `plotting.py`). The implementation
matches that for the estimator. The utility layer does not. `dsp.py` (about
1450 lines) mixes a few spectral helpers with a personal time-series toolkit.
`systems.py` implements a real Bendat calculation through an ad-hoc driver.
`utils.py` is already on the estimator path and does not need a rewrite.

The publication problem is the import graph, not the existence of the helpers.

`speckit/__init__.py` imports `SpectrumAnalyzer` from `analysis.py` and
`default_rc` from `plotting.py`. That pulls, at import time:

| Import | Why it loads | Used by `compute_spectrum`? |
| --- | --- | --- |
| `pandas` | `analysis.py` (`to_dataframe`), `plotting.py` (module level), `dsp.py` (module level) | No. `to_dataframe` and two plot helpers only. |
| `control` | `analysis.py` module level | No. `ct.mag2db` is `20*log10`, used for `cf_db` and Bode. |
| `matplotlib` | `analysis.py` and `plotting.py` | No for the numbers. Yes for `SpectrumResult.plot` and `default_rc`, which the paper shows. |
| `py7zr`, `zipfile`, `tarfile`, `gzip` | `dsp.py` module level, because `analysis.py` imports `integral_rms` from `dsp` | No. Archive reading inside `multi_file_timeseries_loader` only. |
| `scipy.optimize`, `scipy.signal.welch` | `dsp.py` module level | No. `peak_finder` and `optimal_linear_combination` only. |
| `scipy.integrate.cumulative_trapezoid` | `dsp.py` | Yes, indirectly: `SpectrumResult.get_rms` calls `integral_rms`. |

So a user who only wants a spectrum currently installs pandas and a 7-zip
library, and `import speckit` executes them. `tqdm` is required and is used
only in `tests/integration/test_errors.py`.

## Context

### What is already the library

Leave the formulas and the Numba/NumPy pairing alone:

- `speckit/core.py` — Goertzel bin statistics, segment detrending through
  `_build_Q`, Numba kernels and `_np` fallbacks.
- `speckit/schedulers.py` — `lpsd_plan`, `ltf_plan`, `vectorized_ltf_plan`.
  Leave `new_ltf_plan` as it is.
- `speckit/analysis.py` — `compute_spectrum`, `lpsd`, `compute_single_bin`,
  `SpectrumAnalyzer`, `SpectrumResult`. One-sided densities. CSD is
  `Pxy = X * conj(Y)`.
- `speckit/flattop.py` — Heinzel flat-top windows, `olap_dict`, `win_dict`.
- `speckit/noise.py` — generators. Their `psd` arguments are two-sided. That
  convention is a separate open issue
  (`docs/handoffs/20260905_noise-generator-normalization.md`).
- `speckit/utils.py` — `kaiser_alpha`, `kaiser_rov`, `round_half_up`,
  `find_Jdes_binary_search`, `is_function_in_dict`, `get_key_for_function`,
  `chunker`. `analysis.py` imports the first six. No pandas. No rewrite.

### `dsp.py`: three different jobs in one module

NumPy or SciPy helpers the estimator or the paper actually use:

- `integral_rms(f, asd, pass_band=None)` — RMS from a one-sided ASD,
  `sqrt(∫ ASD² df)` via `cumulative_trapezoid` with `initial=0`, taking the
  last sample. `SpectrumResult.get_rms` calls it. Keep this integral.
- `timeshift`, `lagrange_taps` — fractional delay. Defined in
  `speckit/lagrange.py` under the PyTDI copyright and re-exported from `dsp`.
  Notebook `notebooks/0.1_time-delay-bias-error.ipynb` and the time-delay
  application in the paper depend on `timeshift`. Unit tests in
  `tests/unit/test_lagrange.py` lock integer and fractional shifts.
- `polynomial_detrend` — one polynomial, `np.polyfit` / `np.polyval`, on the
  whole record. This is not the per-segment orthonormal detrend in
  `core._build_Q`. Both stay. A later session must not replace one with the
  other to make a plot match.
- `frequency2phase`, `crop_data`, `truncation`, `peak_finder` — small array
  helpers with unit tests. `peak_finder` uses `scipy.optimize.curve_fit`.
  Keep them as array functions. They are not part of the estimator.

Pandas wrappers. The statistic is in the array function underneath. The
DataFrame is only the container:

- `df_detrend` wraps `polynomial_detrend`.
- `df_timeshift` wraps `timeshift`.
- `optimal_linear_combination` requires a DataFrame. It minimizes
  `RMS[output + Σ c_i * shift(input_i)]` with `scipy.optimize.minimize`.
  Time-domain RMS is `sqrt(mean(square))`. Frequency-domain RMS calls
  `scipy.signal.welch(..., scaling="density")` and integrates that spectrum.
  That Welch call is not `compute_spectrum`. Do not point it at
  `compute_spectrum` during this reorganization: Welch's default one-sided
  density and the LPSD one-sided density are different estimators, and the
  coefficients would move. Notebook `notebooks/1.1_optimal-linear-combination.ipynb`.
- `resample_to_common_grid`, `multi_file_timeseries_loader`,
  `multi_file_timeseries_resampler` — overlapping time window, interpolation
  onto one grid, optional per-stream shift. The loader reads plain text,
  zip, tar, gzip, and 7z (`py7zr`). Notebook
  `notebooks/0.8_multi-file-processing.ipynb`. No unit test covers the loader
  or the resampler.

### `systems.py`: keep the calculation, replace the driver

All three functions return `(f, ASD)` and pass `**kwargs` through to
`compute_spectrum`. The ASD is one-sided: it is `sqrt` of a residual built
from `Gxx`, `Gyy`, and `Gxy`.

- `SISO_optimal_spectral_analysis` is `f` and `sqrt(GyySx)` from one
  two-channel `compute_spectrum`. `GyySx` already lives on `SpectrumResult`.
- `MISO_numeric_optimal_spectral_analysis` builds the input spectral matrix
  `T` and the input–output vector `S` from repeated `compute_spectrum` calls,
  then solves `T H = S` per frequency with `np.linalg.solve`, falling back
  to `pinv` when `cond > 1e12` or the solve fails. The residual is Bendat
  and Piersol, *Engineering Applications of Correlation and Spectral
  Analysis*, eq. 8.16, p. 191:
  `S00 - Σ H_i S0i - Σ conj(H_i) Si0 + Σ conj(H_j) H_i T_ji`.
  The published ASD is `abs(sqrt(that))`.
- `MISO_analytic_optimal_spectral_analysis` does the same residual, but
  builds `H` by calling `sympy.solve` on a fresh symbolic system at runtime
  and `lambdify`. The two ASDs agree in
  `test_miso_analytic_vs_numeric_consistency` to 8 decimal places after
  dropping the first four bins (`assert_array_almost_equal(..., decimal=8)`).

The numeric solve is the first-principles form. The symbolic solve is an
implementation detail that adds an undeclared dependency and recomputes a
linear system SymPy can only turn back into arithmetic. A commented-out test
at the bottom of `tests/unit/test_systems.py` would also check SISO against
one-input MISO. It is not running. Do not turn it on until both sides share
one frequency grid and one residual formula; a failure there is a real
discrepancy to report, not a tolerance to widen.

`systems.py` imports `compute_spectrum` via `from speckit import ...`, which
runs `speckit/__init__.py`. After the import cleanup, import it from
`speckit.analysis` so loading `systems` does not depend on package-level
side effects.

The diagonal of `T` in the numeric solver is filled with
`if not np.any(Tmat[i, i, :])` while walking cross-spectra. A spectrum that
is identically zero on a later input would be treated as "not yet filled".
When rewriting the driver, index the autospectra explicitly. Do not copy
that condition.

### Plotting

`plotting.py` is Matplotlib and is part of the paper. `time_plot` accepts a
DataFrame or an array. `time_histogram` requires a DataFrame and
`pd.to_datetime`. Those two functions are the only reason `plotting.py`
imports pandas at import time. `default_rc` does not need pandas.

`SpectrumResult.plot` and `cf_db` live in `analysis.py`, which is why
Matplotlib and `control` load with the estimator.

### PSD convention this work must keep

- `compute_spectrum`, `GyyCx` / `GyyRx` / `GyySx`, `integral_rms`, and the
  systems ASDs are one-sided.
- Noise-generator `psd` arguments are two-sided. Do not "fix" that here.
- `optimal_linear_combination(..., domain="frequency")` integrates a SciPy
  Welch density. Leave that call on Welch unless a separate handoff defines
  the LPSD objective and a test for the coefficients.

## Settled decisions

1. Required runtime dependencies become NumPy, SciPy, Numba, and Matplotlib.
   Matplotlib stays required because the paper's first snippet ends in
   `obj.plot` and the package exports `default_rc`.
2. pandas, `control`, `py7zr`, and `tqdm` leave the required set. `tqdm`
   moves to the `dev` extra. SymPy is not added.
3. No public function listed below is deleted, renamed, or given a new
   required argument. Notebooks `0.1`, `0.8`, `1.0`, and `1.1` must keep
   importing the same names.
4. DataFrame and archive code may move to a new module (`speckit/io.py` is
   the natural place) only if `speckit.dsp` still exports the old names.
5. pandas and `py7zr` are imported inside the functions that use them, so a
   missing extra fails at the call, with a message that names the extra,
   rather than at `import speckit`.
6. `ct.mag2db(x)` becomes `20 * np.log10(x)` at the two call sites
   (`cf_db`, and the Bode error bars). Same value for positive magnitudes.
   The Bode path already floors magnitudes at `1e-300` before the conversion.
7. `MISO_analytic_optimal_spectral_analysis` and
   `MISO_numeric_optimal_spectral_analysis` both call one NumPy solver.
   The analytic name stays so existing callers do not break. It no longer
   imports SymPy.
8. `SISO_optimal_spectral_analysis` stays `sqrt(GyySx)` from one
   two-channel spectrum. Do not reimplement it as a second residual formula.
9. `polynomial_detrend` and `core._build_Q` remain different operations.
10. `integral_rms` keeps `sqrt` of the trapezoidal integral of `ASD**2`,
    including `initial=0` and the last-sample reduction. Moving the function
    next to `get_rms` is encouraged so `analysis.py` does not import `dsp.py`.
    Changing the quadrature is not part of this work.
11. Do not edit `core.py`, the scheduler formulas, `noise.py`, or `flattop.py`
    for this reorganization.

## Success criteria

- `import speckit` succeeds in an environment with NumPy, SciPy, Numba, and
  Matplotlib, and without pandas, `control`, `py7zr`, `tqdm`, or SymPy.
- `compute_spectrum` on a short white series still returns a one-sided PSD.
  `SpectrumResult.get_rms` still matches `integral_rms` on that ASD.
- `cf_db` matches `20 * log10(abs(Hxy))` on a two-channel result with
  nonzero `Gxx`.
- With the `io` extra installed (pandas and `py7zr`), the current
  `tests/unit/test_dsp.py` DataFrame tests pass unchanged, including
  `df_detrend`, `df_timeshift`, and `optimal_linear_combination`.
- `tests/unit/test_systems.py` passes without SymPy.
  `test_miso_analytic_vs_numeric_consistency` still meets decimal 8 after
  the first four bins. That comparison is the lock on the residual, not a
  tolerance to relax.
- `tests/unit/test_utils.py`, `tests/unit/test_dsp.py` (array tests), and
  `tests/integration/test_analysis_api.py::test_get_measurement_and_to_dataframe`
  pass. The DataFrame smoke test runs in the `io` extra job.
- `ruff check` and `ruff format --check` on every touched module.

There is no new numerical target. Agreement with the existing tests is the
target.

## Recommended next steps

1. Explain the SISO versus one-input MISO ASD before changing either
   formula. Frequencies already match. SISO returns `sqrt(GyySx)` from one
   two-channel spectrum. One-input MISO builds `T` from a two-channel
   spectrum of the input with itself, then applies eq. 8.16. Do not multiply
   either ASD by a constant.
2. Adjust the architecture paragraph in `docs/paper/main.tex` so the utility
   sentence names `spectral.py` and the optional DataFrame helpers. Do that
   in the manuscript session. Do not delete notebooks. The paper's sentence
   about a future `control` interface can stay.
3. `requests` in `requirements.txt` is unused by `speckit/`. Leave it unless
   a packaging pass is already editing that file.

The import split, the shared solver, and the array wrappers are done. Do not
reopen them unless the SISO/MISO explanation forces a real formula change.

## Test gaps

- `tests/unit/test_dsp.py::test_resample_series_on_overlap` locks
  `numpy.interp` on two monotone clocks and their overlap. It does not load
  pandas. The file loader and the DataFrame resampler still have no test.
- SISO versus one-input MISO is commented out again. See **What was tried**.
- `tests/integration/test_analysis_api.py::test_import_speckit_skips_optional_modules`
  starts a fresh interpreter and checks that pandas, `control`, `py7zr`,
  SymPy, and `tqdm` are absent from `sys.modules` after importing `speckit`,
  `speckit.dsp`, and `speckit.systems`.

## Files touched

- `speckit/spectral.py` — new. `crop_data` and `integral_rms`.
- `speckit/analysis.py` — `integral_rms` from `spectral`; Matplotlib and
  pandas imported at the call; `20 * log10` instead of `control`.
- `speckit/dsp.py` — re-exports the two spectral helpers; DataFrame and
  `py7zr` imports are local; `resample_series` and
  `optimal_linear_combination_arrays` added. Old names remain.
- `speckit/systems.py` — analytic entry point calls the numeric solver; no
  SymPy; `compute_spectrum` imported from `speckit.analysis`.
- `speckit/plotting.py` — pandas imported in `time_histogram`.
- `pyproject.toml` — required set and `io` / `dev` extras; repository URLs
  point at `https://github.com/mdovale/SpecKit`.
- `requirements.txt` — `control` removed. pandas, `py7zr`, and `tqdm` remain
  so a flat install still runs the DataFrame tests.
- `tests/unit/test_dsp.py`, `tests/unit/test_systems.py`,
  `tests/integration/test_analysis_api.py`.

Do not edit `speckit/core.py`, `speckit/schedulers.py`, `speckit/noise.py`,
or `speckit/flattop.py`.

## References

- `docs/paper/main.tex` — architecture list (analysis, schedulers, core,
  utilities) and the MISO application. The `control` and GWpy sentences are
  future work, not current imports to preserve.
- Bendat and Piersol, *Engineering Applications of Correlation and Spectral
  Analysis*, section 8.1, eq. 8.16, p. 191. ISBN 978-0-471-57055-4.
- `speckit/analysis.py` — `GyySx` is the SISO residual in spectral units.
- `tests/unit/test_dsp.py`, `tests/unit/test_systems.py`,
  `tests/unit/test_utils.py`,
  `tests/integration/test_analysis_api.py`.
- Notebooks `0.1_time-delay-bias-error.ipynb`,
  `0.8_multi-file-processing.ipynb`, `1.0_systems.ipynb`,
  `1.1_optimal-linear-combination.ipynb`.
- `docs/handoffs/20260905_noise-generator-normalization.md` — two-sided
  generator `psd` versus one-sided `compute_spectrum`. Do not mix that fix
  into this split.
- `.cursor/rules/no-hacky-workarounds.mdc` — do not absorb a residual
  mismatch with a scale factor or a wider tolerance.

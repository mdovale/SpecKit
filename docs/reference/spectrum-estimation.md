# Spectrum estimation

`compute_spectrum` plans a logarithmic frequency grid and returns a
`SpectrumResult` of one-sided auto-spectra, or of cross-spectra when the input
has two channels. `SpectrumResult` derives densities, a transfer function,
coherence, conditioned spectra, and both analytic and empirical errors from
the segment statistics.

## Goal

Estimate PSD, ASD, CSD, transfer function, coherence, and the two-channel
conditioned spectra of a uniformly sampled series, with a per-bin segment
count and a scatter statistic.

## Scope

In: `compute_spectrum`, `lpsd`, `compute_single_bin`, `SpectrumAnalyzer`, and
`SpectrumResult`.

Out: the frequency-plan recurrence
([frequency-scheduling.md](frequency-scheduling.md)), the Goertzel kernels
([bin-statistics.md](bin-statistics.md)), window coefficients
([windows.md](windows.md)), the RMS integral
([band-rms.md](band-rms.md)), and multi-input residual ASD
([optimal-subtraction.md](optimal-subtraction.md)). `SpectrumResult` has no
unit strings.

## Behavior and conventions

`data` is a 1-D series, or a `2×N` or `N×2` pair. Channel 0 is `X` and
channel 1 is `Y`. A square `2×2` array is read as two channels of length 2.
Non-finite samples are replaced with 0. `fs` is in hertz and must be a
positive finite float.

Defaults on `SpectrumAnalyzer`: `olap="default"`, `bmin=1`, `Lmin=1`,
`Jdes=500`, `Kdes=100`, `order=0`, `psll=200`, `win=numpy.kaiser`,
`scheduler="vectorized_ltf"`, `band=None`, `force_target_nf=False`. The
constructor docstring says the scheduler default is `"ltf"`. The signature
default is `"vectorized_ltf"`.

`order` selects the segment detrend: `-1` window only, `0` mean removal,
`1` linear, `2` quadratic. `band=(f_min, f_max)` keeps plan frequencies inside
that closed interval after the scheduler runs.

Densities are one-sided, in signal²/Hz for `psd` / `Gxx` / `Gxy`, and in
signal/√Hz for `asd`. On an auto result, `psd`, `asd`, `csd`, `coh`, and `tf`
that belong only to a cross result are `None`, and the reverse.

`K` is the segment count stored with the start indices `D`. `plan()` requires
`K[i]` to equal the length of `D[i]`. `navg` is that same count on the plans
this analyzer builds. Analytic and empirical errors do not divide by `navg`.
They divide by `n_d = nu / 2`, where `nu` is the overlap-corrected degrees of
freedom from [bin-statistics.md](bin-statistics.md). With no overlap,
`nu = 2 * navg` and `n_d = navg`.

`get_measurement` interpolates with `numpy.interp`, component-wise for complex
values, and clamps outside the tabulated frequencies. `get_rms` is auto-only
and integrates the frequencies on the result
([band-rms.md](band-rms.md)). `to_dataframe` imports pandas when called.
That import is the `io` extra in `pyproject.toml` (`pandas`, `py7zr`).

`plot` defaults to `asd` for an auto result and `bode` for a cross result.
With `errors=True`, the ASD band uses `Gxx_dev / (2 * sqrt(Gxx))`, with `Gxx`
floored at `1e-300` inside the square root.

## Algorithm

For each planned frequency `f`, segment length `L`, and start list `D`, the
analyzer builds a window `w` of length `L` and calls `_bin_stats` at
`omega = 2 π f / fs`. The kernel returns `BinStats`: mean powers `XX` and
`YY`, mean cross-spectrum `XY`, and scatter `M2`. `_overlap_dof(w, starts)`
stores `nu` on the same row.

Window sums, with `S1 = sum(w)` and `S2 = sum(w²)`, are stored as
`S12 = S1²` and `S2`. The one-sided scaling is

```
Gxx = 2 * XX / (fs * S2)
Gyy = 2 * YY / (fs * S2)
Gxy = 2 * XY / (fs * S2)
ENBW = fs * S2 / S12
```

`XY` is the mean of `X * conj(Y)` across segments. Derived quantities:

```
asd = sqrt(Gxx)          # auto only
ps  = Gxx * ENBW         # auto only
cs  = Gxy * ENBW
Hxy = conj(XY) / XX      # estimate of Y/X
Hyx = conj(Hxy)
coh = |XY|² / (XX * YY)  # magnitude-squared coherence
ccoh = XY / sqrt(XX * YY)
```

`cf` is `|Hxy|`. `cf_db` is `20 * log10(cf)`. Phase attributes are
`angle(Hxy)` in radians or degrees, with an unwrapped pair.

Conditioned spectra, cross results only:

```
GyyCx = coh * Gyy
GyyRx = max(1 - coh, 0) * Gyy
GyySx = GyyRx
```

`GyyRx` and `GyySx` are the same array. The code comment identifies that
product with `Gyy - |Gxy|² / Gxx`. The floor at 0 is for a `1 - coh` that
went negative by roundoff, which the comment says happens at `navg == 1`.

Analytic errors use `n_d = nu / 2`. `Gxx_dev = Gxx / sqrt(n_d)`, which is
`Gxx * sqrt(2 / nu)`.

```
Gxx_error = 1 / sqrt(n_d)
Gxy_dev   = sqrt(|Gxy|² / (coh * n_d))
Gxy_error = 1 / sqrt(coh * n_d)
Hxy_dev   = |Hxy| * sqrt(|1 - coh|) / sqrt(coh * 2 * n_d)
Hxy_mag_error = sqrt(|1 - coh|) / sqrt(coh * 2 * n_d)
Hxy_rad_error = Hxy_mag_error
Hxy_deg_error = rad2deg(Hxy_rad_error)
coh_dev   = sqrt(|(2 * coh / n_d) * (1 - coh)²|)
coh_error = sqrt(2) * (1 - coh) / (sqrt(coh) * sqrt(n_d))
```

`Gyy_*` follows `Gxx_*` on an auto result and the same `1/sqrt(n_d)` factor
on a cross result. Empirical scatter divides `M2` by `n_d` rather than by
`navg`:

```
XY_emp_var = M2 / n_d
XY_emp_dev = sqrt(M2 / n_d)
Gxx_emp_dev = 2 / (fs * S2) * XY_emp_dev    # auto only
Gxy_emp_dev = 2 / (fs * S2) * XY_emp_dev    # cross only
```

`compute_single_bin(freq, fres=... or L=...)` skips the full plan. Segment
length is the supplied `L`, or `round(fs / fres)`. Starts use the same overlap
spacing as a single plan row, with `K = navg`. It calls `_bin_stats`, so it
follows the same Numba or NumPy choice as `compute()`.

## Numerical checks

`tests/integration/test_accuracy.py` requires, above 1 Hz, a median relative
PSD error below 0.1, a median absolute coherence error below 0.05, and a
median magnitude of the complex transfer-function error, divided by `|H|`,
below 0.2. `tests/integration/test_analysis_api.py` locks SciPy CSD phase
within a median of 5 degrees where `coh > 0.2`, single-bin agreement with the
full run, band masking, and finite empirical statistics.
`tests/integration/test_errors.py` compares analytic errors with a Monte Carlo
ensemble.

## Key files

- `speckit/analysis.py` — `SpectrumAnalyzer`, `SpectrumResult`, `compute_spectrum`
- `speckit/__init__.py` — public exports and thread limits
- `tests/integration/test_analysis_api.py`, `test_accuracy.py`, `test_errors.py`
- `notebooks/0.0_quickstart.ipynb`, `notebooks/0.2_cross-spectrum.ipynb`,
  `notebooks/0.5_errors.ipynb`, `notebooks/0.9_conditioned-spectrum.ipynb`

## Related docs

- [frequency-scheduling.md](frequency-scheduling.md),
  [bin-statistics.md](bin-statistics.md), [windows.md](windows.md),
  [band-rms.md](band-rms.md)
- `docs/handoffs/20261024_phase-error-arcsin.md` — described an `arcsin` in
  `Hxy_rad_error`. The code now sets that error equal to `Hxy_mag_error`.
- `docs/handoffs/20261024_overlap-degrees-of-freedom.md` — `nu` and
  `n_d = nu / 2` are what the errors divide by. The handoff's "not started"
  status is behind `speckit/core.py` and `speckit/analysis.py`.
- `docs/handoffs/20261024_spectrum-units.md` — proposed unit strings. Not on
  `SpectrumResult`.
- `docs/handoffs/20260905_noise-generator-normalization.md` — generator `psd`
  is two-sided, so a generate-then-estimate round trip is a factor of 2 if
  the levels are compared naively.

## Known limitations

`nu` uses the window autocorrelation and the actual starts. The kernel
docstring says the covariance model is a Gaussian process whose spectrum is
flat across the window bandwidth, and that detrending is ignored. `M2` is
still the nonparametric scatter.

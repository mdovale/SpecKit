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

`GyyRx` and `GyySx` are the same array, the optimal residual
`Gyy - |Gxy|² / Gxx` (`(1 - coh) * Gyy` is that expression exactly). The
Cauchy–Schwarz inequality on the segment averages gives `coh <= 1`. The
floor at 0 only removes a `1 - coh` that is negative by roundoff, which
happens where `navg == 1` and `coh` is 1 in exact arithmetic.
[optimal-subtraction.md](optimal-subtraction.md) derives the residual in this
`Gxy` convention.

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
`test_gyysx_is_coherent_residual` requires `GyySx == GyyRx` and
`|GyySx - (Gyy - |Gxy|²/Gxx)| <= 1e-12 * Gyy`. It runs a 7-sample delay,
including bins with `|phase(Gxy)| > 60°`, and a zero-delay control.

`tests/integration/test_errors.py` holds these checks:

- The original 200-realization Monte Carlo at Hann, 50% overlap.
- `test_phase_error_is_bendat_closed_form` checks `Hxy_rad_error`,
  `Hxy_deg_error`, and `Hxy_mag_error` against the closed form at
  `rtol=1e-12`. It also checks that a `coh < 0.3` bin is more than 15% away
  from the former `arcsin` numerator.
- `test_zero_overlap_errors_are_independent_segment_limit` requires
  `nu == 2 * navg` and the `navg` form of `Gxx_dev`, `Gxx_error`, and
  `Gxx_emp_dev` for Hann and Kaiser at `olap=0`.
- `test_overlap_corrected_psd_error_matches_monte_carlo` uses 400
  realizations at Hann, 75% overlap, `L = 512`. Both `Gxx_dev` and
  `Gxx_emp_dev` must predict the measured `ddof=1` scatter within ±0.12,
  about 3.4 standard errors. The independent-segment prediction must be low
  by more than 1.25.

Monte Carlo evidence recorded on 24 Oct 2026 (400 realizations, standard
error about 3.5%) gave a measured-to-predicted `Gxx` scatter ratio of
0.97, 1.03, and 1.00 at Hann 0%, 50%, and 75% overlap, and 1.00 at the
default Kaiser (`psll=200`, overlap 0.766). With `n_d = navg` the 75% and
Kaiser ratios were 1.39 and 1.05. At 75% overlap and `coh = 0.5`, 1500
realizations measured the phase and `|H|` scatter at 0.97–1.03 of the
`n_d = nu / 2` Bendat prediction.

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
- `docs/handoffs/20261024_phase-error-arcsin.md` — `Hxy_rad_error` had an
  `arcsin` in its numerator. It is now Bendat's phase standard deviation,
  equal to `Hxy_mag_error`. Resolved.
- `docs/handoffs/20261024_overlap-independent-segments.md` — errors divided
  by the overlapped segment count. They now use `n_d = nu / 2`. Resolved.
- `docs/handoffs/20261024_overlap-degrees-of-freedom.md` — the same `nu`.
  The chi-square interval on `Gxx` is still open.
- `docs/handoffs/20261024_gyysx-residual.md` — the former `GyySx` expansion.
  Resolved.
- `docs/handoffs/20261024_csd-convention-vs-scipy.md` — `Gxy = X * conj(Y)`
  is the complex conjugate of `scipy.signal.csd`. `Hxy` matches
  `csd(x, y) / csd(x, x)`.
- `docs/handoffs/20261024_spectrum-units.md` — proposed unit strings. Not on
  `SpectrumResult`.
- `docs/handoffs/20260905_noise-generator-normalization.md` — generator `psd`
  is two-sided, so a generate-then-estimate round trip is a factor of 2 if
  the levels are compared naively.

## Known limitations

`nu` uses the window autocorrelation and the actual starts. Its covariance
model is a Gaussian process whose spectrum is flat across the window
bandwidth, away from DC and Nyquist, with the detrend ignored. Applying the
same `n_d = nu / 2` to the cross-spectral Bendat formulas assumes every
segment-pair covariance scales with `rho²`. That holds for jointly Gaussian
channels under the same assumptions, and the Monte Carlo above agrees.

`M2` is still the nonparametric per-segment scatter. Converting it to an
error of the mean needs the same correlation model, so `*_emp_*` divides by
`n_d`, not `navg`.

`Gxy_dev` uses Bendat's `|Gxy| / (|γ| sqrt(n_d))`. For jointly Gaussian
channels the scatter of `|Gxy|` is
`sqrt((Gxx Gyy + |Gxy|²) / (2 n_d))`, a factor
`sqrt((1 + coh) / 2)` below that (0.87 at `coh = 0.5`). The 50% overlap
Monte Carlo test allows `rel=0.5` for this quantity. It is unrelated to
overlap and was not changed.

Computing `nu` takes about 6% of `compute_spectrum` on a 2×10⁶-sample
series with the default plan.

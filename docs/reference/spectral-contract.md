# Spectral contract

This note states the estimand behind `compute_spectrum` as of 1.0.3, the
scaling and the cross-spectrum orientation, and the claims that the pointwise
interval is allowed to make. The other reference notes own the pieces.

## Goal

Give one place that says what a reported spectrum is, which samples went
into it, and which uncertainties are not part of that statement.

## Scope

In: the one-sided WOSA estimator in `speckit/analysis.py`, the bin
statistics in `speckit/core.py`, the logarithmic plans in
`speckit/schedulers.py`, the missing-sample policy, and
`SpectrumResult.gxx_interval`.

Out: a typed `SpectralPlan`, a spectral-response operator, replicate
storage, debiasing, multitaper, robust averaging, and the N-channel matrix.
Those are later work in `docs/handoffs/20261024_speckit-2.0-blueprint.md`.

## Estimand

The public estimator is a one-sided windowed overlapped segment average of a
uniformly sampled, complete, real series. The frequency grid is the
logarithmic scheduling heuristic in
[frequency-scheduling.md](frequency-scheduling.md). It is not the solution
of a stated bias-variance loss.

For bin `j`, segment length `L_j`, starts `D_j`, window `w`, and detrend
order, each segment is detrended, multiplied by `w`, and correlated with
`exp(-i ω n)` at `ω = 2 π f_j / fs`. The Goertzel kernels return that sum
times `exp(i ω (L-1))`. The factor is common to both channels of a segment,
so it cancels in `|X|²` and in `X * conj(Y)`. The NumPy kernels form the
direct sum. Both then average over segments.

The result is a description of that finite record under this operator. It is
not, by itself, an estimate of a local spectrum of a slowly varying process,
a transfer function under a named noise model, or a spectral matrix.

## Observation model

Samples are uniformly spaced and present. `missing="raise"` is the default:
a non-finite sample raises `ValueError`, and the message includes the count.
`missing="zero"` replaces non-finite samples with 0, which is the 1.0.3
behavior, and stores the count on `SpectrumAnalyzer.n_replaced`. That
replacement is a rectangular window of its own. There is no validity mask
and no interpolation.

A finite input that produces a non-finite kernel value keeps that value.
The reduction no longer passes outputs through `numpy.nan_to_num`.

## One-sided scaling and the cross-spectrum

`Gxx`, `Gyy`, and `Gxy` are one-sided densities on `[0, fs/2]`, in
signal²/Hz:

```
Gxx = 2 * XX / (fs * S2)
Gxy = 2 * XY / (fs * S2)
```

with `S2 = sum(w²)` and `XX`, `XY` the averaged segment products. For a real
signal with two-sided density `S2(f)`, `Gxx = 2 S2` on `(0, fs/2)`. Noise
generators in `speckit.noise` take and produce two-sided levels, so a
generate-then-estimate round trip compares `Gxx` with `2 * psd`. DC and
Nyquist are not given a separate one-sided factor of 1.

`Gxy = X * conj(Y)`, channel 0 as `X` and channel 1 as `Y`. That is the
complex conjugate of `scipy.signal.csd(x, y)`. `Hxy = conj(XY) / XX` matches
`csd(x, y) / csd(x, x)`.

## Detrend

`order` selects the projection removed from each segment before the window:

- `-1` — window only
- `0` — subtract the segment mean
- `1` — subtract the projection onto `{1, t}`
- `2` — subtract the projection onto `{1, t, t²}`

For orders 1 and 2, `t` runs from `-1` to `1` and the basis is a reduced QR
of that Vandermonde (`speckit.core._build_Q`). The same projection is what
`y - Q Q.T y` removes. The window is applied after the projection.

## Scheduler

`SpectrumAnalyzer` defaults to `vectorized_ltf_plan`. The same compromise is
implemented by `ltf_plan`, which walks one frequency at a time. The
vectorized plan evaluates it on a logarithmic grid of `10 * Jdes` points and
then steps by the resolution it looks up. Lengths on that path use
`numpy.round`. `ltf_plan` uses a half-up rounder. The two plans both satisfy
`r L = fs` and `f[j+1] - f[j] = r[j]` to roundoff. They do not return the
same `L` on every bin. The measured differences are in
[frequency-scheduling.md](frequency-scheduling.md).

## Degrees of freedom and the pointwise interval

`nu` is Percival and Walden (1993), eq. 292b, from the window autocorrelation
and the actual starts. With no overlap, `nu = 2 * navg`. Analytic and
empirical errors divide by `n_d = nu / 2`. The formula assumes a Gaussian
process whose spectrum is flat across the window, ignores the detrend, and
drops the pseudo-covariance that matters at DC and Nyquist.

`SpectrumResult.gxx_interval(conf)` inverts that model. With
`alpha = 1 - conf`,

```
G_low  = nu * Gxx / chi2.ppf(1 - alpha/2, nu)
G_high = nu * Gxx / chi2.ppf(alpha/2, nu)
```

`nu` is unrounded. The interval is pointwise. It does not say that a
fraction `conf` of the bins on one spectrum fall inside it together, because
neighboring bins share samples. `Gxx_dev = Gxx * sqrt(2 / nu)` is the
plug-in standard deviation under the same model, not this interval.

`M2` is the mean squared distance of the per-segment cross-spectrum from its
mean. `*_emp_*` divides that scatter by `nu / 2`. It is not an
assumption-free uncertainty of the mean.

The coverage claim for `gxx_interval` is the Monte Carlo in
`tests/integration/test_estimator_contract.py`: Hann window, `order=-1`, no
overlap, one interior bin, white Gaussian noise. Detrended bins, colored
spectra, DC, and Nyquist are outside that claim.

## What this contract does not claim

- A bias for a sloped continuum, or a leakage level from a line.
- A covariance between frequency bins.
- A stationarity test.
- A coherence interval. `coh_dev` remains the Bendat formula.
- That `vectorized_ltf_plan` optimizes a declared risk.

## Numerical checks

- `tests/integration/test_estimator_contract.py` locks the 1.0.3 `Gxx`,
  `Gxy`, `nu`, and plan arrays for a seeded series at every detrend order,
  and checks the missing-sample policy and the interval coverage.
- `tests/unit/test_direct_dft.py` compares both kernel families with
  `tests/direct_dft_oracle.py`. The absolute tolerance is
  `8 * L * eps * sum(|y_windowed|)` on the Fourier coefficient.
- `tests/unit/test_schedulers.py` includes `vectorized_ltf_plan` and the
  counted differences from `ltf_plan`.

## Key files

- `speckit/analysis.py` — `missing`, `n_replaced`, `gxx_interval`
- `speckit/core.py` — kernels and `_overlap_dof`
- `speckit/schedulers.py` — `vectorized_ltf_plan`, `ltf_plan`
- `tests/direct_dft_oracle.py`
- [spectrum-estimation.md](spectrum-estimation.md),
  [bin-statistics.md](bin-statistics.md),
  [frequency-scheduling.md](frequency-scheduling.md)

## Known limitations

The interval and `nu` share one locally white Gaussian model. A spectrum that
varies across the main lobe, a polynomial detrend, and the bins next to DC
and Nyquist are not covered by the Monte Carlo above. Cross-frequency
dependence is not reported.

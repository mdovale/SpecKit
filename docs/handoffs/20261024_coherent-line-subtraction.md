# Handoff: coherent line fit and subtraction

## Scope

Item 5 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Fit tonal lines and subtract
them in the time domain. This note does not cover `new_ltf_plan`.

## Goal

From frequencies returned by `peak_finder`, or from user-supplied tones, fit
amplitude, phase, and a polynomial whose `order` means the same thing as the
spectral detrend (`-1` none, `0` mean, `1` linear, `2` quadratic). Subtract
in the time domain and return the residual. An optional second step
re-estimates with `compute_spectrum` and checks that the line is gone.

## Current status

Not started. `speckit/dsp.py` `peak_finder` flags peaks by a carrier-to-noise
ratio in dB. It does not subtract. `optimal_linear_combination` subtracts
scaled copies of other channels. It is not a sine fit.

## Context

**LTPDA.** `ao/linedetect.m` finds lines in an `fsdata` spectrum.
`ao/fit_sine_poly.m` fits `x(t)` to a polynomial plus sines.
`ao/spsdSubtraction.m` is an sPSD-weighted iterative least-squares
subtraction. That third routine is a different algorithm from a single sine
fit.

**Owner.** `speckit/dsp.py`, next to `peak_finder`.

**PSD convention.** The check that a line is gone uses a one-sided
`compute_spectrum` PSD. The fit itself is in the time domain. Do not multiply
the PSD by a notch gain and call that the subtraction.

**Dependencies.** None of the other nine items block the sine-plus-polynomial
fit. Comparing the residual continuum to `Gxx_dev` is enough. The
overlap-corrected interval
(`docs/handoffs/20261024_overlap-degrees-of-freedom.md`) is the better bar
once it exists.

## Settled decisions

- Implement the sine-plus-polynomial fit first.
- Port `spsdSubtraction.m` only if the single fit leaves a biased continuum
  on a steep spectrum, and say which routine was ported.
- Two close tones that the fitter cannot separate must fail the test
  honestly. Do not merge tones to force a pass.
- Polynomial order uses the same integers as `SpectrumAnalyzer`'s `order`.

## Success criteria

A unit-amplitude sine at a non-integer bin frequency, plus white noise, seed
fixed. Record seed, `N`, `fs`, tone frequency, amplitude, and noise
standard deviation.

- After subtraction, the residual power at that tone, read from
  `compute_spectrum` in a band of a few resolution bins, drops by the factor
  the fit's parameter covariance predicts.
- The continuum away from the tone stays inside `Gxx_dev` of a noise-only
  spectrum of the same noise (same seed stream for the noise, or a paired
  comparison).
- A two-tone case documents the minimum separation the fitter claims, and
  fails if the tones are closer than that.
- `ruff check` and `ruff format --check` on the touched files. Existing
  `peak_finder` tests stay green.

## Recommended next steps

1. Read `fit_sine_poly.m` far enough to write the design matrix (polynomial
   columns plus `cos` and `sin` at each tone) in the docstring.
2. Implement the fit and the time-domain residual.
3. Add the spectral residual test.
4. Stop before porting `spsdSubtraction.m` unless the continuum test fails
   on a steep spectrum.

This item is step 6 of the cross-item order.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- `speckit/dsp.py` — `peak_finder`
- `ltpda/classes/@ao/linedetect.m`, `fit_sine_poly.m`, `spsdSubtraction.m`

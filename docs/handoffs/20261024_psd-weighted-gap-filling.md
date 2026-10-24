# Handoff: PSD-weighted gap filling

## Scope

Item 1 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Fill missing samples in a
time series so that `compute_spectrum` can run on a contiguous record. This
note does not cover the other nine items, and it does not cover `new_ltf_plan`.

## Goal

Two public functions in `speckit/dsp.py`:

- A linear or cubic-spline fill across a gap mask (or NaNs).
- An optimized fill that minimizes a chi-square weighted by a one-sided PSD.

The spectral estimator stays gap-free. The fill returns a contiguous series.

## Current status

Not started. SpecKit has no gap fill. `gapfillingoptim.m` has not been
transcribed into an objective yet. No gold values exist.

## Context

**LTPDA.** `ltpda/classes/@ao/gapfillingoptim.m` minimizes a chi-square
weighted by the expected PSD. Short gaps use `ao/optSubtraction`. Long gaps
use `optSubstitution`. The header also returns the PSD used as the weight
and the windows used to estimate it. `ltpda/classes/@ao/gapfilling.m` is the
simpler path: linear or spline interpolation between the samples on either
side of the gap.

**Owner.** `speckit/dsp.py`. Do not put the fill inside `compute_spectrum`.

**PSD convention.** The weight is a one-sided PSD, the same convention as
`compute_spectrum` / `SpectrumResult.psd` (power per Hz, positive frequencies
only). Do not pass a noise-generator `psd` in as that weight. Generator
`psd` arguments are two-sided. See
`docs/handoffs/20260905_noise-generator-normalization.md`.

**Dependencies.** The optimized-fill test needs a colored series with a
known one-sided level. Until item 2
(`docs/handoffs/20261024_color-whiten-spectrum.md`) exists, build that series
with `white_noise` using the **code** normalization `rms = sqrt(psd * fs)`
and state in the test that `psd` is two-sided. The linear/spline fill does
not wait on item 2. "Inside the error bar" should use the overlap-corrected
interval from `docs/handoffs/20261024_overlap-degrees-of-freedom.md` once
that exists. Until then, compare against today's `Gxx_dev` and say so.

## Settled decisions

- Linear/spline and the optimized fill are separate named functions.
- The estimator is not modified to skip NaNs.
- Do not invent a spectral extrapolation and call it the LTPDA method. Read
  `gapfillingoptim.m` and write the objective in the docstring before coding.
- A plot helper must not multiply by 2, `sqrt(2)`, or an ENBW so a filled
  spectrum lines up with an ungapped one.

## Success criteria

Build a long colored series, punch a gap of a few percent of the length, and
fill it. Record the seed, `N`, `fs`, gap location, and the PSD used as the
weight in the test docstring. Compute the expected level from the generator
convention above. Do not invent the number.

- Linear/spline fill: filled samples match `numpy.interp` or a cubic spline
  on the gap edges, to interpolation tolerance.
- Optimized fill: the one-sided PSD of the filled series, from
  `compute_spectrum`, stays within the analytic deviation of the ungapped
  series on the band the weight was defined on.
- Negative control: a linear fill of a steep spectrum must **not** be
  asserted to pass that spectral test.
- If the optimized fill fails the spectral test, fix the weight or the
  objective. Do not widen the tolerance until the residual is consistent
  with the number of averages.
- `python -m pytest` on the new test, plus `ruff check` and
  `ruff format --check` on the touched Python.

## Recommended next steps

1. Read `gapfilling.m` and implement the linear/spline function. It has no
   PSD dependence and can land first.
2. Read `gapfillingoptim.m`. Write the chi-square objective, including which
   PSD (one-sided) multiplies which residual, into the docstring.
3. Implement the optimized fill and the spectral test.
4. Stop and extend this note if the MATLAB routine's short-gap and long-gap
   branches are different objectives. Do not collapse them into one formula
   without saying so.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md` — ranking and cross-item order.
  This item is step 8 of the recommended order. The linear/spline fill may
  land earlier.
- `docs/handoffs/20260905_noise-generator-normalization.md`
- `docs/handoffs/20261024_color-whiten-spectrum.md`
- `docs/handoffs/20261024_overlap-degrees-of-freedom.md`
- `ltpda/classes/@ao/gapfillingoptim.m`, `ltpda/classes/@ao/gapfilling.m`

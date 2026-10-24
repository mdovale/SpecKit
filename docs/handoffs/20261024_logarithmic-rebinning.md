# Handoff: logarithmic rebinning of a uniform spectrum

## Scope

Item 9 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Rebin a uniform-grid spectrum
onto logarithmic, linear, or caller-supplied edges. This note does not cover
`new_ltf_plan`, and it does not change the LPSD/LTF schedulers.

## Goal

One function, in `speckit/dsp.py` or used by `SpectrumResult`, that rebins a
uniform-grid one-sided PSD (SciPy `welch` output, or a linear FFT) onto log
edges. The default reduction is the **mean** of the samples that fall in
each bin, matching LTPDA `bin_data`. A variance-weighted mean, if added, is
a separate `how=` value and is not the default.

## Current status

Not started. `compute_spectrum` already returns a log-frequency plan.
Nothing rebins a linear spectrum onto that kind of grid. `get_measurement`
interpolates. Interpolation is not a bin mean.

## Context

**LTPDA.** `ltpda/classes/@ao/bin_data.m`. Rebins onto logarithmic, linear, or
user edges by the mean of the samples that fall in each bin. Inherited from
D. Nicolodi (UniTN). It does not weight by bin width or by degrees of
freedom.

**Owner.** One implementation. A `SpectrumResult` method may call it. Do not
copy the loop into `analysis.py` and `dsp.py`.

**PSD convention.** Preserve the input convention. Rebinning does not
multiply by 2 and does not convert a two-sided density to a one-sided one.
The docstring says that the output carries the caller's convention.

**Not for LPSD output.** SpecKit's own spectrum is already on a log grid
whose spacing is the resolution bandwidth. Do not run this function inside
`compute_spectrum`. Applying it to an LPSD result as a user call is allowed
and must still be a plain mean. It is not a second estimator.

## Settled decisions

- Default `how="mean"`.
- Empty edges are either dropped or returned as NaN. Pick one, document it,
  and test it.
- No ENBW weight in the default.

## Success criteria

- A constant spectrum rebins to that constant, for log edges and for linear
  edges.
- A pure power law sampled on a fine linear grid, rebinned by the mean,
  matches the mean of the samples in each edge computed independently with
  NumPy in the test. The tolerance is float64, not a spectral error bar.
- An empty edge follows the documented policy.
- A two-sided input is not doubled. A test can pass a constant `1.0` and
  require a constant `1.0` back.
- `ruff check` and `ruff format --check` on the touched file and the test.

The test records the input grid, the edges, and the expected bin means. Do
not invent those means. Compute them in the test from the samples.

## Recommended next steps

1. Read `bin_data.m` far enough to match edge inclusion (which sample belongs
   to which bin when it lands on an edge).
2. Implement the mean reduction.
3. Add the constant, power-law, and empty-edge tests.

This item is independent. The cross-item order places it at step 9 and says
to pull it forward when a Welch comparison needs it.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- `ltpda/classes/@ao/bin_data.m`
- SciPy `signal.welch` — a typical uniform-grid input. Its default scaling
  is a one-sided density. This function still must not apply an extra factor
  of 2.

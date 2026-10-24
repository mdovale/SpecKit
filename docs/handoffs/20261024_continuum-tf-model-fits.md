# Handoff: continuum and transfer-function model fits

## Scope

Item 8 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. A log-polynomial fit of a
one-sided spectrum, and, later, a z-domain fit of a complex transfer
function. The plant fit does not start in the same change as the continuum
fit. This note does not cover `new_ltf_plan`.

## Goal

Continuum, first:

```
log10(S) ≈ sum_{k=0}^{n} a_k (log10(f / f_ref))^k
```

with `f_ref` documented. LTPDA fits `log10(f)` directly, so `f_ref = 1` Hz.
Return `a_k` and the one-sided model `S_hat(f)` on the input grid or on a
caller-supplied grid.

Plant, later: fit a complex `H(f)` from a `SpectrumResult` transfer function
as a z-domain partial-fraction expansion. This waits on the continuum fit and
on the residue helper from
`docs/handoffs/20261024_color-whiten-spectrum.md`.

## Current status

Not started. No log-polynomial spectrum fit and no pole-residue fit exist in
`speckit/`. `SpectrumResult.Hxy` is `conj(Gxy) / Gxx`, which is the estimator
that matches a physical delay (`Y` from `X` under SciPy's
`Pxy = X * conj(Y)`). A plant fit must use that `Hxy`, not `Gxy / Gxx`.

## Context

**LTPDA.** `ao/polyfitSpectrum.m` takes `log10` of **both** axes
(`plist('axis','xy')`), optionally adds `regularizecoeff`, and calls
`polynomfit` with user `orders`. The model is evaluated at `log10(fout)`.
`ao/zDomainFit.m` fits `sum z*r_k/(z-p_k)` via `utils.math.vdfit`, choosing
order by a stop test. `@pzmodel` stores poles, zeros, gain, and delay, and
can convert to IIR/FIR.

**Owner.** Log-polynomial continuum: a function that takes `f` and one-sided
`S`. Put it in `speckit/dsp.py` unless it needs `SpectrumResult` internals,
in which case a method on `SpectrumResult` may call the same function.
Z-domain plant fit: `speckit/systems.py`. Do not create `speckit/models.py`
unless both fits share a residue representation. The only shared residue
code is item 2's partial-fraction fitter. Call it. Do not fit poles in three
modules.

**PSD convention.** Fit the one-sided spectrum the caller passes.
`orders` is the polynomial degree in `log10(f/f_ref)` and `log10(S)`. Say
that in the docstring. Do not fit the ASD unless the function is named for
the ASD and converts `2 * log10` explicitly.

**Known trap.** A fitter that is accurate only for slope `-2` is the
`alpha_noise` failure mode
(`docs/handoffs/20260905_noise-generator-normalization.md`). Test at least
slopes `0` and `-1`.

## Settled decisions

- `f_ref = 1` Hz unless the docstring and the test say otherwise.
- Continuum fit lands alone. `zDomainFit.m` does not start in that change.
- If the residue helper from item 2 is not stable, leave the plant fit
  unstarted and say so in **Current status**.
- Non-positive frequency or PSD bins are excluded. The test band excludes
  them. Do not clip `log10(0)` to a large negative number and keep the bin.

## Success criteria

Synthetic one-sided `S(f) = A * (f / f_ref)^b` over a band with `f > 0` and
`S > 0`.

- Noise-free: recovered slope `b` and amplitude `A` match to a float64
  tolerance. Test `b = 0` and `b = -1` at minimum. Include one non-integer
  slope.
- On a `compute_spectrum` estimate of a series with that spectrum: `b` and
  `A` fall inside the PSD error bars propagated to the fit. Use `Gxx_dev`
  until `docs/handoffs/20261024_overlap-degrees-of-freedom.md` lands, then
  use `nu`. A series with spectrum `S` has to be colored by
  `docs/handoffs/20261024_color-whiten-spectrum.md` or by an explicit
  two-sided generator level. State which, and do not hide a factor 2 in the
  amplitude tolerance.
- `ruff check` and `ruff format --check` on the touched files.

## Recommended next steps

1. Implement the log-polynomial fit and the noise-free tests.
2. Add the estimated-spectrum test only after item 2 can build the series,
   or build it with `white_noise` for `b = 0` first.
3. Do not open `zDomainFit.m` in that change.

This item's continuum half is step 3 of the cross-item order.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- `docs/handoffs/20261024_color-whiten-spectrum.md`
- `docs/handoffs/20260905_noise-generator-normalization.md`
- `ltpda/classes/@ao/polyfitSpectrum.m`, `zDomainFit.m`
- `speckit/analysis.py` — `Hxy = conj(XY) / XX`

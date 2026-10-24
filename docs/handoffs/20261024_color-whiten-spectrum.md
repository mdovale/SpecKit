# Handoff: color and whiten an arbitrary one-sided spectrum

## Scope

Item 2 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Synthesize and whiten a
series from a one-sided PSD on an arbitrary frequency grid. Two-channel
whitening is deferred until the N-channel matrix exists. This note does not
cover `new_ltf_plan`.

## Goal

Public functions whose model argument is a one-sided PSD `(f, S1(f))` in Hz
and power/Hz. The grid need not be uniform and need not include DC or
Nyquist.

- Coloring a unit-variance white series, in the **code** convention of
  `white_noise`, yields a series whose `compute_spectrum` PSD matches `S1`.
- Whitening a series that has spectrum `S1` yields a series whose one-sided
  PSD is flat. Pick the flat level (1, or 2 if the residual is documented as
  two-sided unit white), test it, and write it in the docstring.

## Current status

Not started. `fftnoise` colors a full Hermitian FFT vector, not a log-grid
one-sided ASD. No partial-fraction fitter exists in `speckit/noise.py`.

Unblocked 2026-10-24: the noise-generator handoff is fixed. `alpha_noise`
now meets its documented two-sided level, `f**-alpha`, to within 1% inside
its band. The `white_noise` docstring matches `rms = sqrt(psd * fs)`. The
`noise` module and `compute_spectrum` both document `Gxx = 2 * S2`. The
"Blocked by" paragraph below is kept as the record of what was wrong.

## Context

**LTPDA.** `ltpda/classes/@ao/noisegen1D.m` fits z-domain partial fractions
to a one-sided PSD model (`utils.math.psd2tf`), builds a parallel bank of
MIIR filters, and colors a white series. `ltpda/classes/@ao/whiten1D.m` and
`buildWhitener1D.m` call `ltpda/classes/+utils/@math/psd2wf.m`, which fits
the square root of the one-sided PSD, stabilizes with an all-pass, and
inverts. `whiten2D.m` whitens a 2×2 cross-spectral matrix from its
eigendecomposition. The `psd2wf.m` header states that the input model is the
one-sided PSD.

**Owner.** The fit and the filter coefficients live in `speckit/noise.py`.
Applying the filter to an existing series may live in `speckit/dsp.py` and
must call those same coefficients. Do not fit in both places.

**PSD convention.** The public model is one-sided, because that is what
`SpectrumResult.psd` is. `white_noise` and `alpha_noise` take a two-sided
`psd`. Convert between those conventions in one place inside `noise.py`.

**Blocked by** (resolved 2026-10-24, see Current status).
`docs/handoffs/20260905_noise-generator-normalization.md`.

- `white_noise` sets `rms = sqrt(psd * fs)`. The docstring says
  `sqrt(psd * fs / 2)`. The code is the two-sided formula and is the one to
  match.
- `alpha_noise` misses its documented normalization by up to about 13% in
  PSD, depending on alpha.
- A naive generate / `compute_spectrum` round trip is off by 2 until that
  handoff is fixed, because the generator argument is two-sided and the
  estimator is one-sided.

**Dependencies.** Two-channel whitening waits on
`docs/handoffs/20261024_nchannel-cross-spectral-matrix.md`. The continuum fit
in `docs/handoffs/20261024_continuum-tf-model-fits.md` should call this
item's residue helper rather than fitting poles again. Error bars for the
recovered PSD should use
`docs/handoffs/20261024_overlap-degrees-of-freedom.md` once it exists. Until
then, use `Gxx_dev` and say so.

## Settled decisions

- The public argument is one-sided `(f, S1)`. The two-sided generator
  convention is an internal conversion, tested by a round trip.
- Do not ship `whiten2D` as two independent `whiten1D` calls.
- If the partial-fraction order is free, stop when the fit residual is
  inside the PSD error bars and record the order rule. Do not hard-code an
  order that only works for one alpha.
- Widening `pytest.approx` to hide a factor 2 is a failed fix.
- The new fitter must not inherit the `alpha_noise` level bias. If it cannot
  beat that bias on a `1/f²` model, stop and extend the noise handoff instead
  of shipping both.

## Success criteria

Record seed, `N`, `fs`, the model grid, and the expected one-sided level in
the test docstring.

- Analytic case: `S1(f)` constant. The colored series matches `white_noise`
  at the equivalent two-sided level, to sampling error, and
  `compute_spectrum` recovers `S1` inside `Gxx_dev` (or the overlap-corrected
  interval, once it exists).
- Steep case: a one-sided `1/f²` model on a log grid. Recovered slope and
  level match the model. This is the case `alpha_noise` gets wrong by up to
  about 13%.
- Round trip: color then whiten returns a white residual at the documented
  level. A factor 2 is a convention bug.
- `python -m pytest tests/unit/test_noise.py -q`, plus the new tests, and
  `ruff check` / `ruff format --check` on the touched files.
- Existing `compute_spectrum` integration values for 1-D inputs stay
  unchanged.

## Recommended next steps

1. Read the noise-generator handoff and `psd2wf.m` / `noisegen1D.m`.
2. Write the one-sided to two-sided conversion in one function and test it
   against `white_noise`'s code (`rms = sqrt(psd * fs)` for a flat spectrum).
3. Land one-channel color and whiten.
4. Leave `whiten2D` until the N-channel matrix can form the 2×2 matrix.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md` — this item is step 2 of the
  recommended order.
- `docs/handoffs/20260905_noise-generator-normalization.md`
- `docs/handoffs/20261024_nchannel-cross-spectral-matrix.md`
- `docs/handoffs/20261024_continuum-tf-model-fits.md`
- `ltpda/classes/@ao/noisegen1D.m`, `whiten1D.m`, `whiten2D.m`,
  `buildWhitener1D.m`
- `ltpda/classes/+utils/@math/psd2wf.m`

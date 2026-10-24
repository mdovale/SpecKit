# Handoff: units that follow the spectrum

## Scope

Item 10 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Carry unit strings on a
spectral result. No unit algebra and no new dependency. This note does not
cover `new_ltf_plan`.

## Goal

`SpectrumResult` carries the input unit string and the derived strings for
PSD, ASD, CSD, and coherence. Coherence is dimensionless. `get_rms` documents
the unit of the integral. Numeric values do not change.

A series passed with `y_unit="m"` yields PSD unit `m²/Hz` and ASD unit
`m/√Hz`. Omitting the unit leaves the strings empty and the numbers
identical to today.

## Current status

Not started. `SpectrumResult` has no unit fields. Quantities are bare
NumPy arrays. `get_rms` integrates the one-sided ASD over the frequencies
the plan actually contains. It does not attach a unit.

## Context

**LTPDA.** `@unit` propagates through arithmetic, and analysis objects store
`xunits` and `yunits`. `projectNoise` sets output units to
`output_units * Hz^{-1/2}` for an ASD. SpecKit will not do that arithmetic
in a unit library.

**Owner.** `SpectrumResult` in `speckit/analysis.py`. Generators in
`speckit/noise.py` gain matching strings when
`docs/handoffs/20261024_color-whiten-spectrum.md` lands. Do not build a unit
algebra. Do not add `pint` or `astropy.units` unless the user asks.

**What the strings must agree with.**

| Quantity | String for input unit `U` | Numeric convention |
|----------|---------------------------|--------------------|
| PSD, `Gxx`, `Gyy` | `U²/Hz` | one-sided |
| ASD | `U/√Hz` | `sqrt` of that PSD |
| CSD | `U²/Hz` | one-sided, `Pxy = X * conj(Y)` |
| Coherence | dimensionless (`1` or an empty unit) | `γ²` |
| `get_rms` | `U` | integral of the one-sided ASD squared |

A string that says `U²/Hz` on a two-sided number is a bug. This item does
not fix the generator factor-of-2 problem. That stays in
`docs/handoffs/20260905_noise-generator-normalization.md`.

**RMS coverage.** `get_rms` integrates only the frequencies present on the
plan. A log plan that starts at `bmin * fs / N` and stops short of Nyquist
does not recover the full variance of a white series. The unit string does
not change that. Do not scale the integral so the RMS matches `std(x)`.

## Settled decisions

- Explicit strings, not a unit type.
- Missing unit → empty strings, unchanged numbers.
- Ask before adding `pint` or `astropy`.

## Success criteria

- `y_unit="m"` produces PSD `m²/Hz` and ASD `m/√Hz` as strings on the result.
- Omitting `y_unit` leaves those strings empty.
- Array comparisons in the existing integration tests stay green. The change
  is metadata.
- A test that the PSD of unit-variance white noise is still about `2/fs`
  (one-sided) when a unit string is set. The string must not have scaled the
  array.
- `ruff check` and `ruff format --check` on the touched files.

## Recommended next steps

1. Add the strings on `SpectrumResult` and on `compute_spectrum`'s input
   path.
2. Document `get_rms` units in the docstring, including that the band is the
   plan's band.
3. Leave generator unit strings until item 2's one-sided model exists, so
   the generator string is not attached to the two-sided `psd` argument by
   mistake.

This item is step 10 of the cross-item order. Land it after the numeric
conventions it describes are stable.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- `docs/handoffs/20260905_noise-generator-normalization.md`
- `docs/handoffs/20261024_color-whiten-spectrum.md`
- `docs/handoffs/20261024_component-noise-projection.md` — ASD output string
  `U/√Hz` when that function grows a unit argument
- `ltpda/classes/@ssm/projectNoise.m` — output unit `Hz^{-1/2}` for an ASD
- `speckit/analysis.py` — `SpectrumResult.get_rms`, `Gxx` scale
  `2 / (fs * S2)`

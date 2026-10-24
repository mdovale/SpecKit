# Handoff: phase-error formula has a misplaced arcsin

## Scope

`SpectrumResult.Hxy_rad_error` and `Hxy_deg_error` insert `arcsin` in the
numerator of the Bendat phase standard deviation. The magnitude errors do
not. This note is that formula. It does not change `Gxx_dev` or the overlap
treatment of `navg` (that is a separate handoff).

## Problem

Bendat's standard deviation of the transfer-function phase, in radians, is

```
σ[φ] = sqrt(1 - γ²) / sqrt(2 K γ²)
```

where `γ²` is the magnitude-squared coherence (`SpectrumResult.coh`) and `K`
is `navg`. The manuscript table states this expression for `Hxy_deg_error`
(see the unit mismatch below).

The code (`speckit/analysis.py`, `Hxy_rad_error`) computes

```
arcsin(sqrt(|1 - coh|)) / sqrt(coh * 2 * navg)
```

and `Hxy_deg_error = rad2deg` of that. The ratio to Bendat is

```
arcsin(sqrt(1 - γ²)) / sqrt(1 - γ²)
```

which is about 1 only when `γ²` is near 1.

| `coh` | `arcsin` form / Bendat |
|-------|------------------------|
| 0.20  | 1.238                  |
| 0.50  | 1.111                  |
| 0.90  | 1.017                  |
| 0.99  | 1.002                  |

`Hxy_mag_error` and `Hxy_dev` already use `sqrt(|1 - coh|) / sqrt(coh * 2 * navg)`
without `arcsin`. Those match Bendat's magnitude formulas. Only the phase
pair is wrong.

`Hxy_deg_error` converting radians to degrees is the right unit for a
property named `_deg_`. The manuscript table writes the radian formula in a
column labeled degrees and does not show `180/π`. After the code fix, the
degree values are `rad2deg` of the Bendat radian expression. Updating the
table is a manuscript edit, not part of the numerical fix, unless the user
asks.

## Current status

**Resolved 24 Oct 2026.** The `arcsin` is gone. `Hxy_rad_error` is now
`sqrt(|1 - coh|) / sqrt(coh * 2 * n_d)`, the same value as `Hxy_mag_error`,
and `Hxy_deg_error = rad2deg` of it. `n_d` is `nu / 2` from the overlap
fix in `docs/handoffs/20261024_overlap-independent-segments.md`, which
landed in the same change. Without overlap it equals `navg`. The `coh == 0`
singularity is unchanged and has no NaN guard.

`arcsin(r)` is the phase bound for a confidence circle of relative radius
`r` around `H`. Even that form wraps the whole ratio. Wrapping only the
numerator matches no reference.

Evidence beyond the algebra: a 400-realization Monte Carlo of `angle(Hxy)`
(Hann, no overlap, `K = 78`, 20 Hz, 3-sample delay, seed 7), measured
against each formula:

| true `coh` | measured / Bendat | measured / old `arcsin` |
|------------|-------------------|-------------------------|
| 0.333      | 1.03              | 0.88                    |
| 0.500      | 1.02              | 0.92                    |
| 0.750      | 1.02              | 0.97                    |
| 0.900      | 1.02              | 1.00                    |

The standard error is about 3.5%. These numbers are evidence, not test
tolerances.

Test: `tests/integration/test_errors.py::test_phase_error_is_bendat_closed_form`
builds the delay spectrum from this note and checks `Hxy_rad_error`,
`Hxy_deg_error`, and `Hxy_mag_error` against the closed form at
`rtol=1e-12`, at no overlap and at 50% overlap. It also checks that the
former `arcsin` value is more than 15% larger on the `coh < 0.3` bins.

The manuscript table (`docs/paper/main.tex`, row `Hxy_deg_error`) still
shows the radian formula in a degree column. It was not edited, as
recommended below.

The original status: reproduced from source and closed form, and
`test_errors.py` did not read `Hxy_rad_error`.

On the delay series in `docs/handoffs/20261024_gyysx-residual.md` (`N = 80000`,
`fs = 200`, seed 1, 7-sample delay), bins grouped by coherence had median
`Hxy_rad_error / Bendat` of 1.25 (`coh < 0.3`), 1.12 (`0.3–0.6`), 1.03
(`0.6–0.9`), and 1.01 (`coh > 0.9`). A handful of bins went to `inf` where
`coh` hit 0 in the denominator. That divide-by-zero is the usual `γ → 0`
singularity of the Bendat formula, not the `arcsin` bug. Do not hide it by
clipping `coh` up to a large floor. A guard that returns NaN at `coh == 0`
is optional and must be tested.

## Repro

```python
import numpy as np

def bendat(coh, navg):
    return np.sqrt(1 - coh) / np.sqrt(coh * 2 * navg)

def code(coh, navg):
    return np.arcsin(np.sqrt(np.abs(1 - coh))) / np.sqrt(coh * 2 * navg)

for c in (0.2, 0.5, 0.9, 0.99):
    print(c, code(c, 100) / bendat(c, 100))
```

Expected after the fix: the ratio is 1 at each of those coherences, for any
`navg > 0`.

No random series is required to prove the algebra. A `SpectrumResult` test
should still build a two-channel spectrum so the property, not a retyped
formula, is what gets checked.

## What was tried

Compared the source with Bendat and with the manuscript table. Did not edit
the formula. Did not run a Monte Carlo of phase estimates. The 40-realization
error test in `test_errors.py` is aimed at magnitude deviations and uses a
20% relative tolerance. It is not evidence that the `arcsin` form is right,
and it is not a reason to keep `arcsin` because a loose phase Monte Carlo
might pass.

## Context

`coh` in this codebase is `γ² = |XY|² / (XX YY)`, not `|γ|`. The Bendat
denominator `sqrt(2 K γ²)` is `sqrt(coh * 2 * navg)`. Using `sqrt(coh)` in
place of `coh` would be a second bug. The current denominator already has
`coh` in the right place.

`navg` here is the segment count, not an overlap-corrected degree of freedom.
Fixing `arcsin` does not address
`docs/handoffs/20261024_overlap-independent-segments.md`. Land the `arcsin`
removal first so that a later `nu` change has one formula to scale.

One-sided versus two-sided does not enter. Phase is dimensionless.

## Success criteria

- For a two-channel `SpectrumResult` with several bins of `coh` in
  `(0.2, 0.99)` and `navg > 1`,
  `Hxy_rad_error` equals `sqrt(1 - coh) / sqrt(coh * 2 * navg)` to float
  tolerance on every such bin.
- `Hxy_deg_error` equals `rad2deg` of that, to the same tolerance.
- `Hxy_mag_error` is unchanged (it already matches its Bendat form).
- A bin with `coh = 0.2` must show the ratio to the old `arcsin` form near
  `1.24`, so the test cannot pass against the old formula.
- `python -m pytest tests/integration/test_errors.py -q` still passes.
- `ruff check` and `ruff format --check` on the touched files.

## Recommended next steps

1. Replace the `Hxy_rad_error` numerator with `sqrt(abs(1 - coh))`. Leave the
   denominator as `sqrt(coh * 2 * navg)`.
2. Keep `Hxy_deg_error = rad2deg(Hxy_rad_error)`.
3. Add a direct comparison to the closed form at low and high coherence.
   Do not fit a new tolerance to a phase Monte Carlo.
4. Leave the manuscript table until the user asks. If it is edited, the
   degree row needs the `180/π` factor, or the cell should say the formula
   is in radians.

## References

- `speckit/analysis.py` — `Hxy_rad_error`, `Hxy_deg_error`, `Hxy_mag_error`
- `docs/paper/main.tex` — table of spectral quantities, row `Hxy_deg_error`:
  `σ[φ] ≈ (1 - γ²)^{1/2} / (|γ| √(2K))`, unit column "deg"
- Bendat and Piersol, *Random Data* — phase random error equals the normalized
  magnitude error of `|H|`, in radians
- `tests/integration/test_errors.py` — does not cover this property
- `docs/handoffs/20261024_overlap-independent-segments.md` — separate; `K` in
  this formula is still the raw segment count until that note is done

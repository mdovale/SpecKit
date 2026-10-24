# Handoff: overlapped segments counted as independent

## Scope

`Gxx_dev` and the other Bendat errors divide by the number of overlapped
segments, `navg`, as if those segments were independent. At the overlaps
SpecKit actually uses, that understates the scatter. This note is that
discrepancy. Choosing the effective sample size is a convention choice, so
the fix does not belong in a one-line scale factor at a call site.

A longer note,
`docs/handoffs/20261024_overlap-degrees-of-freedom.md`, was written for a
different request (an LTPDA confidence-interval feature). It already compares
`getdof.m` and `psdconf.m`. Use it for the LTPDA reading. The success
criteria for **this** bug are below. Do not treat a full confidence-interval
API as required to close this bug.

## Problem

For a Gaussian periodogram average of `K` independent segments,

```
σ[Gxx] = Gxx / sqrt(K)
```

and `ν = 2 K` degrees of freedom, since `σ = Gxx * sqrt(2/ν)`. The code
(`speckit/analysis.py`) sets `K = navg`, the number of starts in `D`, and
uses that in `Gxx_dev`, `Gyy_dev`, `Gxy_dev`, `Hxy_dev`, `coh_dev`, and the
`*_error` properties. Overlap correlates neighboring segments, so the
variance is larger than `Gxx² / navg`.

`M2` / `Gxx_emp_dev` is a different estimator (the scatter of the segments
inside one record). It is not a substitute for the analytic formula, and
this bug is not fixed by switching callers over to `M2`.

## Current status

Reproduced on 24 Oct 2026 with 40 realizations. Not fixed. The ratios below
are evidence of a bias. They are not constants to multiply into `Gxx_dev`.

## Repro

```python
import numpy as np
from speckit.analysis import SpectrumAnalyzer

def mc(olap, nreal=40):
    fs, N = 200.0, 40_000
    freq, L = 20.0, 512
    rng = np.random.default_rng(4)
    vals = []
    pred = K = None
    for i in range(nreal):
        x = rng.standard_normal(N)
        ana = SpectrumAnalyzer(
            x, fs, win="hann", olap=olap, order=-1,
            Jdes=30, Kdes=20, scheduler="ltf",
        )
        r = ana.compute_single_bin(freq, L=L)
        vals.append(r.Gxx[0])
        if i == 0:
            pred = r.Gxx_dev[0]
            K = int(r.navg[0])
    vals = np.array(vals)
    return np.std(vals, ddof=1), pred, K, np.mean(vals)

for olap in (0.0, 0.5, 0.75):
    std, pred, K, mean = mc(olap)
    print(olap, K, std / pred, mean)
```

`np.std` in the original run was the default `ddof = 0`. With `nreal = 40`
the `ddof` choice moves the ratio by about 1%. Use `ddof = 1` in a new test
and say so. The original numbers:

| Overlap | `K` | measured std / `Gxx_dev` | mean PSD | `2/fs` |
|---------|-----|--------------------------|----------|--------|
| 0.00    | 78  | 1.08                     | 1.031e-2 | 0.01   |
| 0.50    | 155 | 1.20                     | 1.006e-2 | 0.01   |
| 0.75    | 310 | 1.64                     | 1.010e-2 | 0.01   |

The standard error of a standard-deviation estimate at `n = 40` is about
10%, so 1.08 at zero overlap is consistent with 1, and 1.64 at 75% overlap
is not. Hann at 50% is only mildly optimistic. The default Kaiser window
(`psll = 200`) asks for overlap about 0.766 (`kaiser_rov(kaiser_alpha(200))`),
so the default configuration sits on the 75% side of this table. That Kaiser
case was not Monte Carlo'd. Do not copy the Hann factor 1.64 onto Kaiser.

One-sided level of the mean PSD is about `2/fs` for unit-variance
`standard_normal`, as required. This bug is the width of the error bar, not
the PSD scale.

## What was tried

The Monte Carlo above. No effective-`K` formula was coded. Multiplying
`Gxx_dev` by 1.64, or by `1/sqrt(1 - olap)`, would hide the Hann 75% case
and fail for another window. That is the patch this note exists to prevent.

## Context

**Formula to implement.** For a window normalized to unit energy,
`w / sqrt(sum(w**2))`, and hop `Δ` samples between starts,

```
sm = Σ_{m=1}^{K-1} (1 - m/K) |w · w_shifted_by_m_Δ|²
ν  = 2 K / (1 + 2 sm)
σ[Gxx] = Gxx * sqrt(2 / ν)
```

When `Δ = L` or `K = 1`, `sm = 0` and `ν = 2 K`, so `σ` equals today's
`Gxx / sqrt(navg)`. Overlap makes `sm > 0` and `ν < 2 K`.

This is the same algebra as LTPDA `overlapCorr` plus `getdof`, and as
`psdconf`, once both have divided the window by `sqrt(sum w²)`. It is
Percival and Walden, *Spectral Analysis for Physical Applications* (1993),
p. 292. `getdof`'s LPSD path recomputes one segment length for the whole
record. SpecKit's `L` changes with frequency. Use this bin's `L`, this bin's
`navg`, this bin's window, and the hop implied by `D` (the median gap between
starts is enough when the plan spaces starts evenly). Do not call a
whole-record overlap routine.

`psdconf` rounds `ν`. Keep `ν` unrounded in the division. Rounding changes
a large `K` by less than 1 and is not the bug.

**Which properties.** Today's `Gxx_dev = Gxx / sqrt(navg)` should become
`Gxx * sqrt(2/ν)`. The same replacement of `navg` by `ν/2` applies to every
formula that treats `navg` as an independent-segment count: `Gyy_dev`,
`Gxy_dev`, `Gxy_error`, `Hxy_dev`, `Hxy_mag_error`, `Hxy_rad_error`,
`coh_dev`, `coh_error`, `Gxx_error`, `Gyy_error`. Do that only after
`docs/handoffs/20261024_phase-error-arcsin.md` has removed the `arcsin`, so
the phase formula is scaled once.

Do not scale `M2` or `Gxx_emp_dev`. Those are measurements.

**Zero-overlap lock.** Any window at `olap = 0` must reproduce the current
`Gxx_dev` to float tolerance. That is the test that the new formula is the
old one in the independent limit, not a second definition.

**`test_errors.py`.** It uses Hann, overlap 0.5, and `rel = 0.2` against a
200-realization Monte Carlo of one bin. A correct `ν` should move the
prediction toward the measured scatter. If that test fails after the change,
read the measured ratio before widening `rel`. A failure at overlap 0 would
mean the `sm = 0` reduction is wrong.

Confidence intervals (`chi2` bounds on `Gxx`) are optional for closing this
bug. They are specified in the other handoff. Landing `ν` and using it in
`Gxx_dev` is the fix.

## Success criteria

- Hann, overlap 0, the repro's `L = 512`: `ν == 2 * navg`, and `Gxx_dev`
  matches `Gxx / sqrt(navg)` to float tolerance.
- Hann, 50% overlap, large `navg`: `ν / (2 * navg)` matches the Percival and
  Walden factor for the Hann window at that overlap. Record the reference
  value in the test docstring. Do not tune it to the 1.20 ratio above.
- Repeating the 75% Hann Monte Carlo, the ratio measured std / `Gxx_dev`
  moves from about 1.6 toward 1. With 40 realizations, "toward 1" means
  inside roughly 1.0 ± 0.15, not equality to 1.64. State `ddof`.
- Kaiser `psll = 200` at its default overlap is a separate check, not a copy
  of the Hann factor. `ν` must come from that window's autocorrelation.
- Existing PSD levels (`Gxx` itself) are unchanged.
  `tests/integration/test_accuracy.py` stays green.
- `ruff check` and `ruff format --check` on the touched files.

## Recommended next steps

1. Finish the `arcsin` removal in the phase-error handoff first, or do it in
   the same change before touching `navg`.
2. Add a NumPy helper that returns `ν` from `w`, `L`, `navg`, and the hop.
   No Numba kernel.
3. Point `Gxx_dev` at `Gxx * sqrt(2/ν)`. Confirm overlap 0 is unchanged.
4. Repeat the Hann Monte Carlo at 0, 0.5, and 0.75. Then apply `ν/2` in
   place of `navg` in the other Bendat properties.
5. Stop and update this note if the Hann 50% factor disagrees with Percival
   and Walden by more than the discretization of an integer hop. Do not
   average that book with a fitted 1.20.

## References

- `speckit/analysis.py` — `Gxx_dev` and the `*_dev` / `*_error` block
- `docs/handoffs/20261024_overlap-degrees-of-freedom.md` — LTPDA `getdof` /
  `psdconf` reading and the chi-square interval. Same `ν`. Not a second
  formula.
- `docs/handoffs/20261024_phase-error-arcsin.md` — fix before scaling the
  phase error by `ν`
- Percival and Walden (1993), p. 292
- Bendat and Piersol, *Random Data* — the independent-segment limit already
  implemented
- `ltpda/classes/+utils/@math/overlapCorr.m`, `ltpda/classes/@ao/getdof.m`,
  `ltpda/classes/@ao/psdconf.m`
- `tests/integration/test_errors.py` — Hann, overlap 0.5, `rel = 0.2`

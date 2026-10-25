# Handoff: overlap-corrected degrees of freedom and confidence intervals

## Scope

Item 4 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Per-bin degrees of freedom
from the overlap correlation of the window actually used, and a confidence
interval on the one-sided PSD. Coherence gets its own interval. This note
does not cover `new_ltf_plan`.

## Goal

On `SpectrumResult`, expose a per-bin `nu` and a confidence interval for the
one-sided PSD. Empirical `M2` stays. It is the nonparametric scatter. The new
interval does not replace it, and it does not change `Gxx`, `Gxy`, or the
segment count `navg`.

## Current status

**Steps 1 and 3 done (25 Oct 2026), steps 2 and 4 open.** Per-bin `nu` is
implemented as `speckit/core.py::_overlap_dof` and exposed as
`SpectrumResult.nu`. It uses SpecKit's `L`, the window, and the actual
starts in `D`, unrounded. The analytic and empirical errors divide by
`nu / 2`. See `docs/handoffs/20261024_overlap-independent-segments.md` for
the formula, tests, and Monte Carlo.

`SpectrumResult.gxx_interval` is the chi-square interval on `Gxx` (step 3).
The assumptions and the single-bin coverage test are in
`docs/reference/spectral-contract.md` and
`tests/integration/test_estimator_contract.py`.

A 400-realization rerun puts the Hann 75% ratio below at 1.39, not 1.64.
P&W predicts 1.385.

Not done: the LTPDA `getdof.m` / `psdconf.m` side-by-side at one `L` (step
2) and coherence intervals (step 4).

The original status: not started, with `Gxx_dev = Gxx / sqrt(navg)` and the
Bendat formulas using the segment count. That is the large-`K`, independent-segment limit, where
`nu = 2 * navg` and `std(S) = S * sqrt(2/nu) = S / sqrt(navg)`.

A 40-realization check (24 Oct 2026) shows that formula is optimistic once
segments overlap. Hann window, `compute_single_bin`, `fs = 200`, `N = 40000`,
`freq = 20` Hz, `L = 512`, `order = -1`, `numpy` Generator seed 4:

| Overlap | `K` | measured std / `Gxx_dev` |
|---------|-----|--------------------------|
| 0.00    | 78  | 1.08                     |
| 0.50    | 155 | 1.20                     |
| 0.75    | 310 | 1.64                     |

The ratio at 75% overlap is the target behavior of `sqrt(nu_independent / nu)`,
not a number to hard-code. Forty realizations put the standard error of a
standard-deviation estimate near 10%, so 1.64 is a real bias and not a gold
value.

## Context

**What SpecKit stores.** Each bin has `L`, `navg`, starts `D`, and the window
used to form `S2 = sum(w**2)`. The default scheduler is `vectorized_ltf_plan`.
Segment length and overlap vary with frequency. A single `nu` for the whole
spectrum is the wrong object.

**LTPDA has two call sites. The algebra matches. The segment length does not.**

Both of the following normalize the window to unit energy,
`w / sqrt(sum(w**2))`, and then sum lagged products.

`ltpda/classes/+utils/@math/overlapCorr.m` (Percival and Walden, *Spectral
Analysis for Physical Applications*, Cambridge, 1993, p. 292) returns

```
R = (2 / navs) * sum_{j=1}^{navs-1} (1 - j/navs) |w_0 · w_j|^2
```

with `w` already divided by `sqrt(ws2)`. `ltpda/classes/@ao/getdof.m` then
sets

```
dof = 2 * navs / (R * navs + 1)
```

which is `2 * navs / (1 + 2 * sum ...)`. For `navs == 1`, `dof = 2`. The
Welch branch rounds. The LPSD branch stores the unrounded value, but it calls
`overlapCorr(Win.type, nx, navg, olap)`, which **recomputes one segment
length for the whole record**. That is wrong for an LPSD/LTF plan, where `L`
changes with frequency.

`ltpda/classes/@ao/psdconf.m` does normalize (`win ./ sqrt(w.ws2)` near the
PSD branch and `owin = w.win ./ sqrt(w.ws2)` in the LPSD branch). It
accumulates

```
sm = sum_{m=1}^{navg-1} (1 - m/navg) |w · w_shifted_by_m|^2
dof = 2 * navg / (1 + 2 * sm)
```

and rounds. The LPSD branch uses the bin's own segment length and
`n = floor(shift)`. With a non-integer shift, `floor` is not the same overlap
as SpecKit's per-start rounding.

So `getdof`'s formula and `psdconf`'s formula agree when they see the same
normalized window and the same lags. They disagree when one of them invents
a segment length. **Use SpecKit's `L`, `navg`, window samples, and the actual
hop between starts in `D`.** Do not port `overlapCorr`'s whole-record `L`
search into the LPSD path.

**Chi-square interval** for a one-sided Gaussian PSD estimate with `nu`
degrees of freedom, `alpha = 1 - conf`:

```
S_low  = nu * S_hat / chi2.ppf(1 - alpha/2, nu)
S_high = nu * S_hat / chi2.ppf(alpha/2, nu)
var(S_hat) = 2 * S_hat**2 / nu
```

`psdconf.m` writes `c = dof ./ chi2inv([1 - alpha/2, alpha/2], dof)` and
multiplies `S * c`. Same interval. Default confidence there is 95%. Reported
variance is `S**2 * 2 / dof`.

**Coherence.** Do not reuse the PSD chi-square multiplier for `coh`. Bendat's
`coh_dev` is a different expression. A coherence interval is a separate
quantity.

**Phase errors.** Out of scope here. `Hxy_rad_error` currently wraps
`arcsin` around `sqrt(1 - coherence)` and is not the Bendat phase standard
deviation. Do not "fix" that inside the PSD interval.

## Settled decisions

- `nu = 2 * navg` when the hop is the full segment (`R = 0`). That must
  reproduce today's `Gxx_dev`.
- Keep unrounded `nu` in the chi-square interval. Round only if a test
  against LTPDA's published integers requires it, and document that choice.
- Prefer a NumPy helper. Do not add this sum to a Goertzel kernel unless the
  window correlation becomes a measured bottleneck.
- The PSD values are unchanged. Existing spectrum tests must stay green.
- If a Hann, 50% case still disagrees between `getdof.m` and `psdconf.m` by
  more than rounding after both are evaluated at the **same** `L` and hop,
  report both numbers and follow Percival and Walden. Do not average them.

## Success criteria

- No overlap, any window: `nu == 2 * navg`, and `S_hat / sqrt(navg)` equals
  `S_hat * sqrt(2/nu)`.
- Hann, 50% overlap, large `navg`: `nu / (2 * navg)` matches the Percival and
  Walden overlap factor for that window. The tolerance comes from that
  reference value, not from eye.
- The Hann 75% Monte Carlo above moves from a predicted/measured ratio of
  about 1.6 toward 1 when the prediction uses `S * sqrt(2/nu)` instead of
  `S / sqrt(navg)`. State the same `N`, `fs`, `L`, and seed if you repeat it.
- A white Gaussian series, seed fixed, `N` large enough that the lowest bin
  used has `navg >= 30`: the fraction of bins whose true one-sided PSD
  (`2/fs` for unit-variance `standard_normal`, Hann or boxcar, `order = -1`)
  lies inside the 95% interval is consistent with 95% at binomial
  uncertainty. Bins with small `navg` are the interesting case. State `N`,
  `fs`, `Jdes`, `Kdes`, window, and overlap in the test.
- Coherence interval, if implemented, is tested against a known coherence,
  not against the PSD coverage fraction.
- `python -m pytest` on the new test and the existing error test
  `tests/integration/test_errors.py`. `ruff check` and `ruff format --check`
  on the touched files.

## Recommended next steps

1. Implement `nu` from the normalized window and the actual hop in `D`.
2. Check a single Hann, 50% bin against both LTPDA expressions evaluated at
   that `L`. Record the two numbers in this section if they differ.
3. Add the chi-square interval on `Gxx` only.
4. Leave coherence intervals as a second change.

This item is step 1 of the cross-item order. Items 1 and 2 want it before
they claim a residual is "inside the error bar."

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- Percival and Walden (1993), p. 292
- Bendat and Piersol, *Random Data* — `Gxx_dev`, `coh_dev`, `Hxy_dev` already
  in `SpectrumResult`
- `ltpda/classes/+utils/@math/overlapCorr.m`
- `ltpda/classes/@ao/getdof.m`
- `ltpda/classes/@ao/psdconf.m`
- `README.md` — analytic errors and Welford `M2`

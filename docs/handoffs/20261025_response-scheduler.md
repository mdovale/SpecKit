# Handoff: response-aware plan risk

## Scope

Phase 2 of `docs/handoffs/20261024_speckit-2.0-blueprint.md`, limited to
the first prototype: a scheduler that minimizes a loss written down before
the search, using the bin response that Phase 1 already computes.

In:

- a predicted mean, variance, and risk for one plan row, from
  `BinResponse` and `_overlap_dof`, for a spectrum the caller supplies
- a white-noise identity test of that prediction
- three frozen comparison cases against `vectorized_ltf_plan`, at a
  matched segment-sample cost
- a prototype search that does not replace the default scheduler
- a notebook that prints both losses

Out:

- debiasing, dynamic multitaper, and robust replicate reduction. Those
  are the other Phase 2 prototypes. They wait until this loss exists
- any edit to `ltf_plan`, `vectorized_ltf_plan`, `lpsd_plan`, or
  `new_ltf_plan`
- making the prototype the default, or registering it on
  `SpectrumAnalyzer` unless it wins every frozen case
- the N-channel matrix, gap filling, coherence intervals, and adapters
- a real instrument record
- a version bump. Stay on `1.0.3`. A public scheduler name, if one is
  earned, is recorded under `## Unreleased` in `CHANGELOG.md`

Do not delete or rewrite the ten feature handoffs. Phase 0 and Phase 1
stay as written.

## Goal

Judge a frequency plan by predicted bias and predicted variance together.
The baseline is the current default, `vectorized_ltf_plan`. A candidate
replaces nothing unless the frozen cases say it has a lower risk at the
same segment-sample cost. A loss that the candidate does not win is still
a finished result.

## Current status

Implemented 25 Oct 2026. `speckit/plan_risk.py` predicts the one-sided mean
on the DFT grid of the segment and the variance from `_overlap_dof`. A flat
`G` returns `G` at FFT roundoff. The sum uses `L * S2` and no extra scale.
The constant-Hann-length search is not exported from `speckit` and is not a
scheduler name. `vectorized_ltf_plan` is still the default. The package
version is still `1.0.3`. `ltf_plan` and `vectorized_ltf_plan` were not
edited.

Hann, `order=-1`, cost within 5 percent:

- White. `fs = 256`, `N = 256 * 200`, band 8–64 Hz, `Jdes = 48`,
  `Kdes = 40`, overlap 0.5, `bmin = 1`, `Lmin = 1`. Search mean risk
  `5.20e-9` against the baseline `3.50e-8`, at 967110 segment-samples
  against 921067. The search kept `L = 10`. The fixed reference
  `L = 256`, no overlap, matches `2 G**2 / 400`.
- Pink. `fs = 256`, `N = 2**16`, band 2–8 Hz, same scheduler settings.
  The spectrum in the sum is `2 / max(f, 0.2)`. Search mean risk
  `1.57e-3` against the baseline `1.29e-3`, at 822528 against 784599.
  The search kept `L = 336`. This loss keeps it off `SpectrumAnalyzer`.
- Half-bin tone. `fs = 64`, `N = 64 * 32`, `Jdes = 32`, `Kdes = 20`.
  Squared leakage `8.06e-4` against `1.79e-2`, at 8128 against 8136.
  The search kept `L = 8` with the analysis frequency at 8 Hz.

## Context

**Estimand.** One-sided WOSA of a uniformly sampled, complete, real series.
`Gxx = 2 * XX / (fs * S2)` with `S2 = sum(w²)`. Generator `psd` is
two-sided, so a flat generator level compares with `Gxx = 2 * psd`. For
unit-variance `standard_normal`, or `white_noise(psd=1/fs)`, the one-sided
level is `2/fs`. Pink noise is two-sided `f**(-1)` and one-sided `2/f`.
`Gxy = X * conj(Y)`.

**What a row already costs.** For length `L`, window `w`, and starts `D`,
`nu = _overlap_dof(w, D)`. With no overlap, `nu = 2 * K`. The locally white
variance is `2 * G**2 / nu`. That formula ignores the detrend, the slope
of `G` across the main lobe, and the DC and Nyquist bins. `ENBW` for
`order=-1` is `fs * S2 / S1**2`. `BinResponse.enbw` matches that and the
DFT integral of the window. `SpectrumResult.ENBW` stays on the raw window
even when `order >= 0`.

**Prediction this note is allowed to use.** `order=-1` only. The analysis
frequency is `f`. `H(ξ) = BinResponse.coefficient(ξ)` is the direct
coefficient of `exp(i 2π ξ n / fs)`. For a one-sided interior target `G`,

```
W(f, ν)  proportional to  |H(ν)|**2 + |H(-ν)|**2
∫_0^{fs/2} W(f, ν) dν  =  1
predicted_mean(f)  =  ∫_0^{fs/2} W(f, ν) G(ν) dν
predicted_var(f)   =  2 * predicted_mean(f)**2 / nu
risk(f)            =  (predicted_mean(f) - G(f))**2 + predicted_var(f)
```

`ν` here is a frequency, not degrees of freedom. The integral is a sum on
a grid that is stated next to the code. A DFT grid of the segment is the
one that should make a flat `G` exact, because that is the same identity
as `BinResponse.enbw_numerical`. A flat `G` must return
`predicted_mean = G` without an extra scale. If it does not, stop.

**Cost.** Segment-samples are `sum_j K_j * L_j`. Wall time is recorded and
is not the loss. Two plans are cost-matched when their segment-sample
totals differ by at most 5 percent. The prototype may not buy a lower risk
by processing more samples. A win at higher cost is reported and is not
the predeclared win.

**Where the code lives.** The search is not a method on `SpectrumAnalyzer`
and is not exported from `speckit/__init__.py`. Put it in a module the
public scheduler map does not import, or under `benchmarks/`. It may call
`BinResponse`, `_overlap_dof`, and `SpectralPlan.from_dict`. It must not
call the Goertzel kernel to build `W`. Numba and NumPy bin kernels stay
on the summary statistic they already compute.

**Annotation style.** Python 3.9. Keep `typing.Optional`, `Union`, `Tuple`,
`List`, and `Dict` in new modules. `ruff` `UP006`, `UP035`, and `UP045` on
those forms are not a reason to rewrite them.

## Settled decisions

- `vectorized_ltf_plan` remains the default. `ltf_plan` and
  `vectorized_ltf_plan` keep their present lengths, including the 424
  differences on the scheduler fixture.
- The loss above is the objective. It is not refit after seeing which
  plan looks better on a plot.
- The white-noise identity is a stop. A factor of 2, an ENBW patched in
  at the call site, or a wider tolerance that hides a scale error is not
  a fix.
- `order=-1` is the only order in the prediction. A detrended risk is a
  later note. Do not claim `SpectrumResult.ENBW` is the detrended one.
- Variance uses `nu` from the actual starts. Overlap does not make
  `nu = 2 * K`.
- Neighboring bins share samples. A mean of `risk(f)` over a plan is a
  sum of pointwise risks, not a variance of the sum.
- The prototype becomes a named scheduler only when its mean risk is lower
  on every frozen case at matched cost. Otherwise it stays out of
  `SpectrumAnalyzer`. The comparison is still committed.
- The 1.0.3 lock hashes stay put. `missing="raise"` stays the default.
- One-sided estimates and two-sided generator levels stay as they are.
  `Gxy = X * conj(Y)`. A delayed channel 1 has cross-spectrum phase
  `+ω τ` and `Hxy` phase `-ω τ`.

## Success criteria

- On Hann, `order=-1`, one interior bin, a flat one-sided `G`, the
  predicted mean equals `G` at a relative tolerance written next to the
  assertion and justified by the integration grid. The DFT-grid form of
  the same identity is at most a few times `1e-8` relative, in line with
  `BinResponse.enbw_numerical`. Predicted variance equals
  `2 * G**2 / nu` for that row's starts.
- The frozen cases, all `order=-1`, Hann, seed `20261025` where a series
  is drawn:
  - White. `fs = 256`, `N = 256 * 200`, `white_noise(psd=1/fs)`, band
    8–64 Hz. Truth `2/fs`. Fixed-`L` reference `L = 256`, no overlap.
    Baseline also includes `vectorized_ltf_plan` with stated `Jdes` and
    `Kdes`.
  - Pink. `fs = 256`, `N = 2**16`, `f_min = 0.2`, `f_max = 80`, band
    2–8 Hz. Two-sided generator level `f**(-1)`, one-sided truth `2/f`.
  - Half-bin tone. `fs = 64`, `L` candidate around 64, tone at
    `f + 0.5 * fs/L` for the analysis frequency `f = 8` Hz, rectangular
    or Hann window stated in the test. The quiet neighbor is one
    resolution bandwidth away. The leakage term in the risk is the
    predicted power in that neighbor from `cosine_coefficient`.
- For each case the test reports mean risk and segment-samples for the
  baseline and the prototype. Cost differs by at most 5 percent. The
  prototype is registered as a scheduler only if its mean risk is lower
  on all three. A loss is asserted as a loss, not adjusted.
- `python -m pytest tests/integration/test_estimator_contract.py tests/unit/test_schedulers.py -q`
  passes, plus the new prediction and comparison tests. The lock hashes
  and the 424 length differences are unchanged.
- `ruff format --check` passes on the new Python. `ruff check` on new
  tests passes. Annotation findings that repeat the `typing` forms above
  are left in place.
- `docs/reference/spectral-contract.md` gains a short paragraph that the
  default plan is still the heuristic, and points here for the risk
  prototype. `notebooks/1.7_plan-risk.ipynb` prints the baseline loss, the
  prototype loss, and the segment-sample totals on the pink case and the
  tone case.

## What was tried

The DFT sum `sum |H(ν_k)|**2 * G(|ν_k|) / (L * S2)` recovers a flat
one-sided `G`. On Hann, `L = 256`, `fs = 256`, `f = 32`, the relative
residual was about `1e-16`. No factor of 2 was applied. An off-grid
analysis frequency, `f = 32.5`, recovers the same flat level because
Parseval holds for the modulated window.

Scoring every feasible segment count with `_overlap_dof` is the same
objective and was slow: the white band admits on the order of `10**4`
counts. Bias does not depend on the starts, so each length keeps the
count with the largest `nu`. A uniform-hop form of Percival & Walden
eq. 292b ranks those counts. `_overlap_dof` on the rounded starts then
walks to a neighboring count while `nu` increases. The reported risk
uses that value. On `N = 600`, `fs = 60`, lengths 20, 30, and 40, the
search matches an exhaustive score of the cost band.

The pink result was not reweighted after the run. The mean is the
unweighted mean of the pointwise risks. Baseline lengths on 2–8 Hz run
from 479 down to 162. The search kept one length, 336, and lost.

The tone risk is the squared leaked density of a deterministic cosine.
`2 G**2 / nu` is the continuous-spectrum variance and was not added.
A half-bin tone's leaked density scales with the line height, which
grows as the resolution narrows, so the short length wins that case.
The three results together are not three wins, so the search was not
given a scheduler name.

```
python -m pytest tests/unit/test_plan_risk.py \
  tests/integration/test_response_scheduler.py \
  tests/integration/test_estimator_contract.py \
  tests/unit/test_schedulers.py -q
```

That run passed, 39 tests. The 1.0.3 lock and the 424 length
differences are unchanged.

## Recommended next steps

1. Leave `speckit/plan_risk.py` unexported. Register a scheduler name
   only if a later comparison, on a loss fixed before the search, is
   lower on every frozen case, including the pink band.
2. The next design note is one remaining Phase 2 estimator: debiased
   dynamic WOSA, dynamic multitaper, or robust replicate reduction.
3. Keep `ltf_plan` and `vectorized_ltf_plan` as they are in that note.
   A second copy with a patched rounder is a retune, not a new loss.

## Files touched

- `speckit/plan_risk.py` — predicted mean, variance, risk, and the
  constant-length search. Not imported by `speckit/__init__.py`.
- `tests/unit/test_plan_risk.py` — flat-spectrum identity, leakage scale,
  exhaustive cost band
- `tests/integration/test_response_scheduler.py` — white, pink, and tone
  comparisons
- `docs/reference/spectral-contract.md` — the default plan is still the
  heuristic
- `notebooks/1.7_plan-risk.ipynb` — pink and tone losses
- `speckit/response.py` was not required. `enbw` for `order=-1` is unchanged.
- No scheduler name, so `CHANGELOG.md` has no new entry. Version stays
  `1.0.3`.

## Test gaps

The flat identity and the three frozen cases are tested. Still open:
a detrended risk (`order >= 0`), debiasing, dynamic multitaper, robust
replicate reduction, and a real instrument record. The pointwise risks
are averaged as a sum of bin risks. They are not a variance of the sum,
and they are not a covariance between bins.

## References

- `docs/handoffs/20261024_speckit-2.0-blueprint.md` — Phase 2
- `docs/handoffs/20261025b_spectral-plan.md` — `BinResponse`, `SpectralPlan`
- `docs/handoffs/20261025_estimator-contract.md` — the 1.0.3 lock
- `docs/reference/spectral-contract.md`
- `docs/reference/frequency-scheduling.md` — the heuristic and the 424
  length differences
- `docs/reference/spectrum-estimation.md` — `Gxx` and `ENBW`
- Heinzel, Rüdiger, and Schilling (2002) — the kernel behind ENBW
- Percival and Walden (1993), eq. 292b — `nu`
- Tröbs and Heinzel (2006) — the logarithmic plan this search is
  compared with, not an objective
- `speckit/response.py` — `coefficient`, `cosine_coefficient`, `enbw`
- `speckit/core.py` — `_overlap_dof`
- `tests/integration/test_spectral_corpus.py`
- `notebooks/0.6_schedulers.ipynb`, `notebooks/1.6_bin-response.ipynb`

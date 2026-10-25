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

Not started. Phase 1 is implemented. `SpectralPlan` copies a scheduler
dict, `BinResponse` evaluates one row, and `compute_single_bin` can keep
segment coefficients. The seeded corpus in
`tests/integration/test_spectral_corpus.py` locks white noise, pink noise,
a half-bin tone, an integer delay, and a one-pole. It does not score a
plan.

`vectorized_ltf_plan` is the `SpectrumAnalyzer` default. On `N = 10**6`,
`fs = 1`, overlap 0.75, `bmin = 5`, `Lmin = 1000`, `Jdes = 1000`, and
`Kdes = 100` it returns 849 frequencies, one `K = 1` bin, and 425 lengths
at `Lmin` or `N`. `ltf_plan` differs in `L` on 424 of those 849 bins. Neither
plan is the solution of a stated loss.

The 1.0.3 lock in `tests/integration/test_estimator_contract.py` is
`seed = 20261025`, `N = 4096`, `fs = 200`, `Jdes = 64`, `Kdes = 20`,
default scheduler and Kaiser window. Those hashes must not move.

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

## Recommended next steps

1. Implement the predicted mean, variance, and risk for one row. Stop if
   flat `G` is not recovered without a scale factor.
2. Freeze the three cases and the cost rule in tests, with the baseline
   `vectorized_ltf_plan` scored first.
3. Search segment lengths under the cost cap. Keep the existing
   schedulers unchanged.
4. Register a public name only if every frozen case improves. Otherwise
   leave the prototype unexported and record the case it loses.
5. Update the spectral-contract paragraph and the notebook.

If the search needs a second copy of `vectorized_ltf_plan` with a patched
rounder, stop. That is a retune, not a new objective.

## Files touched

Expected, not yet edited:

- a new module outside `speckit/__init__.py`, or a script under
  `benchmarks/`, that builds candidate plans and scores them
- `speckit/response.py` only if the kernel `W` belongs next to
  `BinResponse`. Do not change `enbw` for `order=-1`
- new tests under `tests/unit/` for the white identity and under
  `tests/integration/` for the three cases
- `docs/reference/spectral-contract.md`
- `CHANGELOG.md` under `## Unreleased` only if a scheduler name is added
- `notebooks/1.7_plan-risk.ipynb`

## Test gaps

Nothing scores `predicted_mean` against a known `G`. Nothing compares
`vectorized_ltf_plan` to another plan on bias and variance together.
The corpus checks levels and one tone. It does not check a scheduler
objective. Debiasing, multitaper, and robust reduction have no benchmark
yet. Real-data cases remain in the blueprint.

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

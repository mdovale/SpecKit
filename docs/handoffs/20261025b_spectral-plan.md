# Handoff: spectral plan, bin response, and replicate retention

## Scope

Phase 1 of `docs/handoffs/20261024_speckit-2.0-blueprint.md`: make the
measurement operator that 1.0.3 already applies available as a checked plan,
a numerical response, and an optional set of segment coefficients.

In:

- a typed, immutable `SpectralPlan` built from the dict `plan()` already
  returns, plus a compatibility view of that dict
- validation of the invariants `plan()` already enforces, before any kernel
  runs
- `report()` that states boundary facts and does not change the plan
- a schema version and a stable hash of the plan and the estimator
  configuration
- a serialization round trip
- a numerical response for one plan row: coherent gain, ENBW, a few leakage
  offsets, negative-frequency folding, the detrend, and a pure delay
- optional per-segment Fourier coefficients, on the direct-DFT oracle first
  and then on one production path
- a small seeded synthetic corpus that later estimator work can extend
- notebook updates that show the plan report, one bin response, and the
  replicate switch

Out:

- a scheduler that optimizes a bias or variance objective, and any retune
  of `ltf_plan` or `vectorized_ltf_plan`
- debiasing, multitaper, robust reduction, and the N-channel matrix
- gap filling, a validity mask, and coherence intervals
- xarray, Astropy, Pint, and GWpy adapters
- a real instrument record. The repository has none, and this note does not
  add one
- a version bump. Stay on `1.0.3`. Record breaks under `## Unreleased` in
  `CHANGELOG.md`

Do not delete or rewrite the ten feature handoffs. Phase 0 stays as written
in `docs/handoffs/20261025_estimator-contract.md`.

## Goal

A caller can recover the exact plan, window, detrend, convention, and
missing-sample policy that produced a spectrum, can hash and reload that
record, and can evaluate how one bin responds to a tone. The summary
spectrum stays the default. Segment coefficients are available when asked
for, and they reduce to the same bin statistic.

## Current status

Not started. Phase 0 is implemented: the spectral contract, the direct-DFT
oracle, `vectorized_ltf_plan` tests, `missing="raise"`, and
`SpectrumResult.gxx_interval`. The seeded 1.0.3 lock in
`tests/integration/test_estimator_contract.py` is the regression this note
must keep.

`SpectrumAnalyzer.plan()` returns a dict. Required keys are `f`, `r`, `b`,
`L`, `K`, `navg`, `D`, and `O`. The method adds `nf`, casts the vectors to
contiguous float64 or int64, and stores `D` as a list of int64 arrays. It
checks length agreement, `K[i] == len(D[i])`, starts inside `[0, N - L]`,
and `L` at least 1 (and at least `Lmin`, except on the `lpsd_plan` path).
A `band` drops rows after the scheduler returns. There is no schema, no
hash, and no response evaluation. Window sums `S1` and `S2` are computed
later, per distinct `L`, inside `compute()`.

The oracle in `tests/direct_dft_oracle.py` returns coefficients for the
segments it is given. It does not import the production package. Production
`BinStats` stores only the reduction `(MXX, MYY, mu_r, mu_i, M2)`.

## Context

**Estimand.** Still the one-sided WOSA of a uniformly sampled, complete,
real series in `docs/reference/spectral-contract.md`. Densities are
one-sided. `Gxx = 2 * XX / (fs * S2)` with `S2 = sum(w²)`. Generator `psd`
arguments are two-sided, so a round trip compares `Gxx` with `2 * psd` on
`(0, fs/2)`. `Gxy = X * conj(Y)`. `ENBW = fs * S2 / S1²`, and `S12` on the
result is `S1²`.

**Plan identity.** The scheduler functions keep returning dicts. `SpectralPlan`
is a validated copy of the dict `plan()` caches, together with the
configuration that the dict does not carry: `fs`, `N`, resolved overlap,
`bmin`, `Lmin`, `Jdes`, `Kdes`, scheduler name, window name, `psll` when the
window is Kaiser, detrend `order`, `missing`, and the convention strings
`one-sided` and `X*conj(Y)`. `band`, when set, is part of the record because
it deletes rows.

**Response.** For one row, the operator is the detrend projection, then the
window, then `sum y[n] exp(-i ω n)` at the bin frequency. `order=-1` leaves
the samples unchanged before the window. Orders `0`, `1`, and `2` use the
projection in the contract: mean, or `y - Q Q.T y` with `t` in `[-1, 1]`.
The production Goertzel value is that sum times `exp(i ω (L-1))`. A response
quoted as a direct-DFT coefficient uses the direct sum. Phase comparisons
use the same convention on both sides.

**Replicates.** `storage="summary"` is the default and is today's
`BinStats`. `storage="replicates"` keeps the complex coefficient of each
segment at that bin. For a cross-spectrum that is the pair `(X_k, Y_k)`.
The mean of `|X_k|²` is `MXX`. The mean of `X_k * conj(Y_k)` is the
cross-spectrum mean. Overlap still couples those replicates. Retaining them
does not make `nu` equal `2 * K`.

**Modules.** `speckit/schedulers.py` owns the dict producers and should gain
the plan type only if the type does not import the analyzer. `speckit/analysis.py`
owns `plan()`, the window, the missing policy, and the result. A response
helper that duplicates `_build_Q` inside `analysis.py` is the wrong home.
Put the numerical response next to the window and the detrend definition it
evaluates, and call the existing `_build_Q` for orders 1 and 2. The oracle
stays free of production imports. Numba and NumPy stay on the same
statistic. New Python modules copy the BSD header from a neighboring
SpecKit file.

**Annotation style.** Python 3.9. Keep `typing.Optional`, `Union`, `Tuple`,
`List`, and `Dict`. The pre-existing `ruff` UP findings in `analysis.py` and
`core.py` are not part of this work.

## Settled decisions

- `plan()` still returns the dict. Callers that read `plan["f"]` and
  `plan["D"]` keep working. `SpectralPlan` is available beside that dict.
  Changing the return type of `plan()` is a break this note does not take.
- The plan object is immutable. Its compatibility dict is a copy. Mutating
  the copy does not change the plan, and mutating the cached dict is not a
  supported way to edit a spectrum.
- The hash is SHA-256 over a documented byte layout: schema integer, then
  the configuration fields in a fixed order, then `f`, `r`, `b`, `L`, `K`,
  `navg`, `O` as little-endian float64 or int64, then each `D[j]` the same
  way. It is not `pickle` and not Python's `hash()`.
- Schema version starts at 1 and is independent of the package version
  `1.0.3`. A field added to the canonical bytes increments the schema.
- `report()` prints or returns counts and ranges. It does not change `Jdes`,
  drop bins, or rewrite `L`.
- On finite inputs the summary `Gxx`, `Gxy`, `nu`, and the plan arrays stay
  on the 1.0.3 lock. Replicate storage is opt-in.
- The response recovers `ENBW = fs * S2 / S1²` for `order=-1` from the
  window that production builds. A mismatch is a stop. It is not a scale
  factor at the call site. For `order >= 0` the reported ENBW is the ENBW
  of the window after the projection, and the note says so.
- The response helper may call `_build_Q`. It must not call the Goertzel
  kernel or a scheduler. Agreement with the oracle is the check.
- One production path retains replicates by exposing the coefficients the
  existing reduction already forms. A second kernel, or a NumPy path that
  averages a different quantity than the Numba path, is a stop.
- The synthetic corpus publishes seeds and the analytic target. It does not
  declare a winner against Welch or LTPDA. A Welch comparison is in scope
  only for a fixed `L`, a window SciPy has, and an overlap SciPy applies the
  same way. Otherwise record that the estimators differ.
- `ltf_plan` and `vectorized_ltf_plan` stay as they are, including the 424
  length differences on the scheduler fixture.
- `missing="raise"` stays the default. The plan records the policy. It does
  not grow a mask.

## Success criteria

- `SpectralPlan` accepts the dict from each of `lpsd_plan`, `ltf_plan`,
  `vectorized_ltf_plan`, and `new_ltf_plan` on the scheduler fixture, and
  rejects a missing key, a length mismatch, an empty `D[j]`, a start outside
  `[0, N-L]`, and `K[j] != len(D[j])`.
- On the lock series (`seed=20261025`, `N=4096`, `fs=200`, `Jdes=64`,
  `Kdes=20`, default scheduler and Kaiser window) the compatibility view
  matches `f`, `r`, `L`, `K`, and `D` that the existing test hashes. Building
  the plan does not change those hashes or the `Gxx` and `Gxy` prefixes.
- Two builds of the same analyzer give the same digest. Changing one start,
  the window name, or `order` changes it. A JSON round trip (metadata as
  JSON, arrays as the canonical bytes or as nested lists) rebuilds a plan
  with the same digest.
- `report()` on that lock plan states `nf`, the frequency range, the `L`
  range, the median `K`, the achieved overlap, and how many bins have
  `K=1` or `L` at a bound. The plan arrays before and after the report are
  identical.
- For one interior row, Hann and the default Kaiser window, and each
  `order` in `{-1, 0, 1, 2}`: the complex response at the bin center to a
  complex exponential matches the direct-DFT oracle within
  `8 * L * eps * sum(|y_windowed|)`. For `order=-1`, the numerical ENBW
  matches `fs * S2 / S1²` at a tolerance written next to the assertion.
  The same row reports the response at the bin center, at half a resolution
  bandwidth, at the first sidelobe the window claims or at `3 * r`, and at
  the negative frequency. A pure tone on channel 0 and the same tone delayed
  by an integer number of samples on channel 1, `order=-1`, has cross-spectrum
  phase `-ω * delay` within that coefficient tolerance.
- The oracle returns per-segment coefficients. With `storage="replicates"`
  on one production path, both kernel families return those coefficients for
  one auto bin and one cross bin, and the reduction of the replicates matches
  `MXX` and the cross-spectrum mean of `storage="summary"` on the lock
  series. `storage="summary"` remains the default. An unknown storage value
  raises.
- The corpus is executable and states, for each case, `N`, `fs`, the seed,
  the estimator settings, and the target:
  - white unit-variance Gaussian, true one-sided level `2/fs`
  - one power-law spectrum from `speckit.noise`, with the two-sided generator
    level and the one-sided target both written down
  - a tone on an integer bin and a tone halfway between bins, with the
    expected peak bin and a neighbor leakage level that the response
    predicts
  - one stable filter or an integer delay, with the expected `Hxy` phase
- `python -m pytest tests/unit/test_schedulers.py tests/unit/test_direct_dft.py tests/integration/test_estimator_contract.py -q`
  passes, plus the new plan, response, replicate, and corpus tests.
  `ruff check` and `ruff format --check` pass on the new Python. Pre-existing
  findings in `speckit/analysis.py` and `speckit/core.py` stay untouched.
- `docs/reference/spectral-contract.md` gains a short section that names the
  plan fields, the hash, and the response, and points here. The notebooks
  below run the new cells.

## Recommended next steps

1. Add `SpectralPlan`, the copy-out dict, the hash, and `report()`, with
   tests against the four schedulers and the 1.0.3 lock. Stop if the lock
   hashes move.
2. Add the single-row response and check it against the oracle and against
   `fs * S2 / S1²` at `order=-1`. Stop on a scale mismatch.
3. Return per-segment coefficients from the oracle, then from one production
   path, and reduce them to the summary statistic.
4. Add the four synthetic cases. The leakage case uses the response from
   step 2 as the predicted neighbor level.
5. Update `docs/reference/spectral-contract.md` and the notebooks.

If step 2 or 3 needs a new kernel or a change to `Gxx` on finite data, stop
and extend this note with the repro, the formula, and the files. Do not land
a special case for one `order` or one window.

## Files touched

Expected, not yet edited:

- `speckit/analysis.py` and, if the plan type lives with the schedulers,
  `speckit/schedulers.py`
- `speckit/core.py` only if the existing reduction grows an optional
  coefficient output. Both the Numba kernel and the NumPy fallback change
  together
- `tests/direct_dft_oracle.py` and new tests under `tests/unit/` and
  `tests/integration/`
- `docs/reference/spectral-contract.md`
- `CHANGELOG.md` under `## Unreleased`
- `notebooks/0.6_schedulers.ipynb` or `notebooks/1.3_vectorized-ltf.ipynb`
  for `report()` and the digest
- a new notebook, `notebooks/1.6_bin-response.ipynb`, for one row's response,
  the detrend, the delay phase, and `storage="replicates"` against the summary

## Test gaps

`plan()` invariants are enforced and untested as a separate object. Nothing
hashes a plan. Nothing evaluates a bin's response to a tone, a negative
frequency, or a delay. Production code cannot return segment coefficients.
The accuracy tests use broad aggregate tolerances and are not this corpus.
Real-data acceptance cases remain in the blueprint. Coherence intervals
remain in `docs/handoffs/20261024_overlap-degrees-of-freedom.md`.

## References

- `docs/handoffs/20261024_speckit-2.0-blueprint.md` — Phase 1
- `docs/handoffs/20261025_estimator-contract.md` — Phase 0, the lock, the oracle
- `docs/reference/spectral-contract.md`
- `docs/reference/frequency-scheduling.md` — plan keys and the measured
  scheduler differences
- `docs/reference/spectrum-estimation.md` — `Gxx` and `ENBW = fs * S2 / S12`
- `docs/reference/windows.md` — Kaiser samples and `S1`, `S2`
- Heinzel, Rüdiger, and Schilling (2002) — coherent gain, ENBW, leakage
- Seybert and Hamilton (1978) — delay bias of the frequency-response estimate
- Percival and Walden (1993), eq. 292b — `nu` is not `2 * K` under overlap
- `tests/integration/test_estimator_contract.py`
- `tests/direct_dft_oracle.py`
- `notebooks/0.6_schedulers.ipynb`, `notebooks/1.3_vectorized-ltf.ipynb`,
  `notebooks/1.5_direct-dft.ipynb`

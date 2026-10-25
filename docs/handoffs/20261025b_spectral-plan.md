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

Implemented on 25 Oct 2026. `SpectralPlan` validates a copy of the scheduler
dict, `plan()` still returns that dict, and the 1.0.3 lock hashes are
unchanged. `BinResponse` matches the direct-DFT oracle inside the stated
roundoff bound. For `order=-1`, its ENBW matches `fs * S2 / S1**2` and the
DFT integral of the window. `storage="replicates"` on `compute_single_bin`
reduces to the summary `XX` and `XY`. The synthetic corpus passes.

Phase 2 (a response-aware scheduler, debiasing, multitaper, robust
reduction) has not been started.

`ruff check` on the new test modules passes. `speckit/plan.py` and
`speckit/response.py` still report `UP006`, `UP035`, and `UP045` because
they keep `typing.Dict`, `Tuple`, and `Optional`, as
`.cursor/rules/type-annotations.mdc` requires. Those findings were not
rewritten to `dict` or `X | None`. `ruff format --check` passes on the new
Python.

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

## What was tried

The fused Numba reductions were left as they are. They square a Goertzel
value and do not return it. `segment_coefficients` calls the existing
`_goertzel_real_imag` and multiplies by `exp(-i ω (L-1))`, which is the
direct sum. The NumPy path is the same `(rows * w) @ exp(-i ω n)` product
the NumPy kernels reduce. Both match the oracle, and the mean of `|X|**2`
and of `X * conj(Y)` match `_bin_stats`. The summary kernels were not given
an optional output buffer.

The oracle casts its segment to float64, so a complex exponential is
compared as the oracle's cosine coefficient plus `i` times its sine
coefficient. The operator is real and linear. That comparison passes for
Hann and Kaiser and for every `order`.

A delayed channel 1 gives `Y = X exp(-i ω τ)` at an integer bin with a
rectangular window. `Gxy = X * conj(Y)` then has phase `+ω τ`. `Hxy =
conj(XY) / XX` has phase `-ω τ`. The minus sign in the original success
criterion is the transfer-function phase, not the phase of `X * conj(Y)`.

The corpus, all seed `20261025`:

- White noise from `white_noise(fs=256, psd=1/fs)`, Hann, `order=-1`, no
  overlap, `L=256`, `K=200`, bin at 32 Hz. Target one-sided level `2/fs`.
  The draw sits inside four `Gxx_dev`.
- Pink noise, two-sided `f**(-1)`, one-sided target `2/f`, on 2–8 Hz.
  Median `Gxx * f / 2` is inside 0.75–1.25. That checks the factor of two.
  It is not a leakage calibration.
- A half-bin cosine, rectangular window, one segment. `XX` matches
  `|cosine coefficient|**2` from `BinResponse`, which includes the negative
  frequency.
- The integer delay above, and a one-pole `0.2 / (1 - 0.8 z^{-1})` whose
  `Hxy` is within 5% of the filter response (Hann, 80 segments, 32 Hz).

On the scheduler fixture the vectorized plan reports `nf=849`, one `K=1`
bin, and 425 lengths at `Lmin` or `N`. `report()` does not change the arrays.

## Recommended next steps

The criteria in this note are met. The next design note is a response-aware
scheduler or one research estimator from the blueprint's Phase 2. Do not
start it in the same change as a retune of `ltf_plan` or
`vectorized_ltf_plan`.

Verification on 25 Oct 2026:

`python -m pytest -q`

180 passed.

`ruff check` passes on `tests/unit/test_spectral_plan.py`,
`tests/unit/test_bin_response.py`, `tests/unit/test_segment_coefficients.py`,
`tests/integration/test_spectral_corpus.py`, and
`tests/integration/test_estimator_contract.py`. `ruff format --check` passes
on the new Python. Annotation findings in `speckit/plan.py` and
`speckit/response.py` are the `typing` forms named under Current status.

## Files touched

- `speckit/plan.py` (new), `speckit/response.py` (new)
- `speckit/analysis.py` — `spectral_plan`, `storage` on `compute_single_bin`,
  `Xseg` and `Yseg`
- `speckit/core.py` — `segment_coefficients`. The fused summary kernels are
  unchanged
- `speckit/__init__.py` — exports `SpectralPlan` and `BinResponse`
- `tests/unit/test_spectral_plan.py`, `tests/unit/test_bin_response.py`,
  `tests/unit/test_segment_coefficients.py`
- `tests/integration/test_spectral_corpus.py`
- `tests/integration/test_estimator_contract.py` — lock, report, replicates
- `docs/reference/spectral-contract.md`
- `CHANGELOG.md` under `## Unreleased`
- `notebooks/0.6_schedulers.ipynb`, `notebooks/1.6_bin-response.ipynb`

## Test gaps

Closed for this note. The four schedulers build a `SpectralPlan`. The lock
hashes, the digest, the JSON round trip, the response, both kernel families'
coefficients, and the four synthetic cases each have a test. Replicate
storage is the single-bin path only. A full-spectrum replicate buffer was
not added. Real-data acceptance cases remain in the blueprint. Coherence
intervals remain in `docs/handoffs/20261024_overlap-degrees-of-freedom.md`.

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
  `notebooks/1.5_direct-dft.ipynb`, `notebooks/1.6_bin-response.ipynb`

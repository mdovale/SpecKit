# Handoff: estimator contract, direct-DFT oracle, and pointwise PSD interval

## Scope

Phase 0 of `docs/handoffs/20261024_speckit-2.0-blueprint.md`: make the
1.0.3 estimator meet a written contract before any new estimator, plan type,
or LTPDA feature work.

In:

- a spectral-contract reference note
- a seeded lock of released 1.0.3 numbers, taken before any estimator edit
- unit coverage of the default scheduler, `vectorized_ltf_plan`, including
  measured differences from `ltf_plan`
- an independent direct-DFT oracle and tests against both kernel families
- an explicit missing-sample policy, replacing silent `nan_to_num`
- the chi-square interval on the one-sided PSD, with its assumptions labeled
- notebook updates that show the new behavior on the existing examples

Out:

- `SpectralPlan`, response-operator evaluation, and replicate storage
- debiasing, multitaper, robust reduction, and the N-channel matrix
- gap filling, masks beyond the named policy, and coherence intervals
- a version bump. Stay on `1.0.3`. Record breaks under `## Unreleased` in
  `CHANGELOG.md`.

Do not delete or rewrite the ten feature handoffs. When the interval lands,
revise **Current status** in
`docs/handoffs/20261024_overlap-degrees-of-freedom.md` in place.

## Goal

For the quantities 1.0.3 already returns, state the estimand, check the
production kernels against a direct DFT, put the default scheduler under
test, stop treating non-finite samples as zeros, and expose a pointwise
PSD interval whose coverage is tested on the process class it claims.

## Current status

Not started. No code has been added for this note.

What 1.0.3 already has:

- `SpectrumResult.nu` from `speckit/core.py::_overlap_dof`, and analytic and
  empirical errors that divide by `n_d = nu / 2`. With no overlap,
  `nu = 2 * navg`.
- Numba and NumPy bin kernels that both use `exp(-i ω n)`. Auto-spectra
  agree. The cross-spectrum is `X * conj(Y)`.
- Scheduler unit tests for `ltf_plan`, `lpsd_plan`, and `new_ltf_plan` only.
  `vectorized_ltf_plan` is the `SpectrumAnalyzer` default and is outside
  `tests/unit/test_schedulers.py`.
- Silent replacement of non-finite input samples and non-finite bin outputs
  with zero. No test locks either wipe.

The chi-square interval on `Gxx` is still open. Coherence intervals stay
open and are not part of this note.

## Repro

**Scheduler gap.** `tests/unit/test_schedulers.py` builds plans with

```python
SCHEDULER_PARAMS = {
    "N": int(1e6),
    "fs": 1.0,
    "olap": 0.75,
    "bmin": 5.0,
    "Lmin": 1000,
    "Jdes": 1000,
    "Kdes": 100,
}
```

and parametrizes `ltf_plan`, `lpsd_plan`, and `new_ltf_plan`. The same
fixture does not call `vectorized_ltf_plan`. `frequency-scheduling.md`
already says so. The vectorized walker looks up `r` on a logarithmic grid
of `10 * Jdes` points and steps by that `r`. Lengths use `numpy.round`.
`ltf_plan` uses a local half-up rounder. Expected: both plans satisfy
`r * L = fs` and strictly increasing `f` at most Nyquist. Actual differences
in `f`, `L`, and `D` are not yet measured. Do not force the two plans onto
one grid. Record the differences.

**Non-finite samples.** `SpectrumAnalyzer.__init__` in `speckit/analysis.py`
replaces non-finite input samples with 0 after a warning, on both the
one-channel and two-channel paths. After the reduction, `XX`, `YY`, `XY`,
`S12`, `S2`, and `M2` are passed through `numpy.nan_to_num`. Several
`SpectrumResult` properties do it again. A series with one NaN is therefore
a different quadratic form, and a non-finite kernel output is stored as a
physical zero. No current test passes a NaN, so a default of `missing="raise"`
is invisible to the existing suite.

**Interval.** The formula to implement, for a one-sided Gaussian PSD with
unrounded `nu` and `alpha = 1 - conf`, is

```
S_low  = nu * Gxx / chi2.ppf(1 - alpha/2, nu)
S_high = nu * Gxx / chi2.ppf(alpha/2, nu)
var    = 2 * Gxx**2 / nu
```

`Gxx_dev = Gxx * sqrt(2 / nu)` is the standard deviation, not this interval.
Coverage is not yet tested. Use one interior bin and independent
realizations. A fraction taken over a whole spectrum is not a binomial
check: neighboring dynamic bins share samples.

The existing scatter check, which this interval must not disturb, is in
`tests/integration/test_errors.py`: 400 realizations, Hann, 75% overlap,
`L = 512`, `fs = 200`, `N = 40000`, `freq = 20` Hz, `order = -1`. Measured
`Gxx` scatter matched `Gxx_dev` at about 1.00. See
`docs/reference/spectrum-estimation.md`.

**Direct DFT.** There is no oracle. Production agreement today is Numba
versus NumPy at `rtol=5e-3` on `Gxx` and about `1e-9` of `sqrt(Gxx * Gyy)`
on `Gxy` (`tests/integration/test_analysis_api.py`). That is a parity check,
not a metrology tolerance.

## Context

**Estimand this note is allowed to claim.** A one-sided WOSA spectrum of a
uniformly sampled, complete, real series. The plan is the logarithmic
scheduling heuristic already in `speckit/schedulers.py`. Each bin is a
quadratic form: segment, detrend projection, window, then
`sum x[n] exp(-i ω n)`, averaged, then scaled. Densities are one-sided.
Generator `psd` arguments in `speckit.noise` are two-sided, so
`Gxx = 2 * S2` on `(0, fs/2)` for a real signal.
`Gxy = X * conj(Y)`, channel 0 as `X` and channel 1 as `Y`.

**What the claim must exclude.** Leakage bias, a cross-frequency covariance,
a stationarity test, and a missing-data model. `nu` is Percival and Walden
(1993), eq. 292b, from the window autocorrelation and the actual starts. It
assumes a locally white Gaussian process, ignores the detrend, and drops the
DC/Nyquist pseudo-covariance. `M2` is per-segment scatter. `*_emp_*` divides
by the same `nu / 2`, so it is not an assumption-free error of the mean.

**Modules.** `speckit/analysis.py` owns the public result, the missing-sample
policy, and the interval. `speckit/schedulers.py` owns the plans.
`speckit/core.py` owns the kernels and `_overlap_dof`. The oracle lives under
`tests/` and must not call the Goertzel kernel, `_build_Q`, or a scheduler.
It may reimplement the detrend projection from the documented definition of
`order` (`-1` window only, `0` mean removal, `1`–`2` polynomial). Numba and
NumPy stay on the same statistic. `__init__.py` pins Numba to the workqueue
layer and BLAS threads to 1.

**Constructor docstring.** `SpectrumAnalyzer` documents the scheduler default
as `"ltf"`. The signature default is `"vectorized_ltf"`. The contract and
the docstring follow the signature.

## Settled decisions

- LTPDA output is not the oracle. The direct DFT is.
- `missing="raise"` is the default. `missing="zero"` is the named 1.0.3
  behavior: replace non-finite samples with 0 and record how many were
  replaced. No interpolation and no validity-mask API in this note.
- Finite inputs must keep non-finite kernel outputs. Remove the output
  `nan_to_num` only after the seeded 1.0.3 lock and the existing integration
  tests stay finite. A finite series that already produces a non-finite bin
  is a bug to report, not a value to zero.
- Keep unrounded `nu` in the chi-square interval. The interval is pointwise.
  It is not a simultaneous band, and it is not claimed at DC, at Nyquist, or
  for detrended or colored series until a separate Monte Carlo says so.
- Coherence intervals are a later change. Do not reuse the PSD multiplier
  for `coh`.
- `ltf_plan` and `vectorized_ltf_plan` may differ. Document the difference.
  Do not retune either scheduler so a plot matches.
- The summary path and `Gxx`, `Gxy`, `navg`, and `M2` stay numerically the
  same on finite inputs. Existing spectrum tests must stay green.
- If the direct DFT disagrees with production, stop. Write the repro, the
  expected formula, and the files into this note. Do not apply a scale
  factor, widen a tolerance, or special-case one `order`, frequency, or
  window.
- If coverage fails inside the labeled class, narrow the sentence in the
  contract or fix the estimator. Do not widen `pytest.approx`.
- Optional integrations and new public estimators wait until this note's
  success criteria hold.

## Success criteria

- `docs/reference/spectral-contract.md` names the estimand, the one-sided
  folding, the CSD orientation, the detrend orders, the missing-sample
  policy, the `nu` assumptions, and the quantities that are not claimed.
  `docs/reference/index.md` links it.
- A seeded test records `Gxx`, `Gxy`, `nu`, and the plan arrays for the
  default `vectorized_ltf` plan, `order` in `{-1, 0, 1, 2}`, one auto series
  and one delayed cross-spectrum, before any estimator edit. State `N`,
  `fs`, and the seed in the test.
- `vectorized_ltf_plan` is in the scheduler unit tests: required keys,
  `r * L = fs` within `rtol=1e-6`, positive strictly increasing `f` at most
  Nyquist, and `L` a positive integer no larger than `N`. The walker claim
  `f[j+1] - f[j] = r[j]` is tested on its own, tighter than the mean relative
  error of `0.01` used for the iterative plans. A second test reports where
  `f`, `L`, and `D` differ from `ltf_plan` on the fixture above, at the
  `bmin` boundary, and on a `K = 1` bin.
  `docs/reference/frequency-scheduling.md` states those differences.
- For every `order`, one window, an integer bin and a fractional bin, a short
  segment and a long one, and a frequency near zero plus one in the interior:
  each production Fourier coefficient matches the direct DFT on both the
  Numba and NumPy paths, one channel and two. The tolerance is a stated
  roundoff scaling, written next to the assertion.
- `missing="raise"` rejects a non-finite sample and the message includes the
  count. `missing="zero"` matches the 1.0.3 substitution and records the
  count. An unknown policy raises.
- `SpectrumResult` exposes the chi-square bounds on `Gxx`. On white Gaussian
  noise, Hann or boxcar, `order=-1`, one interior bin, `navg` large enough
  that the locally white model applies, the fraction of seeded realizations
  whose true one-sided level lies inside the 95% interval is consistent with
  95% at the binomial uncertainty of that trial count. For unit-variance
  `standard_normal` the true one-sided level is `2 / fs`. State `N`, `fs`,
  `L`, the seed, and the trial count.
- `python -m pytest tests/unit/test_schedulers.py tests/integration/test_errors.py tests/integration/test_analysis_api.py -q`
  passes, plus the new oracle, missing-sample, lock, and interval tests.
  `ruff check` and `ruff format --check` pass on the touched Python.
- Notebooks below run their new cells and describe the convention they plot.

## Recommended next steps

1. Write `docs/reference/spectral-contract.md` from
   `docs/reference/spectrum-estimation.md`, `bin-statistics.md`, and
   `frequency-scheduling.md`. Fix the scheduler-default docstring in
   `SpectrumAnalyzer` to `"vectorized_ltf"`.
2. Add the seeded 1.0.3 lock while the kernels and the default plan are
   still unchanged.
3. Extend `tests/unit/test_schedulers.py` with `vectorized_ltf_plan` and the
   parity measurement against `ltf_plan`. Update
   `docs/reference/frequency-scheduling.md` with what the measurement finds.
4. Add the direct-DFT oracle under `tests/` and the comparisons listed in
   the success criteria. Stop if a bin disagrees.
5. Add `missing` on `SpectrumAnalyzer` and `compute_spectrum`. Default
   `"raise"`. Implement `"zero"` as the recorded 1.0.3 path. Check that
   finite inputs still produce finite bins, then remove the output and
   property `nan_to_num` wipes. Note the break in `CHANGELOG.md`.
6. Add the chi-square interval on `Gxx` only, and the single-bin coverage
   test. Label the assumptions in the docstring, the contract, and
   `docs/reference/bin-statistics.md`.
7. Update the notebooks, after the API they call exists.
   - `notebooks/0.4_noise.ipynb` already states `Gxx = 2 S2`. Add the
     pointwise interval on the white series against that known one-sided
     level. Add a short cell: one non-finite sample raises under the
     default, and `missing="zero"` is the old substitution. Keep the
     existing ASD round trip and the `A / sqrt(2)` scaling.
   - `notebooks/0.5_errors.ipynb` plots the interval beside `Gxx_dev` and
     says the interval is pointwise, locally white, and not the scatter
     bar. Leave the existing empirical-versus-analytic comparison in place.
   - `notebooks/0.6_schedulers.ipynb` already draws `vectorized_ltf`. After
     the parity measurement, name in the prose where that plan leaves
     `ltf`, instead of only overlaying the curves.
   - `notebooks/1.3_vectorized-ltf.ipynb` times the two planners and
     currently declines to compare plans. Add the plan comparison the unit
     test records: `r L`, the frequency step, and the bins where `L` or `D`
     differ. Keep the timing figure.
   - `notebooks/1.2_polynomial-detrend.ipynb` shows each `order`. Add a cell
     that the production bin matches the direct DFT for that order on one
     interior frequency. If fractional bins, a near-zero frequency, and the
     long-segment case do not fit there, add
     `notebooks/1.5_direct-dft.ipynb` for those cases. Do not add that
     notebook if `1.2` can carry them without changing its question.
   - `notebooks/0.0_quickstart.ipynb` needs a sentence only if a reader would
     otherwise think non-finite samples are still dropped silently. Do not
     turn it into a second errors notebook.
8. Stop. If the oracle or the labeled coverage test fails, update this
   note and do not start `SpectralPlan`. If both pass, the next design note
   is the plan object in the blueprint's Phase 1, still without a public
   type.

## Files touched

None yet. Expected:

- `docs/reference/spectral-contract.md` (new)
- `docs/reference/index.md`, `spectrum-estimation.md`,
  `frequency-scheduling.md`, `bin-statistics.md`
- `docs/handoffs/20261024_overlap-degrees-of-freedom.md` (status only)
- `CHANGELOG.md`
- `speckit/analysis.py`
- `tests/unit/test_schedulers.py`
- new tests for the 1.0.3 lock, the direct DFT, the missing-sample policy,
  and the interval
- `notebooks/0.4_noise.ipynb`, `0.5_errors.ipynb`, `0.6_schedulers.ipynb`,
  `1.3_vectorized-ltf.ipynb`, `1.2_polynomial-detrend.ipynb`, and
  `0.0_quickstart.ipynb` as far as step 7 requires
- `notebooks/1.5_direct-dft.ipynb` only if `1.2` cannot hold the oracle cases

## Test gaps

- `vectorized_ltf_plan` has no unit test.
- No direct-DFT comparison exists.
- No test feeds a non-finite sample.
- No test checks chi-square coverage. The Monte Carlo in
  `tests/integration/test_errors.py` checks `Gxx_dev` and the Bendat
  formulas, not an interval.

## References

- `docs/handoffs/20261024_speckit-2.0-blueprint.md` — Phase 0
- `docs/handoffs/20261024_overlap-degrees-of-freedom.md` — `nu` done, interval open
- `docs/handoffs/20260905_noise-generator-normalization.md` — generator `psd` is two-sided
- `docs/handoffs/20261024_csd-convention-vs-scipy.md` — `Gxy = X * conj(Y)`
- `docs/reference/spectrum-estimation.md`, `bin-statistics.md`,
  `frequency-scheduling.md`
- Percival and Walden (1993), p. 292, eq. 292b
- Heinzel, Rüdiger, and Schilling (2002), window normalization and one-sided scaling
- `notebooks/0.4_noise.ipynb`, `0.5_errors.ipynb`, `0.6_schedulers.ipynb`,
  `1.2_polynomial-detrend.ipynb`, `1.3_vectorized-ltf.ipynb`

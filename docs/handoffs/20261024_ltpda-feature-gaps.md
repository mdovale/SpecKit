# Handoff: LTPDA features to add to SpecKit

Each item now has its own handoff. Work from that file. This note stays the
ranking and the cross-item order. `new_ltf_plan` is not one of these items
and has no handoff.

| Item | Handoff |
|------|---------|
| 1 PSD-weighted gap filling | `docs/handoffs/20261024_psd-weighted-gap-filling.md` |
| 2 Color and whiten a one-sided spectrum | `docs/handoffs/20261024_color-whiten-spectrum.md` |
| 3 Shared-plan N-channel matrix | `docs/handoffs/20261024_nchannel-cross-spectral-matrix.md` |
| 4 Overlap-corrected degrees of freedom | `docs/handoffs/20261024_overlap-degrees-of-freedom.md` |
| 5 Coherent line fit and subtraction | `docs/handoffs/20261024_coherent-line-subtraction.md` |
| 6 Delay estimate | `docs/handoffs/20261024_delay-estimate.md` |
| 7 Component noise projection | `docs/handoffs/20261024_component-noise-projection.md` |
| 8 Continuum and transfer-function fits | `docs/handoffs/20261024_continuum-tf-model-fits.md` |
| 9 Logarithmic rebinning | `docs/handoffs/20261024_logarithmic-rebinning.md` |
| 10 Units | `docs/handoffs/20261024_spectrum-units.md` |

## Scope

Ten analysis features present in the vendored LTPDA tree and absent from
SpecKit, ranked by usefulness to people who publish spectra of real
instruments. A later session should add them one at a time. This note is the
design record. It is not a patch.

Ranked order, from the comparison of `ltpda/` with `speckit/` on 24 Oct 2026:

1. PSD-weighted gap filling
2. Color and whiten an arbitrary one-sided spectrum
3. Shared-plan N-channel cross-spectral matrix
4. Overlap-corrected degrees of freedom and confidence intervals
5. Coherent line fit and subtraction
6. Delay estimate from the cross-spectrum
7. Component noise projection
8. Continuum and transfer-function model fits
9. Logarithmic rebinning of a uniform spectrum
10. Units that follow the spectrum

Out of scope, even though LTPDA has them: provenance (`@history`,
`@provenance`), bicoherence (`ao/bicohere`), LPF/DFACS/OMS/IS calibrations,
`simLPF`, `commonExcessNoise`, the repository, the GUI, plist/XML, a general
IIR/FIR designer, a general state-space simulator, and an MCMC sampler.
`ao/heterodyne` and `ao/spectrogram` wrap SciPy-shaped operations. The
spectrogram is a uniform-grid FFT. A log-frequency running spectrum is not in
LTPDA, so it is not part of this handoff.

## Goal

Each feature should be a public function or a `SpectrumResult` quantity in the
module that owns it, with the same statistic on the Numba and NumPy paths
when a kernel is involved, and with the PSD convention stated in the
docstring. Tests lock the scale. A plot helper must not multiply by 2,
`sqrt(2)`, or an ENBW to make a figure match.

## Current status

Not started. No SpecKit code implements these ten features. Each item has its
own handoff, listed at the top of this file. The ranking and the LTPDA entry
points below were read from the vendored tree. No numerical gold values have
been generated yet.

Item 4's original note said `psdconf.m` skips the `sqrt(ws2)` normalization.
A reread on 24 Oct 2026 shows both `psdconf.m` and `overlapCorr.m` divide by
`sqrt(ws2)`. The remaining disagreement is the segment length, not the
algebra. The item 4 handoff is the record to follow.

Unrelated dirty tree at the time of writing: modified `.gitignore`, untracked
`docs/paper/`. Leave both alone unless the user asks.

---

## Context

### Conventions the next session must keep

- `compute_spectrum` spectral densities are **one-sided**. Integrating an ASD
  over \([0, f_s/2]\) recovers variance, up to the window's ENBW.
- Noise-generator `psd` arguments are **two-sided**. See
  `docs/handoffs/20260905_noise-generator-normalization.md`. A generate /
  estimate round trip is off by 2 until that handoff is fixed. Item 2 depends
  on it.
- CSD follows SciPy: `Pxy = X * conj(Y)`. LTPDA is not consistent with this.
  `ao/delayEstimate.m` forms `conj(fft(x1)).*fft(x2)`, which is the conjugate
  of the SciPy CSD. Port the lag **sign** by test, not by copying that line.
- Detrend `order`: `-1` window only, `0` mean removal, `1`–`2` polynomial via
  `_build_Q`. A new kernel uses the same order switch.
- Numba and NumPy paths compute the same statistic. Do not disable Numba,
  swallow JIT errors, or skip the fallback.
- `python-control` is imported in `speckit/analysis.py` for `mag2db` only. It
  may be used for Bode data in item 7. Do not grow a second SSM class beside
  it.
- Window overlap defaults live in `speckit/flattop.py` `olap_dict` (Heinzel
  flat-tops, Kaiser, Nuttall, Hann). `win_dict` maps those flat-top names to
  the callables in the same module. Segment overlap for a named window already
  comes from `SpectrumAnalyzer._process_window_config`.

### What SpecKit already does

Do not reimplement these. Extend them.

| Already present | Where |
|-----------------|-------|
| LPSD and LTF plans, including vectorized LTF | `speckit/schedulers.py` |
| One- and two-channel spectra, TF, coherence, complex coherence, conditioned spectra `GyyCx`, `GyyRx`, `GyySx` | `speckit/analysis.py` |
| Analytic deviations `Gxx/sqrt(navg)` and Bendat formulas; empirical `M2` | `SpectrumResult.__getattr__` |
| SISO and MISO optimal subtraction | `speckit/systems.py` |
| Lagrange fractional shift of a **known** delay | `speckit/lagrange.py` `timeshift`, re-exported from `speckit/dsp` |
| CNR peak flags, no subtraction | `speckit/dsp.py` `peak_finder` |
| Band-limited RMS | `SpectrumResult.get_rms`, `dsp.integral_rms` |
| White, pink, red, 1/f^α generators; `fftnoise` on a full Hermitian FFT vector | `speckit/noise.py` |
| Resample onto a common grid | `speckit/dsp.py` |

`SpectrumAnalyzer` accepts a 1-D array or a `2×N` / `N×2` array
(`speckit/analysis.py`, the shape check near the `iscsd` branch). `N>2`
raises `ValueError`. The plan dict carries per-frequency `f`, `r`, `b`, `L`,
`K`, `navg`, `D`, `O`. `navg` is the segment count, not an overlap-corrected
degree of freedom.

### PSD convention for every new quantity

Say it in the docstring of the function that returns the quantity.

| Feature | Density convention |
|---------|--------------------|
| 1 Gap fill | Weight by the one-sided PSD the user would pass to `compute_spectrum` as truth, or by an estimate from `compute_spectrum`. Do not feed a two-sided generator `psd` in as if it were that estimate. |
| 2 Color / whiten | The public model argument is a **one-sided** PSD on an arbitrary frequency grid (Hz, power/Hz), because that is what `SpectrumResult.psd` is. Convert to the two-sided generator convention inside `noise.py` in one place. |
| 3 N-channel matrix | One-sided auto-spectra. Cross-spectra use `Pxy = X * conj(Y)`. |
| 4 Confidence intervals | Intervals on the one-sided PSD (and, separately, on coherence). Do not apply the PSD χ² factor to the ASD without transforming it. |
| 5 Line subtraction | Operate in the time domain, then re-estimate. Do not multiply the PSD by a notch gain. |
| 6 Delay | Seconds. The lag sign follows SpecKit's CSD, not LTPDA's `Gx12`. |
| 7 Noise projection | One-sided in, one-sided out. `S_out(f) = |H(f)|² S_in(f)` with both densities one-sided. |
| 8 Continuum fit | Fit the one-sided spectrum the user passes. Document whether `orders` is the polynomial degree in `log10(f)` and `log10(S)`. |
| 9 Rebin | Preserve the input convention. Rebinning does not convert one-sided to two-sided. |
| 10 Units | A unit string rides along. It does not change the numeric convention. |

---

## Settled decisions

- Add the ten features above. Do not widen the handoff into LTPDA's container,
  history, or mission packages.
- One feature per change, in the owning module listed under that item. A
  shared helper is allowed when two items compute the same quantity. A second
  copy of a scheduler or Goertzel kernel is not.
- Hot loops that grow the bin kernels (item 3) get a Numba kernel and a
  `_np` fallback that return the same `BinStats` fields. Items that are fits,
  rebinned means, or closed-form intervals may be NumPy-only. Say so in the
  docstring.
- Do not add `pint` or `astropy.units` as a required dependency for item 10
  until the user asks. Start with explicit unit strings on the result.
- Item 2 does not ship until the noise-generator handoff's factor-of-2
  convention is either fixed or the new function converts in one documented
  place and tests the round trip. Widening `pytest.approx` to hide the factor
  2 is a failed fix.
- LTPDA is the behavioral reference, not the code style. Match a documented
  formula and a numerical case. When two LTPDA functions disagree, follow the
  citation and record which LTPDA path was wrong.

---

## Item 1 — PSD-weighted gap filling

**LTPDA.** `ltpda/classes/@ao/gapfillingoptim.m` minimizes a chi-square
weighted by the expected PSD. Short gaps use `ao/optSubtraction`. Long gaps
use `optSubstitution`. The header also returns the PSD used as the weight and
the windows used to estimate it. `ltpda/classes/@ao/gapfilling.m` is the
simpler path: linear or spline interpolation between the segment before the
gap and the segment after it.

**Owner.** `speckit/dsp.py`. The spectral estimator stays gap-free. The fill
returns a contiguous series that `compute_spectrum` can take.

**Goal.** Given a series, a sample rate, and a gap mask (or NaNs), replace
the missing samples. The optimized fill must use a one-sided PSD weight. The
linear/spline fill is a separate, named function so callers can see which one
they used.

**Success.** Build a long colored series from a known one-sided PSD (after
item 2 exists, use that; until then use `white_noise` with its **code**
normalization, `rms = sqrt(psd * fs)`, and state that `psd` is two-sided).
Punch a gap of a few percent of the length. Fill it.

- Linear/spline fill: the filled samples match `numpy.interp` or a cubic
  spline on the gap edges, exact to interpolation tolerance.
- Optimized fill: the one-sided PSD of the filled series, estimated with
  `compute_spectrum`, stays within the analytic `Gxx_dev` of the ungapped
  series across the band the weight was defined on. A linear fill of a steep
  spectrum is the negative control: it must **not** be asserted to pass that
  test. If the optimized fill fails it, fix the weight or the objective. Do
  not loosen the tolerance until the residual is consistent with `navg`.

**Stop.** `gapfillingoptim.m` is a large MATLAB routine. Read it and write the
objective in the docstring before coding. Do not invent a spectral
extrapolation and call it the LTPDA method.

---

## Item 2 — Color and whiten an arbitrary one-sided spectrum

**LTPDA.** `ltpda/classes/@ao/noisegen1D.m` fits z-domain partial fractions to
a one-sided PSD model (`utils.math.psd2tf`), builds a parallel bank of MIIR
filters, and colors a white series. `ltpda/classes/@ao/whiten1D.m` and
`buildWhitener1D.m` call `ltpda/classes/+utils/@math/psd2wf.m`, which fits the
square root of the one-sided PSD, stabilizes with an all-pass, and inverts.
`whiten2D.m` whitens a 2×2 cross-spectral matrix from its eigendecomposition.
The `psd2wf.m` header states that the input model is the one-sided PSD.

**Owner.** `speckit/noise.py` for synthesis from a spectrum and for the
partial-fraction fit. Applying the filter to an existing series can live in
`speckit/dsp.py` and must call the same coefficients. Do not fit in both
places.

**Blocked by.** `docs/handoffs/20260905_noise-generator-normalization.md`.
`white_noise` code uses two-sided `rms = sqrt(psd * fs)`. Its docstring says
`sqrt(psd * fs / 2)`. `alpha_noise` misses its documented level by up to
about 13% in an alpha-dependent way. `fftnoise` expects a full FFT-length
Hermitian spectrum, not a log-grid one-sided ASD.

**Goal.** Public functions whose model argument is a one-sided PSD
`(f, S1(f))` in Hz and power/Hz, on a grid that need not be uniform and need
not include DC or Nyquist. Coloring a unit-variance white series (in the
**code** convention of `white_noise`) yields a series whose `compute_spectrum`
PSD matches `S1`. Whitening a series that has spectrum `S1` yields a series
whose one-sided PSD is flat at 1 (or at 2 if the implementation documents a
two-sided unit white residual — pick one, test it, write it down).

**Success.**

- Analytic case: `S1(f) = constant`. The colored series matches `white_noise`
  at the equivalent level, to sampling error, and `compute_spectrum` recovers
  `S1` inside `Gxx_dev`.
- Steep case: a one-sided `1/f²` model on a log grid. The recovered PSD slope
  and level match the model. This is the case `alpha_noise` already gets
  wrong by up to ~13%. The new fitter must not inherit that bias. If it
  cannot beat it, stop and extend the noise handoff instead of shipping both.
- Round trip: color then whiten returns a white residual at the documented
  level. A factor 2 here is a convention bug, not a tolerance change.
- Two-channel whitening waits until item 3 can form the 2×2 matrix. Do not
  fake `whiten2D` with two independent `whiten1D` calls.

**Stop.** If the partial-fraction order is free, stop when the fit residual
is inside the PSD error bars, and record the order rule. Do not hard-code an
order that only works for one alpha.

---

## Item 3 — Shared-plan N-channel cross-spectral matrix

**LTPDA.** `ao/cpsd`, `ao/lcpsd`, `ao/lcohere`, `ao/ltfe` run on arrays of
analysis objects, so every pair shares a frequency plan.

**Owner.** `speckit/analysis.py` for the API and `speckit/core.py` for the
kernels. The scheduler stays in `speckit/schedulers.py` and is called **once**
per matrix, not once per pair.

**Goal.** Accept `(n_chan, n_time)` with `n_chan >= 1`. Keep the existing 1-D
and `2×N` / `N×2` calls working with the same numbers as today. For `n_chan >
2`, produce the one-sided auto-spectra and the cross-spectra `P_{ij} = X_i *
conj(X_j)` on one plan (`f`, `L`, `navg` identical for every pair).

**Success.**

- `n_chan == 1` and `n_chan == 2` match current `compute_spectrum` on a fixed
  seed, to numerical noise of the existing kernels (tight tolerance, not
  `atol=1e-2`).
- For a 3-channel series, `G_ij` computed by the matrix call matches three
  separate 2-channel `compute_spectrum` calls on the **same** plan arrays.
  Calling the scheduler separately per pair and hoping the grids coincide is
  not the test.
- At each frequency the Hermitian matrix is positive semidefinite up to
  roundoff (eigenvalues ≥ a small negative floor set by the kernel's dtype,
  not by a loosened spectral tolerance).
- Numba and `_np` agree on `MXX`, cross means, and `M2` for one auto bin and
  one cross bin.

**Stop.** Do not change `BinStats` field meanings for the 2-channel path.
Extend the reduction so the old fields are still the `(0,0)`, `(1,1)`, and
`(0,1)` entries. Conditioned spectra and `systems.py` may keep taking pairs
until a follow-up wires them to the matrix.

---

## Item 4 — Overlap-corrected degrees of freedom and confidence intervals

**LTPDA.** Two implementations, and they are not the same expression.

- `ltpda/classes/+utils/@math/overlapCorr.m` cites Percival and Walden,
  *Spectral Analysis for Physical Applications* (Cambridge, 1993), p. 292.
  The window is normalized by `sqrt(ws2)`. The returned `R` is
  `2 * sum_{j=1}^{navs-1} (1 - j/navs) |w_0 · w_j|² / navs`.
- `ltpda/classes/@ao/getdof.m` sets `dof = 2 * navs / (R * navs + 1)` with
  that `R`, and rounds. For `navs == 1`, `dof = 2`.
- `ltpda/classes/@ao/psdconf.m` also divides the window by `sqrt(ws2)`, then
  sets `dof = 2 * navs / (1 + 2 * sm)` and rounds. Confidence level default is
  95%. With `alpha = 1 - conf`,
  `c = dof / chi2inv([1 - alpha/2, alpha/2], dof)` and the lower and upper
  levels are `S * c[0]` and `S * c[1]`. Reported variance is `S² * 2 / dof`.
  An earlier draft of this note said `psdconf` skipped that normalization.
  It does not.

A reread on 24 Oct 2026: `psdconf.m` does divide the window by `sqrt(ws2)`
before the `sm` loop (PSD branch near the `win ./ sqrt(w.ws2)` line, LPSD
branch at `owin = w.win ./ sqrt(w.ws2)`). With that normalization,
`dof = 2*navs/(R*navs+1)` and `dof = 2*navs/(1+2*sm)` are the same expression.
What still differs is the segment length. `getdof`'s LPSD path asks
`overlapCorr` to invent one `L` for the whole record. `psdconf`'s LPSD path
uses the bin length and `floor(shift)`. SpecKit must use the plan's `L` and
the actual hop. Details and the Hann overlap Monte Carlo are in
`docs/handoffs/20261024_overlap-degrees-of-freedom.md`. Do not copy either
MATLAB path unchanged.

**Owner.** `speckit/analysis.py`, on `SpectrumResult`. The inputs are already
on the result: per-bin `L`, `navg`, the window, and the overlap.

**What SpecKit does now.** `Gxx_dev = Gxx / sqrt(navg)` and the Bendat
formulas use the segment count. That is the large-`K`, independent-segment
limit, where `dof = 2 * navg` and `std(S) = S * sqrt(2/dof) = S/sqrt(navg)`.
Empirical `M2` stays. It is the nonparametric scatter. The new interval does
not replace it.

**Goal.** Per-bin `dof` from the overlap correlation of the window actually
used, and a confidence interval on the one-sided PSD. Coherence needs its own
interval. Do not reuse the PSD χ² multiplier for `coh`.

**Formula to implement, after checking Percival and Walden p. 292.** For a
one-sided PSD estimate of a Gaussian series, with `nu` degrees of freedom,

```
S_low  = nu * S_hat / chi2.ppf(1 - alpha/2, nu)
S_high = nu * S_hat / chi2.ppf(alpha/2, nu)
var(S_hat) = 2 * S_hat² / nu
```

`nu = 2 * navg` when segments do not overlap (`R = 0`). Overlap reduces `nu`
below `2 * navg`. Round only if matching LTPDA's published integers is an
explicit test. The unrounded `nu` is the quantity the interval should use,
unless a test against `getdof` requires the round. Document the choice.

**Success.**

- No overlap, any window: `nu == 2 * navg`, and `S_hat / sqrt(navg)` equals
  `S_hat * sqrt(2/nu)`. This must reproduce today's `Gxx_dev`.
- Hann, 50% overlap, large `navg`: `nu / (2 * navg)` matches the Percival and
  Walden / Welch overlap factor for that window, to a tolerance set by the
  reference value, not by eye.
- A white Gaussian series, seed fixed, `N` large enough that the lowest bin
  used has `navg >= 30`: the fraction of bins whose true one-sided PSD lies
  inside the 95% interval is consistent with 95% (binomial uncertainty). Bins
  with small `navg` are the interesting case. State `N`, `fs`, `Jdes`,
  `Kdes`, window, and overlap in the test.
- Numba vs NumPy does not apply unless the overlap sum is added to a kernel.
  Prefer a NumPy helper. The PSD itself is unchanged, so existing spectrum
  tests must stay green.

**Stop.** If `getdof.m` and `psdconf.m` disagree on a Hann, 50% case by more
than rounding, report both numbers and follow Percival and Walden. Do not
average them.

---

## Item 5 — Coherent line fit and subtraction

**LTPDA.** `ao/linedetect.m` finds lines in an `fsdata` spectrum.
`ao/fit_sine_poly.m` fits `x(t)` to a polynomial plus sines.
`ao/spsdSubtraction.m` is an sPSD-weighted iterative least-squares
subtraction.

**Owner.** `speckit/dsp.py`, next to `peak_finder`.

**Goal.** From frequencies returned by `peak_finder` (or from user-supplied
tones), fit amplitude, phase, and a polynomial of the same `order` meaning as
the spectral detrend, subtract in the time domain, and return the residual.
Optional second step: re-estimate with `compute_spectrum` and check the line
is gone.

**Success.** A unit-amplitude sine at a non-bin frequency, plus white noise,
seed fixed. After subtraction the residual RMS at that tone, read from
`compute_spectrum` in a band of a few bins, drops by the factor the fit's
parameter covariance predicts. The continuum level away from the tone stays
inside `Gxx_dev` of the noise-only spectrum. A test with two close tones must
fail honestly if the fitter cannot separate them. Do not merge tones to force
a pass.

**Stop.** `spsdSubtraction.m` is a different algorithm from a single sine fit.
Implement the sine-plus-polynomial fit first. Port the iterative weighted
subtraction only if the single fit leaves a biased continuum on a steep
spectrum, and say which routine was ported.

---

## Item 6 — Delay estimate from the cross-spectrum

**LTPDA.** `ltpda/classes/@ao/delayEstimate.m`. Weights named in the file:
`roth` (`G11`), `scot` (`sqrt(G11*G22)`), `phat` (`|G12|`), and `scc`,
`eckart`, `ML` (all coded as weight 1). The method zero-pads, forms a
two-sided FFT cross-spectrum, inverse-transforms `G12/weight`, and takes the
lag of the maximum. The triangular correction `(N./(N-r))` compensates the
zero-pad. **Its `G12 = conj(fft(x1))*fft(x2)` is not SpecKit's CSD.**

**Owner.** `speckit/dsp.py`. Apply the lag with `timeshift` from `speckit/lagrange.py` (re-exported from `dsp`).

**Goal.** Estimate `tau` in seconds such that shifting series 2 by `-tau`
with `timeshift` aligns it with series 1. Document the sign in the docstring
with one sentence and a test.

**Success.** Series 2 is series 1 delayed by a known `tau` that is not an
integer sample, using `timeshift`, plus a little noise. The estimate recovers
`tau` to a tolerance smaller than one sample. Repeat with the opposite sign.
Integer-sample delays must be exact to interpolation tolerance. A pure delay
in noise with no coherence must return a documented failure, not a silent lag
of 0.

**Stop.** Do not estimate the delay from `angle(CSD)/(-2πf)` alone and call it
`delayEstimate`. That estimator is biased by the wrap and by the LTF
`bmin` cutoff. A phase-slope fit may be a second, named function. The LTPDA
method is the correlation peak.

---

## Item 7 — Component noise projection

**LTPDA.** `ltpda/classes/@ssm/projectNoise.m`. The frequency-domain branch
evaluates the state-space Bode response from each noise input to each output
and stores `sqrt(sum |H|²)`. The header says this branch does **not** keep
phase between outputs. The time-domain branch simulates. Commented code in
the frequency branch shows an abandoned one-sided white-noise normalization
(`sqrt(bodeOut*2./(fs*ncols))`). Do not revive that comment. It is not the
running path.

**Owner.** `speckit/systems.py`, which already owns SISO and MISO optimal
subtraction.

**Goal.** A function that takes one-sided input PSDs and complex (or
magnitude) transfer functions on the same frequency grid and returns the
one-sided output contribution of each input, plus the incoherent sum. Phase
between outputs is out of scope for this function. A time-domain projection
that keeps phase is a later function and must say so.

**Success.** Analytic: one input, `H = 1`, `S_out == S_in`. One input,
`H = 1/(1 + i f/f0)`, `S_in` white and one-sided, `S_out(f) = |H(f)|² S_in`.
Two incoherent inputs sum in power, not in amplitude. A coherent pair is
**not** accepted by this function. Send the user to the cross-spectrum (item
3) rather than adding the cross term under a flag that defaults to off.

**Stop.** Do not port `@ssm`. If a state-space model is required, use
`python-control` to obtain `H(f)`, then call this function. Ask before adding
a new model object.

---

## Item 8 — Continuum and transfer-function model fits

**LTPDA.** `ao/polyfitSpectrum.m` takes `log10` of **both** axes
(`plist('axis','xy')`), optionally adds `regularizecoeff`, and calls
`polynomfit` with user `orders`. The model is evaluated at `log10(fout)`.
`ao/zDomainFit.m` fits a z-domain partial-fraction expansion
`sum z*r_k/(z-p_k)` via `utils.math.vdfit`, choosing order by a stop test.
`@pzmodel` stores poles, zeros, gain, and delay, and can convert to IIR/FIR.

**Owner.** Log-polynomial continuum: a function that takes `f` and one-sided
`S` and returns coefficients and the model evaluated on `f` or on a new grid.
Put it in `speckit/dsp.py` unless it needs `SpectrumResult` internals, in
which case a method on `SpectrumResult` may call it. Z-domain plant fit:
`speckit/systems.py`. Do not create `speckit/models.py` unless both fits
share a residue representation. Item 2's partial-fraction fitter is the only
shared residue code. Call it. Do not fit poles in three modules.

**Goal.**

- Continuum: `log10(S) ≈ sum_{k=0}^{n} a_k (log10(f/f_ref))^k`, with `f_ref`
  documented (LTPDA fits `log10(f)` directly, so `f_ref = 1` Hz). Return `a_k`
  and the one-sided model `S_hat(f)`.
- Plant: fit a complex frequency response `H(f)` from a `SpectrumResult`
  transfer function. This can wait until the continuum fit and item 2's
  residue helper exist.

**Success.** Synthetic one-sided `S(f) = A * (f/f_ref)^b` over a band that
excludes non-positive bins. The recovered slope `b` and amplitude `A` match
to a tolerance set by float64 on noise-free data, and to the PSD error bars
on a `compute_spectrum` estimate of a series with that spectrum. A fit that
is accurate only for `b = -2` is the `alpha_noise` failure mode. Test at
least `b = 0` and `b = -1`.

**Stop.** `zDomainFit.m` is hundreds of lines and depends on `vdfit`. Do not
start it in the same change as the log-polynomial fit. If the residue helper
from item 2 is not stable, leave the plant fit unstarted and say so in this
section's status when you update it.

---

## Item 9 — Logarithmic rebinning of a uniform spectrum

**LTPDA.** `ltpda/classes/@ao/bin_data.m`. Rebins onto logarithmic, linear, or
user edges by the **mean** of the samples that fall in each bin. Inherited
from D. Nicolodi (UniTN). It does not weight by bin width or by degrees of
freedom.

**Owner.** `speckit/dsp.py` or a small function used by `SpectrumResult`. One
implementation.

**Goal.** Rebin a uniform-grid one-sided PSD (SciPy `welch` output, or a
linear FFT) onto log edges. Default reduction is the mean, matching
`bin_data`. If a variance-weighted mean is added, it is a separate `how=`
value and is not the default.

**Success.** A constant spectrum rebins to that constant. A pure power law
sampled on a fine linear grid, rebinned by the mean, matches the mean of the
samples in each edge, computed independently with NumPy in the test. Edges
that contain no samples are dropped or returned as NaN, documented, and
tested. The function does not multiply by 2 to convert a two-sided density.

**Stop.** SpecKit's own LPSD output is already on a log grid. Do not run
`bin_data` on it as a post-process inside `compute_spectrum`.

---

## Item 10 — Units that follow the spectrum

**LTPDA.** `@unit` propagates through arithmetic, and analysis objects store
`xunits` and `yunits`. `projectNoise` sets output units to
`output_units * Hz^{-1/2}` for an ASD.

**Owner.** `SpectrumResult` in `speckit/analysis.py` for spectral results.
Generators in `speckit/noise.py` gain the matching strings when item 2 lands.
Do not build a unit algebra.

**Goal.** The result carries the input unit string and the derived strings for
PSD, ASD, CSD, and coherence (coherence is dimensionless). `get_rms`
documents the unit of the integral. Numeric values do not change.

**Success.** A series passed with `y_unit="m"` yields PSD unit `m²/Hz` and
ASD unit `m/√Hz` as strings on the result. Omitting the unit leaves the
strings empty and the numbers unchanged. Existing tests that compare arrays
keep passing. This item does not fix the factor-of-2 generator bug. That
stays in the noise handoff. Unit strings that disagree with the numeric
convention are a bug.

**Stop.** Ask before adding `pint` or `astropy` as a dependency.

---

## Success criteria

The work is done when all ten items have the tests in their sections, or this
note's **Current status** says which items landed and which are still open.
For the whole set:

- `python -m pytest tests/unit -q` and `python -m pytest tests/integration -q`
  pass, including the new tests.
- `ruff check speckit tests` and `ruff format --check speckit tests` pass on
  the touched files.
- One-sided `compute_spectrum` values for 1-D and 2-channel inputs are
  unchanged on the existing integration tests.
- No new scale factor at a plot call site. No widened tolerance that absorbs
  a factor 2 or an overlap bias.
- Each new public function names its PSD convention in the docstring.

There is no single `(seed, N, fs)` for the whole handoff. Each item's test
must record the seed, `N`, `fs`, and the expected value or the analytic
expression. Do not invent those numbers in the implementation. Compute them
from the analytic case or from a recorded LTPDA run, and paste the command in
the test docstring.

## Recommended next steps

Work in this order. The rank order is usefulness. This order is what unblocks
the next item.

1. **Item 4** (degrees of freedom). Smallest change, no new kernel, and it
   tells items 1 and 2 what "inside the error bar" means. Read Percival and
   Walden p. 292 before choosing between `getdof.m` and `psdconf.m`.
2. **Item 2**, after reading the noise-generator handoff. Fix or explicitly
   convert the two-sided generator convention in the new function. Land the
   one-channel color and whiten path. Leave `whiten2D` until step 4.
3. **Item 8** continuum fit only, using spectra from step 2 as the synthetic
   truth.
4. **Item 3** N-channel matrix. Then two-channel whitening from item 2.
5. **Item 6** delay, with a sign test against `timeshift`.
6. **Item 5** line subtraction, reusing `peak_finder`.
7. **Item 7** incoherent projection in `systems.py`.
8. **Item 1** optimized gap fill. The linear/spline fill can land earlier. It
   has no PSD dependence.
9. **Item 9** rebin. Independent. Fit it in whenever a Welch comparison test
   needs it.
10. **Item 10** unit strings, after the numeric conventions above are stable.

Update **Current status** at the top of this file when an item lands. Do not
delete the item section. Mark it done and point at the function and the test.

## Files touched

None yet. Expected owners when work starts:

| Item | Files |
|------|--------|
| 1 | `speckit/dsp.py`, `tests/unit/` |
| 2 | `speckit/noise.py`, `speckit/dsp.py`, `tests/unit/test_noise.py` |
| 3 | `speckit/analysis.py`, `speckit/core.py`, `tests/unit/`, `tests/integration/` |
| 4 | `speckit/analysis.py`, `tests/unit/` or `tests/integration/` |
| 5, 6, 9 | `speckit/dsp.py`, `tests/unit/` |
| 7, 8 plant | `speckit/systems.py`, `tests/unit/` |
| 8 continuum | `speckit/dsp.py` |
| 10 | `speckit/analysis.py`, `speckit/noise.py` |

LTPDA sources to read first, and not to edit:

- `ltpda/classes/@ao/gapfillingoptim.m`, `gapfilling.m`
- `ltpda/classes/@ao/noisegen1D.m`, `whiten1D.m`, `whiten2D.m`, `buildWhitener1D.m`
- `ltpda/classes/+utils/@math/psd2wf.m`
- `ltpda/classes/@ao/cpsd.m`, `lcpsd.m`, `lcohere.m`, `ltfe.m`
- `ltpda/classes/@ao/getdof.m`, `psdconf.m`
- `ltpda/classes/+utils/@math/overlapCorr.m`
- `ltpda/classes/@ao/linedetect.m`, `fit_sine_poly.m`, `spsdSubtraction.m`
- `ltpda/classes/@ao/delayEstimate.m`
- `ltpda/classes/@ssm/projectNoise.m`
- `ltpda/classes/@ao/polyfitSpectrum.m`, `zDomainFit.m`
- `ltpda/classes/@ao/bin_data.m`

## References

- This comparison: conversation that ranked the ten features, 24 Oct 2026.
  Canvas notes are not in the repo. The rank and the entry points are copied
  into this file.
- `docs/handoffs/20260905_noise-generator-normalization.md` — two-sided
  generator `psd`, one-sided `compute_spectrum`, `white_noise` docstring
  factor `sqrt(2)`, `alpha_noise` level error.
- `README.md` — error analysis, Welford `M2`, one-sided densities.
- `docs/paper/main.tex` — LPSD, LTF, `bmin`, delay bias in transfer functions,
  and the statement that SpecKit stays a spectral library rather than a port
  of LTPDA. These ten features are additive. They are not a port of `@ao`.
- Bendat and Piersol, *Random Data* — analytic variances already implemented
  as `Gxx_dev`, `coh_dev`, and `Hxy_dev`.
- Percival and Walden, *Spectral Analysis for Physical Applications* (1993),
  p. 292 — overlap correlation used by `overlapCorr.m`.
- Tröbs and Heinzel, *Measurement* 39 (2006) — log-frequency LPSD. Item 3 must
  keep a single LPSD/LTF plan across channels.
- Heinzel, Rüdiger, and Schilling (2002), AEI technical note — windows and
  ENBW. Overlap defaults in `speckit/flattop.py`.
- SciPy `signal.csd` — `Pxy = X * conj(Y)`.

# Handoff: delay estimate from the cross-spectrum

## Scope

Item 6 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Estimate a time delay from
two series. The lag sign follows SpecKit's CSD, not LTPDA's `Gx12`. This
note does not cover `new_ltf_plan`.

## Goal

A function in `speckit/dsp.py` that returns `tau` in seconds such that
shifting series 2 by `-tau` with `timeshift` aligns it with series 1.
`timeshift` is defined in `speckit/lagrange.py` and re-exported from `dsp`.
The docstring states that sign in one sentence, and a test locks it.

## Current status

Not started. `timeshift` in `speckit/lagrange.py` applies a **known**
fractional delay with Lagrange taps. Nothing estimates the delay.
`frequency2phase` converts a frequency series to phase. It is not a delay
estimator.

## Context

**LTPDA.** `ltpda/classes/@ao/delayEstimate.m` zero-pads, forms a two-sided
FFT cross-spectrum, inverse-transforms `G12 / weight`, and takes the lag of
the maximum. The triangular factor `(N./(N-r))` compensates the zero-pad.

Weights named in the file:

| Name | Weight actually used |
|------|----------------------|
| `roth` | `G11` |
| `scot` | `sqrt(G11 * G22)` |
| `phat` | `abs(G12)` |
| `scc`, `eckart`, `ML` | `1` |

The cross-spectrum in that file is

```
Gx12 = conj(fft(x1)) .* fft(x2)
```

which is `X2 * conj(X1)`. SpecKit and SciPy use `Pxy = X * conj(Y)`, i.e.
`X1 * conj(X2)` when series 1 is `x` and series 2 is `y`. Those are
conjugates, so the correlation peak moves to the opposite lag if the line is
copied. Port the lag **sign** by the `timeshift` test below, not by copying
`Gx12`.

**Owner.** `speckit/dsp.py`.

**What this is not.** A fit of `angle(CSD) / (-2 π f)` is a different
estimator. It is biased by phase wraps and by the LTF `bmin` cutoff, which
drops the lowest bins. It may exist later as a second, named function. The
LTPDA method is the correlation peak.

**NumPy phase.** The Numba Goertzel kernels use `e^{-j ω l}`. The NumPy
fallbacks use `e^{+j ω l}`. This estimator should be built on `numpy.fft`
(or on one documented sign) and tested against `timeshift`, not against
`SpectrumResult.Gxy`, until the fallback sign matches the Goertzel kernel.
See `docs/handoffs/20261024_nchannel-cross-spectral-matrix.md`.

## Settled decisions

- Seconds, not samples, in the public return. The sample lag may be a
  secondary return.
- Default weight is one of the named LTPDA weights. Document which, and test
  at least `scc` (weight 1) and `phat`.
- A pure delay in noise with no coherence returns a documented failure, not a
  silent lag of 0.
- Integer-sample delays are exact to interpolation tolerance. Fractional
  delays are within a fraction of one sample.

## Success criteria

Series 2 is series 1 delayed by a known `tau` that is not an integer number
of samples, using `timeshift`, plus a little noise. Record seed, `N`, `fs`,
and `tau`.

- The estimate recovers `tau` to better than one sample.
- The opposite sign (`series 1` delayed, or `tau` negated) recovers the
  negated lag.
- An integer-sample delay matches to interpolation tolerance.
- Incoherent noise returns the documented failure.
- `ruff check` and `ruff format --check` on `speckit/dsp.py` and the new test.
  Existing `timeshift` tests stay green.

## Recommended next steps

1. Read `delayEstimate.m` through the `ifft` and the peak pick, including the
   triangular pad correction.
2. Implement the correlation peak with SpecKit's CSD sign.
3. Lock the sign with the `timeshift` test before adding the other weights.

This item is step 5 of the cross-item order. It does not wait on the
N-channel matrix. It uses two series.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md`
- `docs/handoffs/20261024_nchannel-cross-spectral-matrix.md` — Fourier sign
- `speckit/lagrange.py` — `timeshift`, `lagrange_taps` (re-exported from `speckit/dsp.py`)
- `tests/unit/test_lagrange.py`
- `ltpda/classes/@ao/delayEstimate.m`
- `docs/paper/main.tex` — time-delay bias in transfer functions. That section
  is about `Lmin`, not this estimator.

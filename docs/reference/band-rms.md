# Band RMS

`integral_rms` is the square root of a trapezoidal integral of a one-sided
ASD squared. `SpectrumResult.get_rms` calls it on `result.asd`.

## Goal

Report a band-limited RMS from the ASD samples the estimator actually
returned.

## Scope

In: `integral_rms` and `crop_data` in `speckit/spectral.py`. `speckit/dsp.py`
re-exports both names so existing imports keep working.

Out: scaling the integral so that it equals `std(x)`. A log plan starts near
`bmin * fs / N` and stops short of Nyquist, so the integral is not the full
variance of a white series. No unit string is attached.

## Behavior and conventions

`fourier_freq` is in hertz and `asd` is a one-sided amplitude spectral
density, one sample per frequency. Lengths must match and must be non-empty.

`pass_band=(f_min, f_max)` is inclusive. The integrated interval is the
overlap of that band with the frequency samples present. If the overlap is
empty or a single point, the function raises `ValueError`. `get_rms` swaps a
reversed finite pair before calling. `integral_rms` itself rejects
`f_min > f_max`.

Cross-spectrum results raise `NotImplementedError` from `get_rms`.

## Algorithm

`crop_data` keeps samples with `xmin <= x <= xmax`. The RMS is

```
sqrt( cumulative_trapezoid(asd², f, initial=0)[-1] )
```

`initial=0` makes the first cumulative sample zero. The returned value is the
last sample, the integral from the first retained frequency through the last.

## Numerical checks

`tests/unit/test_dsp.py` locks `crop_data` bounds and `integral_rms` against
a trapezoidal integral, including the error cases.

## Key files

- `speckit/spectral.py` — `crop_data`, `integral_rms`
- `speckit/dsp.py` — re-export
- `speckit/analysis.py` — `SpectrumResult.get_rms`
- `tests/unit/test_dsp.py`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md)
- `docs/handoffs/20261024_spectrum-units.md` — the integral stays on the
  planned frequencies, and a future unit string for this RMS would be the
  input unit.

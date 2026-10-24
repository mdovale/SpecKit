# Noise generators

`speckit.noise` builds white, red, pink, and `1/f^α` series, plus an FFT
phase-randomization helper and a Butterworth low-pass. Generator `psd`
arguments are two-sided. `compute_spectrum` returns one-sided densities.

## Goal

Stream Gaussian white noise at a requested two-sided level, and shape it into
`1/f^α` with a stable IIR cascade.

## Scope

In: `white_noise`, `red_noise`, `pink_noise`, `alpha_noise`, `fftnoise`,
`band_limited_noise`, `butter_lowpass`, and `butter_lowpass_filter`.

Out: making a generator's two-sided level match a one-sided `SpectrumResult`
without a factor of 2. That comparison is an open discrepancy.

## Behavior and conventions

`white_noise(f_sample, psd=1, seed=None)` draws `Normal(0, rms)` with

```
rms = sqrt(psd * f_sample)
```

The class docstring instead writes `sqrt(psd * f_sample / 2)`. The
constructor and `tests/unit/test_noise.py::test_white_noise_statistics` use
the formula without the 2. For a constant two-sided density `S2`, the
variance over `[-fs/2, fs/2]` is `S2 * fs`, so the code matches that
integral. `get_series(n)` returns `n` samples. `get_sample` refills an
internal buffer of 4096 samples.

`red_noise` is a `1/f²` spectrum, flat below `f_min`, scaled so the class
docstring's two-sided PSD is 1 at 1 Hz. The driving white generator uses
`psd=1`. The filter is one `scipy.signal.lfilter` section:

```
a = [2 π f_min]
b = [1, -exp(-2 π f_min / fs)]
scale = 1 / (fs * f_min)
```

`alpha_noise` requires `alpha` in `[0.01, 2]` and `f_sample >= 2 * f_max`.
The slope is whitened below `f_min` and above `f_max`. The cascade has

```
n = ceil(4.5 * log10(ω_max / ω_min))
```

first-order sections on poles spaced in log frequency, implemented by
`_numba_lfilter_cascade` (direct form II transposed). The output scale is
`1 / f_max^(alpha / 2)`. The class docstring says the two-sided PSD is 1 at
1 Hz. The tests lock the log-log slope, not that level.

`pink_noise` is `alpha_noise` with `alpha=1`.

`init_filter=True` (the default) runs the cascade until the state settles
before the first requested sample. `seed` fixes the driving white generator.

`fftnoise` keeps the magnitudes of a length-`N` complex FFT, replaces phases
on bins `1 .. (N-1)//2` with uniform random phases, forces Hermitian symmetry,
leaves DC and a real Nyquist bin unchanged, and returns `ifft`.
`band_limited_noise` puts unit magnitude on `rfftfreq` bins inside
`[min_freq, max_freq]` and calls `fftnoise`. It does not take a `psd` level.

`butter_lowpass(cutoff, fs, order=5)` is `scipy.signal.butter` in SOS-free
`b, a` form. `butter_lowpass_filter` applies those coefficients with
`lfilter`.

## Numerical checks

`tests/unit/test_noise.py` locks white-noise reproducibility and
`std(x) ≈ sqrt(psd * fs)` within 1 percent relative on the large fixture.
`alpha_noise` slopes for `alpha` in `{0.5, 1, 1.5, 2}` must lie within
`0.15` of `-alpha` on a Welch log-log fit between `2 f_min` and `f_max / 2`.
`red_noise` must lie within `0.1` of slope `-2` between `2 f_min` and
`fs / 4`.

## Key files

- `speckit/noise.py`
- `tests/unit/test_noise.py`
- `notebooks/0.4_noise.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md) — one-sided estimator output
- `docs/handoffs/20260905_noise-generator-normalization.md` — the white-noise
  docstring, the `alpha_noise` absolute level, and the one-sided versus
  two-sided factor of 2. Not fixed. Slope tests do not cover the level.
- Plaszczynski, *Generating long streams of 1/f^α noise*,
  doi:10.1142/S0219477507003635, cited in the module docstring. The pole count
  `4.5` decades factor and the scale `1 / f_max^(α/2)` are what the code
  implements.

## Known limitations

Do not read `white_noise`'s docstring RMS formula. Use `sqrt(psd * fs)`.
Do not treat the `alpha_noise` sentence "PSD is 1 at 1 Hz" as a tested level.
A one-sided spectrum of these generators is a different convention by a
factor of 2 even where the two-sided level is right.

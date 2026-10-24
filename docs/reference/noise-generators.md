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
without a factor of 2. The two conventions are documented in the `noise`
module docstring and in `compute_spectrum`. `Gxx = 2 * S2` for a real signal.

## Behavior and conventions

`white_noise(f_sample, psd=1, seed=None)` draws `Normal(0, rms)` with

```
rms = sqrt(psd * f_sample)
```

For a constant two-sided density `S2`, the variance over `[-fs/2, fs/2]` is
`S2 * fs`, so the constructor, the class docstring, and
`tests/unit/test_noise.py::test_white_noise_statistics` agree on that
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
`_numba_lfilter_cascade` (direct form II transposed). Each section is the
bilinear transform of `(s + ω_zero) / (s + ω_pole)`. Poles sit at
`log10 ω_min + dp (i + 1/2 - α/4)` and zeros `dp α / 2` above them, with
`dp = log10(ω_max / ω_min) / n`.

The output scale is `1 / sqrt(C)`, where `C` is the log-average of
`|H(f)|² f^α` over one pole spacing `dp` centred on the band
(`alpha_noise._power_law_level`, evaluated on the digital `H`). The ripple
is periodic in `dp`, so this is the intercept of the best-fit power law, and
the two-sided PSD is `f^-α`, 1 at 1 Hz. From a decade above `f_min` to a
decade below `f_max`, and below about `fs / 40`, `|H|² / C` is within 1% of
`f^-α` for every `α` in `[0.01, 2]`. Closer to `f_min` the first-order
corner pulls the PSD down (about 9% at `sqrt(10) f_min` for `α = 2`). Near
Nyquist the bilinear transform compresses frequency, which lowers the PSD
below the law.

Before this, the scale was the closed form `1 / f_max^(α/2)`. The power-law
intercept of the staircase actually sits `(dp/2)(1 - α/2)` decades above the
last zero, so that scale was high by `10^(α dp (1 - α/2) / 2)` in PSD: about
13% at `α = 1`, 10% at `α = 0.5` and `1.5`, and exact only at `α = 2` (and
negligible as `α → 0`).

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

The absolute level is locked too. For `alpha_noise` at `α` in
`{0.01, 0.5, 1, 1.5, 2}`, `pink_noise`, and `red_noise` (`fs = 1000`,
`f_min = 0.1`, `f_max = 500`, `2^21` samples), the mean over 1 to 25 Hz of
the two-sided Welch PSD times `f^α` must be within 3% of 1. Measured values
are 0.9995 to 1.002. The old `alpha_noise` scale gives 1.10 to 1.13 at
`α` in `{0.5, 1, 1.5}` and fails.

## Key files

- `speckit/noise.py`
- `tests/unit/test_noise.py`
- `notebooks/0.4_noise.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md) — one-sided estimator output
- `docs/handoffs/20260905_noise-generator-normalization.md` — the white-noise
  docstring, the `alpha_noise` absolute level, and the one-sided versus
  two-sided factor of 2. Fixed in the tree; see the handoff for the
  derivation and the release note still owed.
- Plaszczynski, *Generating long streams of 1/f^α noise*,
  doi:10.1142/S0219477507003635, cited in the module docstring. The code
  implements its `4.5` sections per decade. The level comes from the
  cascade's own transfer function, not from a closed form.

## Known limitations

A one-sided spectrum of these generators is a different convention by a
factor of 2: `Gxx = 2 * S2`. The `alpha_noise` level holds away from the
corners and from Nyquist. A cascade narrower than about two decades has no
region where both corners are negligible.

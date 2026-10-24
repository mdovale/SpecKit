# Handoff: noise generator normalization and PSD-convention documentation

## Scope

Three defects found in `speckit/noise.py` (v1.0.2) while building a colored
measurement-noise layer on top of SpecKit in a downstream project (GravNav's
`gravnav.sensor.noise`, which wraps `white_noise` and `alpha_noise` to synthesize
prescribed one-sided amplitude spectral densities).

None of them break SpecKit's own test suite. All three are about the **level** of
the generated noise, or about the documentation of that level, so they silently
propagate into anything that trusts the docstrings for absolute normalization.

Ordered by severity:

1. `white_noise` docstring states an RMS formula inconsistent with the code
   (off by $\sqrt{2}$). Documentation bug, code is correct.
2. `alpha_noise` misses its own documented normalization by up to ~13 % in PSD,
   with an alpha-dependent bias. Real behavior bug, or at minimum an
   accuracy claim that needs qualifying.
3. Generators use a two-sided PSD convention while `compute_spectrum` returns
   one-sided, with no cross-reference in either docstring, so a naive
   generate/estimate round trip is off by exactly a factor 2.

Plus one trivial packaging nit: `pyproject.toml` `[project.urls]` still points at
`https://github.com/mdovale/spectools`, while the repository is
`https://github.com/mdovale/SpecKit`.

## Current status

Not fixed. Reproducible with the snippets below against v1.0.2 as installed from
PyPI (also present in the working tree at commit `4f81ae2`).

The downstream project works around item 2 by widening its own test tolerance and
by using only `alpha = 2` in shipped instrument models (the accurate end of the
range); it does not compensate for the bias numerically, because a documented
generator normalization is the right place to fix it.

---

## Defect 1: `white_noise` docstring RMS formula

`speckit/noise.py`, `class white_noise`, lines ~141-144:

```
    psd : float, optional
        The desired two-sided Power Spectral Density (PSD) in units of
        (signal units)^2 / Hz. The RMS value of the generated time-series
        will be `sqrt(psd * f_sample / 2)`. Defaults to 1.0.
```

The constructor does

```python
self._rms = np.sqrt(psd * f_sample)
```

with no factor of 2. The two statements cannot both be true.

**The code is right and the docstring formula is wrong.** For a genuinely
two-sided density $S_2$, the variance is

$$
\operatorname{var}(x) = \int_{-f_s/2}^{+f_s/2} S_2 \, \mathrm{d}f = S_2 f_s ,
$$

so $\mathrm{RMS} = \sqrt{S_2 f_s}$, matching the code. The
`sqrt(psd * f_sample / 2)` in the docstring is the **one-sided** answer, so the
prose label ("two-sided") and the formula disagree with each other.

### Repro

```python
import numpy as np
from speckit.noise import white_noise

fs, psd = 10.0, 4.0
w = white_noise(f_sample=fs, psd=psd, seed=0)
x = w.get_series(2_000_000)

print(w.rms)                    # 6.324555 == sqrt(psd*fs)
print(np.sqrt(psd*fs/2))        # 4.472136 == docstring formula
print(x.std())                  # 6.323462 -> code, not docstring
print(x.var()/fs)               # 3.9986   -> psd is two-sided, as labelled
```

### Fix

Drop the `/ 2` from the docstring formula:

```
        will be `sqrt(psd * f_sample)`.
```

Consider also stating the variance identity explicitly, since this is the
constant everything else in the module is normalized against.

---

## Defect 2: `alpha_noise` normalization bias

`speckit/noise.py`, `class alpha_noise`, lines ~318-321:

```
    to achieve a stable and high-performance simulation. The two-sided PSD
    is scaled to be 1.0 at f=1 Hz. The spectrum is whitened below `f_min`
    and above `f_max`.
```

Measured against that claim, the realized two-sided PSD is high by up to ~13 %,
and the error depends on `alpha`:

| alpha | median $S_2^{\rm meas}/f^{-\alpha}$ | p5 | p95 |
|-------|------------------------------------:|---:|----:|
| 0.50 | 1.088 | 1.028 | 1.154 |
| 1.00 | 1.120 | 1.049 | 1.189 |
| 1.50 | 1.085 | 1.026 | 1.154 |
| 2.00 | 0.988 | 0.925 | 1.049 |

(`f_sample=10`, `f_min=1e-3`, `f_max=5`, $2^{21}$ samples, seed 1, ratio taken
over 5 mHz to 2 Hz.)

The p5/p95 spread is the expected Plaszczynski cascade ripple and is not the
issue. The issue is the **median**, which is a systematic gain error: about
+6 % in amplitude at `alpha = 1`, and correct to ~1 % only at `alpha = 2`.

The bias is essentially independent of how many decades the cascade spans, which
points at the closed-form gain rather than at cascade density:

| `f_min` | decades | sections | median ratio |
|---------|--------:|---------:|-------------:|
| 1e-1 | 1.70 | 8 | 1.099 |
| 1e-2 | 2.70 | 13 | 1.111 |
| 1e-3 | 3.70 | 17 | 1.121 |
| 1e-4 | 4.70 | 22 | 1.123 |

The suspect line is the single analytic scale factor

```python
self._scaling = 1.0 / np.power(self.fmax, self.alpha / 2.0)
```

evaluated at `self._fmax`, which is `filter_f_max_vals[-1]`, i.e. the corner of
the last cascade section rather than a frequency at which the cascade's transfer
function actually has the ideal $f^{-\alpha/2}$ magnitude. The pole/zero
placement in `__init__` interleaves corners at half-decade offsets
(`log_p_i` uses `- alpha/2` and `+ dp*alpha/2`), so the true gain at 1 Hz picks
up an alpha-dependent offset that this scaling does not absorb.

### Repro

```python
import numpy as np
from speckit.noise import alpha_noise
from speckit import compute_spectrum

fs, n = 10.0, 2**21
for alpha in (0.5, 1.0, 1.5, 2.0):
    g = alpha_noise(f_sample=fs, f_min=1e-3, f_max=fs/2, alpha=alpha, seed=1)
    x = g.get_series(n)
    sp = compute_spectrum(x, fs=fs, olap="default", bmin=1, Lmin=0, Jdes=500,
                          Kdes=100, order=0, win="Kaiser", psll=200)
    S2 = sp.Gxx / 2.0                      # Gxx is one-sided
    band = (sp.f > 5e-3) & (sp.f < 2.0)
    r = S2[band] / sp.f[band] ** (-alpha)   # ideal is 1.0
    print(alpha, np.median(r))
```

### Recommended approach

Two options, in order of preference:

1. **Normalize numerically.** Evaluate the cascade's own transfer function at
   the reference frequency and divide it out, instead of using the closed-form
   `fmax**(alpha/2)`. The coefficients are already assembled in `__init__`, so
   this is a `scipy.signal.freqz` (or a direct product of the first-order
   sections) at $f = 1\,\mathrm{Hz}$, evaluated once at construction. This makes
   the docstring claim exactly true by construction and is robust to any later
   change in pole placement.
2. **Derive the correct analytic gain** for the interleaved pole/zero layout and
   keep the closed form. Cheaper at construction, but it has to be re-derived if
   the layout ever changes.

Either way, please add a regression test that pins the median realized PSD at
1 Hz to within a few per cent for `alpha` in `{0.5, 1.0, 1.5, 2.0}`. There is
currently no test asserting the absolute level of `alpha_noise`, which is why
this went unnoticed.

Note that `pink_noise` subclasses `alpha_noise` and `red_noise` carries the same
"scaled to be 1.0 at f=1 Hz" claim (line ~249), so both should be checked and
covered by the same test.

**Compatibility warning:** fixing this changes the amplitude of every
`alpha_noise` / `pink_noise` realization by up to ~6 %, for a given seed. That is
a behavioral change, so it wants a minor-version bump and a changelog note
rather than a patch release.

---

## Defect 3: two-sided generators versus one-sided estimator

The generators in `speckit/noise.py` are documented and implemented in a
**two-sided** PSD convention. `compute_spectrum` returns a **one-sided**
estimate. Neither docstring mentions the other, so the natural sanity check a
new user writes is off by exactly 2:

```python
import numpy as np, speckit
from speckit.noise import white_noise

fs, psd = 10.0, 4.0
x  = white_noise(f_sample=fs, psd=psd, seed=0).get_series(2**20)
sp = speckit.compute_spectrum(x, fs, olap="default", order=0, win="kaiser", psll=200)
print(np.median(sp.Gxx) / psd)   # 1.9994
```

This is arguably the most costly of the three in practice, because it looks like
a factor-of-2 bug in the estimator rather than a convention mismatch.

### Fix

No code change needed. Add a short "PSD conventions" note to the `noise` module
docstring and a cross-reference in `compute_spectrum`, stating that generators
take two-sided densities, that `Gxx` is one-sided, and that
$S_1 = 2 S_2$ for a real signal. A worked round-trip example in the docs would
close this permanently.

Alternatively, and more invasively, consider accepting a one-sided `psd=` in the
generators to match the estimator, since one-sided is what instrument data
sheets quote and what most users will have in hand. That is a breaking change
and should not be bundled with defects 1 and 2.

---

## Defect 4 (trivial): stale project URLs

`pyproject.toml` lines 41-43:

```toml
[project.urls]
Homepage = "https://github.com/mdovale/spectools"
Repository = "https://github.com/mdovale/spectools"
```

The git remote is `git@github.com:mdovale/SpecKit.git`. The published PyPI
metadata for 1.0.2 therefore points at the old project name.

---

## Context

- Affected file: `speckit/noise.py` (`white_noise`, `_base_colored_noise`,
  `red_noise`, `alpha_noise`, `pink_noise`).
- Existing tests: `tests/unit/test_noise.py`, `tests/conftest.py`,
  `tests/integration/test_analysis_api.py`. These cover shape, slope, and
  reproducibility, but not absolute level for the colored generators.
- Version under test: 1.0.2 (PyPI wheel and working tree `4f81ae2`).
- The measurements above used `compute_spectrum` with
  `olap="default", bmin=1, Lmin=0, Jdes=500, Kdes=100, order=0, win="Kaiser", psll=200`.

## Success criteria

1. `white_noise` docstring RMS formula agrees with the code, and the two-sided
   label is stated unambiguously.
2. `alpha_noise`, `pink_noise`, and `red_noise` hit their documented
   normalization at the reference frequency to within a few per cent for all
   `alpha` in the supported `[0.01, 2.0]` range, with a regression test that
   would have caught the current bias.
3. The two-sided-generator / one-sided-estimator relationship is documented in
   both places, ideally with a round-trip example.
4. `pyproject.toml` URLs point at the current repository.
5. If the `alpha_noise` level changes, the release is versioned and changelogged
   as a behavioral change.

## References

- Plaszczynski, S. (2007), "Generating long streams of $1/f^\alpha$ noise",
  *Fluctuation and Noise Letters* 7(1), R1-R13,
  doi:10.1142/S0219477507003635 — the cascade design `alpha_noise` implements
  (already cited in the `speckit.noise` module docstring), including its
  accuracy discussion. Worth re-reading Sec. 3 for the intended normalization.
- Kasdin, N. J. (1995), "Discrete simulation of colored noise and stochastic
  processes and $1/f^\alpha$ power law noise generation",
  *Proc. IEEE* 83(5), 802-827, doi:10.1109/5.381848 — alternative exact-spectrum
  approach, useful as an independent cross-check when writing the level test.
- Downstream consumer and its workaround: `gravnav/sensor/noise.py` in the
  GravNav repository (`_AlphaGenerator`, and the tolerance comment in
  `tests/test_noise_models.py`).

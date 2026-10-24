# Handoff: GyySx residual used by SISO optimal analysis

## Scope

`SpectrumResult.GyySx` is not the coherent residual, and
`SISO_optimal_spectral_analysis` returns its square root. The MISO solvers
already use a residual that matches `GyyRx`. This note is that bug. It is
not a new LTPDA feature.

## Problem

For a cross-spectrum, `GyyRx` is

```
Gyy * (1 - coh) = Gyy - |Gxy|^2 / Gxx
```

`GyySx` (`speckit/analysis.py`, the `GyySx` branch in `SpectrumResult.__getattr__`)
is

```
abs(Gyy + Hxy * Hyx * Gxx - Hyx * Gxy - Hxy * Gyx)
```

with `Hxy = conj(XY) / XX` and `Hyx = conj(Hxy)`. Let `Gxy = a + ib`. The
expression inside the absolute value is real and equals

```
Gyy + (-a^2 + 3 b^2) / Gxx
```

which is `Gyy - |Gxy|^2 / Gxx` only when `b = 0`. In amplitude and phase,
`Gxy = |Gxy| e^{i φ}`, `coh = γ²`,

```
GyySx / GyyRx = [1 + γ² (3 sin²φ - cos²φ)] / (1 - γ²)
```

when `GyyRx > 0`. At `φ = 0` the ratio is 1. At `φ = 90°` and `γ² = 0.8` it
is 17. `np.abs` then reports a large positive residual where the coherent
subtraction should be small.

`SISO_optimal_spectral_analysis` (`speckit/systems.py`) returns

```
csd.f, np.sqrt(csd.GyySx)
```

so the public SISO ASD inherits the ratio above, as a square root.

The MISO residual is the one that matches `GyyRx`. Both
`MISO_analytic_optimal_spectral_analysis` and
`MISO_numeric_optimal_spectral_analysis` compute

```
S00 - sum_i H_i S0i - sum_i conj(H_i) Si0 + sum_{ij} conj(H_j) H_i T_ji
```

and return `abs(sqrt(...))`. For one input, `H = Gxy / Gxx` (this `H` is
`conj` of `SpectrumResult.Hxy`), and that expression equals
`Gyy - |Gxy|^2 / Gxx`.

PSD convention: `Gxx`, `Gyy`, and `Gxy` are one-sided,
`2 * (bin power) / (fs * S2)`. The ratio `GyySx / GyyRx` does not depend on
that scale. CSD sign is SciPy, `Pxy = X * conj(Y)`.

## Current status

Reproduced on 24 Oct 2026 against the working tree. Not fixed. No test locks
`GyySx` against `GyyRx` or against a delayed system. `tests/unit/test_systems.py`
checks that the SISO ASD is finite and non-negative. It does not check the
level.

`sympy` was not installed in `.venv`, so `MISO_analytic_optimal_spectral_analysis`
was not executed. The numeric MISO formula was evaluated on the same `Gxx`,
`Gyy`, and `Gxy` arrays. It matched `GyyRx` to about `1e-15`.

## Repro

```python
import numpy as np
from speckit.analysis import compute_spectrum

rng = np.random.default_rng(1)
N, fs, delay = 80_000, 200.0, 7
x = rng.standard_normal(N)
y = 0.8 * np.roll(x, delay) + 0.3 * rng.standard_normal(N)
res = compute_spectrum(
    np.vstack([x, y]), fs=fs,
    win="hann", olap=0.5, Jdes=100, Kdes=30, order=-1, scheduler="ltf",
)
rx, sx = res.GyyRx, res.GyySx
rel = np.abs(sx - rx) / np.maximum(np.abs(rx), 1e-30)
```

Observed:

- Median `|angle(Gxy)|` about 22°.
- Median `|GyySx - GyyRx| / |GyyRx|` about 1.54. The 95th percentile was
  tens, and one low-frequency bin overflowed because `GyyRx` crossed through
  zero (`coh` exceeded 1 by roundoff; max coherence was `1 + 2e-16`, about
  1.4% of bins).
- On bins with `0.2 < coh < 0.9` and `f > 1` Hz (42 bins): median
  `GyySx / GyyRx` was **7.5**.
- Example bin: `f = 9.46` Hz, `coh = 0.842`, phase `119.1°`,
  `GyySx / GyyRx = 17.3`, which matches the closed form above.
- About 37% of bins had `GyySx > Gyy`. A residual larger than the output
  spectrum is not a coherent subtraction.
- Median relative difference between `GyyRx` and `Gyy - |Gxy|^2 / Gxx` was
  about `1e-15`.
- The one-input MISO expression, evaluated on these arrays, matched `GyyRx`
  to about `1e-15`.

Expected: `GyySx` equals `GyyRx` (non-negative after a float clip, not after
the current expansion). `SISO_optimal_spectral_analysis` on this series
returns `sqrt(GyyRx)`, and a one-input MISO call returns the same ASD.

## What was tried

Algebraic expansion of the implemented expression. No code change. No scale
factor was applied to force `GyySx` onto `GyyRx`.

## Context

| Quantity | Where | Role |
|----------|--------|------|
| `GyyRx` | `analysis.py`, `(1 - coh) * Gyy` | Correct coherent residual |
| `GyySx` | `analysis.py`, the `abs(...)` expansion | Wrong when `Gxy` is complex |
| `Hxy` | `analysis.py`, `conj(XY) / XX` | Physical TF from x to y. Keep this sign. |
| SISO return | `systems.py`, `sqrt(GyySx)` | Public bug |
| MISO return | `systems.py`, `S00 - Sum1 - Sum2 + Sum3` | Already matches `GyyRx` for one input |

`GyyCx = coh * Gyy` is the coherent part and was not the failing quantity.

Do not "fix" this by conjugating `Hxy` only inside `GyySx`. The transfer
function `Hxy` matches a pure delay (median phase error 2.3° on this same
setup against `-360 * delay * f / fs`). The manuscript table that writes
`Hxy = Gxy / Gxx` disagrees with that. The code's `Hxy` is the one to keep.

## Success criteria

- On the repro above, median `|GyySx / GyyRx - 1|` on bins with
  `0.2 < coh < 0.9` is at the level of roundoff (`1e-12` or tighter), not
  O(1).
- The `f ≈ 9.46` Hz bin, or any bin with `|phase(Gxy)| > 60°` and
  `coh > 0.5`, satisfies the same test. A real-valued `Gxy` is not a
  sufficient test.
- `SISO_optimal_spectral_analysis(x, y, fs, **same kwargs)` matches
  `sqrt(GyyRx)` to that tolerance, and matches
  `MISO_numeric_optimal_spectral_analysis([x], y, fs, **same kwargs)`.
- A zero-phase control (no delay, real positive gain) still matches. The fix
  must not break the case the current formula gets right.
- `python -m pytest tests/unit/test_systems.py tests/integration/test_accuracy.py -q`
  passes, plus the new delay test.
- `ruff check` and `ruff format --check` on `speckit/analysis.py`,
  `speckit/systems.py`, and the new test.

Tolerance basis: the identity `Gyy (1 - γ²) = Gyy - |Gxy|² / Gxx` is exact
in exact arithmetic. The test tolerance is float error, not the scatter of
the spectrum.

## Recommended next steps

1. Define the residual once as `Gyy - |Gxy|² / Gxx`, which is `GyyRx`.
   Clip only a tiny negative from roundoff. Do not wrap the old expansion in
   `abs`.
2. Point `GyySx` at that value, or delete the special case and make `GyySx`
   an alias of `GyyRx`. Callers of the name are `SISO_optimal_spectral_analysis`
   and anything that reads the attribute. Search before removing the name.
3. Change the SISO return to that residual's square root.
4. Add the delay repro as a test. Include one MISO `q = 1` comparison.
5. Leave the MISO summation alone unless the new test shows it diverges from
   `GyyRx`. It did not on this repro.

## References

- `speckit/analysis.py` — `GyySx`, `GyyRx`, `Hxy`
- `speckit/systems.py` — `SISO_optimal_spectral_analysis`, both MISO functions,
  comment "Equation 8.16, page 191"
- Bendat and Piersol, *Engineering Applications of Correlation and Spectral
  Analysis*, the optimal residual `Syy - |Sxy|² / Sxx`
- `docs/paper/main.tex` — MISO residual
  `Syy - Σ H_i S_yxi - Σ H_j* S_xj y + Σ H_j* H_i S_xj xi`. That is the MISO
  code, not `GyySx`.
- `docs/paper/main.tex` table of spectral quantities — lists `GyyRx`, not
  `GyySx`, and writes `Hxy = Gxy / Gxx`. The implemented `Hxy` is
  `conj(Gxy) / Gxx`.

# Handoff: `Gxy` is the conjugate of `scipy.signal.csd`

## Scope

Code, docstrings, the project rule file, and the manuscript all say SpecKit's
cross-spectrum "follows SciPy: `Pxy = X * conj(Y)`". The formula matches the
code, but SciPy uses the opposite one. This note records the discrepancy and
the decision it needs. Nothing was changed here except reference docs that
now describe the code correctly. This came up while fixing
`docs/handoffs/20261024_gyysx-residual.md` and
`docs/handoffs/20261024_numpy-csd-phase.md`. It is not part of either fix.

## Problem

SpecKit's kernels average `X * conj(Y)` (`speckit/core.py`, `xyi = i1*r2 -
r1*i2`), and `Gxy = 2 * XY / (fs * S2)`. SciPy's `_spectral_helper`
(`scipy/signal/_spectral_py.py`, SciPy 1.18.1 in `.venv`) forms

```
result = np.conjugate(result) * result_y      # conj(X) * Y
```

so `scipy.signal.csd(x, y) = conj(SpecKit Gxy)` under the same DFT sign.
Bendat and Piersol also define `S_xy = E[X* Y]`, which is SciPy's
convention.

What agrees and what doesn't:

| Quantity | SpecKit | SciPy / Bendat | Same? |
|----------|---------|----------------|-------|
| `Gxy` | `X conj(Y)` | `conj(X) Y` | conjugates |
| `Hxy` | `conj(Gxy) / Gxx` = Y/X | `Pxy / Pxx` = Y/X | yes |
| `coh` | `|Gxy|² / (Gxx Gyy)` | same | yes |
| `ccoh` | `XY / sqrt(XX YY)` | `Pxy / sqrt(...)` | conjugates |
| `cs` | `Gxy * ENBW` | — | conjugate of SciPy-based |

The manuscript (`docs/paper/main.tex`, table of spectral quantities) states
both `P_xy = X Y^*` and `H_xy = G_xy / G_xx`. Those are inconsistent with
each other. With `Gxy = X Y*`, `Gxy / Gxx = conj(Y/X)`. The code's `Hxy` is
right, and the manuscript's `Hxy` formula holds only in SciPy's convention.

This mismatch is why the old `GyySx` expansion was wrong: it is the correct
residual for SciPy's `Gxy`, with `Hxy = Gxy / Gxx`.

## Current status

Not changed. Reproduced on 24 Oct 2026.

Docs that state the code's convention correctly:
`docs/reference/index.md`, `docs/reference/spectrum-estimation.md`
(related docs), `docs/reference/optimal-subtraction.md`.

Still claiming "SciPy-compatible":
- `speckit/core.py` module docstring and the `_stats_*_csd` docstrings
- `speckit/analysis.py`: `compute_single_bin`, `_lpsd_core`, and the
  `SpectrumResult` notes
- `tests/integration/test_analysis_api.py::test_csd_phase_matches_scipy`,
  comment `# should follow X * conj(Y) convention`. The test compares `Hxy`,
  which does match.
- `.cursor/rules/project-context.mdc`: "CSD follows SciPy: `Pxy = X * conj(Y)`"
- `docs/paper/main.tex`: the table rows for `Gxy` and `Hxy`, and the
  sufficient-statistics set near line 487

## Repro

```python
import numpy as np
from scipy import signal
from speckit import compute_spectrum

rng = np.random.default_rng(1)
N, fs, delay = 80_000, 200.0, 7
x = rng.standard_normal(N)
y = 0.8 * np.roll(x, delay) + 0.3 * rng.standard_normal(N)
res = compute_spectrum(np.vstack([x, y]), fs=fs, win="hann", olap=0.5,
                       Jdes=100, Kdes=30, order=-1, scheduler="ltf")
f, Pxy = signal.csd(x, y, fs=fs, nperseg=1024, noverlap=512, window="hann")
```

Observed near 9.5 Hz: `angle(Pxy) = -119.2°`, `angle(res.Gxy) = +119.1°`,
`angle(res.Hxy) = -119.1°`. The expected phase of Y/X is
`-360 * 7 * f / fs ≈ -120.6°`.

## Context

`systems.py` is internally consistent with SpecKit's `Gxy`. MISO solves
`T H = S` with `T`, `S` built from `X conj(Y)`, so its `H` is the conjugate
of the physical transfer function. The residual is a Hermitian form and does
not depend on the choice. Flipping `Gxy` would conjugate `T`, `S`, and `H`
together and leave the residual unchanged.

The PSD scale is not involved (`Gxx`, `Gyy` are one-sided and real).

## Options

1. **Keep `X conj(Y)` and fix the words.** Rewrite every "SciPy-compatible"
   to say `Gxy` is the conjugate of `scipy.signal.csd` and `Hxy` matches
   `csd(x, y) / csd(x, x)`. Fix the manuscript to `Hxy = G_xy^* / G_xx`, or
   to `Hxy = G_yx / G_xx`. No numerical change.
2. **Switch to `conj(X) Y`.** Flip the sign of `xyi` in the six CSD kernels
   and their `_np` fallbacks, and set `Hxy = Gxy / Gxx` and
   `ccoh = Gxy / sqrt(...)`. That matches SciPy, Bendat and Piersol, and the
   manuscript table as written. It changes the sign of `Gxy.imag`, `ccoh`,
   and `cs` for every user. `Hxy`, `coh`, and all residuals are unchanged.

Recommendation: option 2 if the published convention is meant to be SciPy's,
since the paper already promises it. Option 1 if downstream users depend on
the current `Gxy` phase. This is a public-convention choice for the
maintainer. Do not apply either option at one call site.

LTPDA's `cpsd` convention was not checked.

## Success criteria

- One statement of the convention, identical in code docstrings, reference
  docs, the rule file, and the manuscript.
- Option 2 only: `res.Gxy` matches `scipy.signal.csd(x, y)` phase within the
  5° median used by `test_csd_phase_matches_scipy`, with a new test on
  `Gxy` itself. `test_numpy_vs_numba_cross_parity`,
  `test_gyysx_is_coherent_residual`, and the SISO/MISO tests still pass. The
  `Hxy` formula in `SpectrumResult` changes in the same commit as the kernel
  sign.
- `ruff check` and `ruff format --check` on touched files.

## References

- SciPy `scipy/signal/_spectral_py.py`, `_spectral_helper`,
  `np.conjugate(result) * result_y`
- Bendat and Piersol, *Random Data*, cross-spectral density definition
- `speckit/core.py`, `speckit/analysis.py`, `speckit/systems.py`
- `docs/paper/main.tex` table of spectral quantities
- `docs/handoffs/20261024_gyysx-residual.md`,
  `docs/handoffs/20261024_numpy-csd-phase.md`

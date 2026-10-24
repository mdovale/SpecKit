# Handoff: NumPy kernels conjugate the cross-spectrum

## Scope

The NumPy fallbacks in `speckit/core.py` use the opposite Fourier sign from
the Numba Goertzel kernels. Auto-spectra match. Cross-spectrum phase flips.
The parity test does not see it. This note is that bug.

## Problem

The manuscript and the Numba kernels evaluate

```
A = Σ_l g[l] exp(-2 π i m l / L)
```

Goertzel in `_stats_win_only_auto` (and the five sibling kernels) finishes with

```
r = s1 - s2 * cos(ω)
i = s2 * sin(ω)
```

which is that sign. The six `_np` fallbacks build

```
e = np.exp(1j * omega * n)
```

and correlate the windowed segment with `e`. That is `exp(+j ω n)`, the
conjugate transform. `|X|²` is unchanged. `X * conj(Y)` is conjugated, so
`mu_i` and `Gxy`'s phase flip. `Hxy = conj(XY) / XX` flips with them.

`compute()` switches on `_NUMBA_ENABLED`. `compute_single_bin` always calls
the Goertzel names (`_stats_win_only_csd` and siblings), so a run with the
fallback flag off still returns the Numba phase from the single-bin path.

## Current status

Reproduced on 24 Oct 2026. Not fixed. Powers agree to about `1e-14`. Phases
are negatives of each other.

## Repro

Kernel-level, no scheduler:

```python
import numpy as np
from speckit.core import _stats_win_only_csd, _stats_win_only_csd_np

rng = np.random.default_rng(0)
N, L, fs = 8000, 512, 200.0
x = rng.standard_normal(N)
y = np.roll(x, 3)
w = np.hanning(L)
starts = np.arange(0, N - L + 1, L // 2, dtype=np.int64)
omega = 2 * np.pi * 40.0 / fs
nb = _stats_win_only_csd(x, y, starts, L, w, omega)
np_ = _stats_win_only_csd_np(x, y, starts, L, w, omega)
Znb = nb[2] + 1j * nb[3]
Znp = np_[2] + 1j * np_[3]
```

Observed:

- `|Znb|` and `|Znp|` differ by a relative `8e-15`.
- Phase of `Znb` was `-144.181°`. Phase of `Znp` was `+144.181°`.
- `angle(Znb / Znp)` was `71.6°`, which is `2 * angle(Znb)` wrapped. That is
  what you get when `Znp = conj(Znb)`, not a random phase error.
- The auto kernel on the same `x` matched `MXX` to a relative `3e-15`.

Pipeline check: `N = 30000`, `fs = 200`, seed 0, `y = roll(x, 5)`, Hann,
overlap 0.5, `Jdes = 40`, `Kdes = 20`, `order = 0`, `scheduler = "ltf"`.
Setting `speckit.analysis._NUMBA_ENABLED = False` (the flag `compute()`
actually reads) made `Gxx` agree at `1e-14` and flipped the sign of
`Gxy.imag`. `compute_single_bin` on that same frequency still matched the
Numba imaginary part (`+8.95e-4` versus the NumPy pipeline's `-8.95e-4`).

Setting `speckit.core._NUMBA_ENABLED = False` after import does **not**
change `speckit.analysis._NUMBA_ENABLED`. The bool is copied at
`from .core import _NUMBA_ENABLED`. `tests/integration/test_analysis_api.py`
`test_numpy_vs_numba_parity` patches the core module and then compares
`Gxx` only, so both runs stay on Numba and the phase bug is invisible.

## What was tried

Direct comparison of the two kernels and of `compute_spectrum` with the
analysis-module flag forced off. No change to the exponential. A wider
tolerance on `Gxx` would not detect this.

## Context

PSD convention is unaffected: one-sided `Gxx = 2 * MXX / (fs * S2)` uses
`MXX = |X|²`. The bug is the sign of the DFT inside `X` and `Y`, which
shows up in `mu_r`, `mu_i`, `Gxy`, `Hxy`, and the phase of `ccoh`.

All six fallbacks use `exp(+1j * omega * n)`: win-only, mean-detrend, and
polynomial, auto and cross. The polynomial and detrend paths were not given
their own delay repro. They share the same exponential. Fix all six.

The sign to keep is the Goertzel / manuscript sign, `exp(-j ω n)`. A pure
delay of `k` samples then has `Hxy` phase `-2 π f k / fs`. That was checked
on the Numba path (median absolute phase error 2.3° for a 7-sample delay).
Do not flip the Numba kernels to match the fallback.

`tests/integration/test_analysis_api.py` `test_csd_phase_matches_scipy`
compares `Hxy` to `scipy.signal.csd(x, y) / csd(x, x)` within 5° median.
That test runs the default (Numba) path. It is the regression to keep.

## Success criteria

- On the kernel repro, `Znp` equals `Znb` in both real and imaginary parts
  to about `1e-12` relative on the modulus, and the phase difference is
  about `0°`, not `180°` of the spectrum (a conjugate).
- The same holds for `_stats_detrend0_csd_np` and `_stats_poly_csd_np` on
  one delay series (`order = 0` and `order = 1`).
- `compute_spectrum` with the NumPy path selected the way `compute()`
  actually branches matches Numba `Gxy` phase, not only `Gxx`.
- `compute_single_bin` uses that same branch. With NumPy selected, its
  `Gxy.imag` matches the NumPy pipeline.
- `test_csd_phase_matches_scipy` and `test_cross_spectrum_accuracy` still
  pass on the Numba path.
- `ruff check` and `ruff format --check` on `speckit/core.py`,
  `speckit/analysis.py`, and the test file.

## Recommended next steps

1. Change the six fallbacks to `np.exp(-1j * omega * n)`.
2. Make `compute()` and `compute_single_bin` read one flag. The robust
   choice is to look up `speckit.core._NUMBA_ENABLED` at call time, so a
   test patch of the core module works. Patching a copied bool in
   `analysis.py` is how the current parity test misses the fallback.
3. Extend `test_numpy_vs_numba_parity` to a two-channel series and to
   `Gxy.real` and `Gxy.imag`, for `order` in `{-1, 0, 1, 2}`. Keep the
   `Gxx` check. Do not loosen it.
4. Add the 3-sample `roll` kernel assertion so a future edit cannot pass on
   power alone.

## References

- `speckit/core.py` — Goertzel `i = s2 * sinw`; fallbacks at the
  `np.exp(1j * omega * n)` lines
- `speckit/analysis.py` — import of `_NUMBA_ENABLED`, `_lpsd_core` branch,
  `compute_single_bin` kernel calls
- `tests/integration/test_analysis_api.py` — `test_numpy_vs_numba_parity`,
  `test_csd_phase_matches_scipy`
- `docs/paper/main.tex` — DFT sum with `exp(-2 π i m l / L)`
- SciPy `signal.csd` — `Pxy = X * conj(Y)` under the same DFT sign

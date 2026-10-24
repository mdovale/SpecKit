# Handoff: shared-plan N-channel cross-spectral matrix

## Scope

Item 3 of the LTPDA feature set in
`docs/handoffs/20261024_ltpda-feature-gaps.md`. Estimate the one-sided
auto-spectra and cross-spectra of `n_chan >= 1` channels on a single
frequency plan. This note does not cover `new_ltf_plan`.

## Goal

`compute_spectrum` accepts `(n_chan, n_time)` with `n_chan >= 1`. Existing
1-D and `2×N` / `N×2` calls keep the same numbers as today. For `n_chan > 2`,
every pair shares one plan (`f`, `L`, `navg` identical) and

```
P_ij = X_i * conj(X_j)
```

on that plan. Densities are one-sided, with the same scale as today's
`Gxx` and `Gxy` (`2 / (fs * S2)` times the averaged bin power).

## Current status

Not started. `SpectrumAnalyzer` accepts a 1-D array or a `2×N` / `N×2`
array. `N > 2` raises `ValueError` (`speckit/analysis.py`, the shape check
near the `iscsd` branch). The scheduler runs once per `compute_spectrum`
call. Kernels in `speckit/core.py` return `BinStats`
`(MXX, MYY, mu_r, mu_i, M2)` for one series or one pair.

## Context

**LTPDA.** `ao/cpsd`, `ao/lcpsd`, `ao/lcohere`, and `ao/ltfe` run on arrays
of analysis objects, so every pair shares a frequency plan.

**Owner.** `speckit/analysis.py` for the API. `speckit/core.py` for the
kernels. The scheduler stays in `speckit/schedulers.py` and is called
**once** per matrix, not once per pair.

**CSD convention.** SciPy: `Pxy = X * conj(Y)`, which is what the Numba
Goertzel kernels implement (`e^{-j ω l}`). The NumPy fallbacks currently
correlate with `e^{+j ω l}`, so their cross-spectrum phase is conjugated
relative to Numba while the power matches. A new N-channel kernel must use
one sign on both paths. The sign to keep is the Numba / SciPy sign. Fixing
the existing fallback is part of making the matrix test meaningful. Do not
paper over it with a looser tolerance on `Gxx` only.
`test_numpy_vs_numba_parity` sets `speckit.core._NUMBA_ENABLED` after
`analysis.py` has bound its own copy of that flag, and it compares `Gxx`
only, so it does not lock the phase.

**BinStats.** Do not change the field meanings for the 2-channel path.
Extend the reduction so the old fields are still the `(0,0)`, `(1,1)`, and
`(0,1)` entries.

**Dependencies.** Two-channel whitening
(`docs/handoffs/20261024_color-whiten-spectrum.md`) waits on this matrix.
`systems.py` and the conditioned spectra `GyyCx`, `GyyRx`, `GyySx` may keep
taking pairs until a follow-up wires them to the matrix. `GyySx` is not the
same quantity as `GyyRx` when `Gxy` has a nonzero phase. Do not use `GyySx`
as the check that a 3-channel residual matches the pair path.

## Settled decisions

- One plan per matrix. Building the matrix by calling `compute_spectrum` once
  per pair is not the implementation, and it is not the test.
- Hot loops get a Numba kernel and a `_np` fallback that return the same
  statistics.
- Conditioned spectra and optimal subtraction stay pair-wise in this change.

## Success criteria

- `n_chan == 1` and `n_chan == 2` match current `compute_spectrum` on a fixed
  seed, to the numerical noise of the existing kernels. Tight tolerance, not
  `atol=1e-2`.
- For a 3-channel series, `G_ij` from the matrix call matches three separate
  2-channel `compute_spectrum` calls that were given the **same** plan arrays
  (`f`, `L`, `D`). Letting each pair call the scheduler on its own is not
  the test.
- At each frequency the Hermitian matrix is positive semidefinite up to
  roundoff. The eigenvalue floor is set by the kernel dtype, not by a
  loosened spectral tolerance.
- Numba and `_np` agree on `MXX`, the cross mean, and `M2` for one auto bin
  and one cross bin, including the sign of `mu_i`.
- Existing integration tests for 1-D and 2-channel spectra stay green.
- `ruff check` and `ruff format --check` on the touched files.

## Recommended next steps

1. Fix the NumPy fallback Fourier sign so it matches the Goertzel kernel, and
   extend the parity test to `Gxy` phase. That is a precondition, not a
   separate product.
2. Add an N-channel reduction that reuses one plan and the same window `S2`.
3. Keep `BinStats` for the pair path as the `(0,0)`, `(1,1)`, `(0,1)` slice.
4. Only then consider `whiten2D`.

## References

- `docs/handoffs/20261024_ltpda-feature-gaps.md` — this item is step 4 of the
  recommended order.
- `docs/handoffs/20261024_color-whiten-spectrum.md`
- `speckit/core.py` — CSD comment `Pxy = X * conj(Y)`; NumPy path uses
  `exp(+1j * omega * n)`.
- `ltpda/classes/@ao/cpsd.m`, `lcpsd.m`, `lcohere.m`, `ltfe.m`
- SciPy `signal.csd`
- Tröbs and Heinzel, *Measurement* 39 (2006) — one LPSD/LTF plan across
  channels

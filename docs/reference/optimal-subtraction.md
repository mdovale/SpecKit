# Optimal subtraction

`speckit.systems` estimates the ASD of an output after removing the part
linearly predicted by one or more inputs. The spectra are one-sided
`compute_spectrum` results. The return value is a frequency array and a
residual ASD, not a `SpectrumResult`.

## Goal

Subtract the coherent contribution of measured inputs from an output and
report the residual amplitude spectral density.

## Scope

In: `SISO_optimal_spectral_analysis`,
`MISO_numeric_optimal_spectral_analysis`, and
`MISO_analytic_optimal_spectral_analysis`.

Out: time-domain coefficient fits
([time-series-tools.md](time-series-tools.md)) and the two-channel
conditioned spectra `GyyCx`, `GyyRx`, and `GyySx` already on
`SpectrumResult`. The SISO function is the square root of `GyySx` for one
input. The MISO function is the Bendat multi-input residual.

## Behavior and conventions

All series are 1-D, finite, and the same length. `fs` is a positive finite
float. Extra keyword arguments are passed through to `compute_spectrum`, so
the default scheduler, window, and detrend are those of
[spectrum-estimation.md](spectrum-estimation.md).

Channel order in each cross-spectrum is `[reference, other]`: input then
output, or input `i` then input `j`. `Gxy` is therefore `X * conj(Y)` with
`X` the first series.

`MISO_analytic_optimal_spectral_analysis` calls the numeric function and
returns its arrays. The two names exist so older callers keep working.

## Algorithm

SISO stacks `[input, output]`, calls `compute_spectrum`, and returns

```
f, sqrt(GyySx)
```

`GyySx` is `max(1 - coh, 0) * Gyy`, the same array as `GyyRx`. See
[spectrum-estimation.md](spectrum-estimation.md).

MISO builds, at every planned frequency, the input spectral matrix `T` and
the input-output cross-spectrum vector `S`:

- `T[i, i]` is `Gxx` of input `i` (or `Gyy` when that input was the second
  channel of a pair)
- `T[i, j] = Gxy` of `[input_i, input_j]`, and `T[j, i] = conj(Gxy)`
- `S[i]` is `Gxy` of `[input_i, output]`
- `S00` is the auto-spectrum of the output

A single input has no pair, so its diagonal is the auto-spectrum of a
two-channel call of that input with itself.

The optimal transfer row `H` solves `T H = S`. If `cond(T) > 1e12`, or if
`solve` raises `LinAlgError`, `H` comes from `numpy.linalg.pinv`. The residual
ASD is the absolute value of the square root of

```
S00 - sum_i H_i * conj(S_i) - sum_i conj(H_i) * S_i
    + sum_ij conj(H_j) * H_i * T[j, i]
```

The function labels this combination as Bendat and Piersol, *Engineering
Applications of Correlation and Spectral Analysis*, equation 8.16.

## Numerical checks

`tests/unit/test_systems.py` checks shapes, kwargs pass-through, length
mismatches, and that a repeated SISO call returns the same arrays. The
analytic and numeric MISO names agree from frequency index 4 at 8 decimal
places on the two-input fixture. No test compares the SISO residual with an
independent evaluation of `sqrt(GyySx)`. The code path is that evaluation.

## Key files

- `speckit/systems.py`
- `tests/unit/test_systems.py`
- `notebooks/1.0_systems.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md)
- [time-series-tools.md](time-series-tools.md) — `optimal_linear_combination`
  minimizes a time-series or Welch RMS. It does not call `compute_spectrum`.
- `docs/handoffs/20261024_lean-library-reorganization.md` — a one-input MISO
  residual and `SISO_optimal_spectral_analysis` are not the same series.
  `test_siso_vs_miso_single_input` in `tests/unit/test_systems.py` is
  commented out for that reason.

## Known limitations

A single-input call of the MISO solver fills the input auto-spectrum from a
two-channel spectrum of that input with itself, then solves the 1×1 system.
SISO instead returns `sqrt(GyySx)` of `[input, output]`. Those are different
calculations. The disabled test records that the ASDs do not agree.

# SpecKit reference

These notes describe the behavior in `speckit/` and the tests that lock it.
`compute_spectrum` returns one-sided spectral densities. Noise-generator `psd`
arguments are two-sided. A cross-spectrum is `Gxy = X * conj(Y)`, with
channel 0 as `X` and channel 1 as `Y`. That is the complex conjugate of
`scipy.signal.csd(x, y)`, which forms `conj(X) * Y`. `Hxy` matches SciPy's
`csd(x, y) / csd(x, x)`. See `docs/handoffs/20261024_csd-convention-vs-scipy.md`.

## Goal

Give a caller one place to see what each public area computes, in which units,
and which module owns the formula.

## Scope

In:

- the spectrum estimator and its plan, windows, bin kernels, and band RMS
- the noise generators
- two-channel conditioned spectra and the SISO/MISO residual ASD
- the time-series helpers and the figure helpers

Out:

- `ltpda/`, which is a vendored MATLAB tree and not the Python package
- open LTPDA feature work listed in
  `docs/handoffs/20261024_ltpda-feature-gaps.md`
- a unit-string field on `SpectrumResult` (not implemented; see
  `docs/handoffs/20261024_spectrum-units.md`)

## Behavior and conventions

`import speckit` exports `compute_spectrum`, `lpsd`, `compute_single_bin`,
`SpectrumAnalyzer`, `SpectrumResult`, and `default_rc`. That import loads
Matplotlib and applies `default_rc`. It does not import pandas, `control`,
`py7zr`, SymPy, or `tqdm`. Numba is pinned to the workqueue layer, and BLAS
threads are set to 1, in `speckit/__init__.py`. Required dependencies are
NumPy, SciPy, Matplotlib, and Numba.

`lpsd` calls `compute_spectrum`. The name does not select `lpsd_plan`.

## Algorithm

The estimator plans a logarithmic frequency grid, forms one Goertzel bin per
frequency on overlapped segments, and scales the averaged periodogram to a
one-sided density. The other notes own the pieces of that path.

## Key files

| Area | Note | Module |
|------|------|--------|
| Estimand, scaling, missing samples, PSD interval | [spectral-contract.md](spectral-contract.md) | `speckit/analysis.py`, `speckit/core.py` |
| Estimator API, densities, errors | [spectrum-estimation.md](spectrum-estimation.md) | `speckit/analysis.py` |
| Frequency plan | [frequency-scheduling.md](frequency-scheduling.md) | `speckit/schedulers.py` |
| Segment bin statistics | [bin-statistics.md](bin-statistics.md) | `speckit/core.py` |
| Windows and overlap | [windows.md](windows.md) | `speckit/flattop.py`, `speckit/utils.py` |
| Band RMS | [band-rms.md](band-rms.md) | `speckit/spectral.py` |
| Noise generators | [noise-generators.md](noise-generators.md) | `speckit/noise.py` |
| Residual ASD | [optimal-subtraction.md](optimal-subtraction.md) | `speckit/systems.py` |
| Time-series helpers | [time-series-tools.md](time-series-tools.md) | `speckit/dsp.py`, `speckit/lagrange.py` |
| Figures | [plotting.md](plotting.md) | `speckit/plotting.py` |

## Related docs

- `README.md` is the user overview. Where it disagrees with these notes, the
  code and tests win.
- `docs/paper/` is the manuscript narrative.
- `docs/handoffs/` records open discrepancies. Each note links the handoffs
  that touch its formula.
- `notebooks/0.0_quickstart.ipynb` walks through `compute_spectrum`.

# Time-series tools

`speckit.dsp` holds loading, resampling, detrending, peak picking, and a
linear noise-subtraction fit. Fractional delay is `timeshift` and
`lagrange_taps` in `speckit.lagrange`, re-exported from `dsp`. These helpers
do not plan a log spectrum and do not call `compute_spectrum`. `crop_data`
and `integral_rms` are defined in `speckit/spectral.py` and re-exported here
([band-rms.md](band-rms.md)).

## Goal

Prepare one or more real series, or subtract a linear combination of inputs
from an output, outside the spectral estimator.

## Scope

In: the public functions in `speckit/dsp.py`, including the fractional-delay
functions re-exported from `speckit/lagrange.py`.

Out: segment detrending inside a Goertzel kernel
([bin-statistics.md](bin-statistics.md)) and the spectral residual ASD
([optimal-subtraction.md](optimal-subtraction.md)).

Pandas and `py7zr` are imported when a DataFrame or `.7z` helper needs them.
They are the `io` extra in `pyproject.toml`, not required dependencies.
`import speckit.dsp` does not import them.

## Behavior and conventions

`frequency2phase(f, fs, subtract_mean=True)` returns radians,

```
(2 π / fs) * cumsum(f - mean(f))
```

with the mean subtraction skipped when `subtract_mean` is false. `fs` is in
hertz and must be positive.

`polynomial_detrend(x, order=1)` subtracts a least-squares polynomial.
`order=0` subtracts the mean. This is not `_build_Q`.

`truncation(x, n_trunc)` drops `n_trunc` samples from each end.

`resample_series(t, y, t_new)` is `numpy.interp`. `t` must be increasing.
Samples of `t_new` outside `t` take the endpoint values. DataFrame resampling
goes through `resample_to_common_grid` and
`multi_file_timeseries_resampler`.

`timeshift(data, shifts, order=31)` is defined in `speckit/lagrange.py` and
re-exported from `dsp`. It delays when `shifts` is positive, in samples.
`order` is an odd integer. The FIR has `order + 1` taps from `lagrange_taps`,
with `halfp = (order + 1) // 2`. `halfp == 1` is linear interpolation: taps
`[1 - frac, frac]`. A scalar shift is constant. An array shift must match
`data` and may vary with time. `df_timeshift` is the DataFrame wrapper in
`dsp`.

`peak_finder` marks local maxima, including a plateau midpoint when neighbors
agree within `rtol` (default `1e-2`). It fits `a + b * f**alpha` to the
non-peak samples and keeps peaks above `noise * 10**(cnr / 10)`, with `cnr`
in dB and default 10. `edge=True` allows the first and last samples.

`optimal_linear_combination_arrays` minimizes the RMS of

```
output + sum_i coefficient_i * optional_timeshift(input_i)
```

after removing the mean of each series. The time-domain objective is the RMS
of the centered residual. `domain="frequency"` instead takes the square root
of a trapezoidal integral of `scipy.signal.welch(..., scaling="density")`.
That Welch call is not `compute_spectrum`. `gradient=True` minimizes the RMS
of `numpy.gradient` of the residual. `timeshifts=True` adds one delay per
input and uses `timeshift`. The minimizer is `scipy.optimize.minimize` with
method `TNC` and `tol=1e-9`, started at zeros. The DataFrame wrapper is
`optimal_linear_combination`.

`df_detrend` applies `polynomial_detrend` to selected columns.
`multi_file_timeseries_loader` reads tabular series from files and archives,
including `.7z` through `py7zr`.

## Numerical checks

`tests/unit/test_lagrange.py` locks Lagrange taps at first order and constant
and variable integer and fractional shifts. `tests/unit/test_dsp.py` locks
the `dsp` re-export of those functions, polynomial detrend, `frequency2phase`,
`peak_finder` band limiting, DataFrame detrend and timeshift, and the SISO,
MIMO, and frequency-domain linear combination on small fixtures.

## Key files

- `speckit/lagrange.py` — `timeshift`, `lagrange_taps` (PyTDI copyright; see `NOTICE`)
- `speckit/dsp.py` — loading, resampling, detrending, and the linear combination. Re-exports `timeshift` and `lagrange_taps`
- `tests/unit/test_lagrange.py`
- `tests/unit/test_dsp.py`
- `notebooks/0.8_multi-file-processing.ipynb`,
  `notebooks/1.1_optimal-linear-combination.ipynb`,
  `notebooks/0.1_time-delay-bias-error.ipynb`

## Related docs

- [band-rms.md](band-rms.md)
- [optimal-subtraction.md](optimal-subtraction.md)
- [bin-statistics.md](bin-statistics.md)

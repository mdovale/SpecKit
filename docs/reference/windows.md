# Windows

The estimator windows each segment before the Goertzel bin. The default window
is a Kaiser window whose shape comes from a peak-sidelobe level. Flat-top
windows live in `speckit/flattop.py` with a recommended-overlap table.

## Goal

Provide the window samples, the overlap the scheduler should request, and the
sums `S1 = sum(w)` and `S2 = sum(w²)` that set ENBW and the density scale.

## Scope

In: Kaiser construction, `kaiser_alpha`, `kaiser_rov`, `win_dict`, and
`olap_dict`.

Out: the density formula that consumes `S1` and `S2`
([spectrum-estimation.md](spectrum-estimation.md)). The module docstring of
`flattop.py` lists PSLL, differentiability, and scalloping figures for each
flat-top. Those figures are not recomputed in the tests.

## Behavior and conventions

`win` accepts `"kaiser"`, `"hann"` / `"hanning"`, a key of `win_dict`, or a
callable. The default callable is `numpy.kaiser`. `scipy.signal.windows.kaiser`
is treated as the same window. `psll` defaults to 200 and is required for a
Kaiser window. It is a positive decibel sidelobe level.

`olap="default"` resolves as follows.

- Kaiser: `kaiser_rov(alpha)`, where `alpha = kaiser_alpha(psll)`.
- Any other window whose resolved name is a key of `olap_dict`: that table
  entry.
- Otherwise `0.5`, with a verbose warning.

A string window name is lowercased before the overlap lookup. `olap_dict`
keys are mixed case (`"SFT3F"`, `"Hanning"`, `"FTSRS"`). A string such as
`"SFT3F"` therefore misses the table and receives overlap `0.5`. Passing the
callable `SFT3F` recovers the key `"SFT3F"` and uses the table value `0.667`.
`"hann"` also misses `"Hanning"` and receives `0.5`, which is the Hanning
table value.

A numeric `olap` must lie in `[0, 1)`.

Kaiser samples are `numpy.kaiser(L + 1, alpha * π)[:-1]`, so the value passed
as NumPy's `beta` is `π` times `kaiser_alpha(psll)`. Other windows are called
as `win(L)` and must return length `L`.

`win_dict` names: `SFT3F`, `SFT3M`, `FTNI`, `SFT4F`, `SFT5F`, `SFT4M`,
`FTHP`, `HFT70`, `FTSRS` (function `FTSR`), `SFT5M`, `HFT90D`, `HFT95`,
`HFT116D`, `HFT144D`, `HFT169D`, `HFT196D`, `HFT223D`, `HFT248D`. Each is a
cosine sum. Coefficients are the published literals in that function.
`olap_dict` also
lists recommended overlaps for rectangular, Welch, Bartlett (`"Barlett"`),
Hanning, Hamming, Nuttall, Blackman-Harris (`BH92`), and Kaiser3/4/5 windows
that `win_dict` does not construct, plus integer keys `0` through `30`.

## Algorithm

`kaiser_alpha` evaluates a cubic in `x = psll / 100`:

```
a0 = -0.0821377
a1 =  4.71469
a2 = -0.493285
a3 =  0.0889732
alpha = a0 + x * (a1 + x * (a2 + x * a3))
```

`kaiser_rov` evaluates a cubic in `alpha` and returns a fraction:

```
a0 = 0.0061076
a1 = 0.00912223
a2 = -0.000925946
a3 = 4.42204e-05
rov = (100 - 1 / (a0 + alpha * (a1 + alpha * (a2 + alpha * a3)))) / 100
```

`tests/unit/test_utils.py` locks both polynomials at the samples in that file.

Flat-top windows use `z = π * arange(M) / M` and an alternating cosine series
`a0 - a1 cos(2z) + a2 cos(4z) - ...`. ENBW is not tabulated. The analyzer
computes `fs * sum(w²) / sum(w)²` from the samples it actually used.

## Key files

- `speckit/flattop.py` — cosine-sum windows, `olap_dict`, `win_dict`
- `tests/unit/test_flattop.py`
- `speckit/utils.py` — `kaiser_alpha`, `kaiser_rov`
- `speckit/analysis.py` — `_process_window_config`, `_build_window`
- `tests/unit/test_utils.py`
- `tests/integration/test_analysis_api.py` — `test_enbw_positive_and_reasonable`
- `notebooks/0.7_windows.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md)
- Heinzel, Rüdiger, and Schilling, *Spectrum and spectral density estimation
  by the Discrete Fourier transform (DFT)* (AEI, 2002), cited in
  `flattop.py`. The cosine coefficients in the functions are those published
  values. The module is SpecKit code under `LICENSE`. `NOTICE` records the
  citation.

## Known limitations

String flat-top names do not hit `olap_dict` after lowercasing. Callers who
want the tabulated overlap should pass the callable or set `olap` numerically.
The Kaiser3/4/5 rows in `olap_dict` are not what a Kaiser `win` uses.
`kaiser_rov` is.

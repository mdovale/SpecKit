# Plotting

`speckit.plotting` builds Matplotlib figures for time series and spectra.
`import speckit` exports `default_rc` and applies it to `matplotlib.rcParams`.
`SpectrumResult.plot` is a separate method on the result object
([spectrum-estimation.md](spectrum-estimation.md)).

## Goal

Draw log spectra and time series with one font size, one color cycle, and a
single-column figure size.

## Scope

In: `default_rc`, `figsize`, `lin_plot`, `log_plot`, `stem_plot`,
`autoscale_y`, `time_plot`, `time_histogram`, `asd_plot`, and `apply_legend`.

Out: the numbers being plotted. `asd_plot` does not estimate a spectrum.

## Behavior and conventions

`default_rc` sets `figure.dpi` to 150 and font, label, and title sizes to 8,
and installs an 11-color cycle starting at black. The module calls
`plt.rcParams.update(default_rc)` at import.

`figsize(scale)` returns `[width, height]` in inches from a 390 pt text
width, `inches_per_pt = 1/72.27`, and the golden-ratio height
`(sqrt(5) - 1) / 2`, multiplied by `scale`.

`lin_plot`, `log_plot`, and `stem_plot` draw on a caller-supplied axes.
`log_plot` uses logarithmic axes. `autoscale_y` expands the y-limits by
`margin` (default 0.1) of the data span. `apply_legend` uses the module
`legend_params`: best location, fontsize 8, frame on.

`asd_plot` draws one or more curves on a log-log axes. Frequencies are in
hertz. The y-axis label is an ASD in `unit/√Hz`. `unit` defaults to `"A"`.
`psd=True` means each array is a PSD and the function plots its square root.
The figure uses `figsize(1.2)` and dpi 300, which overrides the rcParams dpi
for that figure.

`time_plot` draws columns of a DataFrame against a time column.
`time_histogram` builds a 2-D histogram. It imports pandas on call. Pandas is
the `io` extra (`pandas`, `py7zr`).

## Algorithm

There is no spectral formula in this module. The only transform of the data
is the optional square root in `asd_plot` when `psd=True`, and the y-limit
padding in `autoscale_y`.

`SpectrumResult.plot` chooses `asd` or `bode`, and its error band for an ASD
is `Gxx_dev / (2 * sqrt(Gxx))`. That path does not call `asd_plot`.

## Key files

- `speckit/plotting.py`
- `speckit/__init__.py` — exports `default_rc` and therefore applies it
- `speckit/analysis.py` — `SpectrumResult.plot`
- `notebooks/0.0_quickstart.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md)

## Known limitations

No test locks colors, dpi, or the golden-ratio figure size. Those values are
the literals in `plotting.py`.

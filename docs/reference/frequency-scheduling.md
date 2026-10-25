# Frequency scheduling

A scheduler turns `N`, `fs`, overlap, and the targets `Jdes` and `Kdes` into
per-frequency segment lengths and start indices. `SpectrumAnalyzer` defaults
to `vectorized_ltf_plan`.

## Goal

Place frequencies from about `bmin * fs / N` up to but not including Nyquist,
with resolution that is fine at low frequency and coarse enough at high
frequency to average about `Kdes` segments.

## Scope

In: `lpsd_plan`, `ltf_plan`, `vectorized_ltf_plan`, `new_ltf_plan`, and
`find_Jdes_binary_search`.

Out: the spectral scaling applied to those segments. `num_patch_pts` is
accepted by `SpectrumAnalyzer` and forwarded for `new_ltf_plan`. The function
does not read it.

## Behavior and conventions

Every plan returns `f` (Hz), `r` (Hz), `b` (fractional bin, also stored as
`m`), `L` (samples), `K`, `navg`, `D` (list of start-index arrays), `O`
(achieved fractional overlap), and `nf`.

Shared constants, with `xov = 1 - olap`:

```
fmin    = bmin * fs / N
fmax    = fs / 2
fresmin = fs / N
freslim = fresmin * (1 + xov * (Kdes - 1))
logfact = (N / 2) ** (1 / Jdes) - 1
```

`Jdes` and `Kdes` are targets. The length of `f` is whatever the recurrence
produces. `force_target_nf=True` on `SpectrumAnalyzer` treats the configured
`Jdes` as a target `nf` and binary-searches a replacement `Jdes` in
`[MIN_JDES, MAX_JDES] = [100, 1_000_000]`. The search demands an exact `nf`
match. If none exists, `plan()` raises `RuntimeError`.

`SpectrumAnalyzer.plan` then drops bins outside `band`, if set. The schedulers
themselves do not.

`lpsd_plan` calls `ltf_plan` with `bmin = 1` and `Lmin = 1`, replacing the
caller's values.

## Algorithm

`ltf_plan` walks `fi` from `fmin` while `fi < fmax`. The candidate resolution
is `fi * logfact`. If that is at least `freslim`, it is kept. If
`sqrt(freslim * fres)` is still above `fresmin`, the resolution is that
geometric mean. Otherwise it is `fresmin`. If the bin `fi / fres` would fall
below `bmin`, `fres` becomes `fi / bmin`. Segment length is a half-up rounding
of `fs / fres`, clipped to `[Lmin, N]`, and set to `N` when only one segment
fits. The stored resolution is then `fs / L`, and the next frequency is
`fi + fres`.

Starts use `navg` segments spaced by `(N - L) / (navg - 1)` when `navg > 1`,
with each start rounded. `ltf_plan` computes that count with a local half-up
rounder, separately from the `K` stored in the first loop, using the same
overlap expression. `plan()` requires the two counts to agree with `len(D)`.

`vectorized_ltf_plan` evaluates the same compromise on a logarithmic grid of
`10 * Jdes` points from `fmin` to `fmax`, then walks that map with
`searchsorted` so each step equals the resolution looked up at the current
frequency. Lengths use `numpy.round`. `navg` is set equal to `K`.
`ltf_plan` uses a half-up rounder, so the two plans are not the same
sequence of segment lengths. On `N = 10**6`, `fs = 1`, overlap 0.75,
`bmin = 5`, `Lmin = 1000`, `Jdes = 1000`, and `Kdes = 100`, both return 849
frequencies. `r L` matches `fs` to a unit in the last place, and the
vectorized step `f[j+1] - f[j] - r[j]` is below `10**-13` relative.
Nearest-frequency relative differences reach `2.6 * 10**-3` (median
`7.9 * 10**-4`). Segment length differs on 424 of the 849 bins. The first
disagreement is at `6 * 10**-6` Hz, where the vectorized plan has
`L = 832707` and `ltf_plan` has `L = 833333`. Where `L` and `K` agree, the
starts agree.

At `bmin = 1`, `Lmin = 1`, and `Jdes = 200` (the other parameters unchanged)
both plans have 140 frequencies and the lengths differ on 130 of them. Two
of the ten bins whose lengths match still have different starts. With
`N = 5000`, `Lmin = 4000`, `Jdes = 30`, `Kdes = 2`, `bmin = 1`, and overlap
0.5, the plans agree, including three bins with `K = 1`.

`new_ltf_plan` uses one loop and three stages. Stage 1 is the LTF compromise.
When the ideal resolution `fi * logfact` reaches `freslim`, later steps decay
`L` exponentially from the crossover length toward `Lmin` across the remaining
`Jdes` budget. Once `L` would drop below `Lmin`, it stays at `Lmin` and
frequency spacing becomes linear. `navg` equals `K`. Lengths use `numpy.round`.
The analyzer docstring says `num_patch_pts` prepends low-frequency rows.
`new_ltf_plan` has no such parameter.

## Numerical checks

`tests/unit/test_schedulers.py` runs `ltf_plan`, `lpsd_plan`,
`new_ltf_plan`, and `vectorized_ltf_plan`. It requires the keys above,
`r * L = fs` within `rtol=1e-6`, and a mean relative error below `0.01` for
`f[j+1] - f[j]` against `r[j]`. The vectorized walker is also held to a
maximum relative step error below `10**-9`. Frequencies must be positive and
strictly increasing, and the last frequency must be at most Nyquist.
Segment lengths must be positive integers no larger than `N`. The same file
locks the `L` and start differences from `ltf_plan` quoted above.

## Key files

- `speckit/schedulers.py`
- `speckit/utils.py` — `find_Jdes_binary_search`, `MIN_JDES`, `MAX_JDES`
- `speckit/analysis.py` — `SpectrumAnalyzer.plan`
- `tests/unit/test_schedulers.py`, `tests/unit/test_utils.py`
- `notebooks/0.6_schedulers.ipynb`, `notebooks/1.3_vectorized-ltf.ipynb`,
  `notebooks/1.4_new-ltf.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md)
- Tröbs and Heinzel, *Improved spectrum estimation from digitized time series
  on a logarithmic frequency axis*, doi:10.1016/j.measurement.2005.10.010.
  `lpsd_plan` cites that paper and forces `bmin = 1`, `Lmin = 1`.
- `ltf_plan` cites S2-AEI-TN-3052 in its docstring.

## Known limitations

`new_ltf_plan` ignores `num_patch_pts`. The vectorized and iterative LTF
plans satisfy the same DFT identities and still differ in `L` on a large
fraction of bins, as counted above.

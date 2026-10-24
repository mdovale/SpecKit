# Bin statistics

Each frequency bin is one Goertzel DFT of every overlapped segment. The
kernels return the same five numbers, as a `BinStats` record: mean powers,
the mean cross-spectrum, and the mean squared deviation of that
cross-spectrum.

## Goal

Turn a real series, a segment-start list, a window, and a digital frequency
into the sufficient statistics that `SpectrumResult` scales into one-sided
densities.

## Scope

In: the six Numba kernels (`_stats_*_auto` and `_stats_*_csd` for window-only,
mean removal, and polynomial detrend), their `_np` fallbacks, `_build_Q`, and
`BinStats`.

Out: choosing `L` and the starts, and multiplying by `2 / (fs * S2)`. Those
live in [frequency-scheduling.md](frequency-scheduling.md) and
[spectrum-estimation.md](spectrum-estimation.md).

## Behavior and conventions

`BinStats` is `(MXX, MYY, mu_r, mu_i, M2)`.

- `MXX`, `MYY`: mean over segments of `|X|²` and `|Y|²`. For an auto-spectrum,
  `MYY == MXX`.
- `mu_r`, `mu_i`: mean of `X * conj(Y)`. For an auto-spectrum, `mu_i == 0` and
  `mu_r == MXX`.
- `M2`: mean of `|XY_k - mean(XY)|²`. It is 0 when fewer than two segments
  are present.

`omega` is radians per sample, `2 π f / fs`, and may be fractional. The
analyzer passes that value. The kernels do not assume an integer bin.

Detrend happens on the raw segment, then the window multiplies the residual:

- window only: `v[n] = x[n] * w[n]`
- order 0: subtract the segment mean, then window
- order 1 or 2: subtract `Q @ (Q.T @ x_seg)`, then window

`_build_Q(L, order)` builds a centered Vandermonde on `t` from `-1` to `1`
with columns `1`, `t`, and `t²` when `order == 2`, then a reduced QR. `Q` has
shape `(L, order + 1)`. Only orders 1 and 2 are accepted.

Numba kernels use `prange` over segments. `__init__.py` sets the Numba layer
to `workqueue` and BLAS threads to 1. If Numba cannot be imported,
`_NUMBA_ENABLED` is false and `_njit` is a no-op, so the Goertzel function
bodies still run as Python.

`_bin_stats` is the dispatcher. It reads `_NUMBA_ENABLED` on every call, so
assigning `speckit.core._NUMBA_ENABLED = False` selects the `_np` kernels for
both `compute()` and `compute_single_bin`. `x2 is None` selects an
auto-spectrum. Orders 1 and 2 require `Q`.

## Algorithm

The Goertzel recurrence on the windowed residual `v`, with `coeff = 2 cos ω`,
is

```
s0 = v[n] + coeff * s1 - s2
s2 = s1
s1 = s0
```

and the bin is

```
re = s1 - s2 * cos ω
im = s2 * sin ω
```

That is the unnormalized DFT with kernel `exp(-i ω n)`. For two channels the
cross-spectrum sample is

```
Re(X * conj(Y)) = re_x * re_y + im_x * im_y
Im(X * conj(Y)) = im_x * re_y - re_x * im_y
```

The NumPy fallbacks correlate the windowed segment with `exp(-i ω n)` and
form `X * conj(Y)` from those coefficients. The module note on `core.py`
says the Goertzel bin carries an extra common phase `exp(i ω (L-1))` per
segment, and that this phase cancels in `|X|²` and in `X * conj(Y)`.

`_overlap_dof(w, starts)` returns

```
nu = 2 K² / sum_{j,k} rho(|s_j - s_k|)²
rho(d) = sum_n w[n] w[n+d] / sum_n w[n]²
```

with `rho(d) = 0` for `d >= L`. No overlap gives `nu = 2 K`. The docstring
cites Percival and Walden (1993), eq. 292b, for an even hop. The
implementation uses the actual starts.

`M2` is the mean squared distance of the per-segment `(mu_r, mu_i)` samples
from their mean, not a Welford update written as a running pair. The reduction
matches that definition in both `_reduce_stats` and `_reduce_stats_nb`.

## Numerical checks

`tests/integration/test_analysis_api.py` selects the `_np` kernels by
assigning `speckit.core._NUMBA_ENABLED = False` and wraps them in a call
counter, so each test fails if the fallback did not run.

- `test_numpy_vs_numba_parity` compares auto-spectrum `Gxx` at `rtol=5e-3`
  for `order` in `{-1, 0, 1, 2}`.
- `test_numpy_vs_numba_cross_parity` uses a two-channel series with a
  5-sample delay. For each order it requires `Gxy.real` and `Gxy.imag` to
  agree within `1e-6 * sqrt(Gxx * Gyy)`. The two families agree to about
  `1e-9` of that scale, and a conjugated fallback is off by `O(|Gxy|)`. The
  test also checks the NumPy-path `Hxy` phase against `-2 π f 5 / fs`, and
  that `compute_single_bin` uses the same kernel family as `compute()`.
- `test_numpy_csd_kernel_sign_on_delay` compares the window-only,
  mean-removal, and linear-detrend CSD kernels directly on a 3-sample delay,
  at `1e-12` relative.

`test_csd_phase_matches_scipy` runs the default path and compares
transfer-function phase with `scipy.signal.csd`.

`tests/unit/test_core.py` locks `_overlap_dof`. Non-overlapping starts give
`2 K`, and `K` identical starts give 2. The periodic Hann at 50% overlap
matches Percival and Walden's `nu ≈ 36 K / 19` at `rtol=1e-4`; the
finite-`K` term is about `2.6e-5` at `K = 2000`. For evenly spaced starts
under the default Kaiser window, the actual-start form equals eq. 292b.

## Key files

- `speckit/core.py`
- `speckit/analysis.py` — `_bin_stats` from `compute()` and `compute_single_bin`
- `tests/integration/test_analysis_api.py`
- `notebooks/1.2_polynomial-detrend.ipynb`

## Related docs

- [spectrum-estimation.md](spectrum-estimation.md)
- `docs/handoffs/20261024_numpy-csd-phase.md` — the NumPy kernels used
  `exp(+i ω n)` and conjugated the cross-spectrum. Resolved.
- `docs/handoffs/20261024_overlap-independent-segments.md` and
  `docs/handoffs/20261024_overlap-degrees-of-freedom.md` — `_overlap_dof` is
  the `nu` both notes asked for. The chi-square interval is still open.
- `docs/handoffs/20261024_csd-convention-vs-scipy.md` — `X * conj(Y)` is the
  complex conjugate of `scipy.signal.csd`.

## Known limitations

`_overlap_dof` assumes a locally white Gaussian process and ignores the
detrend. Near DC and Nyquist the pseudo-covariance terms it drops are not
small.

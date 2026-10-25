# BSD 3-Clause License
#
# Copyright (c) 2025, Miguel Dovale (University of Arizona),
# Gerhard Heinzel (Albert Einstein Institute).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""Predict the error of a frequency plan before looking at the data.

``import speckit`` does not export this module, and ``SpectrumAnalyzer``
does not offer it as a scheduler. The default plan is still
``vectorized_ltf_plan``. The formulas below are for ``order=-1``: each
segment is multiplied by the window, and no polynomial is removed.

``G`` is the one-sided spectrum, the same kind ``compute_spectrum``
returns. For a bin of length ``L`` tuned to frequency ``f``, let
``f_k = k * fs / L`` be the DFT frequencies. A negative frequency sits
in the upper half of that grid. ``H(f_k)`` is this bin's response to a
complex tone at ``f_k``, the same number as ``BinResponse.coefficient``.
``S2`` is ``sum(w**2)``. The level we expect the estimator to return is

    predicted_mean = sum_k |H(f_k)|**2 * G(|f_k|) / (L * S2)

A flat ``G`` comes back equal to ``G``. The sum of ``|H|**2`` equals
``L * S2``, which is Parseval's relation for the windowed tone, so the
formula has no extra factor of 2. The variance and the risk are

    predicted_var = 2 * predicted_mean**2 / nu
    risk = (predicted_mean - G(f))**2 + predicted_var

``nu`` is the degrees of freedom from ``_overlap_dof`` and the real
segment start times. The first term is the squared bias. The second is
the variance. The cost of a bin is ``(number of segments) * L``. Two
plans are compared when their total costs differ by at most 5%. The
time the calculation takes is separate from the risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from speckit.core import _overlap_dof
from speckit.response import BinResponse
from speckit.utils import round_half_up

Spectrum = Callable[[np.ndarray], np.ndarray]
WindowFn = Callable[[int], np.ndarray]

COST_FRACTION = 0.05


@dataclass(frozen=True)
class RowRisk:
    """Expected level, variance, and risk for one frequency bin.

    ``mean`` is the one-sided level the estimator should return.
    ``truth`` is the true one-sided spectrum at this frequency. ``var``
    is ``2 * mean**2 / nu``, with ``nu`` the degrees of freedom.
    ``risk`` is ``(mean - truth)**2 + var``.
    """

    mean: float
    var: float
    risk: float
    nu: float
    truth: float


@dataclass(frozen=True)
class BandScore:
    """Average risk of the bins that fall inside one frequency band.

    ``segment_samples`` counts the data those bins touch. Each bin adds
    ``(number of segments) * (segment length)``. ``length`` and ``olap``
    are filled in when every bin uses the same values. ``risks[j]`` is
    the risk at frequency ``f[j]``. ``mean_risk`` is the ordinary average
    of ``risks``, one count per bin.
    """

    mean_risk: float
    segment_samples: int
    n_bins: int
    length: Optional[int]
    olap: Optional[float]
    f: np.ndarray
    risks: np.ndarray


def _require_order(order: int) -> None:
    if int(order) != -1:
        raise ValueError(
            f"Predicted risk is implemented for order=-1. Got order={order!r}."
        )


def _as_window(window: np.ndarray) -> np.ndarray:
    samples = np.ascontiguousarray(window, dtype=np.float64)
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("window must be a 1D array of length L.")
    if float(np.dot(samples, samples)) <= 0.0:
        raise ValueError("window energy S2 must be positive.")
    return samples


def _spectrum_at(spectrum: Spectrum, freq: np.ndarray) -> np.ndarray:
    values = np.asarray(spectrum(np.asarray(freq, dtype=np.float64)), dtype=np.float64)
    if values.shape != freq.shape:
        raise ValueError(
            f"spectrum must return shape {freq.shape}, got {values.shape}."
        )
    return values


def segment_starts(n: int, length: int, olap: float) -> np.ndarray:
    """Where each segment starts, using the same rule as ``compute_single_bin``.

    The segment count is ``((N - L) / (1 - olap)) / L + 1``, rounded
    half up. The starts are ``round(arange(K) * (N - L) / (K - 1))``.
    One segment starts at index 0.

    Parameters
    ----------
    n : int
        Series length.
    length : int
        Segment length. Must lie in ``1 .. N``.
    olap : float
        Requested fractional overlap in ``[0, 1)``.

    Returns
    -------
    (K,) ndarray
        Start indices, int64.

    Raises
    ------
    ValueError
        If ``length`` or ``olap`` is outside its range, or a start falls
        outside ``[0, N - L]``.
    """
    n = int(n)
    length = int(length)
    olap = float(olap)
    if length < 1 or length > n:
        raise ValueError(f"length must lie in 1 .. N, got L={length}, N={n}.")
    if not np.isfinite(olap) or not (0.0 <= olap < 1.0):
        raise ValueError(f"olap must lie in [0, 1), got {olap!r}.")
    if n == length:
        return np.array([0], dtype=np.int64)
    navg = int(round_half_up(((n - length) / (1.0 - olap)) / length + 1.0))
    if navg <= 1:
        return np.array([0], dtype=np.int64)
    shift = (n - length) / (navg - 1)
    starts = np.round(np.arange(navg, dtype=np.float64) * shift).astype(np.int64)
    if np.any(starts < 0) or np.any(starts > (n - length)):
        raise ValueError(f"segment starts fall outside [0, N-L] for L={length}, N={n}.")
    return starts


def band_frequencies(f_min: float, f_max: float, resolution: float) -> np.ndarray:
    """Frequencies from ``f_min`` up to ``f_max``, spaced by ``resolution``.

    The first value is ``f_min``. Each step adds ``resolution`` Hz. For a
    segment of length ``L`` that step is ``fs / L``. The last value is the
    last step that is still at or below ``f_max``.

    Parameters
    ----------
    f_min, f_max : float
        Band edges in Hz. ``f_min`` is included when it does not exceed
        ``f_max``.
    resolution : float
        Positive step in Hz. For a segment of length ``L`` this is ``fs / L``.

    Returns
    -------
    (J,) ndarray
        Frequencies in Hz.
    """
    f_min = float(f_min)
    f_max = float(f_max)
    resolution = float(resolution)
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(f"resolution must be positive, got {resolution!r}.")
    if not np.isfinite(f_min) or not np.isfinite(f_max) or f_min > f_max:
        raise ValueError(f"band must satisfy f_min <= f_max, got {f_min}, {f_max}.")
    count = math.floor((f_max - f_min) / resolution + 1e-9) + 1
    freqs = f_min + resolution * np.arange(count, dtype=np.float64)
    return freqs[freqs <= f_max + 1e-9 * resolution]


def kernel_power(window: np.ndarray, fs: float, freqs: np.ndarray) -> np.ndarray:
    """How strongly one bin hears each DFT frequency.

    For every analysis frequency the result is ``|H(f_k)|**2``, with
    ``f_k = k * fs / L``. ``H`` matches ``BinResponse.coefficient`` at
    ``order=-1``. A negative frequency is stored in the upper half of
    the FFT.

    Parameters
    ----------
    window : (L,) ndarray
        Window samples.
    fs : float
        Sampling frequency in Hz.
    freqs : (J,) ndarray
        Analysis frequencies in Hz.

    Returns
    -------
    (J, L) ndarray
        ``|H(ν_k)|**2`` for each analysis frequency.
    """
    samples = _as_window(window)
    fs = float(fs)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError(f"fs must be a positive finite float, got {fs!r}.")
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    length = int(samples.shape[0])
    n = np.arange(length, dtype=np.float64)
    idx = np.mod(-np.arange(length), length)
    power = np.empty((freqs.size, length), dtype=np.float64)
    chunk = max(1, 2_000_000 // length)
    scale = 2.0 * np.pi / fs
    for start in range(0, freqs.size, chunk):
        stop = min(freqs.size, start + chunk)
        block = freqs[start:stop]
        phase = np.exp(-1j * scale * block[:, None] * n[None, :])
        transform = np.fft.fft(samples[None, :] * phase, axis=1)
        power[start:stop] = np.abs(transform[:, idx]) ** 2
    return power


def _grid_abs_freq(fs: float, length: int) -> np.ndarray:
    grid = np.arange(length, dtype=np.float64) * (fs / length)
    return np.minimum(grid, fs - grid)


def predicted_means(
    window: np.ndarray,
    fs: float,
    freqs: np.ndarray,
    spectrum: Spectrum,
    order: int = -1,
) -> np.ndarray:
    """Expected one-sided spectrum at each analysis frequency.

    Each value is ``sum |H|**2 * G / (L * S2)`` over the DFT frequencies.
    A flat true spectrum comes back unchanged, because ``sum |H|**2``
    equals ``L * S2``.

    Parameters
    ----------
    window : (L,) ndarray
        Window samples. The comparison cases use a Hann window.
    fs : float
        Sampling frequency in Hz.
    freqs : (J,) ndarray
        Analysis frequencies in Hz.
    spectrum : callable
        True one-sided spectrum ``G(f)`` at a vector of frequencies in Hz.
    order : int, optional
        Detrend order. Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    (J,) ndarray
        Expected one-sided levels, one per analysis frequency.

    Raises
    ------
    ValueError
        If ``order`` is not ``-1``, or ``spectrum`` returns the wrong shape.
    """
    _require_order(order)
    samples = _as_window(window)
    freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
    length = int(samples.shape[0])
    s2 = float(np.dot(samples, samples))
    power = kernel_power(samples, fs, freqs)
    g = _spectrum_at(spectrum, _grid_abs_freq(float(fs), length))
    return (power @ g) / (length * s2)


def row_risk(
    window: np.ndarray,
    fs: float,
    freq: float,
    starts: np.ndarray,
    spectrum: Spectrum,
    order: int = -1,
) -> RowRisk:
    """Expected level, variance, and risk for one bin.

    The variance is ``2 * mean**2 / nu``. The risk adds
    ``(mean - G(f))**2`` to that variance.

    Parameters
    ----------
    window : (L,) ndarray
        Window samples.
    fs : float
        Sampling frequency in Hz.
    freq : float
        Analysis frequency in Hz.
    starts : array-like
        Segment start indices. ``nu`` is computed from these and the window.
    spectrum : callable
        True one-sided spectrum ``G(f)``.
    order : int, optional
        Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    RowRisk
        ``var`` is ``2 * mean**2 / nu``. ``risk`` is
        ``(mean - G(f))**2 + var``.
    """
    _require_order(order)
    samples = _as_window(window)
    freq_arr = np.asarray([float(freq)], dtype=np.float64)
    mean = float(predicted_means(samples, fs, freq_arr, spectrum, order=-1)[0])
    nu = float(_overlap_dof(samples, np.asarray(starts, dtype=np.int64)))
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError(f"nu must be positive, got {nu!r}.")
    truth = float(_spectrum_at(spectrum, freq_arr)[0])
    var = 2.0 * mean * mean / nu
    return RowRisk(
        mean=mean,
        var=var,
        risk=(mean - truth) ** 2 + var,
        nu=nu,
        truth=truth,
    )


def score_band(
    plan: dict,
    fs: float,
    spectrum: Spectrum,
    f_min: float,
    f_max: float,
    window_fn: Optional[WindowFn] = None,
    order: int = -1,
) -> BandScore:
    """Average risk of one plan, using frequencies inside a band.

    Each frequency uses the window ``window_fn(L)``. The default is
    ``numpy.hanning``. Segment starts come from the plan's ``D`` rows.
    The sample count is ``(number of segments) * L``, summed over the
    bins in the band. The reported risk is the plain average of the
    bin risks.

    Parameters
    ----------
    plan : dict
        Scheduler output with ``f``, ``L``, and ``D``.
    fs : float
        Sampling frequency in Hz.
    spectrum : callable
        True one-sided spectrum ``G(f)``.
    f_min, f_max : float
        Inclusive band in Hz.
    window_fn : callable, optional
        Maps a segment length to window samples. Defaults to ``numpy.hanning``.
    order : int, optional
        Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    BandScore
        Average of the bin risks, and the total number of samples touched.

    Raises
    ------
    ValueError
        If the band contains no plan frequency.
    """
    _require_order(order)
    if window_fn is None:
        window_fn = np.hanning
    freqs = np.asarray(plan["f"], dtype=np.float64)
    lengths = np.asarray(plan["L"], dtype=np.int64)
    keep = np.flatnonzero((freqs >= f_min) & (freqs <= f_max))
    if keep.size == 0:
        raise ValueError(f"no plan frequency lies in [{f_min}, {f_max}].")
    risks = np.empty(keep.size, dtype=np.float64)
    kept_f = np.empty(keep.size, dtype=np.float64)
    samples = 0
    windows = {}
    for row, index in enumerate(keep.tolist()):
        length = int(lengths[index])
        window = windows.get(length)
        if window is None:
            window = _as_window(window_fn(length))
            windows[length] = window
        starts = np.asarray(plan["D"][index], dtype=np.int64)
        risks[row] = row_risk(
            window, fs, float(freqs[index]), starts, spectrum, order=-1
        ).risk
        kept_f[row] = float(freqs[index])
        samples += int(starts.size) * length
    unique = np.unique(lengths[keep])
    length_out: Optional[int] = int(unique[0]) if unique.size == 1 else None
    olap_out: Optional[float] = None
    if "O" in plan and length_out is not None:
        overlaps = np.asarray(plan["O"], dtype=np.float64)[keep]
        if np.allclose(overlaps, overlaps[0]):
            olap_out = float(overlaps[0])
    return BandScore(
        mean_risk=float(np.mean(risks)),
        segment_samples=int(samples),
        n_bins=int(keep.size),
        length=length_out,
        olap=olap_out,
        f=kept_f,
        risks=risks,
    )


def _within_budget(samples: int, budget: int) -> bool:
    return abs(int(samples) - int(budget)) <= COST_FRACTION * float(budget)


def _length_grid(n: int) -> List[int]:
    hi = int(n)
    length = 8
    values: List[int] = []
    while length <= hi:
        values.append(length)
        grown = round(length * (2.0**0.25))
        if grown % 2:
            grown += 1
        if grown <= length:
            grown = length + 2
        length = grown
    return values


def _overlap_for_count(n: int, length: int, count: int) -> Optional[float]:
    """Overlap in ``[0, 1)`` whose start formula targets ``count`` segments."""
    if count <= 1:
        if n == length and count == 1:
            return 0.0
        return None
    if count > n - length + 1:
        return None
    shift = (n - length) / (count - 1)
    if shift < 1.0:
        return None
    olap = 1.0 - shift / float(length)
    if not (0.0 <= olap < 1.0):
        return None
    return float(olap)


def _count_window(budget: int, n_bins: int, length: int) -> Optional[tuple]:
    span = n_bins * length
    if span <= 0:
        return None
    lo = max(1, math.ceil((1.0 - COST_FRACTION) * budget / span - 1e-9))
    hi = math.floor((1.0 + COST_FRACTION) * budget / span + 1e-9)
    if hi < lo:
        return None
    return lo, hi


def constant_length_score(
    n: int,
    fs: float,
    length: int,
    olap: float,
    f_min: float,
    f_max: float,
    spectrum: Spectrum,
    order: int = -1,
) -> BandScore:
    """Score a plan that uses one segment length at every frequency.

    Frequencies start at ``f_min`` and step by ``fs / L`` until
    ``f_max``. Every bin uses a Hann window of that length and the same
    segment starts.

    Parameters
    ----------
    n : int
        Series length.
    fs : float
        Sampling frequency in Hz.
    length : int
        Segment length.
    olap : float
        Requested overlap passed to ``segment_starts``.
    f_min, f_max : float
        Inclusive band in Hz.
    spectrum : callable
        True one-sided spectrum ``G(f)``.
    order : int, optional
        Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    BandScore
        Average risk, and the total number of samples the plan touches.
    """
    _require_order(order)
    window = _as_window(np.hanning(int(length)))
    resolution = float(fs) / int(length)
    freqs = band_frequencies(f_min, f_max, resolution)
    if freqs.size == 0:
        raise ValueError(f"no frequency of step {resolution} lies in the band.")
    starts = segment_starts(n, length, olap)
    means = predicted_means(window, fs, freqs, spectrum, order=-1)
    truth = _spectrum_at(spectrum, freqs)
    nu = float(_overlap_dof(window, starts))
    if nu <= 0.0:
        raise ValueError(f"nu must be positive, got {nu!r}.")
    risks = (means - truth) ** 2 + 2.0 * means**2 / nu
    achieved = 0.0
    if starts.size > 1:
        achieved = 1.0 - float(starts[1] - starts[0]) / float(length)
    return BandScore(
        mean_risk=float(np.mean(risks)),
        segment_samples=int(starts.size) * int(length) * int(freqs.size),
        n_bins=int(freqs.size),
        length=int(length),
        olap=float(achieved),
        f=freqs,
        risks=risks,
    )


def _predicted_count(n: int, length: int, olap: float) -> int:
    """Segment count ``segment_starts`` returns for this overlap."""
    if n == length:
        return 1
    navg = int(round_half_up(((n - length) / (1.0 - olap)) / float(length) + 1.0))
    return max(navg, 1)


def _rho(window: np.ndarray) -> np.ndarray:
    """Window autocorrelation ``rho(d) = sum w[n] w[n+d] / S2``."""
    energy = float(np.dot(window, window))
    corr = np.correlate(window, window, mode="full")
    return corr[window.size - 1 :] / energy


def _nu_uniform(rho: np.ndarray, count: int, hop: float) -> float:
    """Degrees of freedom when segments are equally spaced.

    This is Percival and Walden (1993), eq. 292b, for a constant hop.
    If the hop is not a whole number of samples, the lag is rounded.
    The plan that is actually returned is scored again with
    ``_overlap_dof`` on the real start times.
    """
    if count <= 1 or hop <= 0.0:
        return 2.0 * float(max(count, 0))
    length = int(rho.shape[0])
    total = float(count)
    m_max = min(count - 1, math.floor((length - 1e-9) / hop))
    for step in range(1, m_max + 1):
        lag = round(step * hop)
        if lag >= length:
            break
        if lag <= 0:
            continue
        total += 2.0 * (count - step) * float(rho[lag]) ** 2
    return 2.0 * count * count / total


def _feasible_counts(
    n: int,
    length: int,
    n_bins: int,
    budget: int,
    lo: int,
    hi: int,
) -> List[int]:
    """Segment counts whose total samples stay within 5% of ``budget``."""
    found: List[int] = []
    seen = set()
    for count in range(lo, hi + 1):
        olap = _overlap_for_count(n, length, count)
        if olap is None:
            continue
        predicted = _predicted_count(n, length, olap)
        if predicted in seen:
            continue
        seen.add(predicted)
        samples = predicted * length * n_bins
        if _within_budget(samples, budget):
            found.append(predicted)
    return found


def _maximum_nu_starts(
    n: int,
    length: int,
    window: np.ndarray,
    counts: Sequence[int],
) -> Tuple[np.ndarray, float]:
    """The start list in ``counts`` with the most degrees of freedom.

    A quick equal-spacing formula ranks the candidates. ``_overlap_dof``
    then checks the neighbors and keeps the one with larger ``nu``. For
    a fixed length the bias depends only on the window, so the largest
    ``nu`` is the lowest risk.
    """
    rho = _rho(window)
    ranked = max(
        counts,
        key=lambda count: _nu_uniform(
            rho,
            int(count),
            (n - length) / (int(count) - 1) if int(count) > 1 else 1.0,
        ),
    )
    pool = {int(ranked), int(counts[0]), int(counts[-1])}

    def exact(count: int) -> tuple:
        olap = _overlap_for_count(n, length, count)
        if olap is None:
            raise ValueError(f"segment count {count} has no overlap in [0, 1).")
        starts = segment_starts(n, length, olap)
        return float(_overlap_dof(window, starts)), starts

    allowed = {int(count) for count in counts}
    nu, starts = exact(int(ranked))
    current = int(starts.size)
    for seed in pool:
        if seed not in allowed:
            continue
        nu_seed, starts_seed = exact(seed)
        if nu_seed > nu:
            nu, starts, current = nu_seed, starts_seed, int(starts_seed.size)
    while True:
        improved = False
        for nxt in (current - 1, current + 1):
            if nxt not in allowed:
                continue
            nu_next, starts_next = exact(nxt)
            if nu_next > nu:
                nu, starts, current = nu_next, starts_next, int(starts_next.size)
                improved = True
                break
        if not improved:
            break
    return starts, nu


def search_constant_length(
    n: int,
    fs: float,
    f_min: float,
    f_max: float,
    spectrum: Spectrum,
    budget: int,
    lengths: Optional[Sequence[int]] = None,
    order: int = -1,
) -> BandScore:
    """Find one Hann length with the lowest average risk, at a similar cost.

    Unless ``lengths`` is given, the lengths tried start at 8 and grow by
    about 19% (a factor of ``2**(1/4)``) up to ``N``. For each length the
    overlap is chosen so that

        (number of bins) * (number of segments) * L

    is within 5% of ``budget``, and so that the degrees of freedom
    ``nu`` are as large as possible. The bias depends only on the length
    and the window, so that choice has the lowest risk for the length.
    Across lengths, the smallest average risk is kept. A tie keeps the
    shorter length, then the plan with fewer segments.

    The returned variance comes from ``_overlap_dof`` on the rounded
    start times. Frequencies are spaced by ``fs / L``.
    ``vectorized_ltf_plan`` is left as it is.

    Parameters
    ----------
    n : int
        Series length.
    fs : float
        Sampling frequency in Hz.
    f_min, f_max : float
        Inclusive band in Hz.
    spectrum : callable
        True one-sided spectrum ``G(f)``.
    budget : int
        Total samples to match. Usually the default plan's cost in the band.
    lengths : sequence of int, optional
        Segment lengths to try. The default is the grid described above.
    order : int, optional
        Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    BandScore
        The plan with the lowest average risk inside the 5% cost window.

    Raises
    ------
    ValueError
        If no length has a cost within 5% of ``budget``, or ``order``
        is not ``-1``.
    """
    _require_order(order)
    if lengths is None:
        grid = _length_grid(n)
    else:
        grid = [int(length) for length in lengths]
    best: Optional[BandScore] = None
    best_count = 0
    closest = None
    for length in grid:
        if length < 2 or length > n:
            continue
        resolution = float(fs) / float(length)
        freqs = band_frequencies(f_min, f_max, resolution)
        n_bins = int(freqs.size)
        if n_bins == 0:
            continue
        window = np.hanning(length)
        if float(np.dot(window, window)) <= 0.0:
            continue
        min_count = _predicted_count(n, length, 0.0)
        min_samples = int(min_count) * length * n_bins
        if min_samples > (1.0 + COST_FRACTION) * float(budget):
            gap = abs(min_samples - int(budget))
            if closest is None or gap < closest[0]:
                closest = (gap, length, min_samples)
            continue
        window_counts = _count_window(budget, n_bins, length)
        if window_counts is None:
            continue
        lo, hi = window_counts
        feasible = _feasible_counts(n, length, n_bins, budget, lo, hi)
        if not feasible:
            continue
        starts, nu = _maximum_nu_starts(n, length, window, feasible)
        samples = int(starts.size) * length * n_bins
        gap = abs(samples - int(budget))
        if closest is None or gap < closest[0]:
            closest = (gap, length, samples)
        if not _within_budget(samples, budget) or nu <= 0.0:
            continue
        means = predicted_means(window, fs, freqs, spectrum, order=-1)
        truth = _spectrum_at(spectrum, freqs)
        risks = (means - truth) ** 2 + 2.0 * means**2 / nu
        mean_risk = float(np.mean(risks))
        achieved = 0.0
        if starts.size > 1:
            achieved = 1.0 - float(starts[1] - starts[0]) / float(length)
        candidate = BandScore(
            mean_risk=mean_risk,
            segment_samples=samples,
            n_bins=n_bins,
            length=int(length),
            olap=float(achieved),
            f=freqs,
            risks=risks,
        )
        replace = best is None or mean_risk < best.mean_risk
        if best is not None and mean_risk == best.mean_risk:
            replace = (int(length), int(starts.size)) < (
                int(best.length),
                best_count,
            )
        if replace:
            best = candidate
            best_count = int(starts.size)
    if best is None:
        detail = (
            "none"
            if closest is None
            else (f"closest L={closest[1]} samples={closest[2]} gap={closest[0]}")
        )
        raise ValueError(
            "no constant length matched the segment-sample budget "
            f"within {COST_FRACTION:.0%}. {detail}."
        )
    return best


def leaked_density(
    window: np.ndarray,
    fs: float,
    neighbor_freq: float,
    tone_freq: float,
    amplitude: float = 1.0,
    order: int = -1,
) -> float:
    """One-sided level a cosine leaks into a neighboring bin.

    A pure cosine has no power of its own in a quiet bin, so this value
    is leakage through the window. It uses the same scale as
    ``SpectrumResult.Gxx``:

        G = 2 * |X|**2 / (fs * S2)

    ``X`` is the cosine's Fourier coefficient at ``order=-1``.

    Parameters
    ----------
    window : (L,) ndarray
        Window samples.
    fs : float
        Sampling frequency in Hz.
    neighbor_freq : float
        Frequency of the quiet bin, in Hz.
    tone_freq : float
        Cosine frequency in Hz.
    amplitude : float, optional
        Cosine amplitude. Defaults to 1.
    order : int, optional
        Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    float
        Leaked one-sided level. In a quiet bin the true level is 0, so
        the square of this value is the bias part of the risk.
    """
    _require_order(order)
    samples = _as_window(window)
    response = BinResponse(samples, fs, float(neighbor_freq), order=-1)
    coef = response.cosine_coefficient(float(tone_freq), amplitude)
    return 2.0 * abs(coef) ** 2 / (float(fs) * response.S2)


def search_leakage(
    n: int,
    fs: float,
    analysis_freq: float,
    budget: int,
    lengths: Optional[Sequence[int]] = None,
    order: int = -1,
) -> BandScore:
    """Hann length with the least leakage from a cosine between two bins.

    The cosine sits halfway between ``analysis_freq`` and the next bin,
    one resolution ``fs / L`` higher. The risk is the square of the
    level that leaks into that next bin. A cosine is not random, so the
    variance term ``2 * G**2 / nu`` is left out.

    Both bins share the length and the starts, so the cost is
    ``2 * (number of segments) * L``. Among lengths whose cost is within
    5% of ``budget``, the smallest squared leakage is kept. The segment
    count is the one closest to ``budget``.

    Parameters
    ----------
    n : int
        Series length.
    fs : float
        Sampling frequency in Hz.
    analysis_freq : float
        Analysis frequency in Hz.
    budget : int
        Sample cost of the two default-plan rows this search is compared with.
    lengths : sequence of int, optional
        Lengths to try. The default runs from 8 up to ``N``.
    order : int, optional
        Only ``-1`` (window only, no polynomial) is available.

    Returns
    -------
    BandScore
        ``mean_risk`` is the squared leakage. ``n_bins`` is 2, the
        analysis bin and its neighbor.
    """
    _require_order(order)
    if lengths is None:
        grid = _length_grid(n)
    else:
        grid = [int(length) for length in lengths]
    best: Optional[BandScore] = None
    best_gap = 0
    closest = None
    for length in grid:
        if length < 2 or length > n:
            continue
        resolution = float(fs) / float(length)
        neighbor = float(analysis_freq) + resolution
        if neighbor >= float(fs) / 2.0:
            continue
        window = np.hanning(length)
        if float(np.dot(window, window)) <= 0.0:
            continue
        tone = float(analysis_freq) + 0.5 * resolution
        leakage = leaked_density(window, fs, neighbor, tone, order=-1)
        risk = leakage * leakage
        counts = _count_window(budget, 2, length)
        if counts is None:
            continue
        lo, hi = counts
        chosen = None
        for count in range(max(1, lo), hi + 1):
            olap = _overlap_for_count(n, length, count)
            if olap is None:
                continue
            starts = segment_starts(n, length, olap)
            samples = 2 * int(starts.size) * length
            gap = abs(samples - int(budget))
            if closest is None or gap < closest[0]:
                closest = (gap, length, samples)
            if not _within_budget(samples, budget):
                continue
            if chosen is None or gap < chosen[0]:
                chosen = (gap, samples, float(olap), starts.size)
        if chosen is None:
            continue
        freqs = np.array(
            [float(analysis_freq), neighbor],
            dtype=np.float64,
        )
        candidate = BandScore(
            mean_risk=float(risk),
            segment_samples=int(chosen[1]),
            n_bins=2,
            length=int(length),
            olap=float(chosen[2]),
            f=freqs,
            risks=np.array([float(risk), float(risk)], dtype=np.float64),
        )
        better = best is None or candidate.mean_risk < best.mean_risk
        tie = best is not None and candidate.mean_risk == best.mean_risk
        if better or (tie and chosen[0] < best_gap):
            best = candidate
            best_gap = int(chosen[0])
    if best is None:
        detail = (
            "none"
            if closest is None
            else (f"closest L={closest[1]} samples={closest[2]} gap={closest[0]}")
        )
        raise ValueError(
            "no length matched the leakage budget "
            f"within {COST_FRACTION:.0%}. {detail}."
        )
    return best

# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale (University of Arizona),
# Gerhard Heinzel (Albert Einstein Institute).

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

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

# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
"""Ultra-fast Python noise generators for 1/f^alpha noise.

This module provides classes for generating white, red (Brownian), and general
colored (1/f^alpha) noise time-series. The core generation method for colored
noise follows the numerically stable approach of using a cascade of first-order
IIR filters to shape white noise, as described by Plaszczynski.

This implementation is highly optimized for performance and stability:
1.  **Numba-JIT Acceleration**: The core filter cascade loop in `alpha_noise`
    is Just-In-Time (JIT) compiled by Numba, providing C-like execution speed.
2.  **Buffered Sampling**: A `get_sample()` method with an internal buffer
    amortizes the cost of generation, allowing for efficient single-sample
    retrieval in state-space models like Kalman filters.
3.  **Vectorized Initialization**: NumPy vectorization is used to rapidly
    calculate filter coefficients during class initialization.
4.  **Modular Class Structure**: A base class shares common logic, improving
    maintainability and reducing code duplication.

Requires the `numba` library (`pip install numba`).

References
----------
GENERATING LONG STREAMS OF $1/f^{alpha}$ NOISE
S. PLASZCZYNSKI
Fluctuation and Noise Letters 2007 07:01, R1-R13
https://doi.org/10.1142/S0219477507003635
"""

from __future__ import annotations
from typing import Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import butter, lfilter

import numba
from scipy import signal

_INDEX_LIMIT = np.iinfo(np.intp).max
_DEFAULT_BUFFER_SIZE = 4096


@numba.jit(nopython=True, cache=True)
def _numba_lfilter_cascade(
    samples: np.ndarray,
    a_coeffs: np.ndarray,
    b_coeffs: np.ndarray,
    zi_states: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies a cascade of IIR filters using Numba for high performance.

    This is a fast, numerically stable replacement for a Python loop over
    SciPy's `lfilter`. It processes the entire sample array through one filter
    at a time to maintain the same operational order as the original algorithm.

    Parameters
    ----------
    samples : np.ndarray
        The input signal (typically white noise).
    a_coeffs : np.ndarray
        A 2D array of numerator coefficients for the filter cascade,
        where `a_coeffs[i]` are the coefficients for the i-th filter.
    b_coeffs : np.ndarray
        A 2D array of denominator coefficients for the filter cascade.
    zi_states : np.ndarray
        A 2D array of initial filter states (delay elements).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - filtered_samples: The output signal after applying the full cascade.
        - zi_states: The final filter states after processing all samples.
    """
    filtered_samples = samples.copy()

    for i in range(a_coeffs.shape[0]):  # Iterate through each filter
        a0, a1 = a_coeffs[i]
        b0, b1 = b_coeffs[i]  # b0 is always 1.0

        z = zi_states[i, 0]  # Initial state for this filter

        for j in range(filtered_samples.shape[0]):  # Apply filter to samples
            x = filtered_samples[j]
            # Direct Form II Transposed structure, matching lfilter_zi
            y = a0 * x + z
            z = a1 * x - b1 * y
            filtered_samples[j] = y

        zi_states[i, 0] = z  # Store final state

    return filtered_samples, zi_states


class white_noise:
    """White noise generator (constant power spectrum).

    Generates a stream of random numbers drawn from a zero-mean Gaussian
    distribution, resulting in a flat power spectral density.

    Parameters
    ----------
    f_sample : float
        Sampling frequency in Hz.
    psd : float, optional
        The desired two-sided Power Spectral Density (PSD) in units of
        (signal units)^2 / Hz. The RMS value of the generated time-series
        will be `sqrt(psd * f_sample / 2)`. Defaults to 1.0.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        If None, a seed is drawn from system entropy. Defaults to None.

    Attributes
    ----------
    fs : float
        The sampling frequency in Hz.
    rms : float
        The Root Mean Square (RMS) value of the noise signal.
    """

    def __init__(
        self, f_sample: float, psd: float = 1.0, seed: Optional[int] = None
    ) -> None:
        self._fs = f_sample
        self._rms = np.sqrt(psd * f_sample)
        self._rng = np.random.default_rng(seed)
        self._buffer: np.ndarray = np.array([])

    @property
    def fs(self) -> float:
        """The sampling frequency in Hz."""
        return self._fs

    @property
    def rms(self) -> float:
        """The Root Mean Square (RMS) value of the noise signal."""
        return self._rms

    def get_sample(self) -> float:
        """Retrieves a single sample from the noise stream.

        Note
        ----
        This method is highly optimized through an internal buffer, making
        sequential calls much faster than repeated calls to `get_series(1)`.

        Returns
        -------
        float
            A single noise sample.
        """
        if self._buffer.size == 0:
            self._buffer = self.get_series(_DEFAULT_BUFFER_SIZE)

        sample = self._buffer[0]
        self._buffer = self._buffer[1:]
        return sample

    def get_series(self, npts: int) -> np.ndarray:
        """Generates an array of `npts` noise samples.

        Parameters
        ----------
        npts : int
            The number of samples to generate.

        Returns
        -------
        np.ndarray
            An array of `npts` noise samples.
        """
        if npts > _INDEX_LIMIT:
            raise ValueError(f"Argument 'npts' must be <= {_INDEX_LIMIT}.")
        return self._rng.normal(loc=0.0, scale=self.rms, size=npts)


class _base_colored_noise:
    """Base class for colored noise generators to share common logic."""

    def __init__(self) -> None:
        self._buffer: np.ndarray = np.array([])

    def _settle_filter_state(self) -> None:
        """Runs noise through the filter to bring it to a settled state.

        This is a crucial step to avoid initialization artifacts. The number of
        samples required for settling is proportional to `fs / fmin`.
        """
        npts_req = int(np.ceil(2.0 * self.fs / self.fmin))
        MAX_CHUNK_SIZE = 150_000_000  # To avoid memory errors for huge requests

        if npts_req <= MAX_CHUNK_SIZE:
            _ = self.get_series(npts_req)
        else:
            num_runs = int(np.ceil(npts_req / MAX_CHUNK_SIZE))
            for _ in range(num_runs):
                _ = self.get_series(MAX_CHUNK_SIZE)

    def get_sample(self) -> float:
        """Retrieves a single sample from the noise stream."""
        if self._buffer.size == 0:
            self._buffer = self.get_series(_DEFAULT_BUFFER_SIZE)

        sample = self._buffer[0]
        self._buffer = self._buffer[1:]
        return sample


class red_noise(_base_colored_noise):
    """Red (Brownian) noise generator (1/f^2 power spectrum).

    Generates noise with a PSD proportional to 1/f^2. The spectrum is
    whitened below `f_min`. The output is scaled such that the two-sided
    PSD is 1.0 at f=1 Hz.

    Parameters
    ----------
    f_sample : float
        Sampling frequency in Hz.
    f_min : float
        Frequency cutoff in Hz. Below `f_min`, the noise spectrum is flat.
    init_filter : bool, optional
        If True, settles the filter during initialization. This can be time-
        consuming for low `f_min`. Defaults to True.
    seed : int, optional
        Seed for the random number generator. Defaults to None.

    Attributes
    ----------
    fs : float
        The sampling frequency in Hz.
    fmin : float
        The lower cutoff frequency in Hz.
    """

    def __init__(
        self,
        f_sample: float,
        f_min: float,
        init_filter: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._fs = f_sample
        self._fmin = f_min
        self._whitenoise = white_noise(self.fs, psd=1.0, seed=seed)

        self._scaling = 1.0 / (self.fs * self.fmin)
        self._a = np.array([2.0 * np.pi * self.fmin])
        self._b = np.array([1.0, -np.exp(-2.0 * np.pi * self.fmin / self.fs)])

        zi_unscaled = signal.lfilter_zi(self._a, self._b)
        initial_random_val = self._whitenoise._rng.normal(scale=self._whitenoise.rms)
        self._zi = zi_unscaled * initial_random_val

        if init_filter:
            self._settle_filter_state()

    @property
    def fs(self) -> float:
        """The sampling frequency in Hz."""
        return self._fs

    @property
    def fmin(self) -> float:
        """The lower cutoff frequency in Hz."""
        return self._fmin

    def get_series(self, npts: int) -> np.ndarray:
        """Generates an array of `npts` red noise samples."""
        if npts > _INDEX_LIMIT:
            raise ValueError(f"Argument 'npts' must be <= {_INDEX_LIMIT}.")

        w_noise = self._whitenoise.get_series(npts)
        samples, self._zi = signal.lfilter(self._a, self._b, w_noise, zi=self._zi)
        return samples * self._scaling


class alpha_noise(_base_colored_noise):
    """Colored noise generator (1/f^alpha power spectrum).

    This generator uses a Numba-accelerated cascade of first-order filters
    to achieve a stable and high-performance simulation. The two-sided PSD
    is scaled to be 1.0 at f=1 Hz. The spectrum is whitened below `f_min`
    and above `f_max`.

    Parameters
    ----------
    f_sample : float
        Sampling frequency in Hz.
    f_min : float
        Lower frequency cutoff in Hz for the 1/f^alpha slope.
    f_max : float
        Upper frequency cutoff in Hz for the 1/f^alpha slope.
    alpha : float
        Exponent of the 1/f^alpha power spectrum. Must be in [0.01, 2.0].
    init_filter : bool, optional
        If True, settles the filter during initialization. Defaults to True.
    seed : int, optional
        Seed for the random number generator. Defaults to None.

    Attributes
    ----------
    fs : float
        The sampling frequency in Hz.
    fmin, fmax : float
        The effective lower and upper cutoff frequencies of the filter cascade.
    alpha : float
        The exponent of the power spectrum.
    """

    def __init__(
        self,
        f_sample: float,
        f_min: float,
        f_max: float,
        alpha: float,
        init_filter: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not (0.01 <= alpha <= 2.0):
            raise ValueError("The exponent alpha must be in [0.01, 2.0].")
        if f_sample < 2.0 * f_max:
            raise ValueError(f"f_sample must be >= 2 * f_max (= {2.0 * f_max}).")

        self._fs = f_sample
        self._alpha = alpha
        self._whitenoise = white_noise(self.fs, psd=1.0, seed=seed)

        log_w_min = np.log10(2.0 * np.pi * f_min)
        log_w_max = np.log10(2.0 * np.pi * f_max)
        self._num_spectra = int(np.ceil(4.5 * (log_w_max - log_w_min)))
        dp = (log_w_max - log_w_min) / self._num_spectra

        i = np.arange(self._num_spectra)
        log_p_i = log_w_min + dp * 0.5 * ((2.0 * i + 1.0) - self.alpha / 2.0)
        filter_f_min_vals = np.power(10.0, log_p_i) / (2.0 * np.pi)
        filter_f_max_vals = np.power(10.0, log_p_i + (dp * self.alpha / 2.0)) / (
            2.0 * np.pi
        )

        self._fmin = filter_f_min_vals[0]
        self._fmax = filter_f_max_vals[-1]

        a0, a1, b1 = self._calc_filter_coeffs(filter_f_min_vals, filter_f_max_vals)

        self._a_coeffs = np.vstack([a0, a1]).T.copy()
        self._b_coeffs = np.vstack([np.ones_like(b1), -b1]).T.copy()
        self._zi_states = np.zeros((self._num_spectra, 1), dtype=np.float64)

        self._scaling = 1.0 / np.power(self.fmax, self.alpha / 2.0)

        if init_filter:
            self._settle_filter_state()

    @property
    def fs(self) -> float:
        """The sampling frequency in Hz."""
        return self._fs

    @property
    def fmin(self) -> float:
        """The effective lower cutoff frequency of the filter cascade in Hz."""
        return self._fmin

    @property
    def fmax(self) -> float:
        """The effective upper cutoff frequency of the filter cascade in Hz."""
        return self._fmax

    @property
    def alpha(self) -> float:
        """The exponent of the 1/f^alpha power spectrum."""
        return self._alpha

    def get_series(self, npts: int) -> np.ndarray:
        """Generates an array of `npts` colored noise samples."""
        if npts > _INDEX_LIMIT:
            raise ValueError(f"Argument 'npts' must be <= {_INDEX_LIMIT}.")

        w_noise = self._whitenoise.get_series(npts)
        samples, self._zi_states = _numba_lfilter_cascade(
            w_noise, self._a_coeffs, self._b_coeffs, self._zi_states
        )
        return samples * self._scaling

    def _calc_filter_coeffs(
        self, f_min: np.ndarray, f_max: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized calculation of first-order filter coefficients."""
        pi_f_min = f_min * np.pi
        pi_f_max = f_max * np.pi

        den = self.fs + pi_f_min
        a0 = (self.fs + pi_f_max) / den
        a1 = -1.0 * (self.fs - pi_f_max) / den
        b1 = (self.fs - pi_f_min) / den
        return (a0, a1, b1)


class pink_noise(alpha_noise):
    """Pink noise generator (1/f power spectrum).

    This is a convenience class that specializes `alpha_noise` for alpha=1.
    The two-sided PSD is scaled to be 1.0 at f=1 Hz.

    Parameters
    ----------
    f_sample : float
        Sampling frequency in Hz.
    f_min : float
        Lower frequency cutoff in Hz.
    f_max : float
        Upper frequency cutoff in Hz.
    init_filter : bool, optional
        If True, settles the filter during initialization. Defaults to True.
    seed : int, optional
        Seed for the random number generator. Defaults to None.
    """

    def __init__(
        self,
        f_sample: float,
        f_min: float,
        f_max: float,
        init_filter: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            f_sample=f_sample,
            f_min=f_min,
            f_max=f_max,
            alpha=1.0,
            init_filter=init_filter,
            seed=seed,
        )


def fftnoise(
    f: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a real-valued time series with a prescribed magnitude spectrum by
    assigning random phases to the positive-frequency bins and enforcing
    Hermitian symmetry before inverse FFT.

    Parameters
    ----------
    f : ndarray
        Complex-valued spectrum of length ``N`` for a standard complex FFT
        (i.e., as returned by :func:`numpy.fft.fft`). Only the magnitudes
        are used; phases for bins ``1..Np`` (see Notes) are replaced with
        random phases. DC (index 0) and the Nyquist bin (if present) are
        kept real.
    rng : numpy.random.Generator or None, optional
        Random number generator to use for phase draws. If ``None``,
        ``numpy.random.default_rng()`` is used.

    Returns
    -------
    x : ndarray
        Real-valued time series of length ``N`` obtained via
        :func:`numpy.fft.ifft`, with shape ``(N,)`` and dtype ``float64`` (or
        the real dtype corresponding to the input).

    Notes
    -----
    Let ``N = len(f)`` and ``Np = (N - 1) // 2``. Random phases are applied
    to bins ``1..Np`` (inclusive). For even ``N``, the Nyquist bin is at
    index ``N/2`` and is left unchanged (must be real). Hermitian symmetry
    is enforced by setting ``f[-1:-1-Np:-1] = conj(f[1:Np+1])`` so that the
    inverse FFT is purely real.

    Examples
    --------
    Create a white spectrum and synthesize a noise waveform:

    >>> import numpy as np
    >>> N = 1024
    >>> f = np.ones(N, dtype=complex)
    >>> x = fftnoise(f)  # real-valued length-N noise
    """
    if f.ndim != 1:
        raise ValueError("`f` must be a 1D array representing an FFT spectrum.")
    N = f.size
    if N < 2:
        raise ValueError("`f` must have length >= 2.")
    rng = np.random.default_rng() if rng is None else rng

    # Work on a complex copy to avoid mutating the caller's array
    F = np.array(f, dtype=complex, copy=True)

    # Number of strictly positive-frequency bins below Nyquist
    Np = (N - 1) // 2
    if Np > 0:
        phases = rng.random(Np) * 2.0 * np.pi
        rot = np.cos(phases) + 1j * np.sin(phases)
        # Apply random phases to positive frequencies (exclude DC and Nyquist)
        F[1 : Np + 1] *= rot
        # Enforce Hermitian symmetry for a real IFFT
        F[-1 : -1 - Np : -1] = np.conj(F[1 : Np + 1])

    # Ensure DC and Nyquist (if present) are real-valued
    F[0] = np.real(F[0])
    if N % 2 == 0:  # even length -> Nyquist bin exists
        F[N // 2] = np.real(F[N // 2])

    # Return the real part of the inverse FFT
    x = np.fft.ifft(F).real
    return x


def band_limited_noise(
    min_freq: float,
    max_freq: float,
    samples: int = 1024,
    samplerate: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Synthesize band-limited white noise using randomized phases in the
    frequency domain.

    Parameters
    ----------
    min_freq : float
        Lower cutoff frequency in Hz (inclusive). Must satisfy
        ``0 <= min_freq <= max_freq``.
    max_freq : float
        Upper cutoff frequency in Hz (inclusive). Must not exceed the
        Nyquist frequency ``samplerate / 2``.
    samples : int, optional
        Number of time-domain samples ``N``. Default is ``1024``.
    samplerate : float, optional
        Sampling rate in Hz. Default is ``1.0``.
    rng : numpy.random.Generator or None, optional
        Random number generator to use for phase draws. If ``None``,
        ``numpy.random.default_rng()`` is used.

    Returns
    -------
    x : ndarray
        Real-valued band-limited noise of shape ``(samples,)``.

    Notes
    -----
    The method constructs a flat (unit-magnitude) spectrum on the frequency
    bins whose absolute frequency falls within ``[min_freq, max_freq]``, assigns
    random phases to the positive-frequency bins (excluding DC and Nyquist),
    mirrors them to enforce Hermitian symmetry, and takes an inverse FFT.

    Examples
    --------
    Generate noise between 10 Hz and 50 Hz at 1 kHz sample rate:

    >>> x = band_limited_noise(10.0, 50.0, samples=4096, samplerate=1000.0)
    """
    if samples <= 1:
        raise ValueError("`samples` must be an integer >= 2.")
    if samplerate <= 0:
        raise ValueError("`samplerate` must be positive.")
    if min_freq < 0:
        raise ValueError("`min_freq` must be >= 0.")
    if max_freq < min_freq:
        raise ValueError("`max_freq` must be >= `min_freq`.")
    nyq = samplerate / 2.0
    if max_freq > nyq + 1e-12:
        raise ValueError(f"`max_freq` must be <= Nyquist ({nyq}).")

    # Construct a magnitude spectrum mask over the two-sided FFT grid
    freqs = np.abs(np.fft.fftfreq(samples, d=1.0 / samplerate))
    F = np.zeros(samples, dtype=complex)
    band = (freqs >= min_freq) & (freqs <= max_freq)
    F[band] = 1.0  # unit magnitude within the passband

    # Randomize phases and synthesize the waveform
    x = fftnoise(F, rng=rng)
    return x


def butter_lowpass(
    cutoff: float,
    fs: float,
    order: int = 5,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Design a digital IIR low-pass Butterworth filter.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in Hz. Must satisfy ``0 < cutoff < fs/2``.
    fs : float
        Sampling frequency in Hz. Must be positive.
    order : int, optional
        Filter order (>= 1). Default is ``5``.

    Returns
    -------
    b : ndarray
        Numerator (feed-forward) filter coefficients of shape ``(order + 1,)``.
    a : ndarray
        Denominator (feedback) filter coefficients of shape ``(order + 1,)``.

    Raises
    ------
    ValueError
        If any of the input parameters are invalid (e.g., non-positive ``fs``,
        ``order < 1``, or ``cutoff`` not in the open interval ``(0, fs/2)``).

    Notes
    -----
    The filter is designed in the digital (discrete-time) domain using
    :func:`scipy.signal.butter` with normalized cutoff ``cutoff / (fs/2)`` and
    ``btype='low'``.
    """
    if fs <= 0:
        raise ValueError("`fs` must be positive.")
    if order < 1:
        raise ValueError("`order` must be an integer >= 1.")
    nyq = 0.5 * fs
    if not (0 < cutoff < nyq):
        raise ValueError(f"`cutoff` must satisfy 0 < cutoff < fs/2 (= {nyq}).")

    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b.astype(float, copy=False), a.astype(float, copy=False)


def butter_lowpass_filter(
    data: ArrayLike,
    cutoff: float,
    fs: float,
    order: int = 5,
    axis: int = -1,
) -> NDArray[np.floating]:
    """
    Apply a digital IIR low-pass Butterworth filter to data using causal filtering.

    Parameters
    ----------
    data : array_like
        Input signal array. Can be 1-D or N-D. Filtering is applied along
        ``axis``.
    cutoff : float
        Cutoff frequency in Hz. Must satisfy ``0 < cutoff < fs/2``.
    fs : float
        Sampling frequency in Hz. Must be positive.
    order : int, optional
        Filter order (>= 1). Default is ``5``.
    axis : int, optional
        Axis along which to filter. Default is ``-1``.

    Returns
    -------
    y : ndarray
        Filtered signal with the same shape as ``data`` and a floating dtype.

    Raises
    ------
    ValueError
        If any of the filter design parameters are invalid.

    Notes
    -----
    This function uses :func:`scipy.signal.lfilter`, which is *causal* and
    introduces phase delay. For zero-phase filtering, consider using
    :func:`scipy.signal.filtfilt` instead.

    Examples
    --------
    Low-pass filter a 1 kHz-sampled signal at 50 Hz:

    >>> import numpy as np
    >>> fs = 1000.0
    >>> t = np.arange(0, 1, 1/fs)
    >>> x = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*200*t)
    >>> y = butter_lowpass_filter(x, cutoff=50.0, fs=fs, order=4)
    """
    x = np.asarray(data)
    b, a = butter_lowpass(cutoff=cutoff, fs=fs, order=order)
    y = lfilter(b, a, x, axis=axis)
    return y.astype(float, copy=False)

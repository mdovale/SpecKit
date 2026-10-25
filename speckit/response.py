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
"""Numerical response of one spectral bin.

The coefficient is the direct sum ``sum y[n] exp(-i omega n)`` after the
detrend and the window. It does not call the Goertzel kernel or a scheduler.
One-sided scaling is not applied. For ``order=-1`` the ENBW is
``fs * S2 / S1**2`` with ``S1 = sum(w)`` and ``S2 = sum(w**2)``. For a
polynomial detrend the reported ENBW is that of the analysis tap after the
projection, not the raw-window formula.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from speckit.core import _build_Q


class BinResponse:
    """Direct-DFT response of one windowed, detrended bin.

    Parameters
    ----------
    window : (L,) array-like
        Window samples. Kaiser windows must already be the samples
        ``kaiser(L + 1, alpha * pi)[:-1]`` that the analyzer builds.
    fs : float
        Sampling frequency in Hz.
    freq : float
        Bin frequency in Hz. The analysis tone is ``exp(-i 2π freq n / fs)``.
    order : int, optional
        Detrend order. ``-1`` windows only, ``0`` removes the mean, ``1``
        and ``2`` remove the polynomial projection. Defaults to -1.
    """

    def __init__(
        self,
        window: ArrayLike,
        fs: float,
        freq: float,
        order: int = -1,
    ) -> None:
        self.window = np.ascontiguousarray(window, dtype=np.float64)
        self.fs = float(fs)
        self.freq = float(freq)
        self.order = int(order)
        if self.window.ndim != 1 or self.window.size < 1:
            raise ValueError("window must be a 1D array of length L.")
        if not np.isfinite(self.fs) or self.fs <= 0.0:
            raise ValueError(f"fs must be a positive finite float, got {fs!r}.")
        if not np.isfinite(self.freq):
            raise ValueError(f"freq must be finite, got {freq!r}.")
        if self.order not in (-1, 0, 1, 2):
            raise ValueError(
                f"order must be one of {{-1, 0, 1, 2}}, got {self.order!r}."
            )
        self.L = int(self.window.size)
        self.S1 = float(np.sum(self.window))
        self.S2 = float(np.dot(self.window, self.window))
        self._omega = 2.0 * np.pi * self.freq / self.fs
        self._n = np.arange(self.L, dtype=np.float64)
        self._Q: Optional[np.ndarray]
        if self.order in (1, 2):
            self._Q = _build_Q(self.L, self.order)
        else:
            self._Q = None
        self._tap = self._analysis_tap()

    def _analysis_tap(self) -> np.ndarray:
        """Effective taps ``a`` such that the coefficient is ``a · y``.

        ``a = (I - P) (w exp(-i ω n))``. ``P`` is the detrend projection.
        """
        tap = self.window * np.exp(-1j * self._omega * self._n)
        if self.order == -1:
            return tap
        if self.order == 0:
            return tap - tap.mean()
        q = self._Q
        assert q is not None
        return tap - q @ (q.T @ tap)

    def coefficient(self, tone_freq: float) -> complex:
        """Coefficient of ``exp(i 2π tone_freq n / fs)``.

        Parameters
        ----------
        tone_freq : float
            Tone frequency in Hz. Negative frequencies are allowed and are
            the negative-frequency image.

        Returns
        -------
        complex
            Direct sum after the detrend and the window. One-sided scaling
            is not applied.
        """
        tone = np.exp(1j * (2.0 * np.pi * float(tone_freq) / self.fs) * self._n)
        if self.order == 0:
            tone = tone - tone.mean()
        elif self.order in (1, 2):
            q = self._Q
            assert q is not None
            tone = tone - q @ (q.T @ tone)
        windowed = tone * self.window
        return complex(np.dot(windowed, np.exp(-1j * self._omega * self._n)))

    def cosine_coefficient(self, tone_freq: float, amplitude: float = 1.0) -> complex:
        """Coefficient of ``amplitude * cos(2π tone_freq n / fs)``.

        The cosine is half the complex tone at ``+tone_freq`` and half the
        tone at ``-tone_freq``. Both halves pass through this bin's detrend
        and window.

        Parameters
        ----------
        tone_freq : float
            Tone frequency in Hz.
        amplitude : float, optional
            Cosine amplitude. Defaults to 1.

        Returns
        -------
        complex
            Direct sum after the detrend and the window.
        """
        half = 0.5 * float(amplitude)
        return half * self.coefficient(tone_freq) + half * self.coefficient(
            -float(tone_freq)
        )

    @property
    def coherent_gain(self) -> complex:
        """Response at the bin frequency to ``exp(i ω n)``.

        For ``order=-1`` this equals ``S1``.
        """
        return complex(np.dot(self._tap, np.exp(1j * self._omega * self._n)))

    @property
    def window_enbw(self) -> float:
        """``fs * S2 / S1**2`` of the raw window, in Hz.

        This is the ENBW ``SpectrumResult.ENBW`` reports. It ignores the
        detrend.
        """
        return self.fs * self.S2 / (self.S1 * self.S1)

    @property
    def enbw_numerical(self) -> float:
        """ENBW from the DFT of the raw window, in Hz.

        ``sum_k |H[k]|**2 * (fs / L) / |H[0]|**2``. For a real window this
        matches ``window_enbw``. It is not the detrended ENBW.
        """
        transform = np.fft.fft(self.window)
        return float(
            np.sum(np.abs(transform) ** 2)
            * (self.fs / self.L)
            / np.abs(transform[0]) ** 2
        )

    @property
    def enbw(self) -> float:
        """ENBW of the analysis tap, in Hz.

        ``fs * sum(|a|**2) / |a · exp(i ω n)|**2``. For ``order=-1`` this
        is ``fs * S2 / S1**2``. For ``order >= 0`` it is the ENBW after the
        detrend projection.
        """
        noise = float(np.vdot(self._tap, self._tap).real)
        gain = self.coherent_gain
        return self.fs * noise / (abs(gain) ** 2)

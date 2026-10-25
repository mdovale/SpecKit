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
"""Immutable spectral plan and its SHA-256 record.

The schedulers still return dicts. ``SpectralPlan`` is a validated copy of
that dict plus the estimator configuration the dict does not carry. Densities
that use the plan are one-sided. The cross-spectrum orientation recorded here
is ``X*conj(Y)``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

SCHEMA = 1
PSD_CONVENTION = "one-sided"
CSD_CONVENTION = "X*conj(Y)"
_REQUIRED = ("f", "r", "b", "L", "K", "navg", "D", "O")
_VECTORS = ("f", "r", "b", "L", "K", "navg", "O")
# Quiet NaN, little-endian, used when a Kaiser PSLL does not apply.
_NAN_LE = bytes.fromhex("000000000000f87f")


def _readonly(values: Any, dtype: np.dtype) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True, order="C")
    array.setflags(write=False)
    return array


def _f64(value: float) -> bytes:
    return struct.pack("<d", float(value))


def _i64(value: int) -> bytes:
    return struct.pack("<q", int(value))


def _str(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<I", len(raw)) + raw


def _hex_f64(text: str) -> float:
    blob = bytes.fromhex(text)
    if len(blob) != 8:
        raise ValueError("Expected 8 bytes of little-endian float64.")
    return struct.unpack("<d", blob)[0]


def _hex_array(text: str, dtype: np.dtype) -> np.ndarray:
    blob = bytes.fromhex(text)
    item = np.dtype(dtype).itemsize
    if len(blob) % item != 0:
        raise ValueError("Hex array length is not a whole number of elements.")
    array = np.frombuffer(blob, dtype=np.dtype(dtype).newbyteorder("<"))
    return np.array(array, dtype=dtype, copy=True)


@dataclass(frozen=True, eq=False)
class SpectralPlan:
    """Validated frequency plan and the configuration that produced it.

    Arrays are write-protected copies. ``as_dict`` returns a writable
    compatibility view. ``plan()`` on ``SpectrumAnalyzer`` still returns
    the scheduler dict; this object sits beside that dict.

    ``digest`` is SHA-256 of ``canonical_bytes``. The layout, all
    little-endian, is:

    1. schema as uint32
    2. scheduler, window, missing, ``one-sided``, and ``X*conj(Y)``,
       each as uint32 byte-length then UTF-8
    3. ``fs``, resolved overlap, ``bmin``, and ``psll`` as float64.
       ``psll`` is a canonical quiet NaN when the window is not Kaiser
    4. ``N``, ``Lmin``, ``Jdes``, ``Kdes``, and ``order`` as int64
    5. band flag as uint8. When the flag is 1, ``fmin`` and ``fmax``
       follow as float64
    6. ``nf`` as int64, then ``f``, ``r``, ``b`` as float64, ``L``,
       ``K``, ``navg`` as int64, and ``O`` as float64
    7. for each bin, ``len(D[j])`` as int64, then that row of starts

    Parameters
    ----------
    f, r, b, O : (nf,) ndarray
        Frequency in Hz, resolution in Hz, fractional bin, and achieved
        fractional overlap.
    L, K, navg : (nf,) ndarray
        Segment length, segment count, and the scheduler's average count.
    D : tuple of (K,) ndarray
        Segment starts for each bin.
    fs : float
        Sampling frequency in Hz.
    N : int
        Series length the plan was built for.
    olap : float
        Resolved requested overlap in ``[0, 1)``.
    bmin : float
        Minimum fractional bin passed to the scheduler.
    Lmin, Jdes, Kdes, order : int
        Scheduler controls and the detrend order.
    scheduler, window, missing : str
        Scheduler name, window name, and missing-sample policy.
    psll : float or None
        Kaiser peak-sidelobe level in dB. None for every other window.
    band : tuple of float or None
        Frequency limits applied after the scheduler, in Hz.
    """

    f: np.ndarray
    r: np.ndarray
    b: np.ndarray
    L: np.ndarray
    K: np.ndarray
    navg: np.ndarray
    O: np.ndarray
    D: Tuple[np.ndarray, ...]
    fs: float
    N: int
    olap: float
    bmin: float
    Lmin: int
    Jdes: int
    Kdes: int
    order: int
    scheduler: str
    window: str
    missing: str
    psll: Optional[float]
    band: Optional[Tuple[float, float]]

    @classmethod
    def from_dict(cls, plan: Dict[str, Any], **config: Any) -> SpectralPlan:
        """Copy ``plan`` and ``config`` into a validated ``SpectralPlan``.

        Parameters
        ----------
        plan : dict
            Scheduler output with keys ``f``, ``r``, ``b``, ``L``, ``K``,
            ``navg``, ``D``, and ``O``.
        **config
            ``fs``, ``N``, ``olap``, ``bmin``, ``Lmin``, ``Jdes``, ``Kdes``,
            ``order``, ``scheduler``, ``window``, ``missing``, and optional
            ``psll`` and ``band``.

        Returns
        -------
        SpectralPlan
            Write-protected copy. Later edits of ``plan`` do not affect it.

        Raises
        ------
        ValueError
            If a required key is missing, array lengths disagree, a start
            list is empty, a start lies outside ``[0, N-L]``, or ``K``
            disagrees with the number of starts.
        """
        missing_keys = [key for key in _REQUIRED if key not in plan]
        if missing_keys:
            raise ValueError(
                "Scheduler output missing key(s): " + ", ".join(missing_keys) + "."
            )
        f = _readonly(plan["f"], np.float64)
        r = _readonly(plan["r"], np.float64)
        b = _readonly(plan["b"], np.float64)
        length = _readonly(plan["L"], np.int64)
        k = _readonly(plan["K"], np.int64)
        navg = _readonly(plan["navg"], np.int64)
        overlap = _readonly(plan["O"], np.float64)
        vectors = (f, r, b, length, k, navg, overlap)
        nf = int(f.shape[0])
        if f.ndim != 1 or any(array.shape != (nf,) for array in vectors):
            raise ValueError(
                "Scheduler arrays have inconsistent lengths: "
                + ", ".join(
                    f"{name}={array.shape}" for name, array in zip(_VECTORS, vectors)
                )
                + "."
            )
        if nf == 0:
            raise ValueError("Scheduler produced no frequencies.")
        if "nf" in plan and int(plan["nf"]) != nf:
            raise ValueError(f"Scheduler nf={plan['nf']!r} does not match len(f)={nf}.")
        if np.any(length < 1):
            raise ValueError("Every segment length L must be at least 1.")

        n = int(config["N"])
        if n < 1:
            raise ValueError(f"N must be positive, got {n}.")
        starts_in = plan["D"]
        if isinstance(starts_in, np.ndarray) and starts_in.dtype == object:
            starts_in = list(starts_in)
        if not isinstance(starts_in, (list, tuple)) or len(starts_in) != nf:
            raise ValueError(
                "Scheduler 'D' must be a list of per-frequency start arrays "
                f"with length {nf}."
            )
        starts = []
        for i, row in enumerate(starts_in):
            arr = _readonly(row, np.int64)
            if arr.ndim != 1:
                raise ValueError(f"D[{i}] must be 1D, got shape {arr.shape}.")
            if arr.size == 0:
                raise ValueError(f"D[{i}] is empty.")
            if int(k[i]) != int(arr.size):
                raise ValueError(
                    f"K[{i}]={int(k[i])} does not match number of starts "
                    f"D[{i}]={arr.size}."
                )
            seg_l = int(length[i])
            if np.any(arr < 0) or np.any(arr > (n - seg_l)):
                raise ValueError(
                    f"D[{i}] contains out-of-bounds starts for L={seg_l} and N={n}."
                )
            starts.append(arr)

        fs = float(config["fs"])
        if not np.isfinite(fs) or fs <= 0.0:
            raise ValueError(f"fs must be a positive finite float, got {fs!r}.")
        olap = float(config["olap"])
        if not np.isfinite(olap) or not (0.0 <= olap < 1.0):
            raise ValueError(f"olap must be in [0, 1), got {olap!r}.")
        order = int(config["order"])
        if order not in (-1, 0, 1, 2):
            raise ValueError(f"order must be one of {{-1, 0, 1, 2}}, got {order!r}.")
        missing = str(config["missing"])
        if missing not in ("raise", "zero"):
            raise ValueError(f"missing must be 'raise' or 'zero', got {missing!r}.")
        psll = config.get("psll")
        psll_value = None if psll is None else float(psll)
        if psll_value is not None and not np.isfinite(psll_value):
            raise ValueError(f"psll must be finite, got {psll_value!r}.")
        band = config.get("band")
        band_value: Optional[Tuple[float, float]]
        if band is None:
            band_value = None
        else:
            fmin, fmax = float(band[0]), float(band[1])
            if not (np.isfinite(fmin) and np.isfinite(fmax) and fmax >= fmin):
                raise ValueError(f"Invalid band (fmin, fmax)={band!r}.")
            band_value = (fmin, fmax)

        return cls(
            f=f,
            r=r,
            b=b,
            L=length,
            K=k,
            navg=navg,
            O=overlap,
            D=tuple(starts),
            fs=fs,
            N=n,
            olap=olap,
            bmin=float(config["bmin"]),
            Lmin=int(config["Lmin"]),
            Jdes=int(config["Jdes"]),
            Kdes=int(config["Kdes"]),
            order=order,
            scheduler=str(config["scheduler"]),
            window=str(config["window"]),
            missing=missing,
            psll=psll_value,
            band=band_value,
        )

    def as_dict(self) -> Dict[str, Any]:
        """Return a writable copy of the scheduler arrays.

        Returns
        -------
        dict
            Keys ``f``, ``r``, ``b``, ``L``, ``K``, ``navg``, ``D``, ``O``,
            and ``nf``. Mutating the copy does not change this plan.
        """
        return {
            "f": np.array(self.f, copy=True),
            "r": np.array(self.r, copy=True),
            "b": np.array(self.b, copy=True),
            "L": np.array(self.L, copy=True),
            "K": np.array(self.K, copy=True),
            "navg": np.array(self.navg, copy=True),
            "D": [np.array(row, copy=True) for row in self.D],
            "O": np.array(self.O, copy=True),
            "nf": int(self.f.shape[0]),
        }

    def _psll_bytes(self) -> bytes:
        if self.psll is None:
            return _NAN_LE
        return _f64(self.psll)

    def canonical_bytes(self) -> bytes:
        """Little-endian byte layout hashed by ``digest``.

        Returns
        -------
        bytes
            The layout documented on this class.
        """
        parts = [
            struct.pack("<I", SCHEMA),
            _str(self.scheduler),
            _str(self.window),
            _str(self.missing),
            _str(PSD_CONVENTION),
            _str(CSD_CONVENTION),
            _f64(self.fs),
            _f64(self.olap),
            _f64(self.bmin),
            self._psll_bytes(),
            _i64(self.N),
            _i64(self.Lmin),
            _i64(self.Jdes),
            _i64(self.Kdes),
            _i64(self.order),
        ]
        if self.band is None:
            parts.append(b"\x00")
        else:
            parts.append(b"\x01")
            parts.append(_f64(self.band[0]))
            parts.append(_f64(self.band[1]))
        nf = int(self.f.shape[0])
        parts.append(_i64(nf))
        parts.append(np.ascontiguousarray(self.f, dtype="<f8").tobytes())
        parts.append(np.ascontiguousarray(self.r, dtype="<f8").tobytes())
        parts.append(np.ascontiguousarray(self.b, dtype="<f8").tobytes())
        parts.append(np.ascontiguousarray(self.L, dtype="<i8").tobytes())
        parts.append(np.ascontiguousarray(self.K, dtype="<i8").tobytes())
        parts.append(np.ascontiguousarray(self.navg, dtype="<i8").tobytes())
        parts.append(np.ascontiguousarray(self.O, dtype="<f8").tobytes())
        for row in self.D:
            parts.append(_i64(row.size))
            parts.append(np.ascontiguousarray(row, dtype="<i8").tobytes())
        return b"".join(parts)

    def digest(self) -> str:
        """SHA-256 hex digest of ``canonical_bytes``.

        Returns
        -------
        str
            64 hexadecimal characters. Not Python's ``hash`` and not a pickle.
        """
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def report(self) -> Dict[str, Any]:
        """Counts and ranges for this plan.

        The plan arrays are not modified. ``K=1`` bins and segment lengths
        sitting on ``Lmin`` or ``N`` are counted and left in place.

        Returns
        -------
        dict
            ``nf``, ``f_min``, ``f_max``, ``L_min``, ``L_max``, ``K_median``,
            ``olap_min``, ``olap_max``, ``n_k1``, ``n_L_at_bound``, and
            ``text``.
        """
        nf = int(self.f.shape[0])
        n_bound = int(np.count_nonzero((self.L == self.Lmin) | (self.L == self.N)))
        facts: Dict[str, Any] = {
            "nf": nf,
            "f_min": float(self.f[0]),
            "f_max": float(self.f[-1]),
            "L_min": int(np.min(self.L)),
            "L_max": int(np.max(self.L)),
            "K_median": float(np.median(self.K)),
            "olap_min": float(np.min(self.O)),
            "olap_max": float(np.max(self.O)),
            "n_k1": int(np.count_nonzero(self.K == 1)),
            "n_L_at_bound": n_bound,
        }
        facts["text"] = (
            f"nf={facts['nf']}\n"
            f"f=[{facts['f_min']:.6g}, {facts['f_max']:.6g}] Hz\n"
            f"L=[{facts['L_min']}, {facts['L_max']}]\n"
            f"K median={facts['K_median']:.6g}\n"
            f"overlap=[{facts['olap_min']:.6g}, {facts['olap_max']:.6g}]\n"
            f"K=1 bins={facts['n_k1']}\n"
            f"L at bound={facts['n_L_at_bound']}\n"
        )
        return facts

    def to_json(self) -> str:
        """JSON metadata with float arrays stored as little-endian hex.

        Hex keeps the SHA-256 digest stable. Decimal JSON would round
        float64 values.

        Returns
        -------
        str
            A JSON object. Floats are hex of their little-endian bytes.
        """
        payload = {
            "schema": SCHEMA,
            "scheduler": self.scheduler,
            "window": self.window,
            "missing": self.missing,
            "psd": PSD_CONVENTION,
            "csd": CSD_CONVENTION,
            "fs": _f64(self.fs).hex(),
            "olap": _f64(self.olap).hex(),
            "bmin": _f64(self.bmin).hex(),
            "psll": None if self.psll is None else _f64(self.psll).hex(),
            "N": int(self.N),
            "Lmin": int(self.Lmin),
            "Jdes": int(self.Jdes),
            "Kdes": int(self.Kdes),
            "order": int(self.order),
            "band": None
            if self.band is None
            else [_f64(self.band[0]).hex(), _f64(self.band[1]).hex()],
            "nf": int(self.f.shape[0]),
            "f": np.ascontiguousarray(self.f, dtype="<f8").tobytes().hex(),
            "r": np.ascontiguousarray(self.r, dtype="<f8").tobytes().hex(),
            "b": np.ascontiguousarray(self.b, dtype="<f8").tobytes().hex(),
            "L": np.ascontiguousarray(self.L, dtype="<i8").tobytes().hex(),
            "K": np.ascontiguousarray(self.K, dtype="<i8").tobytes().hex(),
            "navg": np.ascontiguousarray(self.navg, dtype="<i8").tobytes().hex(),
            "O": np.ascontiguousarray(self.O, dtype="<f8").tobytes().hex(),
            "D": [
                np.ascontiguousarray(row, dtype="<i8").tobytes().hex() for row in self.D
            ],
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> SpectralPlan:
        """Rebuild a plan from ``to_json``.

        Parameters
        ----------
        text : str
            JSON produced by ``to_json``.

        Returns
        -------
        SpectralPlan
            Plan with the same ``digest`` as the one that was serialized.

        Raises
        ------
        ValueError
            If the schema or the PSD or CSD convention string is not the
            one this module writes.
        """
        payload = json.loads(text)
        if int(payload["schema"]) != SCHEMA:
            raise ValueError(
                f"SpectralPlan schema {payload['schema']!r} is not {SCHEMA}."
            )
        if payload["psd"] != PSD_CONVENTION or payload["csd"] != CSD_CONVENTION:
            raise ValueError(
                "JSON PSD or CSD convention is not the one-sided WOSA record."
            )
        band = payload["band"]
        if band is not None:
            band = (_hex_f64(band[0]), _hex_f64(band[1]))
        plan = {
            "f": _hex_array(payload["f"], np.float64),
            "r": _hex_array(payload["r"], np.float64),
            "b": _hex_array(payload["b"], np.float64),
            "L": _hex_array(payload["L"], np.int64),
            "K": _hex_array(payload["K"], np.int64),
            "navg": _hex_array(payload["navg"], np.int64),
            "O": _hex_array(payload["O"], np.float64),
            "D": [_hex_array(row, np.int64) for row in payload["D"]],
            "nf": int(payload["nf"]),
        }
        psll = payload["psll"]
        return cls.from_dict(
            plan,
            fs=_hex_f64(payload["fs"]),
            N=int(payload["N"]),
            olap=_hex_f64(payload["olap"]),
            bmin=_hex_f64(payload["bmin"]),
            Lmin=int(payload["Lmin"]),
            Jdes=int(payload["Jdes"]),
            Kdes=int(payload["Kdes"]),
            order=int(payload["order"]),
            scheduler=str(payload["scheduler"]),
            window=str(payload["window"]),
            missing=str(payload["missing"]),
            psll=None if psll is None else _hex_f64(psll),
            band=band,
        )

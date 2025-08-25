# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

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
import time
import logging
from typing import List, Dict, Any, Union, Callable, Optional, Tuple

import numpy as np
from numpy import kaiser as np_kaiser
from scipy.signal.windows import kaiser as sp_kaiser
import pandas as pd
import control as ct
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from speckit.flattop import olap_dict, win_dict
from speckit.dsp import integral_rms, polynomial_detrend
from speckit.schedulers import lpsd_plan, ltf_plan, new_ltf_plan
from speckit.utils import (
    kaiser_alpha,
    kaiser_rov,
    round_half_up,
    is_function_in_dict,
    get_key_for_function,
    find_Jdes_binary_search,
)
from .core import (_NUMBA_ENABLED,
    _build_Q,
    _stats_detrend0_auto, 
    _stats_detrend0_csd,
    _stats_poly_auto_np,
    _stats_poly_csd_np,
    _stats_win_only_auto,
    _stats_win_only_csd,
    _stats_poly_auto,
    _stats_poly_csd,
)


class SpectrumAnalyzer:
    """
    Configures and executes a high-resolution spectral analysis task.

    This class serves as the main configuration object for the speckit library.
    It takes time-series data and a rich set of parameters to define how the
    spectral analysis should be performed. The heavy computation is deferred
    until the `.compute()` method is called.
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: float,
        *,
        olap: Union[str, float] = "default",
        bmin: float = 1.0,
        Lmin: int = 1,
        Jdes: int = 500,
        Kdes: int = 100,
        num_patch_pts: Optional[int] = 50,
        order: int = 0,
        psll: Optional[float] = 200,
        win: Union[str, Callable] = np_kaiser,
        scheduler: Union[str, Callable] = "ltf",
        band: Optional[Tuple[float, float]] = None,
        force_target_nf: Optional[bool] = False,
        verbose: bool = False,
    ):
        """
        Initializes the spectral analyzer.

        Parameters
        ----------
        data : np.ndarray
            Input time-series. A 1D array for auto-spectral analysis or a
            2D (2xN or Nx2) array for cross-spectral analysis.
        fs : float
            The sampling frequency of the data in Hz (must be > 0).
        olap : str or float, optional
            Desired fractional overlap between segments. Use "default" to
            automatically select an optimal value based on the window function.
            Defaults to "default".
        bmin : float, optional
            Minimum fractional bin number to use, discarding lower bins which may
            be biased. Defaults to 1.0.
        Lmin : int, optional
            The smallest allowable segment length. Useful for mitigating
            time-delay bias in cross-spectra. Defaults to 1.
        Jdes : int, optional
            The desired number of frequency bins in the final spectrum. This is a
            target, and the actual number may vary. Defaults to 500.
        Kdes : int, optional
            The desired number of segments to average. This is a control
            parameter for the scheduler. Defaults to 100.
        num_patch_pts: int, optional
            [new_ltf_plan] Number of lower frequencies to prepend 
            to new_ltf_plan scheduler output. Defaults to 50. 
        order : int, optional
            Order of polynomial detrending to apply to each segment.
            Allowed values: -1 (window only), 0 (mean removal), 1 (linear), 2 (quadratic).
            Defaults to 0.
        psll : float, optional
            Peak Side-Lobe Level in dB for the Kaiser window. Required if
            `win` is 'kaiser'. Defaults to 200.
        win : str or callable, optional
            Window function to use. Can be a string name (e.g., 'kaiser', 'hann')
            or a callable function. Defaults to `np.kaiser`.
        scheduler : str or callable, optional
            The scheduling algorithm to use ('lpsd', 'ltf', 'new_ltf') or a
            custom scheduler function. Defaults to 'ltf'.
        band : tuple of (float, float), optional
            Frequency band `(f_min, f_max)` to restrict the analysis to.
            Defaults to None (full band).
        force_target_nf : bool, optional
            If True, performs a search to find the `Jdes` value that
            produces a plan with this target number of frequency bins.
            Defaults to False.
        verbose : bool, optional
            If True, prints progress and diagnostic information. Defaults to False.
        """
        # ---- Basic validation -------------------------------------------------
        if not np.isfinite(fs) or fs <= 0:
            raise ValueError(f"`fs` must be a positive finite float, got {fs!r}.")
        if order not in (-1, 0, 1, 2):
            raise ValueError(f"`order` must be one of {{-1, 0, 1, 2}}, got {order!r}.")

        self.fs = float(fs)
        self.verbose = bool(verbose)

        self.config = {
            "olap": olap,
            "bmin": float(bmin),
            "Lmin": int(Lmin),
            "Jdes": int(Jdes),
            "Kdes": int(Kdes),
            "num_patch_pts": None if num_patch_pts is None else int(num_patch_pts),
            "order": int(order),
            "psll": psll,
            "win": win,
            "scheduler": scheduler,
            "band": band,
            "force_target_nf": bool(force_target_nf),
        }

        # --- Process and validate input data ---
        x = np.asarray(data)
        if x.ndim == 2 and (x.shape[0] == 2 or x.shape[1] == 2):
            self.iscsd = True
            # Ensure canonical shape == (2, N)
            if x.shape[0] == 2 and x.shape[1] != 2:
                data_2n = x
            elif x.shape[1] == 2 and x.shape[0] != 2:
                data_2n = x.T
            else:
                # both dims are 2 (2x2) or ambiguous; prefer channels-first
                data_2n = x if x.shape[0] == 2 else x.T
            self.data = np.ascontiguousarray(data_2n, dtype=np.float64)  # (2, N)
            self.x1 = self.data[0]
            self.x2 = self.data[1]
            N = self.data.shape[1]
            if self.verbose:
                logging.info(f"Detected two-channel data with N={N}")
        elif x.ndim == 1:
            self.iscsd = False
            self.data = np.ascontiguousarray(x, dtype=np.float64)       # (N,)
            self.x1 = self.data
            self.x2 = None
            N = self.data.shape[0]
            if self.verbose:
                logging.info(f"Detected single-channel data with N={N}")
        else:
            raise ValueError("Input data must be a 1D array or a 2xN/Nx2 array.")

        # Warn (don’t fail) on NaN/Inf in input
        if not np.all(np.isfinite(self.data)):
            logging.warning("Input data contains NaN/Inf; results may be undefined.")

        self.nx = int(len(self.x1))
        self.config["N"] = self.nx

        # Resolve dependent config
        self._process_window_config()
        self._process_scheduler_config()

        # Cache for the plan
        self._plan_cache: Optional[Dict[str, Any]] = None

        # Concise summary (verbose only)
        if self.verbose:
            win_func = self.config.get("win_func")
            alpha = self.config.get("alpha")
            if win_func in (np_kaiser, sp_kaiser):
                win_name = f"kaiser(alpha={alpha:.3f})" if alpha is not None else "kaiser"
            elif is_function_in_dict(win_func, win_dict):
                win_name = get_key_for_function(win_func, win_dict)
            else:
                win_name = getattr(win_func, "__name__", "custom_win")

            sched_param = self.config["scheduler"]
            sched_name = sched_param if isinstance(sched_param, str) else getattr(sched_param, "__name__", "custom_sched")

            logging.info(
                f"SpectrumAnalyzer: fs={self.fs:g} Hz | N={self.nx} | "
                f"mode={'CSD' if self.iscsd else 'auto'} | order={self.config['order']} | "
                f"win={win_name} | olap={self.config.get('final_olap','?')} | "
                f"scheduler={sched_name}"
            )


    def _process_window_config(self):
        """Resolve window function, Kaiser alpha, and default overlap."""
        win_param = self.config["win"]
        psll = self.config["psll"]

        # 1) Resolve window function and human-readable name
        win_name = "custom_win"
        if isinstance(win_param, str):
            w = win_param.lower()
            win_name = w
            if w == "kaiser":
                self.config["win_func"] = np_kaiser
                if psll is None:
                    raise ValueError("PSLL must be specified for the Kaiser window.")
                self.config["alpha"] = kaiser_alpha(psll)
            elif w in ("hann", "hanning"):
                self.config["win_func"] = np.hanning
                self.config["alpha"] = None
            elif win_param in win_dict:
                self.config["win_func"] = win_dict[win_param]
                self.config["alpha"] = None
            else:
                raise ValueError(f"Window function '{win_param}' not recognized.")
        elif callable(win_param):
            # Normalize scipy's kaiser to numpy's for consistent API (expects beta*pi)
            if win_param in (np_kaiser, sp_kaiser):
                self.config["win_func"] = np_kaiser
                win_name = "kaiser"
                if psll is None:
                    raise ValueError("PSLL must be specified for the Kaiser window.")
                self.config["alpha"] = kaiser_alpha(psll)
            else:
                self.config["win_func"] = win_param
                # Try to recover a friendly name if it's one of ours
                if is_function_in_dict(win_param, win_dict):
                    win_name = get_key_for_function(win_param, win_dict)
                self.config["alpha"] = None
        else:
            raise TypeError("Window must be a recognized string or a callable function.")

        # 2) Resolve automatic overlap based on the selected window
        olap_req = self.config["olap"]
        if olap_req == "default":
            if self.config["win_func"] is np_kaiser:
                alpha = self.config.get("alpha", None)
                if alpha is None:
                    raise RuntimeError("Internal error: Kaiser 'alpha' missing.")
                self.config["final_olap"] = float(kaiser_rov(alpha))
            elif win_name in olap_dict:
                self.config["final_olap"] = float(olap_dict[win_name])
            else:
                if self.verbose:
                    logging.warning(
                        f"Optimal overlap for window '{win_name}' not found; defaulting to 0.5."
                    )
                self.config["final_olap"] = 0.5
        else:
            # Validate user-provided overlap
            try:
                olap_val = float(olap_req)
            except Exception as exc:
                raise TypeError(
                    f"`olap` must be 'default' or a float in [0,1); got {olap_req!r}"
                ) from exc
            if not np.isfinite(olap_val) or not (0.0 <= olap_val < 1.0):
                raise ValueError(f"`olap` must be in [0, 1); got {olap_val!r}")
            self.config["final_olap"] = olap_val

        # Save a friendly window name for logging/debugging
        self.config["win_name"] = win_name


    def _process_scheduler_config(self):
        """Resolve scheduler function and store a friendly name."""
        scheduler_param = self.config["scheduler"]
        if isinstance(scheduler_param, str):
            sched_map = {"lpsd": lpsd_plan, "ltf": ltf_plan, "new_ltf": new_ltf_plan}
            try:
                self.config["scheduler_func"] = sched_map[scheduler_param]
            except KeyError as exc:
                raise ValueError(
                    f"Scheduler '{scheduler_param}' not recognized. "
                    f"Available: {list(sched_map.keys())}"
                ) from exc
            self.config["scheduler_name"] = scheduler_param
        elif callable(scheduler_param):
            self.config["scheduler_func"] = scheduler_param
            self.config["scheduler_name"] = getattr(scheduler_param, "__name__", "custom_sched")
        else:
            raise TypeError("Scheduler must be a recognized string or a callable function.")


    def plan(self) -> Dict[str, Any]:
        """
        Generates, caches, and returns the computation plan.

        The plan is a dictionary detailing the segmentation (lengths, overlaps)
        and frequency bins for the analysis. It is generated on the first call
        and cached for subsequent calls unless parameters are changed.

        Returns
        -------
        dict
            A dictionary containing the full computation plan, including arrays
            for frequencies (`f`), segment lengths (`L`), number of averages (`K`),
            per-frequency segment starts (`D`), actual overlaps (`O`), etc.
        """
        if self._plan_cache is not None:
            return self._plan_cache

        scheduler_func = self.config["scheduler_func"]
        sched_name = self.config.get("scheduler_name", getattr(scheduler_func, "__name__", "scheduler"))

        # ---- Common kwargs for any scheduler (tolerates extra keys) ------------
        common_kwargs = dict(
            N=self.nx,
            fs=self.fs,
            olap=self.config["final_olap"],
            bmin=self.config["bmin"],
            Lmin=self.config["Lmin"],
            Kdes=self.config["Kdes"],
        )

        # Some schedulers accept 'num_patch_pts' (new_ltf). Detect by name.
        if getattr(scheduler_func, "__name__", "") == "new_ltf_plan":
            common_kwargs["num_patch_pts"] = self.config["num_patch_pts"]

        # ---- Optional: force target nf via Jdes search --------------------------
        if self.config["force_target_nf"]:
            if self.verbose:
                logging.info(f"[plan] Adjusting Jdes to target nf={self.config['Jdes']} using {sched_name}...")
            target_nf = int(self.config["Jdes"])
            solved_Jdes = find_Jdes_binary_search(scheduler_func, target_nf, **common_kwargs)
            if solved_Jdes is None:
                raise RuntimeError("Failed to generate plan with forced number of frequencies.")
            self.config["Jdes"] = int(solved_Jdes)

        # ---- Generate plan ------------------------------------------------------
        call_kwargs = dict(common_kwargs, Jdes=int(self.config["Jdes"]))
        plan_output = scheduler_func(**call_kwargs)

        # ---- Validate scheduler output ------------------------------------------
        REQUIRED = ("f", "r", "b", "L", "K", "navg", "D", "O")
        for k in REQUIRED:
            if k not in plan_output:
                raise ValueError(f"Scheduler output missing key '{k}'.")

        # Ensure arrays have consistent lengths
        lens = [len(plan_output[k]) for k in ("f", "r", "b", "L", "K", "navg", "O")]
        if not all(L == lens[0] for L in lens):
            raise ValueError("Scheduler arrays have inconsistent lengths: "
                            + ", ".join(f"{name}={len(plan_output[name])}" for name in ("f","r","b","L","K","navg","O")))
        if not isinstance(plan_output["D"], (list, tuple)) or len(plan_output["D"]) != lens[0]:
            raise ValueError("Scheduler 'D' must be a list of per-frequency start arrays with matching length.")

        nf = int(lens[0])
        plan_output["nf"] = nf

        # ---- Type & contiguity normalization -----------------------------------
        # 1D arrays
        plan_output["f"]    = np.ascontiguousarray(plan_output["f"], dtype=np.float64)
        plan_output["r"]    = np.ascontiguousarray(plan_output["r"], dtype=np.float64)
        plan_output["b"]    = np.ascontiguousarray(plan_output["b"], dtype=np.float64)
        plan_output["L"]    = np.ascontiguousarray(plan_output["L"], dtype=np.int64)
        plan_output["K"]    = np.ascontiguousarray(plan_output["K"], dtype=np.int64)
        plan_output["navg"] = np.ascontiguousarray(plan_output["navg"], dtype=np.int64)
        plan_output["O"]    = np.ascontiguousarray(plan_output["O"], dtype=np.float64)

        # 'D' is ragged: normalize each to int64 arrays and validate bounds
        D_norm = []
        N = self.nx
        for i, d in enumerate(plan_output["D"]):
            arr = np.asarray(d, dtype=np.int64)
            if arr.ndim != 1:
                raise ValueError(f"D[{i}] must be 1D, got shape {arr.shape}.")
            L_i = int(plan_output["L"][i])
            if L_i < self.config["Lmin"] or L_i < 1:
                raise ValueError(f"L[{i}]={L_i} invalid (min {self.config['Lmin']}).")
            # bounds: 0 <= start <= N-L
            if arr.size == 0:
                raise ValueError(f"D[{i}] is empty; scheduler produced zero segments for bin {i}.")
            if (arr < 0).any() or (arr > (N - L_i)).any():
                raise ValueError(f"D[{i}] contains out-of-bounds starts for L={L_i} and N={N}.")
            # K / navg sanity
            K_i = int(plan_output["K"][i])
            if K_i != arr.size:
                raise ValueError(f"K[{i}]={K_i} does not match number of starts D[{i}]={arr.size}.")
            D_norm.append(arr)
        plan_output["D"] = D_norm  # keep as list of np.ndarray[int64]

        # ---- Optional band-pass on f -------------------------------------------
        if self.config["band"] is not None:
            fmin, fmax = self.config["band"]
            if not (np.isfinite(fmin) and np.isfinite(fmax) and fmax >= fmin):
                raise ValueError(f"Invalid band (fmin, fmax)={self.config['band']!r}.")
            mask = (plan_output["f"] >= fmin) & (plan_output["f"] <= fmax)
            if not np.any(mask):
                raise ValueError("No frequencies found in the specified band.")
            # apply mask to vector fields
            for key in ("f", "r", "b", "L", "K", "navg", "O"):
                plan_output[key] = plan_output[key][mask]
            # D is ragged -> filter by mask
            plan_output["D"] = [d for d, keep in zip(D_norm, mask) if keep]
            plan_output["nf"] = int(plan_output["f"].shape[0])

        # ---- Final sanity: nf coherence ----------------------------------------
        nf2 = int(plan_output["f"].shape[0])
        if nf2 != len(plan_output["D"]):
            raise RuntimeError("Internal plan error: length mismatch after band filter.")
        if self.verbose:
            logging.info(f"[plan] {sched_name}: nf={nf2}, "
                        f"f∈[{plan_output['f'][0]:.6g}, {plan_output['f'][-1]:.6g}] Hz, "
                        f"L∈[{int(np.min(plan_output['L']))}, {int(np.max(plan_output['L']))}], "
                        f"K median={int(np.median(plan_output['K']))}")

        self._plan_cache = plan_output
        return self._plan_cache


    def compute_single_bin(
        self, freq: float, *, fres: Optional[float] = None, L: Optional[int] = None
    ) -> "SpectrumResult":
        """
        Executes spectral analysis for a single, user-defined frequency bin.

        This method is optimized for calculating spectral quantities at one
        specific frequency, defined by either a frequency resolution (`fres`)
        or a segment length (`L`). Returns a SpectrumResult where all arrays
        have length 1 (single-bin).

        Notes
        -----
        - Cross-spectrum sign convention matches SciPy: Pxy = X * conj(Y)
        (i.e., Im{Pxy} = i_x*r_y - r_x*i_y).
        """
        import numpy as _np
        import time as _time

        # -------- Basic validation ----------
        if not _np.isfinite(freq) or freq < 0:
            raise ValueError(f"`freq` must be a finite, non-negative float. Got {freq!r}.")
        if freq > self.fs * 0.5 + 1e-12:
            logging.warning(
                f"Requested freq={freq:g} Hz exceeds Nyquist ({self.fs/2:g} Hz). "
                "Proceeding, but results may be meaningless in one-sided interpretation."
            )

        # -------- Segment length & resolution ----------
        if L is not None and fres is not None:
            raise ValueError("Provide only one of `fres` or `L`, not both.")
        if L is not None:
            segL = int(L)
            if segL < 1:
                raise ValueError(f"Invalid segment length segL={segL}.")
            final_fres = float(self.fs) / segL
        elif fres is not None:
            if not _np.isfinite(fres) or fres <= 0:
                raise ValueError(f"`fres` must be a positive finite float. Got {fres!r}.")
            final_fres = float(fres)
            segL = int(round(float(self.fs) / final_fres))
            if segL < 1:
                segL = 1
                final_fres = float(self.fs) / segL
        else:
            raise ValueError("Provide either `fres` or `L`.")

        if segL > self.nx:
            raise ValueError(f"Invalid segment length segL={segL} for N={self.nx}.")

        # -------- Segmentation (starts) ----------
        if self.nx == segL:
            navg = 1
            starts = _np.array([0], dtype=_np.int64)
        else:
            olap = float(self.config["final_olap"])
            navg = int(
                round_half_up(((self.nx - segL) / (1.0 - olap)) / segL + 1.0)
            )
            if navg <= 1:
                navg = 1
                starts = _np.array([0], dtype=_np.int64)
            else:
                shift = (self.nx - segL) / (navg - 1)
                starts = _np.round(_np.arange(navg) * shift).astype(_np.int64)

        # bounds check to convert segfaults into clear errors
        if (starts < 0).any() or (starts > (self.nx - segL)).any():
            raise ValueError(
                f"Computed segment starts out of bounds for segL={segL}, N={self.nx}."
            )

        # -------- Window (helper) ----------
        def _build_window(win_func, Lint: int, alpha_val):
            if win_func in (np_kaiser, sp_kaiser):
                if alpha_val is None:
                    raise ValueError("Kaiser window selected but 'alpha' is not set.")
                wloc = win_func(Lint + 1, alpha_val * _np.pi)[:-1]
            else:
                wloc = win_func(Lint)
            wloc = _np.asarray(wloc, dtype=_np.float64)
            if wloc.shape[0] != Lint:
                raise ValueError(f"Window length {wloc.shape[0]} != L {Lint}")
            S1loc = float(_np.sum(wloc))
            S2loc = float(_np.sum(wloc * wloc))
            return wloc, S1loc, S2loc

        win_func = self.config["win_func"]
        alpha = self.config.get("alpha", None)
        w, S1, S2 = _build_window(win_func, segL, alpha)

        # -------- Inputs for kernels ----------
        x1 = _np.ascontiguousarray(self.x1, dtype=_np.float64)
        x2 = _np.ascontiguousarray(self.x2, dtype=_np.float64) if self.iscsd else None
        is_cross = self.iscsd
        order = int(self.config.get("order", 0))
        omega = 2.0 * _np.pi * float(freq) / float(self.fs)

        # -------- Select kernel set ----------
        from .core import (
            _build_Q,
            _stats_win_only_auto, _stats_win_only_csd,
            _stats_detrend0_auto, _stats_detrend0_csd,
            _stats_poly_auto, _stats_poly_csd,
        )

        if order == -1:
            detrend_mode = "win"
        elif order == 0:
            detrend_mode = "mean0"
        elif order in (1, 2):
            detrend_mode = "poly"
        else:
            raise ValueError("order must be one of {-1, 0, 1, 2}.")

        Q = None
        if detrend_mode == "poly":
            Q = _build_Q(segL, order).astype(_np.float64, copy=False)

        # -------- Run kernel ----------
        t0 = _time.perf_counter()
        if detrend_mode == "win":
            if is_cross:
                MXX, MYY, mu_r, mu_i, M2 = _stats_win_only_csd(x1, x2, starts, segL, w, omega)
            else:
                MXX, MYY, mu_r, mu_i, M2 = _stats_win_only_auto(x1, starts, segL, w, omega)
        elif detrend_mode == "mean0":
            if is_cross:
                MXX, MYY, mu_r, mu_i, M2 = _stats_detrend0_csd(x1, x2, starts, segL, w, omega)
            else:
                MXX, MYY, mu_r, mu_i, M2 = _stats_detrend0_auto(x1, starts, segL, w, omega)
        else:  # poly
            if is_cross:
                MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd(x1, x2, starts, segL, w, omega, Q)
            else:
                MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto(x1, starts, segL, w, omega, Q)
        tm = _time.perf_counter() - t0

        # -------- Package SpectrumResult ----------
        m = float(freq) / final_fres
        XY = complex(mu_r, mu_i)

        single_bin_results = {
            "f": _np.array([freq], dtype=_np.float64),
            "r": _np.array([final_fres], dtype=_np.float64),
            "b": _np.array([m], dtype=_np.float64),
            "L": _np.array([segL], dtype=_np.int64),
            "K": _np.array([int(starts.shape[0])], dtype=_np.int64),
            "navg": _np.array([int(starts.shape[0])], dtype=_np.int64),
            "D": _np.array([starts], dtype=object),
            "O": _np.array([self.config["final_olap"]], dtype=_np.float64),
            "i": _np.array([0], dtype=_np.int64),
            "XX": _np.array([float(MXX)], dtype=_np.float64),
            "YY": _np.array([float(MYY)], dtype=_np.float64),
            "XY": _np.array([XY], dtype=_np.complex128),
            "S12": _np.array([S1 * S1], dtype=_np.float64),
            "S2": _np.array([S2], dtype=_np.float64),
            "M2": _np.array([float(M2)], dtype=_np.float64),
            "compute_t": _np.array([float(tm)], dtype=_np.float64),
        }

        return SpectrumResult(single_bin_results, self.config, self.iscsd, self.fs)


    def compute(self) -> "SpectrumResult":
        """
        Executes the spectral analysis and returns a SpectrumResult object.

        This method performs the core computation. It uses the generated plan
        to segment the data, apply windowing/Goertzel reductions, and average
        the results.

        Returns
        -------
        SpectrumResult
            An object containing all computed spectral quantities and helper methods.
        """
        plan = self.plan()
        nf = int(plan["nf"])

        if self.verbose:
            logging.info(f"Computing {nf} frequencies...")

        t0 = time.perf_counter()
        # Single chunk path (kept simple & explicit)
        results_list = [self._lpsd_core(np.arange(nf, dtype=np.int64))]
        t_total = time.perf_counter() - t0

        if self.verbose:
            logging.info(f"Computation completed in {t_total:.2f} seconds.")

        # Collect into contiguous arrays
        XX  = np.empty(nf, dtype=np.float64)
        YY  = np.empty(nf, dtype=np.float64)
        XY  = np.empty(nf, dtype=np.complex128)
        S12 = np.empty(nf, dtype=np.float64)
        S2  = np.empty(nf, dtype=np.float64)
        M2  = np.empty(nf, dtype=np.float64)
        tms = np.empty(nf, dtype=np.float64)

        for chunk in results_list:
            for (i, xy, mxx, myy, s12, s2, m2, tm) in chunk:
                XX[int(i)]  = mxx
                YY[int(i)]  = myy
                XY[int(i)]  = xy
                S12[int(i)] = s12
                S2[int(i)]  = s2
                M2[int(i)]  = m2
                tms[int(i)] = tm

        final_results = {
            **plan,
            "XX": XX, "YY": YY, "XY": XY,
            "S12": S12, "S2": S2, "M2": M2,
            "compute_t": tms,
        }
        return SpectrumResult(final_results, self.config, self.iscsd, self.fs)


    def _lpsd_core(self, f_indices: np.ndarray) -> List[Any]:
        """
        Core processing loop for a block of frequency indices.

        Notes
        -----
        - Cross-spectrum sign convention matches SciPy: Pxy = X * conj(Y)
        (Im{Pxy} = i_x * r_y - r_x * i_y).
        - Uses per-L window cache and per-(L,order) Q cache.
        """
        plan = self._plan_cache  # plan() already validated & cached before calling
        assert plan is not None

        results_block: List[Any] = []

        # Per-L window cache: L -> (w, S1, S2)
        window_cache: Dict[int, Tuple[np.ndarray, float, float]] = {}
        # Per-(L,order) Q cache
        Q_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Contiguous, typed views for kernels
        x1 = np.ascontiguousarray(self.x1, dtype=np.float64)
        x2 = np.ascontiguousarray(self.x2, dtype=np.float64) if self.iscsd else None

        win_func = self.config["win_func"]
        alpha    = self.config.get("alpha", None)
        order    = int(self.config["order"])
        fs       = float(self.fs)
        N        = int(self.nx)

        # Import kernels & fallbacks once
        from .core import (
            _build_Q,
            _stats_win_only_auto, _stats_win_only_csd,
            _stats_detrend0_auto, _stats_detrend0_csd,
            _stats_poly_auto, _stats_poly_csd,
            _stats_poly_auto_np, _stats_poly_csd_np,
        )

        def _build_window(L: int) -> Tuple[np.ndarray, float, float]:
            """Return (w, S1, S2) for given L with current window config."""
            if L in window_cache:
                return window_cache[L]
            if win_func in (np_kaiser, sp_kaiser):
                if alpha is None:
                    raise RuntimeError("Kaiser window selected but 'alpha' is not set.")
                w = win_func(L + 1, alpha * np.pi)[:-1]
            else:
                w = win_func(L)
            w = np.ascontiguousarray(w, dtype=np.float64)
            if w.shape[0] != L:
                raise ValueError(f"Window length {w.shape[0]} != L {L}.")
            S1 = float(np.sum(w))
            S2 = float(np.sum(w * w))
            window_cache[L] = (w, S1, S2)
            return window_cache[L]

        for i in f_indices:
            i = int(i)
            t0 = time.perf_counter()

            L      = int(plan["L"][i])
            starts = plan["D"][i]  # np.ndarray[int64] (validated in plan())
            # Optional extra guard (keeps crashes friendly if external schedulers are used)
            if (starts < 0).any() or (starts > (N - L)).any():
                raise ValueError(f"D[{i}] contains out-of-bounds starts for L={L}, N={N}.")

            w, S1, S2 = _build_window(L)

            # Use frequency in Hz directly (clearer than b/L; both equivalent)
            f_i   = float(plan["f"][i])
            omega = 2.0 * np.pi * f_i / fs

            # Choose & run kernel
            if order == -1:
                if self.iscsd:
                    if _NUMBA_ENABLED:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_win_only_csd(x1, x2, starts, L, w, omega)
                    else:
                        # Fallback via poly_np with a degenerate basis (no detrend effect)
                        Q = np.zeros((L, 1), dtype=np.float64)
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q)
                else:
                    if _NUMBA_ENABLED:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_win_only_auto(x1, starts, L, w, omega)
                    else:
                        Q = np.zeros((L, 1), dtype=np.float64)
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto_np(x1, starts, L, w, omega, Q)

            elif order == 0:
                if self.iscsd:
                    if _NUMBA_ENABLED:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_detrend0_csd(x1, x2, starts, L, w, omega)
                    else:
                        Q = np.zeros((L, 1), dtype=np.float64)
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q)
                else:
                    if _NUMBA_ENABLED:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_detrend0_auto(x1, starts, L, w, omega)
                    else:
                        Q = np.zeros((L, 1), dtype=np.float64)
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto_np(x1, starts, L, w, omega, Q)

            elif order in (1, 2):
                key = (L, order)
                Q = Q_cache.get(key)
                if Q is None:
                    Q = _build_Q(L, order)
                    Q_cache[key] = Q
                if self.iscsd:
                    if _NUMBA_ENABLED:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd(x1, x2, starts, L, w, omega, Q)
                    else:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q)
                else:
                    if _NUMBA_ENABLED:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto(x1, starts, L, w, omega, Q)
                    else:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto_np(x1, starts, L, w, omega, Q)
            else:
                raise ValueError(f"Unsupported detrend order: {order}.")

            XY = complex(mu_r, mu_i)
            elapsed = time.perf_counter() - t0
            # Append tuple in the order consumed by compute()
            results_block.append([
                i,
                XY,
                float(MXX),
                float(MYY),
                float(S1 * S1),
                float(S2),
                float(M2),
                float(elapsed),
            ])

        return results_block


class SpectrumResult:
    """
    An immutable container for the results of a spectral analysis.

    This object holds all computed spectral quantities, such as PSDs, CSDs,
    coherence, and transfer functions, along with their associated uncertainties.
    It provides convenient properties for access and methods for plotting and
    further analysis.

    Attributes
    ----------
    f : np.ndarray
        Array of Fourier frequencies in Hz.
    psd : np.ndarray or None
        One-sided Power Spectral Density (auto-spectrum).
    asd : np.ndarray or None
        One-sided Amplitude Spectral Density (auto-spectrum).
    csd : np.ndarray or None
        One-sided Cross Spectral Density.
    coh : np.ndarray or None
        Magnitude-squared coherence.
    tf : np.ndarray or None
        Complex transfer function estimate.
    Gxx_dev : np.ndarray or None
        Standard deviation of the PSD estimate.
    coh_error : np.ndarray or None
        Normalized random error of the coherence estimate.
    ... and many others. Use tab-completion to explore.
    """

    def __init__(
        self,
        results_dict: Dict[str, Any],
        config_dict: Dict[str, Any],
        iscsd: bool,
        fs: float,
    ):
        """Initializes the result object."""
        self._data = results_dict
        self._config = config_dict
        self.iscsd = iscsd
        self.fs = fs
        self._cache: Dict[str, Any] = {}

        # Ensure all list-based data from dict are numpy arrays
        for key, value in self._data.items():
            if isinstance(value, list):
                if key == "D":
                    # The 'D' key is a list of lists with varying lengths.
                    # Force object dtype to handle this inhomogeneous shape.
                    self._data[key] = np.array(value, dtype=object)
                else:
                    # For all other keys, a standard array conversion is expected.
                    self._data[key] = np.array(value)

    def __getattr__(self, name: str) -> Any:
        """Lazy computation and caching of spectral properties."""
        if name in self._cache:
            return self._cache[name]

        val: Any = None
        # --- Base Quantities ---
        if name == "Gxx":
            val = 2.0 * self._data["XX"] / self.fs / self._data["S2"]
        elif name == "Gyy":
            val = (
                2.0 * self._data["YY"] / self.fs / self._data["S2"]
                if self.iscsd
                else self.Gxx
            )
        elif name == "Gxy":
            val = (
                2.0 * self._data["XY"] / self.fs / self._data["S2"]
                if self.iscsd
                else self.Gxx
            )
        elif name == "ENBW":
            val = self.fs * self._data["S2"] / self._data["S12"]

        # --- Derived Auto-Spectral Quantities ---
        elif name in ["psd", "G", "asd", "ps"]:
            if self.iscsd:
                val = None
            else:
                if name in ["psd", "G"]:
                    val = self.Gxx
                elif name == "asd":
                    val = np.sqrt(self.psd)
                elif name == "ps":
                    val = self.psd * self.ENBW

        # --- Derived Cross-Spectral Quantities ---
        elif name in [
            "csd",
            "Gyx",
            "Hxy",
            "Hyx",
            "coh",
            "ccoh",
            "cs",
            "tf",
            "cf",
            "cf_db",
            "cf_rad",
            "cf_deg",
            "cf_rad_unwrapped",
            "cf_deg_unwrapped",
        ]:
            if not self.iscsd:
                val = None
            else:
                if name == "csd":
                    val = self.Gxy
                elif name == "Gyx":
                    val = np.conj(self.Gxy)
                elif name == "Hxy":
                    val = np.divide(
                        np.conj(self._data["XY"]),
                        self._data["XX"],
                        out=np.zeros_like(self._data["XX"], dtype=complex),
                        where=self._data["XX"] != 0,
                    )
                elif name == "Hyx":
                    val = np.conj(self.Hxy)
                elif name == "coh":
                    val = np.divide(
                        np.abs(self._data["XY"]) ** 2,
                        self._data["XX"] * self._data["YY"],
                        out=np.zeros_like(self._data["XX"]),
                        where=(self._data["XX"] != 0) & (self._data["YY"] != 0),
                    )
                elif name == "ccoh":
                    val = np.divide(
                        self._data["XY"],
                        np.sqrt(self._data["XX"] * self._data["YY"]),
                        out=np.zeros_like(self._data["XX"], dtype=complex),
                        where=(self._data["XX"] != 0) & (self._data["YY"] != 0),
                    )
                elif name == "cs":
                    val = self.csd * self.ENBW
                elif name == "tf":
                    val = self.Hxy
                elif name == "cf":
                    val = np.abs(self.Hxy)
                elif name == "cf_db":
                    val = ct.mag2db(self.cf)
                elif name == "cf_rad":
                    val = np.angle(self.Hxy)
                elif name == "cf_deg":
                    val = np.angle(self.Hxy, deg=True)
                elif name == "cf_rad_unwrapped":
                    val = np.unwrap(self.cf_rad)
                elif name == "cf_deg_unwrapped":
                    val = np.rad2deg(self.cf_rad_unwrapped)

        # --- Conditional Spectra ---
        elif name in ["GyyCx", "GyyRx", "GyySx"]:
            if not self.iscsd:
                val = None
            else:
                if name == "GyyCx":
                    val = self.coh * self.Gyy
                elif name == "GyyRx":
                    val = (1 - self.coh) * self.Gyy
                elif name == "GyySx":
                    val = np.abs(
                        self.Gyy
                        + self.Hxy * self.Hyx * self.Gxx
                        - self.Hyx * self.Gxy
                        - self.Hxy * self.Gyx
                    )

        # --- Errors and Deviations ---
        elif name.endswith(("_dev", "_error")):
            navg = self._data["navg"]
            coh = self.coh if self.iscsd else np.ones_like(navg)

            # Standard Deviations
            if name == "Gxx_dev":
                val = self.Gxx / np.sqrt(navg)
            elif name == "Gyy_dev":
                val = self.Gyy / np.sqrt(navg) if self.iscsd else self.Gxx_dev
            elif name == "Hxy_dev":
                val = (
                    np.abs(self.Hxy)
                    * np.sqrt(np.abs(1 - coh))
                    / np.sqrt(coh * 2 * navg)
                    if self.iscsd
                    else None
                )
            elif name == "Gxy_dev":
                val = (
                    np.sqrt(np.abs(self.Gxy) ** 2 / coh / navg) if self.iscsd else None
                )
            elif name == "coh_dev":
                val = (
                    np.sqrt(np.abs((2 * coh / navg) * (1 - coh) ** 2))
                    if self.iscsd
                    else None
                )

            # Normalized Random Errors
            elif name == "Gxx_error":
                val = 1 / np.sqrt(navg)
            elif name == "Gyy_error":
                val = 1 / np.sqrt(navg) if self.iscsd else self.Gxx_error
            elif name == "Gxy_error":
                val = 1 / np.sqrt(coh * navg) if self.iscsd else None
            elif name == "Hxy_mag_error":
                val = (
                    np.sqrt(np.abs(1 - coh)) / (np.sqrt(coh * 2 * navg))
                    if self.iscsd
                    else None
                )
            elif name == "Hxy_rad_error":
                val = (
                    np.arcsin(np.sqrt(np.abs(1 - coh))) / np.sqrt(coh * 2 * navg)
                    if self.iscsd
                    else None
                )
            elif name == "Hxy_deg_error":
                val = np.rad2deg(self.Hxy_rad_error) if self.iscsd else None
            elif name == "coh_error":
                val = (
                    np.sqrt(2) * (1 - coh) / (np.sqrt(coh) * np.sqrt(navg))
                    if self.iscsd
                    else None
                )

        elif name in self._data:
            val = self._data[name]
        else:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        self._cache[name] = val
        return val

    def __dir__(self) -> List[str]:
        """Enhances tab-completion to include dynamic attributes."""
        default_attrs = super().__dir__()
        dynamic_attrs = [
            "Gxx",
            "Gyy",
            "Gxy",
            "ENBW",
            "psd",
            "asd",
            "ps",
            "csd",
            "Gyx",
            "Hxy",
            "Hyx",
            "coh",
            "ccoh",
            "cs",
            "tf",
            "cf",
            "cf_db",
            "cf_rad",
            "cf_deg",
            "cf_rad_unwrapped",
            "cf_rad_unwrapped",
            "GyyCx",
            "GyyRx",
            "GyySx",
            "Gxx_dev",
            "Gyy_dev",
            "Gxy_dev",
            "Hxy_dev",
            "coh_dev",
            "Gxx_error",
            "Gyy_error",
            "Gxy_error",
            "Hxy_mag_error",
            "Hxy_rad_error",
            "Hxy_deg_error",
            "coh_error",
        ]
        return sorted(
            list(set(default_attrs + list(self._data.keys()) + dynamic_attrs))
        )

    def get_rms(self, pass_band: Optional[Tuple[float, float]] = None) -> float:
        """
        Computes the Root Mean Square (RMS) of the signal by integrating the ASD.

        Parameters
        ----------
        pass_band : tuple of (float, float), optional
            The frequency band `(f_min, f_max)` over which to compute the RMS.
            If None, the entire frequency range is used. Defaults to None.

        Returns
        -------
        float
            The computed RMS value.
        """
        if self.iscsd:
            raise NotImplementedError(
                "RMS calculation is only available for auto-spectra."
            )
        return integral_rms(self.f, self.asd, pass_band)

    def get_measurement(
        self, freq: Union[float, np.ndarray], which: str
    ) -> Union[float, np.ndarray]:
        """
        Evaluates a spectral quantity at given frequencies via interpolation.

        Parameters
        ----------
        freq : float or np.ndarray
            The frequency or frequencies at which to evaluate the result.
        which : str
            The name of the spectral quantity to retrieve (e.g., 'asd', 'coh').

        Returns
        -------
        float or np.ndarray
            The interpolated value(s) of the spectral quantity.
        """
        target_signal = getattr(self, which)
        if np.iscomplexobj(target_signal):
            real_part = np.interp(freq, self.f, np.real(target_signal))
            imag_part = np.interp(freq, self.f, np.imag(target_signal))
            return real_part + 1j * imag_part
        else:
            return np.interp(freq, self.f, target_signal)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Exports all computed spectral quantities to a pandas DataFrame.

        The DataFrame will be indexed by frequency and will contain columns for
        all available spectral estimates and their uncertainties.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the spectral analysis results.
        """
        df_dict = {"f": self.f}
        for attr in dir(self):
            if (
                not attr.startswith("_")
                and attr not in ["iscsd", "fs"]
                and not callable(getattr(self, attr))
            ):
                try:
                    value = getattr(self, attr)
                    if isinstance(value, np.ndarray) and len(value) == len(self.f):
                        df_dict[attr] = value
                except AttributeError:
                    continue  # Skip attributes that fail to compute
        return pd.DataFrame(df_dict).set_index("f")

    def plot(
        self,
        which: Optional[str] = None,
        *,
        ax: Optional[Axes] = None,
        ylabel: Optional[str] = None,
        dB: bool = False,
        deg: bool = True,
        unwrap: bool = True,
        errors: bool = False,
        sigma: int = 1,
        **kwargs,
    ) -> Tuple[Figure, Union[Axes, Tuple[Axes, Axes]]]:
        """
        A flexible plotting method for various spectral quantities.

        Parameters
        ----------
        which : str, optional
            The quantity to plot. Options include 'asd', 'psd', 'coh', 'csd', 'cf',
            and 'bode'. If None, defaults to 'bode' for cross-spectra and
            'asd' for auto-spectra.
        ax : matplotlib.axes.Axes, optional
            An existing Axes object to plot on. If None, a new Figure and Axes
            are created. Defaults to None.
        ylabel : str, optional
            Custom label for the y-axis. Defaults to None.
        dB : bool, optional
            For 'bode' or 'cf' plots, display magnitude in decibels. Defaults to False.
        deg : bool, optional
            For 'bode' plots, display phase in degrees. If False, uses radians.
            Defaults to True.
        unwrap : bool, optional
            For 'bode' plots, unwrap the phase angle. Defaults to True.
        errors : bool, optional
            If True, plot error bands around the main trace. Defaults to False.
        sigma : int, optional
            The number of standard deviations (sigma) to show in the error band.
            Defaults to 1.
        **kwargs
            Additional keyword arguments passed to the `matplotlib.pyplot.plot` function.

        Returns
        -------
        tuple
            A tuple containing the matplotlib Figure and either a single Axes
            object or a tuple of two Axes objects (for bode plots).
        """
        plot_options = {
            "psd": ("loglog", self.f, self.psd, "Power Spectral Density"),
            "asd": ("loglog", self.f, self.asd, "Amplitude Spectral Density"),
            "coh": ("semilogx", self.f, self.coh, "Coherence"),
            "csd": (
                "loglog",
                self.f,
                np.abs(self.csd) if self.csd is not None else None,
                "|Cross Spectral Density|",
            ),
            "cf": ("loglog", self.f, self.cf, "Coupling Factor Magnitude"),
            "bode": "bode",
        }

        if which is None:
            which = "bode" if self.iscsd else "asd"

        if which not in plot_options:
            raise ValueError(
                f"Plot type '{which}' not recognized. Available options are: {list(plot_options.keys())}"
            )

        # --- Bode Plot Logic ---
        if which == "bode":
            if not self.iscsd:
                raise ValueError("Bode plot is only available for cross-spectra.")
            fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(7, 5))

            mag_data = self.cf_db if dB else self.cf
            plot_func_mag = ax_mag.semilogx if dB else ax_mag.loglog
            plot_func_mag(self.f, mag_data, **kwargs)
            ax_mag.set_ylabel(f"Magnitude {'(dB)' if dB else ''}")
            if errors:
                mag_error = self.Hxy_mag_error
                lower = mag_data * (1 - sigma * mag_error)
                upper = mag_data * (1 + sigma * mag_error)
                if dB:
                    lower, upper = ct.mag2db(np.maximum(1e-15, lower)), ct.mag2db(upper)
                ax_mag.fill_between(
                    self.f,
                    lower,
                    upper,
                    alpha=0.3,
                    label=f"±{sigma}σ",
                    color=kwargs.get("color"),
                )
                ax_mag.legend()

            phase_data_rad = np.unwrap(self.cf_rad) if unwrap else self.cf_rad
            phase_data = np.rad2deg(phase_data_rad) if deg else phase_data_rad
            ax_phase.semilogx(self.f, phase_data, **kwargs)
            ax_phase.set_ylabel(f"Phase {'(deg)' if deg else '(rad)'}")
            if errors:
                phase_error = self.Hxy_deg_error if deg else self.Hxy_rad_error
                ax_phase.fill_between(
                    self.f,
                    phase_data - sigma * phase_error,
                    phase_data + sigma * phase_error,
                    alpha=0.3,
                    label=f"±{sigma}σ",
                    color=kwargs.get("color"),
                )
                ax_phase.legend()

            ax_phase.set_xlabel("Frequency (Hz)")
            fig.tight_layout()
            return fig, (ax_mag, ax_phase)

        # --- Single Axis Plot Logic ---
        else:
            plot_type, x, y, default_label = plot_options[which]
            if y is None:
                raise ValueError(f"'{which}' is not available for this analysis type.")

            fig, ax1 = (ax.get_figure(), ax) if ax is not None else plt.subplots()
            plot_func = getattr(ax1, plot_type)
            plot_func(x, y, **kwargs)
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel(ylabel if ylabel is not None else default_label)

            if errors:
                error_map = {
                    "asd": self.Gxx_dev / 2 / np.sqrt(self.Gxx),
                    "psd": self.Gxx_dev,
                    "coh": self.coh_dev,
                    "csd": self.Gxy_dev,
                    "cf": self.Hxy_dev,
                }
                error_data = error_map.get(which)
                if error_data is not None:
                    ax1.fill_between(
                        x,
                        y - sigma * error_data,
                        y + sigma * error_data,
                        alpha=0.3,
                        label=f"±{sigma}σ",
                        color=kwargs.get("color"),
                    )
                    ax1.legend()

            fig.tight_layout()
            return fig, ax1


def lpsd(
    data: np.ndarray, fs: float, **kwargs
) -> SpectrumResult:
    """Same as compute_spectrum."""
    return compute_spectrum(data, fs, **kwargs)


def compute_spectrum(
    data: np.ndarray, fs: float, **kwargs
) -> SpectrumResult:
    """
    Computes spectral estimates for one or two time-series in a single call.

    This is the primary high-level function for performing spectral analysis.
    It handles configuration, planning, and computation in one step,
    returning a comprehensive result object. It serves as a convenient
    wrapper around the `SpectrumAnalyzer` and `SpectrumResult` classes.

    Parameters
    ----------
    data : np.ndarray
        Input time-series. A 1D array for auto-spectral analysis or a
        2D (2xN or Nx2) array for cross-spectral analysis.
    fs : float
        The sampling frequency of the data in Hz.
    **kwargs :
        Additional keyword arguments to configure the analysis, passed
        directly to the `SpectrumAnalyzer`. Common arguments include:
        - `win` (str or callable): The window function (e.g., 'kaiser').
        - `olap` (str or float): The fractional segment overlap.
        - `Jdes` (int): The desired number of frequency bins.
        - `bmin` (float): Minimum fractional bin number.
        - `Lmin` (int): Minimum segment length.
        - `order` (int): Order of polynomial detrending.
        - `psll` (float): Peak side-lobe level for Kaiser window.
        - `band` (tuple): Frequency band `(f_min, f_max)`.
        - `verbose` (bool): Enable verbose output.

    Returns
    -------
    SpectrumResult
        An object containing all computed spectral quantities and helper methods.
    """
    # 1. Instantiate the analyzer with all provided parameters
    analyzer = SpectrumAnalyzer(data, fs, **kwargs)

    # 2. Immediately call the compute method
    result = analyzer.compute()

    # 3. Return the final result object
    return result


def compute_single_bin(
    data: np.ndarray,
    fs: float,
    freq: float,
    *,
    fres: Optional[float] = None,
    L: Optional[int] = None,
    **kwargs,
) -> SpectrumResult:
    """
    Computes spectral estimates for a single frequency bin in one call.

    This high-level function provides a simple one-line interface for
    single-bin analysis.

    Parameters
    ----------
    data : np.ndarray
        Input time-series data.
    fs : float
        The sampling frequency in Hz.
    freq : float
        The target Fourier frequency in Hz for the analysis.
    fres : float, optional
        The desired frequency resolution in Hz. Either `fres` or `L` must
        be provided.
    L : int, optional
        The desired segment length in samples. Either `fres` or `L` must
        be provided.
    **kwargs :
        Additional keyword arguments for configuration (e.g., `win`, `olap`),
        passed to the `SpectrumAnalyzer`.

    Returns
    -------
    SpectrumResult
        A result object containing the scalar spectral estimates for the bin.
    """
    # 1. Instantiate the analyzer
    analyzer = SpectrumAnalyzer(data, fs, **kwargs)

    # 2. Call the single-bin computation method
    result = analyzer.compute_single_bin(freq=freq, fres=fres, L=L)

    # 3. Return the result
    return result

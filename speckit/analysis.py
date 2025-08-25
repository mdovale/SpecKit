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
    chunker,
    is_function_in_dict,
    get_key_for_function,
    find_Jdes_binary_search,
)
from .core import (_HAS_NUMBA,
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
            The sampling frequency of the data in Hz.
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
            0 for mean removal, 1 for linear, etc. Defaults to 0.
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
        self.fs = fs
        self.verbose = verbose

        self.config = {
            "olap": olap,
            "bmin": bmin,
            "Lmin": Lmin,
            "Jdes": Jdes,
            "Kdes": Kdes,
            "num_patch_pts": num_patch_pts,
            "order": order,
            "psll": psll,
            "win": win,
            "scheduler": scheduler,
            "band": band,
            "force_target_nf": force_target_nf,
        }

        # --- Process and validate input data ---
        x = np.asarray(data)
        if len(x.shape) == 2 and (x.shape[0] == 2 or x.shape[1] == 2):
            self.iscsd = True
            self.data = x.T if x.shape[0] == 2 else x
            if self.verbose:
                logging.info(f"Detected two-channel data with length {len(self.data)}")
        elif len(x.shape) == 1:
            self.iscsd = False
            self.data = x
            if self.verbose:
                logging.info(
                    f"Detected single-channel data with length {len(self.data)}"
                )
        else:
            raise ValueError("Input data must be a 1D array or a 2xN/Nx2 array.")

        self.nx = len(self.data)
        self.config["N"] = self.nx

        self._process_window_config()
        self._process_scheduler_config()

        self._plan_cache: Optional[Dict[str, Any]] = None

    def _process_window_config(self):
        """Internal method to resolve window function and related parameters."""
        win_param = self.config["win"]
        psll = self.config["psll"]
        win_str_name = "Unknown"

        if isinstance(win_param, str):
            win_str_name = win_param
            if win_param.lower() in ["kaiser"]:
                self.config["win_func"] = np_kaiser
                if psll is None:
                    raise ValueError("PSLL must be specified for the Kaiser window.")
                self.config["alpha"] = kaiser_alpha(psll)
            elif win_param.lower() in ["hann", "hanning"]:
                self.config["win_func"] = np.hanning
            elif win_param in win_dict:
                self.config["win_func"] = win_dict[win_param]
            else:
                raise ValueError(f"Window function '{win_param}' not recognized.")
        elif callable(win_param):
            self.config["win_func"] = win_param
            if win_param in [np_kaiser, sp_kaiser]:
                self.config["win_func"] = np_kaiser
                if psll is None:
                    raise ValueError("PSLL must be specified for the Kaiser window.")
                self.config["alpha"] = kaiser_alpha(psll)
            if is_function_in_dict(win_param, win_dict):
                win_str_name = get_key_for_function(win_param, win_dict)
        else:
            raise TypeError(
                "Window must be a recognized string or a callable function."
            )

        # Resolve automatic overlap based on the selected window
        if self.config["olap"] == "default":
            if self.config["win_func"] == np_kaiser:
                self.config["final_olap"] = kaiser_rov(self.config["alpha"])
            elif win_str_name in olap_dict:
                self.config["final_olap"] = olap_dict[win_str_name]
            else:
                if self.verbose:
                    logging.warning(
                        f"Optimal overlap for window '{win_str_name}' not found. Defaulting to 0.5."
                    )
                self.config["final_olap"] = 0.5
        else:
            self.config["final_olap"] = self.config["olap"]

    def _process_scheduler_config(self):
        """Internal method to resolve the scheduler function."""
        scheduler_param = self.config["scheduler"]
        if isinstance(scheduler_param, str):
            schedulers = {"lpsd": lpsd_plan, "ltf": ltf_plan, "new_ltf": new_ltf_plan}
            if scheduler_param in schedulers:
                self.config["scheduler_func"] = schedulers[scheduler_param]
            else:
                raise ValueError(f"Scheduler '{scheduler_param}' not recognized.")
        elif callable(scheduler_param):
            self.config["scheduler_func"] = scheduler_param
        else:
            raise TypeError(
                "Scheduler must be a recognized string or a callable function."
            )

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
            for frequencies (`f`), segment lengths (`L`), number of averages (`K`), etc.
        """
        if self._plan_cache is not None:
            return self._plan_cache

        scheduler_func = self.config["scheduler_func"]

        # Common kwargs for any scheduler (**args style); extra keys are tolerated.
        common_kwargs = dict(
            N=self.nx,
            fs=self.fs,
            olap=self.config["final_olap"],
            bmin=self.config["bmin"],
            Lmin=self.config["Lmin"],
            Kdes=self.config["Kdes"],
        )

        if scheduler_func == new_ltf_plan:
            common_kwargs["num_patch_pts"] = self.config["num_patch_pts"]

        # If we must force an exact target number of frequencies, solve for Jdes
        if self.config["force_target_nf"]:
            if self.verbose:
                logging.info(f"Adjusting plan to target {self.config['Jdes']} frequencies...")
            target_nf = self.config["Jdes"]
            solved_Jdes = find_Jdes_binary_search(scheduler_func, target_nf, **common_kwargs)
            assert solved_Jdes is not None, (
                "Failed to generate plan with forced number of frequencies"
            )
            self.config["Jdes"] = int(solved_Jdes)

        # Generate plan using the selected scheduler (all take **args now)
        call_kwargs = dict(common_kwargs, Jdes=self.config["Jdes"])
        plan_output = scheduler_func(**call_kwargs)

        # Apply frequency band filter if specified
        if self.config["band"] is not None:
            fmin, fmax = self.config["band"]
            mask = (plan_output["f"] >= fmin) & (plan_output["f"] <= fmax)
            if not np.any(mask):
                raise ValueError("No frequencies found in the specified band.")

            for key in ["f", "r", "b", "L", "K", "navg", "O"]:
                plan_output[key] = plan_output[key][mask]
            plan_output["D"] = [row for row, keep in zip(plan_output["D"], mask) if keep]
            plan_output["nf"] = len(plan_output["f"])

        # Normalize D to arrays
        plan_output["D"] = [np.array(d) for d in plan_output["D"]]

        self._plan_cache = plan_output
        return self._plan_cache


    def compute_single_bin(
        self, freq: float, *, fres: Optional[float] = None, L: Optional[int] = None
    ) -> "SpectrumResult":
        """
        Executes spectral analysis for a single, user-defined frequency bin.

        This method is optimized for calculating spectral quantities at one
        specific frequency, defined by either a frequency resolution (`fres`)
        or a segment length (`L`).

        Parameters
        ----------
        freq : float
            The target Fourier frequency in Hz for the analysis.
        fres : float, optional
            The desired frequency resolution in Hz for the bin.
            Either `fres` or `L` must be provided.
        L : int, optional
            The desired segment length in samples for the bin.
            Either `fres` or `L` must be provided.

        Returns
        -------
        SpectrumResult
            A result object containing the spectral estimates for the single bin.
            All result attributes will be scalar values instead of arrays.
        """
        if L is not None:
            len = int(L)
            final_fres = self.fs / len
        elif fres is not None:
            final_fres = fres
            len = int(self.fs / fres)
        else:
            raise ValueError(
                "Either `fres` (frequency resolution) or `L` (segment length) must be provided."
            )

        m = freq / final_fres  # Fractional bin number

        # --- DFT Kernel Generation ---
        if self.config["win_func"] in [np_kaiser, sp_kaiser]:
            window = self.config["win_func"](len + 1, self.config["alpha"] * np.pi)[:-1]
        else:
            window = self.config["win_func"](len)

        p = 1j * 2 * np.pi * m / len * np.arange(len)
        C = window * np.exp(p)  # Complex DFT kernel

        # --- Segmentation ---
        navg = int(
            round_half_up(((self.nx - len) / (1 - self.config["final_olap"])) / len + 1)
        )
        if navg == 1:
            shift = 0.0
            starts = [0]
        else:
            shift = (self.nx - len) / (navg - 1)
            starts = np.round(np.arange(navg) * shift).astype(int)

        segments = np.array([self.data[d_start : d_start + len] for d_start in starts])

        # --- Detrending ---
        x1s_all = segments[:, :, 0] if self.iscsd else segments
        order = self.config["order"]
        if order == -1:  # No detrending
            pass
        elif order == 0:
            x1s_all -= np.mean(x1s_all, axis=1, keepdims=True)
            if self.iscsd:
                x2s_all = segments[:, :, 1]
                x2s_all -= np.mean(x2s_all, axis=1, keepdims=True)
        elif order > 0:
            x1s_all = np.apply_along_axis(polynomial_detrend, 1, x1s_all, order)
            if self.iscsd:
                x2s_all = segments[:, :, 1]
                x2s_all = np.apply_along_axis(polynomial_detrend, 1, x2s_all, order)

        if self.iscsd and order != -1:  # Re-assign detrended data if needed
            segments[:, :, 0] = x1s_all
            segments[:, :, 1] = x2s_all

        # --- DFT and Averaging ---
        rxsums = np.dot(x1s_all, np.real(C))
        ixsums = np.dot(x1s_all, np.imag(C))

        if self.iscsd:
            x2s_all = segments[:, :, 1]
            rysums = np.dot(x2s_all, np.real(C))
            iysums = np.dot(x2s_all, np.imag(C))
        else:
            rysums, iysums = rxsums, ixsums

        XYr_all = rysums * rxsums + iysums * ixsums
        XYi_all = iysums * rxsums - rysums * ixsums
        XX_all = rxsums**2 + ixsums**2
        YY_all = rysums**2 + iysums**2

        MXX = np.mean(XX_all)
        MYY = np.mean(YY_all)
        XY = np.mean(XYr_all) + 1j * np.mean(XYi_all)
        M2 = np.var(XYr_all + 1j * XYi_all) if navg > 1 else 0.0

        S1 = np.sum(window)
        S2 = np.sum(window**2)

        # --- Package results for SpectrumResult ---
        # Note: We package results as single-element lists/arrays so SpectrumResult
        # can process them just like a multi-bin result.
        single_bin_results = {
            "f": np.array([freq]),
            "r": np.array([final_fres]),
            "b": np.array([m]),
            "L": np.array([len]),
            "K": np.array([navg]),
            "navg": np.array([navg]),
            "D": np.array([starts], dtype=object),
            "O": np.array([self.config["final_olap"]]),
            "i": np.array([0]),
            "XX": np.array([MXX]),
            "YY": np.array([MYY]),
            "XY": np.array([XY]),
            "S12": np.array([S1**2]),
            "S2": np.array([S2]),
            "M2": np.array([M2]),
            "compute_t": np.array([0.0]),  # Timer not used for single bin
        }

        return SpectrumResult(single_bin_results, self.config, self.iscsd, self.fs)


    def compute(self) -> "SpectrumResult":
        """
        Executes the spectral analysis and returns a SpectrumResult object.

        This method performs the core computation. It uses the generated plan 
        to segment the data, apply windowing and FFTs, and average the results.

        Returns
        -------
        SpectrumResult
            An object containing all computed spectral quantities and helper methods.
        """
        plan = self.plan()
        if self.verbose:
            logging.info(f"Computing {plan['nf']} frequencies...")

        start_time = time.time()

        # Compute:
        results_list = [self._lpsd_core(np.arange(plan["nf"]))]

        if self.verbose:
            logging.info(
                f"Computation completed in {time.time() - start_time:.2f} seconds."
            )

        nf = plan["nf"]
        XX = np.empty(nf, np.float64)
        YY = np.empty(nf, np.float64)
        XY = np.empty(nf, np.complex128)
        S12 = np.empty(nf, np.float64)
        S2  = np.empty(nf, np.float64)
        M2  = np.empty(nf, np.float64)
        tms = np.empty(nf, np.float64)

        for chunk in results_list:
            for (i, xy, mxx, myy, s12, s2, m2, tm) in chunk:
                XX[i]  = mxx
                YY[i]  = myy
                XY[i]  = xy
                S12[i] = s12
                S2[i]  = s2
                M2[i]  = m2
                tms[i] = tm

        final_results = {**plan,
                        "XX": XX, "YY": YY, "XY": XY,
                        "S12": S12, "S2": S2, "M2": M2,
                        "compute_t": tms}
        return SpectrumResult(final_results, self.config, self.iscsd, self.fs)

    def _lpsd_core(self, f_indices: np.ndarray) -> List[Any]:
        """Core processing loop for a block of frequency indices."""
        plan = self._plan_cache
        results_block: List[Any] = []

        # Cache window (and sums) per L, and Q per (L, order)
        window_cache: Dict[int, Tuple[np.ndarray, float, float]] = {}
        Q_cache: Dict[Tuple[int, int], np.ndarray] = {}

        # Contiguous views
        if self.iscsd:
            x1 = np.ascontiguousarray(self.data[:, 0], dtype=np.float64)
            x2 = np.ascontiguousarray(self.data[:, 1], dtype=np.float64)
        else:
            x1 = np.ascontiguousarray(self.data, dtype=np.float64)
            x2 = None  # type: ignore

        for i in f_indices:
            t0 = time.time()
            L = int(plan["L"][i])
            m = float(plan["m"][i])     # fractional bin
            starts = plan["D"][i]       # np.ndarray of start indices

            # Window cache
            if L not in window_cache:
                if self.config["win_func"] in (np_kaiser, sp_kaiser):
                    w = self.config["win_func"](L + 1, self.config["alpha"] * np.pi)[:-1]
                else:
                    w = self.config["win_func"](L)
                w = np.asarray(w, dtype=np.float64)
                S1 = float(w.sum()); S2 = float((w*w).sum())
                window_cache[L] = (w, S1, S2)
            else:
                w, S1, S2 = window_cache[L]

            omega = 2.0 * np.pi * (m / L)
            order = int(self.config["order"])

            if order == -1:
                if self.iscsd:
                    if _HAS_NUMBA:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_win_only_csd(x1, x2, starts, L, w, omega)
                    else: 
                        _stats_poly_csd_np(x1, x2, starts, L, w, omega, np.zeros((L,1)))
                else:
                    if _HAS_NUMBA:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_win_only_auto(x1, starts, L, w, omega)
                    else: 
                        _stats_poly_auto_np(x1, starts, L, w, omega, np.zeros((L,1)))
            elif order == 0:
                if self.iscsd:
                    if _HAS_NUMBA:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_detrend0_csd(x1, x2, starts, L, w, omega)
                    else: 
                        _stats_poly_csd_np(x1, x2, starts, L, w, omega, np.zeros((L,1)))
                else:
                    if _HAS_NUMBA:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_detrend0_auto(x1, starts, L, w, omega)
                    else:
                        _stats_poly_auto_np(x1, starts, L, w, omega, np.zeros((L,1)))
            elif order in (1, 2):
                key = (L, order)
                Q = Q_cache.get(key)
                if Q is None:
                    Q = _build_Q(L, order)
                    Q_cache[key] = Q
                if self.iscsd:
                    if _HAS_NUMBA:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd(x1, x2, starts, L, w, omega, Q)
                    else:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q)
                else:
                    if _HAS_NUMBA:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto(x1, starts, L, w, omega, Q)
                    else:
                        MXX, MYY, mu_r, mu_i, M2 = _stats_poly_auto_np(x1, starts, L, w, omega, Q)
            else:
                raise NotImplementedError

            XY = complex(mu_r, mu_i)
            results_block.append([i, XY, float(MXX), float(MYY), S1*S1, S2, float(M2), time.time() - t0])

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

    Examples
    --------
    >>> import numpy as np
    >>> import speckit
    >>> fs = 1000
    >>> t = np.arange(0, 10, 1/fs)
    >>> signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.randn(len(t))

    >>> # Compute the ASD in one line
    >>> result = speckit.compute_spectrum(signal, fs=fs, win='hann')

    >>> # Access the results
    >>> print(result.asd)

    >>> # Use the built-in plotting
    >>> fig, ax = result.plot('asd')
    >>> plt.show()
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

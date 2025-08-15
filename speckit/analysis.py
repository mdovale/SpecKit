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
from typing import Union, Callable, Optional, Tuple

import numpy as np
import multiprocessing as mp

from .core import SpectrumAnalyzer, SpectrumResult

def compute_spectrum(
    data: np.ndarray,
    fs: float, *,
    pool: Optional[mp.Pool] = None,
    **kwargs
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
    pool : multiprocessing.Pool, optional
        A multiprocessing pool to parallelize the computation across
        frequency bins. If None, the computation is done serially.
        Defaults to None.
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

    # 2. Immediately call the compute method, passing the pool
    result = analyzer.compute(pool=pool)

    # 3. Return the final result object
    return result

def compute_single_bin(
    data: np.ndarray,
    fs: float,
    freq: float, *,
    fres: Optional[float] = None,
    L: Optional[int] = None,
    **kwargs
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
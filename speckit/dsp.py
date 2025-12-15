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
import os
import numpy as np
import pandas as pd
import zipfile
import tarfile
import gzip
from py7zr import SevenZipFile
from copy import deepcopy
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit, minimize
from scipy.signal import welch
from typing import List, Optional, Callable
import warnings

import logging

logger = logging.getLogger(__name__)


def frequency2phase(f, fs, subtract_mean=True):
    """
    Integrate frequency in Hz to find phase in radians.

    Parameters
    ----------
    f : numpy.ndarray
        The input signal frequency in Hz.
    fs : float
        The sampling frequency in Hz. Must be positive.
    subtract_mean : bool, optional
        If True, subtract the mean frequency before integration (default behavior).
        If False, integrate the frequency directly, resulting in a ramp for constant
        frequency. Default is True.

    Returns
    -------
    numpy.ndarray
        The phase in radians.

    Raises
    ------
    ValueError
        If `fs` is not positive or if `f` is empty.
    """
    f = np.asarray(f)
    if f.size == 0:
        raise ValueError("Input frequency array `f` must not be empty.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling frequency `fs` must be positive, got {fs!r}.")
    if subtract_mean:
        f_integrate = f - np.mean(f)
    else:
        f_integrate = f
    return (2 * np.pi / fs) * np.cumsum(f_integrate)


def polynomial_detrend(x, order=1):
    """
    Detrend an input signal using a fast and stable polynomial fit.

    Parameters
    ----------
    x : numpy.ndarray
        The input signal to be detrended.
    order : int, optional
        The order of the polynomial fit. Must be non-negative. Default is 1.

    Returns
    -------
    numpy.ndarray
        The detrended signal.

    Raises
    ------
    ValueError
        If `x` is empty or if `order` is negative.
    """
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("Input signal `x` must not be empty.")
    if order < 0:
        raise ValueError(f"Polynomial order must be non-negative, got {order}.")
    if order == 0:
        return x - np.mean(x)
    if len(x) < order + 1:
        logger.warning(
            f"Signal length ({len(x)}) is less than order+1 ({order+1}). "
            f"Using order={len(x)-1} instead."
        )
        order = len(x) - 1
    t = np.arange(len(x))
    # Use the numerically stable polyfit to find coefficients
    coeffs = np.polyfit(t, x, deg=order)
    # Evaluate the polynomial trend
    trend = np.polyval(coeffs, t)
    return x - trend


def crop_data(x, y, xmin, xmax):
    """
    Crop data to a specified range in x.

    Parameters
    ----------
    x : array-like
        The x-axis data (e.g., frequency, time).
    y : array-like
        The y-axis data corresponding to x.
    xmin : float
        Lower bound for cropping (inclusive).
    xmax : float
        Upper bound for cropping (inclusive).

    Returns
    -------
    tuple of numpy.ndarray
        A tuple (x_cropped, y_cropped) containing the cropped arrays.

    Raises
    ------
    ValueError
        If `x` and `y` have different lengths, or if `xmin > xmax`.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError(
            f"Arrays `x` and `y` must have the same length, "
            f"got {len(x)} and {len(y)}."
        )
    if xmin > xmax:
        raise ValueError(
            f"`xmin` ({xmin}) must be <= `xmax` ({xmax})."
        )
    if x.size == 0:
        return x, y

    # Create a boolean mask for the range condition
    mask = (x >= xmin) & (x <= xmax)

    # Apply the mask to both x and y arrays and return
    return x[mask], y[mask]


def truncation(x: np.ndarray, n_trunc: int) -> np.ndarray:
    """Truncate both ends of a time-series array.

    Args:
        x: Array of data to truncate.
        n_trunc: Number of points to remove from each end of the array.
                 Must be a non-negative integer. If zero or negative, the
                 array is returned unchanged.

    Returns:
        Truncated array.

    Raises:
        ValueError: If n_trunc is not an integer or is too large.
    """
    if not isinstance(n_trunc, int):
        raise ValueError(f"n_trunc must be an integer, got {n_trunc}")
    if n_trunc < 0:
        raise ValueError(f"n_trunc must be non-negative, got {n_trunc}")
    if n_trunc * 2 > len(x):
        raise ValueError(
            f"Cannot truncate {n_trunc} elements from each side of array with length {len(x)}"
        )

    return x[n_trunc:-n_trunc] if n_trunc > 0 else x


def integral_rms(fourier_freq, asd, pass_band=None):
    """
    Compute the RMS as integral of an Amplitude Spectral Density (ASD).

    Parameters
    ----------
    fourier_freq : array-like
        Fourier frequency array in Hz.
    asd : array-like
        Amplitude spectral density from which RMS is computed.
    pass_band : tuple of float, optional
        Frequency band for integration as (fmin, fmax). If None, integrates
        over the entire frequency range. Default is None.

    Returns
    -------
    float
        The RMS value computed from the integral of the squared ASD.

    Raises
    ------
    ValueError
        If `fourier_freq` and `asd` have different lengths, or if `pass_band`
        is invalid.
    """
    fourier_freq = np.asarray(fourier_freq)
    asd = np.asarray(asd)
    
    if len(fourier_freq) != len(asd):
        raise ValueError(
            f"Arrays `fourier_freq` and `asd` must have the same length, "
            f"got {len(fourier_freq)} and {len(asd)}."
        )
    if fourier_freq.size == 0:
        raise ValueError("Input arrays must not be empty.")
    
    if pass_band is None:
        pass_band = [-np.inf, np.inf]
    else:
        if len(pass_band) != 2:
            raise ValueError(
                f"`pass_band` must be a tuple of length 2, got {len(pass_band)}."
            )
        if pass_band[0] > pass_band[1]:
            raise ValueError(
                f"`pass_band[0]` ({pass_band[0]}) must be <= `pass_band[1]` ({pass_band[1]})."
            )

    integral_range_min = max(np.min(fourier_freq), pass_band[0])
    integral_range_max = min(np.max(fourier_freq), pass_band[1])
    
    if integral_range_min >= integral_range_max:
        logger.warning(
            f"No valid frequency range for integration. "
            f"Returning 0.0."
        )
        return 0.0
    
    f_tmp, asd_tmp = crop_data(
        fourier_freq, asd, integral_range_min, integral_range_max
    )
    
    if len(f_tmp) == 0:
        logger.warning("No data points in the specified frequency band. Returning 0.0.")
        return 0.0
    
    integral_rms2 = cumulative_trapezoid(asd_tmp**2, f_tmp, initial=0)
    return np.sqrt(integral_rms2[-1])


def peak_finder(frequency, measurement, cnr=10, edge=True, freq_band=None, rtol=1e-2):
    """
    Detects peaks in a measurement array based on CNR(dB) threshold.

    Parameters
    ----------
    frequency : array-like
        The frequency array corresponding to the measurements.
    measurement : array-like
        The measurement array where peaks are to be detected.
    cnr : float, optional
        Carrier-to-noise density ratio in dB. Peaks must exceed this ratio to be considered valid.
        Default is 10.
    edge : bool, optional
        If True, consider peaks that are on the boundary of the spectrum.
        Default is True.
    freq_band : tuple of (float, float), optional
        Frequency band to search for peaks, specified as (low_freq, high_freq).
        Only frequencies within this range are considered. Default is None.
    rtol : float, optional
        Relative tolerance for identifying flat peaks. Default is 1e-2.

    Returns
    -------
    peak_frequencies : ndarray
        Array of frequencies at which peaks were detected.
    peak_measurements : ndarray
        Array of measurement values at the detected peak frequencies.

    Raises
    ------
    ValueError
        If `frequency` and `measurement` have different lengths, or if `freq_band`
        is invalid, or if `rtol` is not positive.

    Notes
    -----
    The function first applies an optional frequency band filter and then manually
    detects peaks by identifying points that are higher than their immediate neighbors.
    Peaks that do not meet the specified carrier-to-noise density ratio are discarded.
    The function returns the frequencies and measurements of the detected peaks.
    """

    def noise_model(x, a, b, alpha):
        return a + b * x**alpha

    frequency = np.asarray(frequency)
    measurement = np.asarray(measurement)
    
    if len(frequency) != len(measurement):
        raise ValueError(
            f"Arrays `frequency` and `measurement` must have the same length, "
            f"got {len(frequency)} and {len(measurement)}."
        )
    if frequency.size == 0:
        raise ValueError("Input arrays must not be empty.")
    if rtol <= 0:
        raise ValueError(f"`rtol` must be positive, got {rtol}.")

    if freq_band is not None:
        if len(freq_band) != 2:
            raise ValueError(
                f"`freq_band` must be a tuple of length 2, got {len(freq_band)}."
            )
        low_freq, high_freq = freq_band
        if low_freq > high_freq:
            raise ValueError(
                f"`freq_band[0]` ({low_freq}) must be <= `freq_band[1]` ({high_freq})."
            )
        mask = (frequency >= low_freq) & (frequency <= high_freq)
        frequency = frequency[mask]
        measurement = measurement[mask]

    if len(frequency) == 0:
        return np.array([]), np.array([])

    # Initial peak finding
    peaks = []
    i = 1
    while i < len(measurement) - 1:
        if measurement[i - 1] < measurement[i] > measurement[i + 1]:
            peaks.append(i)
        elif (measurement[i - 1] < measurement[i]) and np.isclose(
            measurement[i], measurement[i + 1], rtol=rtol
        ):
            start = i
            while i < len(measurement) - 1 and np.isclose(
                measurement[i], measurement[i + 1], rtol=rtol
            ):
                i += 1
            if measurement[i] > measurement[i + 1]:
                mid = (start + i) // 2
                peaks.append(mid)
        i += 1

    if edge:
        if measurement[0] > measurement[1]:
            peaks.insert(0, 0)
        if measurement[-1] > measurement[-2]:
            peaks.append(len(measurement) - 1)
    else:
        peaks = [p for p in peaks if p != 0 and p != len(measurement) - 1]

    # Exclude peaks for noise fitting
    non_peak_mask = np.ones(len(measurement), dtype=bool)
    non_peak_mask[peaks] = False

    # Fit to noise model
    popt, _ = curve_fit(
        noise_model, frequency[non_peak_mask], measurement[non_peak_mask]
    )
    noise_level = noise_model(frequency, *popt)

    # Calculate CNR threshold
    cnr_threshold = noise_level * (10 ** (cnr / 10))
    valid_peaks = [p for p in peaks if measurement[p] > cnr_threshold[p]]

    peak_frequencies = frequency[valid_peaks]
    peak_measurements = measurement[valid_peaks]

    return peak_frequencies, peak_measurements


def optimal_linear_combination(
    df,
    inputs,
    output,
    timeshifts=False,
    gradient=False,
    domain="time",
    method="TNC",
    tol=1e-9,
    *args,
    **kwargs,
):
    """
    Computes the coefficients of a linear combination of optionally timeshifted "input" signals
    that minimize noise when added to the "output" signal.

    Target: `RMS[ output + Sum [coefficient_i * timeshift(input_i, shift_i) ] ]`

    Parameters
    ----------
    df : pd.DataFrame
        Data from signals.
    inputs : list of str
        Labels of the input signal columns in the input DataFrame.
    output : str
        Label of the output signal column in the input DataFrame.
    timeshifts : bool, optional
        Whether the input signals should be timeshifted. Default is False.
    gradient : bool, optional
        Whether to minimize rms in the time series or on its derivative. Default is False.
    domain : str, optional
        Whether to compute RMS in the time domain or in the frequency domain.
        Must be 'time' or 'frequency'. Default is 'time'.
    method : str, optional
        The minimizer method. Default is 'TNC'.
    tol : float, optional
        The minimizer tolerance parameter. Must be positive. Default is 1e-9.
    *args
        Additional positional arguments passed to `welch` when domain='frequency'.
    **kwargs
        Additional keyword arguments passed to `welch` when domain='frequency'.

    Returns
    -------
    OptimizeResult
        The optimization result object from scipy.optimize.minimize.
    np.ndarray
        The output with optimal combination of inputs subtracted.

    Raises
    ------
    ValueError
        If `df` is empty, if required columns are missing, if `domain` is invalid,
        or if `tol` is not positive.
    TypeError
        If `df` is not a pandas DataFrame, or if `inputs` is not a list.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Input DataFrame `df` must not be empty.")
    if not isinstance(inputs, (list, tuple)):
        raise TypeError(f"`inputs` must be a list or tuple, got {type(inputs).__name__}.")
    if len(inputs) == 0:
        raise ValueError("`inputs` must not be empty.")
    if not isinstance(output, str):
        raise TypeError(f"`output` must be a string, got {type(output).__name__}.")
    if domain not in ["time", "frequency"]:
        raise ValueError(f"`domain` must be 'time' or 'frequency', got {domain!r}.")
    if tol <= 0:
        raise ValueError(f"`tol` must be positive, got {tol}.")
    
    # Check that all required columns exist
    missing_inputs = [inp for inp in inputs if inp not in df.columns]
    if missing_inputs:
        raise ValueError(f"Input columns not found in DataFrame: {missing_inputs}.")
    if output not in df.columns:
        raise ValueError(f"Output column '{output}' not found in DataFrame.")
    
    # Check that columns are numeric
    for inp in inputs:
        if df[inp].dtype.kind not in "biufc":
            raise ValueError(f"Input column '{inp}' must be numeric, got dtype {df[inp].dtype}.")
    if df[output].dtype.kind not in "biufc":
        raise ValueError(f"Output column '{output}' must be numeric, got dtype {df[output].dtype}.")

    def print_optimization_result(res):
        logger.info("Optimization Results:")
        logger.info("=====================")
        logger.info(f"Success: {res.success}")
        logger.info(f"Message: {res.message}")
        logger.info(f"Function value at minimum: {res.fun}")
        logger.info("Solution:")
        for idx, val in enumerate(res.x, start=1):
            logger.info(f"Variable {idx}: {val}")

    def fun(x):
        y = np.array(df[output] - np.mean(df[output]))

        if timeshifts:
            for i, input in enumerate(df[inputs]):
                Si = np.array(df[input] - np.mean(df[input]))
                y += x[len(inputs) + i] * timeshift(Si, x[i])
            max_delay = np.max(x[: len(inputs)])
            y = truncation(y, n_trunc=int(2 * max_delay))
        else:
            for i, input in enumerate(df[inputs]):
                Si = np.array(df[input] - np.mean(df[input]))
                y += x[i] * Si

        if gradient:
            y = np.gradient(y)

        if domain == "time":
            rms_value = np.sqrt(np.mean(np.square(y - np.mean(y))))
        elif domain == "frequency":
            f, Sxx = welch(y, scaling="density", *args, **kwargs)
            rms_value = np.sqrt(np.trapezoid(Sxx, f))
        else:
            raise ValueError(
                "The `domain` parameter must be set to 'time' or 'frequency'"
            )

        return rms_value

    if timeshifts:
        x_initial = np.zeros(len(inputs) * 2)
    else:
        x_initial = np.zeros(len(inputs))

    logger.info(f"Solving {len(x_initial)}-dimensional problem...")

    res = minimize(fun, x_initial, method=method, tol=tol)

    print_optimization_result(res)

    if timeshifts:
        y = np.array(df[output] - np.mean(df[output]))
        for i, input in enumerate(df[inputs]):
            Si = np.array(df[input] - np.mean(df[input]))
            y += res.x[len(inputs) + i] * timeshift(Si, res.x[i])
        max_delay = np.max(res.x[: len(inputs)])
        y = truncation(y, int(2 * max_delay))
    else:
        y = np.array(df[output] - np.mean(df[output]))
        for i, input in enumerate(df[inputs]):
            Si = np.array(df[input] - np.mean(df[input]))
            y += res.x[i] * Si

    return res, y


def df_timeshift(
    df, fs, seconds, columns=None, truncate=None, inplace=False, suffix="_shifted"
):
    """
    Timeshift columns of a pandas DataFrame or the entire DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    fs : float
        The sampling frequency of the data. Must be positive.
    seconds : float
        Amount of seconds to shift the data by.
    columns : list or None, optional
        List of columns to shift. If None, all columns are shifted.
        Default is None.
    truncate : bool or int or None, optional
        If True, truncate the resulting DataFrame based on the shift.
        If int, specify the exact number of rows to truncate at both ends.
        Default is None.
    inplace : bool, optional
        If True, overwrite the original columns. If False, add shifted columns with suffix.
        Default is False.
    suffix : str, optional
        Suffix to add to column names when inplace is False. Default is "_shifted".

    Returns
    -------
    pd.DataFrame
        The timeshifted DataFrame.

    Raises
    ------
    ValueError
        If `df` is empty, if `fs` is not positive, or if specified columns don't exist.
    TypeError
        If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Input DataFrame `df` must not be empty.")
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"Sampling frequency `fs` must be positive, got {fs!r}.")

    if seconds == 0.0:
        return df

    df_shifted = df.copy()

    if columns is None:
        columns = df.columns.tolist()
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Columns not found in DataFrame: {missing_cols}."
            )

    for c in columns:
        if df[c].dtype.kind not in "biufc":
            logger.warning(
                f"Column '{c}' is not numeric (dtype: {df[c].dtype}), skipping timeshift."
            )
            continue
        shifted = timeshift(df[c].to_numpy(), seconds * fs)
        if inplace:
            df_shifted[c] = shifted
        else:
            df_shifted[f"{c}{suffix}"] = shifted

    if truncate is not None:
        if isinstance(truncate, bool):
            n_trunc = int(2 * abs(seconds * fs))
        else:
            if not isinstance(truncate, (int, np.integer)):
                raise ValueError(
                    f"`truncate` must be bool, int, or None, got {type(truncate).__name__}."
                )
            n_trunc = int(truncate)
        if n_trunc > 0:
            if n_trunc * 2 >= len(df_shifted):
                logger.warning(
                    f"Truncation amount ({n_trunc}) is too large for DataFrame length "
                    f"({len(df_shifted)}). Returning empty DataFrame."
                )
                return df_shifted.iloc[0:0]
            df_shifted = df_shifted.iloc[n_trunc:-n_trunc]

    return df_shifted


def df_detrend(df, columns=None, order=1, inplace=False, suffix="_detrended"):
    """
    Detrend all or specified columns of a pandas DataFrame using polynomial_detrend.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    columns : list, optional
        List of column names to detrend. If None, all columns are detrended.
        Default is None.
    order : int, optional
        The order of the polynomial fit. Must be non-negative. Default is 1.
    inplace : bool, optional
        If True, overwrite the original columns. If False, create new columns with suffix.
        Default is False.
    suffix : str, optional
        Suffix to add to column names when inplace is False. Default is "_detrended".

    Returns
    -------
    pd.DataFrame
        A DataFrame with detrended data.

    Raises
    ------
    ValueError
        If `df` is empty, if specified columns don't exist, or if `order` is negative.
    TypeError
        If `df` is not a pandas DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"`df` must be a pandas DataFrame, got {type(df).__name__}.")
    if df.empty:
        raise ValueError("Input DataFrame `df` must not be empty.")
    if order < 0:
        raise ValueError(f"Polynomial order must be non-negative, got {order}.")
    
    df_detrended = df.copy()
    if columns is None:
        columns = df.columns.tolist()
    else:
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Columns not found in DataFrame: {missing_cols}."
            )

    for col in columns:
        if df[col].dtype.kind in "biufc":  # Check if the column is numeric
            detrended_data = polynomial_detrend(df[col].values, order=order)
            if inplace:
                df_detrended[col] = detrended_data
            else:
                df_detrended[f"{col}{suffix}"] = detrended_data
        else:
            logger.warning(
                f"Column '{col}' is not numeric (dtype: {df[col].dtype}), skipping detrend."
            )

    return df_detrended


def multi_file_timeseries_loader(
    file_list: List[str],
    fs_list: List[float],
    names_list: Optional[List[str]] = None,
    start_time: Optional[float] = 0.0,
    duration_hours: Optional[float] = None,
    timeshifts: Optional[List[float]] = None,
    delimiter_list: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """
    Loads time-series data from multiple files, restricting the output to the maximum overlapping time window across the datasets.

    Parameters
    ----------
    file_list : List[str]
        A list of file paths for the input data files. Each file must contain time-series data sampled with the
        sampling frequencies provided in `fs_list`.

    fs_list : List[float]
        A list of sampling frequencies (in Hz) corresponding to each file in `file_list`. The length of `fs_list` must
        match the length of `file_list`, and each value must be positive.

    names_list: Optional[List[str]]
        A list of the column names corresponding to each file.

    start_time : float, optional
        The starting time (in seconds) from which the data will be extracted in each file. The function will extract data
        starting at this time in each file, adjusting for the sampling frequency. Default is 0.0 seconds.

    timeshifts : List[float], optional
        Time shifts (in seconds) to apply to each data stream.

    delimiter_list : List[str], optional
        A list of delimiters to be used for reading each file. If not provided, a space (' ') will be assumed as the
        delimiter for all files. The length of `delimiter_list` must match the length of `file_list` if provided.

    Returns
    -------
    List[pd.DataFrame]
        A list of pandas DataFrames containing the synchronized time-series data for each file. Each DataFrame will have
        been sliced to ensure the maximum overlap duration between all datasets, starting from `start_time`.

    Raises
    ------
    ValueError
        If the length of `file_list` does not match the length of `fs_list` or `delimiter_list` (if provided), or if any
        value in `fs_list` is non-positive.

    Notes
    -----
    - The function assumes that the first row in each file (after skipping comment rows) contains the column names.
    - The data in each file will be truncated based on the maximum overlapping time window. The function computes
      this by calculating the number of rows in each file and their corresponding time durations, based on the sampling
      frequencies (`fs_list`).
    - The function supports files with different delimiters and skips any header rows that begin with comment symbols
      such as `#`, `%`, `!`, etc.
    """

    def count_header_rows(file):
        header_symbols = [
            "#",
            "%",
            "!",
            "@",
            ";",
            "&",
            "*",
            "/",
        ]  # Add more symbols as needed
        header_count = 0

        def process_file(file_obj):
            """Helper function to count headers in a file object."""
            nonlocal header_count
            for line in file_obj:
                line = (
                    line.decode("utf-8") if isinstance(line, bytes) else line
                )  # Handle binary content
                if any(line.startswith(symbol) for symbol in header_symbols):
                    header_count += 1
                else:
                    break

        if zipfile.is_zipfile(file):  # Check if it's a zip file
            with zipfile.ZipFile(file, "r") as zip_ref:
                first_file_name = zip_ref.namelist()[0]
                with zip_ref.open(first_file_name, "r") as target_file:
                    process_file(target_file)

        elif tarfile.is_tarfile(file):  # Check if it's a tar file
            with tarfile.open(file, "r") as tar_ref:
                first_member = tar_ref.getmembers()[0]
                with tar_ref.extractfile(first_member) as target_file:
                    process_file(target_file)

        elif file.endswith(".gz"):  # Check if it's a gzip file
            with gzip.open(file, "rt") as target_file:  # 'rt' for reading text
                process_file(target_file)

        elif file.endswith(".7z"):  # Check if it's a 7z file
            with SevenZipFile(file, "r") as seven_zip_ref:
                first_file_name = seven_zip_ref.getnames()[0]
                with seven_zip_ref.open(first_file_name) as target_file:
                    process_file(target_file)

        else:  # Treat it as a regular text file
            with open(file, "r") as target_file:
                process_file(target_file)

        return header_count

    # Ensure matching lengths of input lists
    if len(file_list) != len(fs_list):
        raise ValueError(
            "The length of `fs_list` must match the length of `file_list`."
        )
    if names_list is not None and len(file_list) != len(names_list):
        raise ValueError(
            "The length of `names_list` must match the length of `file_list`."
        )
    if delimiter_list is not None and len(file_list) != len(delimiter_list):
        raise ValueError(
            "The length of `delimiter_list` must match the length of `file_list`."
        )
    if timeshifts is not None and len(file_list) != len(timeshifts):
        raise ValueError(
            "The length of `timeshifts` must match the length of `file_list`."
        )
    for fs in fs_list:
        if fs <= 0:
            raise ValueError("Sampling frequency `fs` must be positive.")

    delimiter_list = delimiter_list or [" "] * len(file_list)
    timeshifts = timeshifts or [None] * len(file_list)
    header_rows = []  # Stores the number of header rows for each file
    record_lengths = []  # Stores the duration for each file
    df_list = []  # Store the actual dataframes
    max_duration = None  # Will hold the maximum overlapping time duration

    # File names and metadata discovery:
    file_names = []
    header_rows = []
    for i, file in enumerate(file_list):
        file_names.append(os.path.basename(file))
        rows = count_header_rows(file)
        header_rows.append(rows)
        logger.info(f"File '{file_names[i]}' contains {header_rows[i]} header rows.")

    # Data ingestion:
    logger.info("Loading data and calculating maximum time series overlap...")
    for i, file in enumerate(file_list):
        try:
            if names_list is not None and names_list[i] is not None:
                df = pd.read_csv(
                    file,
                    delimiter=delimiter_list[i],
                    skiprows=header_rows[i],
                    names=names_list[i],
                    engine="c",
                )
            else:
                df = pd.read_csv(
                    file,
                    delimiter=delimiter_list[i],
                    skiprows=header_rows[i],
                    header=0,
                    engine="c",
                )
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            logger.warning(
                f"Reading {file} with Python engine due to {type(e).__name__}: {e}"
            )
            if names_list is not None and names_list[i] is not None:
                df = pd.read_csv(
                    file,
                    delimiter=delimiter_list[i],
                    skiprows=header_rows[i],
                    names=names_list[i],
                    engine="python",
                )
            else:
                df = pd.read_csv(
                    file,
                    delimiter=delimiter_list[i],
                    skiprows=header_rows[i],
                    header=0,
                    engine="python",
                )
        logger.info(f"Loaded data from file '{file_names[i]}' with length {len(df)}")
        record_lengths.append(len(df) / fs_list[i])  # Data stream duration in seconds
        df_list.append(df)

    # Drop NaN columns and log warning:
    for i, df in enumerate(df_list):
        initial_columns = list(df.columns)  # Store initial column names
        df.dropna(axis=1, how="all", inplace=True)  # Drop columns with all NaN values
        dropped_columns = set(initial_columns) - set(df.columns)  # Find dropped columns
        # Log a warning if columns were dropped:
        if dropped_columns:
            logger.warning(
                f"File '{file_names[i]}' had columns dropped due to NaN values: {dropped_columns}"
            )

    # Determine the maximum overlap between datasets:
    max_duration = (
        min(record_lengths) - start_time
    )  # Maximum overlapping time between datasets in seconds
    logger.info(
        f"Maximum overlap: {max_duration:.2f} seconds ({max_duration / 3600.0:.2f} hours)"
    )

    # Apply optional timeshifts:
    samples_shifted = []
    for i, df in enumerate(df_list):
        if (timeshifts[i] is not None) and (timeshifts[i] != 0.0):
            logger.info(
                f"Applying {timeshifts[i]} seconds timeshift to the '{file_names[i]}' data stream"
            )
            df_list[i] = df_timeshift(
                df,
                seconds=timeshifts[i],
                fs=fs_list[i],
                columns=df.select_dtypes(include=["number"]).columns,
            )
            samples_shifted.append(timeshifts[i] * fs_list[i])
        else:
            samples_shifted.append(0.0)

    # Readjust of start_time and max_duration in the case of large time shifts:
    if any(samples_shifted):
        for i, df in enumerate(df_list):
            if (samples_shifted[i] < 0.0) and (
                abs(samples_shifted[i]) > int(start_time * fs_list[i])
            ):
                start_time = int(2 * abs(samples_shifted[i]))
            if (samples_shifted[i] > 0.0) and (
                abs(samples_shifted[i])
                > len(df)
                - (int(start_time * fs_list[i]) + int(max_duration * fs_list[i]))
            ):
                max_duration = (
                    len(df)
                    - int(start_time * fs_list[i])
                    - int(2 * abs(samples_shifted[i]))
                ) / fs_list[i]
        logger.info(
            f"Maximum overlap after timeshift and truncation: {max_duration:.2f} seconds"
        )

    # Adjust duration according to user input:
    total_time = max_duration  # Total measurement time in seconds
    if duration_hours is not None:
        if (duration_hours * 3600.0 > max_duration) or (duration_hours <= 0.0):
            logger.warning(
                f"Specified duration of {duration_hours:.2f} hours is not possible, setting to {max_duration / 3600.0:.2f} hours"
            )
        else:
            total_time = duration_hours * 3600.0

    # Truncation to the overlapping section and timestamping:
    final_df_list = []
    for i, df in enumerate(df_list):
        start_row = int(start_time * fs_list[i])  # Convert start time to row index
        end_row = start_row + int(
            total_time * fs_list[i]
        )  # Calculate the end row based on max overlap

        new_df = df.iloc[
            start_row : end_row + 1
        ].copy()  # Slice the dataframe to get the relevant rows
        new_df.reset_index(drop=True, inplace=True)
        time_column_name = "time"
        if time_column_name in new_df:
            time_column_name = "new_time"
        new_df[time_column_name] = np.linspace(
            start_row / fs_list[i], end_row / fs_list[i], len(new_df)
        )
        logger.info(f"""File \'{file_names[i]}\':
                                        Start row: {start_row}; Start time: {new_df[time_column_name].iloc[0]:.2f} seconds
                                        End row: {end_row}; End time: {new_df[time_column_name].iloc[-1]:.2f} seconds
                                        Total time: {(new_df[time_column_name].iloc[-1] - new_df[time_column_name].iloc[0]):.2f} seconds.""")
        final_df_list.append(new_df)

    return final_df_list


def resample_to_common_grid(
    df_list: List[pd.DataFrame],
    fs: float,
    t_col_list: Optional[List[str]] = None,
    tolerance: Optional[float] = 0.1,
    preprocessors: Optional[List[Callable]] = None,
    suffixes: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Resample one or multiple DataFrames to a common time grid by interpolating
    data to align with a unified time axis.

    Parameters
    ----------
    df_list : List[pd.DataFrame]
        A list containing one or more DataFrames, each with a time column
        and associated data columns to be interpolated onto a common time grid.

    fs : float
        Sampling frequency (Hz) for the common time grid. Must be positive.

    t_col_list : List[str], optional
        Column names representing time for each DataFrame in `df_list`. If not provided,
        defaults to 'time' for all DataFrames.

    tolerance : float, optional
        The allowable deviation (in seconds) from the mean sampling interval. Values
        exceeding this tolerance trigger warnings.

    preprocessors: List[Callable], optional
        Pre-processing functions to apply to each data stream before resampling. Defaults to None.

    suffixes : Optional[bool], default=True
        If True, suffixes the column names of each DataFrame with its index in `df_list`
        to avoid name conflicts. If False, original column names are retained (name
        conflicts may arise if columns have identical names).

    Returns
    -------
    pd.DataFrame
        DataFrame containing interpolated values of each input DataFrame aligned
        on a common time grid, with time values in 'common_time'. If `suffixes=True`,
        data columns are suffixed with the DataFrame index to clarify origin.

    Raises
    ------
    ValueError
        If `fs` is non-positive, or any DataFrame lacks the specified or default time column.
    """
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    _df_list = deepcopy(df_list)

    if fs <= 0:
        raise ValueError("Sampling frequency `fs` must be positive.")
    if len(_df_list) == 0:
        raise ValueError("At least one DataFrame must be provided.")

    t_col_list = t_col_list or ["time"] * len(_df_list)
    preprocessors = preprocessors or [None] * len(_df_list)

    # If only one DataFrame is provided, handle separately
    if len(_df_list) == 1:
        df = _df_list[0]
        t_col = t_col_list[0]

        if t_col not in df:
            raise ValueError(
                f"The provided DataFrame does not contain the time column '{t_col}'"
            )

        # Determine time range and common time grid
        start_time, end_time = df[t_col].min(), df[t_col].max()
        common_time = np.arange(start_time, end_time, 1 / fs)

        logger.info(
            f"Single DataFrame provided: Resampling to {len(common_time)} samples from {start_time:.2f}s to {end_time:.2f}s"
        )

        # Apply preprocessing if specified
        if preprocessors[0] is not None:
            logger.info("Applying preprocessor to DataFrame")
            df = preprocessors[0](df)

        # Interpolation
        df_interp = pd.DataFrame({"common_time": common_time})
        for col in df.columns:
            if col == t_col:
                continue  # Skip the time column
            df_interp[col] = np.interp(common_time, df[t_col], df[col])

        return df_interp

    # Handling multiple DataFrames (original logic)
    start_time, end_time = 0.0, float("inf")
    df_interp_list = []

    # Determine the overlapping time range
    for i, (df, t) in enumerate(zip(_df_list, t_col_list)):
        if t not in df:
            raise ValueError(
                f"DataFrame #{i + 1} does not contain the time column '{t}'"
            )
        start_time = max(start_time, df[t].min())
        end_time = min(end_time, df[t].max())
        logger.info(f"""DataFrame #{i + 1}:
                                        Start time: {df[t].iloc[0]:.2f} seconds
                                        End time: {df[t].iloc[-1]:.2f} seconds
                                        Samples: {len(df)}""")
    if start_time >= end_time:
        logger.warning(
            "No overlapping time range between DataFrames; output DataFrame is empty."
        )
        return pd.DataFrame(columns=["common_time"])

    # Time grid consistency checks:
    for i, (df, t) in enumerate(zip(_df_list, t_col_list)):
        monotonic = np.all(np.diff(df[t]) > 0)
        if not monotonic:
            logger.warning(
                f"Time array is not monotonically increasing in DataFrame #{i + 1}."
            )

        # Report intervals exceeding tolerance:
        intervals = np.diff(df[t])
        mean_interval = np.mean(intervals)
        problematic_indices = np.where(np.abs(intervals - mean_interval) > tolerance)[0]
        problematic_intervals = [(idx, intervals[idx]) for idx in problematic_indices]
        if problematic_intervals:
            logger.warning(
                f"DataFrame #{i + 1}: found {len(problematic_intervals)} problematic time intervals exceeding the tolerance:"
            )
            for idx, interval in problematic_intervals:
                logger.warning(f"    Interval at index {idx} = {interval:.6f} s")

    # Generate the common time grid
    common_time = np.arange(start_time, end_time, 1 / fs)
    logger.info(
        f"New common time grid created: {len(common_time)} samples from {start_time:.2f}s to {end_time:.2f}s"
    )

    # Application of preprocessors:
    for i, (df, proc) in enumerate(zip(_df_list, preprocessors)):
        if proc is not None:
            logger.info(f"Applying pre-processor {proc} to DataFrame #{i + 1}")
            _df_list[i] = proc(df)
            logger.info(f"Columns: {list(_df_list[i].columns)}")

    # Downsampling and interpolation to the common grid:
    for i, (df, t) in enumerate(zip(_df_list, t_col_list)):
        logger.info(f"Resampling DataFrame #{i + 1} based on column '{t}'...")
        df_interp = pd.DataFrame({"common_time": common_time})
        for col in df.columns:
            col_name = col + f"_{i + 1}" if suffixes else col
            df_interp[col_name] = np.interp(common_time, df[t], df[col])
        df_interp_list.append(df_interp)

    # Merging all data streams to single DataFrame:
    logger.info("Merging...")
    resampled_df = pd.concat(df_interp_list, axis=1).loc[
        :, ~pd.concat(df_interp_list, axis=1).columns.duplicated()
    ]
    logger.info("Done.")

    return resampled_df


def multi_file_timeseries_resampler(
    file_list: List[str],
    fs_list: List[float],
    fs: float,
    start_time: Optional[float] = 0.0,
    duration_hours: Optional[float] = None,
    timeshifts: Optional[List[float]] = None,
    delimiter_list: Optional[List[str]] = None,
    t_col_list: Optional[List[str]] = None,
    tolerance: Optional[float] = 0.1,
    preprocessors: Optional[List[callable]] = None,
    suffixes: bool = False,
) -> pd.DataFrame:
    """
    Loads time-series data from multiple files, truncates to the maximum overlapping time window,
    and resamples to a common time grid by interpolating each data stream.

    Parameters
    ----------
    file_list : List[str]
        A list of file paths for the input data files. Each file must contain time-series data sampled with the
        sampling frequencies provided in `fs_list`.

    fs_list : List[float]
        A list of sampling frequencies (in Hz) corresponding to each file in `file_list`. The length of `fs_list` must
        match the length of `file_list`, and each value must be positive.

    fs : float
        Sampling frequency (Hz) for the common time grid. Must be positive.

    start_time : float, optional
        The starting time (in seconds) from which the data will be extracted in each file. Default is 0.0 seconds.

    timeshifts : List[float], optional
        Time shifts (in seconds) to apply to each data stream.

    delimiter_list : List[str], optional
        A list of delimiters to be used for reading each file. If not provided, a space (' ') will be assumed as the
        delimiter for all files.

    t_col_list : List[str], optional
        Column names representing time for each DataFrame in `file_list`. Defaults to 'time' for all.

    tolerance : float, optional
        The allowable deviation (in seconds) from the mean sampling interval. Values
        exceeding this tolerance trigger warnings.

    preprocessors: List[callable], optional
        Pre-processing functions to apply to each data stream before resampling. Defaults to None for all.

    suffixes : bool, default=False
        If True, suffixes the column names of each DataFrame with its index in `file_list` to avoid name conflicts.

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing the resampled time-series data aligned on a common time grid.
    """

    # Load the time series data from multiple files using the loader function
    df_list = multi_file_timeseries_loader(
        file_list=file_list,
        fs_list=fs_list,
        start_time=start_time,
        duration_hours=duration_hours,
        timeshifts=timeshifts,
        delimiter_list=delimiter_list,
    )

    # Resample the loaded data to a common time grid using the resampler function
    resampled_df = resample_to_common_grid(
        df_list=df_list,
        fs=fs,
        t_col_list=t_col_list,
        tolerance=tolerance,
        preprocessors=preprocessors,
        suffixes=suffixes,
    )

    return resampled_df


# BSD 3-Clause License
#
# Copyright (c) 2022, California Institute of Technology and
# Max Planck Institute for Gravitational Physics (Albert Einstein Institute)
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
def lagrange_taps(shift_fracs, halfp):
    """Computes the coefficients for a Lagrange fractional delay filter.

    This function calculates the FIR filter tap coefficients for a centered
    Lagrange interpolating polynomial of order `2 * halfp - 1`. It uses an
    efficient, vectorized formulation to compute coefficients for multiple
    fractional delays simultaneously.

    This is a core utility function used by `timeshift`.

    Parameters
    ----------
    shift_fracs : np.ndarray
        An array of fractional time shifts, with each value typically in the
        range [0, 1). Each value corresponds to the sub-sample shift for
        which a set of filter coefficients is required.
    halfp : int
        The number of filter taps on each side of the interpolation center.
        This is related to the filter order `p` by `halfp = (p + 1) // 2`.
        The total number of taps will be `2 * halfp`.

    Returns
    -------
    np.ndarray
        A 2D array of the calculated Lagrange coefficients. The shape of the
        array is `(N, 2 * halfp)`, where `N` is the number of fractional
        shifts in `shift_fracs`. Each row contains the `2 * halfp` filter
        taps for the corresponding shift in `shift_fracs`.
    """
    num_taps = 2 * halfp
    taps = np.zeros((num_taps, shift_fracs.size), dtype=np.float64)

    # --- Case 1: Linear Interpolation (order=1, halfp=1) ---
    if halfp == 1:
        taps[0] = 1 - shift_fracs
        taps[1] = shift_fracs
        return taps.T

    # --- Case 2: Higher-Order Interpolation (halfp > 1) ---
    # The algorithm is structured to build parts of the formula and then apply
    # common factors to all taps at the end.

    # This running 'factor' is used to set the initial values for the outer taps.
    # These values are NOT final until the last two multiplications are applied.
    factor = np.ones(shift_fracs.size, dtype=np.float64)
    factor *= shift_fracs * (1 - shift_fracs)

    for j in range(1, halfp):
        # Iteratively build the product term for the outer taps
        factor *= (-1) * (1 - j / halfp) / (1 + j / halfp)
        taps[halfp - 1 - j] = factor / (j + shift_fracs)
        taps[halfp + j] = factor / (j + 1 - shift_fracs)

    # Set the initial values for the two central taps.
    taps[halfp - 1] = 1 - shift_fracs
    taps[halfp] = shift_fracs

    # Now, apply the remaining common factors to ALL taps to finalize them.
    # First common factor: product over (1 - (d/j)^2)
    for j in range(2, halfp):
        taps *= 1 - (shift_fracs / j) ** 2

    # Second common factor: final normalization term.
    taps *= (1 + shift_fracs) * (1 - shift_fracs / halfp)

    return taps.T


def timeshift(data, shifts, order=31):
    """Time-shift data using high-order Lagrange interpolation.

    This function applies a fractional time delay or advancement to a signal
    by convolving it with a time-varying finite-impulse response (FIR)
    Lagrange interpolation filter.

    Parameters
    ----------
    data : np.ndarray
        The 1D input signal to be shifted.
    shifts : Union[float, np.ndarray]
        The desired time shift(s) in units of samples. A positive value
        delays the signal (shifts it to the right), while a negative value
        advances it. Can be a single float for a constant shift or an array
        of the same size as `data` for a time-varying shift.
    order : int, optional
        The order of the Lagrange interpolator, which must be an odd integer.
        Higher orders provide better accuracy but have higher computational
        cost and longer filter ringing. Defaults to 31.

    Returns
    -------
    Union[float, np.ndarray]
        The time-shifted signal.

    Raises
    ------
    ValueError
        If `order` is not an odd integer or if `data` and `shifts` have
        mismatched shapes for a time-varying shift.

    Notes
    -----
    The implementation is fully vectorized using NumPy for high performance,
    avoiding slow Python loops even for time-varying shifts.
    """
    if order % 2 == 0:
        raise ValueError(f"`order` must be an odd integer (got {order})")

    data = np.asarray(data)
    shifts = np.asarray(shifts)

    # --- Handle trivial cases ---
    if data.size <= 1:
        logger.debug("Input data is scalar or empty, returning as is.")
        return data.item() if data.size == 1 else data
    if np.all(shifts == 0):
        logger.debug("Time shifts are all zero, returning original data.")
        return data

    logger.debug("Time shifting data samples (order=%d)", order)

    halfp = (order + 1) // 2
    # num_taps = 2 * halfp

    shift_ints = np.floor(shifts).astype(int)
    shift_fracs = shifts - shift_ints

    logger.debug("Computing Lagrange coefficients")
    taps = lagrange_taps(shift_fracs, halfp)

    # --- Constant Shift Path (Optimized for a single shift value) ---
    if shifts.size == 1:
        logger.debug("Constant shifts, using correlation method")
        shift_int = shift_ints.item()

        i_min = shift_int - (halfp - 1)
        i_max = shift_int + halfp + data.size

        if i_max - 1 < 0:
            return np.repeat(data[0], data.size)
        if i_min > data.size - 1:
            return np.repeat(data[-1], data.size)

        pad_left = max(0, -i_min)
        pad_right = max(0, i_max - data.size)
        logger.debug("Padding data (left=%d, right=%d)", pad_left, pad_right)
        data_trimmed = data[max(0, i_min) : min(data.size, i_max)]
        data_padded = np.pad(data_trimmed, (pad_left, pad_right), mode="edge")

        logger.debug("Computing correlation product")
        return np.correlate(data_padded, taps[0], mode="valid")

    # --- Time-Varying Shift Path ---
    if data.size != shifts.size:
        raise ValueError(
            f"`data` and `shift` must be of the same size (got {data.size}, {shifts.size})"
        )

    logger.debug("Time-varying shifts, using sliding window view")
    indices = np.clip(
        np.arange(data.size) + shift_ints, -(halfp + 1), data.size + (halfp - 1)
    )
    padded = np.pad(data, 2 * halfp)
    slices = np.lib.stride_tricks.sliding_window_view(padded, 2 * halfp)
    slices = slices[indices + 2 * halfp - (halfp - 1)]
    logger.debug("Computing matrix-vector product")
    return np.einsum("ij,ij->i", taps, slices)

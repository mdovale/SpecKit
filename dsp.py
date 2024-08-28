""" This module contains signal processing functions.

Miguel Dovale (Hannover, 2024)
"""
import numpy as np
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
from pytdi.dsp import timeshift

def numpy_detrend(x, order=1):
    """
    Detrend an input signal using linear regression with Numpy.

    Args:
        y (numpy.ndarray): The input signal to be detrended.
        order (int): The order of the polynomial fit

    Returns:
        The detrended signal.
    """
    t = range(len(x))
    p = np.polyfit(t, x, order)
    residual = x - np.polyval(p, t)
    return residual

def crop_data(x, y, xmin, xmax):
    """ Crop data.

    Args:
        x: data in x
        y: data in y
        xmin: lower bound of x
        xmax: upper bound of x
    """
    x = np.array(x)
    y = np.array(y)

    # Create a boolean mask for the range condition
    mask = (x >= xmin) & (x <= xmax)
    
    # Apply the mask to both x and y arrays and return
    return x[mask], y[mask]

def truncation(x, n_trunc):
    """ Truncate both ends of time-series data.

    Args:
        x: data to truncate
        n_trunc: number of points to be truncated at each end of array
    """
    if n_trunc > 0:
        return x[n_trunc:-n_trunc]
    else:
        return x

def df_timeshift(df, fs, seconds, columns=None, truncate=None):
    """ Time shift an entire DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        fs (float): The sampling frequency of the data.
        seconds (float): Amount of seconds to shift the data by.
        columns (list or None): List of columns to shift. If None, all columns are shifted.
        truncate (bool or int or None): If True, truncate the resulting DataFrame based on the shift. 
                                        If int, specify the exact number of rows to truncate at both ends.

    Returns:
        pd.DataFrame: The time-shifted DataFrame.
    """
    

    df_shifted = df.copy()

    if columns is None:
        columns = df.columns

    for c in columns:
        df_shifted[c] = timeshift(np.array(df[c]), seconds*fs)

    if truncate is not None:
        if isinstance(truncate, bool):
            n_trunc = int(2*abs(seconds*fs))
        else:
            n_trunc = int(truncate)
        df_shifted = df_shifted.iloc[n_trunc:-n_trunc]

    return df_shifted

def integral_rms(fourier_freq, asd, pass_band=[-np.inf,np.inf]):
    """ Compute the RMS as integral of the ASD.
    
    Args:
        fourier_freq: fourier frequency (Hz)
        asd: amplitude spectral density from which RMS is computed
        pass_band: [0] = min, [1] = max
    """

    integral_range_min = max(np.min(fourier_freq), pass_band[0])
    integral_range_max = min(np.max(fourier_freq), pass_band[1])
    f_tmp, asd_tmp = crop_data(fourier_freq, asd, integral_range_min, integral_range_max)
    integral_rms2 = cumtrapz(asd_tmp**2, f_tmp, initial=0)
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
    edge : bool, optional, default=True
        If True, consider peaks that are on the boundary of the spectrum.
    freq_band : tuple of (float, float), optional
        Frequency band to search for peaks, specified as (low_freq, high_freq). Only frequencies within this range are considered.
    rtol : float, optional, default=1e-2
        Relative tolerance for identifying flat peaks.

    Returns
    -------
    peak_frequencies : ndarray
        Array of frequencies at which peaks were detected.
    peak_measurements : ndarray
        Array of measurement values at the detected peak frequencies.

    Notes
    -----
    The function first applies an optional frequency band filter and then manually detects peaks by identifying points that are higher than their immediate neighbors.
    Peaks that do not meet the specified carrier-to-noise density ratio are discarded. The function returns the frequencies and measurements of the detected peaks.

    Example
    -------
    >>> frequency = np.linspace(0, 100, 1000)
    >>> measurement = np.sin(frequency) + 0.5 * np.random.randn(1000)
    >>> peaks_freq, peaks_meas = peak_finder(frequency, measurement, cnr=5, edge=True, freq_band=(10, 90))
    >>> print("Peak Frequencies:", peaks_freq)
    >>> print("Peak Measurements:", peaks_meas)
    """
    def noise_model(x, a, b, alpha):
        return a + b * x**alpha

    frequency = np.array(frequency)
    measurement = np.array(measurement)
    
    if freq_band is not None:
        low_freq, high_freq = freq_band
        mask = (frequency >= low_freq) & (frequency <= high_freq)
        frequency = frequency[mask]
        measurement = measurement[mask]

    if len(frequency) == 0:
        return
    
    # Initial peak finding
    peaks = []
    i = 1
    while i < len(measurement) - 1:
        if measurement[i - 1] < measurement[i] > measurement[i + 1]:
            peaks.append(i)
        elif (measurement[i - 1] < measurement[i]) and np.isclose(measurement[i], measurement[i + 1], rtol=rtol):
            start = i
            while i < len(measurement) - 1 and np.isclose(measurement[i], measurement[i + 1], rtol=rtol):
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
    popt, _ = curve_fit(noise_model, frequency[non_peak_mask], measurement[non_peak_mask])
    noise_level = noise_model(frequency, *popt)
    
    # Calculate CNR threshold
    cnr_threshold = noise_level * (10 ** (cnr / 10))
    valid_peaks = [p for p in peaks if measurement[p] > cnr_threshold[p]]

    peak_frequencies = frequency[valid_peaks]
    peak_measurements = measurement[valid_peaks]
    
    return peak_frequencies, peak_measurements

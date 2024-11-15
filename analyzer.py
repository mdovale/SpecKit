""" This module contains spectral analysis functions.

Miguel Dovale (Hannover, 2024)
E-mail: spectools@pm.me
"""
import numpy as np
from spectools.lpsd import ltf

def SISO_optimal_spectral_analysis(input, output, fs, band=None, olap=None, bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, scheduler=None, adjust_Jdes=False):
    """
    Optimal spectral analysis on a Single-Input Single-Output system.

    Parameters
    ----------
    input: array-like
        The input time series.

    output: array-like
        The output time series.

    fs: float
        The sampling frequency of the input and output time series.

    band: iterable of two floats
        Frequency band to restrict computations to.

    olap: float or str, optional
        Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".

    bmin: int, optional
        Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.

    Lmin: int, optional
        The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.

    Jdes: int, optional
        Desired number of Fourier frequencies. Default is None.

    Kdes: int, optional
        Desired number of segments to be averaged. Default is None.

    order: int, optional
        -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.

    win: str, optional
        Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.

    psll: float, optional
        Target peak side-lobe level supression.  Default is None.

    pool: multiprocessing.Pool instance, optional
        Allows performing parallel computations. Default is None.

    scheduler: str or callable, optional
        Scheduler algorithm to use (e.g., 'lpsd', 'ltf', 'new_ltf'). Default is None.

    adjust_Jdes: bool, optional 
        Whether to force the scheduler to produce the desired number of bins. Default is False.

    Returns
    -------
    np.ndarray
        The spectrum of Fourier frequencies at which the output is calculated.

    np.ndarray
        The amplitude spectral density of the output with the influence of the input subtracted
        via the optimal spectral analysis method.
    """

    csd = ltf([input, output], fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, pool, scheduler, adjust_Jdes)

    S11 = csd.Gxx
    S22 = csd.Gyy
    S12 = csd.Gxy
    S21 = np.conj(csd.Gxy)
    H = S12 / S11
    optimal_asd = np.abs(np.sqrt(S22 + (H)*np.conj(H)*S11 - np.conj(H)*S12 - H*S21))

    return csd.f, optimal_asd
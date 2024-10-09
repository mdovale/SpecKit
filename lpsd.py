""" This module contains functions to perform spectral analysis on time series 
data using the LTFObject class.

Miguel Dovale, AEI (2024)
"""
import sys
import numpy as np
from spectools.ltf import LTFObject

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)

version = 1.0

def lpsd(x, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None,\
         object_return=False, pool=None, verbose=False):
    """Main function to perform LPSD/LTF algorithm on data.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        object_return (bool, optional): Whether to return an instance of LTFObject or a tuple of lists containing the traditional output. Default is False.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
    if object_return is True:
        LTFObject: Instance of LTFObject.
    else:
        f (list): Array of Fourier frequencies.
        XX or XY (list): Array of power spectrum or cross spectrum.
        Gxx or Gxy (list): Array of power spectral density or cross spectral density.
        np.nan
        Gxx_dev or Gxy_dev (list): Array of standard deviation of power spectral density or cross spectral density.
    """
    ltf_obj = LTFObject(data=x, fs=fs, verbose=verbose)

    ltf_obj.load_params(verbose, default=False, fs=fs, olap=olap, bmin=bmin, Lmin=Lmin, Jdes=Jdes, Kdes=Kdes, order=order, win=win, psll=psll)

    if verbose: logging.info(f"Attempting to schedule {ltf_obj.Jdes} frequencies...")

    ltf_obj.calc_ltf_plan(band)

    if verbose: logging.info(f"Scheduler returned {ltf_obj.nf} frequencies.")

    if verbose: logging.info("Computing {} frequencies, discarding first {} bins".format(ltf_obj.nf, ltf_obj.bmin))

    ltf_obj.calc_lpsd(pool=pool, verbose=verbose)

    if object_return:
        return ltf_obj
    else:
        return ltf_obj.legacy_return()

def ltf(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Main function to perform LPSD/LTF algorithm on data. Returns an LTFObject instance.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        LTFObject: Instance of LTFObject.
    """
    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    return ltf_obj


def lpsd_legacy(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None):
    """Main function to perform LPSD/LTF algorithm on data. Returns the "traditional" output tuple.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        XX or XY (list): Array of power spectrum or cross spectrum.
        Gxx or Gxy (list): Array of power spectral density or cross spectral density.
        np.nan
        Gxx_dev or Gxy_dev (list): Array of standard deviation of power spectral density or cross spectral density.
    """
    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=False)
    
    return ltf_obj.legacy_return()
    
def asd(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the amplitude spectral density.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        asd (list): Array of amplitude spectral density.
    """
    x = np.asarray(data)
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, np.sqrt(ltf_obj.Gxx)

def psd(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the power spectral density.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        psd (list): Array of power spectral density.
    """
    x = np.asarray(data)
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, ltf_obj.Gxx

def ps(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the power spectrum.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        psd (list): Array of power spectral density.
    """
    x = np.asarray(data)
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, ltf_obj.G

def csd(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the cross spectral density.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        csd (list): Array of cross spectral density.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, ltf_obj.Gxy

def tf(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the transfer function.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        tf (list): Array of transfer function estimate.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, ltf_obj.Hxy

def cf(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the coupling coefficient.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        cf (list): Array of coupling coefficient.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, np.abs(ltf_obj.Hxy)

def coh(data, fs, band=None, olap="default", bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD/LTF algorithm on data and return the coherence or cross-correlation.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        band (iterable of two floats): Frequency band to restrict computations to.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        bmin (int, optional): Minimum bin number to be used. The optimal value depends on the chosen window function, with typical values between 1 and 8. Default is None.
        Lmin (int, optional): The smallest allowable segment length to be processed. Of special use in multi-channel applications which have a delay between their signal contents. Default is None.
        Jdes (int, optional): Desired number of Fourier frequencies. Default is None.
        Kdes (int, optional): Desired number of segments to be averaged. Default is None.
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        f (list): Array of Fourier frequencies.
        coh (list): Array of coherence.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = lpsd(data, fs, band, olap, bmin, Lmin, Jdes, Kdes, order, win, psll, object_return=True, pool=pool, verbose=verbose)
    
    return ltf_obj.f, ltf_obj.coh

def ltf_single_bin(x, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, verbose=False):
    """Main function to perform the LPSD algorithm for a single frequency bin.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        LTFObject: Instance of LTFObject.
    """
    ltf_obj = LTFObject(data=x, fs=fs, verbose=verbose)

    ltf_obj.load_params(verbose, default=False, fs=fs, olap=olap, order=order, win=win, psll=psll)

    ltf_obj.calc_lpsd_single_bin(freq, fres, L)

    return ltf_obj

def asd_single_bin(data, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, verbose=False):
    """Perform the LPSD algorithm on a single frequency bin and return the amplitude spectral density.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        asd (float): Amplitude spectral density.
    """
    x = np.asarray(data)
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(data, fs, freq, fres, L, olap, order, win, psll, verbose=verbose)
    
    return np.sqrt(ltf_obj.Gxx)

def psd_single_bin(data, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD algorithm on a single frequency bin and return the power spectral density.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        psd (float): Power spectral density.
    """
    x = np.asarray(data)
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(data, fs, freq, fres, L, olap, order, win, psll, verbose=verbose)
    
    return ltf_obj.Gxx

def csd_single_bin(data, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD algorithm on a single frequency bin and return the cross spectral density.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        csd (list): Cross spectral density.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(data, fs, freq, fres, L, olap, order, win, psll, verbose=verbose)
    
    return ltf_obj.Gxy

def tf_single_bin(data, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD algorithm on a single frequency bin and return the transfer function estimate.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        tf (list): Transfer function estimate.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(data, fs, freq, fres, L, olap, order, win, psll, verbose=verbose)
    
    return ltf_obj.Hxy

def cf_single_bin(data, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD algorithm on a single frequency bin and return the coupling coefficient.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        cf (list): Coupling coefficient.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(data, fs, freq, fres, L, olap, order, win, psll, verbose=verbose)
    
    return np.abs(ltf_obj.Hxy)

def coh_single_bin(data, fs, freq, fres=None, L=None, olap="default", order=None, win=None, psll=None, pool=None, verbose=False):
    """Perform the LPSD algorithm on a single frequency bin and return the coherence or cross-correlation.

    Args:
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is None.

    Returns:
        coh (list): Coherence.
    """
    x = np.asarray(data)
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(data, fs, freq, fres, L, olap, order, win, psll, verbose=verbose)
    
    return ltf_obj.coh
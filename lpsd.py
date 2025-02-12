""" This module contains functions to perform spectral analysis on time series 
data using the LTFObject class.

Miguel Dovale (Hannover, 2024)
E-mail: spectools@pm.me
"""
from ._config import *
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
__version__ = version

def lpsd(x, fs, band=None, olap=None, bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None,\
         return_type='object', pool=None, scheduler=None, adjust_Jdes=False, verbose=False):
    """Main function to perform LPSD/LTF algorithm on data.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Parameters
    ----------
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
        return_type (str, optional): Specifies the return of this function ("object" for LTFObject, "plan" for scheduler output, otherwise it returns a tuple). Default is "object".
        pool (multiprocessing.Pool instance, optional): Allows performing parallel computations. Default is None.
        scheduler (str or callable, optional): Scheduler algorithm to use (e.g., 'lpsd', 'ltf', 'new_ltf'). Default is None.
        adjust_Jdes (bool, optional): Whether to force the scheduler to produce the desired number of bins. Default is False.
        verbose (bool, optional): Whether to print out some useful information. Default is False.

    Returns
    -------
    if return_type == "plan":
        f (np.ndarray): Frequency vector in Hz (array of floats).
        r (np.ndarray): For each frequency, resolution bandwidth in Hz (array of floats).
        b (np.ndarray): For each frequency, fractional bin number (array of floats).
        L (np.ndarray): For each frequency, length of the segments to be processed (array of ints).
        K (np.ndarray): For each frequency, number of segments to be processed (array of ints).
        D (list): A list that, for each frequency, contains a list with the starting indices (int) of each segment to be processed.
        O (np.ndarray): For each frequency, actual fractional overlap between segments (array of floats).
        nf (int): Total number of frequencies produced.
    elif return_type == "object":
        LTFObject: Instance of LTFObject.
    else:
        f (np.ndarray): Array of Fourier frequencies.
        XX or XY (np.ndarray): Array of power spectrum or cross spectrum.
        Gxx or Gxy (np.ndarray): Array of power spectral density or cross spectral density.
        np.nan
        Gxx_dev or Gxy_dev (np.ndarray): Array of standard deviation of power spectral density or cross spectral density.
    """
    ltf_obj = LTFObject(data=x, fs=fs, olap=olap, bmin=bmin, Lmin=Lmin, Jdes=Jdes, Kdes=Kdes, order=order, win=win, psll=psll, scheduler=scheduler, verbose=verbose)

    if adjust_Jdes:
        if verbose: logging.info(f"Forcing {ltf_obj.Jdes} frequencies...")
        ltf_obj.adjust_Jdes_to_target_nf(ltf_obj.Jdes)
    else:
        if verbose: logging.info(f"Attempting to schedule {ltf_obj.Jdes} frequencies...")
        ltf_obj.calc_plan()

    if band is not None:
        logging.info(f"Restricting frequencies to the desired band.")
        ltf_obj.filter_to_band(band)

    if return_type == 'plan':
        return ltf_obj.get_plan()

    if verbose: logging.info("Computing {} frequencies, discarding frequency bins with b < {}".format(ltf_obj.nf, ltf_obj.bmin))

    ltf_obj.calc_lpsd(pool=pool, verbose=verbose)

    if verbose: logging.info("Done.")

    if return_type == 'object':
        return ltf_obj
    else:
        return ltf_obj.legacy_return()

def ltf(*args, **kwargs):
    """Main function to perform LPSD/LTF algorithm on data. Returns an LTFObject instance.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to lpsd.
        **kwargs (dict, optional): Additional keyword arguments passed to lpsd.

    Returns
    -------
        LTFObject: Instance of LTFObject.
    """
    ltf_obj = lpsd(*args, **kwargs, return_type='object')
    return ltf_obj

def lpsd_legacy(*args, **kwargs):
    """Main function to perform LPSD/LTF algorithm on data. Returns the "traditional" output tuple.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        XX or XY (np.ndarray): Array of power spectrum or cross spectrum.
        Gxx or Gxy (np.ndarray): Array of power spectral density or cross spectral density.
        np.nan
        Gxx_dev or Gxy_dev (np.ndarray): Array of standard deviation of power spectral density or cross spectral density.
    """
    ltf_obj = ltf(*args, **kwargs, scheduler="ltf", verbose=False)
    
    return ltf_obj.legacy_return()
    
def asd(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the amplitude spectral density.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        asd (np.ndarray): Array of amplitude spectral density.
    """
    x = np.asarray(args[0])
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, np.sqrt(ltf_obj.Gxx)

def psd(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the power spectral density.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        psd (np.ndarray): Array of power spectral density.
    """
    x = np.asarray(args[0])
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, ltf_obj.Gxx

def ps(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the power spectrum.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        psd (np.ndarray): Array of power spectral density.
    """
    x = np.asarray(args[0])
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, ltf_obj.G

def csd(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the cross spectral density.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        csd (np.ndarray): Array of cross spectral density.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, ltf_obj.Gxy

def tf(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the transfer function.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        tf (np.ndarray): Array of transfer function estimate.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, ltf_obj.Hxy

def cf(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the coupling coefficient.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        cf (np.ndarray): Array of coupling coefficient.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, np.abs(ltf_obj.Hxy)

def coh(*args, **kwargs):
    """Perform the LPSD/LTF algorithm on data and return the coherence or cross-correlation.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf.

    Returns
    -------
        f (np.ndarray): Array of Fourier frequencies.
        coh (np.ndarray): Array of coherence.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf(*args, **kwargs)
    
    return ltf_obj.f, ltf_obj.coh

def ltf_single_bin(x, fs, freq, fres=None, L=None, olap=None, order=None, win=None, psll=None, verbose=False):
    """Main function to perform the LPSD algorithm for a single frequency bin.

    Computes power spectrum and power spectral density on 1-dimensional arrays. 
    Alternatively, computes cross spectrum and cross spectral density on 2-dimensional arrays.
    It also computes an estimate of the variance of the spectrum based on the Welford algorithm.
    Takes in an optional multiprocessing.Pool argument to enable parallel computations.

    Parameters
    ----------
        data (array-like): Input data.
        fs (float): Sampling frequency.
        freq (float): Fourier frequency.
        fres (float): Frequency resolution.
        olap (float or str, optional): Overlap factor ("default" will use an optimal overlap based on the window function). Default is "default".
        order (int, optional): -1: no detrending, 0: remove mean, n >= 1: remove an n-th order polynomial fit. Default is None.
        win (str, optional): Window function to be used (e.g., "Kaiser", "Hanning"). Default is None.
        psll (float, optional): target peak side-lobe level supression.  Default is None.
        verbose (bool, optional): Whether to print out some useful information. Default is False.

    Returns
    -------
        LTFObject: Instance of LTFObject.
    """
    ltf_obj = LTFObject(data=x, fs=fs, olap=olap, order=order, win=win, psll=psll, verbose=verbose)

    ltf_obj.calc_lpsd_single_bin(freq, fres, L)

    return ltf_obj

def asd_single_bin(*args, **kwargs):
    """Perform the LPSD algorithm on a single frequency bin and return the amplitude spectral density.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf_single_bin.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf_single_bin.

    Returns
    -------
        asd (float): Amplitude spectral density.
    """
    x = np.asarray(args[0])
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(*args, **kwargs)
    
    return np.sqrt(ltf_obj.Gxx)

def psd_single_bin(*args, **kwargs):
    """Perform the LPSD algorithm on a single frequency bin and return the power spectral density.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf_single_bin.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf_single_bin.

    Returns
    -------
        psd (float): Power spectral density.
    """
    x = np.asarray(args[0])
    if len(x.shape) != 1:
        logging.error("Input array size must be 1xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(*args, **kwargs)
    
    return ltf_obj.Gxx

def csd_single_bin(*args, **kwargs):
    """Perform the LPSD algorithm on a single frequency bin and return the cross spectral density.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf_single_bin.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf_single_bin.

    Returns
    -------
        csd (float): Cross spectral density.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(*args, **kwargs)
    
    return ltf_obj.Gxy

def tf_single_bin(*args, **kwargs):
    """Perform the LPSD algorithm on a single frequency bin and return the transfer function estimate.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf_single_bin.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf_single_bin.

    Returns
    -------
        tf (float): Transfer function estimate.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(*args, **kwargs)
    
    return ltf_obj.Hxy

def cf_single_bin(*args, **kwargs):
    """Perform the LPSD algorithm on a single frequency bin and return the coupling coefficient.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf_single_bin.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf_single_bin.

    Returns
    -------
        cf (float): Coupling coefficient.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(*args, **kwargs)
    
    return np.abs(ltf_obj.Hxy)

def coh_single_bin(*args, **kwargs):
    """Perform the LPSD algorithm on a single frequency bin and return the coherence or cross-correlation.

    Parameters
    ----------
        *args (tuple): Positional arguments passed to ltf_single_bin.
        **kwargs (dict, optional): Additional keyword arguments passed to ltf_single_bin.

    Returns
    -------
        coh (float): Coherence.
    """
    x = np.asarray(args[0])
    if not(len(x.shape) == 2 and ((x.shape[0] == 2) or (x.shape[1] == 2))):
        logging.error("Input array size must be 2xN")
        sys.exit(-1)

    ltf_obj = ltf_single_bin(*args, **kwargs)
    
    return ltf_obj.coh
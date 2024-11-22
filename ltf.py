""" This module contains the LTFObject class with methods for spectral analysis
of digitized time series.

This class implements a very fast pure-Python implementation of the LPSD/LTF algorithm.

Reference: LPSD algorithm by Gerhard Heinzel and Michael Troebs.
https://doi.org/10.1016/j.measurement.2005.10.010

Miguel Dovale (Hannover, 2024)
E-mail: spectools@pm.me
"""
import sys
import copy
import numpy as np
import control as ct
import scipy.signal.windows as windows
import pandas as pd
import math
import time
from spectools.flattop import olap_dict, win_dict
from spectools.dsp import integral_rms, numpy_detrend
from spectools.aux import kaiser_alpha, kaiser_rov, round_half_up, chunker
from spectools.aux import is_function_in_dict, get_key_for_function, find_Jdes_binary_search
import matplotlib.pyplot as plt

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)

class LTFObject:
    def __init__(self, default=False, data=None, N=None, fs=None, olap=None, bmin=None, Lmin=None,\
                 Jdes=None, Kdes=None, order=None, psll=None, win=None, scheduler=None, verbose=False):
        ########################
        # LTFObject attributes #
        ########################
        self.fs = fs # Sampling frequency of input data
        self.olap = "default" # Desired fractional overlap between segments
        self.bmin = 1.0 # Fractional bin numbers < bmin will be discarded
        self.Lmin = 0 # The smallest allowable segment length to be processed
        self.Jdes = 500 # Desired number of frequencies in spectrum
        self.Kdes = 100 # Desired number of averages (control parameter)
        self.order = 0 # Detrending order (0: remove mean, 1: linear regression, 2: quadratic regression)
        self.win = np.kaiser # Window function to use
        self.psll = 200 # Peak side-lobe level
        self.scheduler = "ltf" # Scheduler algorithm to be used
        self.alpha = None # Alpha parameter for Kaiser window
        self.iscsd = None # True if it is a cross-spectrum
        self.f = None # Fourier frequency vector
        self.r = None # Frequency resolution vector
        self.m = None # Frequency bin vector
        self.L = None # Segment lengths vector
        self.K = None # Number of segments vector
        self.D = None # Starting indices of segments
        self.O = None # Actual overlap factors
        self.navg    = None  # Actual number of averages vector (should be equal to K)
        self.compute_t = None # Computation time per frequency
        self.data    = None  # Input data as columns
        self.df      = None  # DataFrame containing the results
        self.nx      = None  # Total length of the time series
        self.nf      = None  # Number of frequencies in spectrum
        self.ENBW    = None  # Equivalent noise bandwidth
        self.rms     = None  # Root mean square of the signal computed from the integral of the ASD

        # Spectral estimates
        self.XX      = None  # Spectrum (S)
        self.YY      = None  # Spectrum (S)
        self.G       = None  # Power spectrum (PS)
        self.ps      = None  # Power spectrum (PS)
        self.Gxx     = None  # Power spectral density (PSD)
        self.Gyy     = None  # Power spectral density (PSD)
        self.psd     = None  # Power spectral density (PSD)
        self.asd     = None  # Amplitude spectral density (ASD)
        self.XY      = None  # Cross spectrum (CS)
        self.cs      = None  # Cross power spectrum (CPS)
        self.Gxy     = None  # Cross spectral density (CSD)
        self.csd     = None  # Cross spectral density (CSD)
        self.cpsd    = None  # Cross power spectral density (CPSD)
        self.Hxy     = None  # Transfer function (TF)
        self.tf      = None  # Transfer function (TF)
        self.cf      = None  # Magnitude of the transfer function
        self.cf_db   = None  # Magnitude of the transfer function in dB
        self.cf_deg  = None  # Argument of the transfer function in degrees
        self.cf_rad  = None  # Argument of the transfer function in rad
        self.cf_deg_unwrapped = None # Argument of the transfer function in degrees, unwrapped
        self.cf_rad_unwrapped = None # Argument of the transfer function in degrees, unwrapped
        self.coh     = None  # Coherence
        self.ccoh    = None  # Complex coherence

        # For uncertainty estimations, see Bendat, Piersol - ISBN10:0471058874, Chapter 11
        # Standard deviations of estimates:
        self.Gxx_dev = None  # Standard deviation of PSD
        self.Gyy_dev = None  # Standard deviation of PSD
        self.Gxy_dev = None  # Standard deviation of CSD
        self.Hxy_dev = None  # Standard deviation of Hxy
        self.coh_dev = None  # Standar deviation of coherence
        self.ccoh_dev = None # Standar deviation of complex coherence
 
        # Normalized random errors:
        # A normalized random error multiplied with the value of the function estimate becomes the standard deviation. 
        self.Gxx_error = None  # Normalized random error of PSD
        self.Gyy_error = None  # Normalized random error of PSD
        self.Gxy_error = None  # Normalized random error of CSD
        self.Hxy_mag_error = None  # Normalized random error of |Hxy|
        self.Hxy_rad_error = None  # Normalized random error of arg(Hxy) in radians
        self.Hxy_deg_error = None  # Normalized random error of arg(Hxy) in degrees
        self.coh_error = None  # Normalized random error of coherence

        # Load data:
        if data is not None:
            x = np.asarray(data)
            if len(x.shape) == 2 and ((x.shape[0] == 2)or(x.shape[1] == 2)):
                self.iscsd = True
                if x.shape[0] == 2:
                    x = x.T
                if verbose: logging.info(f"Detected two channel data with length {len(x)}")
            elif len(x.shape) == 1:
                if verbose: logging.info(f"Detected one channel data with length {len(x)}")
                self.iscsd = False
            else:
                logging.error("Input array size must be 1xN or 2xN")
                sys.exit(-1)
            self.data = copy.deepcopy(x)
            self.nx = len(self.data)
        else:
            self.nx = int(N)
            self.data = np.zeros(N)

        # Load parameters:
        if default:
            self.load_defaults()

        if fs is not None: self.fs = fs
        if olap is not None: self.olap = olap
        if bmin is not None: self.bmin = bmin
        if Lmin is not None: self.Lmin = Lmin
        if Jdes is not None: self.Jdes = Jdes
        if Kdes is not None: self.Kdes = Kdes
        if order is not None: self.order = order
        if psll is not None: self.psll = psll
        if scheduler is not None: self.scheduler = scheduler

        # Load scheduler function:
        if self.scheduler == 'ltf':
            self.scheduler = ltf_plan
        elif self.scheduler == 'lpsd':
            self.scheduler = lpsd_plan
        elif self.scheduler == 'new_ltf':
            self.scheduler = new_ltf_plan
        elif callable(self.scheduler):
            self.scheduler = scheduler
        else:
            logging.error("The specified scheduler not recognized")
            sys.exit(-1)

        # Load window parameters:
        if (win=="Kaiser")or(win=="kaiser")or(win==np.kaiser)or(win==windows.kaiser)or(win==None):
            assert self.psll > 0, logging.error("Need to specify PSLL for the Kaiser window")
            self.win = np.kaiser
            self.alpha = kaiser_alpha(self.psll)
        elif (win == "Hann")or(win=="hann")or(win == "Hanning")or(win=="hanning")or(win==np.hanning)or(win==windows.hann):
            self.win = np.hanning
        elif isinstance(win, str):
            win_str = win
            if win_str in win_dict:
                self.win = win_dict[win_str]
        elif callable(win):
            self.win = win
            if is_function_in_dict(win, win_dict):
                win_str = get_key_for_function(win, win_dict)
            else:
                win_str = "Unknown"
        else:
            logging.error("This window function is not implemented, exiting...")
            sys.exit(-1)
        
        if verbose: logging.info(f"Selected window: {self.win}")
        
        # Load overlap parameters:
        if self.olap == "default":
            if self.win == np.kaiser:
                self.olap = kaiser_rov(self.alpha)
            elif self.win == np.hanning:
                self.olap = 0.5
            elif win_str in olap_dict:
                self.olap = olap_dict[win_str]
                if verbose: logging.info(f"Automatic setting of overlap for window {win_str}: {self.olap}")
            else:
                logging.warning(f"Automatic setting of overlap for window {win_str} failed, setting to 0.5")
                self.olap = 0.5

    #####################
    # LTFObject methods #
    #####################
    def load_defaults(self):
        """
        Loads default values of the parameters.
        """
        self.fs = 2.0
        self.olap = "default"
        self.bmin = 1.0
        self.Lmin = int(1)
        self.Jdes = 500
        self.Kdes = 100
        self.order = 0
        self.psll = 200
        self.win = np.kaiser
        self.scheduler = ltf_plan

    def calc_plan(self):
        """
        Uses the callable self.scheduler function to compute the frequency and time series segmentation plans.
        """
        params = {"N": self.nx, 
                  "fs": self.fs, 
                  "olap": self.olap, 
                  "bmin": self.bmin, 
                  "Lmin": self.Lmin, 
                  "Jdes": self.Jdes,
                  "Kdes": self.Kdes}

        output = self.scheduler(params)
        
        self.f = output["f"]
        self.r = output["r"]
        self.m = output["b"]
        self.L = output["L"]
        self.K = output["K"]
        self.navg = output["navg"]
        self.D = output["D"]
        self.O = output["O"]
        self.nf = output["nf"]

    def get_plan(self):
        """
        Returns the scheduler output
        """
        if self.f is None:
            self.calc_plan()
        
        return self.f, self.r, self.m, self.L, self.K, self.D, self.O, self.nf

    def filter_to_band(self, band):
        """
        Filters the scheduler output to the desired frequency band.
        """
        fmin = band[0]
        fmax = band[1]

        if not np.any((self.f >= fmin) & (self.f <= fmax)):
            logging.error("Cannot compute a spectrum in the specified frequency band")
            return
    
        mask = (self.f >= fmin) & (self.f <= fmax)
        self.f = self.f[mask]
        self.r = self.r[mask]
        self.m = self.m[mask]
        self.L = self.L[mask]
        self.K = self.K[mask]
        self.navg = self.navg[mask]
        self.D = [row for row, keep in zip(self.D, mask) if keep]
        self.O = self.O[mask]
        self.nf = len(self.f)

    def adjust_Jdes_to_target_nf(self, target_nf, min_Jdes=100, max_Jdes=5000):
        """
        Adjusts the Jdes parameter to achieve the desired number of bins.
        """
        params = {"N": self.nx, 
                  "fs": self.fs, 
                  "olap": self.olap, 
                  "bmin": self.bmin, 
                  "Lmin": self.Lmin, 
                  "Jdes": self.Jdes,
                  "Kdes": self.Kdes}

        self.Jdes = find_Jdes_binary_search(self.scheduler, params, target_nf, min_Jdes, max_Jdes)
        self.calc_plan()
        
    def info(self):
        print(f"LTF/LPSD object properties:")
        print(f"Input data shape: {np.shape(self.data)}")

    def legacy_return(self):
        """
        Legacy return.
        """
        if self.iscsd:
            return self.f, self.XY, self.Gxy, np.nan, self.Gxy_dev, self.ENBW
        else:
            return self.f, self.XX, self.Gxx, np.nan, self.Gxx_dev, self.ENBW

    def calc_lpsd_single_bin(self, freq, fres=None, L=None):
        """
        Implements the LPSD algorithm on a single frequency bin.
        
        Args:
            freq (float): Fourier frequency at which to perform DFT.
            fres (float, Optional): Desired frequency resolution.
            L (int, Optional): Desired segment length.
        """
        csd = self.iscsd

        if L is not None:
            l = int(L)
            fres = self.fs/l
            self.r = fres
        elif fres is not None:
            self.r = fres
            l = int(self.fs/fres)
        else:
            logging.error(f"You need to provide either `fres` (frequency resolution) or `L` (segment length)")
            sys.exit(-1)

        m = freq/fres

        if self.win == np.kaiser:
            window = self.win(l + 1, self.alpha*np.pi)[0:-1]
        else:
            window = self.win(l)        

        p = 1j * 2 * np.pi * m / l * np.arange(0, l)
        C = window * np.exp(p)

        MXYr = 0.0
        MXYi = 0.0
        MXY2 = 0.0
        MXX = 0.0
        MYY = 0.0
        MXX2 = 0.0
        MYY2 = 0.0

        self.navg = round_half_up(((self.nx - l) / (1 - self.olap)) / l + 1)

        if self.navg == 1:
            shift = 1.0
        else:
            shift = (float)(self.nx - l) / (float)(self.navg - 1)
        if shift < 1:
            shift = 1.0

        start = 0.0
        for j in range(self.navg):
            istart = int(round_half_up(start))
            start = start + shift

            x1s = self.data[istart:istart + l].copy()
            if csd:
                x1s = self.data[istart:istart + l, 0].copy()
                x2s = self.data[istart:istart + l, 1].copy()

            if self.order == -1:
                pass  # do nothing
            elif self.order == 0:
                x1s = x1s - np.mean(x1s)
                if csd:
                    x2s = x2s - np.mean(x2s)                    
            else:
                x1s = numpy_detrend(x1s, self.order)
                if csd:
                    x2s = numpy_detrend(x2s, self.order)
            
            rxsum = 0.0
            rysum = 0.0
            ixsum = 0.0
            iysum = 0.0

            if csd:
                rxsum = float(np.dot(np.real(C),x1s))
                rysum = float(np.dot(np.real(C),x2s))
                ixsum = float(np.dot(np.imag(C),x1s))
                iysum = float(np.dot(np.imag(C),x2s))
            else:
                rxsum = float(np.dot(np.real(C),x1s))
                rysum = rxsum
                ixsum = float(np.dot(np.imag(C),x1s))
                iysum = ixsum

            assert isinstance(rxsum, float)
            assert isinstance(rysum, float)
            assert isinstance(ixsum, float)
            assert isinstance(iysum, float)

            if (j == 0):
                # /* for XY  */
                MXYr = rysum * rxsum + iysum * ixsum
                MXYi = iysum * rxsum - rysum * ixsum
                # /* for XX  */
                MXX = rxsum * rxsum + ixsum * ixsum
                # /* for YY  */
                MYY = rysum * rysum + iysum * iysum
            else:
                # /* for XY cross - power */
                XYr = rysum * rxsum + iysum * ixsum
                XYi = iysum * rxsum - rysum * ixsum
                QXYr = XYr - MXYr
                QXYi = XYi - MXYi
                MXYr += QXYr / j
                MXYi += QXYi / j
                # /* new Qs, using new mean */
                QXYrn = XYr - MXYr
                QXYin = XYi - MXYi
                # /* taking abs to get real variance */
                MXY2 += math.sqrt((QXYr * QXYrn - QXYi * QXYin)**2 + (QXYr * QXYin + QXYi * QXYrn)**2)
                # /* for XX  */
                XX = rxsum * rxsum + ixsum * ixsum
                QXX = XX - MXX
                MXX += QXX / j
                MXX2 += QXX * (XX - MXX)
                # /* for YY  */
                YY = rysum * rysum + iysum * iysum
                QYY = YY - MYY
                MYY += QYY / j
                MYY2 += QYY * (YY - MYY)

        # /* Outputs */
        Pxyr = MXYr
        Pxyi = MXYi
        if (self.navg == 1):
            Vr = MXYr * MXYr
        else:
            Vr = MXY2 / (self.navg - 1)
        Pxx = MXX
        Pyy = MYY

        S1 = np.sum(window)
        S12 = S1 * S1
        S2 = np.sum(window**2)

        self.XY = Pxyr + 1j*Pxyi
        self.XX = Pxx
        self.YY = Pyy
        M2 = Vr
        try:
            self.Hxy = self.XY.conjugate() / (self.XX)
            self.Gxx = 2.0 * self.XX / self.fs / S2
            self.Gyy = 2.0 * self.YY / self.fs / S2
            self.Gxy = 2.0 * self.XY / self.fs / S2
            self.coh = (abs(self.XY)**2) / (self.XX * self.YY)
            self.ccoh = self.XY / math.sqrt(self.XX * self.YY)
        except ZeroDivisionError:
            self.XX = 0.0
            self.Hxy = 0.0
            self.Gxx = 0.0
            self.Gyy = 0.0
            self.Gxy = 0.0
            self.coh = 0.0
            self.ccoh = 0.0

        self.ENBW = self.fs * S2 / S12

        self.Hxy_dev = 1.0
        self.Gxy_dev = 1.0
        self.coh_dev = 1.0
        self.ccoh_dev = 1.0
        self.Gxy_error = 1.0
        self.Gxx_dev = self.Gxx/np.sqrt(self.navg)
        self.Gxx_error = 1/np.sqrt(self.navg)
        self.Gyy_dev = self.Gyy/np.sqrt(self.navg)
        self.Gyy_error = 1/np.sqrt(self.navg)
        self.cf_mag_error = 1.0
        self.cf_rad_error = 0.0
        self.cf_deg_error = 0.0
        self.coh_error = 1.0

        if (self.navg > 1) and (self.XX != 0):
            self.Hxy_dev = math.sqrt(abs((self.navg / (self.navg - 1)**2) * (self.YY / self.XX) * (1 - (abs(self.XY)**2) / (self.XX * self.YY))))
            self.Gxy_dev = math.sqrt(abs(4.0 * M2 / self.fs**2 / S2**2 / self.navg))
            self.coh_dev = math.sqrt(abs((2 * self.coh / self.navg) * (1 - self.coh)**2))
            self.ccoh_dev = math.sqrt(abs((2 * abs(self.ccoh) / self.navg) * (1 - abs(self.ccoh))**2))
            try:
                self.Gxy_error = 1 / ( math.sqrt(self.coh) * math.sqrt(self.navg))
            except ValueError:
                pass
            try:
                self.cf_mag_error = math.sqrt(1 - self.coh) / ( math.sqrt(self.coh) * math.sqrt(2*self.navg))
            except ValueError:
                pass
            try:
                # cf_rad_error = math.asin(math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*self.navg)))
                # cf_rad_error = math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*self.navg))
                self.cf_rad_error = self.Hxy_dev/abs(self.Hxy)
                self.cf_deg_error = self.cf_rad_error*180.0/np.pi/2.0
            except ValueError:
                pass
            try:
                self.coh_error = math.sqrt(2) * (1 - self.coh) / (math.sqrt(self.coh) * math.sqrt(self.navg))
            except ValueError:
                pass

        if self.iscsd: # We are computing cross-spectra:
            self.csd = copy.copy(self.Gxy) # Cross spectral density
            self.cpsd = np.sqrt(self.csd) # Cross power spectral density
            self.cs = self.csd * self.ENBW # Cross spectrum
            self.cf = np.abs(self.Hxy) # Coupling coefficient
            self.cf_db  = ct.mag2db(self.cf)  # Magnitude of the transfer function in dB
            self.cf_deg = np.angle(self.Hxy, deg=True)  # Argument of the transfer function in degrees
            self.cf_rad = np.angle(self.Hxy, deg=False)  # Argument of the transfer function in rad
        
        if np.imag(self.Gxy) == 0:  # The computed cross-spectrum is actually an auto-spectrum:
            self.Gxy = self.Gxy.real # Power spectral density
            self.psd = copy.copy(self.Gxx) # Power spectral density
            self.asd = np.sqrt(self.psd) # Amplitude spectral density
            self.G = self.psd * self.ENBW # Power spectrum
            self.ps = copy.copy(self.G) # Power spectrum

    def calc_lpsd(self, pool, verbose):
        """
        Executes calls to _calc_lpsd and gathers the output.

        Args:
            pool (multiprocessing.Pool): For parallel computation.
            verbose (bool): Prints additional information if True.

        Computes:
            self.G
            self.Gxy
            self.Gxx
            self.ENBW
            self.compute_t
        """
        csd = self.iscsd

        assert self.nf is not None

        before = time.time()
        if (pool is not None)and(pool._processes >= 2):
            """
            starmap method with equal-length chunks of shuffled frequencies
            """
            chunk_size = int(np.ceil(self.nf/pool._processes))
            frequencies = np.arange(self.nf)
            np.random.shuffle(frequencies)
            f_blocks = chunker(frequencies, chunk_size)
            if verbose:
                logging.info("{} frequency blocks to process:".format(len(f_blocks)))
                for _ in f_blocks: logging.info("Block {}".format(_))
            tasks = [(f_blocks[i], csd, verbose) for i in range(len(f_blocks))]
            r = pool.starmap(self._lpsd_core, tasks)
        else:
            r = []
            for i in range(self.nf):
                r.append(self._lpsd_core(i, csd, verbose))
        
        after = time.time()
        total_time = after - before
        if verbose: logging.info('Completed in {} seconds'.format(total_time))

        self.df = pd.concat([pd.DataFrame(chunk, columns=['i', 'XY', 'XX', 'YY', 'Gxy', 'Gxx', 'Gyy', 'Hxy', 'coh', 'ccoh', 
                                                          'Gxy_dev', 'Gxy_error', 'Hxy_dev', 'cf_mag_error', 'cf_rad_error', 'cf_deg_error', 
                                                          'coh_dev', 'ccoh_dev', 'coh_error',
                                                          'ENBW', 'navg', 'compute_t']) \
                for chunk in r], axis=0, ignore_index=True)

        self.df = self.df.sort_values('i')

        assert np.array_equal(np.array(range(self.nf)), np.array(self.df.i))

        self.XX = np.array(self.df.XX) # Spectrum
        self.YY = np.array(self.df.YY) # Spectrum
        self.XY = np.array(self.df.XY) # Cross spectrum
        self.Gxx = np.array(self.df.Gxx) # Power spectral density
        self.Gyy = np.array(self.df.Gyy) # Power spectral density
        self.Gxy = np.array(self.df.Gxy) # Cross spectral density
        self.Hxy = np.array(self.df.Hxy) # Transfer function estimate
        self.coh = np.array(self.df.coh) # Coherence
        self.ccoh = np.array(self.df.ccoh) # Complex coherence

        self.Gxx_dev = np.array(self.df.Gxx/np.sqrt(self.df.navg))
        self.Gxx_error = np.array(1/np.sqrt(self.df.navg))
        self.Gyy_dev = np.array(self.df.Gyy/np.sqrt(self.df.navg))
        self.Gyy_error = np.array(1/np.sqrt(self.df.navg))
        self.Gxy_dev = np.array(self.df.Gxy_dev)
        self.Gxy_error = np.array(self.df.Gxy_error)
        self.Hxy_dev = np.array(self.df.Hxy_dev)
        self.cf_mag_error = np.array(self.df.cf_mag_error)
        self.cf_rad_error = np.array(self.df.cf_rad_error)
        self.cf_deg_error = np.array(self.df.cf_deg_error)
        self.coh_dev = np.array(self.df.coh_dev)
        self.coh_error = np.array(self.df.coh_error)
        self.ccoh_dev = np.array(self.df.ccoh_dev)
        
        self.ENBW = np.array(self.df.ENBW) # Equivalent noise bandwidth
        self.navg = np.array(self.df.navg) # Number of averages
        self.compute_t = np.array(self.df.compute_t) # Compute time

        if self.iscsd: # We are computing cross-spectra:
            self.csd = self.Gxy.copy() # Cross spectral density
            self.cpsd = np.sqrt(self.csd) # Cross power spectral density
            self.cs = self.csd * self.ENBW # Cross spectrum
            self.tf = self.Hxy.copy() # Transfer function
            self.cf = np.abs(self.tf) # Magnitude of the transfer function
            self.cf_db  = ct.mag2db(self.cf)  # Magnitude of the transfer function in dB
            self.cf_deg = np.angle(self.tf, deg=True)  # Argument of the transfer function in degrees
            self.cf_rad = np.angle(self.tf, deg=False)  # Argument of the transfer function in rad
            self.cf_rad_unwrapped = np.unwrap(self.cf_rad) # Argument of the transfer function in degrees, unwrapped
            self.cf_deg_unwrapped = np.rad2deg(self.cf_rad_unwrapped) # Argument of the transfer function in degrees, unwrapped
        
        if np.all(np.imag(self.Gxy) == 0): # The computed cross-spectrum is actually an auto-spectrum:
            self.Gxy = self.Gxy.real
            self.psd = self.Gxx.copy() # Power spectral density
            self.asd = np.sqrt(self.psd) # Amplitude spectral density
            self.G = self.psd * self.ENBW # Power spectrum
            self.ps = self.G # Power spectrum

    def _lpsd_core(self, f_block, csd, verbose):
        """
        Implements the core LPSD algorithm.
        
        Args:
            f_block (iterable): List of frequencies at which to compute spectrum.

        Returns:
            r_block (list): List with the results at the frequencies in `f_block`.
        """

        r_block = []
        if verbose: logging.info('Processing block {}'.format(f_block))

        if type(f_block) == int:
            f_block = [f_block]

        before = time.time()

        for i in f_block:

            now = time.time()

            if self.win == np.kaiser:
                window = self.win(self.L[i] + 1, self.alpha*np.pi)[0:-1]
            else:
                window = self.win(self.L[i])        

            p = 1j * 2 * np.pi * self.m[i] / self.L[i] * np.arange(0, self.L[i])
            C = window * np.exp(p)

            MXYr = 0.0
            MXYi = 0.0
            MXY2 = 0.0
            MXX = 0.0
            MYY = 0.0
            MXX2 = 0.0
            MYY2 = 0.0

            for j in range(self.navg[i]):

                x1s = self.data[self.D[i][j]:self.D[i][j] + self.L[i]].copy()
                if csd:
                    x1s = self.data[self.D[i][j]:self.D[i][j] + self.L[i], 0].copy()
                    x2s = self.data[self.D[i][j]:self.D[i][j] + self.L[i], 1].copy()

                if self.order == -1:
                    pass  # do nothing
                elif self.order == 0:
                    x1s = x1s - np.mean(x1s)
                    if csd:
                        x2s = x2s - np.mean(x2s)                    
                else:
                    x1s = numpy_detrend(x1s, self.order)
                    if csd:
                        x2s = numpy_detrend(x2s, self.order)
                
                rxsum = 0.0
                rysum = 0.0
                ixsum = 0.0
                iysum = 0.0

                if csd:
                    rxsum = float(np.dot(np.real(C),x1s))
                    rysum = float(np.dot(np.real(C),x2s))
                    ixsum = float(np.dot(np.imag(C),x1s))
                    iysum = float(np.dot(np.imag(C),x2s))
                else:
                    rxsum = float(np.dot(np.real(C),x1s))
                    rysum = rxsum
                    ixsum = float(np.dot(np.imag(C),x1s))
                    iysum = ixsum

                assert isinstance(rxsum, float)
                assert isinstance(rysum, float)
                assert isinstance(ixsum, float)
                assert isinstance(iysum, float)

                 # /* Welford's incremental mean and variance calculation  */
                if (j == 0):
                    # /* for XY  */
                    MXYr = rysum * rxsum + iysum * ixsum
                    MXYi = iysum * rxsum - rysum * ixsum
                    # /* for XX  */
                    MXX = rxsum * rxsum + ixsum * ixsum
                    # /* for YY  */
                    MYY = rysum * rysum + iysum * iysum
                else:
                    # /* for XY cross - power */
                    XYr = rysum * rxsum + iysum * ixsum
                    XYi = iysum * rxsum - rysum * ixsum
                    QXYr = XYr - MXYr
                    QXYi = XYi - MXYi
                    MXYr += QXYr / j
                    MXYi += QXYi / j
                    # /* new Qs, using new mean */
                    QXYrn = XYr - MXYr
                    QXYin = XYi - MXYi
                    # /* taking abs to get real variance */
                    MXY2 += math.sqrt((QXYr * QXYrn - QXYi * QXYin)**2 + (QXYr * QXYin + QXYi * QXYrn)**2)
                    # /* for XX  */
                    XX = rxsum * rxsum + ixsum * ixsum
                    QXX = XX - MXX
                    MXX += QXX / j
                    MXX2 += QXX * (XX - MXX)
                    # /* for YY  */
                    YY = rysum * rysum + iysum * iysum
                    QYY = YY - MYY
                    MYY += QYY / j
                    MYY2 += QYY * (YY - MYY)

            # /* Outputs */
            Pxyr = MXYr
            Pxyi = MXYi
            if (self.K[i] == 1):
                Vr = MXYr * MXYr
            else:
                Vr = MXY2 / (self.K[i] - 1)
            Pxx = MXX
            Pyy = MYY

            S1 = np.sum(window)
            S12 = S1 * S1
            S2 = np.sum(window**2)

            XY = Pxyr + 1j*Pxyi
            XX = Pxx
            YY = Pyy
            M2 = Vr
            try:
                Hxy = XY.conjugate() / (XX)
                Gxx = 2.0 * XX / self.fs / S2
                Gyy = 2.0 * YY / self.fs / S2
                Gxy = 2.0 * XY / self.fs / S2
                coh = (abs(XY)**2) / (XX * YY)
                ccoh = XY / math.sqrt(XX * YY)
            except ZeroDivisionError:
                XX = 0.0
                Hxy = 0.0
                Gxx = 0.0
                Gyy = 0.0
                Gxy = 0.0
                coh = 0.0
                ccoh = 0.0

            ENBW = self.fs * S2 / S12

            Hxy_dev = 1.0
            Gxy_dev = 1.0
            coh_dev = 1.0
            ccoh_dev = 1.0
            Gxy_error = 1.0
            cf_mag_error = 1.0
            cf_rad_error = 0.0
            cf_deg_error = 0.0
            coh_error = 1.0

            navg = self.navg[i]

            if (navg > 1) and (XX != 0):
                Hxy_dev = math.sqrt(abs((navg / (navg - 1)**2) * (YY / XX) * (1 - (abs(XY)**2) / (XX * YY))))
                Gxy_dev = math.sqrt(abs(4.0 * M2 / self.fs**2 / S2**2 / navg))
                coh_dev = math.sqrt(abs((2 * coh / navg) * (1 - coh)**2))
                ccoh_dev = math.sqrt(abs((2 * abs(ccoh) / navg) * (1 - abs(ccoh))**2))
                try:
                    Gxy_error = 1 / ( math.sqrt(coh) * math.sqrt(navg))
                except ValueError:
                    pass
                try:
                    cf_mag_error = math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*navg))
                except ValueError:
                    pass
                try:
                    # cf_rad_error = math.asin(math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*navg)))
                    # cf_rad_error = math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*navg))
                    cf_rad_error = Hxy_dev/abs(Hxy)
                    cf_deg_error = np.rad2deg(cf_rad_error)
                except ValueError:
                    pass
                try:
                    coh_error = math.sqrt(2) * (1 - coh) / (math.sqrt(coh) * math.sqrt(navg))
                except ValueError:
                    pass

            then = time.time()

            compute_t = then-now

            r_block.append([i, XY, XX, YY, Gxy, Gxx, Gyy, Hxy, coh, ccoh,
                            Gxy_dev, Gxy_error, 
                            Hxy_dev, cf_mag_error, cf_rad_error, cf_deg_error,
                            coh_dev, ccoh_dev, coh_error, 
                            ENBW, navg, compute_t])

        after = time.time()
        block_time = after - before
        
        if verbose: logging.info('Completed {} in {} seconds'.format(f_block, block_time))

        return r_block

    def plot(self, which=None, ylabel=None, dB=False, deg=True, unwrap=True, errors=False, sigma=1, *args, **kwargs):
        # Dictionary to map 'which' to appropriate data and y-label
        plot_options = {
            'ps': (self.f, self.G, r"PS") if not self.iscsd else None,
            'asd': (self.f, self.asd, r"ASD") if not self.iscsd else None,
            'psd': (self.f, self.psd, r"PSD") if not self.iscsd else None,
            'cs': (self.f, self.cs, r"CS") if self.iscsd else None,
            'csd': (self.f, np.abs(self.csd), r"CSD") if self.iscsd else None,
            'cpsd': (self.f, self.cpsd, r"CPSD") if self.iscsd else None,
            'coh': (self.f, self.coh, r"Coherence") if self.iscsd else None,
            'cf': (self.f, self.cf, r"Coupling coefficient") if self.iscsd else None,
            'bode': (self.f, self.cf, self.cf_rad) if self.iscsd else None,
        }
        plot_errors = {
            'ps': None if not self.iscsd else None,
            'asd': self.Gxx_dev/2/self.asd if not self.iscsd else None,
            'psd': self.Gxx_dev if not self.iscsd else None,
            'cs': None if self.iscsd else None,
            'csd': self.Gxy_dev if self.iscsd else None,
            'cpsd': self.Gxy_dev/2/self.csd if self.iscsd else None,
            'coh': self.coh_dev if self.iscsd else None,
            'cf': self.Hxy_dev if self.iscsd else None,
            'bode': (self.cf_mag_error, self.cf_rad_error) if self.iscsd else None,
        }

        if which is None:
            if self.iscsd:
                which = 'bode'
            else:
                which = 'asd'

        # Retrieve plot data:
        plot_data = plot_options.get(which.lower())
        if plot_data is None:
            return
        if errors:
            error = plot_errors.get(which.lower())
        else:
            error = None
        
        with plt.rc_context({
            # Figure and axes properties
            'figure.figsize': (5, 3),            # Figure size in inches
            'figure.dpi': 150,                   # Dots per inch for the figure (controls resolution)
            'figure.facecolor': 'white',         # Background color of the figure
            'figure.edgecolor': 'white',         # Edge color of the figure
            'axes.facecolor': 'white',           # Background color of the axes
            'axes.edgecolor': 'black',           # Edge color of the axes
            'axes.linewidth': 1.0,               # Line width of the axes' frame
            'axes.grid': True,                   # Whether to show grid lines by default
            'grid.color': 'gray',                # Grid line color
            'grid.linestyle': '--',              # Grid line style
            'grid.linewidth': 0.7,               # Grid line width
            # Font and text properties
            'font.size': 8,                      # Default font size
            'font.family': 'sans-serif',         # Font family
            'font.sans-serif': ['Arial'],        # Specify sans-serif font
            'axes.labelsize': 8,                 # Font size for axes labels
            'axes.titlesize': 9,                 # Font size for the plot title
            'axes.labelcolor': 'black',          # Color of the axes labels
            'axes.titleweight': 'regular',       # Weight of the title font
            'legend.fontsize': 8,                # Font size of the legend
            'legend.frameon': True,              # Whether to draw a frame around the legend
            # Tick properties
            'xtick.labelsize': 8,                # Font size for x-axis tick labels
            'ytick.labelsize': 8,                # Font size for y-axis tick labels
            'xtick.color': 'black',              # Color of x-axis ticks
            'ytick.color': 'black',              # Color of y-axis ticks
            'xtick.direction': 'out',            # Direction of x-axis ticks ('in' or 'out')
            'ytick.direction': 'out',            # Direction of y-axis ticks
            # Lines properties
            'lines.linewidth': 1.5,              # Default line width
            'lines.color': 'black',              # Default line color
            'lines.linestyle': '-',              # Default line style ('-', '--', '-.', ':')
            'lines.marker': '',                  # Default marker for points
            'lines.markersize': 3.5,             # Marker size
            # Legend properties
            'legend.loc': 'best',                # Legend location ('best', 'upper right', etc.)
            'legend.framealpha': 1.0,            # Transparency of the legend frame
            'legend.edgecolor': 'black',         # Edge color of the legend box
            # Save figure properties
            'savefig.dpi': 300,                  # Resolution when saving figures
            'savefig.format': 'pdf',             # Default format for saving figures
            'savefig.bbox': 'tight',             # Bounding box when saving figures
            'savefig.facecolor': 'white',        # Background color for saved figures

            # Color properties
            'axes.prop_cycle': plt.cycler('color', ['#000000', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']),
        }):
            if which == 'bode':
                fig, (ax2, ax1) = plt.subplots(2,1, figsize=(5,5))
                f, cf, cf_rad = plot_data
                # Magnitude plot:
                if dB:
                    ax2.semilogx(f, ct.mag2db(cf), *args, **kwargs)
                    ax2.set_ylabel('Magnitude (dB)')
                else:
                    ax2.loglog(f, cf, *args, **kwargs)
                    ax2.set_ylabel('Magnitude')
                # Phase plot:
                if deg:
                    if unwrap:
                        ax1.semilogx(f, np.rad2deg(np.unwrap(cf_rad)), *args, **kwargs)
                        ax1.set_ylabel('Phase (deg)')
                    else:
                        ax1.semilogx(f, np.rad2deg(cf_rad), *args, **kwargs)
                        ax1.set_ylabel('Phase (deg)')
                else:
                    if unwrap:
                        ax1.semilogx(f, np.unwrap(cf_rad), *args, **kwargs)
                        ax1.set_ylabel('Phase (rad)')
                    else:
                        ax1.semilogx(f, cf_rad, *args, **kwargs)
                        ax1.set_ylabel('Phase (rad)')
                if error is not None:
                    # Magnitude plot:
                    if dB:
                        cf_lower_bound = np.maximum(cf * (1 - sigma*error[0]), 1e-10)
                        cf_upper_bound = cf * (1 + sigma*error[0])
                        cf_dB_lower = 20 * np.log10(cf_lower_bound)
                        cf_dB_upper = 20 * np.log10(cf_upper_bound)
                        ax2.fill_between(f,
                            cf_dB_lower,
                            cf_dB_upper,
                            color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                    else:
                        ax2.fill_between(f,
                            cf*(1 - sigma*error[0]),
                            cf*(1 + sigma*error[0]),
                            color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                    ax2.legend()
                    # Phase plot:
                    cf_rad_lower = sigma*error[1]
                    cf_rad_upper = sigma*error[1]
                    if deg:
                        if unwrap:
                            ax1.fill_between(f, np.rad2deg(np.unwrap(cf_rad) - cf_rad_lower), np.rad2deg(np.unwrap(cf_rad) + cf_rad_upper), 
                                            color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                        else:
                            ax1.fill_between(f, np.rad2deg(cf_rad - cf_rad_lower), np.rad2deg(cf_rad + cf_rad_upper), 
                                            color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                    else:
                        if unwrap:
                            ax1.fill_between(f, np.unwrap(cf_rad) - cf_rad_lower, np.unwrap(cf_rad) + cf_rad_upper, 
                                            color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                        else:
                            ax1.fill_between(f, cf_rad - cf_rad_lower, cf_rad + cf_rad_upper, 
                                            color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                    ax1.legend()
                ax1.set_xlim([f[0], f[-1]])
                ax2.set_xlim([f[0], f[-1]])
                fig.align_ylabels()
            else:
                fig, ax1 = plt.subplots()
                f, y_data, y_label = plot_data        
                if ylabel is not None:
                    y_label = ylabel
                ax1.set_ylabel(y_label)
                if which == 'coh':
                    ax1.semilogx(f, y_data, *args, **kwargs)
                    ax1.set_ylim(0,1.1)
                else:
                    ax1.loglog(f, y_data, *args, **kwargs)
                if error is not None:
                    ax1.fill_between(f,
                        y_data - sigma*error,
                        y_data + sigma*error,
                        color='pink', label=r'$\pm' + str(sigma) + '\sigma$')
                    ax1.legend()
                ax1.set_xlim([f[0], f[-1]])

            ax1.set_xlabel("Fourier frequency (Hz)")
            fig.tight_layout()
            if which == 'bode':
                return fig, (ax2, ax1)
            else:
                return fig, ax1
    
    def get_measurement(self, freq, which):
        """
        Evaluates the value of a result at given frequency/frequencies, using interpolation if necessary.

        Args:
            freq (float, list of floats, or numpy array): The frequency or frequencies at which to evaluate the result.
            which (str): The result to evaluate ('asd', 'psd', 'csd', etc.)

        Returns:
            float or numpy array of floats: The measurement value(s) at the requested frequency/frequencies.

        Raises:
            ValueError: If any requested frequency is outside the range of the freq array,
                        or if the 'which' parameter is invalid.
        """
        signal_options = {
            'ps': self.ps if not self.iscsd else None,
            'asd': self.asd if not self.iscsd else None,
            'psd': self.psd if not self.iscsd else None,
            'cs': self.cs if self.iscsd else None,
            'csd': self.csd if self.iscsd else None,
            'cpsd': self.cpsd if self.iscsd else None,
            'cf': self.cf if self.iscsd else None,
            'Hxy': self.Hxy if self.iscsd else None,
            'Hxy_mag': abs(self.Hxy) if self.iscsd else None,
            'Hxy_deg': np.angle(self.Hxy, deg=True) if self.iscsd else None,
            'Hxy_rad': np.angle(self.Hxy, deg=False) if self.iscsd else None,
            'coh': self.coh if self.iscsd else None,
            'm': self.m,
            'r': self.r,
            'f': self.f
        }

        signal = signal_options.get(which)

        if signal is None:
            raise ValueError(f"Invalid measurement type '{which}'.")

        freqs = np.array(self.f)
        signal = np.array(signal)

        if np.isscalar(freq):
            # Single frequency case
            if freq < freqs.min() or freq > freqs.max():
                raise ValueError(f"Requested frequency {freq} is outside the range of the frequency array.")
            measurement = np.interp(freq, freqs, signal)
            return measurement
        else:
            # Array-like frequency case
            freq = np.asarray(freq)
            if np.any((freq < freqs.min()) | (freq > freqs.max())):
                raise ValueError("One or more requested frequencies are outside the range of the frequency array.")
            measurements = np.interp(freq, freqs, signal)
            return measurements
    
    def get_rms(self):
        self.rms = integral_rms(self.f, self.asd)
        return self.rms
    
    def get_timeseries(self, fs, size):
        import warnings
        warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
        import pycbc.psd
        import pycbc.noise

        if self.iscsd:
            return
                
        pycbc_psd = pycbc.psd.read.from_numpy_arrays(self.f, self.Gxx, int(self.f[-1]/self.f[0]), self.f[0], self.f[0])
        return pycbc.noise.noise_from_psd(int(size), 1/fs, pycbc_psd).data


def lpsd_plan(params):
    """
    Original LPSD scheduler.

    Like ltf_plan, but bmin = 1 and Lmin = 1.
    """

    params["bmin"] = 1.0
    params["Lmin"] = int(1)
    return ltf_plan(params)

def ltf_plan(params):
    """
    LTF scheduler from S2-AEI-TN-3052 (Gerhard Heinzel).

    Based on the input parameters, the algorithm generates an array of 
    frequencies (f), with corresponding resolution bandwidths (r), bin 
    numbers (b), segment lengths (L), number of averages (K), and starting 
    indices (D) for subsequent spectral analysis of time series using 
    the windowed, overlapped segmented averaging method.

    The time series will then be segmented for each frequency as follows:
    [---------------------------------------------------------------------------------] total length N
    [---------] segment length L[j], starting at index D[j][0] = 0                    .
    .     [---------] segment length L[j], starting at index D[j][1]                  .
    .           [---------] segment length L[j], starting at index D[j][2]            .
    .                 [---------] segment length L[j], starting at index D[j][3]      .
    .                           ... (total of K[j] segments to average)               .
    .                                                                       [---------] segment length L[j]
                                                                                        starting at index D[j][-1]

    Inputs: 
        N (int): Total length of the input data.
        fs (float): Sampling frequency of the input data.
        olap (float): Desired fractional overlap between segments of the input data.
        bmin (float): Minimum bin number to be used (used to discard the lower bins with biased estimates due to power aliasing from negative bins).
        Lmin (int): Smallest allowable segment length to be processed (used to tackle time delay bias error in cross spectra estimation).
        Jdes (int): Desired number of frequencies to produce. This value is almost never met exactly.
        Kdes (int): Desired number of segments to be averaged. This value is almost nowhere met exactly, and is actually only used as control parameter in the algorithm to ﬁnd a compromise between conflicting goals.

    The algorithm balances several conflicting goals:
        - Desire to compute approximately Jdes frequencies.
        - Desire for those frequencies to be approximately log-spaced.
        - For each frequency, desire to have approximately `olap` fractional overlap between segments while using the full time series.

    Computes:
        f (array of float): Frequency vector in Hz.
        r (array of float): For each frequency, resolution bandwidth in Hz.
        b (array of float): For each frequency, fractional bin number.
        L (array of int): For each frequency, length of the segments to be processed.
        K (array of float): For each frequency, number of segments to be processed.
        D (array of arrays of int): For each frequency, array containing the starting indices of each segment to be processed.
        O (array of float): For each frequency, actual fractional overlap between segments.
        nf (int): Total number of frequencies produced.

    Constraints:
        f[j] = r[j] * m[j]: Definition of the non-integer bin number
        r[j] * L[j] = fs: DFT constraint
        f[j+1] = f[j] + r[j]: Local spacing between frequency bins equivalent to original WOSA method.
        L[j] <= nx: Time series segment length cannot be larger than total length of the time series
        L[j] >= Lmin: Time series segment length must be greater or equal to Lmin
        b[j] >= bmin: Discard frequency bin numbers lower or equal to bmin
        f[0] = fmin: Lowest possible frequency must be met.
        f[-1] <= fmax: Maximum possible frequency must be met.

    Internal constants:
        xov (float): Desired non-overlapping fraction, xov = 1 - olap.
        fmin (float): Lowest possible frequency, fmin = fs/nx*bmin.
        fmax (float): Maximum possible frequency (Nyquist criterion), fmax = fs/2.
        logfact (float): Constant factor that would ensure logarithmic frequency spacing, logfact = (nx/2)^(1/Jdes)-1.
        fresmin (float): The smallest possible frequency resolution bandwidth in Hz, fresmin = fs/nx.
        freslim (float): The smallest possible frequency resolution bandwidth in Hz when Kdes averages are performed, freslim = fresmin*(1+xov(Kdes-1)).

    Targets:
    1. r[j]/f[j] = x1[j] with x1[j] -> logfact:
    This targets the approximate logarithmic spacing of frequencies on the x-axis, 
    and also the desired number of frequencies Jdes.

    2. if K[j] = 1, then L[j] = nx:
    This describes the requirement to use the complete time series. In the case of K[j] > 1, the starting points of the individual segments
    can and will be adjusted such that the complete time series is used, at the expense of not precisely achieving the desired overlap.

    3. K[j] >= Kdes:
    This describes the desire to have at least Kdes segments for averaging at each frequency. As mentioned above, 
    this cannot be met at low frequencies but is easy to over-achieve at high frequencies, such that this serves only as a 
    guideline for ﬁnding compromises in the scheduler.
    """
    def round_half_up(val):
        if (float(val) % 1) >= 0.5:
            x = math.ceil(val)
        else:
            x = round(val)
        return x
    
    N = params['N']
    fs = params['fs']
    olap = params['olap']
    bmin = params['bmin']
    Lmin = params['Lmin']
    Jdes = params['Jdes']
    Kdes = params['Kdes']

    # Init constants:
    xov = (1 - olap)
    fmin = fs / N * bmin
    fmax = fs / 2
    fresmin = fs / N
    freslim = fresmin * (1 + xov * (Kdes - 1))
    logfact = (N / 2)**(1 / Jdes) - 1

    # Init lists:
    f = []
    r = []
    b = []
    L = []
    K = []
    O = []
    D = []
    navg = []

    # Scheduler algorithm:
    fi = fmin
    while fi < fmax:
        fres = fi * logfact
        if fres >= freslim:
            pass
        elif fres < freslim and (freslim * fres)**0.5 > fresmin:
            fres = (freslim * fres)**0.5
        else:
            fres = fresmin

        fbin = fi / fres
        if fbin < bmin:
            fbin = bmin
            fres = fi / fbin

        dftlen = int(round_half_up(fs / fres))
        if dftlen > N:
            dftlen = N
        if dftlen < Lmin:
            dftlen = Lmin

        nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))
        if nseg == 1:
            dftlen = N

        fres = fs / dftlen
        fbin = fi / fres

        f.append(fi)
        r.append(fres)
        b.append(fbin)
        L.append(dftlen)
        K.append(nseg)

        fi = fi + fres

    nf = len(f)

    # Compute actual averages and starting indices:
    for j in range(nf):
        L_j = int(L[j])
        L[j] = L_j
        averages = int(round_half_up(((N - L_j) / (1 - olap)) / L_j + 1))
        navg.append(averages)

        if averages == 1:
            shift = 1.0
        else:
            shift = (float)(N - L_j) / (float)(averages - 1)
        if shift < 1:
            shift = 1.0

        start = 0.0
        D.append([])
        for _ in range(averages):
            istart = int(float(start) + 0.5) if start >= 0 else int(float(start) - 0.5)
            start = start + shift
            D[j].append(istart)

    # Compute the actual overlaps:
    O = []
    for j in range(nf):
        indices = np.array(D[j])
        if len(indices) > 1:
            overlaps = indices[1:] - indices[:-1]
            O.append(np.mean((L[j] - overlaps) / L[j]))
        else:
            O.append(0.0)

    # Convert lists to numpy arrays:
    f = np.array(f)
    r = np.array(r)
    b = np.array(b)
    L = np.array(L)
    K = np.array(K)
    O = np.array(O)
    navg = np.array(navg)

    # Constraint verification (note that some constraints are "soft"):
    if not np.isclose(f[-1], fmax, rtol=0.05): logging.warning(f"ltf::ltf_plan: f[-1]={f[-1]} and fmax={fmax}")
    if not np.allclose(f, r * b): logging.warning(f"ltf::ltf_plan: f[j] != r[j]*b[j]")
    if not np.allclose(r * L, np.full(len(r), fs)): logging.warning(f"ltf::ltf_plan: r[j]*L[j] != fs")
    if not np.allclose(r[:-1], np.diff(f), rtol=0.05): logging.warning(f"ltf::ltf_plan: r[j] != f[j+1] - f[j]")
    if not np.all(L < N+1): logging.warning(f"ltf::ltf_plan: L[j] >= N+1")
    if not np.all(L >= Lmin): logging.warning(f"ltf::ltf_plan: L[j] < Lmin")
    if not np.all(b >= bmin * (1 - 0.05)): logging.warning(f"ltf::ltf_plan: b[j] < bmin")
    if not np.all(L[K == 1] == N): logging.warning(f"ltf::ltf_plan: L[K==1] != N")

    # Final number of frequencies:
    nf = len(f)
    if nf == 0:
        logging.error("Error: frequency scheduler returned zero frequencies")
        sys.exit(-1)

    output = {"f": f, "r": r, "b": b, "L": L, "K": K, "navg": navg, "D": D, "O": O, "nf": nf}

    return output

def new_ltf_plan(params):
    """
    New LTF scheduler.

    Work in progress.
    """
    def round_half_up(val):
        if (float(val) % 1) >= 0.5:
            x = math.ceil(val)
        else:
            x = round(val)
        return x
    
    N = params['N']
    fs = params['fs']
    olap = params['olap']
    bmin = params['bmin']
    Lmin = params['Lmin']
    Jdes = params['Jdes']
    Kdes = params['Kdes']

    # Init constants:
    xov = (1 - olap)
    fmin = fs / N * bmin
    fmax = fs / 2
    fresmin = fs / N
    freslim = fresmin * (1 + xov * (Kdes - 1))
    logfact = (N / 2)**(1 / Jdes) - 1

    # Init lists:
    f = []
    r = []
    b = []
    L = []
    K = []
    O = []
    D = []
    navg = []

    # Scheduler algorithm:
    dftlen_crossover = int(0)
    f_crossover = 0.0
    third_stage = False
    fourth_stage = False

    j = 0
    fi = fmin
    while fi < fmax:
        fres = fi * logfact
        if fres >= freslim:
            third_stage = True
            break
            pass
        elif fres < freslim and (freslim * fres)**0.5 > fresmin:
            fres = (freslim * fres)**0.5
        else:
            fres = fresmin

        fbin = fi / fres
        if fbin < bmin:
            fbin = bmin
            fres = fi / fbin

        dftlen = int(round_half_up(fs / fres))

        dftlen_crossover = dftlen
        f_crossover = fi

        if dftlen > N:
            dftlen = N
        if dftlen < Lmin:
            dftlen = Lmin

        nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))
        if nseg == 1:
            dftlen = N

        fres = fs / dftlen
        fbin = fi / fres

        f.append(fi)
        r.append(fres)
        b.append(fbin)
        L.append(dftlen)
        K.append(nseg)

        fi = fi + fres
        j = j + 1
    
    if third_stage:
        alpha = 1.0*np.log(Lmin / dftlen_crossover) / (Jdes - j - 1)
        k = 0
        while fi < fmax:
            dftlen = int(dftlen_crossover * np.exp(alpha * (k + 0)))
            nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))

            if dftlen < Lmin:
                fourth_stage = True
                break

            if k == 0:
                fres = fs / dftlen 
                fi = fi + fres
                fi = (fi + f_crossover)/2
                if fi > fmax: break
                fres = fi - f_crossover
                fbin = fi / fres
            else:
                fres = fs / dftlen
                fi = fi + fres
                if fi > fmax: break
                fbin = fi / fres

            f.append(fi)
            r.append(fres)
            b.append(fbin)
            L.append(dftlen)
            K.append(nseg)

            k = k + 1

    if fourth_stage:
        while fi < fmax:
            dftlen = Lmin
            nseg = int(round_half_up((N - dftlen) / (xov * dftlen) + 1))
            fres = fs / dftlen
            fi = fi + fres
            if fi > fmax: break
            fbin = fi / fres
            f.append(fi)    
            r.append(fres)
            b.append(fbin)
            L.append(dftlen)
            K.append(nseg)

    nf = len(f)

    # Compute actual averages and starting indices:
    for j in range(nf):
        L_j = int(L[j])
        L[j] = L_j
        averages = int(round_half_up(((N - L_j) / (1 - olap)) / L_j + 1))
        navg.append(averages)

        if averages == 1:
            shift = 1.0
        else:
            shift = (float)(N - L_j) / (float)(averages - 1)
        if shift < 1:
            shift = 1.0

        start = 0.0
        D.append([])
        for _ in range(averages):
            istart = int(float(start) + 0.5) if start >= 0 else int(float(start) - 0.5)
            start = start + shift
            D[j].append(istart)

    # Compute the actual overlaps:
    O = []
    for j in range(nf):
        indices = np.array(D[j])
        if len(indices) > 1:
            overlaps = indices[1:] - indices[:-1]
            O.append(np.mean((L[j] - overlaps) / L[j]))
        else:
            O.append(0.0)

    # Convert lists to numpy arrays:
    f = np.array(f)
    r = np.array(r)
    b = np.array(b)
    L = np.array(L)
    K = np.array(K)
    O = np.array(O)
    navg = np.array(navg)

    # Constraint verification (note that some constraints are "soft"):
    if not np.isclose(f[-1], fmax, rtol=0.05): logging.warning(f"ltf::ltf_plan: f[-1]={f[-1]} and fmax={fmax}")
    if not np.allclose(f, r * b): logging.warning(f"ltf::ltf_plan: f[j] != r[j]*b[j]")
    if not np.allclose(r * L, np.full(len(r), fs), rtol=0.05): logging.warning(f"ltf::ltf_plan: r[j]*L[j] != fs")
    if not np.allclose(r[:-1], np.diff(f), rtol=0.05): logging.warning(f"ltf::ltf_plan: r[j] != f[j+1] - f[j]")
    if not np.all(L < N+1): logging.warning(f"ltf::ltf_plan: L[j] >= N+1")
    if not np.all(L >= Lmin): logging.warning(f"ltf::ltf_plan: L[j] < Lmin")
    if not np.all(b >= bmin * (1 - 0.05)): logging.warning(f"ltf::ltf_plan: b[j] < bmin")
    if not np.all(L[K == 1] == N): logging.warning(f"ltf::ltf_plan: L[K==1] != N")

    # Final number of frequencies:
    nf = len(f)
    if nf == 0:
        logging.error("Error: frequency scheduler returned zero frequencies")
        sys.exit(-1)

    output = {"f": f, "r": r, "b": b, "L": L, "K": K, "navg": navg, "D": D, "O": O, "nf": nf}

    return output
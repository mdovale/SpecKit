""" This module contains the LTFObject class with methods for spectral analysis
of digitized time series.

This class implements a very fast pure-Python implementation of the LPSD/LTF algorithm.

Reference: LPSD algorithm by Gerhard Heinzel and Michael Troebs.
https://doi.org/10.1016/j.measurement.2005.10.010

Miguel Dovale (Hannover, 2024)
"""
import sys
import copy
import types
import numpy as np
import pandas as pd
import math
import time
from spectools.flattop import olap_dict, win_dict
from spectools.dsp import integral_rms, numpy_detrend
from spectools.aux import chunker
import matplotlib.pyplot as plt

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)

class LTFObject:
    def __init__(self, data=None, fs=None, verbose=False):
        self.fs = fs # Sampling frequency of input data
        self.olap = "default" # Segment overlap
        self.bmin = 1 # Number of the first frequency bin to consider
        self.Lmin = 0 # The smallest allowable segment length to be processed
        self.Jdes = 500 # Desired number of frequencies in spectrum
        self.Kdes = 100 # Desired number of averages
        self.order = 0 # Detrending order (0: remove mean, 1: linear regression, 2: quadratic regression)
        self.win = np.kaiser # Window function to use
        self.psll = 200 # Peak side-lobe level
        self.alpha = None # 
        self.iscsd = None # True if it is a cross-spectrum
        self.f = None # Fourier frequency vector
        self.r = None # Frequency resolution vector
        self.m = None # Frequency bin vector
        self.L = None # Segment lengths vector
        self.K = None # Number of segments vector (should be the same as navg)
        self.navg    = None  # Number of averages vector
        self.compute_t = None # Computation time per frequency
        self.data    = None  # Input data as columns
        self.df      = None  # DataFrame containing the results
        self.nx      = None  # Total length of the time series
        self.nf      = None  # Number of frequencies in spectrum
        self.ENBW    = None  # Equivalent noise bandwidth
        self.rms     = None  # Root mean square of the signal computed from the integral of the ASD

        # Spectral estimates
        self.XX      = None  # Spectrum
        self.YY      = None  # Spectrum
        self.G       = None  # Power spectrum (PS)
        self.ps      = None  # Power spectrum (PS)
        self.Gxx     = None  # Power spectral density (PSD)
        self.Gyy     = None  # Power spectral density (PSD)
        self.psd     = None  # Power spectral density (PSD)
        self.asd     = None  # Amplitude spectral density (ASD)
        self.XY      = None  # Cross spectrum
        self.cs      = None  # Cross power spectrum
        self.Gxy     = None  # Cross spectral density (CSD)
        self.csd     = None  # Cross spectral density (CSD)
        self.cpsd    = None  # Cross power spectral density (CPSD)
        self.Hxy     = None  # Transfer function (TF)
        self.cf      = None  # Coupling coefficient (magnitude of the transfer function)
        self.coh     = None  # Coherence
        self.ccoh    = None  # Complex coherence

        # For uncertainty estimations, see Bendat, Piersol - ISBN10:0471058874, Chapter 11
        # Standard deviation of estimates:
        self.Gxx_dev = None  # Standard deviation of PSD
        self.Gxy_dev = None  # Standard deviation of CSD
        self.Hxy_dev = None  # Standard deviation of Hxy
        self.coh_dev = None  # Standar deviation of coherence
        self.ccoh_dev = None # Standar deviation of complex coherence
 
        # Normalized random errors:
        self.Gxx_error = None  # Normalized random error of PSD
        self.Gxy_error = None  # Normalized random error of CSD
        self.Hxy_magnitude_error = None  # Normalized random error of |Hxy|
        self.Hxy_angle_error = None  # Normalized random error of arg(Hxy)
        self.coh_error = None  # Normalized random error of coherence

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

    def _myround(self, val):
        if (float(val) % 1) >= 0.5:
            x = math.ceil(val)
        else:
            x = round(val)
        return x

    def _kaiser_alpha(self, psll):
        a0 = -0.0821377
        a1 = 4.71469
        a2 = -0.493285
        a3 = 0.0889732

        x = psll / 100
        return (((((a3 * x) + a2) * x) + a1) * x + a0)

    def _kaiser_rov(self, alpha):
        a0 = 0.0061076
        a1 = 0.00912223
        a2 = -0.000925946
        a3 = 4.42204e-05
        x = alpha
        return (100 - 1 / (((((a3 * x) + a2) * x) + a1) * x + a0)) / 100

    def load_params(self, verbose, default=False, fs=None, olap=None, bmin=None, Lmin=None, Jdes=None, Kdes=None, order=None, win=None, psll=None):
        """
        Function to load or update parameters.
        """
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

        if (win=="Kaiser")or(win=="kaiser")or(win==np.kaiser)or(win==None):
            self.win = np.kaiser
        elif (win == "Hanning")or(win=="hanning")or(win==np.hanning):
            self.win = np.hanning
        elif isinstance(win, str):
            win_str = win
            if win_str in win_dict:
                self.win = win_dict[win_str]
        elif isinstance(win, types.FunctionType):
            self.win = win
            win_str = "User"
        else:
            logging.error("This window function is not implemented in Python")
        
        if verbose: logging.info(f"Selected window: {self.win}")
        
        if self.olap == "default":
            if self.win == np.kaiser:
                self.alpha = self._kaiser_alpha(self.psll)
                self.olap = self._kaiser_rov(self.alpha)
            elif self.win == np.hanning:
                self.olap = 0.5
            elif win_str in olap_dict:
                self.olap = olap_dict[win_str]
                if verbose: logging.info(f"Automatic setting of overlap for window {win_str}: {self.olap}")
            else:
                logging.warning(f"Automatic setting of overlap for window {win_str} failed, setting to 0.5")
                self.olap = 0.5

        if (self.win==-1)or(self.win==-2):
            assert self.psll > 0, logging.error("Need to specify PSLL if window is -1 or -2")
        
        if (self.win==-1):
            assert isinstance(self.olap, float), logging.error("Need to specify an overlap if window is -1")

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

    def load_defaults(self):
        """
        Loads default values of the parameters.
        """
        self.fs = 1
        self.olap = "default"
        self.bmin = 1
        self.Lmin = 0
        self.Jdes = 500
        self.Kdes = 100
        self.order = 0
        self.psll = 200
        self.win = np.kaiser

    def ltf_plan(self, band):
        """
        LTF frequency plan from S2-AEI-TN-3052 (Gerhard Heinzel).

        Computes:
            self.f
            self.r
            self.m
            self.L
            self.K
            self.nf
        """

        xov = (1 - self.olap)
        fmin = self.fs / self.nx * self.bmin
        fmax = self.fs / 2
        fresmin = self.fs / self.nx
        freslim = fresmin * (1 + xov * (self.Kdes - 1))
        logfact = (self.nx / 2)**(1 / self.Jdes) - 1

        self.f = []
        self.r = []
        self.m = []
        self.L = []
        self.K = []

        fi = fmin
        while fi < fmax:
            fres = fi * logfact
            if fres <= freslim:
                fres = np.sqrt(fres * freslim)
            if fres < fresmin:
                fres = fresmin

            fbin = fi / fres
            if fbin < self.bmin:
                fbin = self.bmin
                fres = fi / fbin

            dftlen = self._myround(self.fs / fres)
            if dftlen > self.nx:
                dftlen = self.nx
            if dftlen < self.Lmin:
                dftlen = self.Lmin

            nseg = self._myround((self.nx - dftlen) / (xov * dftlen) + 1)
            if nseg == 1:
                dftlen = self.nx

            fres = self.fs / dftlen
            fbin = fi / fres

            self.f.append(fi)
            self.r.append(fres)
            self.m.append(fbin)
            self.L.append(dftlen)
            self.K.append(nseg)

            fi = fi + fres

        self.f = np.array(self.f)
        self.r = np.array(self.r)
        self.m = np.array(self.m)
        self.L = np.array(self.L)
        self.K = np.array(self.K)

        if band is not None:
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

        self.nf = len(self.f)

        if self.nf == 0:
            logging.error("Error: frequency scheduler returned zero frequencies")
            sys.exit(-1)

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

        segLen = l  # Segment length
        ovfact = 1 / (1 - self.olap)

        davg = (((self.nx - segLen)) * ovfact) / segLen + 1
        self.navg = self._myround(davg)

        if self.navg == 1:
            shift = 1.0
        else:
            shift = (float)(self.nx - segLen) / (float)(self.navg - 1)
        if shift < 1:
            shift = 1.0

        start = 0.0
        for j in range(self.navg):
            istart = int(self._myround(start))
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
        self.Hxy_magnitude_error = 1.0
        self.Hxy_angle_error = 0.0
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
                self.Hxy_magnitude_error = math.sqrt(1 - self.coh) / ( math.sqrt(self.coh) * math.sqrt(2*self.navg))
            except ValueError:
                pass
            try:
                # Hxy_angle_error = math.asin(math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*self.navg)))
                # Hxy_angle_error = math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*self.navg))
                self.Hxy_angle_error = self.Hxy_dev/abs(self.Hxy)
            except ValueError:
                pass
            try:
                self.coh_error = math.sqrt(2) * (1 - self.coh) / (math.sqrt(self.coh) * math.sqrt(self.navg))
            except ValueError:
                pass

        if self.iscsd:
            self.csd = self.Gxy # Cross spectral density
            self.cpsd = np.sqrt(self.Gxy) # Cross power spectral density
            self.cf = np.abs(self.Hxy) # Coupling coefficient
            self.cs = self.csd * self.ENBW # Cross spectrum
        else:
            self.Gxx = self.Gxy.real # Power spectral density
            self.Gxx_dev = self.Gxy_dev # Standar deviation of Gxx
            self.psd = self.Gxx # Power spectral density
            self.asd = np.sqrt(self.Gxx) # Amplitude spectral density
            self.G = self.Gxx * self.ENBW # Power spectrum
            self.ps = self.G # Power spectrum

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
                                                          'Gxy_dev', 'Gxy_error', 'Hxy_dev', 'Hxy_magnitude_error', 'Hxy_angle_error', 
                                                          'coh_dev', 'ccoh_dev', 'coh_error',
                                                          'ENBW', 'navg', 'compute_t']) \
                for chunk in r], axis=0, ignore_index=True)

        self.df = self.df.sort_values('i')

        assert np.array_equal(np.array(range(self.nf)), np.array(self.df.i))

        self.XX = np.array(self.df.XX) # Spectrum
        self.YY = np.array(self.df.YY) # Spectrum
        self.XY = np.array(self.df.XY) # Cross spectrum
        self.Gxy = np.array(self.df.Gxy) # Cross spectral density
        self.Gxx = np.array(self.df.Gxx) # Power spectral density
        self.Gyy = np.array(self.df.Gyy) # Power spectral density
        self.Hxy = np.array(self.df.Hxy) # Transfer function estimate
        self.coh = np.array(self.df.coh) # Coherence
        self.ccoh = np.array(self.df.ccoh) # Complex coherence
        self.Gxy_dev = np.array(self.df.Gxy_dev)
        self.Gxy_error = np.array(self.df.Gxy_error)
        self.Hxy_dev = np.array(self.df.Hxy_dev)
        self.Hxy_magnitude_error = np.array(self.df.Hxy_magnitude_error)
        self.Hxy_angle_error = np.array(self.df.Hxy_angle_error)
        self.coh_dev = np.array(self.df.coh_dev)
        self.coh_error = np.array(self.df.coh_error)
        self.ccoh_dev = np.array(self.df.ccoh_dev)
        self.ENBW = np.array(self.df.ENBW) # Equivalent noise bandwidth
        self.navg = np.array(self.df.navg) # Number of averages
        self.compute_t = np.array(self.df.compute_t) # Compute time

        if self.iscsd:
            self.csd = self.Gxy.copy() # Cross spectral density
            self.cpsd = np.sqrt(self.Gxy) # Cross power spectral density
            self.cf = np.abs(self.Hxy) # Coupling coefficient
            self.cs = self.csd * self.ENBW # Cross spectrum
        
        if np.all(np.imag(self.Gxy) == 0):
            self.Gxx = self.Gxy.real # Power spectral density
            self.Gxx_dev = self.Gxy_dev.copy()
            self.Gxx_error = self.Gxy_error.copy()
            self.psd = self.Gxx.copy() # Power spectral density
            self.asd = np.sqrt(self.Gxx) # Amplitude spectral density
            self.G = self.Gxx * self.ENBW # Power spectrum
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

            l = int(self.L[i]) # segment length

            if self.win == np.kaiser:
                window = self.win(l + 1, self.alpha*np.pi)[0:-1]
            else:
                window = self.win(l)        

            p = 1j * 2 * np.pi * self.m[i] / l * np.arange(0, l)
            C = window * np.exp(p)

            MXYr = 0.0
            MXYi = 0.0
            MXY2 = 0.0
            MXX = 0.0
            MYY = 0.0
            MXX2 = 0.0
            MYY2 = 0.0

            segLen = l  # Segment length
            ovfact = 1 / (1 - self.olap)

            davg = (((self.nx - segLen)) * ovfact) / segLen + 1
            navg = self._myround(davg)

            if navg == 1:
                shift = 1.0
            else:
                shift = (float)(self.nx - segLen) / (float)(navg - 1)
            if shift < 1:
                shift = 1.0

            start = 0.0
            for j in range(navg):
                istart = int(self._myround(start))
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
            if (navg == 1):
                Vr = MXYr * MXYr
            else:
                Vr = MXY2 / (navg - 1)
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
            Hxy_magnitude_error = 1.0
            Hxy_angle_error = 0.0
            coh_error = 1.0

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
                    Hxy_magnitude_error = math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*navg))
                except ValueError:
                    pass
                try:
                    # Hxy_angle_error = math.asin(math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*navg)))
                    # Hxy_angle_error = math.sqrt(1 - coh) / ( math.sqrt(coh) * math.sqrt(2*navg))
                    Hxy_angle_error = Hxy_dev/abs(Hxy)
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
                            Hxy_dev, Hxy_magnitude_error, Hxy_angle_error,
                            coh_dev, ccoh_dev, coh_error, 
                            ENBW, navg, compute_t])

        after = time.time()
        block_time = after - before
        
        if verbose: logging.info('Completed {} in {} seconds'.format(f_block, block_time))

        return r_block

    def plot(self, which=None, ylabel=None, figsize=(3, 2), dpi=300, fontsize=8, *args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Dictionary to map 'which' to appropriate data and y-label
        plot_options = {
            'ps': (self.f, self.G, r"PS") if not self.iscsd else None,
            'asd': (self.f, self.asd, r"ASD") if not self.iscsd else None,
            'psd': (self.f, self.psd, r"PSD") if not self.iscsd else None,
            'cs': (self.f, self.cs, r"CS") if self.iscsd else None,
            'csd': (self.f, self.csd, r"CSD") if self.iscsd else None,
            'cpsd': (self.f, self.cpsd, r"CPSD") if self.iscsd else None,
            'coh': (self.f, self.coh, r"Coherence") if self.iscsd else None,
            'cf': (self.f, self.cf, r"Coupling coefficient") if self.iscsd else None,
        }
        
        if which is None:
            if self.iscsd:
                which = 'csd'
            else:
                which = 'asd'

        # Retrieve plot data and y-label
        plot_data = plot_options.get(which)
        if plot_data is None:
            return  # Return if plot data is None (invalid 'which' or 'iscsd' is False)

        x_data, y_data, y_label = plot_data
        if which == 'coh':
            ax.semilogx(x_data, y_data, *args, **kwargs);
        else:
            ax.loglog(x_data, y_data, *args, **kwargs);
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=fontsize);
        else:
            ax.set_ylabel(y_label, fontsize=fontsize);

        ax.set_xlim([self.f[0], self.f[-1]]);
        ax.set_xlabel("Fourier frequency (Hz)", fontsize=fontsize);
        ax.tick_params(which='both', labelsize=fontsize);
        ax.grid();
        fig.tight_layout();

        plt.close(fig)
        return fig
    
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
            'Hxy_angle': np.angle(self.Hxy, deg=True) if self.iscsd else None,
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

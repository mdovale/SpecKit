""" This module contains spectral analysis functions.

Miguel Dovale (Hannover, 2024)
E-mail: spectools@pm.me
"""
import numpy as np
from spectools.lpsd import ltf

def SISO_optimal_spectral_analysis(y1, y2, fs, pool=None):

    y1y2csd = ltf([y1, y2], fs=fs, pool=pool)

    S11 = y1y2csd.Gxx
    S22 = y1y2csd.Gyy
    S12 = y1y2csd.Gxy
    S21 = np.conj(y1y2csd.Gxy)
    H = S12 / S11
    optimal_asd = np.abs(np.sqrt(S22 + (H)*np.conj(H)*S11 - np.conj(H)*S12 - H*S21))

    return y1y2csd.f, optimal_asd
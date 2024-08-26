"""This module implements several flat-top window functions.

It is based on a Matlab script from Joe Henning from 6 May 2013.

Implemented flat-top window functions:
SFT3F, 31.7 dB PSLL, differentiable, 0.0082 dB emax 
SFT4F, 44.7 dB PSLL, 2nd differentiable, 0.0041 dB emax 
SFT5F, 57.3 dB PSLL, 3rd differentiable, -0.0025 dB emax 
SFT3M, 44.2 dB PSLL, -0.0115 dB emax 
SFT4M, 66.5 dB PSLL, -0.0067 dB emax 
SFT5M, 89.9 dB PSLL, 0.0039 dB emax 
FTNI (National Instruments), 44.4 dB PSLL, 0.0169 dB emax 
FTHP (old Hewlett Packard), 70.4 dB PSLL, 0.0096 dB emax 
FTSRS (Stanford Research SR785), 76.6 dB PSLL, differentiable, -0.0156 dB emax 
Matlab, 93.0 dB PSLL, 0.0097 dB emax 
HFT70 (3-term cosine), 70.4 dB PSLL, -0.0065 dB emax 
HFT95 (4-term cosine), 95.0 dB PSLL, 0.0044 dB emax 
HFT90D (4-term cosine), 90.2 dB PSLL, differentiable, -0.0039 dB emax 
HFT116D (5-term cosine), 116.8 dB PSLL, differentiable, -0.0028 dB emax 
HFT144D (6-term cosine), 144.1 dB PSLL, differentiable, 0.0021 dB emax 
HFT169D (7-term cosine), 169.5 dB PSLL, differentiable, 0.0017 dB emax 
HFT196D (8-term cosine), 196.2 dB PSLL, differentiable, 0.0013 dB emax 
HFT223D (9-term cosine), 223.0 dB PSLL, differentiable, -0.0011 dB emax 
HFT248D (10-term cosine), 248.4 dB PSLL, differentiable, 0.0009 dB emax 

In the above, "F" indicates "fast decaying" which enforces a high
degree of differentiability, "M" indicates minimum sidelobe, and
"HFT" indicates the window with lowest sidelobe level.

It also includes a dictionary in which the optimal overlap values are given.

Ref:
Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new flat-top windows
G. Heinzel, A. Rudiger, and R. Schilling
Maz-Planck-Institut fur Gravitationsphysik
(Albert-Einstein-Institut)
Teilinstitut Hannover
February 15, 2002
"""
import numpy as np


def SFT3F(M):
    """Flat-top window function SFT3F. 
    
    Differentiable, -31.7 dB, NBW 3.1681 bins, 0.0082 dB, first zero at +/- 3 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.26526
    a1 = 0.5
    a2 = 0.23474
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z)
    return w


def SFT4F(M):
    """Flat-top window function SFT4F. 
    
    2nd differentiable, -44.7 dB, NBW 3.7970 bins, 0.0041 dB, first zero at +/- 4 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.21706
    a1 = 0.42103
    a2 = 0.28294
    a3 = 0.07897
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(6 * z)
    return w


def SFT5F(M):
    """Flat-top window function SFT5F. 
    
    3rd differentiable, -57.3 dB, NBW 4.3412 bins, -0.0025 dB, first zero at +/- 5 bins
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.1881
    a1 = 0.36923
    a2 = 0.28702
    a3 = 0.13077
    a4 = 0.02488
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z)
    return w


def SFT3M(M):
    """Flat-top window function SFT3M. 
    
    -44.2 dB, NBW 2.9452 bins, -0.0115 dB, first zero at +/- 3 bins
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.28235
    a1 = 0.52105
    a2 = 0.19659
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z)
    return w


def SFT4M(M):
    """Flat-top window function SFT4M. 
    
    -66.5 dB, NBW 3.3868 bins, -0.0067 dB, first zero at +/- 4 bins
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.241906
    a1 = 0.460841
    a2 = 0.255381
    a3 = 0.041872
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(6 * z)
    return w


def SFT5M(M):
    """Flat-top window function SFT5F. 
    
    -89.9 dB, NBW 3.8852 bins, 0.0039 dB, first zero at +/- 5 bins
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.209671
    a1 = 0.407331
    a2 = 0.281225
    a3 = 0.092669
    a4 = 0.0091036
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z)
    return w


def FTNI(M):
    """Flat-top window function FTNI (National Instruments). 
    
    -44.4 dB, NBW 2.9656 bins, 0.0169 dB, first zero at +/- 3 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.2810639
    a1 = 0.5208972
    a2 = 0.1980399
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z)
    return w


def FTHP(M):
    """Flat-top window function # FTHP (old Hewlett Packard). 
    
    -70.4 dB, NBW 3.4279 bins, 0.0096 dB, first zero at +/- 4 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.912510941
    a2 = 1.079173272
    a3 = 0.1832630879
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(6 * z)
    return w


def FTSR(M):
    """Flat-top window function FTSRS (Stanford Research SR785). 
    
    Differentiable, -76.6 dB, NBW 3.7702 bins, -0.0156 dB, first zero at +/- 4.72 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.93
    a2 = 1.29
    a3 = 0.388
    a4 = 0.028
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z)
    return w


def Matlab(M):
    """Flat-top window function Matlab. 
    
    -93.0 dB, NBW 3.774 bins, 0.0025 dB, first zero at +/- 5.01 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 0.21557895
    a1 = 0.41663158
    a2 = 0.277263158
    a3 = 0.083578947
    a4 = 0.006947368
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z)
    return w


def HFT70(M):
    """Flat-top window function HFT70 (lowest sidelobe level with 3 np.cosine terms). 
    
    -70.4 dB, NBW 3.4129 bins, -0.0065 dB, first zero at +/- 4 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """

    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.90796
    a2 = 1.07349
    a3 = 0.18199
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(6 * z)
    return w


def HFT95(M):
    """Flat-top window function HFT95 (lowest sidelobe level with 4 np.cosine terms),  
    
    -95.0 dB, NBW 3.8112 bins, 0.0044 dB, first zero at +/- 5 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.9383379
    a2 = 1.3045202
    a3 = 0.4028270
    a4 = 0.0350665
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z)
    return w


def HFT90D(M):
    """Flat-top window function HFT90D (lowest sidelobe level with 4 np.cosine terms). 
    
    Differentiable, -90.2, NBW 3.8832 bins, -0.0039 dB, first zero at +/- 5 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.942604
    a2 = 1.340318
    a3 = 0.440811
    a4 = 0.043097
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z)
    return w


def HFT116D(M):
    """Flat-top window function HFT116D (lowest sidelobe level with 5 np.cosine terms).
    
    Differentiable, -116.8 dB, NBW 4.2186 bins, -0.0028 dB, first zero at +/- 6 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.9575375
    a2 = 1.4780705
    a3 = 0.6367431
    a4 = 0.1228389
    a5 = 0.0066288
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z) - a5 * np.cos(10 * z)
    return w


def HFT144D(M):
    """Flat-top window function HFT144D (lowest sidelobe level with 6 np.cosine terms).
    
    Differentiable, -144.1 dB, NBW 4.5386 bins, 0.0021 dB, first zero at +/- 7 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.96760033
    a2 = 1.57983607
    a3 = 0.81123644
    a4 = 0.22583558
    a5 = 0.02773848
    a6 = 0.00090360
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z) - a5 * np.cos(10 * z) + a6 * np.cos(12 * z)
    return w


def HFT169D(M):
    """Flat-top window function HFT169D (lowest sidelobe level with 7 np.cosine terms).
    
    Differentiable, -169.5 dB, NBW 4.8347 bins, 0.0017 dB, first zero at +/- 8 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.97441842
    a2 = 1.65409888
    a3 = 0.95788186
    a4 = 0.33673420
    a5 = 0.06364621
    a6 = 0.00521942
    a7 = 0.00010599
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z) - a5 * np.cos(10 * z) + a6 * np.cos(
            12 * z) - a7 * np.cos(14 * z)
    return w


def HFT196D(M):
    """Flat-top window function HFT196D (lowest sidelobe level with 8 np.cosine terms).
    
    Differentiable, -196.2 dB, NBW 5.1134 bins, 0.0013 dB, first zero at +/- 9 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.979280420
    a2 = 1.710288951
    a3 = 1.081629853
    a4 = 0.448734314
    a5 = 0.112376628
    a6 = 0.015122992
    a7 = 0.000871252
    a8 = 0.000011896
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z) - a5 * np.cos(10 * z) + a6 * np.cos(
            12 * z) - a7 * np.cos(14 * z) + a8 * np.cos(16 * z)
    return w


def HFT223D(M):
    """Flat-top window function HFT223D (lowest sidelobe level with 9 np.cosine terms).
    
    Differentiable, -223.0 dB, NBW 5.3888 bins, -0.0011 dB, first zero at +/- 10 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.98298997309
    a2 = 1.75556083063
    a3 = 1.19037717712
    a4 = 0.56155440797
    a5 = 0.17296769663
    a6 = 0.03233247087
    a7 = 0.00324954578
    a8 = 0.00013801040
    a9 = 0.00000132725
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z) - a5 * np.cos(10 * z) + a6 * np.cos(
            12 * z) - a7 * np.cos(14 * z) + a8 * np.cos(16 * z) - a9 * np.cos(
                18 * z)
    return w


def HFT248D(M):
    """Flat-top window function HFT248D (lowest sidelobe level with 10 np.cosine terms).
    
    Differentiable, -248.4 dB, NBW 5.6512 bins, 0.0009 dB, first zero at +/- 11 bins.
    
    Args:
        M (int): Points in the output window.
    
    Returns:
        np.array(): Window values as vector.
    """
    z = np.pi / M * np.arange(0, M)
    a0 = 1.0
    a1 = 1.985844164102
    a2 = 1.791176438506
    a3 = 1.282075284005
    a4 = 0.667777530266
    a5 = 0.240160796576
    a6 = 0.056656381764
    a7 = 0.008134974479
    a8 = 0.000624544650
    a9 = 0.000019808998
    a10 = 0.000000132974
    w = a0 - a1 * np.cos(2 * z) + a2 * np.cos(4 * z) - a3 * np.cos(
        6 * z) + a4 * np.cos(8 * z) - a5 * np.cos(10 * z) + a6 * np.cos(
            12 * z) - a7 * np.cos(14 * z) + a8 * np.cos(16 * z) - a9 * np.cos(
                18 * z) + a10 * np.cos(20 * z)
    return w


c_windows = {
    "Kaiser": -2,
    "Flat-top": -1,
    "Rectangular": 0,
    "Welch": 1,
    "Barlett": 2,
    "Hanning": 3,
    "Hamming": 4,
    "Nuttall3": 5,
    "Nuttall4": 6,
    "Nuttall3a": 7,
    "Nuttall3b": 8,
    "Nuttall4a": 9,
    "BH92": 10,
    "Nuttall4b": 11,
    "Nuttall4c": 12,
    "SFT3F": 13,
    "SFT3M": 14,
    "FTNI": 15,
    "SFT4F": 16,
    "SFT5F": 17,
    "SFT4M": 18,
    "FTHP": 19,
    "HFT70": 20,
    "FTSRS": 21,
    "SFT5M": 22,
    "HFT90D": 23,
    "HFT95": 24,
    "HFT116D": 25,
    "HFT144D": 26,
    "HFT169D": 27,
    "HFT196D": 28,
    "HFT223D": 29,
    "HFT248D": 30,
    -2: "Flat-top",
    -1: "Kaiser",
    0: "Rectangular",
    1: "Welch",
    2: "Barlett",
    3: "Hanning",
    4: "Hamming",
    5: "Nuttall3",
    6: "Nuttall4",
    7: "Nuttall3a",
    8: "Nuttall3b",
    9: "Nuttall4a",
    10: "BH92",
    11: "Nuttall4b",
    12: "Nuttall4c",
    13: "SFT3F",
    14: "SFT3M",
    15: "FTNI",
    16: "SFT4F",
    17: "SFT5F",
    18: "SFT4M",
    19: "FTHP",
    20: "HFT70",
    21: "FTSRS",
    22: "SFT5M",
    23: "HFT90D",
    24: "HFT95",
    25: "HFT116D",
    26: "HFT144D",
    27: "HFT169D",
    28: "HFT196D",
    29: "HFT223D",
    30: "HFT248D",
}

olap_dict = {
    "Rectangular": 0.0,
    "Welch": 0.293,
    "Barlett": 0.500,
    "Hanning": 0.500,
    "Hamming": 0.500,
    "Nuttall3": 0.647,
    "Nuttall4": 0.705,
    "Nuttall3a": 0.612,
    "Kaiser3": 0.619,
    "Nuttall3b": 0.598,
    "Nuttall4a": 0.680,
    "BH92": 0.661,
    "Nuttall4b": 0.663,
    "Kaiser4": 0.670,
    "Nuttall4c": 0.656,
    "Kaiser5": 0.705,
    "SFT3F": 0.667,
    "SFT3M": 0.655,
    "FTNI": 0.656,
    "SFT4F": 0.750,
    "SFT5F": 0.785,
    "SFT4M": 0.721,
    "FTHP": 0.723,
    "HFT70": 0.722,
    "FTSRS": 0.754,
    "SFT5M": 0.760,
    "HFT90D": 0.760,
    "HFT95": 0.756,
    "HFT116D": 0.782,
    "HFT144D": 0.799,
    "HFT169D": 0.812,
    "HFT196D": 0.823,
    "HFT223D": 0.833,
    "HFT248D": 0.841,
    0: 0.0,
    1: 0.293,
    2: 0.500,
    3: 0.500,
    4: 0.500,
    5: 0.647,
    6: 0.705,
    7: 0.612,
    8: 0.598,
    9: 0.680,
    10: 0.661,
    11: 0.663,
    12: 0.656,
    13: 0.667,
    14: 0.655,
    15: 0.656,
    16: 0.750,
    17: 0.785,
    18: 0.721,
    19: 0.723,
    20: 0.722,
    21: 0.754,
    22: 0.760,
    23: 0.760,
    24: 0.756,
    25: 0.782,
    26: 0.799,
    27: 0.812,
    28: 0.823,
    29: 0.833,
    30: 0.841
}

win_dict = {
    "SFT3F": SFT3F,
    "SFT3M": SFT3M,
    "FTNI": FTNI,
    "SFT4F": SFT4F,
    "SFT5F": SFT5F,
    "SFT4M": SFT4M,
    "FTHP": FTHP,
    "HFT70": HFT70,
    "FTSRS": FTSR,
    "SFT5M": SFT5M,
    "HFT90D": HFT90D,
    "HFT95": HFT95,
    "HFT116D": HFT116D,
    "HFT144D": HFT144D,
    "HFT169D": HFT169D,
    "HFT196D": HFT196D,
    "HFT223D": HFT223D,
    "HFT248D": HFT248D
}

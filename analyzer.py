""" This module contains functions for spectral analysis in
Single-Input Single-Output (SISO) and Multiple-Input Single-Output (MISO) systems.

Miguel Dovale (Hannover, 2024)
E-mail: spectools@pm.me
"""
import numpy as np
import sympy as sp
from spectools.lpsd import lpsd, ltf, LTFObject

import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S'
)

def SISO_optimal_spectral_analysis(input, output, fs, **kwargs):
    """
    Performs optimal spectral analysis on a Single-Input Single-Output (SISO) system 
    using an exact solution to estimate the amplitude spectral density (ASD) of the output, 
    with the influence of the input subtracted.

    Parameters
    ----------
    input : array-like
        The input time series signal.
    
    output : array-like
        The output time series signal.
    
    fs : float
        The sampling frequency of the input and output time series.
    
    **kwargs : dict, optional
        Additional keyword arguments passed to the `ltf` function for spectral analysis.

    Returns
    -------
    np.ndarray
        Fourier frequencies at which the analysis is performed.
    
    np.ndarray
        The amplitude spectral density of the output signal, calculated using the 
        optimal spectral analysis method.
    """

    logging.info("Computing all spectral estimates and optimal solution...")

    csd = ltf([input, output], fs, **kwargs)
    
    logging.info("Done.")

    return csd.f, np.sqrt(csd.GyySx)

def MISO_analytic_optimal_spectral_analysis(inputs, output, fs, **kwargs):
    """
    Performs optimal spectral analysis on a Multiple-Input Single-Output (MISO) system 
    using an exact analytic solution to the system of linear equations involving the
    optimal transfer functions between the inputs and the output, and estimates the 
    amplitude spectral density (ASD) of the output with the influence of the inputs subtracted.

    Reference
    ---------
    Bendat, Piersol - "Engineering Applications of Correlation and Spectral Analysis"
    Section 8.1: Multiple Input/Output Systems
    ISBN: 978-0-471-57055-4
    https://archive.org/details/engineeringappli0000bend
    
    Parameters
    ----------
    inputs : array-like
        List of multiple input time series signals.
    
    output : array-like
        The output time series signal.
    
    fs : float
        Sampling frequency of the input and output time series.
    
    **kwargs : dict, optional
        Additional keyword arguments passed to the `ltf` function for spectral analysis.

    Returns
    -------
    np.ndarray
        Fourier frequencies at which the analysis is performed.
    
    np.ndarray
        Amplitude spectral density of the output signal, calculated using the 
        optimal spectral analysis method.
    """
    q = len(inputs)
    if q > 5:
        logging.warning(f"The problem dimension ({q}) is too large for the analytic solver, you may want to use MISO_numeric_optimal_spectral_analysis.")

    N = len(inputs[0])
    for input in inputs:
        if len(input) != N:
            raise ValueError("All input time series must be of equal length")
    if len(output) != N:
            raise ValueError("The output time series must have the same length as the inputs")

    logging.info(f"Solving {q}-dimensional symbolic problem...")

    # Automatically generate symbolic elements for the vector of CSDs between inputs and outout, Sj0:
    Svec = sp.Matrix([sp.Symbol(f'S{i}0') for i in range(1, q+1)])

    # Automatically generate symbolic elements for the matrix of input CSDs, Tij:
    Tmat = sp.Matrix(q, q, lambda i, j: sp.symbols(f'T{i+1}{j+1}'))  # This creates Matrix([[T11, T12...], [T21, T22...], ...])

    # Vector of unknown optimal transfer functions:
    Hvec = sp.Matrix(sp.symbols(f'H1:{q+1}'))  # This creates (H1, H2...)

    # Set up the system of equations:
    eqns = [Svec[i] - sum(Tmat[i, j] * Hvec[j] for j in range(q)) for i in range(q)]

    # Solve the system symbolically:
    solution = sp.solve(eqns, Hvec)

    logging.info(f"Solution: {solution}")
    logging.info("Computing all spectral estimates...")
    result = {}
    for i in range(q):
        for j in range(i+1,q):
            obj = ltf([inputs[i], inputs[j]], fs, **kwargs)
            result.setdefault(f'T{i+1}{j+1}', obj.Gxy)
            result.setdefault(f'T{j+1}{i+1}', np.conj(obj.Gxy))
            result.setdefault(f'T{i+1}{i+1}', obj.Gxx)
            result.setdefault(f'T{j+1}{j+1}', obj.Gyy)

    for i in range(q):
        obj = ltf([inputs[i], output], fs, **kwargs) 
        result[f'S{i+1}0'] = obj.Gxy
        result[f'S0{i+1}'] = np.conj(obj.Gxy)
    
    result['S00'] = obj.Gyy
    result['f'] = obj.f

    logging.info("Computing solution...")
    for Hi_symbol, Hi_expr in solution.items():
        try:
            # Convert the symbolic expression to a numerical lambda function
            # Pass keys from `result` as symbols for substitution
            Hi_numeric_func = sp.lambdify(list(result.keys()), Hi_expr, modules="numpy")
            
            # Evaluate the numerical function using the numpy arrays in `result`
            Hi_numeric_value = Hi_numeric_func(*result.values())
            
            # Store the evaluated numerical value in the result dictionary
            result[str(Hi_symbol)] = np.asarray(Hi_numeric_value, dtype=complex)

        except Exception as e:
            logging.error(f"Error during numerical computation for {Hi_symbol}: {e}")
            raise

    Sum1 = np.array([0]*len(result['f']), dtype=complex)
    Sum2 = np.array([0]*len(result['f']), dtype=complex)
    Sum3 = np.array([0]*len(result['f']), dtype=complex)
    for i in range(q):
        Sum1 += result[f'H{i+1}']*result[f'S0{i+1}']
        Sum2 += np.conj(result[f'H{i+1}'])*result[f'S{i+1}0']
        for j in range(q):
            Sum3 += np.conj(result[f'H{j+1}'])*result[f'H{i+1}']*result[f'T{j+1}{i+1}']

    # Compute optimal analysis (Equation 8.16, page 191):
    result['optimal_asd'] = np.abs(np.sqrt(result['S00'] - Sum1 - Sum2 + Sum3))

    logging.info("Done.")

    # return result
    return result['f'], result['optimal_asd']

def MISO_numeric_optimal_spectral_analysis(inputs, output, fs, **kwargs):
    """
    Performs optimal spectral analysis on a Multiple-Input Single-Output (MISO) system 
    using by numerically solving the system of linear equations involving the
    optimal transfer functions between the inputs and the output, and estimates the 
    amplitude spectral density (ASD) of the output with the influence of the inputs subtracted.

    Reference
    ---------
    Bendat, Piersol - "Engineering Applications of Correlation and Spectral Analysis"
    Section 8.1: Multiple Input/Output Systems
    ISBN: 978-0-471-57055-4
    https://archive.org/details/engineeringappli0000bend
    
    Parameters
    ----------
    inputs : array-like
        List of multiple input time series signals.
    
    output : array-like
        The output time series signal.
    
    fs : float
        Sampling frequency of the input and output time series.
    
    **kwargs : tuple, optional
        Additional keyword arguments passed to the `ltf` function for spectral analysis.

    Returns
    -------
    np.ndarray
        Fourier frequencies at which the analysis is performed.
    
    np.ndarray
        Amplitude spectral density of the output signal, calculated using the 
        optimal spectral analysis method.
    """
    q = len(inputs)

    N = len(inputs[0])
    for input in inputs:
        if len(input) != N:
            raise ValueError("All input time series must be of equal length")
    if len(output) != N:
        raise ValueError("The output time series must have the same length as the inputs")

    logging.info(f"Solving {q}-dimensional problem...")

    # Dictionary to cache results of ltf calls:
    result = {}

    def get_ltf_result(key, *args, **kwargs):
        """Helper function to retrieve or compute ltf result."""
        if key not in result:
            obj = ltf(*args, **kwargs)
            result[key] = obj
        return result[key]

    logging.info("Computing the auto-spectrum of the output...")
    obj = get_ltf_result("S00", output, fs, **kwargs)
    S00 = obj.Gxx
    frequencies = obj.f
    nf = obj.nf

    # Prepare data for solving the linear system:
    Tmat = np.zeros((q, q, nf), dtype=complex)  # Coherence matrix of inputs
    Svec = np.zeros((q, nf), dtype=complex)  # Cross-spectral densities of inputs and output

    logging.info("Computing all other spectral estimates...")
    for i in range(q):
        for j in range(i+1, q):
            obj = get_ltf_result(f"T{i+1}{j+1}", [inputs[i], inputs[j]], fs, **kwargs)
            if not np.any(Tmat[i, j, :]):
                Tmat[i, j, :] = obj.Gxy
            if not np.any(Tmat[j, i, :]):
                Tmat[j, i, :] = np.conj(obj.Gxy)
            if not np.any(Tmat[i, i, :]):
                Tmat[i, i, :] = obj.Gxx
            if not np.any(Tmat[j, j, :]):
                Tmat[j, j, :] = obj.Gyy

        obj = get_ltf_result(f"S{i+1}0", [inputs[i], output], fs, **kwargs)
        Svec[i, :] = obj.Gxy

    logging.info("Computing solution...")
    # Solve for the optimal transfer functions numerically:
    Hvec = np.zeros((q, nf), dtype=complex)
    for k in range(nf):
        Hvec[:, k] = np.linalg.solve(Tmat[:, :, k], Svec[:, k])

    # Compute the optimal spectral density
    Sum1 = np.sum(Hvec * Svec.conj(), axis=0)
    Sum2 = np.sum(Hvec.conj() * Svec, axis=0)
    Sum3 = np.zeros(nf, dtype=complex)
    for i in range(q):
        for j in range(q):
            Sum3 += Hvec[j, :].conj() * Hvec[i, :] * Tmat[j, i, :]

    # Compute optimal analysis (Equation 8.16, page 191):
    optimal_asd = np.abs(np.sqrt(S00 - Sum1 - Sum2 + Sum3))

    logging.info("Done.")

    return frequencies, optimal_asd
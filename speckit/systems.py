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
import numpy as np
import sympy as sp
from typing import List, Tuple, Any
from numpy.typing import ArrayLike
from speckit import compute_spectrum as ltf
import logging

logger = logging.getLogger(__name__)


def SISO_optimal_spectral_analysis(
    input: ArrayLike, output: ArrayLike, fs: float, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs optimal spectral analysis on a Single-Input Single-Output (SISO) system
    using an exact solution to estimate the amplitude spectral density (ASD) of the output,
    with the influence of the input subtracted.

    Parameters
    ----------
    input : array-like
        The input time series signal. Must be a 1D array-like object with finite values.

    output : array-like
        The output time series signal. Must be a 1D array-like object with finite values
        and the same length as `input`.

    fs : float
        The sampling frequency of the input and output time series. Must be a positive,
        finite value.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `ltf` function for spectral analysis.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Fourier frequencies at which the analysis is performed.
        - The amplitude spectral density of the output signal, calculated using the
          optimal spectral analysis method.

    Raises
    ------
    ValueError
        If `fs` is not positive and finite, if `input` or `output` are empty, if they
        have different lengths, if they are not 1D arrays, or if they contain non-finite values.
    TypeError
        If `input` or `output` cannot be converted to numpy arrays.
    """
    # Validate fs
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"`fs` must be a positive finite float, got {fs!r}.")

    # Convert to numpy arrays and validate
    input_arr = np.asarray(input)
    output_arr = np.asarray(output)

    # Validate dimensions
    if input_arr.ndim != 1:
        raise ValueError(f"`input` must be a 1D array, got {input_arr.ndim}D array.")
    if output_arr.ndim != 1:
        raise ValueError(f"`output` must be a 1D array, got {output_arr.ndim}D array.")

    # Validate non-empty
    if input_arr.size == 0:
        raise ValueError("`input` must not be empty.")
    if output_arr.size == 0:
        raise ValueError("`output` must not be empty.")

    # Validate lengths match
    if len(input_arr) != len(output_arr):
        raise ValueError(
            f"`input` and `output` must have the same length, "
            f"got {len(input_arr)} and {len(output_arr)}."
        )

    # Validate finite values
    if not np.all(np.isfinite(input_arr)):
        raise ValueError("`input` must contain only finite values.")
    if not np.all(np.isfinite(output_arr)):
        raise ValueError("`output` must contain only finite values.")

    logger.info("Computing all spectral estimates and optimal solution...")

    csd = ltf([input_arr, output_arr], fs, **kwargs)

    logger.info("Done.")

    return csd.f, np.sqrt(csd.GyySx)


def MISO_analytic_optimal_spectral_analysis(
    inputs: List[ArrayLike], output: ArrayLike, fs: float, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
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
    inputs : List[array-like]
        List of multiple input time series signals. Each input must be a 1D array-like
        object with finite values. All inputs must have the same length.

    output : array-like
        The output time series signal. Must be a 1D array-like object with finite values
        and the same length as all inputs.

    fs : float
        Sampling frequency of the input and output time series. Must be a positive,
        finite value.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `ltf` function for spectral analysis.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Fourier frequencies at which the analysis is performed.
        - Amplitude spectral density of the output signal, calculated using the
          optimal spectral analysis method.

    Raises
    ------
    ValueError
        If `fs` is not positive and finite, if `inputs` is empty, if any input or `output`
        are empty, if they have different lengths, if they are not 1D arrays, or if they
        contain non-finite values.
    TypeError
        If `inputs` is not a list/sequence, or if any input or `output` cannot be converted
        to numpy arrays.
    """
    # Validate fs
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"`fs` must be a positive finite float, got {fs!r}.")

    # Validate inputs is a non-empty sequence
    if not isinstance(inputs, (list, tuple)):
        raise TypeError(f"`inputs` must be a list or tuple, got {type(inputs).__name__}.")
    q = len(inputs)
    if q == 0:
        raise ValueError("`inputs` must not be empty.")

    if q > 5:
        logger.warning(
            f"The problem dimension ({q}) is very large for the analytic solver, "
            f"you may want to use MISO_numeric_optimal_spectral_analysis."
        )

    # Convert to numpy arrays and validate
    input_arrays = []
    for i, inp in enumerate(inputs):
        inp_arr = np.asarray(inp)
        if inp_arr.ndim != 1:
            raise ValueError(
                f"`inputs[{i}]` must be a 1D array, got {inp_arr.ndim}D array."
            )
        if inp_arr.size == 0:
            raise ValueError(f"`inputs[{i}]` must not be empty.")
        if not np.all(np.isfinite(inp_arr)):
            raise ValueError(f"`inputs[{i}]` must contain only finite values.")
        input_arrays.append(inp_arr)

    output_arr = np.asarray(output)
    if output_arr.ndim != 1:
        raise ValueError(f"`output` must be a 1D array, got {output_arr.ndim}D array.")
    if output_arr.size == 0:
        raise ValueError("`output` must not be empty.")
    if not np.all(np.isfinite(output_arr)):
        raise ValueError("`output` must contain only finite values.")

    # Validate all have the same length
    N = len(input_arrays[0])
    for i, inp_arr in enumerate(input_arrays):
        if len(inp_arr) != N:
            raise ValueError(
                f"All input time series must be of equal length, "
                f"got length {len(inp_arr)} for `inputs[{i}]` but expected {N}."
            )
    if len(output_arr) != N:
        raise ValueError(
            f"The output time series must have the same length as the inputs, "
            f"got {len(output_arr)} but expected {N}."
        )

    logger.info(f"Solving {q}-dimensional symbolic problem...")

    # Automatically generate symbolic elements for the vector of CSDs between inputs and outout, Sj0:
    Svec = sp.Matrix([sp.Symbol(f"S{i}0") for i in range(1, q + 1)])

    # Automatically generate symbolic elements for the matrix of input CSDs, Tij:
    Tmat = sp.Matrix(
        q, q, lambda i, j: sp.symbols(f"T{i + 1}{j + 1}")
    )  # This creates Matrix([[T11, T12...], [T21, T22...], ...])

    # Vector of unknown optimal transfer functions:
    Hvec = sp.Matrix(sp.symbols(f"H1:{q + 1}"))  # This creates (H1, H2...)

    # Set up the system of equations:
    eqns = [Svec[i] - sum(Tmat[i, j] * Hvec[j] for j in range(q)) for i in range(q)]

    # Solve the system symbolically:
    solution = sp.solve(eqns, Hvec)

    logger.info(f"Solution: {solution}")
    logger.info("Computing all spectral estimates...")
    result = {}
    
    # First, compute all auto-spectra explicitly to establish consistent frequency grid
    # Use the first input to establish the reference frequency grid
    obj_ref = ltf(input_arrays[0], fs, **kwargs)
    result["f"] = obj_ref.f
    result[f"T11"] = obj_ref.Gxx
    f_ref = obj_ref.f  # Reference frequency grid
    
    # Compute remaining auto-spectra and verify they use the same frequency grid
    for i in range(1, q):
        obj = ltf(input_arrays[i], fs, **kwargs)
        if not np.allclose(obj.f, f_ref, rtol=1e-10):
            raise ValueError(
                f"Frequency grid mismatch for input {i+1}. "
                f"This suggests inconsistent scheduler behavior. "
                f"All ltf calls must produce identical frequency grids."
            )
        result[f"T{i + 1}{i + 1}"] = obj.Gxx
    
    # Compute cross-spectra between inputs
    for i in range(q):
        for j in range(i + 1, q):
            obj = ltf([input_arrays[i], input_arrays[j]], fs, **kwargs)
            if not np.allclose(obj.f, f_ref, rtol=1e-10):
                raise ValueError(
                    f"Frequency grid mismatch for cross-spectrum T{i+1}{j+1}. "
                    f"This suggests inconsistent scheduler behavior. "
                    f"All ltf calls must produce identical frequency grids."
                )
            result[f"T{i + 1}{j + 1}"] = obj.Gxy
            result[f"T{j + 1}{i + 1}"] = np.conj(obj.Gxy)

    # Compute cross-spectra between inputs and output
    for i in range(q):
        obj = ltf([input_arrays[i], output_arr], fs, **kwargs)
        if not np.allclose(obj.f, f_ref, rtol=1e-10):
            raise ValueError(
                f"Frequency grid mismatch for cross-spectrum S{i+1}0. "
                f"This suggests inconsistent scheduler behavior. "
                f"All ltf calls must produce identical frequency grids."
            )
        result[f"S{i + 1}0"] = obj.Gxy
        result[f"S0{i + 1}"] = np.conj(obj.Gxy)

    # Compute auto-spectrum of output
    obj_output = ltf(output_arr, fs, **kwargs)
    if not np.allclose(obj_output.f, f_ref, rtol=1e-10):
        raise ValueError(
            f"Frequency grid mismatch for output auto-spectrum S00. "
            f"This suggests inconsistent scheduler behavior. "
            f"All ltf calls must produce identical frequency grids."
        )
    result["S00"] = obj_output.Gxx
    
    # Final verification: ensure all arrays have the same length
    nf_ref = len(result["f"])
    for key, value in result.items():
        if key != "f" and isinstance(value, np.ndarray):
            if len(value) != nf_ref:
                raise ValueError(
                    f"Array {key} has length {len(value)} but expected {nf_ref}. "
                    f"This should not happen if frequency grids match."
                )

    logger.info("Computing solution...")
    for Hi_symbol, Hi_expr in solution.items():
        try:
            # Extract the free symbols from the expression (symbols it actually uses)
            free_symbols = sorted(Hi_expr.free_symbols, key=str)
            
            # Convert the symbolic expression to a numerical lambda function
            # Pass symbols in sorted order for consistency
            Hi_numeric_func = sp.lambdify(free_symbols, Hi_expr, modules="numpy")

            # Extract the corresponding values from result in the same order
            symbol_values = [result[str(sym)] for sym in free_symbols]

            # Evaluate the numerical function using the arrays in the correct order
            Hi_numeric_value = Hi_numeric_func(*symbol_values)

            # Store the evaluated numerical value in the result dictionary
            result[str(Hi_symbol)] = np.asarray(Hi_numeric_value, dtype=complex)

        except Exception as e:
            logger.error(f"Error during numerical computation for {Hi_symbol}: {e}")
            raise

    Sum1 = np.array([0] * len(result["f"]), dtype=complex)
    Sum2 = np.array([0] * len(result["f"]), dtype=complex)
    Sum3 = np.array([0] * len(result["f"]), dtype=complex)
    for i in range(q):
        Sum1 += result[f"H{i + 1}"] * result[f"S0{i + 1}"]
        Sum2 += np.conj(result[f"H{i + 1}"]) * result[f"S{i + 1}0"]
        for j in range(q):
            Sum3 += (
                np.conj(result[f"H{j + 1}"])
                * result[f"H{i + 1}"]
                * result[f"T{j + 1}{i + 1}"]
            )

    # Compute optimal analysis (Equation 8.16, page 191):
    result["optimal_asd"] = np.abs(np.sqrt(result["S00"] - Sum1 - Sum2 + Sum3))

    logger.info("Done.")

    # return result
    return result["f"], result["optimal_asd"]


def MISO_numeric_optimal_spectral_analysis(
    inputs: List[ArrayLike], output: ArrayLike, fs: float, **kwargs: Any
) -> Tuple[np.ndarray, np.ndarray]:
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
    inputs : List[array-like]
        List of multiple input time series signals. Each input must be a 1D array-like
        object with finite values. All inputs must have the same length.

    output : array-like
        The output time series signal. Must be a 1D array-like object with finite values
        and the same length as all inputs.

    fs : float
        Sampling frequency of the input and output time series. Must be a positive,
        finite value.

    **kwargs : dict, optional
        Additional keyword arguments passed to the `ltf` function for spectral analysis.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - Fourier frequencies at which the analysis is performed.
        - Amplitude spectral density of the output signal, calculated using the
          optimal spectral analysis method.

    Raises
    ------
    ValueError
        If `fs` is not positive and finite, if `inputs` is empty, if any input or `output`
        are empty, if they have different lengths, if they are not 1D arrays, or if they
        contain non-finite values.
    TypeError
        If `inputs` is not a list/sequence, or if any input or `output` cannot be converted
        to numpy arrays.
    """
    # Validate fs
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError(f"`fs` must be a positive finite float, got {fs!r}.")

    # Validate inputs is a non-empty sequence
    if not isinstance(inputs, (list, tuple)):
        raise TypeError(f"`inputs` must be a list or tuple, got {type(inputs).__name__}.")
    q = len(inputs)
    if q == 0:
        raise ValueError("`inputs` must not be empty.")

    # Convert to numpy arrays and validate
    input_arrays = []
    for i, inp in enumerate(inputs):
        inp_arr = np.asarray(inp)
        if inp_arr.ndim != 1:
            raise ValueError(
                f"`inputs[{i}]` must be a 1D array, got {inp_arr.ndim}D array."
            )
        if inp_arr.size == 0:
            raise ValueError(f"`inputs[{i}]` must not be empty.")
        if not np.all(np.isfinite(inp_arr)):
            raise ValueError(f"`inputs[{i}]` must contain only finite values.")
        input_arrays.append(inp_arr)

    output_arr = np.asarray(output)
    if output_arr.ndim != 1:
        raise ValueError(f"`output` must be a 1D array, got {output_arr.ndim}D array.")
    if output_arr.size == 0:
        raise ValueError("`output` must not be empty.")
    if not np.all(np.isfinite(output_arr)):
        raise ValueError("`output` must contain only finite values.")

    # Validate all have the same length
    N = len(input_arrays[0])
    for i, inp_arr in enumerate(input_arrays):
        if len(inp_arr) != N:
            raise ValueError(
                f"All input time series must be of equal length, "
                f"got length {len(inp_arr)} for `inputs[{i}]` but expected {N}."
            )
    if len(output_arr) != N:
        raise ValueError(
            f"The output time series must have the same length as the inputs, "
            f"got {len(output_arr)} but expected {N}."
        )

    logger.info(f"Solving {q}-dimensional problem...")

    # Dictionary to cache results of ltf calls:
    result = {}

    def get_ltf_result(key, *args, **kwargs):
        """Helper function to retrieve or compute ltf result."""
        if key not in result:
            obj = ltf(*args, **kwargs)
            result[key] = obj
        return result[key]

    logger.info("Computing the auto-spectrum of the output...")
    obj = get_ltf_result("S00", output_arr, fs, **kwargs)
    S00 = obj.Gxx
    frequencies = obj.f
    nf = obj.nf

    # Prepare data for solving the linear system:
    Tmat = np.zeros((q, q, nf), dtype=complex)  # Coherence matrix of inputs
    Svec = np.zeros(
        (q, nf), dtype=complex
    )  # Cross-spectral densities of inputs and output

    logger.info("Computing all other spectral estimates...")
    # Compute cross-spectra between inputs (this also gives us diagonal elements)
    for i in range(q):
        for j in range(i + 1, q):
            obj = get_ltf_result(
                f"T{i + 1}{j + 1}", [input_arrays[i], input_arrays[j]], fs, **kwargs
            )
            Tmat[i, j, :] = obj.Gxy
            Tmat[j, i, :] = np.conj(obj.Gxy)
            # Diagonal elements: Gxx is auto-spectrum of first input, Gyy is auto-spectrum of second
            if not np.any(Tmat[i, i, :]):
                Tmat[i, i, :] = obj.Gxx
            if not np.any(Tmat[j, j, :]):
                Tmat[j, j, :] = obj.Gyy
    
    # For single input case (q=1), compute diagonal separately
    if q == 1:
        obj = get_ltf_result("T11", [input_arrays[0], input_arrays[0]], fs, **kwargs)
        Tmat[0, 0, :] = obj.Gxx
    
    # Compute cross-spectra between inputs and output
    for i in range(q):
        obj = get_ltf_result(f"S{i + 1}0", [input_arrays[i], output_arr], fs, **kwargs)
        Svec[i, :] = obj.Gxy

    logger.info("Computing solution...")
    # Solve for the optimal transfer functions numerically:
    Hvec = np.zeros((q, nf), dtype=complex)
    for k in range(nf):
        Tk = Tmat[:, :, k]
        Sk = Svec[:, k]
        # Check condition number to detect near-singular matrices
        try:
            cond = np.linalg.cond(Tk)
            if cond > 1e12:  # Very ill-conditioned, use pinv
                Tmat_pinv = np.linalg.pinv(Tk)
                Hvec[:, k] = Tmat_pinv @ Sk
            else:
                Hvec[:, k] = np.linalg.solve(Tk, Sk)
        except np.linalg.LinAlgError:
            # Matrix is singular, use pseudo-inverse
            Tmat_pinv = np.linalg.pinv(Tk)
            Hvec[:, k] = Tmat_pinv @ Sk

    # Compute the optimal spectral density
    Sum1 = np.sum(Hvec * Svec.conj(), axis=0)
    Sum2 = np.sum(Hvec.conj() * Svec, axis=0)
    Sum3 = np.zeros(nf, dtype=complex)
    for i in range(q):
        for j in range(q):
            Sum3 += Hvec[j, :].conj() * Hvec[i, :] * Tmat[j, i, :]

    # Compute optimal analysis (Equation 8.16, page 191):
    optimal_asd = np.abs(np.sqrt(S00 - Sum1 - Sum2 + Sum3))

    logger.info("Done.")

    return frequencies, optimal_asd

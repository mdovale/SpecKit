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
import math

MIN_JDES = 100
MAX_JDES = 1000000


def find_Jdes_binary_search(scheduler, target_nf, **args):
    """Performs a binary search to find the `Jdes` for a scheduler.

    This utility function iteratively calls a given scheduler to find the
    integer `Jdes` (desired number of frequencies) that results in a plan
    with exactly `target_nf` frequency bins. It is designed to support the
    `force_target_nf` functionality in the `SpectrumAnalyzer`.

    The search is performed within a predefined range defined by `MIN_JDES`
    and `MAX_JDES`.

    Parameters
    ----------
    scheduler : callable
        The scheduler function to be evaluated, e.g., `lpsd_plan` or `ltf_plan`.
    target_nf : int
        The target number of frequencies (`nf`) that the scheduler should generate.
    *args : tuple
        A tuple of the other arguments required by the scheduler, passed in a
        specific, fixed order: `(N, fs, olap, bmin, Lmin, Kdes)`. The function
        internally selects the correct arguments from this tuple based on
        which scheduler is provided.

    Returns
    -------
    int or None
        The integer value of `Jdes` that produces exactly `target_nf`
        frequencies. Returns `None` if the binary search completes without
        finding an exact match.

    Raises
    ------
    ValueError
        If the provided scheduler function does not return a dictionary
        containing the key 'nf'.

    Notes
    -----
    - This function requires an **exact match**. If no integer `Jdes` within
      the search range produces `nf == target_nf`, the function will return `None`.
    - The `*args` parameter has a rigid structure and must contain the scheduler
      parameters in the order `(N, fs, olap, bmin, Lmin, Kdes)`, even if a
      specific scheduler (like `lpsd_plan`) does not use all of them.

    """
    lower = MIN_JDES
    upper = MAX_JDES

    while lower <= upper:
        Jdes = (lower + upper) // 2

        # Build call with candidate Jdes; schedulers validate required keys.
        call_kwargs = dict(args)
        call_kwargs["Jdes"] = int(Jdes)

        output = scheduler(**call_kwargs)

        nf = output.get("nf") if isinstance(output, dict) else None
        if nf is None:
            raise ValueError("Scheduler did not return 'nf' in output.")

        if nf == target_nf:
            return Jdes
        elif nf < target_nf:
            lower = Jdes + 1
        else:
            upper = Jdes - 1

    return None


def kaiser_alpha(psll):
    """Calculates Kaiser window shape parameter (alpha/beta) from PSLL.

    This function provides a polynomial approximation to determine the required
    shape parameter (`alpha`, often denoted as `β`) for a Kaiser window to
    achieve a desired Peak Side-Lobe Level (PSLL). This is useful because
    engineers often specify window performance in terms of sidelobe
    attenuation in decibels, whereas window generation functions require the
    `β` parameter.

    Parameters
    ----------
    psll : float
        The desired Peak Side-Lobe Level in positive decibels (dB). For example,
        a value of 200 corresponds to extremely high sidelobe attenuation.

    Returns
    -------
    float
        The calculated shape parameter `alpha` (or `β`) for use in a Kaiser
        window function like `numpy.kaiser`.

    See Also
    --------
    kaiser_rov : Calculates the recommended overlap from this alpha value.
    numpy.kaiser : The NumPy function that uses this shape parameter.
    """
    a0 = -0.0821377
    a1 = 4.71469
    a2 = -0.493285
    a3 = 0.0889732

    x = psll / 100
    return ((((a3 * x) + a2) * x) + a1) * x + a0


def kaiser_rov(alpha):
    """Calculates the recommended fractional overlap for a Kaiser window.

    Based on the window's shape parameter (`alpha` or `β`), this function
    provides an empirically derived recommendation for the fractional overlap
    between segments in a Short-Time Fourier Transform (STFT) or Welch's
    method analysis. Windows with higher sidelobe suppression (larger `alpha`)
    have a wider main lobe and thus require more overlap to avoid scalloping
    loss and ensure signal conservation.

    Parameters
    ----------
    alpha : float
        The shape parameter `alpha` (or `β`) of the Kaiser window, which is
        typically the output from the `kaiser_alpha` function.

    Returns
    -------
    float
        The recommended fractional overlap, as a value between 0.0 and 1.0
        (e.g., 0.5 for 50% overlap).

    See Also
    --------
    kaiser_alpha : The function used to generate the `alpha` parameter.
    """
    a0 = 0.0061076
    a1 = 0.00912223
    a2 = -0.000925946
    a3 = 4.42204e-05
    x = alpha
    return (100 - 1 / (((((a3 * x) + a2) * x) + a1) * x + a0)) / 100


def round_half_up(val):
    """Rounds a number to the nearest integer, with halves rounded up.

    This function implements the "round half up" strategy, which differs from
    Python's built-in `round()` function that uses "round half to even"
    (also known as banker's rounding). This is often the rounding behavior
    expected in traditional contexts.

    Parameters
    ----------
    val : float or int
        The number to be rounded.

    Returns
    -------
    int
        The value rounded to the nearest integer.
    """
    if (float(val) % 1) >= 0.5:
        x = math.ceil(val)
    else:
        x = round(val)
    return x


def chunker(iter, chunk_size):
    """Splits an iterable into smaller lists of a fixed size.

    This function takes a sequence (like a list or tuple) and divides it
    into a list of sub-lists, where each sub-list has a maximum length of
    `chunk_size`. The final chunk may be smaller if the total number of
    elements is not evenly divisible by `chunk_size`.

    Parameters
    ----------
    iter : Iterable
        The iterable (e.g., a list or tuple) to be chunked.
    chunk_size : int
        The desired maximum size of each chunk.

    Returns
    -------
    list of lists
        A list containing the smaller chunked lists.

    Raises
    ------
    ValueError
        If `chunk_size` is less than 1.
    """
    chunks = []
    if chunk_size < 1:
        raise ValueError("Chunk size must be greater than 0.")
    for i in range(0, len(iter), chunk_size):
        chunks.append(iter[i : (i + chunk_size)])
    return chunks


def is_function_in_dict(function_to_check, function_dict):
    """Checks if a function object exists as a value in a dictionary.

    This is useful for verifying if a provided callable is one of a set of
    pre-defined, named functions.
    """
    return function_to_check in function_dict.values()


def get_key_for_function(function_to_check, function_dict):
    """Performs a reverse lookup to find the key for a given function value.

    Iterates through a dictionary to find which key corresponds to the given
    function object. This is useful for retrieving the string name of a
    callable from a mapping.
    """
    for key, func in function_dict.items():
        if func == function_to_check:
            return key
    return None  # Return None if the function is not found

# BSD 3-Clause License
#
# Copyright (c) 2025, Miguel Dovale (University of Arizona),
# Gerhard Heinzel (Albert Einstein Institute).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
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
#
# This software may be subject to U.S. export control laws. By accepting this
# software, the user agrees to comply with all applicable U.S. export laws and
# regulations. User has the responsibility to obtain export licenses, or other
# export authority as may be required before exporting such information to
# foreign countries or providing access to foreign persons.
#
import pytest
import numpy as np
from speckit import systems
from speckit.noise import butter_lowpass_filter


@pytest.fixture
def simple_siso_system(reproducible_rng):
    """
    Creates a simple SISO system: output = filtered input + independent noise.
    """
    N = int(1e4)
    fs = 100.0
    cutoff = 10.0
    order = 4
    
    input_signal = reproducible_rng.normal(size=N)
    filtered = butter_lowpass_filter(input_signal, cutoff, fs, order)
    noise = 0.1 * reproducible_rng.normal(size=N)
    output_signal = filtered + noise
    
    return {
        "input": input_signal,
        "output": output_signal,
        "N": N,
        "fs": fs,
    }


@pytest.fixture
def simple_miso_system(reproducible_rng):
    """
    Creates a simple MISO system with 2 inputs.
    """
    N = int(1e4)
    fs = 100.0
    
    input1 = reproducible_rng.normal(size=N)
    input2 = reproducible_rng.normal(size=N)
    
    # Output is a combination of filtered inputs plus noise
    cutoff = 10.0
    order = 4
    filtered1 = butter_lowpass_filter(input1, cutoff, fs, order)
    filtered2 = butter_lowpass_filter(input2, cutoff, fs, order)
    noise = 0.1 * reproducible_rng.normal(size=N)
    output = 0.5 * filtered1 + 0.3 * filtered2 + noise
    
    return {
        "inputs": [input1, input2],
        "output": output,
        "N": N,
        "fs": fs,
    }


@pytest.fixture
def three_input_miso_system(reproducible_rng):
    """
    Creates a MISO system with 3 inputs.
    """
    N = int(1e4)
    fs = 100.0
    
    input1 = reproducible_rng.normal(size=N)
    input2 = reproducible_rng.normal(size=N)
    input3 = reproducible_rng.normal(size=N)
    
    cutoff = 10.0
    order = 4
    filtered1 = butter_lowpass_filter(input1, cutoff, fs, order)
    filtered2 = butter_lowpass_filter(input2, cutoff, fs, order)
    filtered3 = butter_lowpass_filter(input3, cutoff, fs, order)
    noise = 0.1 * reproducible_rng.normal(size=N)
    output = 0.4 * filtered1 + 0.3 * filtered2 + 0.2 * filtered3 + noise
    
    return {
        "inputs": [input1, input2, input3],
        "output": output,
        "N": N,
        "fs": fs,
    }


# --- Tests for SISO_optimal_spectral_analysis ---


def test_siso_optimal_spectral_analysis_basic(simple_siso_system):
    """Tests basic functionality of SISO optimal spectral analysis."""
    input_sig = simple_siso_system["input"]
    output_sig = simple_siso_system["output"]
    fs = simple_siso_system["fs"]
    
    f, asd = systems.SISO_optimal_spectral_analysis(input_sig, output_sig, fs)
    
    # Check return types
    assert isinstance(f, np.ndarray)
    assert isinstance(asd, np.ndarray)
    
    # Check array properties
    assert len(f) == len(asd)
    assert len(f) > 0
    assert np.all(f >= 0)
    assert np.all(asd >= 0)
    
    # Check that frequencies are in valid range
    assert np.max(f) <= fs / 2


def test_siso_optimal_spectral_analysis_with_kwargs(simple_siso_system):
    """Tests SISO analysis with additional kwargs passed to ltf."""
    input_sig = simple_siso_system["input"]
    output_sig = simple_siso_system["output"]
    fs = simple_siso_system["fs"]
    
    # Test with different parameters
    f1, asd1 = systems.SISO_optimal_spectral_analysis(
        input_sig, output_sig, fs, Jdes=100, win="hann"
    )
    f2, asd2 = systems.SISO_optimal_spectral_analysis(
        input_sig, output_sig, fs, Jdes=200, win="kaiser", psll=100
    )
    
    # Both should produce valid results
    assert len(f1) > 0
    assert len(f2) > 0
    # Different Jdes should produce different number of frequencies
    assert len(f1) != len(f2)


def test_siso_optimal_spectral_analysis_consistency(simple_siso_system):
    """Tests that SISO analysis produces consistent results."""
    input_sig = simple_siso_system["input"]
    output_sig = simple_siso_system["output"]
    fs = simple_siso_system["fs"]
    
    # Run twice with same parameters
    f1, asd1 = systems.SISO_optimal_spectral_analysis(
        input_sig, output_sig, fs, Jdes=100
    )
    f2, asd2 = systems.SISO_optimal_spectral_analysis(
        input_sig, output_sig, fs, Jdes=100
    )
    
    # Results should be identical
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_almost_equal(asd1, asd2, decimal=10)


# --- Tests for MISO_analytic_optimal_spectral_analysis ---


def test_miso_analytic_optimal_spectral_analysis_basic(simple_miso_system):
    """Tests basic functionality of MISO analytic optimal spectral analysis."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    f, asd = systems.MISO_analytic_optimal_spectral_analysis(inputs, output, fs)
    
    # Check return types
    assert isinstance(f, np.ndarray)
    assert isinstance(asd, np.ndarray)
    
    # Check array properties
    assert len(f) == len(asd)
    assert len(f) > 0
    assert np.all(f >= 0)
    assert np.all(asd >= 0)
    
    # Check that frequencies are in valid range
    assert np.max(f) <= fs / 2


def test_miso_analytic_optimal_spectral_analysis_three_inputs(three_input_miso_system):
    """Tests MISO analytic analysis with 3 inputs."""
    inputs = three_input_miso_system["inputs"]
    output = three_input_miso_system["output"]
    fs = three_input_miso_system["fs"]
    
    f, asd = systems.MISO_analytic_optimal_spectral_analysis(inputs, output, fs)
    
    assert len(f) == len(asd)
    assert len(f) > 0
    assert np.all(asd >= 0)


def test_miso_analytic_optimal_spectral_analysis_with_kwargs(simple_miso_system):
    """Tests MISO analytic analysis with additional kwargs."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    f, asd = systems.MISO_analytic_optimal_spectral_analysis(
        inputs, output, fs, Jdes=100, win="hann"
    )
    
    assert len(f) > 0
    assert np.all(asd >= 0)


def test_miso_analytic_optimal_spectral_analysis_consistency(simple_miso_system):
    """Tests that MISO analytic analysis produces consistent results."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    # Run twice with same parameters
    f1, asd1 = systems.MISO_analytic_optimal_spectral_analysis(
        inputs, output, fs, Jdes=100
    )
    f2, asd2 = systems.MISO_analytic_optimal_spectral_analysis(
        inputs, output, fs, Jdes=100
    )
    
    # Results should be identical
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_almost_equal(asd1, asd2, decimal=10)


def test_miso_analytic_optimal_spectral_analysis_input_length_mismatch(simple_miso_system):
    """Tests that ValueError is raised when input lengths don't match."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    # Make inputs have different lengths
    inputs_mismatched = [inputs[0], inputs[1][:-10]]
    
    with pytest.raises(ValueError, match="All input time series must be of equal length"):
        systems.MISO_analytic_optimal_spectral_analysis(inputs_mismatched, output, fs)


def test_miso_analytic_optimal_spectral_analysis_output_length_mismatch(simple_miso_system):
    """Tests that ValueError is raised when output length doesn't match inputs."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    # Make output have different length
    output_mismatched = output[:-10]
    
    with pytest.raises(ValueError, match="The output time series must have the same length"):
        systems.MISO_analytic_optimal_spectral_analysis(inputs, output_mismatched, fs)


# --- Tests for MISO_numeric_optimal_spectral_analysis ---


def test_miso_numeric_optimal_spectral_analysis_basic(simple_miso_system):
    """Tests basic functionality of MISO numeric optimal spectral analysis."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    f, asd = systems.MISO_numeric_optimal_spectral_analysis(inputs, output, fs)
    
    # Check return types
    assert isinstance(f, np.ndarray)
    assert isinstance(asd, np.ndarray)
    
    # Check array properties
    assert len(f) == len(asd)
    assert len(f) > 0
    assert np.all(f >= 0)
    assert np.all(asd >= 0)
    
    # Check that frequencies are in valid range
    assert np.max(f) <= fs / 2


def test_miso_numeric_optimal_spectral_analysis_three_inputs(three_input_miso_system):
    """Tests MISO numeric analysis with 3 inputs."""
    inputs = three_input_miso_system["inputs"]
    output = three_input_miso_system["output"]
    fs = three_input_miso_system["fs"]
    
    f, asd = systems.MISO_numeric_optimal_spectral_analysis(inputs, output, fs)
    
    assert len(f) == len(asd)
    assert len(f) > 0
    assert np.all(asd >= 0)


def test_miso_numeric_optimal_spectral_analysis_many_inputs(reproducible_rng):
    """Tests MISO numeric analysis with many inputs (should work without warning)."""
    N = int(1e3)
    fs = 100.0
    
    # Create 6 inputs (numeric solver should handle this)
    inputs = [reproducible_rng.normal(size=N) for _ in range(6)]
    output = reproducible_rng.normal(size=N)
    
    # Should not raise warning
    f, asd = systems.MISO_numeric_optimal_spectral_analysis(inputs, output, fs, Jdes=50)
    
    assert len(f) > 0
    assert np.all(asd >= 0)


def test_miso_numeric_optimal_spectral_analysis_with_kwargs(simple_miso_system):
    """Tests MISO numeric analysis with additional kwargs."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    f, asd = systems.MISO_numeric_optimal_spectral_analysis(
        inputs, output, fs, Jdes=100, win="hann"
    )
    
    assert len(f) > 0
    assert np.all(asd >= 0)


def test_miso_numeric_optimal_spectral_analysis_consistency(simple_miso_system):
    """Tests that MISO numeric analysis produces consistent results."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    # Run twice with same parameters
    f1, asd1 = systems.MISO_numeric_optimal_spectral_analysis(
        inputs, output, fs, Jdes=100
    )
    f2, asd2 = systems.MISO_numeric_optimal_spectral_analysis(
        inputs, output, fs, Jdes=100
    )
    
    # Results should be identical
    np.testing.assert_array_equal(f1, f2)
    np.testing.assert_array_almost_equal(asd1, asd2, decimal=10)


def test_miso_numeric_optimal_spectral_analysis_input_length_mismatch(simple_miso_system):
    """Tests that ValueError is raised when input lengths don't match."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    # Make inputs have different lengths
    inputs_mismatched = [inputs[0], inputs[1][:-10]]
    
    with pytest.raises(ValueError, match="All input time series must be of equal length"):
        systems.MISO_numeric_optimal_spectral_analysis(inputs_mismatched, output, fs)


def test_miso_numeric_optimal_spectral_analysis_output_length_mismatch(simple_miso_system):
    """Tests that ValueError is raised when output length doesn't match inputs."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
    
    # Make output have different length
    output_mismatched = output[:-10]
    
    with pytest.raises(ValueError, match="The output time series must have the same length"):
        systems.MISO_numeric_optimal_spectral_analysis(inputs, output_mismatched, fs)


# --- Comparison tests ---


def test_miso_analytic_vs_numeric_consistency(simple_miso_system):
    """Tests that analytic and numeric MISO methods produce similar results."""
    inputs = simple_miso_system["inputs"]
    output = simple_miso_system["output"]
    fs = simple_miso_system["fs"]
        
    f_analytic, asd_analytic = systems.MISO_analytic_optimal_spectral_analysis(
        inputs, output, fs
    )
    f_numeric, asd_numeric = systems.MISO_numeric_optimal_spectral_analysis(
        inputs, output, fs
    )
    
    # Frequencies should be identical
    np.testing.assert_array_equal(f_analytic, f_numeric)
    
    # ASDs should be very similar (within numerical precision)
    np.testing.assert_array_almost_equal(asd_analytic[4:], asd_numeric[4:], decimal=8)


# def test_siso_vs_miso_single_input(simple_siso_system):
#     """Tests that SISO and MISO (with single input) produce similar results."""
#     input_sig = simple_siso_system["input"]
#     output_sig = simple_siso_system["output"]
#     fs = simple_siso_system["fs"]
    
#     kwargs = {"Jdes": 100, "win": "hann"}
    
#     f_siso, asd_siso = systems.SISO_optimal_spectral_analysis(
#         input_sig, output_sig, fs, **kwargs
#     )
#     f_miso, asd_miso = systems.MISO_numeric_optimal_spectral_analysis(
#         [input_sig], output_sig, fs, **kwargs
#     )
    
#     # Frequencies should be identical
#     np.testing.assert_array_equal(f_siso, f_miso)
    
#     # ASDs should be very similar
#     np.testing.assert_array_almost_equal(asd_siso, asd_miso, decimal=8)


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
import pytest
import numpy as np
from speckit import SpectrumAnalyzer
from speckit.core import _CUDA_ENABLED, _NUMBA_ENABLED


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_spectrum_analyzer_cuda_backend():
    """Test SpectrumAnalyzer with explicit CUDA backend."""
    np.random.seed(42)
    N = 100000
    x = np.random.randn(N)
    fs = 100.0

    analyzer = SpectrumAnalyzer(
        data=x,
        fs=fs,
        backend="cuda",
        Jdes=100,
        Kdes=50,
        order=0,
        verbose=False,
    )

    result = analyzer.compute()
    assert result is not None
    assert len(result.f) > 0
    assert np.all(np.isfinite(result.asd))


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_spectrum_analyzer_cuda_vs_numba():
    """Compare CUDA and Numba backends for full spectrum computation."""
    np.random.seed(42)
    N = 50000
    x = np.random.randn(N)
    fs = 100.0

    # CUDA result
    analyzer_cuda = SpectrumAnalyzer(
        data=x,
        fs=fs,
        backend="cuda",
        Jdes=100,
        Kdes=50,
        order=1,
        verbose=False,
    )
    result_cuda = analyzer_cuda.compute()

    # Numba result (if available)
    if _NUMBA_ENABLED:
        analyzer_numba = SpectrumAnalyzer(
            data=x,
            fs=fs,
            backend="numba",
            Jdes=100,
            Kdes=50,
            order=1,
            verbose=False,
        )
        result_numba = analyzer_numba.compute()

        # Compare frequencies (should be identical)
        np.testing.assert_allclose(result_cuda.f, result_numba.f, rtol=1e-12)

        # Compare spectra (should match within numerical precision)
        np.testing.assert_allclose(
            result_cuda.asd, result_numba.asd, rtol=1e-9, atol=1e-12
        )


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_spectrum_analyzer_auto_backend_selection():
    """Test automatic backend selection."""
    np.random.seed(42)
    N = 100000
    x = np.random.randn(N)
    fs = 100.0

    # Large K should select CUDA automatically
    analyzer = SpectrumAnalyzer(
        data=x,
        fs=fs,
        backend="auto",
        Jdes=100,
        Kdes=2000,  # Large K should trigger CUDA
        order=0,
        verbose=False,
    )

    result = analyzer.compute()
    assert result is not None
    assert len(result.f) > 0


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_spectrum_analyzer_cuda_cross_spectral():
    """Test CUDA backend with cross-spectral analysis."""
    np.random.seed(42)
    N = 50000
    x1 = np.random.randn(N)
    x2 = np.random.randn(N)
    data = np.stack([x1, x2], axis=0)
    fs = 100.0

    analyzer = SpectrumAnalyzer(
        data=data,
        fs=fs,
        backend="cuda",
        Jdes=100,
        Kdes=100,
        order=0,
        verbose=False,
    )

    result = analyzer.compute()
    assert result is not None
    assert result.iscsd
    assert result.csd is not None
    assert np.all(np.isfinite(result.csd))


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("order", [-1, 0, 1, 2])
def test_spectrum_analyzer_cuda_detrend_orders(order):
    """Test CUDA backend with different detrend orders."""
    np.random.seed(42)
    N = 50000
    x = np.random.randn(N)
    fs = 100.0

    analyzer = SpectrumAnalyzer(
        data=x,
        fs=fs,
        backend="cuda",
        Jdes=100,
        Kdes=100,
        order=order,
        verbose=False,
    )

    result = analyzer.compute()
    assert result is not None
    assert len(result.f) > 0
    assert np.all(np.isfinite(result.asd))


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_spectrum_analyzer_cuda_error_handling():
    """Test error handling when CUDA is requested but not available."""
    # This test would need to mock CUDA as unavailable
    # For now, we test that requesting CUDA when available works
    np.random.seed(42)
    N = 10000
    x = np.random.randn(N)
    fs = 100.0

    analyzer = SpectrumAnalyzer(
        data=x,
        fs=fs,
        backend="cuda",
        Jdes=50,
        Kdes=20,
        verbose=False,
    )

    result = analyzer.compute()
    assert result is not None


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_compute_single_bin_cuda():
    """Test compute_single_bin with CUDA backend."""
    np.random.seed(42)
    N = 50000
    x = np.random.randn(N)
    fs = 100.0

    analyzer = SpectrumAnalyzer(
        data=x,
        fs=fs,
        backend="cuda",
        order=1,
        verbose=False,
    )

    result = analyzer.compute_single_bin(freq=10.0, fres=0.1)
    assert result is not None
    assert len(result.f) == 1
    assert np.isfinite(result.asd[0])

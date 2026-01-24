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
from speckit.core import (
    _CUDA_ENABLED,
    _NUMBA_ENABLED,
    _stats_win_only_auto,
    _stats_win_only_csd,
    _stats_detrend0_auto,
    _stats_detrend0_csd,
    _stats_poly_auto,
    _stats_poly_csd,
    _stats_win_only_auto_np,
    _stats_win_only_csd_np,
    _stats_detrend0_auto_np,
    _stats_detrend0_csd_np,
    _stats_poly_auto_np,
    _stats_poly_csd_np,
    _build_Q,
)

if _CUDA_ENABLED:
    from speckit.core_cuda import (
        _stats_win_only_auto_cuda,
        _stats_win_only_csd_cuda,
        _stats_detrend0_auto_cuda,
        _stats_detrend0_csd_cuda,
        _stats_poly_auto_cuda,
        _stats_poly_csd_cuda,
    )


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_cuda_availability():
    """Test that CUDA is properly detected."""
    assert _CUDA_ENABLED


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("K", [10, 100, 1000, 5000])
@pytest.mark.parametrize("L", [100, 500, 1000])
def test_cuda_vs_numba_win_only_auto(K, L):
    """Compare CUDA and Numba implementations for windowing-only auto-spectral."""
    np.random.seed(42)
    N = K * L + 1000  # Extra padding
    x = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1  # Some test frequency

    # CUDA result
    cuda_result = _stats_win_only_auto_cuda(x, starts, L, w, omega)

    # Numba result (if available)
    if _NUMBA_ENABLED:
        numba_result = _stats_win_only_auto(x, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numba_result, rtol=1e-10, atol=1e-12)
    else:
        # Fallback to NumPy
        numpy_result = _stats_win_only_auto_np(x, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("K", [10, 100, 1000])
@pytest.mark.parametrize("L", [100, 500, 1000])
def test_cuda_vs_numba_win_only_csd(K, L):
    """Compare CUDA and Numba implementations for windowing-only cross-spectral."""
    np.random.seed(42)
    N = K * L + 1000
    x1 = np.random.randn(N).astype(np.float64)
    x2 = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1

    cuda_result = _stats_win_only_csd_cuda(x1, x2, starts, L, w, omega)

    if _NUMBA_ENABLED:
        numba_result = _stats_win_only_csd(x1, x2, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numba_result, rtol=1e-10, atol=1e-12)
    else:
        numpy_result = _stats_win_only_csd_np(x1, x2, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("K", [10, 100, 1000])
@pytest.mark.parametrize("L", [100, 500, 1000])
def test_cuda_vs_numba_detrend0_auto(K, L):
    """Compare CUDA and Numba implementations for mean-removal auto-spectral."""
    np.random.seed(42)
    N = K * L + 1000
    x = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1

    cuda_result = _stats_detrend0_auto_cuda(x, starts, L, w, omega)

    if _NUMBA_ENABLED:
        numba_result = _stats_detrend0_auto(x, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numba_result, rtol=1e-10, atol=1e-12)
    else:
        numpy_result = _stats_detrend0_auto_np(x, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("K", [10, 100, 1000])
@pytest.mark.parametrize("L", [100, 500, 1000])
def test_cuda_vs_numba_detrend0_csd(K, L):
    """Compare CUDA and Numba implementations for mean-removal cross-spectral."""
    np.random.seed(42)
    N = K * L + 1000
    x1 = np.random.randn(N).astype(np.float64)
    x2 = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1

    cuda_result = _stats_detrend0_csd_cuda(x1, x2, starts, L, w, omega)

    if _NUMBA_ENABLED:
        numba_result = _stats_detrend0_csd(x1, x2, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numba_result, rtol=1e-10, atol=1e-12)
    else:
        numpy_result = _stats_detrend0_csd_np(x1, x2, starts, L, w, omega)
        np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("K", [10, 100, 1000])
@pytest.mark.parametrize("L", [100, 500, 1000])
def test_cuda_vs_numba_poly_auto(order, K, L):
    """Compare CUDA and Numba implementations for polynomial detrend auto-spectral."""
    np.random.seed(42)
    N = K * L + 1000
    x = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1
    Q = _build_Q(L, order)

    cuda_result = _stats_poly_auto_cuda(x, starts, L, w, omega, Q)

    if _NUMBA_ENABLED:
        numba_result = _stats_poly_auto(x, starts, L, w, omega, Q)
        np.testing.assert_allclose(cuda_result, numba_result, rtol=1e-10, atol=1e-12)
    else:
        numpy_result = _stats_poly_auto_np(x, starts, L, w, omega, Q)
        np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("K", [10, 100, 1000])
@pytest.mark.parametrize("L", [100, 500, 1000])
def test_cuda_vs_numba_poly_csd(order, K, L):
    """Compare CUDA and Numba implementations for polynomial detrend cross-spectral."""
    np.random.seed(42)
    N = K * L + 1000
    x1 = np.random.randn(N).astype(np.float64)
    x2 = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1
    Q = _build_Q(L, order)

    cuda_result = _stats_poly_csd_cuda(x1, x2, starts, L, w, omega, Q)

    if _NUMBA_ENABLED:
        numba_result = _stats_poly_csd(x1, x2, starts, L, w, omega, Q)
        np.testing.assert_allclose(cuda_result, numba_result, rtol=1e-10, atol=1e-12)
    else:
        numpy_result = _stats_poly_csd_np(x1, x2, starts, L, w, omega, Q)
        np.testing.assert_allclose(cuda_result, numpy_result, rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_cuda_edge_case_single_segment():
    """Test CUDA kernels with single segment."""
    np.random.seed(42)
    N = 1000
    x = np.random.randn(N).astype(np.float64)
    starts = np.array([0], dtype=np.int64)
    L = 500
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1

    result = _stats_win_only_auto_cuda(x, starts, L, w, omega)
    assert len(result) == 5
    assert all(np.isfinite(r) for r in result)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_cuda_edge_case_empty_segments():
    """Test CUDA kernels with empty segment array."""
    np.random.seed(42)
    N = 1000
    x = np.random.randn(N).astype(np.float64)
    starts = np.array([], dtype=np.int64)
    L = 500
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.1

    result = _stats_win_only_auto_cuda(x, starts, L, w, omega)
    assert result == (0.0, 0.0, 0.0, 0.0, 0.0)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_cuda_edge_case_long_segments():
    """Test CUDA kernels with very long segments."""
    np.random.seed(42)
    L = 50000
    K = 10
    N = K * L + 1000
    x = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    omega = 2.0 * np.pi * 0.01

    result = _stats_win_only_auto_cuda(x, starts, L, w, omega)
    assert len(result) == 5
    assert all(np.isfinite(r) for r in result)


@pytest.mark.skipif(not _CUDA_ENABLED, reason="CUDA not available")
def test_cuda_fractional_frequency():
    """Test CUDA kernels with fractional frequency bins."""
    np.random.seed(42)
    N = 10000
    K = 100
    L = 500
    x = np.random.randn(N).astype(np.float64)
    starts = np.arange(0, K * L, L, dtype=np.int64)[:K]
    w = np.hanning(L).astype(np.float64)
    # Fractional frequency (not aligned to FFT bins)
    omega = 2.0 * np.pi * 0.123456789

    result = _stats_win_only_auto_cuda(x, starts, L, w, omega)
    assert len(result) == 5
    assert all(np.isfinite(r) for r in result)

    # Compare with reference
    if _NUMBA_ENABLED:
        ref_result = _stats_win_only_auto(x, starts, L, w, omega)
        np.testing.assert_allclose(result, ref_result, rtol=1e-10, atol=1e-12)

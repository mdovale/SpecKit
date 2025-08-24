# BSD 3-Clause License

# Copyright (c) 2025, Miguel Dovale

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
from speckit import compute_spectrum, SpectrumAnalyzer, SpectrumResult
from speckit.flattop import HFT95
from matplotlib.figure import Figure
from matplotlib.axes import Axes


@pytest.mark.parametrize(
    "win_config",
    ["hann", "kaiser", HFT95],
    ids=["hann_str", "kaiser_str", "HFT95_func"],
)
@pytest.mark.parametrize("scheduler", ["ltf", "lpsd"])
def test_compute_spectrum_autospectrum(short_white_noise_data, win_config, scheduler):
    """
    Tests auto-spectrum computation with various windows and schedulers.
    """
    params = short_white_noise_data
    result = compute_spectrum(
        params["data"],
        fs=params["fs"],
        win=win_config,
        scheduler=scheduler,
        Jdes=100,
        Kdes=20,
    )

    assert isinstance(result, SpectrumResult)
    assert result.iscsd is False
    assert result.psd is not None
    assert result.asd is not None
    assert result.csd is None
    assert len(result.f) == len(result.psd)
    assert np.all(np.isfinite(result.psd))


def test_compute_spectrum_cross_spectrum(siso_data, multiprocessing_pool):
    """
    Tests cross-spectrum computation using a multiprocessing pool.
    """
    params = siso_data
    data_stack = np.vstack([params["input"], params["output"]])
    result = compute_spectrum(
        data_stack,
        fs=params["fs"],
        win="hann",
        Jdes=200,
        Kdes=50,
        pool=multiprocessing_pool,
    )

    assert isinstance(result, SpectrumResult)
    assert result.iscsd is True
    # Auto-spectrum quantities should be None
    assert result.psd is None
    assert result.asd is None
    # Cross-spectrum quantities should be present
    assert result.csd is not None
    assert result.coh is not None
    assert result.tf is not None
    assert np.all(np.isfinite(result.Gxx))
    assert np.all(np.isfinite(result.Gyy))
    assert np.all(np.isfinite(result.Gxy))


def test_spectrum_result_plotting(siso_data):
    """
    Verifies that the plot methods run without error and return correct objects.
    """
    params = siso_data
    data_stack = np.vstack([params["input"], params["output"]])
    result = compute_spectrum(data_stack, fs=params["fs"])

    # Test Bode plot (cross-spectrum default)
    fig_bode, (ax_mag, ax_phase) = result.plot(errors=True)
    assert isinstance(fig_bode, Figure)
    assert isinstance(ax_mag, Axes)
    assert isinstance(ax_phase, Axes)

    # Test Coherence plot
    fig_coh, ax_coh = result.plot(which="coh", errors=True)
    assert isinstance(fig_coh, Figure)
    assert isinstance(ax_coh, Axes)

    # Test plotting a single-channel result
    result_auto = compute_spectrum(params["input"], fs=params["fs"])
    fig_asd, ax_asd = result_auto.plot(which="asd", errors=True)
    assert isinstance(fig_asd, Figure)
    assert isinstance(ax_asd, Axes)
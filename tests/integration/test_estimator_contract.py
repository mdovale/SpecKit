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
"""Locked 1.0.3 spectra, the missing-sample policy, and the PSD interval."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from scipy.stats import chi2

from speckit import SpectrumAnalyzer, compute_single_bin, compute_spectrum
from speckit.plan import SpectralPlan

# Generator seed 20261025, N = 4096, fs = 200 Hz, Jdes = 64, Kdes = 20,
# default vectorized_ltf plan and Kaiser window. SHA-256 prefixes of the
# raw array bytes, taken from release 1.0.3 before the missing-sample
# change. Finite inputs must keep these values.
_LOCK_N = 4096
_LOCK_FS = 200.0
_LOCK_JDES = 64
_LOCK_KDES = 20
_GXX_SHA = {
    -1: "b07ae51263a2fd37",
    0: "142152a44953f7bd",
    1: "cfe85be791996e1d",
    2: "0da321a96059e67f",
}
_GXY_SHA = {
    -1: "452a1cd3c87f9a1a",
    0: "ba4a8bfef286d10d",
    1: "c1d3e2c98107167c",
    2: "b44fce947e17cea6",
}
_NU_SHA = "19adb46c868c1ece"
_F_SHA = "799d4cf428478b8b"
_L_SHA = "05cf60e2ebeb73aa"
_R_SHA = "41d85d78cc401e26"
_K_SHA = "095e9281063145bf"
_D_SHA = "6d4c23b002c1ab2e"


def _sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _locked_series():
    rng = np.random.default_rng(20261025)
    x = rng.standard_normal(_LOCK_N)
    y = np.empty(_LOCK_N)
    y[:3] = rng.standard_normal(3)
    y[3:] = x[:-3]
    return x, np.vstack([x, y])


def test_released_103_spectra_stay_put():
    x, pair = _locked_series()
    for order, gxx_sha in _GXX_SHA.items():
        auto = compute_spectrum(
            x, _LOCK_FS, Jdes=_LOCK_JDES, Kdes=_LOCK_KDES, order=order
        )
        cross = compute_spectrum(
            pair, _LOCK_FS, Jdes=_LOCK_JDES, Kdes=_LOCK_KDES, order=order
        )
        assert auto.nf == 47
        assert np.isfinite(auto.Gxx).all()
        assert np.isfinite(cross.Gxy).all()
        assert _sha(auto.Gxx) == gxx_sha
        assert _sha(auto.nu) == _NU_SHA
        assert _sha(np.ascontiguousarray(cross.Gxy).view(np.float64)) == _GXY_SHA[order]
        assert _sha(auto.f) == _F_SHA
        assert _sha(np.asarray(auto.L)) == _L_SHA
        assert _sha(auto.r) == _R_SHA
        assert _sha(np.asarray(auto.K, dtype=np.int64)) == _K_SHA
        starts = np.concatenate([np.asarray(d, dtype=np.int64) for d in auto.D])
        assert _sha(starts) == _D_SHA


def _plan_config(plan):
    return {
        "fs": plan.fs,
        "N": plan.N,
        "olap": plan.olap,
        "bmin": plan.bmin,
        "Lmin": plan.Lmin,
        "Jdes": plan.Jdes,
        "Kdes": plan.Kdes,
        "order": plan.order,
        "scheduler": plan.scheduler,
        "window": plan.window,
        "missing": plan.missing,
        "psll": plan.psll,
        "band": plan.band,
    }


def test_spectral_plan_preserves_the_lock_and_the_report():
    x, _pair = _locked_series()
    analyzer = SpectrumAnalyzer(x, _LOCK_FS, Jdes=_LOCK_JDES, Kdes=_LOCK_KDES, order=-1)
    plan = analyzer.spectral_plan()
    view = plan.as_dict()
    assert _sha(view["f"]) == _F_SHA
    assert _sha(np.asarray(view["L"])) == _L_SHA
    assert _sha(view["r"]) == _R_SHA
    assert _sha(np.asarray(view["K"], dtype=np.int64)) == _K_SHA
    starts = np.concatenate([np.asarray(row, dtype=np.int64) for row in view["D"]])
    assert _sha(starts) == _D_SHA
    before = view["f"].tobytes()
    facts = plan.report()
    assert facts["nf"] == 47
    assert facts["f_min"] <= facts["f_max"]
    assert "K median" in facts["text"]
    assert "L at bound" in facts["text"]
    assert analyzer.plan()["f"].tobytes() == before
    again = SpectrumAnalyzer(x, _LOCK_FS, Jdes=_LOCK_JDES, Kdes=_LOCK_KDES, order=-1)
    assert again.spectral_plan().digest() == plan.digest()
    shifted = plan.as_dict()
    moved = False
    for index, row in enumerate(shifted["D"]):
        room = plan.N - int(plan.L[index])
        for sample, start in enumerate(row):
            if int(start) + 1 <= room:
                edited = np.array(row, copy=True)
                edited[sample] = int(start) + 1
                shifted["D"][index] = edited
                moved = True
                break
        if moved:
            break
    assert moved
    assert (
        SpectralPlan.from_dict(shifted, **_plan_config(plan)).digest() != plan.digest()
    )
    other = SpectrumAnalyzer(
        x, _LOCK_FS, Jdes=_LOCK_JDES, Kdes=_LOCK_KDES, order=1, win="hann", olap=0.5
    )
    assert other.spectral_plan().digest() != plan.digest()
    assert SpectralPlan.from_json(plan.to_json()).digest() == plan.digest()
    assert _sha(analyzer.compute().Gxx) == _GXX_SHA[-1]


def test_replicates_reduce_to_the_summary_on_the_lock_series():
    _x, pair = _locked_series()
    kwargs = {
        "freq": 20.0,
        "L": 256,
        "order": -1,
        "win": "hann",
        "olap": 0.0,
        "Jdes": _LOCK_JDES,
        "Kdes": _LOCK_KDES,
    }
    summary = compute_single_bin(pair, _LOCK_FS, storage="summary", **kwargs)
    replicates = compute_single_bin(pair, _LOCK_FS, storage="replicates", **kwargs)
    assert summary.Xseg is None
    assert replicates.Xseg is not None and replicates.Yseg is not None
    assert float(np.mean(np.abs(replicates.Xseg) ** 2)) == pytest.approx(
        float(summary.XX[0]), rel=1e-12
    )
    cross = np.mean(replicates.Xseg * np.conj(replicates.Yseg))
    assert complex(cross) == pytest.approx(complex(summary.XY[0]), rel=1e-12, abs=1e-12)
    with pytest.raises(ValueError, match="storage"):
        compute_single_bin(pair, _LOCK_FS, storage="samples", **kwargs)


def test_missing_raise_reports_the_count():
    x = np.ones(32)
    x[4] = np.nan
    x[5] = np.inf
    with pytest.raises(ValueError, match=r"2 non-finite"):
        SpectrumAnalyzer(x, 10.0)


def test_missing_raise_counts_one_channel_of_a_pair():
    pair = np.ones((2, 16))
    pair[1, 3] = np.nan
    with pytest.raises(ValueError, match=r"1 non-finite"):
        compute_spectrum(pair, 8.0)


def test_missing_zero_matches_an_explicit_zero_series():
    rng = np.random.default_rng(1)
    clean = rng.standard_normal(256)
    dirty = clean.copy()
    dirty[10] = np.nan
    dirty[11] = -np.inf
    clean[10] = 0.0
    clean[11] = 0.0
    replaced = SpectrumAnalyzer(dirty, 16.0, missing="zero", Jdes=16, Kdes=4, order=-1)
    explicit = SpectrumAnalyzer(clean, 16.0, Jdes=16, Kdes=4, order=-1)
    assert replaced.n_replaced == 2
    assert explicit.n_replaced == 0
    assert np.array_equal(replaced.data, explicit.data)
    assert np.array_equal(replaced.compute().Gxx, explicit.compute().Gxx)


def test_missing_unknown_policy_raises():
    with pytest.raises(ValueError, match="missing must"):
        SpectrumAnalyzer(np.ones(8), 1.0, missing="fill")


def test_gxx_interval_is_the_chi_square_formula():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(2048)
    result = compute_single_bin(
        x, 64.0, freq=8.0, L=128, order=-1, win="hann", olap=0.0
    )
    low, high = result.gxx_interval(0.95)
    nu = float(result.nu[0])
    gxx = float(result.Gxx[0])
    assert low[0] == pytest.approx(nu * gxx / chi2.ppf(0.975, nu))
    assert high[0] == pytest.approx(nu * gxx / chi2.ppf(0.025, nu))
    assert low[0] < gxx < high[0]
    with pytest.raises(ValueError, match="conf"):
        result.gxx_interval(0.0)


def test_gxx_interval_covers_white_noise_at_one_interior_bin():
    """Independent trials. A whole-spectrum fraction is not a binomial check.

    Hann, order=-1, no overlap, so nu = 2 K. fs = 256 Hz, L = 256, K = 80,
    f = 32 Hz (an interior integer bin). Unit-variance white noise has
    one-sided level 2/fs. 800 trials, seed 20261025. The half-width is
    three binomial standard errors of a 0.95 rate.
    """
    rng = np.random.default_rng(20261025)
    fs = 256.0
    length = 256
    averages = 80
    n = length * averages
    freq = 32.0
    n_trial = 800
    truth = 2.0 / fs
    inside = 0
    nu = None
    for _ in range(n_trial):
        x = rng.standard_normal(n)
        result = compute_single_bin(
            x, fs, freq=freq, L=length, order=-1, win="hann", olap=0.0
        )
        if nu is None:
            nu = float(result.nu[0])
        low, high = result.gxx_interval(0.95)
        inside += int(low[0] <= truth <= high[0])
    assert nu == pytest.approx(2.0 * averages)
    fraction = inside / n_trial
    standard_error = np.sqrt(0.95 * 0.05 / n_trial)
    assert abs(fraction - 0.95) <= 3.0 * standard_error, (fraction, standard_error)

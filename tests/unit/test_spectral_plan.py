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
"""SpectralPlan validation, digest, and report."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from speckit.plan import SCHEMA, SpectralPlan
from speckit.schedulers import lpsd_plan, ltf_plan, new_ltf_plan, vectorized_ltf_plan

_SCHEDULER_PARAMS = {
    "N": int(1e6),
    "fs": 1.0,
    "olap": 0.75,
    "bmin": 5.0,
    "Lmin": 1000,
    "Jdes": 1000,
    "Kdes": 100,
    "num_patch_pts": 50,
}
_NAMES = {
    lpsd_plan: "lpsd",
    ltf_plan: "ltf",
    vectorized_ltf_plan: "vectorized_ltf",
    new_ltf_plan: "new_ltf",
}


def _config(**overrides):
    config = {
        "fs": 4.0,
        "N": 8,
        "olap": 0.0,
        "bmin": 1.0,
        "Lmin": 1,
        "Jdes": 1,
        "Kdes": 1,
        "order": -1,
        "scheduler": "ltf",
        "window": "hann",
        "missing": "raise",
        "psll": None,
        "band": None,
    }
    config.update(overrides)
    return config


def _one_bin():
    return {
        "f": np.array([1.0]),
        "r": np.array([1.0]),
        "b": np.array([1.0]),
        "L": np.array([4], dtype=np.int64),
        "K": np.array([2], dtype=np.int64),
        "navg": np.array([2], dtype=np.int64),
        "O": np.array([0.0]),
        "D": [np.array([0, 2], dtype=np.int64)],
        "nf": 1,
    }


def _plan(**overrides):
    return SpectralPlan.from_dict(_one_bin(), **_config(**overrides))


def test_schema_is_one_and_independent_of_the_package_version():
    blob = _plan().canonical_bytes()
    assert struct.unpack("<I", blob[:4])[0] == SCHEMA == 1


def test_rejects_a_missing_key_and_a_length_mismatch():
    raw = _one_bin()
    del raw["r"]
    with pytest.raises(ValueError, match="missing key"):
        SpectralPlan.from_dict(raw, **_config())
    raw = _one_bin()
    raw["r"] = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="inconsistent lengths"):
        SpectralPlan.from_dict(raw, **_config())


def test_rejects_empty_starts_bounds_and_k_mismatch():
    raw = _one_bin()
    raw["D"] = [np.array([], dtype=np.int64)]
    raw["K"] = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError, match="empty"):
        SpectralPlan.from_dict(raw, **_config())

    raw = _one_bin()
    raw["D"] = [np.array([-1, 2], dtype=np.int64)]
    with pytest.raises(ValueError, match="out-of-bounds"):
        SpectralPlan.from_dict(raw, **_config())

    raw = _one_bin()
    raw["K"] = np.array([3], dtype=np.int64)
    with pytest.raises(ValueError, match="does not match"):
        SpectralPlan.from_dict(raw, **_config())


def test_copy_and_report_do_not_change_the_plan():
    plan = _plan()
    before = plan.canonical_bytes()
    view = plan.as_dict()
    view["f"][0] = 0.0
    view["D"][0][0] = 1
    facts = plan.report()
    assert plan.canonical_bytes() == before
    assert facts["nf"] == 1
    assert facts["n_k1"] == 0
    assert "K median" in facts["text"]
    assert plan.f[0] == 1.0
    with pytest.raises(ValueError):
        plan.f[0] = 0.0


def test_digest_changes_with_start_window_and_order_and_survives_json():
    plan = _plan()
    assert plan.digest() == _plan().digest()
    raw = plan.as_dict()
    raw["D"][0] = np.array([1, 2], dtype=np.int64)
    moved = SpectralPlan.from_dict(raw, **_config())
    assert moved.digest() != plan.digest()
    assert _plan(window="kaiser", psll=200.0).digest() != plan.digest()
    assert _plan(order=1).digest() != plan.digest()
    restored = SpectralPlan.from_json(plan.to_json())
    assert restored.digest() == plan.digest()


@pytest.mark.parametrize("scheduler", list(_NAMES), ids=list(_NAMES.values()))
def test_each_scheduler_dict_builds_a_plan(scheduler):
    raw = scheduler(**_SCHEDULER_PARAMS)
    params = dict(_SCHEDULER_PARAMS)
    if scheduler is lpsd_plan:
        params["bmin"] = 1.0
        params["Lmin"] = 1
    plan = SpectralPlan.from_dict(
        raw,
        fs=params["fs"],
        N=params["N"],
        olap=params["olap"],
        bmin=params["bmin"],
        Lmin=params["Lmin"],
        Jdes=params["Jdes"],
        Kdes=params["Kdes"],
        order=-1,
        scheduler=_NAMES[scheduler],
        window="hann",
        missing="raise",
    )
    view = plan.as_dict()
    assert view["nf"] == len(raw["f"])
    assert np.array_equal(view["f"], np.asarray(raw["f"], dtype=np.float64))
    assert np.array_equal(view["L"], np.asarray(raw["L"], dtype=np.int64))
    facts = plan.report()
    assert facts["nf"] == view["nf"]
    assert plan.digest() == SpectralPlan.from_json(plan.to_json()).digest()
    assert plan.canonical_bytes() == plan.canonical_bytes()

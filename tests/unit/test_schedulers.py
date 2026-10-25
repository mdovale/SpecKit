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
import pytest

from speckit.schedulers import lpsd_plan, ltf_plan, new_ltf_plan, vectorized_ltf_plan

# Define a standard set of parameters for scheduler tests
SCHEDULER_PARAMS = {
    "N": int(1e6),
    "fs": 1.0,
    "olap": 0.75,
    "bmin": 5.0,
    "Lmin": 1000,
    "Jdes": 1000,
    "Kdes": 100,
    "num_patch_pts": 50,
}


@pytest.fixture(
    params=[ltf_plan, lpsd_plan, new_ltf_plan, vectorized_ltf_plan],
    ids=["ltf_plan", "lpsd_plan", "new_ltf_plan", "vectorized_ltf_plan"],
)
def scheduler_plan(request):
    """Fixture to generate a plan from each available scheduler using **kwargs."""
    scheduler_func = request.param
    params = dict(SCHEDULER_PARAMS)
    plan = scheduler_func(**params)
    return plan, params, scheduler_func


def test_plan_output_structure(scheduler_plan):
    """Verifies required keys and basic shape."""
    plan, _, _ = scheduler_plan
    expected_keys = ["f", "r", "b", "L", "K", "navg", "D", "O", "nf"]
    for key in expected_keys:
        assert key in plan, f"Missing key '{key}' in plan output"

    assert isinstance(plan["nf"], int)
    assert plan["nf"] > 0
    assert len(plan["f"]) == plan["nf"]
    # Basic dtype checks
    assert isinstance(plan["D"], (list, tuple))
    assert isinstance(plan["O"], np.ndarray)


def test_plan_dft_constraint(scheduler_plan):
    """Verifies the fundamental DFT constraint: r * L = fs."""
    plan, params, _ = scheduler_plan
    fs = params["fs"]
    assert np.allclose(plan["r"] * plan["L"], fs, rtol=1e-6, atol=1e-6)


def test_plan_frequency_stepping_constraint(scheduler_plan):
    """Verifies the frequency stepping constraint: f[j+1] - f[j] ≈ r[j]."""
    plan, _, _ = scheduler_plan
    # Approximate relationship; average relative error should be small.
    df = np.diff(plan["f"])
    r_for_diff = plan["r"][:-1]
    # Guard against divide-by-zero in pathological cases
    nonzero = r_for_diff != 0
    assert nonzero.all()
    relative_error = np.abs(df[nonzero] - r_for_diff[nonzero]) / r_for_diff[nonzero]
    assert np.mean(relative_error) < 0.01


def test_plan_boundary_and_monotonicity_constraints(scheduler_plan):
    """Checks frequency range, segment lengths, and monotonicity."""
    plan, params, _ = scheduler_plan
    # Frequencies must be positive and strictly increasing
    assert np.all(plan["f"] > 0)
    assert np.all(np.diff(plan["f"]) > 0)
    # Frequency should not exceed Nyquist
    assert plan["f"][-1] <= params["fs"] / 2.0 + 1e-12
    # Segment lengths must be positive integers and not exceed total length
    assert np.issubdtype(plan["L"].dtype, np.integer)
    assert np.all(plan["L"] > 0)
    assert np.all(plan["L"] <= params["N"])


def _nearest_index(f_ref: np.ndarray, f_other: np.ndarray) -> np.ndarray:
    idx = np.clip(np.searchsorted(f_ref, f_other), 0, f_ref.size - 1)
    idx0 = np.clip(idx - 1, 0, f_ref.size - 1)
    closer_left = np.abs(f_ref[idx0] - f_other) < np.abs(f_ref[idx] - f_other)
    return np.where(closer_left, idx0, idx)


def _starts_equal(left, right) -> bool:
    a = np.asarray(left, dtype=np.int64)
    b = np.asarray(right, dtype=np.int64)
    return a.shape == b.shape and np.array_equal(a, b)


def test_vectorized_ltf_step_equals_resolution():
    """The walker claims f[j+1] - f[j] = r[j], tighter than the 0.01 mean test."""
    cases = (
        SCHEDULER_PARAMS,
        {**SCHEDULER_PARAMS, "bmin": 1.0, "Lmin": 1, "Jdes": 200},
        {
            **SCHEDULER_PARAMS,
            "N": 5000,
            "Lmin": 4000,
            "Jdes": 30,
            "Kdes": 2,
            "bmin": 1.0,
            "olap": 0.5,
        },
    )
    for params in cases:
        plan = vectorized_ltf_plan(**params)
        step = np.diff(plan["f"])
        resolution = plan["r"][:-1]
        relative = np.abs(step - resolution) / resolution
        assert np.max(relative) < 1e-9


def test_vectorized_ltf_documented_differences_from_ltf():
    """Record where the default plan leaves ltf_plan. Do not force them equal.

    Counts below were measured on 25 Oct 2026. A later scheduler edit that
    moves them has to update this test and frequency-scheduling.md together.
    """
    fixture = vectorized_ltf_plan(**SCHEDULER_PARAMS)
    iterative = ltf_plan(**SCHEDULER_PARAMS)
    assert fixture["nf"] == 849
    assert iterative["nf"] == 849
    take = _nearest_index(iterative["f"], fixture["f"])
    length_differs = np.asarray(iterative["L"])[take] != np.asarray(fixture["L"])
    assert int(length_differs.sum()) == 424
    first = int(np.flatnonzero(length_differs)[0])
    assert fixture["f"][first] == pytest.approx(6e-6)
    assert int(fixture["L"][first]) == 832707
    assert int(iterative["L"][take[first]]) == 833333
    for j in np.flatnonzero(~length_differs):
        assert int(iterative["K"][take[j]]) == int(fixture["K"][j])
        assert _starts_equal(iterative["D"][take[j]], fixture["D"][j])

    low = {**SCHEDULER_PARAMS, "bmin": 1.0, "Lmin": 1, "Jdes": 200}
    fixture = vectorized_ltf_plan(**low)
    iterative = ltf_plan(**low)
    assert fixture["nf"] == iterative["nf"] == 140
    take = _nearest_index(iterative["f"], fixture["f"])
    length_differs = np.asarray(iterative["L"])[take] != np.asarray(fixture["L"])
    assert int(length_differs.sum()) == 130
    start_differs = 0
    for j in np.flatnonzero(~length_differs):
        if not _starts_equal(iterative["D"][take[j]], fixture["D"][j]):
            start_differs += 1
    assert start_differs == 2

    clamped = {
        **SCHEDULER_PARAMS,
        "N": 5000,
        "Lmin": 4000,
        "Jdes": 30,
        "Kdes": 2,
        "bmin": 1.0,
        "olap": 0.5,
    }
    fixture = vectorized_ltf_plan(**clamped)
    iterative = ltf_plan(**clamped)
    assert fixture["nf"] == iterative["nf"]
    assert np.array_equal(fixture["f"], iterative["f"])
    assert np.array_equal(fixture["L"], iterative["L"])
    assert np.array_equal(fixture["K"], iterative["K"])
    assert int(np.sum(np.asarray(fixture["K"]) == 1)) == 3
    for left, right in zip(iterative["D"], fixture["D"]):
        assert _starts_equal(left, right)

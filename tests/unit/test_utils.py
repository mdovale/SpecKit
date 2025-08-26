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
from speckit import utils
from speckit import schedulers

# Known values from reference implementations or calculations
# PSLL values and their corresponding alpha (beta)
PSLL_ALPHA_PAIRS = [
    (50, 2.163),
    (100, 4.228),
    (200, 8.086),
]

# Alpha values and their recommended overlap
ALPHA_OLAP_PAIRS = [
    (1, 0.303),
    (5, 0.707),
    (10, 0.796),
]


@pytest.mark.parametrize("psll, expected_alpha", PSLL_ALPHA_PAIRS)
def test_kaiser_alpha(psll, expected_alpha):
    """Tests the Kaiser window alpha calculation against known values."""
    assert utils.kaiser_alpha(psll) == pytest.approx(expected_alpha, rel=1e-3)


@pytest.mark.parametrize("alpha, expected_olap", ALPHA_OLAP_PAIRS)
def test_kaiser_rov(alpha, expected_olap):
    """Tests the recommended overlap calculation for Kaiser windows."""
    assert utils.kaiser_rov(alpha) == pytest.approx(expected_olap, rel=1e-3)


@pytest.mark.parametrize(
    "val, expected", [(3.5, 4), (3.49, 3), (3.0, 3), (-2.5, -2), (-2.51, -3)]
)
def test_round_half_up(val, expected):
    """Tests the custom rounding function."""
    assert utils.round_half_up(val) == expected


def test_chunker():
    """Tests the list chunking utility."""
    data = list(range(10))
    assert utils.chunker(data, 3) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert utils.chunker(data, 10) == [list(range(10))]
    assert utils.chunker(data, 11) == [list(range(10))]
    with pytest.raises(ValueError):
        utils.chunker(data, 0)


def test_find_jdes_binary_search():
    """
    Tests the binary search utility for finding a Jdes that produces a target
    number of frequencies. This is a key part of the `force_target_nf` feature.
    """
    # Mock analysis setup
    N, fs, olap, bmin, Lmin, Kdes = int(1e6), 1.0, 0.5, 1.0, 100, 100
    target_nf = 1000

    # Common kwargs for schedulers (**args style)
    kwargs = dict(N=N, fs=fs, olap=olap, bmin=bmin, Lmin=Lmin, Kdes=Kdes)

    found_jdes = utils.find_Jdes_binary_search(schedulers.ltf_plan, target_nf, **kwargs)
    assert found_jdes is not None

    # Verify that the found Jdes actually produces the target nf
    final_plan = schedulers.ltf_plan(**kwargs, Jdes=int(found_jdes))
    assert final_plan["nf"] == target_nf

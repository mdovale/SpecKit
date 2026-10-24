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
import numpy as np
import pytest

from speckit.core import _overlap_dof
from speckit.utils import kaiser_alpha, kaiser_rov


def _even_hop_dof(w, hop, K):
    """Percival & Walden (1993) eq. 292b for evenly spaced starts."""
    w = w / np.sqrt(np.sum(w**2))
    sm = 0.0
    for m in range(1, K):
        d = m * hop
        if d >= w.size:
            break
        sm += (1 - m / K) * np.dot(w[: w.size - d], w[d:]) ** 2
    return 2 * K / (1 + 2 * sm)


@pytest.mark.parametrize("hop", [512, 513, 600])
def test_no_overlap_gives_two_per_segment(hop):
    L, K = 512, 50
    starts = np.arange(K) * hop
    for w in (np.hanning(L), np.ones(L), np.kaiser(L, 10.0)):
        assert _overlap_dof(w, starts) == 2.0 * K


def test_single_segment_and_identical_segments():
    w = np.hanning(256)
    assert _overlap_dof(w, np.array([10])) == 2.0
    # K copies of one segment carry the information of one segment.
    assert _overlap_dof(w, np.full(8, 10)) == pytest.approx(2.0, rel=1e-14)


def test_hann_half_overlap_matches_percival_walden():
    """
    Percival & Walden (1993), p. 294: Hann (periodic) at 50% overlap gives
    nu ~ 36 K / 19 for large K, since rho(L/2) = 1/6. The finite-K term
    (1 - 1/K) moves the ratio by ~2.6e-5 at K = 2000, hence rtol = 1e-4.
    """
    L, K = 4096, 2000
    w = np.hanning(L + 1)[:-1]
    starts = np.arange(K) * (L // 2)
    assert _overlap_dof(w, starts) / (2 * K) == pytest.approx(18 / 19, rel=1e-4)


def test_pairwise_form_equals_even_hop_form_for_default_kaiser():
    """The actual-start formula reduces to eq. 292b when starts are even."""
    alpha = kaiser_alpha(200)
    olap = kaiser_rov(alpha)
    L, K = 1000, 300
    hop = round(L * (1 - olap))
    w = np.kaiser(L + 1, alpha * np.pi)[:-1]
    starts = np.arange(K) * hop
    nu = _overlap_dof(w, starts)
    assert nu == pytest.approx(_even_hop_dof(w, hop, K), rel=1e-12)
    assert 2.0 < nu < 2.0 * K


def test_order_of_starts_does_not_matter():
    rng = np.random.default_rng(0)
    w = np.hanning(300)
    starts = np.sort(rng.choice(5000, size=60, replace=False))
    assert _overlap_dof(w, starts[::-1]) == pytest.approx(
        _overlap_dof(w, starts), rel=1e-14
    )

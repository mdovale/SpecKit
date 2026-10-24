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

import speckit.flattop as ft

# Float64 bytes of each window at M=8, from the published cosine-sum coefficients.
_M8 = {
    "SFT3F": "000000000000000097120d49659ab6bfd0e47e87a2409f3f7a995dbd51cde33f000000000000f03f7b995dbd51cde33f00e57e87a2409f3f97120d49659ab6bf",
    "SFT4F": "000000000000703cce180297906899bf07e275fd82ddb0bf0ba9124a285fdd3f000000000000f03f0fa9124a285fdd3f03e275fd82ddb0bfd8180297906899bf",
    "SFT5F": "00000000000070bc801bbd12d51a76bff988981249f4b2bf0fc151edcf3cd53f000000000000f03f16c151edcf3cd53ffa88981249f4b2bf001cbd12d51a76bf",
    "SFT3M": "784bab21718fa5bfd7a67cc6dc09b6bfd8114e0b5ef4b53f4838d15441d3e43f1c774a07ebffef3f4b38d15441d3e43fe4114e0b5ef4b53fd6a67cc6dc09b6bf",
    "SFT4M": "2800c5c8923976bf445d895fbdd3abbfcc4082e2c7988bbfd1c7a5239f38e13f000000000000f03fd2c7a5239f38e13f9d4082e2c7988bbf465d895fbdd3abbf",
    "SFT5M": "0000bd9af2d79abec706b84e677596bf50863eb37ff9afbfd4e6bdc08713db3f2b6b4029ffffef3fdbe6bdc08713db3f4a863eb37ff9afbfe306b84e677596bf",
    "FTNI": "409660ccf165a5bf1bfa3f3e1157b6bf7c79c8940f41b53fcc836d98d5c7e43f0b7a6f0c0100f03fce836d98d5c7e43f8879c8940f41b53f16fa3f3e1157b6bf",
    "FTHP": "20bbe5efc9ff90bf333e39817e83ccbf7612f515b344b4bfe39313e837c801404f8e9f6225b31040e59313e837c801405712f515b344b4bf363e39817e83ccbf",
    "FTSR": "000000000000993c0e4640bfc04cbebff953e3a59bc4d0bf896635e5b57f004024068195438b12408d6635e5b57f0040f453e3a59bc4d0bf1e4640bfc04cbebf",
    "Matlab": "b0ff624610983bbf0dad585d62849bbfd36d5f937706acbf77b1a2b4b66cdc3f8f28ce000000f03f7fb1a2b4b66cdc3fcc6d5f937706acbf19ad585d62849bbf",
    "HFT70": "a8c5c4e6e3da90bff1c978748b37ccbf2642959a3dd0b2bf9f8c47b778c30140d76ebbd05ca71040a18c47b778c301400742959a3dd0b2bff3c978748b37ccbf",
    "HFT95": "0031bd5873db59bf0c30b4e926efbebf075152bbba3ed1bffbaa6253d767004078cc8af216b91240ffaa6253d7670040025152bbba3ed1bf2c30b4e926efbebf",
    "HFT90D": "0000000000009ebcf08e6a59ebe2babf78cdab3aab05d3bf4470f2df90260040e25d2ee23b1113404770f2df9026004073cdab3aab05d3bf008f6a59ebe2babf",
    "HFT116D": "0000000000a0a23ce61adbe41aacaabf465e20521dbcd6bf0232e1fe14e7fc3f32df6696a9ce14400932e1fe14e7fc3f445e20521dbcd6bf181bdbe41aacaabf",
    "HFT144D": "0000000000b8993c21c250b3bc7798bfa410f1a4bfb6d6bfd6510169d327f93fac1a84b9dd731640e0510169d327f93fa310f1a4bfb6d6bfedc250b3bc7798bf",
    "HFT169D": "002035e58e79553e1924000cbfb785bffa69edc737a5d4bf19c509b1e864f53fe7312d5eeaf7174024c509b1e864f53ffb69edc737a5d4bf7625000cbfb785bf",
    "HFT196D": "0000000080c98dbce6dd9a86ffe272bfe12aca30e4b4d1bf7342bde1f3b6f13fefbb9b07ad6419407e42bde1f3b6f13fdf2aca30e4b4d1bfefe09a86ffe272bf",
    "HFT223D": "000000000255a4bc6f2169a1dfa45ebf4351af8926f4ccbf819f10651321ec3f11ec0f99f3cb1a40979f10651321ec3f3c51af8926f4ccbfcd3069a1dfa45ebf",
    "HFT248D": "00000000a218833cfa4bd8cb1e3d48bf2fd9274d97f7c6bffe6409d86c53e53f86f997d43f211c40196509d86c53e53f21d9274d97f7c6bf8a72d8cb1e3d48bf",
}


def test_flat_top_samples_match_published_cosine_sums():
    for name, hx in _M8.items():
        w = np.asarray(getattr(ft, name)(8), dtype=np.float64)
        assert w.tobytes().hex() == hx


def test_win_dict_and_overlap_table():
    assert ft.win_dict["FTSRS"] is ft.FTSR
    assert ft.win_dict["SFT3F"] is ft.SFT3F
    assert "Matlab" not in ft.win_dict
    assert ft.olap_dict["SFT3F"] == 0.667
    assert ft.olap_dict["HFT248D"] == 0.841
    assert ft.olap_dict[13] == ft.olap_dict["SFT3F"]
    for func in ft.win_dict.values():
        assert len(func(16)) == 16


def test_flattop_module_is_project_bsd():
    """The flat-top module is SpecKit code, not a reserved third-party file."""
    from pathlib import Path

    text = Path(ft.__file__).read_text()
    assert text.startswith("# BSD 3-Clause License")
    assert "All rights reserved" not in text
    assert "Joe Henning" not in text
    assert "Copyright (c) 2025, Miguel Dovale" in text

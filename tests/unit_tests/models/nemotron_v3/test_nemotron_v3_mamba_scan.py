# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``BackendConfig.mamba_norm_backward_grid_scale``: which scan the mixer runs, on a CPU stand-in.

The comparisons with mamba_ssm's own scan on the real kernels are in
``tests/functional_tests/kernel_parity/test_nemotron_v3_mamba_scan.py``.
"""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3 import layers as nemotron_layers
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer
from nemo_automodel.components.models.nemotron_v3.mamba_scan import split_conv1d_scan
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mamba_out_projection import (
    _install_cpu_scan,
    _pre_projection,
)
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config


def _backend(**mamba: int | float | bool | str) -> BackendConfig:
    return BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", **mamba)


@pytest.mark.cpu_only
def test_default_keeps_the_library_scan():
    backend = _backend()
    assert backend.mamba_norm_backward_grid_scale == 1
    for mixer in (
        NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0),
        NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=backend),
    ):
        assert mixer.norm_backward_grid_scale == 1


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [2.5, 8.0, True, "8"])
def test_grid_scale_that_is_not_an_integer_is_rejected(scale):
    """YAML can deliver a float, a boolean or a string; the value multiplies a kernel's launch grid."""
    with pytest.raises(ValueError, match="mamba_norm_backward_grid_scale must be an integer"):
        _backend(mamba_norm_backward_grid_scale=scale)


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [0, -3])
def test_grid_scale_below_one_is_rejected(scale):
    with pytest.raises(ValueError, match="mamba_norm_backward_grid_scale"):
        _backend(mamba_norm_backward_grid_scale=scale)
    with pytest.raises(ValueError, match="norm_backward_grid_scale"):
        split_conv1d_scan(
            torch.zeros(1, 1, 1),
            torch.zeros(1, 1),
            None,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            chunk_size=1,
            seq_idx=None,
            dt_limit=(0.0, float("inf")),
            activation="silu",
            rmsnorm_weight=torch.zeros(1),
            rmsnorm_eps=1e-5,
            outproj_weight=None,
            outproj_bias=None,
            headdim=1,
            ngroups=1,
            norm_backward_grid_scale=scale,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("scale", [1, 8])
@pytest.mark.parametrize("fused", [True, False])
def test_mixer_selects_the_scan_by_grid_scale(monkeypatch, scale, fused):
    torch.manual_seed(3)
    backend = _backend(mamba_norm_backward_grid_scale=scale, mamba_fused_out_proj=fused)
    mixer = NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=backend).float()
    library_calls = _install_cpu_scan(monkeypatch, mixer)
    own_calls = []

    def own_scan(projected, conv_weight, conv_bias, dt_bias, a, d, **kwargs):
        own_calls.append({"conv_weight": conv_weight, "conv_bias": conv_bias, "dt_bias": dt_bias, "D": d, **kwargs})
        result = _pre_projection(projected, mixer.intermediate_size, kwargs["rmsnorm_weight"], kwargs["rmsnorm_eps"])
        if kwargs["outproj_weight"] is not None:
            result = result @ kwargs["outproj_weight"].T
        return result

    monkeypatch.setattr(nemotron_layers, "split_conv1d_scan", own_scan)
    x = torch.randn(2, 7, mixer.hidden_size)
    actual = mixer(x)

    # Independent expectation: the stand-in scan without a projection, then plain matrix algebra.
    expected = (
        _pre_projection(mixer.in_proj(x), mixer.intermediate_size, mixer.norm.weight, mixer.norm.variance_epsilon)
        @ mixer.out_proj.weight.T
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    if scale == 1:
        assert len(library_calls) == 1 and own_calls == []
        return
    assert library_calls == [] and len(own_calls) == 1
    call = own_calls[0]
    assert call["norm_backward_grid_scale"] == scale
    assert call["rmsnorm_weight"] is mixer.norm.weight
    assert call["rmsnorm_eps"] == mixer.norm.variance_epsilon
    assert (call["outproj_weight"] is mixer.out_proj.weight) == fused
    assert call["outproj_bias"] is None
    assert call["conv_weight"].shape == (mixer.conv_dim, mixer.conv_kernel_size)
    assert call["conv_bias"] is mixer.conv1d.bias
    assert call["chunk_size"] == mixer.chunk_size
    assert call["headdim"] == mixer.head_dim and call["ngroups"] == mixer.n_groups
    assert call["dt_limit"] == (0.0, float("inf"))
    assert call["activation"] == "silu" and call["seq_idx"] is None

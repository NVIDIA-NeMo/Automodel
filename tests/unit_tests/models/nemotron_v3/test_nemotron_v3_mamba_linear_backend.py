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

"""``BackendConfig.mamba_linear``: which module type the Mamba projections use.

The comparison of Transformer Engine projections with the projection applied inside the scan kernel, on the
real kernel, is in ``tests/functional_tests/kernel_parity/test_nemotron_v3_mamba_projections.py``.
"""

import math

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block, NemotronV3Mamba2Mixer
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config


def _backend(mamba_linear: str) -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        mamba_linear=mamba_linear,
    )


@pytest.mark.cpu_only
def test_default_backend_keeps_torch_projections():
    assert BackendConfig(attn="sdpa", linear="torch", experts="torch", dispatcher="torch").mamba_linear == "torch"
    for mixer in (
        NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0),
        NemotronV3Mamba2Mixer(MockNemotronV3Config(), layer_idx=0, backend=_backend("torch")),
    ):
        assert type(mixer.in_proj) is nn.Linear
        assert type(mixer.out_proj) is nn.Linear


@pytest.mark.cpu_only
@pytest.mark.parametrize("with_backend", [False, True])
def test_torch_projection_construction_draws_the_same_initial_values(with_backend):
    """The default path must not change the initialization stream of existing recipes.

    The expectation is built here from plain PyTorch modules, in the order the mixer has always constructed
    them: input projection, depthwise convolution, output projection. The SSM parameters and the norm weight
    draw nothing.
    """
    config = MockNemotronV3Config()
    intermediate = config.mamba_num_heads * config.mamba_head_dim
    conv_dim = intermediate + 2 * config.n_groups * config.ssm_state_size
    torch.manual_seed(7)
    in_proj = nn.Linear(config.hidden_size, intermediate + conv_dim + config.mamba_num_heads, bias=config.use_bias)
    conv1d = nn.Conv1d(
        conv_dim,
        conv_dim,
        kernel_size=config.conv_kernel,
        groups=conv_dim,
        padding=config.conv_kernel - 1,
        bias=config.use_conv_bias,
    )
    out_proj = nn.Linear(intermediate, config.hidden_size, bias=config.use_bias)
    expected = {
        **{f"in_proj.{name}": value for name, value in in_proj.named_parameters()},
        **{f"conv1d.{name}": value for name, value in conv1d.named_parameters()},
        **{f"out_proj.{name}": value for name, value in out_proj.named_parameters()},
    }

    torch.manual_seed(7)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_backend("torch") if with_backend else None)
    actual = dict(mixer.named_parameters())
    assert set(expected) == {name for name in actual if not name.startswith(("_fp32_params.", "norm."))}
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        torch.testing.assert_close(actual[name], value, rtol=0, atol=0, msg=name)
    # Nothing else was drawn: the next random number is the one that follows the three modules.
    after_mixer = torch.rand(())
    torch.manual_seed(7)
    nn.Linear(config.hidden_size, intermediate + conv_dim + config.mamba_num_heads, bias=config.use_bias)
    nn.Conv1d(conv_dim, conv_dim, kernel_size=config.conv_kernel, groups=conv_dim, bias=config.use_conv_bias)
    nn.Linear(intermediate, config.hidden_size, bias=config.use_bias)
    assert torch.equal(after_mixer, torch.rand(()))


@pytest.mark.cpu_only
def test_block_forwards_the_backend_to_the_mixer(monkeypatch):
    seen = []
    original = NemotronV3Mamba2Mixer._build_projection

    def record(mamba_linear, in_features, out_features, config):
        seen.append(mamba_linear)
        return original("torch", in_features, out_features, config)

    monkeypatch.setattr(NemotronV3Mamba2Mixer, "_build_projection", staticmethod(record))
    backend = _backend("torch")
    backend.mamba_linear = "te"  # bypass the constructor check: only the plumbing is under test here
    NemotronV3Block(MockNemotronV3Config(layers_block_type=["mamba", "moe"]), layer_idx=0, backend=backend)
    assert seen == ["te", "te"]


@pytest.mark.cpu_only
def test_unknown_mamba_linear_is_rejected():
    with pytest.raises(ValueError, match="mamba_linear"):
        _backend("quack")


@pytest.mark.parametrize("mamba_linear", ["torch", pytest.param("te", marks=pytest.mark.run_only_on("GPU"))])
def test_init_weights_restores_every_parameter_of_a_zeroed_mixer(mamba_linear):
    """Parameters that hold no usable values get their construction-time values, with either projection type.

    A model on more than one rank is built on the meta device and materialized before ``init_weights``
    runs, so the values its parameters had at construction are gone; zeros stand in for that here. A mixer
    left like that returns zeros and passes no gradient to its parameters.
    """
    if mamba_linear == "te":
        pytest.importorskip("transformer_engine")
    config = MockNemotronV3Config(hidden_size=256, mamba_num_heads=8, mamba_head_dim=64)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_backend(mamba_linear))
    assert (type(mixer.in_proj) is nn.Linear) == (mamba_linear == "torch")
    assert (type(mixer.out_proj) is nn.Linear) == (mamba_linear == "torch")
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.zero_()
    torch.manual_seed(0)
    num_hidden_layers = 4
    mixer.init_weights(num_hidden_layers=num_hidden_layers, buffer_device=torch.device("cpu"))
    # Uniform in +-1/sqrt(fan_in): standard deviation bound / sqrt(3). The output projection is rescaled.
    # Transformer Engine projections hold bfloat16 values, which round up to 1% past the bound.
    for weight, scale in ((mixer.in_proj.weight, 1.0), (mixer.out_proj.weight, math.sqrt(num_hidden_layers))):
        bound = 1.0 / math.sqrt(weight.shape[1]) / scale
        assert float(weight.float().abs().max()) <= bound * 1.01
        assert float(weight.float().std()) == pytest.approx(bound / math.sqrt(3), rel=0.03)
    # The depthwise convolution has a fan-in of its kernel size.
    conv_bound = 1.0 / math.sqrt(config.conv_kernel)
    assert 0.8 * conv_bound < float(mixer.conv1d.weight.abs().max()) <= conv_bound
    assert 0.8 * conv_bound < float(mixer.conv1d.bias.abs().max()) <= conv_bound
    torch.testing.assert_close(mixer.norm.weight, torch.ones_like(mixer.norm.weight))
    torch.testing.assert_close(mixer.D, torch.ones_like(mixer.D))
    torch.testing.assert_close(mixer.A_log, torch.log(torch.arange(1, config.mamba_num_heads + 1.0)))
    assert float(mixer.dt_bias.abs().min()) > 0


@pytest.mark.run_only_on("GPU")
def test_te_projections_start_from_the_torch_linear_distribution():
    pytest.importorskip("transformer_engine")
    config = MockNemotronV3Config(hidden_size=256, mamba_num_heads=8, mamba_head_dim=64)
    mixer = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_backend("te"))
    mixer.init_weights(num_hidden_layers=4, rescale_prenorm_residual=True, buffer_device=torch.device("cpu"))
    # torch.nn.Linear draws U(-1/sqrt(fan_in), 1/sqrt(fan_in)); the output projection is then divided by sqrt(layers).
    in_bound = 1.0 / math.sqrt(config.hidden_size)
    out_bound = 1.0 / math.sqrt(mixer.intermediate_size) / math.sqrt(4)
    for weight, bound in ((mixer.in_proj.weight.float(), in_bound), (mixer.out_proj.weight.float(), out_bound)):
        assert weight.abs().max().item() <= bound * 1.01
        assert weight.std().item() == pytest.approx(bound / math.sqrt(3), rel=0.05)

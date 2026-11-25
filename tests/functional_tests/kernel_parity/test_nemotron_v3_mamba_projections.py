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

"""Mamba output projection of Nemotron V3 applied after the scan kernel against the one applied inside it.

Runs the real scan kernel on CUDA twice on identical weights and inputs: with the output projection inside
the kernel (the default) and with both projections as Transformer Engine linears (``mamba_linear="te"``),
whose output projection runs as a module after the kernel.
The two must agree in output and in every gradient. The first run of a dtype compiles the Triton kernels of
the scan, about a minute each, which is why this comparison is not a unit test.
"""

import math

import pytest
import torch

from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Mamba2Mixer
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mamba_linear_backend import _backend
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="The Mamba scan kernels run on CUDA")


def _weighted_sum(output: torch.Tensor) -> torch.Tensor:
    weights = torch.cos(torch.arange(output.numel(), device=output.device, dtype=torch.float32))
    return (output.float() * weights.reshape(output.shape)).sum()


@pytest.mark.parametrize(
    ("dtype", "rtol"),
    [(torch.float32, 2e-4), (torch.bfloat16, 3e-2)],
    ids=["fp32", "bf16"],
)
def test_projections_after_the_kernel_match_the_fused_projection_on_the_real_kernel(dtype, rtol):
    pytest.importorskip("transformer_engine")
    pytest.importorskip("mamba_ssm")
    device = torch.device("cuda")
    config = MockNemotronV3Config(torch_dtype="float32" if dtype is torch.float32 else "bfloat16")

    torch.manual_seed(20261123)
    fused = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_backend("torch"))
    external = NemotronV3Mamba2Mixer(config, layer_idx=0, backend=_backend("te"))
    assert type(external.in_proj).__module__.startswith("transformer_engine")
    assert type(external.out_proj).__module__.startswith("transformer_engine")
    # Identical parameter names and shapes: checkpoints are interchangeable between the two settings.
    assert [(name, parameter.shape) for name, parameter in fused.named_parameters()] == [
        (name, parameter.shape) for name, parameter in external.named_parameters()
    ]

    with torch.no_grad():
        fused.norm.weight.uniform_(0.5, 1.5)
        fused.conv1d.bias.normal_(std=0.1)
    incompatible = external.load_state_dict(fused.state_dict(), strict=False)
    # Transformer Engine modules carry a non-parameter ``_extra_state`` entry; nothing else may differ.
    assert incompatible.unexpected_keys == []
    assert all(key.endswith("._extra_state") for key in incompatible.missing_keys), incompatible.missing_keys
    for mixer in (fused, external):
        mixer.to(device=device, dtype=dtype)
        # The SSM parameters are fp32 in every recipe.
        mixer._fp32_params.float()

    # 160 tokens = 2.5 chunks of 64, so the scan crosses chunk boundaries and pads the last chunk.
    inputs = torch.randn(2, 160, config.hidden_size, device=device, dtype=dtype)
    results = []
    for mixer in (fused, external):
        hidden = inputs.clone().requires_grad_()
        output = mixer(hidden)
        assert output.shape == inputs.shape and output.dtype == dtype
        _weighted_sum(output).backward()
        gradients = {name: parameter.grad for name, parameter in mixer.named_parameters()}
        assert all(gradient is not None for gradient in gradients.values()), gradients.keys()
        results.append((output.detach(), hidden.grad, gradients))

    (fused_out, fused_dx, fused_grads), (external_out, external_dx, external_grads) = results

    def assert_matches(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
        scale = expected.float().abs().max().item()
        assert scale > 0 and math.isfinite(scale), name
        torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=rtol * scale, msg=name)

    assert_matches(external_out, fused_out, "output")
    assert_matches(external_dx, fused_dx, "input gradient")
    for name, expected in fused_grads.items():
        assert_matches(external_grads[name], expected, name)

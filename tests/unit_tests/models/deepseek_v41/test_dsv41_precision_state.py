# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Block activation-checkpoint ownership and released router precision contracts."""

import copy
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41AttentionState
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config


def _build_model(config=None):
    model = DeepseekV41ForCausalLM(config or _tiny_config(), backend=_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


def test_default_bf16_router_matches_fp32_reference():
    config = _tiny_config()
    config.text_config.n_routed_experts = 384
    config.text_config.num_experts_per_tok = 6
    model = _build_model(config)
    cast_model_to_dtype(model, torch.bfloat16)
    gate = model.model.layers["2"].ffn.gate
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2048, config.text_config.hidden_size, generator=generator).bfloat16()
    with torch.no_grad():
        gate.e_score_correction_bias.copy_(torch.linspace(-0.04, 0.04, 384))
    scores = F.softplus(F.linear(x.float(), gate.weight.float())).sqrt()
    indices = (scores + gate.e_score_correction_bias).topk(6, dim=-1).indices
    expected = scores.gather(1, indices)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * config.text_config.routed_scaling_factor
    weights, actual, _ = gate(x, torch.ones(2048, dtype=torch.bool), None)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(actual, indices, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected, atol=0, rtol=0)
    assert model.backend.gate_precision is None


@pytest.mark.parametrize("checkpoint", [False, True])
def test_copied_layer_inputs_preserve_shared_state_and_gradients(checkpoint):
    torch.manual_seed(91)
    reference = _build_model()
    model = copy.deepcopy(reference)
    parameters = dict(model.named_parameters())
    snapshots = []

    def copy_state(module, args, kwargs):
        state = args[2]
        snapshots.append((state, vars(state).copy()))
        return (*args[:2], replace(state)), kwargs

    for layer in model.model.layers.values():
        layer.register_forward_pre_hook(copy_state, with_kwargs=True)
    if checkpoint:
        apply_ac(model)
    tokens = torch.tensor([[5, 6, 7, 8, 9, 10, 11, 12]])
    expected = reference(tokens).logits
    actual = model(tokens).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for state, fields in snapshots:
        assert all(getattr(state, name) is value for name, value in fields.items())
    for name, parameter in reference.named_parameters():
        grad = parameters[name].grad
        if parameter.grad is None:
            assert grad is None
        else:
            torch.testing.assert_close(grad, parameter.grad, atol=0, rtol=0)
    assert model.model.layers["2"].attn.compressor.wkv.weight.grad is not None


def test_block_rejects_rounded_carried_coefficients():
    model = _build_model()
    block = model.model.layers["0"]
    streams = torch.randn(1, 4, model.config.text_config.hc_mult, model.config.text_config.hidden_size)
    with pytest.raises(TypeError, match="FP32 carried coefficients"):
        block(
            streams,
            torch.zeros(1, 4, model.config.text_config.hc_mult, dtype=torch.bfloat16),
            DeepseekV41AttentionState(),
            position_ids=torch.arange(4)[None],
        )

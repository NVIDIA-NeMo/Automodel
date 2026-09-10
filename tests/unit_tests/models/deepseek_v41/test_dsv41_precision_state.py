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

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4 import fsdp as dsv4_fsdp
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.conftest import build_tiny_model, tiny_config


def test_default_bf16_router_matches_fp32_reference():
    model = build_tiny_model(tiny_config(n_routed_experts=384, num_experts_per_tok=6))
    cast_model_to_dtype(model, torch.bfloat16)
    gate = model.model.layers["2"].mlp.gate
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(2048, model.config.hidden_size, generator=generator).bfloat16()
    with torch.no_grad():
        gate.e_score_correction_bias.copy_(torch.linspace(-0.04, 0.04, 384))
    scores = F.softplus(F.linear(x.float(), gate.weight.float())).sqrt()
    indices = (scores + gate.e_score_correction_bias).topk(6, dim=-1).indices
    expected = scores.gather(1, indices)
    expected = expected / (expected.sum(-1, keepdim=True) + 1e-20) * model.config.routed_scaling_factor
    weights, actual, _ = gate(x, torch.ones(2048, dtype=torch.bool), None)
    assert weights.dtype == torch.float32
    torch.testing.assert_close(actual, indices, atol=0, rtol=0)
    torch.testing.assert_close(weights, expected, atol=0, rtol=0)
    assert model.backend.gate_precision is None


@pytest.mark.parametrize("checkpoint", [False, True])
def test_copied_layer_inputs_preserve_shared_state_and_gradients(checkpoint):
    reference = build_tiny_model(tiny_config(engram_enabled=False))
    model = build_tiny_model(tiny_config(engram_enabled=False))
    parameters = dict(model.named_parameters())
    snapshots = []

    def copy_state(module, args, kwargs):
        """Copy shared tensor metadata while retaining tensor aliases.

        Args:
            module: Transformer block receiving the input.
            args: Positional tensors [batch, sequence, copies, hidden] and
                [batch, sequence, copies].
            kwargs: Attention arguments including the documented shared state.

        Returns:
            Unchanged positional tensors and keyword arguments with copied state.
        """
        state = kwargs["state"]
        snapshots.append((state, vars(state).copy()))
        return args, {**kwargs, "state": replace(state)}

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


@pytest.mark.parametrize("checkpoint", [False, True])
def test_fsdp_block_preserves_mixed_activation_precision(monkeypatch, checkpoint):
    model = build_tiny_model(tiny_config(engram_enabled=False))
    cast_model_to_dtype(model, torch.bfloat16)
    if checkpoint:
        apply_ac(model)
    block = model.model.layers["2"]
    calls = []

    def capture(module, **kwargs):
        calls.append((module, kwargs["mp_policy"]))
        return module

    monkeypatch.setattr(dsv4_fsdp, "fully_shard", capture)
    original = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.bfloat16)
    dsv4_fsdp.fully_shard_deepseek_v4(block, mesh=None, mp_policy=original)
    policy = next(policy for module, policy in calls if module is block)
    assert policy.param_dtype == torch.bfloat16
    assert policy.reduce_dtype == torch.float32
    assert policy.output_dtype is None
    assert policy.cast_forward_inputs is False
    assert original.output_dtype == torch.bfloat16
    assert original.cast_forward_inputs is True

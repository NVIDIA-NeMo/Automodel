# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise cancellation across routed/shared experts at the public MoE boundary."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


def _config(combine_in_fp32: bool = True) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=2.0,
        dim=1,
        inter_dim=1,
        moe_inter_dim=1,
        norm_topk_prob=True,
        expert_activation="relu2",
        shared_expert_activation="relu2",
        router_weights_fp32=True,
        combine_in_fp32=combine_in_fp32,
    )


def _cancelling_moe(combine_in_fp32: bool) -> MoE:
    """Experts produce 1, 1/256 and -1: early BF16 rounding loses 1/256."""
    moe = MoE(_config(combine_in_fp32), BackendConfig(linear="torch", experts="torch", dispatcher="torch"))
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.experts.gate_and_up_projs.fill_(1)
        moe.experts.down_projs.copy_(torch.tensor([1.0, 1 / 256]).reshape(2, 1, 1))
        moe.shared_experts.up_proj.weight.fill_(1)
        moe.shared_experts.down_proj.weight.fill_(-1)
    # The arithmetic is tested on CPU; avoid compiling the small activation.
    activation = moe.experts.expert_activation_grouped
    moe.experts.expert_activation_grouped = getattr(activation, "_torchdynamo_orig_callable", activation)
    return moe


@pytest.mark.parametrize("combine_in_fp32,expected", [(False, 0.0), (True, 1 / 256)])
def test_shared_cancellation_preserves_small_expert_and_gradients(combine_in_fp32: bool, expected: float) -> None:
    moe = _cancelling_moe(combine_in_fp32)
    hidden = torch.ones(1, 1, 1, dtype=torch.bfloat16, requires_grad=True)
    result = moe(hidden)
    assert result.dtype == hidden.dtype
    torch.testing.assert_close(result, torch.full_like(hidden, expected), rtol=0, atol=0)
    result.sum().backward()
    for weight in (moe.experts.down_projs, moe.shared_experts.down_proj.weight):
        torch.testing.assert_close(weight.grad, torch.ones_like(weight), rtol=0, atol=0)
    assert moe.gate.weight.grad is not None and torch.isfinite(moe.gate.weight.grad).all()


def test_routed_boundary_retains_fp32_and_respects_padding() -> None:
    moe = _cancelling_moe(True)
    hidden = torch.ones(2, 1, dtype=torch.bfloat16)
    weights = torch.ones(2, 2, dtype=torch.float32, requires_grad=True)
    indices = torch.tensor([[0, 1], [0, 1]])
    output = moe.experts(hidden, torch.tensor([True, False]), weights, indices)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, torch.tensor([[257 / 256], [0.0]]), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(weights.grad[1], torch.zeros(2), rtol=0, atol=0)


@pytest.mark.parametrize("dispatcher", ["deepep", "uccl_ep", "mok"])
def test_unimplemented_dispatcher_rejects_fp32_contract(dispatcher: str) -> None:
    with pytest.raises(ValueError, match="combine_in_fp32 requires dispatcher"):
        MoE(_config(), BackendConfig(dispatcher=dispatcher))


def test_latent_projection_rejects_fp32_contract() -> None:
    with pytest.raises(ValueError, match="latent projection"):
        MoE(replace(_config(), moe_latent_size=2), BackendConfig(dispatcher="torch"))


def test_output_cast_cannot_silently_drop_routed_precision() -> None:
    moe = _cancelling_moe(True)
    # Reproduce an FSDP output cast; even a FP32 shared output cannot restore
    # routed information already lost at that boundary.
    moe.experts.register_forward_hook(lambda _module, _inputs, output: output.bfloat16())
    moe.shared_experts.register_forward_hook(lambda _module, _inputs, output: output.float())
    with pytest.raises(TypeError, match="FSDP output_dtype=None"):
        moe(torch.ones(1, 1, 1, dtype=torch.bfloat16))

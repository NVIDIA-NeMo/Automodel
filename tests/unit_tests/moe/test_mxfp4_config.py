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

"""CPU checks for expert storage selection, checkpoint loading, and routing contracts."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components._peft.lora_experts_mxfp4 import (
    GroupedExpertsDeepEPLoRAMXFP4,
    GroupedExpertsLoRAMXFP4,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsDeepEPMXFP4, GroupedExpertsMXFP4
from nemo_automodel.components.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4


@pytest.fixture
def moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=64,
        moe_inter_dim=64,
        norm_topk_prob=False,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("passthrough", (False, True))
@pytest.mark.parametrize(
    "cls",
    (
        GroupedExpertsMXFP4,
        GroupedExpertsLoRAMXFP4,
        GroupedExpertsDeepEPMXFP4,
        GroupedExpertsDeepEPLoRAMXFP4,
    ),
)
def test_rejects_router_weight_after_down(
    cls: type[torch.nn.Module], passthrough: bool, moe_config: MoEConfig
) -> None:
    config = replace(moe_config, apply_router_weight_after_down=True)
    base = GroupedExpertsDeepEP if issubclass(cls, GroupedExpertsDeepEP) else GroupedExperts
    with torch.device("meta"):
        orig = base(config, BackendConfig(experts="torch_mm"))
        with pytest.raises(NotImplementedError, match="apply_router_weight_after_down=True"):
            cls(orig, passthrough=passthrough)


@pytest.mark.parametrize("expert_cls", (GroupedExperts, GroupedExpertsDeepEP))
def test_patch_mxfp4_meta_loads_packed_checkpoint(expert_cls: type[torch.nn.Module], moe_config: MoEConfig) -> None:
    with torch.device("meta"):
        original = expert_cls(moe_config, BackendConfig(experts="torch_mm"))
        patched = patch_moe_module(original, dim=4, expert_weight_format="mxfp4")

    assert "gate_and_up_projs" not in patched.state_dict()
    assert "down_projs" not in patched.state_dict()
    assert all(param.is_meta for param in patched.parameters())
    checkpoint = {}
    for name, shape in (
        ("gate_and_up_projs", (4, 128, 64)),
        ("down_projs", (4, 64, 64)),
    ):
        packed, scales = quantize_mxfp4(torch.full(shape, 0.5))
        checkpoint[name + "_packed"] = packed
        checkpoint[name + "_scales"] = scales
    patched.to_empty(device="cpu")
    patched.init_lora_weights("xavier")
    result = patched.load_state_dict(checkpoint, strict=False)
    adapters = {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}
    assert set(result.missing_keys) == adapters
    assert result.unexpected_keys == []
    assert {name for name, param in patched.named_parameters() if param.requires_grad} == adapters
    for name, expected in checkpoint.items():
        actual = getattr(patched, name)
        assert not actual.is_meta
        assert not actual.requires_grad
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.parametrize("expert_cls", (GroupedExperts, GroupedExpertsDeepEP))
def test_patch_mxfp4_preserves_loaded_weights(expert_cls: type[torch.nn.Module], moe_config: MoEConfig) -> None:
    original = expert_cls(moe_config, BackendConfig(experts="torch_mm"))
    with torch.no_grad():
        original.gate_and_up_projs.fill_(0.5)
        original.down_projs.fill_(-0.25)
    patched = patch_moe_module(original, dim=4, expert_weight_format="mxfp4")
    for name in ("gate_and_up_projs", "down_projs"):
        packed, scales = getattr(patched, name + "_packed"), getattr(patched, name + "_scales")
        assert not packed.is_meta
        decoded = dequantize_mxfp4(packed, scales, torch.float32).transpose(-2, -1)
        torch.testing.assert_close(decoded, getattr(original, name), rtol=0, atol=0)


def test_unquantized_storage_supports_fp32(moe_config: MoEConfig) -> None:
    torch.manual_seed(19)
    original = GroupedExperts(moe_config)
    with torch.no_grad():
        original.init_weights(torch.device("cpu"))
    patched = patch_moe_module(original, dim=4, expert_weight_format="bf16")
    assert patched.gate_and_up_projs.dtype == torch.float32
    assert patched.lora_gate_and_up_A.dtype == torch.float32
    x_ref = torch.randn(4, 64, requires_grad=True)
    x = x_ref.detach().clone().requires_grad_()
    indices = torch.tensor([[0, 1], [1, 3], [2, 0], [3, 2]])
    weights = torch.softmax(torch.randn(4, 2), dim=-1)
    mask = torch.ones(4, dtype=torch.bool)
    # Check the FP32 math eagerly; compiler startup is outside this storage-mode test.
    with torch.compiler.set_stance("force_eager"):
        expected = original(x_ref, mask, weights, indices)
        actual = patched(x, mask, weights, indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        actual.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)
        assert patched.gate_and_up_projs.grad is None
        assert patched.lora_gate_and_up_B.grad is not None
        assert torch.isfinite(patched.lora_gate_and_up_B.grad).all()
        assert torch.count_nonzero(patched.lora_gate_and_up_B.grad) > 0


@pytest.mark.parametrize("storage_format", ("mxf4", "fp32"))
def test_patch_rejects_invalid_storage_mode(storage_format: str, moe_config: MoEConfig) -> None:
    original = GroupedExperts(moe_config)
    with pytest.raises(ValueError, match="storage mode"):
        patch_moe_module(original, expert_weight_format=storage_format)

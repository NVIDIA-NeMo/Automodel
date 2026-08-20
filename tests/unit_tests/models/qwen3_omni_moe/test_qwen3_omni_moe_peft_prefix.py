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

"""PEFT save/resume prefix handling for the Qwen3 Omni MoE adapter (CPU)."""

from types import SimpleNamespace

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _tiny_adapter():
    moe = MoEConfig(
        dim=32,
        inter_dim=64,
        moe_inter_dim=16,
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
    )
    backend = BackendConfig(linear="torch", rms_norm="torch", attn="sdpa")
    return Qwen3OmniMoeStateDictAdapter(SimpleNamespace(), moe, backend)


def _peft_lora_state_dict(rank=4, n_experts=2, dim=32, inter=16):
    # grouped expert LoRA params exactly as ModelState.state_dict() emits them on
    # a PEFT save, shapes matching GroupedExpertsLoRA (lora_experts.py): A is
    # [experts, in_features, rank], B is [experts, rank, out_features]. Plus one
    # attention LoRA key for contrast.
    base = "base_model.model.model.layers.0.mlp.experts"
    attn = "base_model.model.model.layers.0.self_attn.q_proj"
    return {
        f"{base}.lora_gate_and_up_A": torch.randn(n_experts, dim, rank),
        f"{base}.lora_gate_and_up_B": torch.randn(n_experts, rank, 2 * inter),
        f"{base}.lora_down_A": torch.randn(n_experts, inter, rank),
        f"{base}.lora_down_B": torch.randn(n_experts, rank, dim),
        f"{attn}.lora_A.weight": torch.randn(rank, dim),
    }


def test_peft_lora_keys_get_thinker_inside_the_peft_prefix():
    """The thinker namespace must land inside the PEFT prefix.

    Before the fix every key got "thinker." prepended on the outside,
    producing ``thinker.base_model.model...`` names that HF PEFT cannot
    attach. On the actual HF omni model PEFT names the text modules
    ``base_model.model.thinker.model.layers...``.
    """
    adapter = _tiny_adapter()
    out = adapter.to_hf(_peft_lora_state_dict())

    assert out, "no keys came back from to_hf"
    for key in out:
        assert ".lora_" in key, key
        assert key.startswith("base_model.model.thinker.model.layers."), key
    # grouped expert tensors still split into per-expert projections
    assert any(".mlp.experts.0.gate_proj.lora_A.weight" in k for k in out)
    assert any(".mlp.experts.1.down_proj.lora_B.weight" in k for k in out)


def test_convert_single_tensor_moves_thinker_inside_the_peft_prefix():
    adapter = _tiny_adapter()
    tensor = torch.randn(4, 32)
    result = adapter.convert_single_tensor_to_hf(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight", tensor
    )

    assert result == [("base_model.model.thinker.model.layers.0.self_attn.q_proj.lora_A.weight", tensor)]


def test_full_weights_still_get_the_thinker_prefix():
    adapter = _tiny_adapter()
    out = adapter.to_hf({"model.layers.0.self_attn.q_proj.weight": torch.randn(32, 32)})

    assert list(out) == ["thinker.model.layers.0.self_attn.q_proj.weight"]


def test_peft_lora_save_round_trips_for_resume():
    """from_hf must rebuild the exact grouped keys ModelState expects on resume."""
    adapter = _tiny_adapter()
    sd = _peft_lora_state_dict()
    out = adapter.to_hf({k: v.clone() for k, v in sd.items()})
    back = adapter.from_hf(dict(out))

    assert set(back) == set(sd)
    for key in sd:
        torch.testing.assert_close(back[key], sd[key])


def test_peft_resume_does_not_flip_the_prefix_flags():
    """A LoRA-only adapter dict must not disable the thinker prefix for later
    full-checkpoint saves; the flags describe the base checkpoint layout."""
    adapter = _tiny_adapter()
    adapter.from_hf(adapter.to_hf(_peft_lora_state_dict()))

    assert adapter._uses_thinker_prefix is True
    assert adapter._uses_model_prefix is True

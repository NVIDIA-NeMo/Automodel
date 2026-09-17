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

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import (
    KimiK3Gate,
    KimiK3MoE,
    _build_moe_config,
)
from nemo_automodel.components.moe.layers import FakeBalancedGate


def _tiny_config() -> KimiK3TextConfig:
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        first_k_dense_replace=0,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": [1],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=1,
    )


def _torch_backend(**overrides) -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
        **overrides,
    )


def _build_moe(backend: BackendConfig) -> KimiK3MoE:
    config = _tiny_config()
    moe_config = _build_moe_config(config, torch.float32, None)
    return KimiK3MoE(config, moe_config, backend)


def test_kimi_k3_moe_uses_learned_gate_by_default():
    moe = _build_moe(_torch_backend())
    assert isinstance(moe.gate, KimiK3Gate)


def test_kimi_k3_moe_honors_fake_balanced_gate():
    moe = _build_moe(_torch_backend(fake_balanced_gate=True, fake_gate_noise=0.25))
    assert isinstance(moe.gate, FakeBalancedGate)
    assert moe.gate.noise == 0.25


def test_kimi_k3_fake_balanced_gate_spreads_tokens_across_experts():
    moe = _build_moe(_torch_backend(fake_balanced_gate=True))
    hidden_states = torch.randn(8, moe.dim)
    token_mask = torch.ones(8, dtype=torch.bool)

    weights, indices, _ = moe.gate(hidden_states, token_mask, None)

    assert weights.shape == indices.shape == (8, moe.n_activated_experts)
    # noise=0.0 assigns tokens round-robin: every expert receives the same
    # number of (token, slot) assignments instead of collapsing onto [0..topk).
    counts = torch.bincount(indices.flatten(), minlength=moe.n_routed_experts)
    assert (counts == counts[0]).all()

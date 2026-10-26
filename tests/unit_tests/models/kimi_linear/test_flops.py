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

"""Independent shape arithmetic and dispatch checks for Kimi Linear FLOPs."""

import pytest

from nemo_automodel.components.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel.components.models.kimi_linear.flops import model_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config


def _config(**overrides: object) -> KimiLinear48BConfig:
    values = dict(
        hidden_size=8,
        intermediate_size=12,
        vocab_size=19,
        num_hidden_layers=3,
        num_attention_heads=2,
        kv_lora_rank=3,
        qk_nope_head_dim=2,
        qk_rope_head_dim=2,
        v_head_dim=4,
        mla_use_nope=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=1,
        moe_intermediate_size=5,
        first_k_dense_replace=1,
        moe_layer_freq=2,
        max_position_embeddings=7,
        linear_attn_config={
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
        },
    )
    values.update(overrides)
    return KimiLinear48BConfig(**values)


@pytest.mark.parametrize("length", [1, 7, 64, 129])
def test_matrix_shapes_and_kda_kernel_work(length: int) -> None:
    # Two dense SwiGLUs and one MoE with two routed + one shared SwiGLU.
    matrices = [(12, 8), (12, 8), (8, 12)] * 2
    matrices += [(5, 8), (5, 8), (8, 5)] * 3 + [(4, 8)]
    # Two KDA blocks: Q/K/V/O, forget gate, output gate, beta.
    matrices += ([(8, 8)] * 4 + [(4, 8), (8, 4)] * 2 + [(2, 8)]) * 2
    # One MLA block: Q, KV-A, KV-B, O. Then LM head.
    matrices += [(8, 8), (5, 8), (12, 3), (8, 8), (19, 8)]
    linear = 6 * length * sum(rows * columns for rows, columns in matrices)
    conv = 6 * length * 2 * 3 * 8 * 3
    # KDA paper forward FLOPs per head, tripled for backward; 2 layers x 2 heads.
    delta = 3 * 2 * 2 * length * (6 * 4**2 + 3 * 64 * 4 + 64**2)
    # QK/PV in two heads, conventional half-square causal estimate.
    mla = 6 * 2 * length**2 * (4 + 4) / 2
    assert model_flops(_config(), gbs=3, seq_len=length) == 3 * (linear + conv + delta + mla)


def test_config_hook_survives_serialization() -> None:
    config = _config()
    assert get_flops_formula_for_hf_config(config) is model_flops
    assert model_flops(config) == model_flops(config, seq_len=7)
    assert "flops_formula" not in config.to_json_string()
    restored = KimiLinear48BConfig.from_dict(config.to_dict())
    assert get_flops_formula_for_hf_config(restored)(restored) == model_flops(config)


def test_expert_selection_not_total_expert_bank() -> None:
    base = model_flops(_config(), seq_len=7)
    # Increasing the bank only expands the router, not the activated SwiGLUs.
    assert model_flops(_config(num_experts=8), seq_len=7) - base == 6 * 7 * 8 * 4
    # One extra routed or shared expert has identical projection work.
    extra = 6 * 7 * 3 * 8 * 5
    assert model_flops(_config(num_experts_per_token=3), seq_len=7) - base == extra
    assert model_flops(_config(num_shared_experts=2), seq_len=7) - base == extra


def test_only_mla_attention_grows_quadratically() -> None:
    config = _config()
    assert model_flops(config, seq_len=14) - 2 * model_flops(config, seq_len=7) == 6 * 2 * 7**2 * 8
    config = _config(
        linear_attn_config={
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [],
            "num_heads": 2,
            "head_dim": 4,
            "short_conv_kernel_size": 3,
        }
    )
    assert model_flops(config, seq_len=14) == 2 * model_flops(config, seq_len=7)


def test_published_48b_architecture() -> None:
    config = KimiLinear48BConfig(
        hidden_size=2304,
        intermediate_size=9216,
        num_hidden_layers=27,
        num_attention_heads=32,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_nope=True,
        num_experts=256,
        num_experts_per_token=8,
        num_shared_experts=1,
        moe_intermediate_size=1024,
        first_k_dense_replace=1,
        linear_attn_config={
            "kda_layers": [i for i in range(1, 28) if i % 4 and i != 27],
            "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],
            "num_heads": 32,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
        },
    )
    # Audited projection matrices + chunk KDA: 18,884,296,704 FLOPs/token.
    # Seven 4K causal MLA blocks add 880,803,840 FLOPs/token.
    assert model_flops(config, gbs=8, seq_len=4096) == 8 * 4096 * 19_765_100_544


@pytest.mark.parametrize("gbs,length", [(0, 7), (1, 0)])
def test_empty_work(gbs: int, length: int) -> None:
    assert model_flops(_config(), gbs=gbs, seq_len=length) == 0


@pytest.mark.parametrize("gbs,length", [(-1, 7), (1, -1)])
def test_negative_work_rejected(gbs: int, length: int) -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        model_flops(_config(), gbs=gbs, seq_len=length)


def test_unsupported_query_lora_rejected() -> None:
    with pytest.raises(ValueError, match="q_lora_rank=None"):
        model_flops(_config(q_lora_rank=2), seq_len=7)

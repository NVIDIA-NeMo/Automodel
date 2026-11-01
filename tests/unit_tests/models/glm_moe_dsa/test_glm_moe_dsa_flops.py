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

"""GLM MoE DSA model-FLOPs formula: hand-derived expectations and registry wiring."""

import pytest
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.glm_moe_dsa.flops import model_flops
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config, mla_moe_flops

pytestmark = pytest.mark.cpu_only

HIDDEN, HEADS, VOCAB = 64, 4, 256
Q_LORA, KV_LORA, QK_NOPE, QK_ROPE, V_HEAD = 32, 16, 12, 4, 16
INTER, MOE_INTER, N_EXPERTS, TOPK_EXPERTS, N_SHARED = 128, 64, 4, 2, 1
INDEX_HEADS, INDEX_DIM, INDEX_TOPK = 2, 16, 8


def _config(
    indexer_types=("full", "full", "shared", "shared"),
    mlp_layer_types=("dense", "dense", "sparse", "sparse"),
    num_nextn_predict_layers: int = 1,
) -> GlmMoeDsaConfig:
    return GlmMoeDsaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_key_value_heads=HEADS,
        num_hidden_layers=len(mlp_layer_types),
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=N_SHARED,
        num_experts_per_tok=TOPK_EXPERTS,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=256,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=QK_ROPE,
        v_head_dim=V_HEAD,
        index_n_heads=INDEX_HEADS,
        index_head_dim=INDEX_DIM,
        index_topk=INDEX_TOPK,
        mlp_layer_types=list(mlp_layer_types),
        indexer_types=list(indexer_types),
        num_nextn_predict_layers=num_nextn_predict_layers,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    )


def _expected_flops(seq_len: int, gbs: int, n_dense: int, n_moe: int, n_indexer: int) -> float:
    """Independent derivation written term by term (no helper shared with the implementation)."""
    attention_params = (
        HIDDEN * Q_LORA  # q_a
        + Q_LORA * HEADS * (QK_NOPE + QK_ROPE)  # q_b
        + HIDDEN * (KV_LORA + QK_ROPE)  # kv_a with the shared rope key
        + KV_LORA * HEADS * (QK_NOPE + V_HEAD)  # kv_b
        + HEADS * V_HEAD * HIDDEN  # o
    )
    dense_ffn = 3 * HIDDEN * INTER
    moe_ffn = 3 * HIDDEN * MOE_INTER * (TOPK_EXPERTS + N_SHARED) + HIDDEN * N_EXPERTS
    layers = n_dense + n_moe
    per_token = HIDDEN * VOCAB + layers * attention_params + n_dense * dense_ffn + n_moe * moe_ffn
    # Causal query at one-based position i selects min(i, topk) keys; QK over the full qk head
    # dim and PV over the value head dim, every head.
    pairs = sum(min(position, INDEX_TOPK) for position in range(1, seq_len + 1))
    attention_macs = layers * pairs * HEADS * ((QK_NOPE + QK_ROPE) + V_HEAD)
    # Indexer: projections at every token, causal score GEMM, forward only, full layers only.
    indexer_proj = Q_LORA * INDEX_HEADS * INDEX_DIM + HIDDEN * INDEX_DIM + HIDDEN * INDEX_HEADS
    indexer_macs = n_indexer * (seq_len * indexer_proj + INDEX_HEADS * INDEX_DIM * seq_len * (seq_len + 1) // 2)
    return float(gbs * (6 * (seq_len * per_token + attention_macs) + 2 * indexer_macs))


def test_matches_hand_derivation_past_the_topk_cap():
    # seq_len 12 > index_topk 8, so the last four queries are capped at eight keys.
    assert model_flops(_config(), gbs=2, seq_len=12) == _expected_flops(12, 2, n_dense=2, n_moe=2, n_indexer=2)


def test_short_sequence_is_plain_causal():
    assert model_flops(_config(), gbs=1, seq_len=5) == _expected_flops(5, 1, n_dense=2, n_moe=2, n_indexer=2)


def test_indexer_counted_only_on_full_layers_and_forward_only():
    all_full = model_flops(_config(indexer_types=("full",) * 4), gbs=3, seq_len=12)
    two_full = model_flops(_config(), gbs=3, seq_len=12)
    indexer_proj = Q_LORA * INDEX_HEADS * INDEX_DIM + HIDDEN * INDEX_DIM + HIDDEN * INDEX_HEADS
    one_layer_indexer_macs = 12 * indexer_proj + INDEX_HEADS * INDEX_DIM * 12 * 13 // 2
    assert all_full - two_full == 3 * 2 * 2 * one_layer_indexer_macs


def test_dense_moe_pattern_falls_back_to_first_k_dense_replace():
    config = _config(mlp_layer_types=("dense", "sparse", "sparse", "sparse"))
    fallback = _config()
    fallback.mlp_layer_types = None
    fallback.first_k_dense_replace = 1
    assert model_flops(fallback, gbs=1, seq_len=12) == model_flops(config, gbs=1, seq_len=12)


def test_mtp_layers_are_not_counted():
    assert model_flops(_config(num_nextn_predict_layers=0), seq_len=12) == model_flops(
        _config(num_nextn_predict_layers=1), seq_len=12
    )


def test_defaults_to_max_position_embeddings():
    config = _config()
    assert model_flops(config) == model_flops(config, seq_len=config.max_position_embeddings)


@pytest.mark.parametrize("kwargs", [{"seq_len": 0}, {"gbs": 0}])
def test_rejects_non_positive_sizes(kwargs):
    with pytest.raises(ValueError):
        model_flops(_config(), **kwargs)


def test_registry_selects_the_model_formula_and_generic_formula_overcounts():
    config = _config()
    assert get_flops_formula_for_hf_config(config) is model_flops
    # The generic MLA+MoE formula counts an indexer on every layer (including MTP) at 6 FLOPs/MAC
    # over a full square score matrix, an MTP layer the model does not run, and S x topk pairs.
    assert mla_moe_flops(config, gbs=1, seq_len=12) > model_flops(config, gbs=1, seq_len=12)

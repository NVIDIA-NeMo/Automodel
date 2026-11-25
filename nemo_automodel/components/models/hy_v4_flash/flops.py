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

"""Model FLOPs of Hy4-flash, in the repository's convention (6 x MACs, causal-aware, recompute excluded).

Registered through ``HYV4FlashConfig.flops_formula`` so no shared utility names this model.

Per token (``hs`` hidden, ``H`` query heads of the layer type, ``nope + rope = 256`` query/key width, ``v = 256``):

* attention linears of every layer (expanded MLA, the mathematical definition; the absorbed kernel execution does extra
  work that belongs to hardware FLOPs, not model FLOPs): ``q_a`` (hs x q_lora), ``q_b`` (q_lora x H*256),
  ``kv_a`` (hs x (kv_lora + rope)), ``kv_b`` (kv_lora x H*(nope + v)), ``gate`` (hs x H*v), ``o`` (H*v x hs);
* attention BMMs, summed over query positions: QK^T and PV each cost ``H * 256`` MACs per key.
  Sliding layers see ``min(i + 1, window)`` keys; sparse layers see ``min(i + 1, index_topk)`` keys;
* the frozen compressed indexer (hard top-k has no gradient; counted forward only, 2 x MACs) on sparse layers:
  query projection (q_lora x n_heads*d), head weights (hs x n_heads), token keys (hs x d), pooling value + gate
  projections (2 x d x d per token) and the group scoring ``n_heads * d`` MACs per visible complete group;
* MoE layers: router (hs x experts), latent down / up projections (2 x hs x latent), ``top_k`` routed experts of
  ``3 x latent x moe_inter`` plus one shared expert of ``3 x hs x moe_inter``; the dense layer is ``3 x hs x
  intermediate``;
* hyper-connections: two sites per layer, each ``2*hc x hc*hs`` MACs; the head ``hc x hc*hs``;
* Engram layers: ``hc`` key projections and one value projection of ``(orders * n_embed)`` x hs;
* the fp32 ``lm_head`` (hs x vocab).

The vision tower is not counted (as for the other VLM entries). Table gathers, hashing, norms, RoPE and gating are
not GEMMs and are not counted.
"""

from __future__ import annotations

from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashConfig, HYV4FlashTextConfig
from nemo_automodel.components.utils.flops_utils import _sum_min_floor_div, _sum_min_window


def hy_v4_flash_flops(config: HYV4FlashConfig | HYV4FlashTextConfig, gbs: int = 1, seq_len: int | None = None) -> float:
    """Model FLOPs of one training step of ``gbs`` sequences of ``seq_len`` tokens.

    Args:
        config: Composite or text configuration.
        gbs: Number of sequences per step.
        seq_len: Tokens per sequence (documents are counted as one sequence); defaults to ``max_position_embeddings``.

    Returns:
        FLOPs of the step (6 x MACs for trainable GEMMs and attention BMMs, 2 x MACs for the frozen indexer).
    """
    text = config.text_config if isinstance(config, HYV4FlashConfig) else config
    seq_len = int(text.max_position_embeddings if seq_len is None else seq_len)
    hs, hc = text.hidden_size, text.hc_mult
    macs_per_token = 0
    attention_keys_macs = 0
    indexer_macs = 0
    for layer, attention_type in enumerate(text.attention_types):
        heads = text.attention_heads(attention_type)
        nope, rope, value = text.attention_dims(attention_type)
        qk = nope + rope
        macs_per_token += hs * text.q_lora_rank + text.q_lora_rank * heads * qk
        macs_per_token += hs * (text.kv_lora_rank + rope) + text.kv_lora_rank * heads * (nope + value)
        macs_per_token += hs * heads * value + heads * value * hs
        if attention_type == "sliding_attention":
            keys = _sum_min_window(seq_len, text.sliding_window)
        else:
            keys = _sum_min_window(seq_len, text.index_topk)
            n, d = text.index_n_heads, text.index_head_dim
            indexer_macs += seq_len * (text.q_lora_rank * n * d + hs * n + hs * d + 2 * d * d)
            indexer_macs += n * d * _sum_min_floor_div(seq_len, text.index_kpool, None)
        attention_keys_macs += heads * (qk + value) * keys
        if text.mlp_layer_types[layer] == "sparse":
            macs_per_token += hs * text.n_routed_experts + 2 * hs * text.routed_expert_hidden_size
            macs_per_token += text.num_experts_per_tok * 3 * text.routed_expert_hidden_size * text.moe_intermediate_size
            macs_per_token += text.n_shared_experts * 3 * hs * text.moe_intermediate_size
        else:
            macs_per_token += 3 * hs * text.intermediate_size
        macs_per_token += 2 * (2 * hc) * hc * hs
    macs_per_token += hc * hc * hs
    engram_in = (text.engram_max_ngram_size - 1) * text.engram_n_embed_per_ngram
    macs_per_token += len(text.engram_layer_indices) * (hc + 1) * engram_in * hs
    macs_per_token += hs * text.vocab_size
    return float(6 * gbs * seq_len * macs_per_token + 6 * gbs * attention_keys_macs + 2 * gbs * indexer_macs)

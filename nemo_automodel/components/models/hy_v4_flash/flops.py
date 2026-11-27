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

Registered through ``HYV4FlashConfig.flops_formula`` so no shared utility names this model. Counted per token:

* the expanded MLA linears of every layer (the absorbed kernel's extra work is hardware FLOPs, not model FLOPs);
* the QK^T and PV products over ``min(i + 1, window)`` keys in sliding layers and, in sparse layers, over the query's
  own partial pool plus at most ``index_topk / index_kpool - 1`` closed pools;
* the frozen compressed indexer on sparse layers, forward only (2 x MACs: a hard top-k has no gradient); documents
  no longer than ``index_topk`` only project their token keys;
* the MoE router, latent projections, ``top_k`` routed experts and the shared expert (a dense MLP in layer 0);
* two hyper-connection sites per layer and the head, the Engram projections, and the fp32 ``lm_head``.

The vision tower, table gathers, hashing, norms, RoPE and gating are not counted.
"""

from __future__ import annotations

from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashConfig, HYV4FlashTextConfig
from nemo_automodel.components.utils.flops_utils import _sum_min_floor_div, _sum_min_window


def hy_v4_flash_flops(config: HYV4FlashConfig | HYV4FlashTextConfig, gbs: int = 1, seq_len: int | None = None) -> float:
    """Model FLOPs of one training step of ``gbs`` sequences of ``seq_len`` tokens.

    ``config`` is the composite or the text configuration; ``seq_len`` defaults to ``max_position_embeddings``.
    Trainable GEMMs and attention products count 6 x MACs, the frozen indexer 2 x MACs.
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
            # Each query attends its own partial pool plus at most topk / pool - 1 closed pools.
            pool = text.index_kpool
            closed = _sum_min_floor_div(seq_len - 1, pool, None)
            recalled = _sum_min_floor_div(seq_len - 1, pool, text.index_topk // pool - 1)
            keys = seq_len * (seq_len + 1) // 2 - pool * (closed - recalled)
            n, d = text.index_n_heads, text.index_head_dim
            indexer_macs += seq_len * hs * d  # token keys are projected even on the dense-prefix path
            if seq_len > text.index_topk:
                pooled_tokens = ((seq_len - 1) // pool) * pool
                indexer_macs += seq_len * (text.q_lora_rank * n * d + hs * n)
                indexer_macs += pooled_tokens * 2 * d * d + n * d * closed
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

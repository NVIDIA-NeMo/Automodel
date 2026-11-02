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

"""Effective Step-5 text FLOPs, including CSA's forward-only discrete indexer."""

from collections import Counter

from nemo_automodel.components.models.step5.configuration import Step5Config


def _sum_history(length: int, region_size: int, cap: int | None) -> int:
    """Sum complete historical regions for document-local queries zero through length-1."""
    full, tail = divmod(length, region_size)
    if cap is None or full <= cap:
        return region_size * full * (full - 1) // 2 + full * tail
    return region_size * cap * (cap - 1) // 2 + cap * (length - cap * region_size)


def _document_work(config: Step5Config, lengths: list[int], supervised_tokens: int) -> float:
    """Count independent documents; each length excludes physical pack padding."""
    sparse = config.sparse_config
    kinds = config.layer_types[: config.num_hidden_layers]
    if sparse.enabled and "sliding_attention" in sparse.apply_to_layer_types:
        raise ValueError("Step5 FLOPs do not support CSA selection intersected with sliding attention")
    swa_layers = kinds.count("sliding_attention")
    csa_layers = sum(sparse.enabled and kind in sparse.apply_to_layer_types for kind in kinds)
    dense_attention_layers = len(kinds) - swa_layers - csa_layers
    moe_layers = sum(config.use_moe and index in config.moe_layer_list for index in range(len(kinds)))
    hidden, heads, dim = config.hidden_size, config.num_attention_heads, config.head_dim
    attention_weights = hidden * ((2 * heads + 2 * config.num_attention_groups) * dim)
    if config.use_head_wise_attn_gate:
        attention_weights += hidden * heads
    dense_ffn_weights = 3 * hidden * config.intermediate_size
    moe_weights = 3 * hidden * (config.moe_top_k * config.moe_intermediate_size + config.share_expert_dim)
    router_weights = hidden * config.moe_num_experts
    trainable_weights = (
        len(kinds) * attention_weights
        + (len(kinds) - moe_layers) * dense_ffn_weights
        + moe_layers * (moe_weights + router_weights)
    )
    head_weights = hidden * config.vocab_size
    indexer_weights = (
        csa_layers
        * hidden
        * ((sparse.sparse_indexer_num_heads + 2) * sparse.proxy_dim + sparse.sparse_indexer_num_heads)
    )
    tokens = sum(lengths)
    if not 0 <= supervised_tokens <= tokens:
        raise ValueError("supervised_tokens must be between zero and the number of valid tokens")
    swa_pairs = csa_pairs = dense_pairs = history_pairs = 0
    for length, count in Counter(lengths).items():
        if length < 0:
            raise ValueError("Document lengths must be nonnegative")
        window = min(length, config.sliding_window)
        swa_pairs += count * (window * (window + 1) // 2 + (length - window) * window)
        region_size = sparse.region_block_size
        blocks, tail = divmod(length, region_size)
        current = blocks * region_size * (region_size + 1) // 2 + tail * (tail + 1) // 2
        csa_pairs += count * (region_size * _sum_history(length, region_size, sparse.topk) + current)
        dense_pairs += count * length * (length + 1) // 2
        history_pairs += count * _sum_history(length, region_size, None)
    linear = 6 * (tokens * trainable_weights + supervised_tokens * head_weights)
    attention = (
        12 * heads * dim * (swa_layers * swa_pairs + csa_layers * csa_pairs + dense_attention_layers * dense_pairs)
    )
    # Hard top-k is under no_grad: count Q/K/Z/W projections and legal Q@proxy
    # scores once. Future/cross-document scores and activation recomputation are
    # execution overhead, not additional effective model work.
    indexer = 2 * (
        tokens * indexer_weights + csa_layers * sparse.sparse_indexer_num_heads * sparse.proxy_dim * history_pairs
    )
    return float(linear + attention + indexer)


def step5_flops(config: Step5Config, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count Step-5 text training GEMMs for independent, fully supervised documents.

    The count includes trainable linear/attention forward and backward (6 x MACs)
    and the forward-only indexer (2 x MACs). Embedding lookup, norms, activations,
    rotary/softmax elementwise work, vision, MTP, optimizer work, kernel padding,
    masked indexer scores and activation recomputation are excluded. A CSA query
    at position t sees region_size * min(topk, floor(t / region_size)) historical
    tokens plus t % region_size + 1 tokens in its current region.

    Args:
        config: Actual complete text configuration, before pipeline layer partitioning.
        gbs: Number of independent documents.
        seq_len: Tokens per document; defaults to max_position_embeddings.

    Returns:
        Effective training FLOPs with all document tokens supervised.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if gbs < 0 or length < 0:
        raise ValueError("gbs and seq_len must be nonnegative")
    return _document_work(config, [length] * gbs, gbs * length)

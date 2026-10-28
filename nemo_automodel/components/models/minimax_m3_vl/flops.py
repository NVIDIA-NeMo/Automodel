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

"""Useful text-backbone FLOPs for MiniMax M3 training."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig


def model_flops(config: "MiniMaxM3VLTextConfig", *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count useful GEMM FLOPs for text-only full-parameter training.

    Trainable projections and attention count forward, input gradients and weight
    gradients (six FLOPs per MAC). Hard block selection has no gradient, so the
    indexer projections and causal scores count forward only (two per MAC).
    Activation recomputation, masked-out dense work, padding, optimizer updates,
    communication, softmax and elementwise operations are excluded. This is MFU,
    not the FLOPs actually executed by a particular backend (HFU).

    Each sequence is one document without padding. Sparse layers force the
    current block into the top-k budget; its causal partial length is counted
    exactly. Packed-document FLOPs require the individual document lengths and
    cannot be inferred from a packed row length.

    Args:
        config: Effective text-backbone configuration, with MTP disabled.
        gbs: Number of full-length sequences in the global optimizer step.
        seq_len: Tokens in each sequence; defaults to max_position_embeddings.

    Returns:
        Useful model FLOPs over all devices for one optimizer step.

    Raises:
        ValueError: Sequence dimensions, MTP, or sparse selection settings cannot
            be represented by this text-backbone accounting.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if length <= 0 or gbs <= 0:
        raise ValueError("MiniMax M3 FLOPs require positive seq_len and gbs")
    if config.num_mtp_modules:
        raise ValueError("MiniMax M3 text FLOPs require num_mtp_modules=0")
    layers = config.num_hidden_layers
    if config.attention_output_gate:
        raise ValueError("MiniMax M3 FLOPs do not support attention_output_gate")
    if config.moe_layer_freq is not None and len(config.moe_layer_freq) < layers:
        raise ValueError("MiniMax M3 moe_layer_freq must cover num_hidden_layers")
    hidden = config.hidden_size
    heads = config.num_attention_heads
    head_dim = config.head_dim
    moe_layers = (
        layers if config.moe_layer_freq is None else sum(value != 0 for value in config.moe_layer_freq[:layers])
    )
    dense_layers = layers - moe_layers
    attention_params = layers * 2 * hidden * (heads + config.num_key_value_heads) * head_dim
    dense_params = dense_layers * 3 * hidden * config.dense_intermediate_size
    expert_params = (
        moe_layers
        * 3
        * hidden
        * (
            config.num_experts_per_tok * config.intermediate_size
            + config.n_shared_experts * config.shared_intermediate_size
        )
    )
    router_params = moe_layers * hidden * config.num_local_experts
    lm_head_params = hidden * config.vocab_size
    trainable = 6 * gbs * length * (attention_params + dense_params + expert_params + router_params + lm_head_params)

    causal_pairs = length * (length + 1) // 2
    sparse_layers = 0
    selected_pairs = causal_pairs
    indexer = 0
    sparse = config.sparse_attention_config
    if sparse is not None and sparse.get("use_sparse_attention", True):
        frequency = sparse.get("sparse_attention_freq")
        if frequency is not None and len(frequency) < layers:
            raise ValueError("MiniMax M3 sparse_attention_freq must cover num_hidden_layers")
        sparse_layers = layers if frequency is None else sum(value != 0 for value in frequency[:layers])
        if sparse_layers:
            block = sparse["sparse_block_size"]
            budget = sparse["sparse_topk_blocks"]
            if block <= 0 or budget <= 0:
                raise ValueError("MiniMax M3 FLOPs require positive sparse block size and top-k budget")
            if length > block * budget:
                if sparse.get("sparse_local_block", 1) <= 0:
                    raise ValueError("MiniMax M3 FLOPs require the local block for exact sparse key counts")
                if sparse.get("sparse_init_block", 0) >= budget:
                    raise ValueError(
                        "MiniMax M3 FLOPs require the forced initial blocks to leave room for the local block"
                    )
                if sparse.get("sparse_init_block", 0) + sparse.get("sparse_local_block", 1) > budget:
                    raise ValueError("MiniMax M3 FLOPs require forced initial and local blocks to fit the top-k budget")
            dense_prefix = min(length, block * budget)
            tail = length - dense_prefix
            full_blocks, remainder = divmod(tail, block)
            selected_pairs = (
                dense_prefix * (dense_prefix + 1) // 2
                + tail * (budget - 1) * block
                + full_blocks * block * (block + 1) // 2
                + remainder * (remainder + 1) // 2
            )
            index_heads = sparse["sparse_num_index_heads"]
            index_dim = sparse["sparse_index_dim"]
            projection_macs = length * hidden * (index_heads + 1) * index_dim
            score_macs = index_heads * index_dim * causal_pairs
            indexer = 2 * gbs * sparse_layers * (projection_macs + score_macs)
    attention = 12 * gbs * heads * head_dim * ((layers - sparse_layers) * causal_pairs + sparse_layers * selected_pairs)
    return float(trainable + attention + indexer)

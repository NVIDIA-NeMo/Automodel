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

"""Matrix-product FLOPs of Inkling2 text training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.inkling2.config import Inkling2TextConfig


def _causal_pairs(length: int, window: int | None) -> int:
    """Count the query/key pairs of one document, each query seeing at most ``window`` keys."""
    if window is None or length <= window:
        return length * (length + 1) // 2
    return window * (window + 1) // 2 + (length - window) * window


def model_flops(config: Inkling2TextConfig, *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count forward and backward FLOPs of ``gbs`` unpacked sequences of ``seq_len`` tokens.

    Every matrix product counts 6 FLOPs per multiply-accumulate (forward and backward),
    once per layer visit; a layer of the loop core is visited once per pass. Counted: the
    fused query/key/value/relative/gate/sink projection with each row computed once, the
    relative-profile product, attention scores and values over the causal keys (a sliding
    window in local layers), the output projection, the dynamic convolution taps of full
    layers, the router, the selected experts, the message controllers of later passes and
    the head over the usable vocabulary. Norms, residual mixing, static convolutions,
    activation recomputation and other elementwise work are excluded, as for the other
    MFU formulas.
    """
    if seq_len is None:
        seq_len = config.max_position_embeddings
    hidden = config.hidden_size
    experts = config.num_experts_per_tok * 3 * hidden * config.intermediate_size
    router = hidden * config.n_routed_experts
    loop = config.loop_config
    core = config.loop_core_layers
    local_layers = set(config.local_layer_ids)
    weights = 0
    pairs = 0
    for layer in range(config.num_hidden_layers):
        is_local = layer in local_layers
        heads = config.swa_num_attention_heads if is_local else config.num_attention_heads
        kv_heads = config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        head_dim = config.swa_head_dim if is_local else config.head_dim
        extent = config.sliding_window_size if is_local else config.rel_extent
        query_width = heads * head_dim
        projection = hidden * (2 * query_width + 2 * kv_heads * head_dim + heads * config.d_rel + heads)
        relative = heads * config.d_rel * extent
        taps = 0 if is_local else 2 * hidden * kv_heads * config.sconv_kernel_size
        layer_weights = projection + relative + query_width * hidden + taps + router + experts
        visits = loop["num_passes"] if loop is not None and layer in core else 1
        weights += visits * layer_weights
        if visits > 1:
            # Four controllers per layer: one down projection of two streams and two up projections.
            weights += (visits - 1) * 16 * hidden * loop["gate_rank"]
        window = config.sliding_window_size if is_local else None
        # Scores and values each cost one multiply-accumulate per head channel and pair.
        pairs += visits * 2 * query_width * _causal_pairs(seq_len, window)
    head = hidden * config.unpadded_vocab_size
    return 6.0 * gbs * (seq_len * (weights + head) + pairs)

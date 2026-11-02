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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config


def model_flops(config: "MistralLarge4Config", *, gbs: int = 1, seq_len: int | None = None) -> float:
    """Count useful GEMM FLOPs for text-only full-parameter training of Mistral Large 4.

    Trainable projections and attention count forward, input gradients and weight
    gradients (six FLOPs per MAC). The DSA indexer is a frozen hard selector, so it
    runs only on ``full`` layers and counts forward only (two FLOPs per MAC):
    its projections plus scores over the causal keys. Per layer:

    * ``q_proj``: ``hidden * heads * head_dim`` (no Q low-rank path).
    * ``kv_a``: ``hidden * (kv_lora_rank + rope_dim)`` on layers that own their KV
      only; NoPE layers have no rope part and layers that share KV have no ``kv_a``.
    * ``kv_b``: ``kv_lora_rank * heads * (nope_dim + v_head_dim)`` on every layer,
      with ``nope_dim = head_dim`` on NoPE layers and ``qk_nope_head_dim`` otherwise.
    * ``o_proj``: ``heads * v_head_dim * hidden``.
    * Attention: ``pairs * heads * (head_dim + v_head_dim)`` MACs, where a query at
      position ``p`` attends ``min(p + 1, index_topk)`` keys (exact, not ``S * topk``).
    * FFN: layers below ``first_k_dense_replace`` are dense SwiGLU; the others run
      ``num_experts_per_tok`` routed experts and ``n_shared_experts`` shared experts
      of ``moe_intermediate_size`` plus the router.
    * LM head: ``hidden * vocab`` (the embedding lookup has no GEMM).

    Counted work is the algorithmic model FLOPs (MFU), independent of the attention
    backend: the absorbed TileLang kernel and dense SDPA execute different amounts
    of work. Activation recomputation, padding, elementwise operations (norms,
    SSMax, LeRoPE, softcap, softmax, top-k), optimizer updates and communication are
    excluded. Vision and the draft head are not executed by the causal LM and are
    excluded.

    Each sequence is one document without padding. Packed-document FLOPs need the
    individual document lengths and cannot be inferred from a packed row length.

    Args:
        config: Effective Mistral Large 4 configuration.
        gbs: Number of full-length sequences in the global optimizer step.
        seq_len: Tokens in each sequence; defaults to max_position_embeddings.

    Returns:
        Useful model FLOPs over all devices for one optimizer step.

    Raises:
        ValueError: ``gbs`` or ``seq_len`` is not positive.
    """
    length = config.max_position_embeddings if seq_len is None else seq_len
    if length <= 0 or gbs <= 0:
        raise ValueError("Mistral Large 4 FLOPs require positive seq_len and gbs")
    hidden = config.hidden_size
    heads = config.num_attention_heads
    head_dim = config.head_dim
    latent = config.kv_lora_rank
    value_dim = config.v_head_dim
    budget = config.index_topk

    causal_pairs = length * (length + 1) // 2
    if length <= budget:
        selected_pairs = causal_pairs
    else:
        selected_pairs = budget * (budget + 1) // 2 + (length - budget) * budget

    params = 0
    full_indexer_layers = 0
    for layer in range(config.num_hidden_layers):
        use_rope = not config.nope_layer_flags[layer]
        nope_dim = config.qk_nope_head_dim if use_rope else head_dim
        params += hidden * heads * head_dim
        if config.kv_source_layers[layer] == layer:
            params += hidden * (latent + (config.qk_rope_head_dim if use_rope else 0))
        params += latent * heads * (nope_dim + value_dim)
        params += heads * value_dim * hidden
        if layer < config.first_k_dense_replace:
            params += 3 * hidden * config.intermediate_size
        else:
            params += 3 * hidden * config.moe_intermediate_size * (config.num_experts_per_tok + config.n_shared_experts)
            params += hidden * config.n_routed_experts
        if config.index_sharing[layer] == "full":
            full_indexer_layers += 1
    params += hidden * config.vocab_size

    trainable = 6 * gbs * length * params
    attention = 6 * gbs * config.num_hidden_layers * selected_pairs * heads * (head_dim + value_dim)
    index_projection = (
        length * hidden * (config.index_n_heads + 1) * config.index_head_dim + length * hidden * config.index_n_heads
    )
    index_scores = config.index_n_heads * config.index_head_dim * causal_pairs
    indexer = 2 * gbs * full_indexer_layers * (index_projection + index_scores)
    return float(trainable + attention + indexer)

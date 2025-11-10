# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Callable, Optional


def calculate_mfu(tflops, world_size, time_seconds, reference_mfu=1979.0):
    """Calculate Model FLOPs Utilization (MFU).

    Args:
        tflops: TFLOPs per GPU
        world_size: Total number of GPUs
        time_seconds: Time taken for computation
        reference_mfu: Peak TFLOPs of the hardware (default: H100)

    Returns:
        MFU as a percentage
    """
    mfu = tflops / (world_size * time_seconds)
    mfu = mfu / reference_mfu
    return mfu * 100


def gpt3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for GPT3 family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    hs = config.hidden_size
    layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (24 * gbs * seq_len * hs * hs + 4 * gbs * seq_len * seq_len * hs * (0.5 if causal_self_attn else 1)) * (
        3 * layers
    ) + (6 * gbs * seq_len * hs * vocab_size)


def llama2_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for llama2 family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def llama3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for llama3 family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def nemotron_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for nemotron family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (12 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def mixtral_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for mixtral family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 2
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * moe_router_topk * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def qwen3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for Qwen3 family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    # Calculate head_dim if not present (for Qwen2) or use directly (for Qwen3)
    head_dim = config.head_dim if hasattr(config, "head_dim") else (hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    # MoE fields - Qwen3 uses "moe_topk" if present, else dense (1)
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 1
    moe_ffn_hidden_size = (
        config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else config.intermediate_size
    )

    causal_self_attn = True
    hidden_size = hs
    gated_linear_multiplier = 2

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + 1)  # QKV gemm
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))  # attention
            + 1  # attention proj gemm
        )
    )

    # mlp flops
    mlp_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_ffn_hidden_size * moe_router_topk)  # MoE layers
    )

    # vocab flops
    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def bert_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for BERT family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 512

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size

    return 72 * gbs * layers * seq_len * hs * hs * (1 + (seq_len / (6 * hs)) + (vocab_size / (12 * hs * layers)))


def transformer_flops(config, gbs=1, seq_len=None):
    """Calculate FLOPs for a standard Transformer model - accepts either AutoConfig or normalized config.
    Note: This does not cover encoder-decoder models.
    """
    batch_size = gbs
    if seq_len is None:
        seq_length = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048
    else:
        seq_length = seq_len

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    # Handle optional parameters with reasonable defaults
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else num_attention_heads
    causal_self_attn = True  # Default to causal for decoder models
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 0
    kv_channels = hidden_size // num_attention_heads  # Standard dimension per head

    # Calculate query projection size and ratio
    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

    # MoE parameters - simplified for NeMo config
    # In this implementation, we assume all layers are dense if num_experts is None
    if moe_router_topk == 0:
        num_dense_layers = num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
    else:
        # Simplified MoE handling - assuming uniform distribution of MoE layers
        # This can be expanded based on NeMo's actual MoE implementation
        num_moe_layers = num_layers // 2  # Simplified assumption
        num_dense_layers = num_layers - num_moe_layers
        num_experts_routed_to = moe_router_topk

    # Handle SwiGLU vs standard GELU/ReLU
    # Default to standard activation (no SwiGLU)
    gated_linear_multiplier = 1

    # Define the expansion factor as described in the paper
    # 3x: Each GEMM needs forward pass, backward wgrad, and backward dgrad
    # 2x: GEMMs are stacked twice in standard Transformer architectures
    # 2x: A GEMM of m*n with n*k requires 2mnk floating-point operations
    expansion_factor = 3 * 2 * 2
    # Attention
    if not causal_self_attn:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio
    else:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size / 2)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio

    # Calculate total FLOPs
    total_flops = (
        expansion_factor
        * batch_size
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            attention_component
            # MLP component
            + (
                (
                    # Dense layers
                    (ffn_hidden_size * num_dense_layers)
                    +
                    # MoE layers
                    (
                        (
                            # Routed experts
                            ffn_hidden_size * num_experts_routed_to
                            # Note: Shared experts are not implemented in this version
                        )
                        * num_moe_layers
                    )
                )
                * gated_linear_multiplier
                / (num_layers * hidden_size)
            )
            # Logit component
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )

    return total_flops


def clip_vit_l_flops(config):
    """Model FLOPs for CLIP ViT"""

    if config.img_seq_len is None:
        config.img_seq_len = (config.img_h * config.img_w) / (
            config.patch_dim * config.patch_dim
        ) + config.class_token_len
    return config.gbs * config.layers * config.hs * config.hs * config.img_seq_len * (
        24 + (4 * config.img_seq_len / config.hs)
    ) + (2 * config.gbs * config.hs * config.in_channels * config.img_h * config.img_w)


def neva_projection_flops(config):
    """Model FLOPs for NeVA Projection"""

    if "mlp" in config.projector_type:
        return 6 * config.gbs * config.img_seq_len * config.ffn_hs * (config.inp_s + config.hs)
    elif config.projector_type == "affine":
        return 6 * config.gbs * config.img_seq_len * config.inp_s * config.hs
    else:
        raise ValueError(
            f"NeVA Projections FLOPs calculator only supports 'mlp', 'mcore_mlp'"
            f" or 'affine' projector_type but found {config.projector_type}"
        )


def flux_flops(config):
    """Model FLOPs for FLUX"""

    hs = config.hs
    seq_len = config.model_channels + config.inp_s
    base_factor = 6 * config.gbs  # common multiplier for most terms

    # Joint layer computations
    joint_layer_flops = (
        base_factor
        * config.layers[0]
        * (
            10 * hs * hs  # hidden size operations
            + 2 * hs * (config.model_channels + config.inp_s) * (1 + hs * 7)  # channel and context joint attention
            + 2 * (config.model_channels + config.inp_s) * hs  # final projection
        )
    )

    # Single layer computations
    single_layer_flops = (
        base_factor
        * config.layers[1]
        * seq_len
        * hs
        * (
            3  # linear Y
            + 1  # Modulation
            + 4 * hs  # Linear computations
            + (3 * hs + 2 * seq_len)  # attention operations
            + 5 * hs  # feed-forward
            + 1  # Modulation
        )
    )

    # Embedding and projection layers
    other_flops = base_factor * (
        config.inp_s * config.in_channels * hs  # image embedding
        + config.inp_s * hs * config.model_channels  # text embedding
        + config.vec_in_dim * hs
        + hs * hs  # vector embedding
        + 2 * (config.model_channels * hs + hs * hs)  # guidance + timestep embedding
        + (config.inp_s * config.in_channels * hs) / config.gbs  # final projection
    )

    return joint_layer_flops + single_layer_flops + other_flops


def deepseekv3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for DeepSeek V3 - accepts either AutoConfig or normalized config"""

    hs = config.hidden_size
    layers = config.num_hidden_layers
    attention_heads = config.num_attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size

    # DeepSeek V3 specific fields
    q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    qk_nope_head_dim = config.qk_nope_head_dim if hasattr(config, "qk_nope_head_dim") else None

    v_head_dim = config.v_head_dim

    # MoE fields
    moe_intermediate_size = config.moe_intermediate_size
    moe_shared_expert_intermediate_size = moe_intermediate_size
    moe_ffn_hidden_size = moe_intermediate_size
    moe_router_topk = config.num_experts_per_tok

    # MoE layer pattern
    first_k_dense_replace = config.first_k_dense_replace if hasattr(config, "first_k_dense_replace") else 0
    if hasattr(config, "moe_layer_freq"):
        moe_layer_freq = config.moe_layer_freq
    else:
        moe_layer_freq = [0] * first_k_dense_replace + [1] * (layers - first_k_dense_replace)

    # MTP layers (optional)
    mtp_num_layers = config.mtp_num_layers if hasattr(config, "mtp_num_layers") else None

    # self-attention flops
    bmm1_flops = 0.5 * (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * (seq_len**2)
    bmm2_flops = 0.5 * v_head_dim * attention_heads * (seq_len**2)
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * layers
    if mtp_num_layers is not None:
        per_input_attention_flops += 6 * (bmm1_flops + bmm2_flops) * mtp_num_layers

    # linear layer flops
    if q_lora_rank is not None:
        per_layer_mla_params = hs * q_lora_rank + q_lora_rank * (
            (qk_nope_head_dim + qk_rope_head_dim) * attention_heads
        )  # Q
    else:
        per_layer_mla_params = hs * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)  # Q

    per_layer_mla_params += hs * qk_rope_head_dim  # K^R
    per_layer_mla_params += hs * kv_lora_rank + kv_lora_rank * (
        (qk_nope_head_dim + v_head_dim) * attention_heads
    )  # K^C and V^C
    per_layer_mla_params += v_head_dim * attention_heads * hs  # Proj
    mla_params = per_layer_mla_params * layers
    if mtp_num_layers is not None:
        mla_params += per_layer_mla_params * mtp_num_layers

    dense_layer_ffn_params = hs * ffn_hs * 3  # gated linear unit
    per_shared_expert_params = hs * moe_shared_expert_intermediate_size * 3
    per_selected_expert_params = hs * moe_ffn_hidden_size * 3
    ffn_params = 0

    if isinstance(moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(layers)]
    else:
        moe_layer_pattern = moe_layer_freq
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    if mtp_num_layers is not None:
        for i in range(mtp_num_layers):
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    per_input_params = mla_params + ffn_params
    per_input_linear_flops = 6 * per_input_params * seq_len

    # vocab flops
    per_input_vocab_flops = 6 * vocab_size * hs * seq_len
    if mtp_num_layers is not None:
        for i in range(mtp_num_layers):
            per_input_vocab_flops += 6 * vocab_size * hs * seq_len
            per_input_vocab_flops += 6 * hs * 2 * hs * seq_len

    return (per_input_attention_flops + per_input_linear_flops + per_input_vocab_flops) * gbs


def _nemotronh_mlp_layer_flops(config, gbs, seq_len):
    """Model FLOPs for MLP layer. Assume gated linear unit."""
    return 6 * gbs * seq_len * config.hidden_size * config.intermediate_size * 3


def _non_mla_attn_layer_flops(config, gbs, seq_len):
    """Model FLOPs for attention layer"""
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads

    return (
        6
        * gbs
        * seq_len
        * hs
        * (
            hs  # Q
            + query_groups / attention_heads * hs * 2  # KV
            + seq_len / 2 * 2
            + hs
        )
    )


def _mamba_layer_flops(config, gbs, seq_len):
    """Model FLOPs for Mamba layer. We ignore part of the flops of scan because the
    chunk size is not known from model config."""
    hs = config.hidden_size
    mamba_state_dim = config.mamba_state_dim
    mamba_head_dim = config.mamba_head_dim
    mamba_num_groups = config.mamba_num_groups

    if hasattr(config, "mamba_num_heads") and config.mamba_num_heads:
        nheads = config.mamba_num_heads
    else:
        nheads = 2 * hs // mamba_head_dim  # default expand is 2
    d_in = nheads * mamba_head_dim

    return (
        (6 * gbs * seq_len * hs * (2 * d_in + 2 * mamba_num_groups * mamba_state_dim + nheads))
        + (3 * 2 * gbs * seq_len * d_in * mamba_state_dim)
        + (6 * gbs * seq_len * d_in * hs)
    )


def _hybrid_model_flops(config, gbs, seq_len):
    """Model FLOPs for hybrid model"""
    if not config.is_hybrid_model:
        raise ValueError("Config must have is_hybrid_model=True")

    hybrid_override_pattern = config.hybrid_override_pattern
    hs = config.hidden_size
    vocab_size = config.vocab_size

    num_attn_layers, num_mamba_layers, num_mlp_layers = 0, 0, 0
    for c in hybrid_override_pattern:
        if c == "M":
            num_mamba_layers += 1
        elif c == "-":
            num_mlp_layers += 1
        elif c == "*":
            num_attn_layers += 1

    return (
        num_attn_layers * _non_mla_attn_layer_flops(config, gbs, seq_len)
        + num_mamba_layers * _mamba_layer_flops(config, gbs, seq_len)
        + num_mlp_layers * _nemotronh_mlp_layer_flops(config, gbs, seq_len)
        + 6 * gbs * seq_len * hs * vocab_size
    )


def nemotronh_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for NemotronH"""
    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    return _hybrid_model_flops(config, gbs, seq_len)


def attention_flops_calculator(
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    kv_channels: Optional[int] = None,
    is_swa: bool = False,
    swa_window_size: int = 128,
):
    """Calculate the flops for the attention part."""
    kv_channels = kv_channels or (hidden_size // num_attention_heads)

    linear_qkv = seqlen * hidden_size * (kv_channels * (num_attention_heads + num_query_groups * 2))

    linear_proj = seqlen * hidden_size * (kv_channels * num_attention_heads)

    if is_swa:
        attention_mask_nz_elem = (
            swa_window_size * (swa_window_size + 1) / 2 + (seqlen - swa_window_size) * swa_window_size
        )
        attention = num_attention_heads * (attention_mask_nz_elem * kv_channels) * 2
    else:
        bmm_k = kv_channels
        bmm_b = num_attention_heads
        attention_mask_nz_elem = seqlen * (seqlen + 1) / 2
        attention = bmm_b * attention_mask_nz_elem * bmm_k * 2

    return (linear_qkv + linear_proj + attention) * 6


def moe_mlp_flops_calculator(
    seqlen,
    hidden_size,
    moe_ffn_hidden_size,
    moe_router_topk,
    gated_linear_unit: bool = True,
):
    """Calculate the flops for the MLP"""
    total_num_tokens = seqlen * moe_router_topk
    linear_fc1 = total_num_tokens * hidden_size * moe_ffn_hidden_size * (2 if gated_linear_unit else 1)
    linear_fc2 = total_num_tokens * moe_ffn_hidden_size * hidden_size
    return (linear_fc1 + linear_fc2) * 6


def loss_flops_calculator(
    seqlen,
    hidden_size,
    vocab_size,
):
    """Calculate the flops for the loss"""
    return (seqlen * hidden_size * vocab_size) * 6


def gpt_oss_flops_calculator(
    gbs,
    num_layers,
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    moe_ffn_hidden_size,
    moe_router_topk,
    vocab_size,
    kv_channels: Optional[int] = None,
    swa_window_size: int = 128,
    window_attn_skip_freq: Optional[int] = 2,
):
    """Calculate the flops for the GPT-OSS model"""
    flops = 0
    for i in range(num_layers):
        if i % window_attn_skip_freq == 0:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=False,
            )
        else:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=True,
                swa_window_size=swa_window_size,
            )
        flops += moe_mlp_flops_calculator(
            seqlen,
            hidden_size,
            moe_ffn_hidden_size,
            moe_router_topk,
        )
    flops += loss_flops_calculator(seqlen, hidden_size, vocab_size)
    flops *= gbs
    return flops


def gpt_oss_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for GPT-OSS"""
    # Map config fields
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else num_attention_heads
    vocab_size = config.vocab_size

    # GPT-OSS specific fields
    moe_ffn_hidden_size = (
        config.moe_ffn_hidden_size if hasattr(config, "moe_ffn_hidden_size") else config.intermediate_size
    )
    moe_router_topk = config.num_experts_per_tok
    kv_channels = config.kv_channels if hasattr(config, "kv_channels") else (hidden_size // num_attention_heads)
    swa_window_size = config.window_size[0] if hasattr(config, "window_size") and config.window_size else 128
    window_attn_skip_freq = config.window_attn_skip_freq if hasattr(config, "window_attn_skip_freq") else 2

    return gpt_oss_flops_calculator(
        gbs=gbs,
        num_layers=num_layers,
        seqlen=seq_len,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_router_topk=moe_router_topk,
        vocab_size=vocab_size,
        kv_channels=kv_channels,
        swa_window_size=swa_window_size,
        window_attn_skip_freq=window_attn_skip_freq,
    )


def glm4_moe_flops(config, gbs=1, seq_len=None):
    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size

    # GLM4 MoE attention config
    head_dim = getattr(config, "head_dim", hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    # MoE config
    ffn_hs = config.intermediate_size  # for dense layers
    moe_intermediate_size = config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else ffn_hs
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 1
    n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
    first_k_dense_replace = config.first_k_dense_replace if hasattr(config, "first_k_dense_replace") else 0

    causal_self_attn = True
    hidden_size = hs
    gated_linear_multiplier = 2  # SwiGLU

    # Attention flops for GQA (Qwen3-style)
    attention_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + 1)  # QKV gemm
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))  # attention
            + 1  # attention proj gemm
        )
    )

    # MLP flops (DeepSeek V3-style MoE)
    # Dense layers: first_k_dense_replace layers
    dense_mlp_flops = (
        3 * 2 * gbs * first_k_dense_replace * seq_len * hidden_size * (1 + gated_linear_multiplier) * ffn_hs
    )

    # MoE layers: (layers - first_k_dense_replace) layers
    # Each MoE layer has: shared experts + routed experts (topk selected)
    num_moe_layers = layers - first_k_dense_replace

    # Shared expert flops (always computed)
    shared_expert_flops = (
        3
        * 2
        * gbs
        * num_moe_layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_intermediate_size * n_shared_experts)
    )

    # Routed expert flops (topk selected)
    routed_expert_flops = (
        3
        * 2
        * gbs
        * num_moe_layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_intermediate_size * moe_router_topk)
    )

    mlp_flops = dense_mlp_flops + shared_expert_flops + routed_expert_flops

    # Vocab flops
    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def get_flops_formula_for_hf_config(config: Any) -> Optional[Callable]:
    """
    Get the appropriate FLOPs formula function for a given HuggingFace config.

    Args:
        config: HuggingFace model config object

    Returns:
        The appropriate FLOPs formula function, or None if model type is not supported
    """
    # Get config class name
    config_class_name = config.__class__.__name__

    # Map config class names to FLOPs formulas
    class_name_to_formula = {
        # GPT family
        "GPT2Config": gpt3_flops,
        "GPTNeoConfig": gpt3_flops,
        "GPTNeoXConfig": gpt3_flops,
        "GPTJConfig": gpt3_flops,
        # Llama family
        "LlamaConfig": llama2_flops,  # Llama 1 and 2 use same formula
        # Mixtral (MoE)
        "MixtralConfig": mixtral_flops,
        # Qwen family
        "Qwen2Config": qwen3_flops,
        "Qwen3Config": qwen3_flops,
        "Qwen3MoeConfig": qwen3_flops,
        # BERT family
        "BertConfig": bert_flops,
        "RobertaConfig": bert_flops,
        "AlbertConfig": bert_flops,
        "ElectraConfig": bert_flops,
        # DeepSeek V3
        "DeepseekV3Config": deepseekv3_flops,
        # GPT-OSS
        "GptOssConfig": gpt_oss_flops,
        # GLM4 MoE
        "Glm4MoeConfig": glm4_moe_flops,
        # T5 family (encoder-decoder)
        "T5Config": transformer_flops,
        "MT5Config": transformer_flops,
        # Nemotron
        "NemotronConfig": nemotron_flops,
        # General transformer fallback
        "OPTConfig": transformer_flops,
        "BloomConfig": transformer_flops,
        "FalconConfig": transformer_flops,
    }

    # Try exact match first
    formula = class_name_to_formula.get(config_class_name)

    # If no exact match, try to match by model_type as fallback
    if formula is None:
        formula = transformer_flops

    return formula

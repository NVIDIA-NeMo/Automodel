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
import copy

from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids

TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(
    target_config,
    model_args,
):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, "markov_head_type must be provided when markov_rank > 0."

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


def get_gemma4_text_config(target_config):
    """Return a deep copy of a Gemma4 target's text sub-config."""
    assert target_config.model_type in ("gemma4", "gemma4_unified"), (
        f"Gemma4 DSpark expects a Gemma4 target config, got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config
    assert text_config.model_type in ("gemma4_text", "gemma4_unified_text"), (
        f"Gemma4 DSpark expects text_config.model_type 'gemma4_text'/'gemma4_unified_text', got {text_config.model_type!r}."
    )
    return copy.deepcopy(text_config)


def build_gemma4_draft_config(target_config, model_args):
    """Build a Gemma4 DSpark draft config from a Gemma4 target's text sub-config."""
    draft_config = get_gemma4_text_config(target_config)
    full_attention_config = draft_config.per_layer_config["full_attention"]

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(model_args.target_layer_ids, num_target_layers)

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, "markov_head_type must be provided when markov_rank > 0."

    draft_config.architectures = ["Gemma4DSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.global_head_dim = full_attention_config.head_dim
    draft_config.num_global_key_value_heads = full_attention_config.num_key_value_heads
    draft_config.layer_types = layer_types
    draft_config.per_layer_config = {
        layer_idx: {
            "head_dim": full_attention_config.head_dim,
            "num_key_value_heads": full_attention_config.num_key_value_heads,
        }
        for layer_idx in range(num_draft_layers)
    }
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


def build_deepseek_v4_draft_config(target_config, model_args):
    """Build a DeepSeek V4 DSpark draft config from a V4 target config.

    The draft shares the target's frozen ``embed_tokens`` / ``lm_head`` and fuses
    its hidden states, so it keeps the target's attention and embedding dims and
    only shrinks the depth, then adds the DSpark-specific fields. The draft is
    always dense: V4's sparse machinery (DSA ``compress_ratios``), hash routing,
    and MTP are disabled.
    """
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(model_args.target_layer_ids, num_target_layers)

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, "markov_head_type must be provided when markov_rank > 0."

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["DeepseekV4DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    # The draft is always dense: no DSA / hash routing / MTP.
    draft_config.compress_ratios = None
    draft_config.num_hash_layers = 0
    draft_config.num_nextn_predict_layers = 0
    draft_config.tie_word_embeddings = False
    # V4 eager attention consumes a dense additive mask, so the draft uses the
    # SDPA (dense-mask) DFlash path, not flex_attention.
    draft_config._attn_implementation = "sdpa"
    draft_config.block_size = int(model_args.block_size)
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


def get_kimi_k3_text_config(target_config):
    """Return a deep copy of a Kimi K3 target's text configuration."""
    if target_config.model_type == "kimi_linear":
        return copy.deepcopy(target_config)
    assert target_config.model_type == "kimi_k3", (
        f"Kimi K3 DSpark expects model_type 'kimi_k3' or 'kimi_linear', got {target_config.model_type!r}."
    )
    assert target_config.text_config.model_type == "kimi_linear", (
        f"Kimi K3 DSpark expects text_config.model_type 'kimi_linear', got {target_config.text_config.model_type!r}."
    )
    return copy.deepcopy(target_config.text_config)


def build_kimi_k3_draft_config(target_config, model_args):
    """Build a dense MLA DSpark draft config from a Kimi K3 target's text configuration.

    The draft shares the target's frozen ``embed_tokens`` / ``lm_head`` and fuses its
    hidden states, so it keeps the target's MLA and embedding dims and only shrinks the
    depth, then adds the DSpark-specific fields. The draft is always dense: K3's KDA
    linear-attention layers, routed / shared experts, and MTP are all disabled.
    """
    draft_config = get_kimi_k3_text_config(target_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(model_args.target_layer_ids, num_target_layers)

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, "markov_head_type must be provided when markov_rank > 0."

    draft_config.architectures = ["KimiK3DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    # The draft is always dense: no KDA layers, no routed / shared experts, no MTP.
    draft_config.num_experts = None
    draft_config.num_shared_experts = 0
    draft_config.linear_attn_config = {
        "kda_layers": [],
        "full_attn_layers": list(range(1, num_draft_layers + 1)),
    }
    draft_config.num_nextn_predict_layers = 0
    draft_config.tie_word_embeddings = False
    # K3 MLA runs the dense eager attention path over a dense additive mask, so the
    # draft uses the SDPA (dense-mask) DFlash path, not flex_attention.
    draft_config._attn_implementation = "sdpa"
    draft_config.block_size = int(model_args.block_size)
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


def build_glm_5_2_draft_config(target_config, model_args):
    """Build a GLM-5.2 DSpark draft config from a GLM-5.2 (glm_moe_dsa) target config.

    The draft shares the target's frozen ``embed_tokens`` / ``lm_head`` and fuses its
    hidden states, so it keeps the target's MLA and embedding dims and only shrinks the
    depth, then adds the DSpark-specific fields. The draft is always dense: GLM's DSA
    sparse machinery (the top-k indexer + IndexShare layer sharing) and its MoE are
    disabled -- every draft layer is a plain dense SwiGLU block with a dense MLA.
    """
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(model_args.target_layer_ids, num_target_layers)

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, "markov_head_type must be provided when markov_rank > 0."

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Glm5_2DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    # The draft is always dense: no MoE, no DSA indexer / IndexShare, no top-k sharing.
    # (GLM-5.2 has no MTP module, so there is nothing MTP-related to disable.)
    draft_config.mlp_layer_types = ["dense"] * num_draft_layers
    draft_config.indexer_types = ["full"] * num_draft_layers
    draft_config.tie_word_embeddings = False
    # GLM MLA runs the dense eager attention path over a dense additive mask, so the
    # draft uses the SDPA (dense-mask) DFlash path, not flex_attention.
    draft_config._attn_implementation = "sdpa"
    draft_config.block_size = int(model_args.block_size)
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


def get_minimax_m3_text_config(target_config):
    """Return a deep copy of a MiniMax M3 VL target's text sub-config."""
    assert target_config.model_type == "minimax_m3_vl", (
        f"MiniMax M3 DSpark expects a MiniMax M3 VL target config, got model_type={target_config.model_type!r}."
    )
    text_config = target_config.text_config
    assert text_config.model_type == "minimax_m3", (
        f"MiniMax M3 DSpark expects text_config.model_type 'minimax_m3', got {text_config.model_type!r}."
    )
    return copy.deepcopy(text_config)


def build_minimax_m3_draft_config(target_config, model_args):
    """Build a MiniMax M3 DSpark draft config from a MiniMax M3 VL target's text sub-config."""
    draft_config = get_minimax_m3_text_config(target_config)

    num_target_layers = int(draft_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(model_args.target_layer_ids, num_target_layers)

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, "markov_head_type must be provided when markov_rank > 0."

    draft_config.architectures = ["MiniMaxM3DSparkModel"]
    draft_config.target_model_type = str(target_config.model_type)
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    # The draft is always dense: no block-sparse indexer, no MoE, no MTP.
    draft_config.num_mtp_modules = 0
    draft_config.tie_word_embeddings = False
    draft_config = type(draft_config).from_dict(draft_config.to_dict())
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.block_size = int(model_args.block_size)
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(model_args.confidence_head_with_markov)
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
    "build_gemma4_draft_config",
    "build_deepseek_v4_draft_config",
    "build_glm_5_2_draft_config",
    "build_kimi_k3_draft_config",
    "get_gemma4_text_config",
    "get_kimi_k3_text_config",
    "build_minimax_m3_draft_config",
    "get_minimax_m3_text_config",
]

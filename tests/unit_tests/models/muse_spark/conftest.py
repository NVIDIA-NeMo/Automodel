# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small Muse Spark Open fixtures that preserve every architectural feature."""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM


def tiny_muse_spark_config() -> MuseSpark1xOmniConfig:
    """Build a four-layer, four-expert Muse configuration for CPU tests."""
    return MuseSpark1xOmniConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        global_attn_cfg=[8, 8, 8, 0],
        ffn_dim_multiplier=2.0,
        moe={
            "is_enabled": True,
            "topk": 2,
            "num_experts": 4,
            "use_shared_expert": False,
            "router_in_fp32": True,
            "route_scale": 1.0,
            "override_routed_hidden_dim": 8,
            "interleave_moe_layer_step": 1,
            "n_dense_layers_before_moe": 0,
            "router_score_normalization": "sigmoid",
            "router_score_division_eps": 1e-15,
            "compression_factor": 2,
            "use_pre_expert_norm": True,
            "use_post_expert_norm": True,
        },
        metap={
            "use_metap": True,
            "metap_mode": "sp",
            "base_width": 4.0,
            "router_has_one_inf_dim": True,
            "m_emb": 1.0,
        },
        vision_config={
            "downsample_factor": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "projector_input_dim": 16,
            "navit_config": {
                "downsample_factor": 2,
                "grid_sample_mode": "bilinear",
                "hidden_size": 8,
                "image_size": 4,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_channels": 3,
                "num_hidden_layers": 2,
                "patch_size": 2,
                "patch_temporal": 2,
                "rope_theta": 10_000.0,
                "sparse_attention_factor": 2,
                "use_rope": True,
                "vision_output_dim": 32,
            },
        },
        use_audio_perception=False,
        image_token_id=61,
        patch_token_id=62,
        video_token_id=63,
        torch_dtype="float32",
    )


def tiny_backend(*, adapter: bool = True) -> BackendConfig:
    """Use portable PyTorch implementations for CPU unit tests."""
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=adapter,
    )


def tiny_muse_spark_model() -> MuseSpark1xOmniForCausalLM:
    """Construct and initialize the tiny CPU model."""
    model = MuseSpark1xOmniForCausalLM(tiny_muse_spark_config(), backend=tiny_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model

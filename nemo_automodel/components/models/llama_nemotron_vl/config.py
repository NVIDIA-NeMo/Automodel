# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

from typing import Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig


class LlamaBidirectionalConfig(LlamaConfig):
    """Configuration for bidirectional (non-causal) LLaMA model."""

    model_type = "llama_bidirec"

    def __init__(
        self,
        pooling="avg",
        temperature=1.0,
        **kwargs,
    ):
        self.pooling = pooling
        self.temperature = temperature
        super().__init__(
            **kwargs,
        )


class LlamaNemotronVLConfig(PretrainedConfig):
    """
    Base configuration for vision-language models combining vision and language components.
    This serves as the foundation for LlamaNemotronVL configurations.
    """

    model_type = "llama_nemotron_vl"
    is_composition = True
    # is_composition was renamed to has_no_defaults_at_init in transformers 4.52.1
    # In PR https://github.com/huggingface/transformers/pull/36263
    has_no_defaults_at_init = True
    # Declare sub-configs so transformers can propagate per-backbone attn_implementation
    # e.g. from_pretrained(attn_implementation={"vision_config": "sdpa", "llm_config": "flash_attention_2"})
    sub_configs = {"vision_config": SiglipVisionConfig, "llm_config": LlamaBidirectionalConfig}

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        mlp_checkpoint=True,
        pre_feature_reduction=False,
        keep_aspect_ratio=False,
        vocab_size=-1,
        q_max_length: Optional[int] = 512,
        p_max_length: Optional[int] = 10240,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
        pooling: str = "last",
        bidirectional_attention: bool = False,
        max_input_tiles: int = 2,
        img_context_token_id: int = 128258,  # tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        **kwargs,
    ):
        if vision_config is not None:
            if vision_config["model_type"] == "siglip_vision_model":
                self.vision_config = SiglipVisionConfig(**vision_config)
            else:
                raise ValueError("Unsupported model_type: {}".format(vision_config["model_type"]))

        if llm_config is not None:
            if llm_config["architectures"][0] in {
                "LlamaBidirectionalModel",
                "LlamaBidirectionalForSequenceClassification",
            }:
                self.llm_config = LlamaBidirectionalConfig(**llm_config)
            else:
                raise ValueError("Unsupported architecture: {}".format(llm_config["architectures"][0]))
            self.vocab_size = self.llm_config.vocab_size
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.mlp_checkpoint = mlp_checkpoint
        self.pre_feature_reduction = pre_feature_reduction
        self.keep_aspect_ratio = keep_aspect_ratio

        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.pooling = pooling
        self.bidirectional_attention = bidirectional_attention
        self.img_context_token_id = img_context_token_id
        self.max_input_tiles = max_input_tiles
        super().__init__(**kwargs)


__all__ = [
    "LlamaBidirectionalConfig",
    "LlamaNemotronVLConfig",
]

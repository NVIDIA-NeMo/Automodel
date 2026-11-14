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
#
# Field names and defaults are adapted from Hugging Face Transformers (Apache-2.0),
# https://github.com/huggingface/transformers at v5.23.0
# (52b080b6416a7897e2af90febeb4dfad22b84d19):
#   src/transformers/models/lfm3/configuration_lfm3.py
#   src/transformers/models/lfm3_vl/configuration_lfm3_vl.py
# Copyright 2026 The HuggingFace Team. All rights reserved.

"""LFM3 and LFM3-VL configurations for Transformers releases that predate them."""

from __future__ import annotations

from huggingface_hub.dataclasses import strict
from transformers.configuration_utils import PreTrainedConfig

# Extra `config.json` entries that Transformers forwards to `__post_init__`.
_JsonValue = str | int | float | bool | list | dict | None


@strict
class Lfm3Config(PreTrainedConfig):
    """Configure the LFM3 decoder of gated short convolutions and grouped-query attention.

    ``layer_types`` names each layer ``"conv"``, ``"full_attention"`` or
    ``"sliding_attention"``. ``rope_parameters`` holds one entry per attention
    type. Full-attention layers rotate only the first ``partial_rotary_factor``
    share of each head and leave the rest unrotated.

    Released checkpoints label that partial rotation ``"default"``, which
    Transformers 5.23 reads as a full rotation. Such an entry is relabelled
    ``"proportional"`` here, the name under which Transformers and the
    publisher's code both build the trained frequencies, so saved configs load
    correctly in either.
    """

    model_type = "lfm3"
    keys_to_ignore_at_inference = ["past_key_values"]
    # The Qwen3 attention block this model extends reads `rms_norm_eps`.
    attribute_map = {"rms_norm_eps": "norm_eps"}
    default_theta = 1_000_000.0
    _rope_type_labels = ("full_attention", "sliding_attention")

    vocab_size: int = 128_000
    hidden_size: int = 2048
    intermediate_size: int = 7680
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    head_dim: int = 128
    max_position_embeddings: int = 128_000
    initializer_range: float = 0.02
    norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int | None = 124_893
    bos_token_id: int | None = 124_894
    eos_token_id: int | list[int] | None = 124_900
    tie_word_embeddings: bool = True
    rope_parameters: dict | None = None
    conv_bias: bool = False
    conv_L_cache: int = 7
    layer_types: list[str] | None = None
    sliding_window: int | None = 256
    use_sandwich_norm: bool = True
    use_attention_gate: bool = True
    attention_bias: bool = False
    attention_dropout: float = 0.0

    def __post_init__(self, **kwargs: _JsonValue) -> None:
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if i % 4 == 2 else "sliding_attention" if i % 4 == 0 and i > 0 else "conv"
                for i in range(self.num_hidden_layers)
            ]
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            }
        attention_types = [layer_type for layer_type in self._rope_type_labels if layer_type in self.layer_types]
        if missing := [layer_type for layer_type in attention_types if layer_type not in self.rope_parameters]:
            raise ValueError(f"`rope_parameters` has no entry for the {missing} layers in `layer_types`.")
        relabelled = {
            layer_type: {**self.rope_parameters[layer_type], "rope_type": "proportional"}
            for layer_type in attention_types
            if self.rope_parameters[layer_type].get("rope_type", "default") == "default"
            and self.rope_parameters[layer_type].get("partial_rotary_factor", 1.0) < 1.0
        }
        self.rope_parameters = {**self.rope_parameters, **relabelled}
        super().__post_init__(**kwargs)
        if unsupported := {
            layer_type: self.rope_parameters[layer_type]["rope_type"]
            for layer_type in attention_types
            if self.rope_parameters[layer_type]["rope_type"] not in ("default", "proportional")
        }:
            raise ValueError(f"LFM3 supports `default` and `proportional` RoPE only, got {unsupported}.")


@strict
class Lfm3VlVisionConfig(PreTrainedConfig):
    """Configure the LFM3-VL vision encoder of gated 2D convolutions and bidirectional attention.

    ``layer_types`` names each layer ``"conv"`` or ``"full_attention"``. Patches
    are ``patch_size`` pixels square and ``temporal_patch_size`` frames deep;
    the projector later merges each ``spatial_merge_size`` square of them.
    """

    model_type = "lfm3_vl_vision"
    base_config_key = "vision_config"
    default_rope_type = "axial"
    default_theta = 1_000_000.0

    hidden_size: int = 1024
    intermediate_size: int = 2560
    num_hidden_layers: int = 14
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    in_channels: int = 3
    patch_size: int = 16
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    conv_kernel_size: int = 3
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    rope_parameters: dict | None = None
    layer_types: list[str] | None = None

    def __post_init__(self, **kwargs: _JsonValue) -> None:
        if self.layer_types is None:
            self.layer_types = ["conv" if i < 2 or i % 2 else "full_attention" for i in range(self.num_hidden_layers)]
        super().__post_init__(**kwargs)


@strict
class Lfm3VlConfig(PreTrainedConfig):
    """Configure LFM3-VL: an LFM3 decoder fed by the LFM3-VL vision encoder through an MLP projector."""

    model_type = "lfm3_vl"
    sub_configs = {"text_config": Lfm3Config, "vision_config": Lfm3VlVisionConfig}

    vision_config: dict | PreTrainedConfig | None = None
    text_config: dict | PreTrainedConfig | None = None
    image_token_id: int = 125_030
    video_token_id: int = 125_029
    vision_start_token_id: int = 125_009
    vision_end_token_id: int = 125_010
    projector_hidden_act: str = "gelu"
    projector_hidden_size: int = 4096
    projector_bias: bool = True
    tie_word_embeddings: bool = True

    def __post_init__(self, **kwargs: _JsonValue) -> None:
        if not isinstance(self.text_config, PreTrainedConfig):
            self.text_config = Lfm3Config(**(self.text_config or {}))
        if not isinstance(self.vision_config, PreTrainedConfig):
            self.vision_config = Lfm3VlVisionConfig(**(self.vision_config or {}))
        super().__post_init__(**kwargs)

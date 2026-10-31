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

"""Serializable configuration for the released D1 Omni composition."""

from __future__ import annotations

from typing import Any

from transformers import Lfm2Config, PretrainedConfig


class D1OmniConfig(PretrainedConfig):
    """Configure the Omni encoder, marker head, and perception modules."""

    model_type = "d1_omni"
    sub_configs = {"text_config": Lfm2Config}

    def __init__(
        self,
        text_config: dict[str, Any] | Lfm2Config | None = None,
        vision_config: dict[str, Any] | None = None,
        audio_config: dict[str, Any] | None = None,
        projector_hidden_size: int = 2048,
        head_layers: int = 2,
        max_length: int = 16384,
        image_text_length: int = 896,
        audio_text_length: int = 15360,
        temperatures: dict[str, float] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("architectures", ["D1OmniModel"])
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.text_config = text_config if isinstance(text_config, Lfm2Config) else Lfm2Config(**(text_config or {}))
        self.vision_config = dict(vision_config or {})
        self.audio_config = dict(audio_config or {})
        self.projector_hidden_size = projector_hidden_size
        self.head_layers = head_layers
        self.max_length = max_length
        self.image_text_length = image_text_length
        self.audio_text_length = audio_text_length
        self.temperatures = dict(temperatures or {})
        if min(projector_hidden_size, head_layers, max_length, image_text_length, audio_text_length) <= 0:
            raise ValueError("projection width, layer count, and sequence limits must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize HF subconfig fields and the release's legacy rotary key.

        Returns:
            Configuration mapping reloadable by native and published Omni classes.
        """
        result = super().to_dict()
        result["text_config"]["rope_theta"] = self.text_config.rope_parameters["rope_theta"]
        return result

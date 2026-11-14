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

"""Configuration-only Qwen2.5-VL positions for CPU preprocessing workers."""

from dataclasses import dataclass

import torch
from transformers import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel


@dataclass(frozen=True, slots=True)
class Qwen2_5_VLPositionState:
    """Carry the HF configuration, without a model or parameters, into workers.

    Position mathematics stays owned by the installed HF model implementation.
    This state implements precisely the configuration/vision-position subset its
    public ``get_rope_index`` method consumes.
    """

    config: Qwen2_5_VLConfig

    get_vision_position_ids = Qwen2_5_VLModel.get_vision_position_ids

    def positions(
        self,
        input_ids: torch.Tensor,
        *,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute image/text positions with HF's own RoPE implementation.

        Args:
            input_ids: Unshifted token IDs [sequence].
            mm_token_type_ids: Processor modality IDs [1, sequence], with text=0
                and images=1. Video inputs are outside this image-only path.
            image_grid_thw: Image grids [images, 3] in temporal/height/width order.
            attention_mask: Processor attention eligibility [sequence], excluding
                injected fake media independently of trailing padding.

        Returns:
            Unshifted integer mRoPE positions [3, sequence], without changing any
            input tensor. No model instance or vision forward is required.
        """
        positions, _ = Qwen2_5_VLModel.get_rope_index(
            self,
            input_ids.unsqueeze(0),
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask.unsqueeze(0),
        )
        return positions[:, 0]

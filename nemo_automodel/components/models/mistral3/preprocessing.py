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

"""Configuration-only positions for the Mistral3 image path."""

from dataclasses import dataclass

import torch
from transformers.configuration_utils import PreTrainedConfig


@dataclass(frozen=True)
class Mistral3PositionState:
    """Keep processor expansion and scalar text positions model-owned."""

    config: PreTrainedConfig

    def __post_init__(self) -> None:
        text = self.config.get_text_config()
        if text.model_type not in ("mistral", "ministral3") or getattr(text, "sliding_window", None) is not None:
            raise ValueError(
                "Zephon Mistral3 images require a Mistral/Ministral3 text backbone without sliding attention"
            )

    def positions(
        self,
        input_ids: torch.Tensor,
        *,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return native scalar offsets over fully expanded input tokens.

        Args:
            input_ids: Unshifted tokens [sequence], including expanded image slots.
            mm_token_type_ids: Processor modality IDs [1, sequence]; unused.
            image_grid_thw: Optional image grids; unused by the Pixtral processor.
            attention_mask: Processor eligibility [sequence]; exclusions keep offsets.

        Returns:
            Integer positions [sequence], without input mutation or a model instance.
        """
        return torch.arange(input_ids.numel(), dtype=torch.long, device=input_ids.device)

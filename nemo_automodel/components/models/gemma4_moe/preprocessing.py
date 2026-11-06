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

"""Configuration-only image preprocessing state for dense Gemma4."""

from dataclasses import dataclass

import torch
from transformers.configuration_utils import PreTrainedConfig


@dataclass(frozen=True)
class Gemma4PositionState:
    """Carry dense31B configuration into workers without constructing a model.

    Image patch coordinates come from the processor's ``image_position_ids``;
    text decoder positions are scalar offsets over the expanded token sequence.
    Audio, MoE, per-layer embeddings and shared-KV variants are unsupported.
    """

    config: PreTrainedConfig

    def __post_init__(self) -> None:
        text = self.config.get_text_config()
        if text.use_bidirectional_attention not in (None, "vision"):
            raise ValueError("Zephon Gemma4 image preprocessing requires causal text attention")
        if (
            self.config.audio_config is not None
            or text.enable_moe_block
            or text.hidden_size_per_layer_input
            or text.num_kv_shared_layers
        ):
            raise ValueError(
                "Zephon Gemma4 image preprocessing supports dense31B configurations without audio, PLE or KV sharing"
            )

    def positions(
        self,
        input_ids: torch.Tensor,
        *,
        mm_token_type_ids: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return scalar positions over one expanded, unshifted document.

        Args:
            input_ids: Token IDs [sequence], including expanded image tokens.
            mm_token_type_ids: Processor modality IDs [1, sequence]; unused.
            image_grid_thw: Optional processor image grids; unused by Gemma4.
            attention_mask: Eligibility [sequence]; exclusions do not renumber positions.

        Returns:
            Integer positions [sequence], matching the native model default.
        """
        return torch.arange(input_ids.numel(), dtype=torch.long, device=input_ids.device)

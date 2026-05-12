# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Model-specific strategies for diffusion LLM (dLLM) training.

Each strategy encapsulates three variation points that differ across dLLM
model families:

1. **Loss function creation** — which loss module to use.
2. **Corruption** — how tokens are corrupted before the forward pass.
3. **Batch preparation** — how the batch dict is shaped for the model's
   forward signature.
4. **Normalization mode** — whether loss is normalised by the total
   supervised token count (``"supervised"``) or by the actually-corrupted
   token count (``"noise"``).

To add a new dLLM variant (e.g., LLADA), implement a new
:class:`DLLMStrategy` subclass and register it in :data:`DLLM_STRATEGIES`.
No changes to the recipe are required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.dllm.corruption import (
    corrupt_uniform,
)
from nemo_automodel.components.loss.dllm_loss import (
    MDLMCrossEntropyLoss,
)


class DLLMStrategy(ABC):
    """Abstract base for dLLM model strategies."""

    @property
    def normalization_mode(self) -> str:
        """Token count used as the loss denominator: ``"supervised"`` or ``"noise"``.

        * ``"supervised"`` — total ``loss_mask == 1`` positions (default).
        * ``"noise"`` — actually-corrupted positions (``noise_mask == True``).

        Also governs gradient-norm scaling so loss and gradients stay consistent.
        """
        return "supervised"

    @abstractmethod
    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        """Return the loss module for this model type."""

    @abstractmethod
    def apply_corruption(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        mask_token_id: int,
        *,
        eps: float,
        block_size: Optional[int],
        half_life_ratio: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(noisy_input_ids, noise_mask, p_mask)``."""

    @abstractmethod
    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_input_ids: torch.Tensor,
        noise_mask: torch.Tensor,
        clean_input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Mutate *batch* in-place for the model's forward pass and return it."""


class MDLMStrategy(DLLMStrategy):
    """Strategy for MDLM models.

    - Loss: :class:`MDLMCrossEntropyLoss`
    - Corruption: always uniform (``corrupt_uniform``)
    - Batch: model receives noisy (corrupted) tokens as ``input_ids``
    """

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        return MDLMCrossEntropyLoss()

    def apply_corruption(self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio):
        return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        batch["input_ids"] = noisy_input_ids
        batch.pop("attention_mask", None)  # MDLM models are bidirectional
        return batch


class DFlashStrategy(MDLMStrategy):
    """Placeholder strategy for DFlash mode.

    The actual DFlash training logic (dual-model forward, anchor masking,
    decay loss) lives in :class:`~nemo_automodel.recipes.dllm.train_dflash.DFlashSFTRecipe`.
    This entry exists only so ``DiffusionLMSFTRecipe.setup()`` accepts
    ``dllm.mode: dflash`` without raising an unknown-mode error.
    """


DLLM_STRATEGIES: Dict[str, type] = {
    "mdlm": MDLMStrategy,
    "dflash": DFlashStrategy,
}


def get_dllm_strategy(mode: str) -> DLLMStrategy:
    """Look up and instantiate a dLLM strategy by mode name.

    Raises:
        ValueError: If *mode* is not registered in :data:`DLLM_STRATEGIES`.
    """
    cls = DLLM_STRATEGIES.get(mode)
    if cls is None:
        raise ValueError(f"Unknown dllm.mode: {mode!r}. Available: {sorted(DLLM_STRATEGIES)}")
    return cls()

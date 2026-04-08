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

"""NEFTune: Noisy Embeddings Fine-Tuning.

Implements the technique from "NEFTune: Noisy Embeddings Improve Instruction Finetuning"
(https://arxiv.org/abs/2310.05914). Adds scaled uniform noise to token embeddings during
training to improve generalization, with no additional compute or data overhead.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class NEFTune:
    """Applies NEFTune noise to a model's embedding layer during training.

    NEFTune adds uniform random noise scaled by ``alpha / sqrt(seq_len * hidden_dim)``
    to the embedding output. The noise is only applied when the model is in training mode.

    Args:
        noise_alpha: Noise magnitude. Higher values add more noise. Typical values
            are 5-15. Set to 0 to disable.

    Example::

        neftune = NEFTune(noise_alpha=5.0)
        neftune.activate(model)
        # ... training loop ...
        neftune.deactivate(model)
    """

    def __init__(self, noise_alpha: float = 5.0):
        if noise_alpha < 0:
            raise ValueError(f"noise_alpha must be non-negative, got {noise_alpha}")
        self.noise_alpha = noise_alpha
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._original_forward = None

    def _neftune_forward_hook(self, module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
        """Forward hook that adds NEFTune noise to embedding output during training."""
        if module.training and self.noise_alpha > 0:
            dims = torch.tensor(output.size(1) * output.size(2), dtype=output.dtype, device=output.device)
            mag = self.noise_alpha / torch.sqrt(dims)
            output = output + torch.zeros_like(output).uniform_(-mag.item(), mag.item())
        return output

    def activate(self, model: nn.Module) -> None:
        """Attach NEFTune noise hook to the model's input embedding layer.

        Args:
            model: The model whose embeddings will be augmented with noise.

        Raises:
            RuntimeError: If NEFTune is already active on this model.
            ValueError: If the model has no recognizable embedding layer.
        """
        if self._hook_handle is not None:
            raise RuntimeError("NEFTune is already active. Call deactivate() first.")

        embeddings = _get_input_embeddings(model)
        if embeddings is None:
            raise ValueError(
                "Could not find input embeddings on the model. "
                "Expected get_input_embeddings() method or model.embed_tokens / model.model.embed_tokens attribute."
            )

        self._hook_handle = embeddings.register_forward_hook(self._neftune_forward_hook)
        logger.info("NEFTune activated with noise_alpha=%.2f", self.noise_alpha)

    def deactivate(self, model: nn.Module) -> None:
        """Remove the NEFTune noise hook from the model.

        Safe to call even if NEFTune is not active (no-op in that case).

        Args:
            model: The model to deactivate NEFTune on.
        """
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
            logger.info("NEFTune deactivated")

    @property
    def is_active(self) -> bool:
        """Whether NEFTune noise is currently being applied."""
        return self._hook_handle is not None


def _get_input_embeddings(model: nn.Module) -> Optional[nn.Module]:
    """Find the input embedding layer on a model.

    Checks for ``get_input_embeddings()`` method first (HF models),
    then falls back to common attribute names.

    Args:
        model: The model to search.

    Returns:
        The embedding module, or None if not found.
    """
    if hasattr(model, "get_input_embeddings") and callable(model.get_input_embeddings):
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb

    for attr_path in ["embed_tokens", "model.embed_tokens", "transformer.wte"]:
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.Module):
                return obj
        except AttributeError:
            continue

    return None

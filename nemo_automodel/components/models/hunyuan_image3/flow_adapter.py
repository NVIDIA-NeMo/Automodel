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

"""Flow-matching adapter for HunyuanImage-3.0 text-to-image training.

HunyuanImage-3.0 has no separate text encoder: the prompt is part of the transformer's own token sequence. The
preprocessing step stores, per sample, the token ids before the image (``<bos> prompt <boi> <img_size_*>
<img_ratio_*> <timestep>``), the matching unconditional ids used for classifier-free guidance (the prompt replaced
by ``<cfg>`` tokens of the same length), and the ids after the image (``<eoi>``). This adapter assembles
``prefix + <img> * (h*w) + suffix`` for every sample, right-pads the batch and calls the model.

The pipeline's convention already matches the release: ``x_t = (1 - sigma) x_0 + sigma * noise``, target
``noise - x_0`` and model timestep ``sigma * 1000``.
"""

from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext, ModelAdapter

PROMPT_IDS_KEY = "prompt_input_ids"
UNCOND_PROMPT_IDS_KEY = "uncond_prompt_input_ids"
SUFFIX_IDS_KEY = "prompt_suffix_ids"


class HunyuanImage3Adapter(ModelAdapter):
    """Builds the joint token sequence around the noisy latents and returns the predicted velocity.

    Args:
        image_token_id: Id of the ``<img>`` placeholder token.
        pad_token_id: Id used to right-pad sequences of different lengths.
    """

    def __init__(self, image_token_id: int = 128006, pad_token_id: int = 128009):
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id

    def prepare_inputs(self, context: FlowMatchingContext) -> dict[str, Any]:
        """Assemble model inputs.

        Args:
            context: ``noisy_latents`` of shape [batch, channels, height, width], ``timesteps`` of shape [batch]
                (``sigma * 1000``) and a batch holding per-sample 1D long tensors under ``prompt_input_ids``,
                ``uncond_prompt_input_ids`` and ``prompt_suffix_ids``.

        Returns:
            ``input_ids`` long [batch, sequence] right-padded with ``pad_token_id``, ``latents`` [batch, channels,
            height, width], fp32 ``timestep`` [batch] and long ``valid_lengths`` [batch].
        """
        batch = context.batch
        noisy = context.noisy_latents
        if noisy.ndim != 4:
            raise ValueError(f"HunyuanImage3Adapter expects 4D latents [B, C, H, W], got {noisy.ndim}D")
        for key in (PROMPT_IDS_KEY, UNCOND_PROMPT_IDS_KEY, SUFFIX_IDS_KEY):
            if key not in batch:
                raise KeyError(f"Batch is missing {key!r}; preprocess the data with the 'hunyuan_image3' processor.")
        batch_size, _, height, width = noisy.shape
        num_image = height * width
        device = context.device
        image_ids = torch.full((num_image,), self.image_token_id, dtype=torch.long)

        rows = []
        for i in range(batch_size):
            drop = context.cfg_dropout_prob > 0 and random.random() < context.cfg_dropout_prob
            prefix = batch[UNCOND_PROMPT_IDS_KEY if drop else PROMPT_IDS_KEY][i]
            rows.append(torch.cat([prefix.long().cpu(), image_ids, batch[SUFFIX_IDS_KEY][i].long().cpu()]))
        valid_lengths = torch.tensor([len(row) for row in rows], dtype=torch.long)
        input_ids = torch.full((batch_size, int(valid_lengths.max())), self.pad_token_id, dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = row

        return {
            "input_ids": input_ids.to(device, non_blocking=True),
            "latents": noisy.to(device, dtype=context.dtype),
            "timestep": context.timesteps.to(device, dtype=torch.float32),
            "valid_lengths": valid_lengths.to(device, non_blocking=True),
        }

    def forward(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        """Run the model on ``prepare_inputs`` output.

        Returns:
            Tensor of shape [batch, channels, height, width]: the predicted velocity ``noise - x0``.
        """
        return self.post_process_prediction(model(**inputs, return_dict=False))

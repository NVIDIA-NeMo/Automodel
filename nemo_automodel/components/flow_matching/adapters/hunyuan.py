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

"""
HunyuanVideo model adapter for FlowMatching Pipeline.

This adapter supports HunyuanVideo 1.5 style models with dual text encoders
and image embeddings for image-to-video conditioning.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .base import FlowMatchingContext, ModelAdapter


class HunyuanAdapter(ModelAdapter):
    """
    Model adapter for HunyuanVideo 1.5 style models.

    These models use:
    - Condition latents concatenated with noisy latents
    - Dual text encoders with attention masks
    - Image embeddings for i2v

    Expected batch keys:
    - text_embeddings: Primary text encoder output [B, seq_len, dim]
    - text_mask: Attention mask for primary encoder [B, seq_len] (optional)
    - text_embeddings_2: Secondary text encoder output [B, seq_len, dim] (optional)
    - text_mask_2: Attention mask for secondary encoder [B, seq_len] (optional)
    - image_embeds: Image embeddings for i2v [B, seq_len, dim] (optional)

    Example:
        adapter = HunyuanAdapter()
        pipeline = FlowMatchingPipelineV2(model_adapter=adapter)
    """

    def __init__(
        self,
        default_image_embed_shape: Tuple[int, int] = (729, 1152),
        use_condition_latents: bool = True,
    ):
        """
        Initialize the HunyuanAdapter.

        Args:
            default_image_embed_shape: Default shape for image embeddings (seq_len, dim)
                when not provided in batch. Defaults to (729, 1152).
            use_condition_latents: Whether to concatenate condition latents with
                noisy latents. Defaults to True.
        """
        self.default_image_embed_shape = default_image_embed_shape
        self.use_condition_latents = use_condition_latents

    def get_condition_latents(self, latents: torch.Tensor, task_type: str) -> torch.Tensor:
        """
        Generate conditional latents based on task type.

        Args:
            latents: Input latents [B, C, F, H, W]
            task_type: Task type ("t2v" or "i2v")

        Returns:
            Conditional latents [B, C+1, F, H, W]
        """
        b, c, f, h, w = latents.shape
        cond = torch.zeros([b, c + 1, f, h, w], device=latents.device, dtype=latents.dtype)

        if task_type == "t2v":
            return cond
        elif task_type == "i2v":
            cond[:, :-1, :1] = latents[:, :, :1]
            cond[:, -1, 0] = 1
            return cond
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        """
        Prepare inputs for HunyuanVideo model.

        Args:
            context: FlowMatchingContext with batch data

        Returns:
            Dictionary containing:
            - latents: Noisy latents (optionally concatenated with condition latents)
            - timesteps: Timestep values
            - encoder_hidden_states: Primary text embeddings
            - encoder_attention_mask: Primary attention mask
            - encoder_hidden_states_2: Secondary text embeddings
            - encoder_attention_mask_2: Secondary attention mask
            - image_embeds: Image embeddings
        """
        batch = context.batch
        batch_size = context.noisy_latents.shape[0]
        device = context.device
        dtype = context.dtype

        # Get text embeddings
        text_embeddings = batch["text_embeddings"].to(device, dtype=dtype)
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)

        # Get optional elements
        text_mask = batch.get("text_mask")
        text_embeddings_2 = batch.get("text_embeddings_2")
        text_mask_2 = batch.get("text_mask_2")

        if text_mask is not None:
            text_mask = text_mask.to(device, dtype=dtype)
        if text_embeddings_2 is not None:
            text_embeddings_2 = text_embeddings_2.to(device, dtype=dtype)
        if text_mask_2 is not None:
            text_mask_2 = text_mask_2.to(device, dtype=dtype)

        # Handle image embeds for i2v
        if context.task_type == "i2v" and "image_embeds" in batch:
            image_embeds = batch["image_embeds"].to(device, dtype=dtype)
        else:
            seq_len, dim = self.default_image_embed_shape
            image_embeds = torch.zeros(
                batch_size,
                seq_len,
                dim,
                dtype=dtype,
                device=device,
            )

        # Prepare latents (with or without condition)
        if self.use_condition_latents:
            cond_latents = self.get_condition_latents(context.latents, context.task_type)
            latents = torch.cat([context.noisy_latents, cond_latents], dim=1)
        else:
            latents = context.noisy_latents

        return {
            "latents": latents,
            "timesteps": context.timesteps.to(dtype),
            "encoder_hidden_states": text_embeddings,
            "encoder_attention_mask": text_mask,
            "encoder_hidden_states_2": text_embeddings_2,
            "encoder_attention_mask_2": text_mask_2,
            "image_embeds": image_embeds,
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Execute forward pass for HunyuanVideo model.

        Args:
            model: HunyuanVideo model
            inputs: Dictionary from prepare_inputs()

        Returns:
            Model prediction tensor
        """
        model_pred = model(
            inputs["latents"],
            inputs["timesteps"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_attention_mask=inputs["encoder_attention_mask"],
            encoder_hidden_states_2=inputs["encoder_hidden_states_2"],
            encoder_attention_mask_2=inputs["encoder_attention_mask_2"],
            image_embeds=inputs["image_embeds"],
            return_dict=False,
        )
        return self.post_process_prediction(model_pred)

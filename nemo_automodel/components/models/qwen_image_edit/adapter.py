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

"""Flow-matching adapter for cached Qwen-Image-Edit-2511 conditioning."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext, ModelAdapter


class QwenImageEditAdapter(ModelAdapter):
    """Adapt cached image-edit batches to the upstream Diffusers transformer.

    Qwen-Image-Edit concatenates the packed noisy target tokens followed by
    the packed context-image tokens. The transformer predicts every image
    token, but flow matching supervises only the leading target-token span.
    """

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack 2-by-2 latent patches in the upstream Diffusers layout.

        Args:
            latents: Tensor of shape [batch, channels, latent_height,
                latent_width]. Both spatial dimensions must be even.

        Returns:
            Tensor of shape [batch, packed_tokens, 4 * channels], where
            ``packed_tokens = latent_height * latent_width / 4`` and the final
            axis stores each channels-first 2-by-2 patch.
        """
        if latents.ndim != 4:
            raise ValueError(
                "QwenImageEditAdapter expects latent tensors with shape [batch, channels, height, width], "
                f"got {tuple(latents.shape)}"
            )

        batch_size, channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(
                "Qwen-Image-Edit latent height and width must be divisible by 2 before patch packing, "
                f"got height={height}, width={width}"
            )

        return (
            latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(batch_size, (height // 2) * (width // 2), channels * 4)
        )

    @staticmethod
    def _unpack_latents(packed_latents: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
        """Restore packed predictions to a channels-first latent grid.

        Args:
            packed_latents: Tensor of shape [batch, packed_tokens,
                4 * channels] in the upstream Diffusers 2-by-2 patch layout.
            height: Output latent height. It must be even.
            width: Output latent width. It must be even.

        Returns:
            Tensor of shape [batch, channels, height, width]. The returned
            tensor does not alias ``packed_latents``.
        """
        if packed_latents.ndim != 3:
            raise ValueError(
                "QwenImageEditAdapter expects packed predictions with shape [batch, tokens, channels], "
                f"got {tuple(packed_latents.shape)}"
            )
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Output latent height and width must be divisible by 2, got {(height, width)}")

        batch_size, token_count, packed_channels = packed_latents.shape
        expected_token_count = (height // 2) * (width // 2)
        if token_count != expected_token_count:
            raise ValueError(
                f"Packed prediction has {token_count} tokens, but latent shape {(height, width)} requires "
                f"{expected_token_count}"
            )
        if packed_channels % 4 != 0:
            raise ValueError(f"Packed prediction channels must be divisible by 4, got {packed_channels}")

        channels = packed_channels // 4
        return (
            packed_latents.reshape(batch_size, height // 2, width // 2, channels, 2, 2)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch_size, channels, height, width)
            .contiguous()
        )

    @staticmethod
    def _validate_context_latents(
        context_latents: Any,
        *,
        batch_size: int,
        channels: int,
    ) -> list[torch.Tensor]:
        """Validate the ordered cached context-latent sequence.

        Args:
            context_latents: Ordered list of tensors, each with shape [batch,
                channels, context_height, context_width].
            batch_size: Required leading size for every context tensor.
            channels: Required latent channel count for every context tensor.

        Returns:
            The validated list of context tensors in its original order. The
            returned tensors alias the input tensors.
        """
        if not isinstance(context_latents, (list, tuple)) or not context_latents:
            raise ValueError("Qwen-Image-Edit batches require a non-empty ordered context_latents list")

        validated = []
        for index, latent in enumerate(context_latents):
            if not isinstance(latent, torch.Tensor):
                raise TypeError(f"context_latents[{index}] must be a torch.Tensor, got {type(latent)!r}")
            if latent.ndim != 4:
                raise ValueError(
                    f"context_latents[{index}] must have shape [batch, channels, height, width], "
                    f"got {tuple(latent.shape)}"
                )
            if latent.shape[0] != batch_size or latent.shape[1] != channels:
                raise ValueError(
                    f"context_latents[{index}] must have batch/channels {(batch_size, channels)}, "
                    f"got {tuple(latent.shape[:2])}"
                )
            if latent.shape[-2] % 2 != 0 or latent.shape[-1] % 2 != 0:
                raise ValueError(
                    f"context_latents[{index}] height and width must be divisible by 2, got {tuple(latent.shape[-2:])}"
                )
            validated.append(latent)
        return validated

    def prepare_inputs(self, context: FlowMatchingContext) -> dict[str, Any]:
        """Build Qwen edit conditioning from a cached flow-matching batch.

        Args:
            context: Flow context whose ``noisy_latents`` tensor has shape
                [batch, channels, target_height, target_width]. Its batch must
                contain ``context_latents`` as an ordered list of tensors with
                shape [batch, channels, context_height, context_width],
                ``text_embeddings`` with shape [batch, text_tokens, hidden],
                and ``text_attention_mask`` with shape [batch, text_tokens].

        Returns:
            Mapping containing ``hidden_states`` with shape [batch, total_image_tokens,
            4 * channels], ``encoder_hidden_states`` with shape [batch,
            text_tokens, hidden], ``encoder_hidden_states_mask`` with shape
            [batch, text_tokens], ``timestep`` with shape [batch], and the
            non-tensor Qwen image-shape metadata needed by ``forward``.
        """
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(
                "QwenImageEditAdapter expects noisy target latents with shape [batch, channels, height, width], "
                f"got {tuple(noisy_latents.shape)}"
            )

        batch_size, channels, target_height, target_width = noisy_latents.shape
        context_latents = self._validate_context_latents(
            context.batch.get("context_latents"),
            batch_size=batch_size,
            channels=channels,
        )

        text_embeddings = context.batch.get("text_embeddings")
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError("Qwen-Image-Edit batches require text_embeddings as a torch.Tensor")
        if text_embeddings.ndim != 3 or text_embeddings.shape[0] != batch_size:
            raise ValueError(
                "text_embeddings must have shape [batch, text_tokens, hidden] with the target batch size, "
                f"got {tuple(text_embeddings.shape)}"
            )

        text_attention_mask = context.batch.get("text_attention_mask")
        if not isinstance(text_attention_mask, torch.Tensor):
            raise TypeError("Qwen-Image-Edit batches require text_attention_mask as a torch.Tensor")
        if text_attention_mask.shape != text_embeddings.shape[:2]:
            raise ValueError(
                "text_attention_mask must have shape [batch, text_tokens] matching text_embeddings, "
                f"got mask={tuple(text_attention_mask.shape)}, embeddings={tuple(text_embeddings.shape)}"
            )

        timesteps = context.timesteps
        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise ValueError(f"timesteps must have shape [batch], got {tuple(timesteps.shape)}")

        packed_target = self._pack_latents(noisy_latents)
        packed_contexts = [
            self._pack_latents(latent.to(device=context.device, dtype=context.dtype, non_blocking=True))
            for latent in context_latents
        ]
        hidden_states = torch.cat([packed_target, *packed_contexts], dim=1)

        per_sample_shapes = [
            (1, target_height // 2, target_width // 2),
            *[(1, latent.shape[-2] // 2, latent.shape[-1] // 2) for latent in context_latents],
        ]
        img_shapes = [list(per_sample_shapes) for _ in range(batch_size)]

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": text_embeddings.to(
                device=context.device,
                dtype=context.dtype,
                non_blocking=True,
            ),
            "encoder_hidden_states_mask": text_attention_mask.to(device=context.device, non_blocking=True),
            "timestep": timesteps.to(device=context.device, dtype=context.dtype) / 1000.0,
            "img_shapes": img_shapes,
            "guidance": None,
            "_target_token_count": packed_target.shape[1],
            "_target_latent_shape": (channels, target_height, target_width),
        }

    def forward(self, model: nn.Module, inputs: dict[str, Any]) -> torch.Tensor:
        """Run the upstream transformer and return only target predictions.

        Args:
            model: Upstream Diffusers Qwen image transformer. Its packed output
                has shape [batch, total_image_tokens, 4 * channels].
            inputs: Mapping returned by ``prepare_inputs``. Tensor-bearing fields
                have the layouts documented by that method.

        Returns:
            Target velocity prediction tensor of shape [batch, channels,
            target_height, target_width]. Context-token predictions are sliced
            off before unpacking.
        """
        model_output = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"],
            timestep=inputs["timestep"],
            img_shapes=inputs["img_shapes"],
            guidance=inputs["guidance"],
            attention_kwargs=None,
            return_dict=False,
        )
        packed_prediction = self.post_process_prediction(model_output)
        if packed_prediction.ndim != 3:
            raise ValueError(
                "Qwen image transformer prediction must have shape [batch, image_tokens, packed_channels], "
                f"got {tuple(packed_prediction.shape)}"
            )

        target_token_count = inputs["_target_token_count"]
        channels, target_height, target_width = inputs["_target_latent_shape"]
        packed_target_prediction = packed_prediction[:, :target_token_count]
        prediction = self._unpack_latents(
            packed_target_prediction,
            height=target_height,
            width=target_width,
        )
        if prediction.shape[1] != channels:
            raise ValueError(
                f"Qwen image transformer predicted {prediction.shape[1]} latent channels, expected {channels}"
            )
        return prediction

    def compute_loss(self, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute unreduced flow-matching velocity MSE in float32.

        Args:
            model_pred: Predicted target velocity tensor of shape [batch,
                channels, target_height, target_width].
            target: Reference ``noise - clean_latent`` velocity tensor with the
                same shape as ``model_pred``.

        Returns:
            Per-element float32 squared-error tensor with the same shape as
            ``model_pred``.
        """
        if model_pred.shape != target.shape:
            raise ValueError(
                "Qwen-Image-Edit flow loss requires prediction and target to have identical shapes, "
                f"got prediction={tuple(model_pred.shape)}, target={tuple(target.shape)}"
            )
        return torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")

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

from __future__ import annotations

import random
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from .base import FlowMatchingContext
from .qwen_image import QwenImageAdapter


class QwenImageEditAdapter(QwenImageAdapter):
    """Adapter for Qwen-Image-Edit flow-matching training."""

    def __init__(
        self,
        caption_drop_rate: float = 0.0,
        vae_scale_factor: int = 8,
        offload_text_encoder_after_encode: bool = False,
        offload_vae_after_encode: bool = False,
        cached_latents_only: bool = False,
    ) -> None:
        super().__init__(guidance_scale=1.0, use_guidance_embeds=False)
        self.caption_drop_rate = float(caption_drop_rate)
        self.vae_scale_factor = int(vae_scale_factor)
        self.offload_text_encoder_after_encode = bool(offload_text_encoder_after_encode)
        self.offload_vae_after_encode = bool(offload_vae_after_encode)
        self.cached_latents_only = bool(cached_latents_only)
        self.pipeline = None
        self.vae = None

    def attach_pipeline(self, pipe: Any, device: torch.device, dtype: torch.dtype) -> None:
        """Attach and freeze the non-trainable Qwen-Image-Edit pipeline modules."""
        self.pipeline = pipe
        if self.cached_latents_only:
            self.vae = getattr(pipe, "vae", None)
            if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
                pipe.text_encoder.requires_grad_(False).eval()
            if self.vae is not None:
                self.vae.requires_grad_(False).eval()
            return

        if not hasattr(pipe, "vae") or pipe.vae is None:
            raise ValueError("QwenImageEditAdapter requires the loaded pipeline to contain a VAE.")
        if not hasattr(pipe, "text_encoder") or pipe.text_encoder is None:
            raise ValueError("QwenImageEditAdapter requires the loaded pipeline to contain a text_encoder.")

        self.vae = pipe.vae.requires_grad_(False).to(device)
        self.vae.eval()

        pipe.text_encoder = pipe.text_encoder.requires_grad_(False).to(device=device, dtype=dtype)
        pipe.text_encoder.eval()

    def prepare_latents(
        self,
        batch: Dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        global_step: int,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """Encode target/context images and prompts before the generic noise step."""
        del global_step
        cached_latents = _extract_cached_target_latents(batch)
        if cached_latents is not None:
            return _prepare_cached_latents(batch, cached_latents, device, dtype)

        if self.cached_latents_only:
            raise KeyError(
                "QwenImageEditAdapter was configured with cached_latents_only=True, but the batch did not contain "
                "'image_latents' or 'target_latents'."
            )
        if self.pipeline is None or self.vae is None:
            raise RuntimeError("QwenImageEditAdapter.attach_pipeline() must be called before training.")

        vae_dtype = _module_dtype(self.vae, torch.float32)
        text_encoder_dtype = _module_dtype(self.pipeline.text_encoder, dtype)
        with torch.profiler.record_function("qwen_edit.move_frozen_modules_to_device"):
            self.vae.to(device=device, dtype=vae_dtype)
            self.pipeline.text_encoder.to(device=device, dtype=text_encoder_dtype)

        with torch.profiler.record_function("qwen_edit.prepare_image_tensors"):
            target_image = _as_batched_image_tensor(batch["target_image"], device, vae_dtype)
            context_images = [_as_batched_image_tensor(image, device, vae_dtype) for image in batch["context_images"]]
            condition_images = _prepare_condition_images(batch.get("condition_images", []))

        prompt = batch.get("prompt", [""])
        if isinstance(prompt, str):
            prompt = [prompt]
        else:
            prompt = [str(item) for item in prompt]
        if self.caption_drop_rate > 0 and random.random() < self.caption_drop_rate:
            prompt = [""] * len(prompt)

        with torch.no_grad():
            with torch.profiler.record_function("qwen_edit.vae_encode_target"):
                target_latents = self._encode_vae_image(target_image)
            with torch.profiler.record_function("qwen_edit.vae_encode_context"):
                context_latents = [self._encode_vae_image(image)[:, :, 0] for image in context_images]
            with torch.profiler.record_function("qwen_edit.encode_prompt"):
                if _is_nested_condition_images(condition_images):
                    prompt_embeds, prompt_embeds_mask = self._encode_prompts_per_sample(
                        prompt,
                        condition_images,
                        device,
                    )
                else:
                    prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
                        prompt=prompt,
                        image=condition_images,
                        device=device,
                    )

        offloaded = False
        with torch.profiler.record_function("qwen_edit.offload_frozen_modules"):
            if self.offload_text_encoder_after_encode:
                self.pipeline.text_encoder.to("cpu")
                offloaded = True
            if self.offload_vae_after_encode:
                self.vae.to("cpu")
                offloaded = True
            if offloaded and device.type == "cuda":
                torch.cuda.empty_cache()

        return target_latents[:, :, 0].to(dtype), {
            "context_latents": [latent.to(device=device, dtype=dtype) for latent in context_latents],
            "prompt_embeds": prompt_embeds.to(device=device, dtype=dtype),
            "prompt_embeds_mask": prompt_embeds_mask.to(device=device) if prompt_embeds_mask is not None else None,
        }

    def prepare_inputs(self, context: FlowMatchingContext) -> Dict[str, Any]:
        noisy_latents = context.noisy_latents
        if noisy_latents.ndim != 4:
            raise ValueError(f"QwenImageEditAdapter expects 4D latents [B, C, H, W], got {noisy_latents.ndim}D")

        batch_size, channels, height, width = noisy_latents.shape
        with torch.profiler.record_function("qwen_edit.pack_target_latents"):
            packed_target = self._pack_latents(noisy_latents)

        packed_latents = [packed_target]
        context_latents = context.batch["context_latents"]
        with torch.profiler.record_function("qwen_edit.pack_context_latents"):
            for latent in context_latents:
                packed_latents.append(self._pack_latents(latent.to(context.device, context.dtype)))
        with torch.profiler.record_function("qwen_edit.cat_hidden_states"):
            hidden_states = torch.cat(packed_latents, dim=1)

        img_shapes = [
            [
                (1, height // 2, width // 2),
                *[(1, latent.shape[-2] // 2, latent.shape[-1] // 2) for latent in context_latents],
            ]
            for _ in range(batch_size)
        ]

        prompt_embeds_mask = context.batch.get("prompt_embeds_mask")
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(context.device)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": context.batch["prompt_embeds"].to(context.device, context.dtype),
            "encoder_hidden_states_mask": prompt_embeds_mask,
            "timestep": context.timesteps.to(context.dtype) / 1000.0,
            "img_shapes": img_shapes,
            "guidance": None,
            "target_token_count": packed_target.size(1),
            "_original_shape": (batch_size, channels, height, width),
        }

    def forward(self, model: nn.Module, inputs: Dict[str, Any]) -> torch.Tensor:
        original_shape = inputs.pop("_original_shape")
        _, _, height, width = original_shape
        target_token_count = inputs.pop("target_token_count")

        with torch.profiler.record_function("qwen_edit.transformer_forward"):
            model_pred = model(
                hidden_states=inputs["hidden_states"],
                timestep=inputs["timestep"],
                guidance=inputs["guidance"],
                encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"],
                encoder_hidden_states=inputs["encoder_hidden_states"],
                img_shapes=inputs["img_shapes"],
                txt_seq_lens=None,
                attention_kwargs=None,
                return_dict=False,
            )
        with torch.profiler.record_function("qwen_edit.unpack_prediction"):
            pred = self.post_process_prediction(model_pred)[:, :target_token_count]
            return self._unpack_latents(pred, height * self.vae_scale_factor, width * self.vae_scale_factor)

    def _encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean)
        latents_mean = latents_mean.view(1, self.vae.config.z_dim, 1, 1, 1).to(image.device, image.dtype)

        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std)
        latents_std = latents_std.view(1, self.vae.config.z_dim, 1, 1, 1).to(image.device, image.dtype)

        latents = self.vae.encode(image.unsqueeze(2)).latent_dist.sample()
        return latents_std * (latents - latents_mean)

    def _encode_prompts_per_sample(
        self,
        prompt: list[str],
        condition_images: list[list[Any]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(prompt) != len(condition_images):
            raise ValueError(
                f"Expected one condition-image list per prompt, got {len(condition_images)} images for "
                f"{len(prompt)} prompts."
            )

        embeds = []
        masks = []
        for prompt_item, image_items in zip(prompt, condition_images):
            prompt_embeds, prompt_embeds_mask = self.pipeline.encode_prompt(
                prompt=[prompt_item],
                image=image_items,
                device=device,
            )
            prompt_embed = prompt_embeds[0]
            if prompt_embeds_mask is None:
                prompt_mask = torch.ones(prompt_embed.shape[0], dtype=torch.long, device=prompt_embed.device)
            else:
                prompt_mask = prompt_embeds_mask[0].to(device=prompt_embed.device)
            embeds.append(prompt_embed)
            masks.append(prompt_mask)

        max_seq_len = max(embed.shape[0] for embed in embeds)
        padded_embeds = []
        padded_masks = []
        for prompt_embed, prompt_mask in zip(embeds, masks):
            pad_len = max_seq_len - prompt_embed.shape[0]
            if pad_len > 0:
                prompt_embed = torch.nn.functional.pad(prompt_embed, (0, 0, 0, pad_len), value=0.0)
                prompt_mask = torch.nn.functional.pad(prompt_mask, (0, pad_len), value=0)
            padded_embeds.append(prompt_embed)
            padded_masks.append(prompt_mask)

        return torch.stack(padded_embeds), torch.stack(padded_masks)


def _module_dtype(module: nn.Module, default: torch.dtype) -> torch.dtype:
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    return default


def _as_batched_image_tensor(image: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4:
        raise ValueError(f"Expected image tensor with shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
    return image.to(device=device, dtype=dtype)


def _prepare_condition_images(condition_images: Any) -> list[Any] | list[list[Any]]:
    if _is_nested_condition_images(condition_images):
        return [[_to_processor_image(image) for image in sample_images] for sample_images in condition_images]
    return [_to_processor_image(image) for image in condition_images]


def _is_nested_condition_images(condition_images: Any) -> bool:
    return (
        isinstance(condition_images, (list, tuple))
        and len(condition_images) > 0
        and isinstance(condition_images[0], (list, tuple))
    )


def _extract_cached_target_latents(batch: Dict[str, Any]) -> torch.Tensor | None:
    for key in ("image_latents", "target_latents"):
        value = batch.get(key)
        if value is not None:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Cached {key} must be a torch.Tensor, got {type(value)!r}")
            return value
    return None


def _prepare_cached_latents(
    batch: Dict[str, Any],
    target_latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    if "context_latents" not in batch:
        raise KeyError("Cached Qwen-Image-Edit batches must include context_latents.")
    if "prompt_embeds" not in batch:
        raise KeyError("Cached Qwen-Image-Edit batches must include prompt_embeds.")

    target_latents = _as_batched_latent_tensor(target_latents, device, dtype)
    prompt_embeds = batch["prompt_embeds"]
    if not isinstance(prompt_embeds, torch.Tensor):
        raise TypeError(f"Cached prompt_embeds must be a torch.Tensor, got {type(prompt_embeds)!r}")
    if prompt_embeds.ndim == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)
    if prompt_embeds.ndim != 3:
        raise ValueError(f"Expected cached prompt_embeds with shape [B,S,D] or [S,D], got {tuple(prompt_embeds.shape)}")

    prompt_embeds_mask = batch.get("prompt_embeds_mask")
    if prompt_embeds_mask is not None:
        if not isinstance(prompt_embeds_mask, torch.Tensor):
            raise TypeError(f"Cached prompt_embeds_mask must be a torch.Tensor, got {type(prompt_embeds_mask)!r}")
        if prompt_embeds_mask.ndim == 1:
            prompt_embeds_mask = prompt_embeds_mask.unsqueeze(0)
        if prompt_embeds_mask.ndim != 2:
            raise ValueError(
                f"Expected cached prompt_embeds_mask with shape [B,S] or [S], got {tuple(prompt_embeds_mask.shape)}"
            )
        prompt_embeds_mask = prompt_embeds_mask.to(device=device)

    return target_latents, {
        "context_latents": _as_context_latent_list(batch["context_latents"], device, dtype),
        "prompt_embeds": prompt_embeds.to(device=device, dtype=dtype),
        "prompt_embeds_mask": prompt_embeds_mask,
    }


def _as_batched_latent_tensor(latents: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if latents.ndim == 3:
        latents = latents.unsqueeze(0)
    if latents.ndim == 5 and latents.shape[2] == 1:
        latents = latents[:, :, 0]
    if latents.ndim != 4:
        raise ValueError(f"Expected latent tensor with shape [C,H,W] or [B,C,H,W], got {tuple(latents.shape)}")
    return latents.to(device=device, dtype=dtype)


def _as_context_latent_list(context_latents: Any, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    if isinstance(context_latents, (list, tuple)):
        if not context_latents:
            raise ValueError("Cached context_latents must not be empty.")
        return [_as_batched_latent_tensor(latent, device, dtype) for latent in context_latents]

    if not isinstance(context_latents, torch.Tensor):
        raise TypeError(
            f"Cached context_latents must be a tensor or sequence of tensors, got {type(context_latents)!r}"
        )

    if context_latents.ndim == 3:
        return [_as_batched_latent_tensor(context_latents, device, dtype)]
    if context_latents.ndim == 4:
        if context_latents.shape[0] == 1:
            return [_as_batched_latent_tensor(context_latents, device, dtype)]
        return [_as_batched_latent_tensor(latent, device, dtype) for latent in context_latents]
    if context_latents.ndim == 5:
        return [
            _as_batched_latent_tensor(context_latents[:, index], device, dtype)
            for index in range(context_latents.shape[1])
        ]

    raise ValueError(
        "Expected cached context_latents with shape [C,H,W], [B,C,H,W], [N,C,H,W], or [B,N,C,H,W], "
        f"got {tuple(context_latents.shape)}"
    )


def _to_processor_image(image: Any) -> Any:
    if not isinstance(image, torch.Tensor):
        return image

    image = image.detach().cpu()
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("Qwen-Image-Edit condition image batching requires local_batch_size=1.")
        image = image.squeeze(0)
    if image.ndim != 3:
        raise ValueError(f"Expected condition image tensor with shape [C,H,W], got {tuple(image.shape)}")

    image = image.clamp(0.0, 1.0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)

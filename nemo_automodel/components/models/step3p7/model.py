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

from __future__ import annotations

import logging
import os
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.step3p5.model import Step3p5Model
from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
from nemo_automodel.components.models.step3p7.state_dict_adapter import Step3p7StateDictAdapter
from nemo_automodel.components.models.step3p7.vision_encoder import StepRoboticsVisionEncoder
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

logger = logging.getLogger(__name__)


def _debug_vision_enabled() -> bool:
    return os.environ.get("NEMO_STEP3P7_DEBUG_VISION", "0").lower() in ("1", "true", "yes")


def _debug_vision_log(message: str, *args: Any) -> None:
    logger.warning(message, *args)
    print(message % args, flush=True)


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


class Step3p7Model(nn.Module):
    """Step3.7 VLM wrapper using the native Step3.5 MoE language backbone."""

    def __init__(
        self,
        config: Step3p7Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.vision_model = StepRoboticsVisionEncoder(config.vision_config)
        self.language_model = Step3p5Model(
            config.text_config,
            backend=backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.vocab_size = config.text_config.vocab_size
        self.vit_large_projector = initialize_linear_module(
            backend.linear,
            config.vision_config.width * 4,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )
        self.image_placeholder_token_id = config.image_token_id
        self.moe_config = self.language_model.moe_config

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def _vision_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        try:
            param = next(self.vision_model.parameters())
            return param.dtype, param.device
        except StopIteration:
            dtype = get_dtype(getattr(self.config.text_config, "torch_dtype", "bfloat16"), torch.bfloat16)
            return dtype, torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    def _process_image_features(self, image_features: torch.Tensor) -> torch.Tensor:
        bsz, patches = image_features.shape[:2]
        grid = int(patches**0.5)
        if grid * grid != patches:
            raise ValueError(f"Step3p7 vision features must form a square grid, got {patches} patches.")

        image_features = image_features.permute(0, 2, 1).reshape(bsz, -1, grid, grid)
        image_features = self.vision_model.vit_downsampler1(image_features)
        image_features = self.vision_model.vit_downsampler2(image_features)
        bsz, channels, grid_h, grid_w = image_features.shape
        image_features = image_features.reshape(bsz, channels, grid_h * grid_w).permute(0, 2, 1)
        return self.vit_large_projector(image_features)

    def _process_image_input(
        self,
        pixel_values: torch.Tensor,
        *,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> list[torch.Tensor]:
        if self.vision_model is None:
            raise ValueError("vision_model is not present on this pipeline stage.")

        vision_dtype, vision_device = self._vision_dtype_device()
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:]).to(device=vision_device, dtype=vision_dtype)

        if num_patches is None:
            num_patches_list = [0] * pixel_values.shape[0]
        elif isinstance(num_patches, torch.Tensor):
            num_patches_list = [int(x) for x in num_patches.detach().cpu().view(-1).tolist()]
        else:
            num_patches_list = [int(x) for x in num_patches]

        patch_image_features = None
        if patch_pixel_values is not None and patch_pixel_values.numel() > 0:
            patch_pixel_values = patch_pixel_values.reshape(-1, *patch_pixel_values.shape[-3:]).to(
                device=vision_device,
                dtype=vision_dtype,
            )
            patch_image_features = self._process_image_features(self.vision_model(patch_pixel_values))

        image_features = self._process_image_features(self.vision_model(pixel_values))

        merged_image_features: list[torch.Tensor] = []
        cur_patch_idx = 0
        for image_idx, num_patch in enumerate(num_patches_list):
            cur_features = []
            if num_patch > 0:
                if patch_image_features is None:
                    raise ValueError("num_patches requested patch features, but patch_pixel_values is missing.")
                patch_slice = patch_image_features[cur_patch_idx : cur_patch_idx + num_patch]
                cur_features.append(patch_slice.reshape(-1, patch_slice.shape[-1]))
            cur_features.append(image_features[image_idx].reshape(-1, image_features.shape[-1]))
            cur_patch_idx += num_patch
            merged_image_features.append(torch.cat(cur_features, dim=0) if len(cur_features) > 1 else cur_features[0])

        if _debug_vision_enabled():
            _debug_vision_log(
                "[step3p7][rank %s] vision tower active: pixel_values=%s patch_pixel_values=%s num_patches=%s "
                "vision_token_tensors=%s total_vision_tokens=%s",
                _rank(),
                tuple(pixel_values.shape),
                None if patch_pixel_values is None else tuple(patch_pixel_values.shape),
                num_patches_list,
                [tuple(t.shape) for t in merged_image_features],
                sum(t.shape[0] for t in merged_image_features),
            )

        return merged_image_features

    def get_multimodal_embeddings(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        image_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> list[torch.Tensor] | torch.Tensor | None:
        if image_embeds is not None:
            return image_embeds.reshape(-1, image_embeds.shape[-1])
        if pixel_values is None:
            return None
        return self._process_image_input(
            pixel_values,
            patch_pixel_values=patch_pixel_values,
            num_patches=num_patches,
        )

    def prepare_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.language_model.embed_tokens is None:
            if input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32):
                return input_ids
            raise ValueError("embed_tokens is not present on this pipeline stage.")

        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        flattened = (
            torch.cat([x.reshape(-1, x.shape[-1]) for x in multimodal_embeddings], dim=0)
            if isinstance(multimodal_embeddings, list)
            else multimodal_embeddings.reshape(-1, multimodal_embeddings.shape[-1])
        )
        image_mask = input_ids == self.config.image_token_id
        expected = int(image_mask.sum().item())
        if flattened.shape[0] != expected:
            raise ValueError(
                f"Step3p7 image token mismatch: got {flattened.shape[0]} vision embeddings for {expected} placeholders."
            )

        flattened = flattened.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[image_mask] = flattened

        if _debug_vision_enabled():
            _debug_vision_log(
                "[step3p7][rank %s] fed vision embeddings into LLM: placeholders=%s inputs_embeds=%s",
                _rank(),
                expected,
                tuple(inputs_embeds.shape),
            )

        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        image_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Step3p7Model requires input_ids or inputs_embeds.")
            if input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32):
                inputs_embeds = input_ids
                input_ids = None
            else:
                multimodal_embeddings = self.get_multimodal_embeddings(
                    pixel_values=pixel_values,
                    patch_pixel_values=patch_pixel_values,
                    num_patches=num_patches,
                    image_embeds=image_embeds,
                )
                inputs_embeds = self.prepare_inputs_embeds(input_ids, multimodal_embeddings)

        if (
            inputs_embeds is not None
            and attention_mask is not None
            and attention_mask.shape[-1] != inputs_embeds.shape[1]
        ):
            attention_mask = None

        return self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )


class Step3p7ForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Native Step3.7 VLM implementation for MedPix fine-tuning with EP and PP."""

    _keep_in_fp32_modules = ["rotary_emb"]

    _pp_keep_self_forward: bool = True

    @classmethod
    def from_config(
        cls,
        config: Step3p7Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        config = Step3p7Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Step3p7Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = Step3p7Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = getattr(config.text_config, "pad_token_id", None)

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Step3p7StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(config.text_config, "torch_dtype", "bfloat16"), torch.bfloat16),
            )

    @property
    def language_model(self):
        return self.model.language_model

    @property
    def visual(self):
        return self.model.vision_model

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def prepare_model_inputs_for_cp(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        patch_pixel_values: torch.Tensor | None = None,
        num_patches: torch.Tensor | list[int] | tuple[int, ...] | None = None,
        image_embeds: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Merge vision features into token embeddings before CP sequence sharding."""
        multimodal_embeddings = self.model.get_multimodal_embeddings(
            pixel_values=pixel_values,
            patch_pixel_values=patch_pixel_values,
            num_patches=num_patches,
            image_embeds=image_embeds,
        )
        inputs_embeds = self.model.prepare_inputs_embeds(input_ids, multimodal_embeddings)
        return {"inputs_embeds": inputs_embeds}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if kwargs.pop("_pre_embed_only", False):
            if input_ids is None:
                raise ValueError("Step3p7 CP pre-embedding requires input_ids.")
            return self.prepare_model_inputs_for_cp(input_ids=input_ids, **kwargs)

        pixel_values = kwargs.get("pixel_values", None)
        has_image_tokens = (
            input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            and (input_ids == self.config.image_token_id).any()
        )

        chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
        consumed_vlm_chunk = False
        if pixel_values is None and has_image_tokens:
            image_chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            if image_chunks is not None and chunk_idx < len(image_chunks):
                kwargs["pixel_values"] = image_chunks[chunk_idx]
                patch_chunks = getattr(self, "_vlm_patch_pixel_values_chunks", None)
                if patch_chunks is not None and chunk_idx < len(patch_chunks):
                    kwargs["patch_pixel_values"] = patch_chunks[chunk_idx]
                num_patches_chunks = getattr(self, "_vlm_num_patches_chunks", None)
                if num_patches_chunks is not None and chunk_idx < len(num_patches_chunks):
                    kwargs["num_patches"] = num_patches_chunks[chunk_idx]
                newline_chunks = getattr(self, "_vlm_patch_newline_mask_chunks", None)
                if newline_chunks is not None and chunk_idx < len(newline_chunks):
                    kwargs["patch_newline_mask"] = newline_chunks[chunk_idx]
                consumed_vlm_chunk = True
                if _debug_vision_enabled():
                    _debug_vision_log(
                        "[step3p7][rank %s] PP consumed vision chunk=%s pixel_values=%s num_patches=%s",
                        _rank(),
                        chunk_idx,
                        tuple(image_chunks[chunk_idx].shape),
                        None
                        if kwargs.get("num_patches", None) is None
                        else [int(x) for x in kwargs["num_patches"].detach().cpu().view(-1).tolist()],
                    )

        if consumed_vlm_chunk:
            self._vlm_chunk_idx = chunk_idx + 1

        if (
            inputs_embeds is not None
            and attention_mask is not None
            and attention_mask.shape[-1] != inputs_embeds.shape[1]
        ):
            attention_mask = None
            padding_mask = None

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids,
                position_ids,
                padding_mask,
                kwargs,
            )
            attention_mask = None
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask

        # MedPix batches are right-padded. For causal LM loss the right-pad
        # keys are future positions for all supervised tokens, while TE's
        # padding_causal + sliding-window path diverges from the CP path.
        if self.backend.attn == "te" and attention_mask is not None:
            attention_mask = None

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        logits = self.lm_head(hidden_states) if self.lm_head is not None else hidden_states
        return logits

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )

        with buffer_device:
            self.model.language_model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.text_config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.language_model.rotary_emb.device = buffer_device


ModelClass = Step3p7ForConditionalGeneration

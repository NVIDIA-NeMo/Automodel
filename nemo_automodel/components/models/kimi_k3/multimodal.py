# Copyright 2025-2026 The Moonshot AI Team and HuggingFace Inc. team. All rights reserved.
#
# The multimodal input merge is adapted from the Kimi K3 checkpoint implementation
# and LLaVA. Apache-licensed portions remain under Apache-2.0; other adapted portions
# are distributed under the Kimi K3 License in KIMI_K3_LICENSE.

"""Native top-level Kimi K3 vision-language model."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config
from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM
from nemo_automodel.components.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from nemo_automodel.components.models.kimi_k3.vision import (
    MLP,
    IdentityMap,
    MoonViT3dPretrainedModel,
    PatchMergerMLP,
    PatchMergerMLPV2,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _vision_tower_config(config: KimiK3Config):
    vision_config = copy.deepcopy(config.vision_config)
    vision_config.kimi_attn_implementation = vision_config._attn_implementation
    vision_config.hidden_size = vision_config.vt_hidden_size
    vision_config.num_attention_heads = vision_config.vt_num_attention_heads
    vision_config.num_hidden_layers = vision_config.vt_num_hidden_layers
    vision_config.intermediate_size = vision_config.vt_intermediate_size
    return vision_config


def _projector_config(config: KimiK3Config) -> SimpleNamespace:
    vision_config = config.vision_config
    return SimpleNamespace(
        mm_projector_type=vision_config.mm_projector_type,
        mm_hidden_size=vision_config.mm_hidden_size,
        hidden_size=config.text_config.hidden_size,
        merge_kernel_size=vision_config.merge_kernel_size,
        projector_hidden_act=vision_config.projector_hidden_act,
        projector_ln_eps=vision_config.projector_ln_eps,
    )


class KimiK3ForConditionalGeneration(KimiK3ForCausalLM):
    """Kimi K3 MoonViT3d tower, projector, and native KDA/MLA language model."""

    # The multimodal wrapper uses the same Kimi Linear execution path as its
    # language backbone, including its model-owned CP and pipeline staging.
    ModelCapabilities = KimiK3ForCausalLM.ModelCapabilities

    @classmethod
    def from_config(
        cls,
        config: KimiK3Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "KimiK3ForConditionalGeneration":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "KimiK3ForConditionalGeneration":
        config = KimiK3Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: KimiK3Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        top_config = config
        super().__init__(
            config.text_config,
            moe_config=moe_config,
            backend=backend,
            **kwargs,
        )
        self.config = top_config
        self.vision_tower = MoonViT3dPretrainedModel(_vision_tower_config(top_config))

        projector_config = _projector_config(top_config)
        projector_types = {
            "identity": IdentityMap,
            "mlp": MLP,
            "patchmerger": PatchMergerMLP,
            "patchmergerv2": PatchMergerMLPV2,
        }
        try:
            projector_cls = projector_types[projector_config.mm_projector_type]
        except KeyError as error:
            raise ValueError(f"Unsupported K3 projector type {projector_config.mm_projector_type!r}.") from error
        self.mm_projector = projector_cls(projector_config)
        model_dtype = get_dtype(getattr(config.text_config, "torch_dtype", None), torch.bfloat16)
        cast_model_to_dtype(self.vision_tower, model_dtype)
        cast_model_to_dtype(self.mm_projector, model_dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = KimiK3StateDictAdapter(
                top_config,
                self.model.moe_config,
                self.backend,
                dtype=model_dtype,
            )

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Expand image placeholders in ``[batch, sequence]`` text tensors."""
        _, embed_dim = image_features[0].shape
        feature_lengths = torch.tensor(
            [features.shape[0] for features in image_features],
            dtype=torch.long,
            device=input_ids.device,
        )
        image_features_tensor = torch.cat(image_features, dim=0)
        image_token_id = self.config.media_placeholder_token_id
        pad_token_id = self.config.pad_token_id

        placeholder_mask = input_ids == image_token_id
        if int(placeholder_mask.sum()) != len(feature_lengths):
            raise ValueError(
                f"Received {int(placeholder_mask.sum())} image placeholders for "
                f"{len(feature_lengths)} image feature groups."
            )
        occupation = torch.ones_like(input_ids)
        occupation[placeholder_mask] = feature_lengths
        max_length = int(occupation.sum(-1).max().item())
        batch_indices, text_indices = torch.where(~placeholder_mask)
        positions = occupation.cumsum(-1) - 1
        left_padding = bool(torch.any(input_ids[:, -1] != pad_token_id))
        left_pad = max_length - 1 - positions[:, -1]
        if left_padding:
            positions = positions + left_pad[:, None]
        text_positions = positions[batch_indices, text_indices]

        final_embeddings = inputs_embeds.new_zeros(input_ids.shape[0], max_length, embed_dim)
        final_attention_mask = attention_mask.new_zeros(input_ids.shape[0], max_length)
        final_labels = (
            input_ids.new_full((input_ids.shape[0], max_length), self.config.ignore_index)
            if labels is not None
            else None
        )
        final_embeddings[batch_indices, text_positions] = inputs_embeds[batch_indices, text_indices]
        final_attention_mask[batch_indices, text_positions] = attention_mask[batch_indices, text_indices]
        if final_labels is not None:
            final_labels[batch_indices, text_positions] = labels[batch_indices, text_indices]

        image_positions = torch.ones(
            input_ids.shape[0],
            max_length,
            dtype=torch.bool,
            device=input_ids.device,
        )
        image_positions[batch_indices, text_positions] = False
        image_positions &= image_positions.cumsum(-1) - 1 >= left_pad[:, None]
        if int(image_positions.sum()) != image_features_tensor.shape[0]:
            raise ValueError(
                f"Expanded placeholders contain {int(image_positions.sum())} positions, "
                f"but the vision tower returned {image_features_tensor.shape[0]} tokens."
            )
        final_embeddings[image_positions] = image_features_tensor.reshape(-1, embed_dim).to(final_embeddings.device)
        final_attention_mask |= image_positions
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill(final_attention_mask == 0, 1)

        pad_batch, pad_indices = torch.where(input_ids == pad_token_id)
        final_embeddings[pad_batch, positions[pad_batch, pad_indices]] = 0
        return final_embeddings, final_attention_mask, final_labels, position_ids

    def _extract_image_features(
        self,
        pixel_values: torch.Tensor,
        grid_thws: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode packed ``pixel_values`` using ``grid_thws`` image/video geometry."""
        pixel_values = pixel_values.to(self.vision_tower.patch_embed.proj.weight.dtype)
        return self.mm_projector(self.vision_tower(pixel_values, grid_thws))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        block_residual: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        grid_thws: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run text-only or multimodal K3 inputs."""
        chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        if (
            pixel_values is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
            and chunks is not None
            and (input_ids == self.config.media_placeholder_token_id).any()
        ):
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
            if chunk_idx < len(chunks):
                pixel_values = chunks[chunk_idx]
                image_grid_hws = self._vlm_image_grid_hws_chunks[chunk_idx]
                if image_grid_hws.shape[-1] == 2:
                    ones = torch.ones(
                        image_grid_hws.shape[0],
                        1,
                        dtype=image_grid_hws.dtype,
                        device=image_grid_hws.device,
                    )
                    grid_thws = torch.cat((ones, image_grid_hws), dim=-1)
                else:
                    grid_thws = image_grid_hws
                self._vlm_chunk_idx = chunk_idx + 1

        if pixel_values is not None:
            if self.vision_tower is None or self.mm_projector is None:
                raise ValueError("Only the first K3 pipeline stage can process pixel values.")
            if input_ids is None:
                raise ValueError("K3 multimodal forward requires input_ids.")
            if grid_thws is None:
                raise ValueError("K3 multimodal forward requires grid_thws.")
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            image_features = self._extract_image_features(pixel_values, grid_thws)
            inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds.to(image_features[0].dtype),
                input_ids,
                attention_mask,
                labels,
            )
            input_ids = None

        output = super().forward(
            input_ids,
            block_residual,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        if labels is None or isinstance(output, tuple) or isinstance(output, torch.Tensor):
            return output

        shift_logits = output.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if attention_mask is not None:
            valid = attention_mask[..., 1:].bool()
            shift_logits = shift_logits[valid]
            shift_labels = shift_labels[valid]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1).to(shift_logits.device),
            ignore_index=self.config.ignore_index,
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=output.logits,
            hidden_states=output.hidden_states,
        )


ModelClass = KimiK3ForConditionalGeneration

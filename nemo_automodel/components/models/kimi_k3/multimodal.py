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
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config
from nemo_automodel.components.models.kimi_k3.cp import shard_batch_for_kimi_vlm_cp
from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM
from nemo_automodel.components.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from nemo_automodel.components.models.kimi_k3.vision import (
    MLP,
    IdentityMap,
    Learnable2DInterpPosEmbDivided_fixed,
    MoonViT3dPretrainedModel,
    PatchMergerMLP,
    PatchMergerMLPV2,
    Rope2DPosEmbRepeated,
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


def _positions_within_documents(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return zero-based positions within each document, skipping masked tokens.

    Masked tokens inside a document, such as deliberately masked fake-image
    spans, do not advance positions or start a new document.

    Args:
        attention_mask: Binary or indexed document mask [batch, sequence].

    Returns:
        Positions [batch, sequence], with masked positions set to one.
    """
    valid = attention_mask > 0
    indices = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
    # Index of the closest valid token strictly before each position, or -1.
    last_valid = indices.masked_fill(~valid, -1).cummax(dim=-1).values
    previous_valid = F.pad(last_valid[:, :-1], (1, 0), value=-1)
    previous_document = attention_mask.gather(1, previous_valid.clamp(min=0)).masked_fill(previous_valid < 0, 0)
    starts = valid & (attention_mask != previous_document)
    counts = valid.long().cumsum(dim=-1)
    offsets = ((counts - 1) * starts).cummax(dim=-1).values
    return (counts - 1 - offsets).masked_fill(~valid, 1)


class KimiK3ForConditionalGeneration(KimiK3ForCausalLM):
    """Kimi K3 MoonViT3d tower, projector, and native KDA/MLA language model."""

    # The multimodal wrapper uses the same Kimi Linear execution path as its
    # language backbone, including its model-owned CP and pipeline staging.
    ModelCapabilities = KimiK3ForCausalLM.ModelCapabilities
    # Keep complete media token streams until image embedding inside forward.
    _cp_batch_sharder = staticmethod(shard_batch_for_kimi_vlm_cp)

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
        self.cp_mesh: DeviceMesh | None = None
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

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Describe global media token inputs and local CP stage activations.

        Args:
            is_first: Whether this is the first pipeline stage.
            microbatch_size: Number of sequences per pipeline microbatch.
            seq_len: Global input token count before CP padding and slicing.
            dtype: Floating point dtype of pipeline activations.

        Returns:
            Input and output meta tensors. Only first-stage token IDs retain
            the global sequence; hidden states and residuals use local length.
        """
        cp_size = 1 if self.cp_mesh is None else self.cp_mesh.size()
        local_seq_len = (seq_len + cp_size - 1) // cp_size
        inputs, outputs = super().get_pipeline_stage_metas(
            is_first=is_first, microbatch_size=microbatch_size, seq_len=local_seq_len, dtype=dtype
        )
        if is_first:
            inputs = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        return inputs, outputs

    def _merge_input_ids_with_image_features(
        self,
        image_features: list[torch.Tensor],
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Merge compact HF placeholders or pre-expanded training image tokens.

        Args:
            image_features: Per-image tensors of shape [image_tokens, hidden], in
                batch order. Each image_tokens count follows spatial patch merging.
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            input_ids: Tensor of shape [batch, sequence], with either one media
                placeholder per image or one per projected image token.
            attention_mask: Binary or indexed document mask of shape [batch, sequence].
            labels: Optional unshifted targets of shape [batch, sequence].

        Returns:
            Embeddings [batch, expanded_sequence, hidden], binary or indexed document mask,
            optional labels, and position IDs [batch, expanded_sequence]. Already
            expanded inputs retain their sequence length. Inputs are not mutated.
        """
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
        if int(placeholder_mask.sum()) == image_features_tensor.shape[0]:
            # A matching total count alone does not prove image ownership: one
            # image must never straddle rows, documents, padding, or text gaps.
            # A wholly masked span is valid: fake-image injection embeds the
            # image for FSDP but masks its tokens from attention and positions.
            locations = placeholder_mask.nonzero()
            offset = 0
            for features in image_features:
                span = locations[offset : offset + features.shape[0]]
                owner = attention_mask[span[:, 0], span[:, 1]]
                if (
                    not bool((span[:, 0] == span[0, 0]).all())
                    or not bool((owner == owner[0]).all())
                    or not bool((span[1:, 1] - span[:-1, 1] == 1).all())
                ):
                    raise ValueError("A K3 image token span crosses a document, row, padding, or text boundary.")
                offset += features.shape[0]
            final_embeddings = inputs_embeds.masked_scatter(
                placeholder_mask.unsqueeze(-1),
                image_features_tensor.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )
            final_labels = (
                labels.masked_fill(placeholder_mask, self.config.ignore_index) if labels is not None else None
            )
            position_ids = _positions_within_documents(attention_mask)
            return final_embeddings, attention_mask, final_labels, position_ids
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
        left_padding = not bool(torch.any(attention_mask[:, -1] == 0))
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
        token_positions = torch.arange(max_length, device=input_ids.device)[None, :]
        if left_padding:
            image_positions &= token_positions >= left_pad[:, None]
        else:
            image_positions &= token_positions < occupation.sum(-1, keepdim=True)
        if int(image_positions.sum()) != image_features_tensor.shape[0]:
            raise ValueError(
                f"Expanded placeholders contain {int(image_positions.sum())} positions, "
                f"but the vision tower returned {image_features_tensor.shape[0]} tokens."
            )
        final_embeddings[image_positions] = image_features_tensor.reshape(-1, embed_dim).to(final_embeddings.device)
        final_attention_mask[image_positions] = torch.repeat_interleave(
            attention_mask[placeholder_mask], feature_lengths
        )
        position_ids = _positions_within_documents(final_attention_mask)

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
        image_grid_hws: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        kimi_vlm_full_input: bool = False,
        kimi_vlm_token_order: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run text-only or multimodal K3 inputs.

        Args:
            input_ids: Token IDs [batch, sequence], or PP activations
                [batch, sequence, hidden] after the first pipeline stage.
            block_residual: PP residual starts [batch * sequence, blocks, hidden].
            pixel_values: Flattened image patches [patches, channels, patch_h, patch_w].
            grid_thws: Image geometry [images, 3], ordered time, height, width.
            image_grid_thw: Canonical packed image geometry [images, 3].
            image_grid_hws: Image geometry [images, 2] from the native VLM collator;
                a [images, 3] time-height-width grid is also accepted.
            attention_mask: Binary or indexed document mask [batch, sequence].
            position_ids: Optional positions [batch, sequence].
            inputs_embeds: Optional embeddings [batch, sequence, hidden].
            labels: Optional unshifted targets [batch, sequence], with ignore_index
                at unsupervised positions. Native recipes compute loss externally.
            logits_to_keep: Trailing logit count or sequence indices [selected_tokens].
            output_hidden_states: Whether to return final hidden states.
            kimi_vlm_full_input: Whether CP left the primary token stream intact
                for image embedding before contiguous sequence sharding.
            kimi_vlm_token_order: Optional source index of each CP sequence
                position [batch, padded_sequence], set when CP moved masked
                tokens behind valid ones.
            **kwargs: Text-backbone metadata, following KimiK3ForCausalLM.forward.

        Returns:
            Causal LM output with logits [batch, expanded_sequence, vocab] and
            optional scalar loss and hidden states [batch, expanded_sequence, hidden].
            Intermediate PP stages return activations and residual starts.
        """
        chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        if (
            pixel_values is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
            and chunks is not None
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
                if pixel_values.numel() == 0:
                    pixel_values = None

        if grid_thws is None and image_grid_thw is not None:
            grid_thws = image_grid_thw

        if grid_thws is None and image_grid_hws is not None:
            if image_grid_hws.shape[-1] == 2:
                grid_thws = torch.cat((torch.ones_like(image_grid_hws[:, :1]), image_grid_hws), dim=-1)
            else:
                grid_thws = image_grid_hws

        full_primary = kimi_vlm_full_input and input_ids is not None and not input_ids.is_floating_point()
        if full_primary:
            doc_ids = kwargs["kimi_packed_doc_ids"]
            if kimi_vlm_token_order is not None:
                # Media ownership is checked in source token order.
                doc_ids = torch.empty_like(doc_ids).scatter_(1, kimi_vlm_token_order, doc_ids)
            attention_mask = doc_ids[:, : input_ids.shape[1]]

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
            inputs_embeds, attention_mask, labels, merged_positions = self._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds.to(image_features[0].dtype),
                input_ids,
                attention_mask,
                labels,
            )
            if position_ids is None:
                position_ids = merged_positions
            elif not full_primary and position_ids.shape != merged_positions.shape:
                raise ValueError("Explicit position_ids must match the expanded image token sequence.")
            if full_primary and inputs_embeds.shape[1] != input_ids.shape[1]:
                raise ValueError("K3 vision CP requires pre-expanded image placeholders from the VLM collator.")
            input_ids = None

        if full_primary:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            global_length = kwargs["kimi_packed_doc_ids"].shape[1]
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, global_length - inputs_embeds.shape[1]))
            if kimi_vlm_token_order is not None:
                order = kimi_vlm_token_order.unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
                inputs_embeds = inputs_embeds.gather(1, order)
            local_length = global_length // kwargs["kimi_packed_cp_size"]
            inputs_embeds = inputs_embeds[
                :, kwargs["kimi_packed_seq_start"] : kwargs["kimi_packed_seq_start"] + local_length
            ].contiguous()
            input_ids = None
            attention_mask = None

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

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize text and vision after distributed materialization."""
        super().initialize_weights(buffer_device=buffer_device, dtype=dtype)
        for component in (self.vision_tower, self.mm_projector):
            if component is None:
                continue
            for module in component.modules():
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                    torch.nn.init.trunc_normal_(module.weight, std=self.config.text_config.initializer_range)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
                    torch.nn.init.ones_(module.weight)
                    if isinstance(module, torch.nn.LayerNorm) and module.bias is not None:
                        torch.nn.init.zeros_(module.bias)
                elif isinstance(module, Learnable2DInterpPosEmbDivided_fixed):
                    module.reset_parameters()
                elif isinstance(module, Rope2DPosEmbRepeated):
                    module.reset_parameters()


ModelClass = KimiK3ForConditionalGeneration

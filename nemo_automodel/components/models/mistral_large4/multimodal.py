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

"""Image/text execution over the existing Large 4 training backbone."""

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import shard_sequence_for_cp_contiguous
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import Float32RMSNorm
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.context_parallel import prepare_cp_inputs
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from nemo_automodel.components.models.mistral_large4.vision import (
    MistralLarge4PatchMerger,
    MistralLarge4VisionAdapter,
    MistralLarge4VisionConfig,
)


def image_token_positions(
    input_ids: torch.Tensor,
    pixels: list[torch.Tensor],
    config: MistralLarge4VisionConfig,
    *,
    cu_seqlens: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Validate complete image grids independently inside each document.

    Args:
        input_ids: Integer [batch, sequence] or packed [tokens] storage.
        pixels: Images [channels, height, width], in document/encounter order.
        config: Native vision dimensions and image delimiter IDs.
        cu_seqlens: Optional integer [documents + 1] partition of packed storage.
        attention_mask: Optional binary keep mask with input_ids' shape.
        padding_mask: Optional boolean padding mask with input_ids' shape.

    Returns:
        Int64 [total_merged_patches] indices into flattened tokens, on input_ids'
        device. Only image tokens are replaced; row/end delimiters remain text.
    """
    if input_ids.ndim not in (1, 2) or input_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("Image alignment requires integer [batch, sequence] or packed [tokens] IDs")
    flat = input_ids.reshape(-1)
    if cu_seqlens is not None:
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.dtype not in (torch.int32, torch.int64)
            or (input_ids.ndim == 2 and input_ids.shape[0] != 1)
        ):
            raise ValueError("Packed image alignment requires one token row and integer document offsets")
        boundaries = cu_seqlens.tolist()
    elif input_ids.ndim == 2:
        boundaries = list(range(0, flat.numel() + 1, input_ids.shape[1])) if input_ids.shape[1] else []
    else:
        raise ValueError("Packed image IDs require cu_seqlens")
    if (
        len(boundaries) < 2
        or boundaries[0] != 0
        or boundaries[-1] != flat.numel()
        or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError("Document offsets must strictly partition image/text token storage")
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape or not bool((attention_mask >= 0).all()):
            raise ValueError("attention_mask must contain nonnegative document IDs matching image/text token axes")
        if not bool((attention_mask == attention_mask.long()).all()):
            raise ValueError("attention_mask must contain integer document IDs")
        valid = valid & attention_mask.bool()
        # Validate every image inside its original document, including packed
        # rows from the shared collator. A partial grid cannot borrow the next
        # document's tokens to appear complete.
        if cu_seqlens is None:
            mask = attention_mask.reshape(-1)
            cuts = (torch.where(mask[1:] != mask[:-1])[0] + 1).tolist()
            boundaries = sorted(set(boundaries + cuts))
    if padding_mask is not None:
        if padding_mask.shape != input_ids.shape or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be boolean and match image/text token axes")
        valid = valid & ~padding_mask
    ids, valid = flat.tolist(), valid.reshape(-1).tolist()
    positions, image = [], 0
    special = {config.image_token_id, config.image_break_token_id, config.image_end_token_id}
    for start, stop in zip(boundaries, boundaries[1:]):
        cursor = start
        while cursor < stop:
            token = ids[cursor]
            if token not in special:
                cursor += 1
                continue
            if token != config.image_token_id or image >= len(pixels):
                raise ValueError("Image tokens/delimiters have no matching complete pixel image")
            height, width = config.grid(pixels[image])
            expected = []
            for row in range(height):
                expected.extend([config.image_token_id] * width)
                expected.append(config.image_end_token_id if row == height - 1 else config.image_break_token_id)
            end = cursor + len(expected)
            if end > stop or ids[cursor:end] != expected or not all(valid[cursor:end]):
                raise ValueError("Image span must match its complete grid inside one unmasked document")
            positions.extend(cursor + i for i, token in enumerate(expected) if token == config.image_token_id)
            cursor, image = end, image + 1
    if image != len(pixels):
        raise ValueError("pixel_values contains images without matching token spans")
    return torch.tensor(positions, dtype=torch.long, device=input_ids.device)


class MistralLarge4ForConditionalGeneration(MistralLarge4ForCausalLM):
    """Native multimodal SFT/LoRA with complete images and packed document isolation.

    Shares the text backbone, loss, and checkpoint adapter without
    wrapping a second causal-model root. All vision checkpoint names stay native.
    """

    parallel_layer_groups = {
        "language": ("model.layers",),
        "vision": ("vision_encoder.transformer.layers",),
    }
    _no_split_modules = ["MistralLarge4Block", "MistralLarge4VisionBlock"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Image/text CP composes with PP/EP; packed THD composes with CP/PP."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep the native vision tower and all image projections on stage zero."""
        stages = super().customize_pipeline_stage_modules(
            module_names_per_stage, layers_prefix=layers_prefix, text_model=text_model
        )
        stages[0].extend(("vision_encoder", "pre_mm_projector_norm", "patch_merger", "vision_language_adapter"))
        return stages

    def __init__(self, config: MistralLarge4Config, backend: BackendConfig | None = None) -> None:
        if not config.vision_config:
            raise ValueError("Conditional generation requires the native vision_config")
        self.vision_settings = MistralLarge4VisionConfig(**config.vision_config)
        if (
            max(
                self.vision_settings.image_token_id,
                self.vision_settings.image_break_token_id,
                self.vision_settings.image_end_token_id,
            )
            >= config.vocab_size
        ):
            raise ValueError("Image delimiter IDs must fit vocab_size")
        config = copy.deepcopy(config)
        config.architectures = ["MistralLarge4ForConditionalGeneration"]
        super().__init__(config, backend)
        dtype = self.model.embed_tokens.weight.dtype
        self.vision_encoder = self.vision_settings.build(dtype=dtype)
        self.pre_mm_projector_norm = Float32RMSNorm(self.vision_settings.hidden_size, eps=1e-5, dtype=dtype)
        self.patch_merger = MistralLarge4PatchMerger(self.vision_settings, dtype=dtype)
        self.vision_language_adapter = MistralLarge4VisionAdapter(self.vision_settings, config.hidden_size, dtype=dtype)

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Keep complete image/token streams until stage-zero embedding.

        Args:
            batch: Global IDs/masks [batch, sequence], optional embeddings
                [batch, sequence, hidden], and complete images [channels, height, width].
            num_chunks: Number of independently scheduled PP microbatches.

        Returns:
            Sharder mapping using prepare_cp_inputs' aux-only contract: primary
            inputs and binary attention_mask stay global; labels, positions and
            padding_mask become [batch, padded_sequence / CP]. No weights are read.
        """
        if self.cp_mesh is None and batch.get("qkv_format") != "thd":
            return {}
        return prepare_cp_inputs(batch, num_chunks=num_chunks, shard_primary=False)

    def get_image_features(self, pixel_values: list[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Encode and project variable-size complete images in encounter order.

        Args:
            pixel_values: Nonempty list of floating [channels, height, width] images on the model device.

        Returns:
            Per-image floating [merged_rows * merged_columns, text_hidden] features.
        """
        features = self.vision_encoder(pixel_values)
        lengths = [x.shape[0] for x in features]
        normalized = self.pre_mm_projector_norm(torch.cat(features)).split(lengths)
        patch = self.vision_settings.patch_size
        sizes = [(p.shape[1] // patch, p.shape[2] // patch) for p in pixel_values]
        projected = self.vision_language_adapter(self.patch_merger(normalized, sizes))
        return projected.split([h * w // self.vision_settings.spatial_merge_size**2 for h, w in sizes])

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *pipeline_tensors: torch.Tensor,
        pixel_values: list[torch.Tensor] | torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
        cu_seqlens: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        qkv_format: str = "bshd",
        use_cache: bool = False,
        _mistral_packed_document_ids: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast | torch.Tensor | tuple[torch.Tensor, ...]:
        """Replace complete image grids and run the existing causal training path.

        Args:
            input_ids: Global integer [batch, sequence] or packed [tokens].
                Required for images; under CP, complete images are spliced before
                the first stage shards the sequence. Later PP stages take local residuals.
            pipeline_tensors: Propagated state in ``_pipeline_boundary_metas``
                layouts; non-first stages receive [batch, sequence, hidden]
                residuals in input_ids and skip the image encoder.
            pixel_values: Optional list of floating [channels, height, width] images in encounter order.
            inputs_embeds: Optional floating [batch, sequence, hidden] or [tokens, hidden],
                replacing the embedding lookup. IDs remain necessary for image alignment.
            position_ids: Optional nonnegative integers matching token axes;
                CP sharder supplies [batch, local_sequence] global positions.
            attention_mask: Binary keep mask or packed document IDs matching
                global token axes. CP combines both user padding masks here.
            padding_mask: Optional boolean padding mask matching token axes;
                CP sharder supplies [batch, local_sequence], including CP padding.
            labels: Optional integer targets matching token axes, -100 ignored;
                [batch, local_sequence] under CP. Image tokens and delimiters are
                masked on a copy before the existing shifted CE.
            logits_to_keep: Zero for all, positive tail length or int64 [kept_positions].
            output_hidden_states: Return final hidden states in addition to logits.
            cu_seqlens: Optional packed integer [documents + 1] boundaries.
            seq_lens: Optional real document lengths [batch, documents], with
                -1000 padding, retained by the packed sharder for document-count validation.
            qkv_format: bshd or thd; THD requires cu_seqlens.
            use_cache: Must be false, as in the causal training path.
            _mistral_packed_document_ids: Prepared integer [batch, local_sequence]
                document identity for prepared THD execution.

        Returns:
            CausalLMOutputWithPast with FP32 [batch, kept_sequence, vocab] capped logits,
            optional scalar loss and [batch, sequence, hidden] final activations.
            PP returns the tensor/tuple contract from ``_finish_pipeline_stage``.
            Inputs are never mutated. Text-only ranks execute one zero-contribution
            dummy image to keep vision FSDP forward/backward collectives symmetric.
        """
        if self.model.embed_tokens is None:
            if pixel_values is not None:
                raise ValueError("Only Mistral Large 4 PP stage zero may receive image pixels")
            return super().forward(
                input_ids,
                *pipeline_tensors,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                labels=labels,
                logits_to_keep=logits_to_keep,
                output_hidden_states=output_hidden_states,
                cu_seqlens=cu_seqlens,
                seq_lens=seq_lens,
                qkv_format=qkv_format,
                use_cache=use_cache,
                _mistral_packed_document_ids=_mistral_packed_document_ids,
            )
        if pipeline_tensors:
            raise ValueError("Mistral Large 4 PP stage zero does not accept propagated state")
        chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        if chunks is not None:
            if pixel_values is not None:
                raise ValueError("Use either staged PP media or explicit pixel_values")
            cursor = int(getattr(self, "_vlm_chunk_idx", 0) or 0)
            if cursor >= len(chunks):
                raise ValueError("Mistral Large 4 PP media cursor exceeds the prepared microbatches")
            pixel_values = chunks[cursor]
            self._vlm_chunk_idx = cursor + 1
        pixels = [] if pixel_values is None else pixel_values
        if isinstance(pixels, torch.Tensor) and pixels.ndim == 4:
            pixels = list(pixels.unbind(0))
        if not isinstance(pixels, list):
            raise ValueError("pixel_values must be a list of complete [channels, height, width] images")
        if input_ids is None and (inputs_embeds is None or pixels):
            raise ValueError("Provide input_ids for images, or inputs_embeds for text-only input")
        positions = None
        if _mistral_packed_document_ids is not None and cu_seqlens is not None:
            if cu_seqlens.ndim == 2 and cu_seqlens.shape[0] == 1:
                cu_seqlens = cu_seqlens[0]
            cu_seqlens = cu_seqlens[cu_seqlens >= 0]
        if input_ids is not None:
            positions = image_token_positions(
                input_ids,
                pixels,
                self.vision_settings,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
                padding_mask=padding_mask if self.cp_mesh is None else None,
            )
        hidden = self.model.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if input_ids is not None and hidden.shape != (*input_ids.shape, self.config.hidden_size):
            raise ValueError("inputs_embeds must match input_ids and hidden_size")
        if pixels:
            features = torch.cat(self.get_image_features([p.to(device=hidden.device) for p in pixels]))
            hidden = (
                hidden.reshape(-1, self.config.hidden_size)
                .index_copy(0, positions, features.to(hidden.dtype))
                .view_as(hidden)
            )
        else:
            size = self.vision_settings.patch_size * self.vision_settings.spatial_merge_size
            dummy = hidden.new_zeros((self.vision_settings.num_channels, size, size))
            hidden = hidden + self.get_image_features([dummy])[0].sum() * 0
        if self.cp_mesh is not None:
            if not self._is_pipeline_parallel_stage():
                # PP stage zero uses the same operation in the causal forward;
                # PP1 performs it here after complete-image validation/splicing.
                hidden, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, hidden)
                if attention_mask is not None:
                    attention_mask, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, attention_mask)
            if labels is not None and input_ids is not None:
                input_ids, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, input_ids)
        if labels is not None and input_ids is not None:
            special = (
                (input_ids == self.vision_settings.image_token_id)
                | (input_ids == self.vision_settings.image_break_token_id)
                | (input_ids == self.vision_settings.image_end_token_id)
            )
            labels = labels.masked_fill(special, -100)
        return super().forward(
            inputs_embeds=hidden,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            labels=labels,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            qkv_format=qkv_format,
            use_cache=use_cache,
            _mistral_packed_document_ids=_mistral_packed_document_ids,
        )

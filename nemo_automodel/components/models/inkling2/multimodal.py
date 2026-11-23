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

"""Image/text entry point sharing Inkling2's native decoder and checkpoint loader."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import BackendConfig

from .config import Inkling2Config, Inkling2TextConfig, resolve_quantization_config
from .model import Inkling2ForCausalLM
from .state_dict_adapter import Inkling2StateDictAdapter
from .vision import Inkling2VisionEncoder


class Inkling2ForConditionalGeneration(Inkling2ForCausalLM):
    """Insert unnormalized image features after the text embedding normalization."""

    config_class = Inkling2Config

    # Shared VLM pipeline staging installs one image chunk per microbatch and
    # clears these references after the schedule completes.
    _vlm_pixel_values_chunks: list[torch.Tensor] | None = None
    _vlm_image_grid_hws_chunks: list[torch.Tensor] | None = None
    _vlm_chunk_idx: int | None = None

    # The registry owns this class-valued hook; the image subclass narrows its flags.
    @dataclass(frozen=True)
    class ModelCapabilities(Inkling2ForCausalLM.ModelCapabilities):  # ty: ignore[invalid-attribute-override]
        """Image training uses FSDP2, EP, PP and unpacked CP; TP remains unsupported."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    def __init__(
        self, config: Inkling2Config, *, backend: BackendConfig | None = None, revision: str | None = None
    ) -> None:
        """Build the decoder and vision encoder from the released configuration.

        Args:
            config: Released image/text configuration; the sidecar declaration is resolved
                on it before the text decoder is built from its text configuration.
            backend: Kernel selection shared with the text decoder.
            revision: Resolved Hub revision of the checkpoint, for the sidecar read.
        """
        resolve_quantization_config(config, revision=revision)
        super().__init__(config.text_config, backend=backend)
        self.config = config
        assert self.embed_tokens is not None
        self.visual = Inkling2VisionEncoder(
            config.vision_config, config.text_config.hidden_size, dtype=self.embed_tokens.weight.dtype
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Inkling2StateDictAdapter(config)

    @classmethod
    def from_config(
        cls, config: Inkling2Config | Inkling2TextConfig, *, backend: BackendConfig | None = None
    ) -> "Inkling2ForConditionalGeneration":
        """Construct the registered multimodal architecture."""
        if not isinstance(config, Inkling2Config):
            raise TypeError("Inkling2 image/text construction requires Inkling2Config")
        return cls(config, backend=backend)

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize the decoder and vision encoder after meta materialization."""
        super().initialize_weights(buffer_device)
        if self.visual is not None:
            self.visual.reset_parameters()

    def customize_pipeline_stage_modules(
        self, module_names_per_stage: list[list[str]], *, layers_prefix: str, text_model: nn.Module | None = None
    ) -> list[list[str]]:
        """Keep the vision encoder and projector exclusively on the first stage.

        Args:
            module_names_per_stage: Generated stage FQNs; caller lists are unchanged.
            layers_prefix: Empty for the native model's top-level decoder layers.
            text_model: Backbone resolved by the shared splitter.

        Returns:
            Copied stage FQNs with one owner for vision, embedding normalization
            and the final residual query.
        """
        stages = super().customize_pipeline_stage_modules(
            module_names_per_stage, layers_prefix=layers_prefix, text_model=text_model
        )
        stages = [[name for name in stage if name != "visual"] for stage in stages]
        stages[0].append("visual")
        return stages

    def forward(
        self,
        input_ids: torch.Tensor,
        *pipeline_tensors: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        vision_patches_bthwc: torch.Tensor | None = None,
        vision_grid_thw: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | CausalLMOutputWithPast:
        """Fuse images into unpacked image/text sequences.

        Args:
            input_ids: Integer [batch, sequence] tokens, including image markers and bodies.
            pipeline_tensors: Non-first-stage int64 [batch, sequence, 2] layout,
                paired with residual history [batch, sequence, sources, hidden]
                in input_ids. Later stages use only these transported inputs.
            inputs_embeds: Optional decoder-ready embeddings [batch, sequence, hidden];
                mutually exclusive with image patches.
            vision_patches_bthwc: BF16 [patches, temporal, side, side, 3] image patches.
            vision_grid_thw: Integer [images, 3] row-major grid metadata.
            pixel_values: Shared VLM alias for vision_patches_bthwc, shape
                [patches, temporal, side, side, 3]. Mutually exclusive with it.
            image_grid_thw: Shared VLM alias for vision_grid_thw, shape [images, 3].
                Shared PP staging supplies patch/grid chunks on stage zero only.
            mm_token_type_ids: Integer/bool [batch, sequence], 1 only for image-body
                positions. Content-image headers share token ID 200005 but must remain 0.
            attention_mask: Optional [batch, sequence] real-token validity.
            padding_mask: Optional boolean [batch, sequence], true for padding.
            position_ids: Optional integer [batch, sequence] consecutive positions.
            logits_to_keep: Tail length or sequence indices for the shared loss path.
            output_hidden_states: Return final decoder features for external loss.

        Returns:
            The text decoder's FP32 logits or output object; inputs are not mutated.
        """
        if self.embed_tokens is None:
            # Every stage receives row-chunked token kwargs from the shared
            # schedule. Only stage zero owns media; later stages consume the
            # authoritative residual history and token layout from the wire.
            return super().forward(
                input_ids,
                *pipeline_tensors,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                logits_to_keep=logits_to_keep,
                output_hidden_states=output_hidden_states,
            )
        if pipeline_tensors:
            raise ValueError("Inkling2 first stage cannot receive pipeline residual history")
        if pixel_values is not None:
            if vision_patches_bthwc is not None:
                raise ValueError("Provide pixel_values or vision_patches_bthwc, not both")
            vision_patches_bthwc = pixel_values
        if image_grid_thw is not None:
            if vision_grid_thw is not None:
                raise ValueError("Provide image_grid_thw or vision_grid_thw, not both")
            vision_grid_thw = image_grid_thw
        if self._vlm_pixel_values_chunks is not None:
            index = self._vlm_chunk_idx or 0
            if (
                self._vlm_image_grid_hws_chunks is None
                or len(self._vlm_image_grid_hws_chunks) != len(self._vlm_pixel_values_chunks)
                or index >= len(self._vlm_pixel_values_chunks)
            ):
                raise ValueError("Inkling2 pipeline image chunks are missing or exhausted")
            if vision_patches_bthwc is not None or vision_grid_thw is not None:
                raise ValueError("Pipeline image chunks cannot be combined with direct image inputs")
            patches = self._vlm_pixel_values_chunks[index]
            grid = self._vlm_image_grid_hws_chunks[index]
            if patches.numel():
                vision_patches_bthwc, vision_grid_thw = patches, grid
            elif grid.numel():
                raise ValueError("Empty Inkling2 pipeline image chunk has nonempty grid metadata")
            self._vlm_chunk_idx = index + 1
        embedded = inputs_embeds
        if vision_patches_bthwc is not None:
            if inputs_embeds is not None:
                raise ValueError("Provide image patches or inputs_embeds, not both")
            if vision_grid_thw is None or mm_token_type_ids is None:
                raise ValueError("Inkling2 image inputs require grids and an explicit image-body mask")
            if mm_token_type_ids.shape != input_ids.shape or torch.any(
                (mm_token_type_ids != 0) & (mm_token_type_ids != 1)
            ):
                raise ValueError("Inkling2 image-body mask must be binary [batch, sequence]")
            mask = mm_token_type_ids.bool()
            if attention_mask is not None and torch.any(mask & ~attention_mask.bool()):
                raise ValueError("Inkling2 image features cannot occupy padding positions")
            if torch.any(input_ids[mask] != 200005):
                raise ValueError("Inkling2 image-body positions must contain token ID 200005")
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                if self.visual is None:
                    raise ValueError("Only the first Inkling2 pipeline stage owns the vision encoder")
                features = self.visual(vision_patches_bthwc, vision_grid_thw)
                if mask.sum().item() != features.shape[0]:
                    raise ValueError("Inkling2 image-body tokens and projected patch counts disagree")
                assert self.embed_tokens is not None
                assert self.embed_norm is not None
                text = self.embed_tokens(input_ids)
                # The release embeds text through embed_norm, then replaces image
                # body positions with the projector output without another norm.
                embedded = self.embed_norm(text.float()).to(text.dtype).masked_scatter(mask.unsqueeze(-1), features)
        elif vision_grid_thw is not None or (mm_token_type_ids is not None and torch.any(mm_token_type_ids != 0)):
            raise ValueError("Inkling2 image metadata requires image patches")
        if vision_patches_bthwc is None and self.visual is not None and self.visual.image_cp_mesh is not None:
            # Trainable vision is sharded per layer. Even a text-only DP peer
            # joins the same image calls, with empty features contributing zero.
            cfg = self.visual.config
            side = cfg.adapter_merge_size * cfg.native_patch_size
            patches = torch.empty(
                (0, cfg.temporal_patch_size, side, side, 3), device=input_ids.device, dtype=torch.bfloat16
            )
            grids = torch.empty((0, 3), device=input_ids.device, dtype=torch.long)
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                empty_features = self.visual(patches, grids)
                if embedded is None:
                    assert self.embed_norm is not None
                    text = self.embed_tokens(input_ids)
                    embedded = self.embed_norm(text.float()).to(text.dtype)
                embedded = embedded + empty_features.sum() * 0
        return super().forward(
            input_ids,
            inputs_embeds=embedded,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
        )

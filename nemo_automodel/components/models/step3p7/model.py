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
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.cp_sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.mtp import roll_tensor
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.gpt_oss.rope_utils import position_ids_to_freqs_cis
from nemo_automodel.components.models.step3p5.model import Step3p5Model
from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
from nemo_automodel.components.models.step3p7.mtp import build_mtp_config_from_hf, build_step3p5_mtp
from nemo_automodel.components.models.step3p7.state_dict_adapter import Step3p7StateDictAdapter
from nemo_automodel.components.models.step3p7.vision_encoder import StepRoboticsVisionEncoder
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

logger = logging.getLogger(__name__)


@dataclass
class Step3p7CausalLMOutput(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` plus optional per-depth MTP logits.

    Subclassing the HF ``ModelOutput`` gives this output the standard
    ``logits``/``hidden_states`` fields (so ``"hidden_states" in out`` and
    ``getattr(out, "hidden_states")`` behave like every other model and the
    fused-CE path can read the final hidden states), while the MTP fields stay
    declared dataclass fields so they survive output-restructuring layers like
    FSDP2's mixed-precision output cast, which rebuild ``ModelOutput``
    instances from declared fields only.
    """

    mtp_per_depth_logits: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


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

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    _keep_in_fp32_modules = ["rotary_emb"]

    _pp_keep_self_forward: bool = True
    mtp_outputs_are_logits = True
    # CP submesh, installed by the MoE parallelizer's apply_cp when context
    # parallelism is active; None means the forward embeds and shards nothing for CP.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

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
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        mtp_loss_scaling_factor = kwargs.pop("mtp_loss_scaling_factor", 0.1)
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
        dtype = get_dtype(getattr(config.text_config, "torch_dtype", "bfloat16"), torch.bfloat16)

        self.mtp_config = build_mtp_config_from_hf(config.text_config, loss_scaling_factor=mtp_loss_scaling_factor)
        if self.mtp_config.enabled:
            self.mtp = build_step3p5_mtp(
                config=config.text_config,
                mtp_config=self.mtp_config,
                backend=self.backend,
                moe_config=self.model.moe_config,
                dtype=dtype,
            )
        else:
            self.mtp = None

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Step3p7StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=dtype,
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

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        stage_modules = [list(modules) for modules in module_names_per_stage]
        if self.mtp is not None and "mtp" not in stage_modules[-1]:
            stage_modules[-1].append("mtp")
        return stage_modules

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        # Under context parallelism the first stage embeds the full sequence and
        # shards it in forward, so every stage output and later-stage input (incl.
        # the propagated MTP hidden states) carries the LOCAL (padded-to-2*cp then
        # //cp) sequence length while the first-stage token-id input stays full.
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_seq_len = seq_len
        if cp_size > 1:
            padded_seq_len = seq_len + (-seq_len) % (2 * cp_size)
            local_seq_len = padded_seq_len // cp_size

        hidden_shape = (microbatch_size, local_seq_len, self.config.text_config.hidden_size)
        vocab_shape = (microbatch_size, local_seq_len, self.config.text_config.vocab_size)
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)

        def meta(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dtype)

        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (meta(hidden_shape), *(meta(hidden_shape) for _ in range(mtp_depth)))

        primary_shape = vocab_shape if self.lm_head is not None else hidden_shape
        mtp_shape = vocab_shape if self.lm_head is not None else hidden_shape
        outputs_meta = (meta(primary_shape), *(meta(mtp_shape) for _ in range(mtp_depth)))
        return inputs_meta, outputs_meta

    def _is_pipeline_parallel_stage(self) -> bool:
        if self.lm_head is None:
            return True
        language_model = getattr(self.model, "language_model", None)
        if language_model is None:
            return False
        if getattr(language_model, "embed_tokens", None) is None:
            return True
        if not hasattr(language_model, "layers"):
            return False
        try:
            return len(language_model.layers) != int(self.config.text_config.num_hidden_layers)
        except TypeError:
            return False

    def _build_mtp_embed_inputs_from_embeds(self, inputs_embeds: torch.Tensor) -> tuple[torch.Tensor, ...]:
        cur_embeds = inputs_embeds
        embed_inputs = []
        for _ in range(self.mtp_config.num_layers):
            cur_embeds = roll_tensor(cur_embeds, shifts=-1, dim=1)
            embed_inputs.append(cur_embeds)
        return tuple(embed_inputs)

    def _make_position_ids(self, hidden: torch.Tensor, position_ids: torch.Tensor | None) -> torch.Tensor:
        if position_ids is not None:
            return position_ids
        return torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0).expand(hidden.shape[0], -1)

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return a sharder-only CP backend; embed + splice + shard happen in forward.

        Embedding and the vision multimodal scatter now run inside ``forward``
        per microbatch (see the CP branch that calls ``get_multimodal_embeddings``
        + ``prepare_inputs_embeds`` + :func:`shard_sequence_for_cp_round_robin`). The returned
        :class:`ContextParallelSharder` round-robin-shards only the no-grad aux
        streams (labels/position_ids/loss_mask/padding_mask) and leaves
        ``input_ids`` and the media inputs full-length for the forward. Step3.7
        uses plain 1-D positions, so no ``position_ids`` are computed here (the
        aux shard injects and slices them).

        Args:
            batch: The full-sequence batch (with ``input_ids`` ``[batch,
                sequence]``); left intact.
            num_chunks: Accepted for hook-signature parity; unused (round-robin CP).
        """
        if batch.get("input_ids") is None:
            raise ValueError("Step3p7 CP pre-embedding requires input_ids.")
        del num_chunks
        return {
            "cp_sharder": ContextParallelSharder._from_strategy(
                shard_batch_aux_only,
                round_robin_local_indices,
            )
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *mtp_embed_inputs: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> torch.Tensor | Step3p7CausalLMOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config.text_config, "output_hidden_states", False)
        )

        is_pp_stage = self._is_pipeline_parallel_stage()
        pp_mtp_enabled = is_pp_stage and self.mtp_config.enabled
        use_mtp = self.mtp is not None and self.training

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

        # Context-parallel: embed + vision-splice the full sequence, then keep this
        # rank's round-robin chunk pair (aux streams aligned by shard_batch_aux_only).
        # Downstream (MTP source embeds, decoder, lm_head) sees what the old
        # dispatch-level pre-embed produced, and stays differentiable.
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if (
            cp_size > 1
            and inputs_embeds is None
            and input_ids is not None
            and input_ids.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            and getattr(self.model.language_model, "embed_tokens", None) is not None
        ):
            if is_pp_stage and self.lm_head is None and kwargs.get("pixel_values") is not None:
                raise NotImplementedError(
                    "Step3p7 does not support image microbatch chunking under combined pipeline + "
                    "context parallelism; use a text-only batch for cp>1 and pp>1."
                )
            multimodal_embeddings = self.model.get_multimodal_embeddings(
                pixel_values=kwargs.get("pixel_values", None),
                patch_pixel_values=kwargs.get("patch_pixel_values", None),
                num_patches=kwargs.get("num_patches", None),
                image_embeds=kwargs.get("image_embeds", None),
            )
            inputs_embeds = self.model.prepare_inputs_embeds(input_ids, multimodal_embeddings)
            inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            input_ids = None
            # Media consumed into inputs_embeds; drop so self.model does not re-splice.
            for media_key in (
                "pixel_values",
                "patch_pixel_values",
                "num_patches",
                "image_embeds",
                "patch_newline_mask",
            ):
                kwargs.pop(media_key, None)

        need_mtp_source_embeds = (use_mtp or (pp_mtp_enabled and self.lm_head is None)) and not mtp_embed_inputs
        if need_mtp_source_embeds and inputs_embeds is None and input_ids is not None:
            if input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32):
                inputs_embeds = input_ids
            elif getattr(self.model.language_model, "embed_tokens", None) is not None:
                multimodal_embeddings = self.model.get_multimodal_embeddings(
                    pixel_values=kwargs.get("pixel_values", None),
                    patch_pixel_values=kwargs.get("patch_pixel_values", None),
                    num_patches=kwargs.get("num_patches", None),
                    image_embeds=kwargs.get("image_embeds", None),
                )
                inputs_embeds = self.model.prepare_inputs_embeds(input_ids, multimodal_embeddings)

        if pp_mtp_enabled and self.lm_head is None and not mtp_embed_inputs:
            if inputs_embeds is None:
                raise ValueError("First PP stage requires input_ids or inputs_embeds to build MTP embeddings.")
            mtp_embed_inputs = self._build_mtp_embed_inputs_from_embeds(inputs_embeds)

        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep).logits

        if pp_mtp_enabled and self.lm_head is None:
            return (logits, *mtp_embed_inputs)

        mtp_per_depth_logits = None
        if use_mtp:
            if is_pp_stage and not mtp_embed_inputs:
                raise ValueError("Final PP stage requires propagated MTP embeddings")
            position_ids = self._make_position_ids(hidden_states, position_ids)
            freqs_cis = position_ids_to_freqs_cis(
                self.model.language_model.rotary_emb,
                position_ids,
                qkv_format=kwargs.get("qkv_format", "bshd"),
                for_fused_rope=self.backend.rope_fusion,
                cp_size=kwargs.get("cp_size", 1),
            )
            mtp_kwargs = {
                "hidden_states": hidden_states,
                "freqs_cis": freqs_cis,
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "padding_mask": padding_mask,
            }
            if mtp_embed_inputs:
                mtp_kwargs["embed_inputs"] = tuple(mtp_embed_inputs)
            elif inputs_embeds is not None:
                mtp_kwargs["embed_inputs"] = self._build_mtp_embed_inputs_from_embeds(inputs_embeds)
            else:
                mtp_kwargs["input_ids"] = input_ids
                mtp_kwargs["embed_fn"] = self.model.language_model.embed_tokens
            mtp_per_depth_logits = self.mtp(**mtp_kwargs)
        elif pp_mtp_enabled and self.lm_head is not None:
            mtp_per_depth_logits = [
                logits.new_empty(logits.shape) for _ in range(int(getattr(self.mtp_config, "num_layers", 0) or 0))
            ]

        if is_pp_stage:
            if pp_mtp_enabled:
                if self.training and self.mtp is None:
                    raise ValueError("Final PP stage has MTP enabled but does not own the MTP module")
                return (logits, *mtp_per_depth_logits)
            return logits

        if mtp_per_depth_logits is None:
            # Preserve the historical bare-tensor return on the default path so
            # existing callers/tests are unaffected; only wrap into a
            # ``ModelOutput`` (carrying the final hidden states) when the
            # fused-CE path asks for hidden states via ``output_hidden_states``.
            if not output_hidden_states:
                return logits
            return Step3p7CausalLMOutput(logits=logits, hidden_states=hidden_states)

        return Step3p7CausalLMOutput(
            logits=logits,
            hidden_states=(hidden_states if output_hidden_states else None),
            mtp_per_depth_logits=mtp_per_depth_logits,
            mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
        )

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
            if self.mtp is not None:
                for sublayer in self.mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)

        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.language_model.rotary_emb.device = buffer_device


ModelClass = Step3p7ForConditionalGeneration

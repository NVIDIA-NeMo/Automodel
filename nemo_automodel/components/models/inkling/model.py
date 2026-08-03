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

"""NeMo AutoModel wrapper for the Inkling multimodal MoE model.

The wrapper reuses the HuggingFace ``InklingForConditionalGeneration`` towers,
attention, norms, embeddings, and language-model head. Decoder feed-forwards
retain the raw checkpoint's fused interleaved projection layout; sparse layers
also use an expert-parallel :class:`InklingMoE`. This avoids full-size weight
conversion copies while preserving Transformers numerics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta

from .configuration import InklingConfig

_INKLING_HF_UNAVAILABLE_MSG = (
    "The Inkling model requires a transformers build that provides "
    "transformers.models.inkling and transformers.masking_utils.create_recurrent_attention_mask "
    "(transformers >= 5.14)."
)


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": _INKLING_HF_UNAVAILABLE_MSG})


try:
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import (
        create_causal_mask,
        create_recurrent_attention_mask,
        create_sliding_window_causal_mask,
    )
    from transformers.modeling_outputs import BaseModelOutputWithPast
    from transformers.models.inkling.configuration_inkling import InklingConfig as HFInklingConfig
    from transformers.models.inkling.modeling_inkling import (
        InklingForConditionalGeneration as HFInklingForConditionalGeneration,
    )
    from transformers.models.inkling.modeling_inkling import InklingMLP as HFInklingMLP
    from transformers.models.inkling.modeling_inkling import InklingMoE as HFInklingMoE
    from transformers.models.inkling.modeling_inkling import InklingTextModel as HFInklingTextModel

    InklingConfig = HFInklingConfig
    _INKLING_HF_AVAILABLE = True
except ImportError:  # transformers < 5.14 ships neither symbol set
    DynamicCache = _make_missing("DynamicCache")
    BaseModelOutputWithPast = _make_missing("BaseModelOutputWithPast")
    HFInklingForConditionalGeneration = _make_missing("HFInklingForConditionalGeneration")
    HFInklingMLP = _make_missing("HFInklingMLP")
    HFInklingMoE = _make_missing("HFInklingMoE")
    HFInklingTextModel = _make_missing("HFInklingTextModel")
    create_causal_mask = _make_missing("create_causal_mask")
    create_recurrent_attention_mask = _make_missing("create_recurrent_attention_mask")
    create_sliding_window_causal_mask = _make_missing("create_sliding_window_causal_mask")
    _INKLING_HF_AVAILABLE = False

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import InklingStateDictAdapter

if _INKLING_HF_AVAILABLE:
    from .cp_attention import attach_inkling_upipe_attention
    from .layers import InklingDenseMLP, InklingMoE, InklingShortConvolution, build_inkling_moe_config
else:
    attach_inkling_upipe_attention = _make_missing("attach_inkling_upipe_attention")
    InklingDenseMLP = _make_missing("InklingDenseMLP")
    InklingMoE = _make_missing("InklingMoE")
    InklingShortConvolution = _make_missing("InklingShortConvolution")
    build_inkling_moe_config = _make_missing("build_inkling_moe_config")


def _full_padding_mask(attention_mask: Any) -> torch.Tensor | None:
    """Reduce an incoming attention mask to the 2D keep-map the convolutions consume."""
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask.get("linear_attention")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        return None
    return attention_mask.bool()


class InklingTextModel(HFInklingTextModel):
    """Inkling text backbone that also accepts AutoModel pipeline stages."""

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if inputs_embeds is None:
            if self.embed_tokens is None:
                if input_ids is None or not input_ids.dtype.is_floating_point:
                    raise ValueError("Pipeline stages without embeddings require hidden states as input")
                inputs_embeds = input_ids
                input_ids = None
            else:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))
        elif input_ids is not None:
            raise ValueError("You must provide exactly one of input_ids or inputs_embeds")

        use_cache = getattr(self.config, "use_cache", False) if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        if getattr(self, "_cp_enabled", False):
            # Under CP the attention masks are rebuilt inside the UPipe attention from
            # global token indices, so only the padding map has to travel. It stays at
            # full length: the K/V convolutions run after the all-to-all and need padding
            # information for positions this rank does not own.
            causal_mask_mapping = {
                "full_attention": None,
                "sliding_attention": None,
                "linear_attention": _full_padding_mask(attention_mask),
            }
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
                "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        layers = self.layers.values() if isinstance(self.layers, nn.ModuleDict) else self.layers
        for decoder_layer in layers:
            attention_type = getattr(decoder_layer, "attention_type", None)
            if attention_type is None:
                # Transformers 5.14 renamed this field and uses Inkling layer types.
                layer_type = decoder_layer.layer_type
                if layer_type == "hybrid":
                    attention_type = "full_attention"
                elif layer_type == "hybrid_sliding":
                    attention_type = "sliding_attention"
                else:
                    raise ValueError(f"Unsupported Inkling layer type: {layer_type!r}")
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[attention_type],
                conv_mask=causal_mask_mapping["linear_attention"],
                past_key_values=past_key_values,
                **kwargs,
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class InklingForConditionalGeneration(HFCheckpointingMixin, HFInklingForConditionalGeneration, MoEFSDPSyncMixin):
    """Inkling VLM with expert-parallel MoE feed-forwards."""

    _msg = _INKLING_HF_UNAVAILABLE_MSG
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # The adapter covers every checkpoint tensor. Avoid initializing the 975B
    # sharded model immediately before loading it, which can also introduce
    # stage-divergent DTensor collectives under PP.
    _skip_init_weights_on_load: bool = True

    # Keep the multimodal forward under PP so stage 0 can consume the media
    # chunks staged by the VLM recipe.
    _pp_keep_self_forward: bool = True

    # Short convolutions and router correction bias use callable fp32 holders.
    _keep_in_fp32_modules_strict = ["_fp32_params"]

    # Attention is model-owned under CP (head-chunked UPipe over FlexAttention), so the
    # runtime's SDPA-backend gate is the right one to pass.
    _supports_cp_sdpa: bool = True

    # CP submesh, installed by the parallelizer when context parallelism is active.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        # CP is supported through the head-chunked UPipe attention in cp_attention.py;
        # the batch is sharded contiguously (see prepare_model_inputs_for_cp).
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> "InklingForConditionalGeneration":
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> "InklingForConditionalGeneration":
        if not _INKLING_HF_AVAILABLE:
            raise UnavailableError(_INKLING_HF_UNAVAILABLE_MSG)
        config = InklingConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> None:
        if not _INKLING_HF_AVAILABLE:
            raise UnavailableError(_INKLING_HF_UNAVAILABLE_MSG)
        reject_unsupported_tie_word_embeddings(type(self), config)
        backend = backend or BackendConfig()

        # Propagate the requested top-level dtype to the nested sub-configs so the
        # HF towers and our MoE parameters are constructed in a consistent dtype.
        top_dtype = getattr(config, "torch_dtype", None)
        if top_dtype is not None:
            for sub_cfg in vars(config).values():
                if sub_cfg is not config and hasattr(sub_cfg, "torch_dtype"):
                    sub_cfg.torch_dtype = top_dtype

        super().__init__(config)

        self.backend = backend
        self.model.language_model.__class__ = InklingTextModel
        # Router scoring is selection-sensitive; keep it in fp32 unless overridden.
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        text_config = config.text_config
        self.moe_config = moe_config or build_inkling_moe_config(text_config, backend)
        # Exposed on the inner model too so the parallelizer can discover it.
        self.model.moe_config = self.moe_config

        # Keep every feed-forward in the raw checkpoint's fused interleaved
        # layout. This lets distributed checkpoint loading write through
        # transpose views instead of allocating converted expert tensors.
        for layer in self.model.language_model.layers:
            layer.self_attn.k_sconv = InklingShortConvolution(layer.self_attn.k_sconv)
            layer.self_attn.v_sconv = InklingShortConvolution(layer.self_attn.v_sconv)
            layer.attn_sconv = InklingShortConvolution(layer.attn_sconv)
            layer.mlp_sconv = InklingShortConvolution(layer.mlp_sconv)
            # Inert until the parallelizer hands over a CP mesh.
            attach_inkling_upipe_attention(layer.self_attn)
            if isinstance(layer.mlp, HFInklingMoE):
                layer.mlp = InklingMoE(text_config, backend, moe_config=self.moe_config)
            elif isinstance(layer.mlp, HFInklingMLP):
                layer.mlp = InklingDenseMLP(text_config)

        model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = InklingStateDictAdapter(
                text_config,
                self.moe_config,
                self.backend,
                dtype=model_dtype,
            )

        # Custom construction bypasses Transformers' from_pretrained dtype plan.
        # Normalize inherited and replacement modules here while retaining the
        # short convolutions and router correction-bias holders in strict fp32.
        cast_model_to_dtype(self, model_dtype)

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module,
    ) -> list[list[str]]:
        """Keep Inkling's post-embedding norm on the first pipeline stage."""
        del text_model
        module_names_per_stage[0].append(f"{layers_prefix}embed_norm")
        return module_names_per_stage

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Hand the recipe Inkling's own context-parallel batch sharding.

        Sharding is contiguous rather than load-balanced round-robin: the UPipe
        all-to-all reassembles the sequence in rank order, and the residual-stream
        convolutions need contiguous neighbors. ``attention_mask`` is left full length so
        the post-all-to-all convolutions can see padding outside this rank's shard.

        Args:
            batch: Full-sequence batch; left untouched until the returned sharder runs.
            num_chunks: Accepted for CP hook signature parity; Inkling uses one
                contiguous shard per rank.

        Returns:
            dict: Batch updates carrying the model-owned context-parallel sharder.
        """
        from nemo_automodel.components.distributed.context_parallel.sharder import (  # noqa: PLC0415
            ContextParallelSharder,
            contiguous_local_indices,
        )

        from .cp import setup_inkling_cp, shard_batch_for_inkling_cp  # noqa: PLC0415

        cp_mesh = getattr(self, "cp_mesh", None)
        if cp_mesh is None:
            raise RuntimeError("Inkling context-parallel input preparation requires a CP mesh.")
        setup_inkling_cp(self, cp_mesh)

        del batch, num_chunks
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_for_inkling_cp,
                local_token_global_indices=contiguous_local_indices,
            )
        }

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return PP metadata using Inkling's unpadded runtime vocabulary."""
        text_config = self.config.text_config
        hidden_shape = (microbatch_size, seq_len, text_config.hidden_size)
        vocab_size = text_config.unpadded_vocab_size or text_config.vocab_size

        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (torch.empty(*hidden_shape, device="meta", dtype=dtype),)

        if self.lm_head is None:
            outputs_meta = (torch.empty(*hidden_shape, device="meta", dtype=dtype),)
        else:
            head_dtype = self.lm_head.weight.dtype
            outputs_meta = (torch.empty(microbatch_size, seq_len, vocab_size, device="meta", dtype=head_dtype),)
        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Any:
        """Run the standard Inkling forward or its pipeline-stage equivalent."""
        language_model = self.model.language_model
        if not isinstance(language_model.layers, nn.ModuleDict):
            return super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                audio_input_ids=audio_input_ids,
                audio_input_ids_mask=audio_input_ids_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if inputs_embeds is None and input_ids is not None and input_ids.dtype.is_floating_point:
            inputs_embeds = input_ids
            input_ids = None

        is_first_stage = language_model.embed_tokens is not None
        if pixel_values is None and is_first_stage:
            chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
            if chunks is not None and chunk_idx is not None and chunk_idx < len(chunks):
                pixel_values = chunks[chunk_idx]
                self._vlm_chunk_idx = chunk_idx + 1

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        if self.lm_head is None:
            return hidden_states

        hidden_states = hidden_states / self.config.text_config.logits_mup_width_multiplier
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        unpadded_vocab_size = self.config.text_config.unpadded_vocab_size
        if unpadded_vocab_size is not None and unpadded_vocab_size < logits.shape[-1]:
            logits = logits[..., :unpadded_vocab_size]
        return logits

    def update_moe_gate_bias(self) -> None:
        """Inkling uses a trained correction bias, so gate-bias updates are a no-op."""
        return

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

"""Native full-sequence Flint causal language model with explicit training scope."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from nemo_automodel.components.distributed.context_parallel.sharder import shard_sequence_for_cp_round_robin
from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, generation_config_from_model_config
from nemo_automodel.components.models.flint.config import FlintConfig
from nemo_automodel.components.models.flint.layers import FlintDecoderLayer
from nemo_automodel.components.models.flint.magi import FlintMagiBatchPlans
from nemo_automodel.components.models.flint.packing import _prepare_inputs_for_cp
from nemo_automodel.components.models.flint.parallelization import FlintModelParallelizer
from nemo_automodel.components.models.flint.state_dict_adapter import FlintStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str


class FlintModel(nn.Module):
    """Untied input embedding, scheduled decoder blocks and final RMSNorm."""

    _flint_preserve_output_dtype: bool = True

    def __init__(self, config: FlintConfig, backend: BackendConfig) -> None:
        """Create the backbone with explicit parameter dtype and layer schedule.

        Args:
            config: Validated architecture settings.
            backend: Eager/torch settings passed to every block.
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.residual_dtype = config.residual_dtype
        dtype = dtype_from_str(config.dtype, torch.float32)
        # EOS can also occur as a valid input, so it is not a frozen embedding row.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            [FlintDecoderLayer(config, index, backend) for index in range(config.num_hidden_layers)]
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        **attn_kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Evaluate full-sequence decoder states without an inference cache.

        Args:
            input_ids: Integer tensor of shape [batch, sequence], or None when
                inputs_embeds is supplied.
            inputs_embeds: Alternative [batch, sequence, hidden] tensor or THD
                [tokens, hidden]; later PP stages receive this local residual.
            position_ids: Optional integer [batch, sequence] or THD [tokens].
            attention_mask: Optional binary tensor of shape [batch, sequence];
                one means valid, independently of whether a token equals EOS.
            padding_mask: Optional True-for-padding expert mask [batch, sequence]
                or THD [tokens], derived from physical document lengths.
            output_hidden_states: Retain embedding, intermediate and final states.
            **attn_kwargs: Backend THD boundaries/formats or immutable Magi plan.

        Returns:
            Output with last_hidden_state [batch, sequence, hidden], optional
            tuple of hidden states in the same layout, and no cache/attentions.
            Padded-query values have no training-target meaning.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Flint requires exactly one of input_ids or inputs_embeds")
        if input_ids is not None:
            expected_rank = 1 if attn_kwargs.get("qkv_format") == "thd" else 2
            if input_ids.ndim != expected_rank or input_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Flint input_ids must be integer [batch, sequence]")
            if self.embed_tokens is None:
                raise ValueError("A non-first Flint pipeline stage requires local floating hidden states")
            hidden_states = self.embed_tokens(input_ids)
        else:
            if (
                inputs_embeds is None
                or inputs_embeds.ndim != (2 if attn_kwargs.get("qkv_format") == "thd" else 3)
                or inputs_embeds.shape[-1] != self.hidden_size
            ):
                raise ValueError("Flint inputs_embeds must have shape [batch, sequence, hidden]")
            hidden_states = inputs_embeds
        if self.residual_dtype is not None:
            hidden_states = hidden_states.to(torch.float32 if self.residual_dtype == "fp32" else torch.bfloat16)
        states = (hidden_states,) if output_hidden_states else None
        layers = self.layers.values() if isinstance(self.layers, nn.ModuleDict) else self.layers
        for index, layer in enumerate(layers):
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
            if states is not None and index < len(self.layers) - 1:
                states += (hidden_states,)
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        if states is not None:
            states += (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=states)


class FlintForCausalLM(HFCheckpointingMixin, PreTrainedModel, MoEFSDPSyncMixin):
    """Eager Flint baseline, with protected trainable sinks and untied FP32 head.

    CPU FP32 correctness is the initial supported execution contract. GPU
    precision, inference caches, packed sequences and distributed axes require
    their own validation before being advertised.
    """

    config_class = FlintConfig
    parallelizer: FlintModelParallelizer = FlintModelParallelizer()
    base_model_prefix = "model"
    _no_split_modules = ["FlintDecoderLayer"]
    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True
    _packed_cp_attn_backends = ("te", "magi")
    supports_thd = True
    packed_mask_type = "document_ids"
    _keep_in_fp32_modules = ["_fp32_params", "lm_head"]
    _keep_in_fp32_modules_strict = ["_fp32_params", "lm_head"]
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared native axes; real numerical gates certify each configuration."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    def __init__(self, config: FlintConfig, backend: BackendConfig | None = None) -> None:
        """Build the source architecture with an explicit CE-only objective.

        Args:
            config: Original local config; the tying guard runs before use.
            backend: Kernel choices, defaulting to the verified torch path.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        if config.router_aux_loss_coef != 0:
            raise ValueError(
                "Flint's original auxiliary-loss formula is unspecified; explicitly set router_aux_loss_coef=0.0 "
                "for the CE-only baseline"
            )
        super().__init__(config)
        self.backend = backend or BackendConfig(
            attn="eager",
            linear="torch",
            rms_norm="torch",
            rope="torch",
            rope_fusion=False,
            experts="torch",
            dispatcher="torch",
            gate_precision=None,
        )
        self.cp_mesh: DeviceMesh | None = None
        self.model = FlintModel(config, self.backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32)
        self.lm_head._flint_fp32_output_head = True
        # Seeded from the model config like PreTrainedModel does for generating
        # models, so bos/eos/pad are set; the loader then restores the checkpoint's
        # generation_config.json and the consolidated export writes it back out.
        self.generation_config = generation_config_from_model_config(config)
        self.vocab_size = config.vocab_size
        self.moe_config = next((layer.moe_config for layer in self.model.layers if layer.moe_config is not None), None)
        self.state_dict_adapter = FlintStateDictAdapter(
            config, self.moe_config, self.backend, dtype=dtype_from_str(config.dtype, torch.float32)
        )
        self.post_init()

    @classmethod
    def from_config(cls, config: FlintConfig, *, backend: BackendConfig | None = None) -> FlintForCausalLM:
        """Construct the native model through the config-owned build boundary.

        Args:
            config: Local architecture settings.
            backend: Runtime kernel choices.

        Returns:
            Initialized native model, without loading any checkpoint weights.
        """
        return config.build(backend=backend)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | PathLike[str], *model_args: object, **kwargs: object
    ) -> FlintForCausalLM:
        """Reject the inherited HF loader, which cannot convert native MoE layouts.

        Args:
            pretrained_model_name_or_path: Source checkpoint directory or Hub ID.
            *model_args: Upstream loader positional arguments.
            **kwargs: Upstream loader arguments.

        Returns:
            No model; use NeMoAutoModelForCausalLM.from_pretrained instead.

        Raises:
            ValueError: Always, with the adapter-aware public entry point.
        """
        raise ValueError("Load Flint with NeMoAutoModelForCausalLM.from_pretrained so checkpoint conversion is applied")

    @torch.no_grad()
    def initialize_weights(self, *, dtype: torch.dtype | None = None) -> None:
        """Initialize every trainable family after construction or meta materialization.

        Args:
            dtype: Optional bulk parameter dtype; sinks and the head stay FP32.
                Installed PEFT adapters retain their explicitly selected dtype.
        """
        if dtype is not None:
            if dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("Flint bulk parameters support only float32 and bfloat16")
            parameters = tuple(self.named_parameters())
            has_adapters = any(any(part.startswith("lora_") for part in name.split(".")) for name, _ in parameters)
            if has_adapters:
                # The public meta loader materializes after PEFT and FSDP have
                # installed independent adapter storage/compute policies. A
                # recursive Module.to() here would silently cast those masters,
                # including direct grouped-expert Parameters, to the bulk dtype.
                mismatched = [
                    name
                    for name, parameter in parameters
                    if parameter.is_floating_point()
                    and parameter.dtype != dtype
                    and not any(part.startswith("lora_") for part in name.split("."))
                    and "_fp32_params" not in name.split(".")
                    and "lm_head" not in name.split(".")
                ]
                if mismatched:
                    raise ValueError(
                        "Set Flint's bulk dtype before installing PEFT adapters; changing it during "
                        f"initialization would invalidate adapter/FSDP dtype policies (examples={mismatched[:3]})"
                    )
            else:
                cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params", "lm_head"))
            self.state_dict_adapter.dtype = dtype
        device = next(self.parameters()).device
        if self.model.embed_tokens is not None:
            nn.init.normal_(self.model.embed_tokens.weight, std=self.config.initializer_range)
        layers = self.model.layers.values() if isinstance(self.model.layers, nn.ModuleDict) else self.model.layers
        for layer in layers:
            layer.init_weights(device, self.config.initializer_range)
        if self.model.norm is not None:
            self.model.norm.reset_parameters()
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=self.config.initializer_range)

    def _is_pipeline_parallel_stage(self) -> bool:
        """Identify a real partial stage using retained modules and active depth."""
        return (
            self.model.embed_tokens is None
            or self.lm_head is None
            or len(self.model.layers) != self.config.num_hidden_layers
        )

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Prepare global [B,S] tokens and [B,N] document lengths before PP splitting.

        Args:
            batch: Global token, target and position tensors; packed batches
                include real/padded document lengths and qkv_format='thd'.
            num_chunks: Independent PP microbatch token spaces.

        Returns:
            Model-owned shared sharder and padding metadata; local coordinates
            stay [Tlocal] or [chunks,Tlocal], while PP stage-zero IDs stay global.
        """
        return _prepare_inputs_for_cp(self, batch, num_chunks=num_chunks)

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Describe PP token/residual/logit boundaries without a probe forward.

        Args:
            is_first: Whether this is the first physical stage.
            microbatch_size: Number of rows in one PP microbatch.
            seq_len: Global sequence extent, including dataset physical padding.
            dtype: Framework-selected bulk compute dtype.

        Returns:
            Input/output meta tuples: global int64 IDs [B,S] on stage zero,
            local residuals [B,S/cp,H] between stages, and FP32 logits [B,S/cp,V]
            or FP32 hidden states [B,S/cp,H] for a linear fused loss.
        """
        residual_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(self.config.residual_dtype, dtype)
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_length = (seq_len + (-seq_len) % (2 * cp_size)) // cp_size if cp_size > 1 else seq_len
        input_shape = (
            (microbatch_size, seq_len) if is_first else (microbatch_size, local_length, self.config.hidden_size)
        )
        inputs = (torch.empty(*input_shape, dtype=torch.long if is_first else residual_dtype, device="meta"),)
        final = self.lm_head is not None
        width = (
            self.config.hidden_size
            if not final or getattr(self, "_pp_return_hidden_states", False)
            else self.vocab_size
        )
        outputs = (
            torch.empty(
                microbatch_size, local_length, width, dtype=torch.float32 if final else residual_dtype, device="meta"
            ),
        )
        return inputs, outputs

    def get_input_embeddings(self) -> nn.Embedding:
        """Return the independently trainable input embedding module.

        Returns:
            Embedding with weight [vocab, hidden], with no frozen padding row.
        """
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        """Replace the input module without changing the output-head parameter.

        Args:
            value: Embedding module with weight [vocab, hidden].
        """
        if (
            value.weight.shape != (self.config.vocab_size, self.config.hidden_size)
            or value.weight.dtype != self.model.embed_tokens.weight.dtype
            or value.padding_idx is not None
            or value.weight is self.lm_head.weight
            or (
                not value.weight.is_meta
                and not self.lm_head.weight.is_meta
                and value.weight.device == self.lm_head.weight.device
                and value.weight.untyped_storage().data_ptr() == self.lm_head.weight.untyped_storage().data_ptr()
            )
        ):
            raise ValueError(
                "Flint input embeddings must preserve vocabulary, bulk dtype, untied storage and trainable EOS"
            )
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        """Return the separate FP32 vocabulary projection module.

        Returns:
            Bias-free linear module with weight [vocab, hidden].
        """
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        """Replace the output module under the protected-FP32 contract.

        Args:
            value: Bias-free linear module with FP32 weight [vocab, hidden].
        """
        if (
            value.weight.shape != (self.config.vocab_size, self.config.hidden_size)
            or value.weight.dtype != torch.float32
            or value.bias is not None
            or value.weight is self.model.embed_tokens.weight
            or (
                not value.weight.is_meta
                and not self.model.embed_tokens.weight.is_meta
                and value.weight.device == self.model.embed_tokens.weight.device
                and value.weight.untyped_storage().data_ptr()
                == self.model.embed_tokens.weight.untyped_storage().data_ptr()
            )
        ):
            raise ValueError(
                "Flint output embeddings require an untied, bias-free FP32 linear module with fixed vocabulary"
            )
        value._flint_fp32_output_head = True
        self.lm_head = value

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        """Preserve the fixed vocabulary used by checkpoint conversion.

        Args:
            new_num_tokens: Vocabulary resizing is not supported by this baseline.
            pad_to_multiple_of: Vocabulary padding is not supported.
            mean_resizing: Upstream initialization option, unused without resizing.

        Returns:
            Existing embedding [vocab, hidden] when no resize is requested.

        Raises:
            ValueError: A vocabulary resize or padding operation is requested.
        """
        if new_num_tokens is not None or pad_to_multiple_of is not None:
            raise ValueError(
                "Flint's baseline uses a fixed checkpoint vocabulary; token embedding resizing is unsupported"
            )
        return self.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Cache | None = None,
        output_hidden_states: bool = False,
        logits_to_keep: int = 0,
        skip_logits: bool = False,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast | torch.Tensor:
        """Return FP32 logits for full-sequence, padded causal language modeling.

        Args:
            input_ids: Integer [batch, sequence], THD [tokens], or floating
                CP-local [batch, sequence, hidden] residuals on later PP stages.
            attention_mask: Binary [batch, sequence] validity, or nonnegative
                integer document IDs for eager/SDPA packed attention.
            position_ids: Optional integer tensor [batch, sequence].
            inputs_embeds: Alternative floating tensor [batch, sequence, hidden].
            labels: Optional unshifted HF-style labels [batch, sequence], with -100
                ignored. This path shifts once. Production calculate_loss consumes
                already shifted labels separately and must not pass them here.
            use_cache: True is unsupported by this full-sequence training baseline.
            past_key_values: Inference cache; only None is supported.
            output_hidden_states: Include all [batch, sequence, hidden] states,
                except logits_to_keep=1 selects the final FP32 hidden tensor for
                the shared linear-loss contract without retaining every layer.
            logits_to_keep: Zero projects all positions; a positive value projects
                that many final positions. Labels require all positions.
            skip_logits: Return only final FP32 hidden states for an external
                already-shifted linear loss; do not run the vocabulary projection.
            padding_mask: Boolean [batch, sequence] or THD [tokens], True for
                physically padded expert rows; document lengths own its meaning.
            **attn_kwargs: Shared TE THD boundaries/formats or a model-owned Magi
                plan. PP stage-zero global IDs are selected by prepared local
                indices once; later stages already receive local residuals.

        Returns:
            CausalLMOutputWithPast with FP32 logits [batch, selected_sequence, vocab],
            optional scalar CE loss, optional hidden states and no inference cache.
        """
        if use_cache or past_key_values is not None:
            raise ValueError("Flint's native baseline supports full-sequence training without KV caching")
        if not isinstance(logits_to_keep, int) or isinstance(logits_to_keep, bool) or logits_to_keep < 0:
            raise ValueError("Flint logits_to_keep must be a nonnegative integer")
        if labels is not None and logits_to_keep:
            raise ValueError("Flint labels require logits_to_keep=0")
        if labels is not None and skip_logits:
            raise ValueError("Flint skip_logits consumes already-shifted labels through a separate linear loss")
        final_hidden_only = skip_logits or (output_hidden_states and logits_to_keep == 1)
        received_residual = inputs_embeds is None and input_ids is not None and torch.is_floating_point(input_ids)
        if received_residual:
            inputs_embeds, input_ids = input_ids, None
        is_thd = attn_kwargs.get("qkv_format") == "thd"
        if labels is not None and (is_thd or (self.cp_mesh is not None and self.cp_mesh.size() > 1)):
            raise ValueError("Packed/CP Flint uses already-shifted labels through the external linear or masked loss")
        magi_plan = attn_kwargs.get("_flint_magi_plan")
        if isinstance(magi_plan, FlintMagiBatchPlans):
            attn_kwargs["_flint_magi_plan"] = magi_plan.select(attn_kwargs.pop("_flint_magi_microbatch"))
        local_indices = attn_kwargs.pop("_thd_local_indices", None)
        if is_thd:
            if input_ids is not None and self._is_pipeline_parallel_stage():
                if local_indices is None:
                    raise ValueError("Flint packed PP stage-zero IDs require CP-prepared local-token indices")
                input_ids = input_ids.reshape(-1).index_select(0, local_indices.reshape(-1).long())
            for name, value in tuple(attn_kwargs.items()):
                if isinstance(value, torch.Tensor):
                    value = value.squeeze(0) if value.ndim > 1 else value
                    if name in ("cu_seqlens", "cu_seqlens_padded"):
                        value = value[value != -1000].contiguous()
                    attn_kwargs[name] = int(value.item()) if name == "max_seqlen" else value
            if input_ids is not None:
                input_ids = input_ids.reshape(-1)
            if inputs_embeds is not None and inputs_embeds.ndim == 3 and inputs_embeds.shape[0] == 1:
                inputs_embeds = inputs_embeds.squeeze(0)
            if position_ids is not None:
                position_ids = position_ids.reshape(-1)
            if padding_mask is not None:
                padding_mask = padding_mask.reshape(-1)
            attention_mask = None
        elif attention_mask is not None and bool((attention_mask > 1).any()):
            if attention_mask.dtype not in (torch.int32, torch.int64) or bool((attention_mask < 0).any()):
                raise ValueError("Flint packed document IDs must be nonnegative integers")
            attn_kwargs["document_ids"] = attention_mask
            if position_ids is None:
                indices = torch.arange(attention_mask.shape[1], device=attention_mask.device).expand_as(attention_mask)
                changes = F.pad(attention_mask[:, 1:] != attention_mask[:, :-1], (1, 0), value=True)
                starts = torch.where(changes, indices, 0).cummax(-1).values
                position_ids = indices - starts
            attention_mask = attention_mask > 0
        if not is_thd and not received_residual and self.cp_mesh is not None and self.cp_mesh.size() > 1:
            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)
                input_ids = None
            global_length = inputs_embeds.shape[1]
            inputs_embeds, indices, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            if position_ids is None:
                position_ids = torch.arange(global_length, device=inputs_embeds.device).expand(
                    inputs_embeds.shape[0], -1
                )
            if position_ids.shape[-1] == global_length:
                position_ids = F.pad(position_ids, (0, int(indices.max()) + 1 - global_length)).index_select(
                    -1, indices
                )
            attention_mask = None
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            output_hidden_states=output_hidden_states and not final_hidden_only,
            **attn_kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if is_thd:
            hidden_states = hidden_states.unsqueeze(0)
        if getattr(self, "_pp_return_hidden_states", False):
            if labels is not None:
                raise ValueError("Flint PP linear loss receives already-shifted targets separately")
            return hidden_states.float()
        if skip_logits:
            return CausalLMOutputWithPast(hidden_states=hidden_states.float())
        if self.lm_head is None:
            return CausalLMOutputWithPast(logits=hidden_states)
        selected = hidden_states[:, -logits_to_keep:] if logits_to_keep else hidden_states
        with torch.autocast(device_type=selected.device.type, enabled=False):
            logits = self.lm_head(selected.float())
        loss = None
        if labels is not None:
            if labels.shape != hidden_states.shape[:2]:
                raise ValueError("Flint labels must match [batch, sequence]")
            targets = labels[:, 1:].clone()
            if attention_mask is not None:
                valid_pairs = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
                targets.masked_fill_(~valid_pairs, -100)
            documents = attn_kwargs.get("document_ids")
            if documents is not None:
                targets.masked_fill_(documents[:, :-1] != documents[:, 1:], -100)
            numerator = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size), targets.reshape(-1), ignore_index=-100, reduction="sum"
            )
            loss = numerator / (targets != -100).sum().clamp_min(1)
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states.float() if final_hidden_only else outputs.hidden_states,
        )


ModelClass = FlintForCausalLM

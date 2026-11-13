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

"""Native FP32/BF16 Inkling2 text reference with explicit accumulation precision."""

from dataclasses import dataclass, replace
from typing import Literal, cast

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    shard_sequence_for_cp_contiguous,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import (
    HFCheckpointingMixin,
)
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str

from .attention import AttentionSources, Inkling2Attention
from .config import Inkling2Config, Inkling2TextConfig, resolve_quantization_config
from .context_parallel import CPBatch, prepare_cp_inputs, unpack_row_layout, validate_document_positions
from .layers import Inkling2LMHead, Inkling2MoE, ResidualQuery, ResidualState, ShortConvolution
from .parallelization import FP32_MODULES, Inkling2Parallelizer
from .state_dict_adapter import Inkling2StateDictAdapter


class Inkling2DecoderLayer(nn.Module):
    """One attention/MoE layer operating on explicit immutable depth state."""

    def __init__(
        self,
        config: Inkling2TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.is_local = layer_idx in config.local_layer_ids
        self.is_boundary = layer_idx % config.n_layers_per_attnres_block == 0
        self.attn_query = (
            None
            if layer_idx == 0
            else ResidualQuery(
                config.hidden_size,
                1 if self.is_local else 3,
                normalize_values=config.attn_res_normalize_values,
            )
        )
        self.mlp_query = ResidualQuery(config.hidden_size, 1, normalize_values=config.attn_res_normalize_values)
        count = 1 if self.is_local else 3
        self.attn_norms = nn.ModuleList(
            [
                initialize_rms_norm_module(
                    backend.rms_norm,
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                    dtype=dtype,
                )
                for _ in range(count)
            ]
        )
        self.mlp_norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.attn = Inkling2Attention(
            config, layer_idx, dtype=dtype, backend=cast(Literal["eager", "flex"], backend.attn)
        )
        self.mlp = Inkling2MoE(config, backend, dtype=dtype)
        self.attn_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, dtype=dtype)
        self.mlp_sconv = ShortConvolution(config.hidden_size, config.sconv_kernel_size, dtype=dtype)

    @property
    def self_attn(self) -> Inkling2Attention:
        """Expose the shared MoE CP setup hook without another parameter owner."""
        return self.attn

    def forward(self, state: ResidualState, valid: torch.Tensor, positions: torch.Tensor) -> ResidualState:
        """Advance a depth block, with a new block beginning after boundary attention.

        Args:
            state: Completed blocks and partial, each [batch, sequence, hidden].
            valid: Boolean tensor of shape [batch, sequence].
            positions: Integer tensor of shape [batch, sequence].

        Returns:
            New state with tensors of shape [batch, sequence, hidden]; previous state is unchanged.
        """
        dtype = self.attn.qkvr.weight.dtype
        streams = (
            (state.partial,) * len(self.attn_norms) if self.attn_query is None else self.attn_query(state).unbind(0)
        )
        normed = tuple(norm(stream.float()).to(dtype) for norm, stream in zip(self.attn_norms, streams))
        sources = AttentionSources(normed[0], normed[0], normed[0]) if self.is_local else AttentionSources(*normed)
        attention = self.attn_sconv(self.attn(sources, valid, positions), valid, positions=positions)
        blocks = (*state.blocks, state.partial) if self.is_boundary else state.blocks
        # Mixture arithmetic sees the FP32 sum before the residual storage cast.
        partial = attention.float() if self.is_boundary else state.partial.float() + attention.float()
        mlp_input = self.mlp_norm(self.mlp_query(ResidualState(blocks, partial))[0]).to(dtype)
        mlp = self.mlp_sconv(self.mlp(mlp_input, padding_mask=~valid), valid, positions=positions)
        return ResidualState(blocks, (partial + mlp.float()).to(dtype))

    @torch.no_grad()
    def reset_parameters(self, init_std: float, buffer_device: torch.device) -> None:
        """Initialize all layer parameters and buffers, including the common experts."""
        # The reference backend contract guarantees Torch RMSNorm instances.
        for norm in self.attn_norms:
            cast(nn.RMSNorm, norm).reset_parameters()
        cast(nn.RMSNorm, self.mlp_norm).reset_parameters()
        if self.attn_query is not None:
            self.attn_query.weight.zero_()
        self.mlp_query.weight.zero_()
        self.attn.reset_parameters(init_std)
        self.mlp.init_weights(buffer_device, init_std)
        self.attn_sconv.reset_parameters(init_std)
        self.mlp_sconv.reset_parameters(init_std)


class Inkling2ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Text implementation using eager or FlexAttention and common Torch experts.

    This entry point consumes Inkling2TextConfig; the multimodal subclass
    inserts projected image features before the same decoder.
    """

    config_class = Inkling2TextConfig
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = FP32_MODULES
    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True
    _pp_return_hidden_states: bool = False
    parallelizer = Inkling2Parallelizer()
    _owns_cp_attention = True
    supports_thd = True
    _packed_cp_attn_backends = ("eager", "flex")

    @dataclass(frozen=True)
    class ModelCapabilities:
        """FSDP2 supports EP, PP and packed contiguous CP; TP remains unsupported."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    def __init__(
        self,
        config: Inkling2TextConfig | Inkling2Config,
        *,
        backend: BackendConfig | None = None,
        revision: str | None = None,
    ) -> None:
        """Build the text decoder from a text or a released image/text configuration.

        Args:
            config: Text configuration, or the released image/text configuration that the
                shared factory passes as is; its text configuration is used, the checkpoint
                path carries over so the loader finds the weights, and no vision tower is built.
            backend: Kernel selection; None selects the eager Torch reference.
            revision: Resolved Hub revision of the checkpoint, for the sidecar read.
        """
        super().__init__()
        reject_unsupported_tie_word_embeddings(type(self), config)
        if isinstance(config, Inkling2Config):
            resolve_quantization_config(config, revision=revision)
            text_config = config.text_config
            text_config.name_or_path = config.name_or_path
            text_config.architectures = [type(self).__name__]
            config = text_config
        backend = backend or BackendConfig(
            attn="eager",
            linear="torch",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
        )
        if (
            (
                backend.linear,
                backend.rms_norm,
            )
            != ("torch", "torch")
            or backend.attn not in ("eager", "flex")
            or (backend.experts, backend.dispatcher) not in (("torch", "torch"), ("torch_mm", "torch"))
        ):
            raise ValueError(
                "Inkling2 requires eager/flex attention, Torch linear/norm, and torch or torch_mm experts with the torch dispatcher"
            )
        if backend.fake_balanced_gate or backend.gate_precision not in (
            None,
            torch.float32,
        ):
            raise ValueError("Inkling2 requires the real FP32 sigmoid router")
        self.cp_mesh: DeviceMesh | None = None
        self.backend = replace(backend, gate_precision=torch.float32)
        self.config = config
        self.text_config = config
        self.name_or_path = config.name_or_path
        resolve_quantization_config(config, revision=revision)
        self.vocab_size = config.vocab_size
        dtype = dtype_from_str(config.dtype, torch.float32)
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("Inkling2 text compute requires float32 or bfloat16")
        self.embed_tokens: nn.Embedding | None = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.embed_norm: nn.Module | None = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.layers = nn.ModuleDict(
            {
                str(i): Inkling2DecoderLayer(config, i, self.backend, dtype=dtype)
                for i in range(config.num_hidden_layers)
            }
        )
        self.moe_config = cast(Inkling2DecoderLayer, self.layers["0"]).mlp.moe_config
        self.output_query: ResidualQuery | None = ResidualQuery(
            config.hidden_size, 1, normalize_values=config.attn_res_normalize_values
        )
        self.norm: nn.Module | None = initialize_rms_norm_module(
            "torch", config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.lm_head: Inkling2LMHead | None = Inkling2LMHead(
            config.hidden_size, config.vocab_size, bias=False, dtype=dtype
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Inkling2StateDictAdapter(config)
        Inkling2ForCausalLM.initialize_weights(self)

    def prepare_model_inputs_for_cp(self, batch: CPBatch, *, num_chunks: int = 1) -> dict[str, ContextParallelSharder]:
        """Prepare packed or unpacked CP labels/layout while retaining full inputs for image splicing."""
        return prepare_cp_inputs(batch, num_chunks=num_chunks)

    @classmethod
    def from_config(
        cls, config: Inkling2TextConfig | Inkling2Config, *, backend: BackendConfig | None = None
    ) -> "Inkling2ForCausalLM":
        """Construct the text model, from a text or a released image/text configuration.

        The constructor accepts both forms; this classmethod keeps the explicit entry point.
        """
        return cls(config, backend=backend)

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize a fresh or materialized model without a checkpoint."""
        device = next(self.parameters()).device if buffer_device is None else buffer_device
        if self.embed_tokens is not None:
            nn.init.normal_(self.embed_tokens.weight, std=self.text_config.initializer_range)
            cast(nn.RMSNorm, self.embed_norm).reset_parameters()
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, std=self.text_config.initializer_range)
            cast(nn.RMSNorm, self.norm).reset_parameters()
            assert self.output_query is not None
            self.output_query.weight.zero_()
        for layer in self.layers.values():
            cast(Inkling2DecoderLayer, layer).reset_parameters(self.text_config.initializer_range, device)

    def get_input_embeddings(self) -> nn.Embedding | None:
        """Return the untied token lookup; embed_norm is applied in forward."""
        return self.embed_tokens

    def get_output_embeddings(self) -> Inkling2LMHead | None:
        """Return the callable untied head for shared FSDP and output-loss setup."""
        return self.lm_head

    def _pipeline_layer_ids(self) -> list[int]:
        """Validate contiguous global layer IDs and exclusive parameter ownership."""
        ids = [int(index) for index in self.layers]
        if not ids or ids != list(range(ids[0], ids[-1] + 1)):
            raise ValueError("Inkling2 PP stages require nonempty contiguous decoder layers")
        first, last = ids[0] == 0, ids[-1] == self.text_config.num_hidden_layers - 1
        if any(first != (module is not None) for module in (self.embed_tokens, self.embed_norm)):
            raise ValueError("Inkling2 PP first stage must exclusively own embeddings and embed_norm")
        if any(last != (module is not None) for module in (self.output_query, self.norm, self.lm_head)):
            raise ValueError("Inkling2 PP last stage must exclusively own output_query, norm and lm_head")
        return ids

    def customize_pipeline_stage_modules(
        self, module_names_per_stage: list[list[str]], *, layers_prefix: str, text_model: nn.Module | None = None
    ) -> list[list[str]]:
        """Assign embedding normalization and the final residual query to one owner.

        Args:
            module_names_per_stage: Generated stage FQNs; caller lists are unchanged.
            layers_prefix: Empty for this native model's top-level layers.
            text_model: Backbone resolved by the shared splitter.

        Returns:
            Copied FQN lists with embed_norm first and output_query last.
        """
        del text_model
        if layers_prefix or not module_names_per_stage:
            raise ValueError("Inkling2 PP expects nonempty stages with top-level layers")
        stages = [
            [name for name in stage if name not in ("embed_norm", "output_query")] for stage in module_names_per_stage
        ]
        stages[0].append("embed_norm")
        stages[-1].append("output_query")
        return stages

    def _pipeline_boundary_metas(
        self, boundary: int, *, batch: int, sequence: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Describe carried residual history and token layout at a decoder cut.

        Returns:
            Compute-dtype history [batch, sequence, sources, hidden] with completed
            blocks followed by the partial block; int64 layout [batch, sequence, 2]
            containing positions and validity. A boundary layer seals the preceding
            block after its attention, so sources = ceil(boundary / block_size) + 1.
        """
        block_size = self.text_config.n_layers_per_attnres_block
        sources = (boundary + block_size - 1) // block_size + 1
        return (
            torch.empty(batch, sequence, sources, self.text_config.hidden_size, device="meta", dtype=dtype),
            torch.empty(batch, sequence, 2, device="meta", dtype=torch.int64),
        )

    def get_pipeline_stage_metas(
        self, *, is_first: bool, microbatch_size: int, seq_len: int, dtype: torch.dtype
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Supply static shapes/dtypes to Automodel's shared pipeline runtime.

        Args:
            is_first: Whether the runtime identifies this as logical stage zero.
            microbatch_size: Number of unpacked sequences per microbatch.
            seq_len: Global input sequence length, before additional CP padding.
            dtype: Body compute dtype, independent of FP32 master storage.

        Returns:
            First-stage inputs are global int64 [batch, sequence]. Other token
            axes use padded_sequence / CP under context parallelism.
            Intermediate boundaries follow _pipeline_boundary_metas. Final output
            is FP32 [batch, sequence, vocab], or compute-dtype [batch, sequence, hidden]
            for the shared linear-projection loss when _pp_return_hidden_states is enabled.
        """
        ids = self._pipeline_layer_ids()
        if is_first != (ids[0] == 0):
            raise ValueError("Inkling2 PP metadata disagrees with embedding ownership")
        global_length = seq_len
        if self.cp_mesh is not None:
            size = self.cp_mesh.size()
            seq_len = (seq_len + (-seq_len) % (2 * size)) // size
        inputs = (
            (torch.empty(microbatch_size, global_length, device="meta", dtype=torch.int64),)
            if is_first
            else self._pipeline_boundary_metas(ids[0], batch=microbatch_size, sequence=seq_len, dtype=dtype)
        )
        if self.lm_head is None:
            outputs = self._pipeline_boundary_metas(ids[-1] + 1, batch=microbatch_size, sequence=seq_len, dtype=dtype)
        else:
            width = self.text_config.hidden_size if self._pp_return_hidden_states else self.vocab_size
            outputs = (
                torch.empty(
                    microbatch_size,
                    seq_len,
                    width,
                    device="meta",
                    dtype=dtype if self._pp_return_hidden_states else torch.float32,
                ),
            )
        return inputs, outputs

    def _receive_pipeline_state(
        self, history: torch.Tensor, tensors: tuple[torch.Tensor, ...]
    ) -> tuple[ResidualState, torch.Tensor, torch.Tensor]:
        """Recover differentiable residual views and per-microbatch token metadata.

        Args:
            history: Compute-dtype [batch, sequence, sources, hidden] residual history.
            tensors: One int64 [batch, sequence, 2] layout tensor; neither input is mutated.

        Returns:
            Residual state with views [batch, sequence, hidden], boolean validity
            [batch, sequence] and int64 positions [batch, sequence]. No activation is detached.
        """
        if history.ndim != 4 or history.dtype not in (torch.float32, torch.bfloat16) or len(tensors) != 1:
            raise ValueError("Inkling2 PP requires complete residual history and token layout")
        metas = self._pipeline_boundary_metas(
            self._pipeline_layer_ids()[0], batch=history.shape[0], sequence=history.shape[1], dtype=history.dtype
        )
        for value, meta in zip((history, *tensors), metas, strict=True):
            if value.shape != meta.shape or value.dtype != meta.dtype or value.device != history.device:
                raise ValueError("Inkling2 PP residual history/layout shape, dtype or device mismatch")
        sources = history.unbind(2)
        positions, valid = tensors[0].unbind(-1)
        return ResidualState(sources[:-1], sources[-1]), valid.bool(), positions

    def forward(
        self,
        input_ids: torch.Tensor,
        *pipeline_tensors: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor | None] | None = None,
        seq_lens: torch.Tensor | None = None,
        seq_lens_padded: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...] | CausalLMOutputWithPast:
        """Return unshifted text logits; the shared loss path owns the target shift.

        Args:
            input_ids: Integer tensor of shape [batch, global_sequence]. Inputs are
                packed or unpacked rows; the first stage slices them under CP after image splicing.
                On non-first pipeline stages, this is floating residual history
                [batch, sequence, sources, hidden] from _pipeline_boundary_metas.
            pipeline_tensors: Non-first-stage int64 [batch, sequence, 2] token layout.
                It carries authoritative positions/validity; repeated mask/position kwargs
                from the shared schedule are only consumed by the first stage.
            inputs_embeds: Optional decoder-ready [batch, sequence, hidden] embeddings,
                with text embedding normalization already applied. Supported on the
                first stage only; token IDs still supply layout.
            attention_mask: Optional tensor of shape [batch, sequence], one/true for real tokens.
                Document boundaries come from position resets, independently of padding.
            padding_mask: Optional boolean tensor of shape [batch, local_sequence], true
                for padding, as supplied by the shared trainer. Combined with attention_mask
                on the first stage; later stages use the carried validity tensor.
            position_ids: Optional integer tensor of shape [batch, local_sequence], increasing by one
                within each document, restarting at zero for a new document.
            causal_mask_mapping: Shared text-pipeline placeholders keyed by attention type.
                Only None entries are accepted; model-owned masks use validity and positions.
                Supplied dense masks with arbitrary leading dimensions are rejected.
            seq_lens: Optional shared packer real lengths [batch, documents].
            seq_lens_padded: Corresponding padded lengths covering each physical row.
            logits_to_keep: Zero for all positions, a positive tail length, or integer
                indices of shape [kept_sequence]. Used by the shared linear-loss path.
            output_hidden_states: Include final hidden states for external loss; targets
                are already aligned by the caller and are never shifted here.

        Returns:
            FP32 logits of shape [batch, kept_sequence, vocab], softcapped before
            unused vocabulary rows are set to negative infinity. When output_hidden_states
            is true, return CausalLMOutputWithPast with these logits and full final
            hidden_states of shape [batch, sequence, hidden]. No input is mutated.
            Non-final pipeline stages return the residual history/layout tuple;
            final stages return logits or hidden states for the external pipeline loss.
        """
        if causal_mask_mapping is not None and any(mask is not None for mask in causal_mask_mapping.values()):
            raise ValueError("Inkling2 constructs its own masks; causal_mask_mapping entries must be None")
        is_stage = self.embed_tokens is None or self.lm_head is None
        if is_stage:
            self._pipeline_layer_ids()
            if output_hidden_states or not isinstance(logits_to_keep, int) or logits_to_keep != 0:
                raise ValueError("Inkling2 PP requires full outputs and the external pipeline loss")
        if self.embed_tokens is None:
            if inputs_embeds is not None:
                raise ValueError("Only the first Inkling2 stage accepts inputs_embeds")
            state, valid, positions = self._receive_pipeline_state(input_ids, pipeline_tensors)
            with torch.autocast(device_type=input_ids.device.type, enabled=False):
                return self._run_decoder(state, valid, positions)
        if pipeline_tensors:
            raise ValueError("Inkling2 first stage cannot receive pipeline residual history")
        if seq_lens is not None or seq_lens_padded is not None:
            packed: CPBatch = {
                "input_ids": input_ids,
                "seq_lens": seq_lens,
                "seq_lens_padded": seq_lens_padded,
                "position_ids": position_ids,
            }
            unpack_row_layout(packed)
            position_ids = cast(torch.Tensor, packed["position_ids"])
            packed_padding = cast(torch.Tensor, packed["padding_mask"])
            padding_mask = packed_padding if padding_mask is None else padding_mask | packed_padding
        if self.cp_mesh is not None:
            if position_ids is None or padding_mask is None:
                raise ValueError("Inkling2 CP requires layout from ContextParallelSharder")
            input_ids, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, input_ids)
            if inputs_embeds is not None:
                inputs_embeds, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, inputs_embeds)
            if attention_mask is not None:
                attention_mask, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, attention_mask)
        if input_ids.ndim != 2 or min(input_ids.shape) == 0:
            raise ValueError("input_ids must have nonempty [batch, sequence] shape")
        valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
        if valid.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids [batch, sequence]")
        if padding_mask is not None:
            if padding_mask.shape != input_ids.shape or padding_mask.dtype != torch.bool:
                raise ValueError("padding_mask must be boolean [batch, sequence]")
            valid = valid & ~padding_mask
        positions = (valid.long().cumsum(-1) - 1).clamp_min(0) if position_ids is None else position_ids
        if positions.shape != input_ids.shape or positions.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("position_ids must be integer [batch, sequence]")
        if torch.any((positions < 0) & valid) or torch.any(
            (positions >= self.text_config.max_position_embeddings) & valid
        ):
            raise ValueError("position_ids exceed the configured context range")
        validate_document_positions(valid, positions)
        # Storage selects projection precision. An outer autocast must not silently
        # downcast FP32 routing, residual mixing, attention reductions or output logits.
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            embedded = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
            if embedded.shape != (*input_ids.shape, self.text_config.hidden_size):
                raise ValueError("inputs_embeds must match [batch, sequence, hidden]")
            if embedded.device != input_ids.device or embedded.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("inputs_embeds must use the input device and FP32/BF16 compute dtype")
            dtype = embedded.dtype
            assert self.embed_norm is not None
            hidden = self.embed_norm(embedded.float()).to(dtype) if inputs_embeds is None else embedded
            hidden = hidden * valid.unsqueeze(-1)
            state = ResidualState((), hidden)
            result = self._run_decoder(state, valid, positions, return_hidden=True)
            if self.lm_head is None:
                return result
            hidden = cast(torch.Tensor, result)
            if isinstance(logits_to_keep, int):
                if logits_to_keep < 0:
                    raise ValueError("logits_to_keep must be nonnegative")
                selected = hidden if logits_to_keep == 0 else hidden[:, -logits_to_keep:]
            else:
                if logits_to_keep.ndim != 1 or logits_to_keep.dtype != torch.int64:
                    raise ValueError("logits_to_keep indices must be int64 [kept_sequence]")
                selected = hidden[:, logits_to_keep]
            # The common head helper casts logits to the head/input dtype; this
            # model's projection and output remain FP32 even with BF16 weights.
            logits = self._project_logits(selected)
            if output_hidden_states:
                # Automodel's loss contract consumes the final tensor here; HF
                # annotates a tuple of per-layer states. Remove this boundary
                # exception when that shared output annotation accepts the tensor.
                return CausalLMOutputWithPast(logits=cast(torch.FloatTensor, logits), hidden_states=hidden)  # ty: ignore[invalid-argument-type]
            return logits

    def _run_decoder(
        self, state: ResidualState, valid: torch.Tensor, positions: torch.Tensor, *, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run local layers; transport immutable depth state or apply final-stage modules.

        Args:
            state: Completed blocks and partial [batch, sequence, hidden].
            valid: Boolean token validity [batch, sequence].
            positions: Integer token positions [batch, sequence].
            return_hidden: Skip final projection for ordinary selected-logit handling.

        Returns:
            Non-final stage history/layout as documented by _pipeline_boundary_metas;
            otherwise hidden states [batch, sequence, hidden] or FP32 logits
            [batch, sequence, vocab]. Packed history owns fresh storage and retains autograd.
        """
        for layer in self.layers.values():
            state = layer(state, valid, positions)
        if self.lm_head is None:
            return torch.stack((*state.blocks, state.partial), dim=2), torch.stack(
                (positions.long(), valid.long()), dim=-1
            )
        assert self.norm is not None and self.output_query is not None
        hidden = self.norm(self.output_query(state)[0]).to(state.partial.dtype)
        return hidden if return_hidden or self._pp_return_hidden_states else self._project_logits(hidden)

    def _project_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Apply the callable head, logit softcap and unused-vocabulary mask.

        Args:
            hidden: Compute-dtype tensor of shape [batch, sequence, hidden].

        Returns:
            FP32 logits [batch, sequence, vocab]; unused vocabulary rows are
            negative infinity. The input is unchanged.
        """
        assert self.lm_head is not None
        logits = self.lm_head(hidden)
        cap = self.text_config.final_logit_softcapping
        logits = cap * torch.tanh(logits / cap)
        return logits.masked_fill(
            torch.arange(self.vocab_size, device=logits.device) >= self.text_config.unpadded_vocab_size, float("-inf")
        )

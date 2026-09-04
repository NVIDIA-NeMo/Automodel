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

"""MiniMax M3 (mixed sparse/dense MoE) text backbone.

Stage 1 implements ``MiniMaxM3TextModel`` and the standalone
``MiniMaxM3SparseForCausalLM`` so the language path can be parity-tested against
the sglang reference before the vision tower / VLM wrapper (Stage 3) embeds the
text model as ``language_model``.
"""

from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import Block, MiniMaxM3RMSNorm
from nemo_automodel.components.models.minimax_m3_vl.msa import _msa_cp_enabled, _reject_unsupported_msa_runtime
from nemo_automodel.components.models.minimax_m3_vl.msa_plan import _MSAPackedLayout, _resolve_canonical_document_map
from nemo_automodel.components.models.minimax_m3_vl.mtp import MiniMaxM3MTP
from nemo_automodel.components.models.minimax_m3_vl.state_dict_adapter import (
    MiniMaxM3StateDictAdapter,
    MiniMaxM3VLStateDictAdapter,
)
from nemo_automodel.components.models.minimax_m3_vl.vision_encoder import MiniMaxM3VisionModel
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass
class MiniMaxM3CausalLMOutput:
    """Forward output carrying the primary logits and optional per-depth MTP logits."""

    logits: torch.Tensor
    mtp_per_depth_logits: list[torch.Tensor] | None = None


def build_moe_config(config: Any, dtype: torch.dtype) -> MoEConfig:
    """Build the routed-expert ``MoEConfig`` for the M3 backbone.

    Shared experts are handled in :class:`~...layers.Block` (SwiGLU-OAI), so
    ``n_shared_experts`` is 0 here. Routed experts use the ``swigluoai``
    activation ``gate * sigmoid(alpha * gate) * (up + 1)`` over the concatenated
    grouped gate/up projection produced by ``MoESplitExpertsStateDictMixin``.
    """
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.intermediate_size,
        n_routed_experts=config.num_local_experts,
        n_shared_experts=0,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="sigmoid" if str(getattr(config, "scoring_func", "sigmoid")).lower() != "softmax" else "softmax",
        route_scale=float(getattr(config, "routed_scaling_factor", 1.0)),
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swigluoai",
        activation_alpha=float(getattr(config, "swiglu_alpha", 1.702)),
        activation_limit=float(getattr(config, "swiglu_limit", 7.0)),
        softmax_before_topk=False,
        force_e_score_correction_bias=bool(getattr(config, "use_routing_bias", True)),
        # Released MiniMax-M3 checkpoints store the router gate weight in fp32
        # (same 1e-3-quantized correction-bias lattice as MiniMax-M2.7); allocate
        # it fp32 so every construction path keeps the gate's FSDP dtype group
        # uniform with its fp32 bias buffer (AMINT-286 pattern).
        gate_dtype=torch.float32,
        dtype=dtype,
    )


class MiniMaxM3TextModel(nn.Module):
    """Embedding + decoder stack + final norm for the M3 text backbone."""

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
    ):
        super().__init__()
        self.backend = backend
        # MiniMax M3 routes experts in fp32 (sglang hardcodes the router gate to
        # fp32: projection + sigmoid + correction-bias add). Forcing it here avoids
        # bf16 top-k drift -> different expert selection -> different logits. Set
        # before the decoder/MTP blocks (which build the MoE gates) are constructed.
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32
        self._attn_impl = backend.attn
        self._rope_fusion = backend.rope_fusion
        self.config = config
        self.config.num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", None))

        dtype = get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16)
        self.moe_config = moe_config or build_moe_config(config, dtype)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, self.backend)
        self._msa_layer_ids = frozenset(layer_id for layer_id, block in self.layers.items() if block.self_attn._use_msa)
        self._msa_model_has_dense_layers = len(self._msa_layer_ids) != len(self.layers)
        if self._msa_layer_ids and int(getattr(config, "num_mtp_modules", 0) or 0) > 0:
            raise NotImplementedError("MiniMax M3 MSA sparse attention supports MTP0 only; set num_mtp_modules=0")

        gemma = getattr(config, "use_gemma_norm", False)
        self.norm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)

        if not hasattr(config, "rope_parameters") or config.rope_parameters is None:
            rotary_dim = getattr(config, "rotary_dim", self.head_dim)
            config.rope_parameters = {
                "rope_theta": getattr(config, "rope_theta", 5000000.0),
                "rope_type": "default",
                "partial_rotary_factor": rotary_dim / self.head_dim,
            }

        base, rope_scaling, partial_rotary_factor = get_rope_config(config)
        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"),
        )

        # Multi-token prediction (DeepSeek-V3 style); shares the main lm_head.
        num_mtp = int(getattr(config, "num_mtp_modules", 0) or 0)
        self.mtp = MiniMaxM3MTP(config, self.moe_config, self.backend, num_mtp) if num_mtp > 0 else None

    def make_freqs_cis(self, position_ids: torch.Tensor, **attn_kwargs: Any) -> torch.Tensor:
        return position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self._rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run the text backbone.

        When the model has MSA layers, the packed document layout is built once
        here and handed to those layers as the model-owned ``_msa_layout``.

        Args:
            input_ids: Token ids shaped ``[batch, sequence]``. A later pipeline
                stage may instead receive floating hidden states shaped
                ``[batch, sequence, hidden]`` through this slot.
            position_ids: Optional positions shaped ``[batch, sequence]``.
            attention_mask: Optional keep/document mask shaped ``[batch,
                sequence]`` or standard bool block-causal mask shaped
                ``[batch, 1, sequence, sequence]``.
            padding_mask: Optional bool padding mask shaped ``[batch,
                sequence]``, where ``True`` marks padding.
            inputs_embeds: Optional input embeddings shaped ``[batch,
                sequence, hidden]`` instead of ``input_ids``.
            **attn_kwargs: Attention runtime metadata. ``_packed_seq_ids`` may
                carry integer document ids shaped ``[batch, sequence]``; callers
                must not provide the model-owned ``_msa_layout``.

        Returns:
            Final or pipeline-stage hidden states shaped ``[batch, sequence,
            hidden]``.

        Raises:
            TypeError: If a caller supplies the model-owned ``_msa_layout``.
            ValueError: If packed dense layers lack a standard 4-D block-causal
                mask or document metadata violates the MSA layout contract.
            NotImplementedError: If MSA is combined with an unsupported runtime
                mode or dense attention backend.
        """
        if "_msa_layout" in attn_kwargs:
            raise TypeError("_msa_layout is model-owned and cannot be supplied by callers.")

        use_msa = bool(self._msa_layer_ids)
        if use_msa:
            _reject_unsupported_msa_runtime(attn_kwargs, cp_enabled=_msa_cp_enabled(self))

        # Pipeline stages after the first receive the previous stage's hidden
        # states in the input_ids slot (a float tensor) with embed_tokens=None.
        if inputs_embeds is None and input_ids is not None and torch.is_floating_point(input_ids):
            inputs_embeds = input_ids
            input_ids = None
        h = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(0, h.shape[1], device=h.device).unsqueeze(0).expand(h.shape[0], -1)

        msa_layout: _MSAPackedLayout | None = None
        if use_msa:
            packed_seq_ids = attn_kwargs.pop("_packed_seq_ids", None)
            doc_ids = _resolve_canonical_document_map(
                h,
                packed_seq_ids=packed_seq_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
            )
            block_causal_mask = attention_mask if attention_mask is not None and attention_mask.dim() == 4 else None
            stage_uses_msa = any(layer_id in self.layers for layer_id in self._msa_layer_ids)
            if stage_uses_msa:
                msa_layout = _MSAPackedLayout.build(doc_ids)
                has_padding = msa_layout.has_padding
                has_multiple_documents = msa_layout.has_multiple_documents_per_row
            else:
                has_padding, has_multiple_documents = _MSAPackedLayout.validate(doc_ids)

            if self._msa_model_has_dense_layers and has_multiple_documents:
                if block_causal_mask is None:
                    raise ValueError(
                        "Packed MiniMax M3 MSA input crosses dense attention layers and therefore requires a "
                        "standard bool attention_mask with shape [batch,1,sequence,sequence]. A 2-D mask or "
                        "_packed_seq_ids alone cannot preserve document isolation in dense layers."
                    )
                if self._attn_impl != "sdpa":
                    raise NotImplementedError(
                        "Packed MiniMax M3 MSA dense layers first support backend.attn='sdpa' with the explicit "
                        f"4-D block-causal mask; got backend.attn={self._attn_impl!r}."
                    )

            # Canonical ids own padding semantics too, so a lower-priority mask
            # cannot reach dense attention or the MoE router after support is built.
            padding_mask = doc_ids == 0
            if block_causal_mask is None:
                attention_mask = doc_ids != 0 if has_padding else None

        freqs_cis = self.make_freqs_cis(position_ids, **attn_kwargs)

        for layer_id, layer in self.layers.items():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                _msa_layout=msa_layout if layer_id in self._msa_layer_ids else None,
                # CP-aware sparse attention derives per-document boundaries from
                # these; the eager path pops them.
                position_ids=position_ids,
                **attn_kwargs,
            )

        # norm is None on non-final pipeline stages.
        return self.norm(h) if self.norm is not None else h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        # embed_tokens / norm / layers can be None under meta-device + PP/sharded
        # init (the framework calls init_weights tolerating absent modules), so
        # guard each (matching minimax_m2 / step3p7).
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)
        if self.mtp is not None:
            self.mtp.init_weights(buffer_device)

    def mtp_logits(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        lm_head: nn.Module,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> list[torch.Tensor]:
        """Per-depth MTP logits from the final hidden states (shares ``lm_head``)."""
        if position_ids is None:
            position_ids = (
                torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
                .unsqueeze(0)
                .expand(hidden_states.shape[0], -1)
            )
        freqs_cis = self.make_freqs_cis(position_ids, **attn_kwargs)
        return self.mtp(
            hidden_states,
            input_ids=input_ids,
            embed_fn=self.embed_tokens,
            lm_head=lm_head,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )


class MiniMaxM3SparseForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Standalone M3 text backbone for causal LM (Stage 1 parity target)."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    _keep_in_fp32_modules_strict = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
    _pp_keep_self_forward: bool = True

    # The state-dict adapter loads every tensor from the checkpoint, so skip HF
    # random init on load (also avoids DTensor-collective hangs under sharding/PP).
    _skip_init_weights_on_load = True

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiniMaxM3VLTextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = copy(backend) if backend is not None else BackendConfig()
        self.model = MiniMaxM3TextModel(config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM3StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(config, "torch_dtype", "bfloat16"), torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor | MiniMaxM3CausalLMOutput:
        """Run standalone MiniMax M3 causal language modeling.

        The MSA runtime rules are enforced here, before the generic THD squeeze,
        because MSA accepts only cache-free BSHD prefill.

        Args:
            input_ids: Token ids shaped ``[batch, sequence]``.
            position_ids: Optional positions shaped ``[batch, sequence]``.
            attention_mask: Optional keep/document mask shaped ``[batch,
                sequence]`` or bool attention relation shaped ``[batch, 1,
                sequence, sequence]``.
            padding_mask: Optional bool padding mask shaped ``[batch,
                sequence]``, where ``True`` marks padding.
            **attn_kwargs: Attention runtime metadata, including optional
                ``_packed_seq_ids`` shaped ``[batch, sequence]`` and
                ``qkv_format``.

        Returns:
            Logits shaped ``[batch, sequence, vocab]`` for BSHD, logits shaped
            ``[1, tokens, vocab]`` for THD, or ``MiniMaxM3CausalLMOutput`` with
            the primary logits and per-depth MTP logits during MTP training.

        Raises:
            NotImplementedError: If MSA is combined with an unsupported runtime
                mode such as THD, cache use, context parallelism, or non-causal
                attention.
        """
        use_msa = bool(self.model._msa_layer_ids)
        if use_msa:
            _reject_unsupported_msa_runtime(attn_kwargs, cp_enabled=_msa_cp_enabled(self))
        if attn_kwargs.get("qkv_format") == "thd":
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        logits = self.lm_head(hidden) if self.lm_head else hidden
        if attn_kwargs.get("qkv_format") == "thd":
            return logits.unsqueeze(0)
        if self.model.mtp is not None and self.training and input_ids is not None:
            mtp_logits = self.model.mtp_logits(
                hidden,
                input_ids,
                self.lm_head,
                position_ids=position_ids,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **attn_kwargs,
            )
            return MiniMaxM3CausalLMOutput(logits=logits, mtp_per_depth_logits=mtp_logits)
        return logits

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight, mean=0.0, std=final_out_std, a=-3 * final_out_std, b=3 * final_out_std
                )
        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


class MiniMaxM3SparseForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """MiniMax M3 VL model with vision splicing and an M3 text backbone."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # Keep text and vision rotary buffers in FP32.
    _keep_in_fp32_modules = ["rotary_emb", "inv_freq"]
    _keep_in_fp32_modules_strict = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
    # Preserve vision splicing by retaining this forward under PP.
    _pp_keep_self_forward: bool = True
    mtp_outputs_are_logits = True
    # Use SDPA for context-parallel dense attention.
    _supports_cp_sdpa = True
    # Let M3 attention own packed document masking.
    _owns_packed_attention = True
    # Let M3 attention own packed CP sharding.
    _owns_cp_attention = True
    # Skip random initialization because checkpoint loading populates every tensor.
    _skip_init_weights_on_load = True
    # Store the CP submesh installed by the MoE parallelizer.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declare the supported parallelism modes."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls, config: Any, moe_config: MoEConfig | None = None, backend: BackendConfig | None = None, **kwargs
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = MiniMaxM3VLConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        text_config = config.text_config
        self.backend = copy(backend) if backend is not None else BackendConfig()
        self.model = MiniMaxM3TextModel(text_config, backend=self.backend, moe_config=moe_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear, text_config.hidden_size, text_config.vocab_size, bias=False
        )
        self.vision_tower = MiniMaxM3VisionModel(
            config.vision_config,
            text_config.hidden_size,
            config.projector_hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias,
            patch_merge_bias=getattr(config, "patch_merge_bias", config.multimodal_projector_bias),
        )
        self.image_token_index = config.image_token_index
        self.video_token_index = config.video_token_index
        self.vocab_size = text_config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MiniMaxM3VLStateDictAdapter(
                config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(text_config, "torch_dtype", "bfloat16"), torch.bfloat16),
            )

    @property
    def language_model(self):
        return self.model

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Rewrite generated pipeline FQNs to M3's module paths."""
        if getattr(self.model, "mtp", None) is not None:
            raise NotImplementedError(
                "MiniMax M3 VL does not support MTP modules under pipeline parallelism yet; "
                "set text_config.num_mtp_modules=0 for pp_size>1 runs."
            )
        from nemo_automodel.components.distributed.pipelining.hf_utils import MULTIMODAL_SUFFIXES

        text_prefix = "model."  # M3's text stack lives directly under self.model
        fixed: list[list[str]] = []
        for stage in module_names_per_stage:
            names: list[str] = []
            for name in stage:
                if layers_prefix != text_prefix and name.startswith(layers_prefix):
                    names.append(text_prefix + name[len(layers_prefix) :])
                elif name.startswith(text_prefix) and name[len(text_prefix) :] in MULTIMODAL_SUFFIXES:
                    names.append(name[len(text_prefix) :])
                else:
                    names.append(name)
            fixed.append(names)
        return fixed

    def _is_pipeline_parallel_stage(self) -> bool:
        """True when this is a partial pipeline stage (some text modules nulled)."""
        if self.lm_head is None:
            return True
        if getattr(self.model, "embed_tokens", None) is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.text_config.num_hidden_layers)
        except (TypeError, AttributeError):
            return False

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Build PP meta tensors with full stage-zero ids and CP-local activations."""
        text_config = self.config.text_config
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size

        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_seq_len = seq_len
        if cp_size > 1:
            padded_seq_len = seq_len + (-seq_len) % (2 * cp_size)
            local_seq_len = padded_seq_len // cp_size

        def meta(*shape: int) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dtype)

        # Keep token ids int64 and inter-stage activations in the requested dtype.
        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (meta(microbatch_size, local_seq_len, hidden_size),)

        if self.lm_head is not None:
            # Match the output meta dtype to the language-model head.
            head_dtype = getattr(getattr(self.lm_head, "weight", None), "dtype", dtype)
            outputs_meta = (torch.empty(microbatch_size, local_seq_len, vocab_size, device="meta", dtype=head_dtype),)
        else:
            outputs_meta = (meta(microbatch_size, local_seq_len, hidden_size),)
        return inputs_meta, outputs_meta

    @staticmethod
    def _to_grid_list(grid_thw) -> list[list[int]]:
        if isinstance(grid_thw, torch.Tensor):
            return grid_thw.detach().cpu().to(torch.int64).tolist()
        return [list(map(int, g)) for g in grid_thw]

    def _splice_multimodal(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        grid_thw,
        token_index: int,
    ) -> torch.Tensor:
        # Suspend CP dispatch while the vision tower runs bidirectional attention.
        from nemo_automodel.components.distributed.context_parallel.utils import (
            cp_dispatcher_suspended,  # noqa: PLC0415
        )

        with cp_dispatcher_suspended(self.cp_mesh):
            features = self.vision_tower(pixel_values, self._to_grid_list(grid_thw))
        mask = input_ids == token_index
        expected = int(mask.sum().item())
        if features.shape[0] != expected:
            raise ValueError(
                f"MiniMax M3 VL: got {features.shape[0]} vision tokens for {expected} placeholder positions "
                f"(token_index={token_index})."
            )
        inputs_embeds[mask] = features.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        return inputs_embeds

    def _embed_and_splice(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw=None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw=None,
    ) -> torch.Tensor:
        """Embed token ids and splice vision or video features."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None or pixel_values_videos is not None:
            inputs_embeds = inputs_embeds.clone()
        if pixel_values is not None:
            inputs_embeds = self._splice_multimodal(
                inputs_embeds, input_ids, pixel_values, image_grid_thw, self.image_token_index
            )
        if pixel_values_videos is not None:
            inputs_embeds = self._splice_multimodal(
                inputs_embeds, input_ids, pixel_values_videos, video_grid_thw, self.video_token_index
            )
        return inputs_embeds

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return a CP sharder while deferring embedding and vision splicing to forward."""
        del batch, num_chunks
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw=None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw=None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | MiniMaxM3CausalLMOutput | dict[str, torch.Tensor]:
        is_pp_stage = self._is_pipeline_parallel_stage()

        # Reject MTP on every pipeline stage using the unsplit config.
        if is_pp_stage and int(getattr(self.config.text_config, "num_mtp_modules", 0) or 0) > 0:
            raise NotImplementedError(
                "MiniMax M3 VL does not support MTP modules under pipeline parallelism yet; "
                "set text_config.num_mtp_modules=0 for pp_size>1 runs."
            )

        # Restore this PP microbatch's staged media before embedding.
        chunks = getattr(self, "_vlm_pixel_values_chunks", None)
        if (
            pixel_values is None
            and pixel_values_videos is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
            and (chunks is not None or getattr(self, "_vlm_pixel_values_videos_chunks", None) is not None)
        ):
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
            consumed = False
            if chunks is not None and (input_ids == self.image_token_index).any() and chunk_idx < len(chunks):
                pixel_values = chunks[chunk_idx]
                image_grid_thw = self._vlm_image_grid_hws_chunks[chunk_idx]
                consumed = True
            video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
            if (
                video_chunks is not None
                and (input_ids == self.video_token_index).any()
                and chunk_idx < len(video_chunks)
            ):
                pixel_values_videos = video_chunks[chunk_idx]
                video_grid_thw = self._vlm_video_grid_thw_chunks[chunk_idx]
                consumed = True
            if consumed:
                self._vlm_chunk_idx = chunk_idx + 1

        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1

        # Route floating PP inputs directly to the text model as hidden states.
        if inputs_embeds is None and input_ids is not None and torch.is_floating_point(input_ids):
            inputs_embeds = input_ids
            input_ids = None

        if inputs_embeds is None:
            inputs_embeds = self._embed_and_splice(
                input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
            # Shard freshly embedded full-sequence activations for CP.
            if cp_size > 1:
                inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)

        hidden = self.model(
            None,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        # Return hidden states directly for fused loss without materializing logits.
        if logits_to_keep is not None and not is_pp_stage:
            if self.model.mtp is not None and self.training and input_ids is not None:
                raise NotImplementedError(
                    "logits_to_keep (fused-loss path) is not supported together with MTP "
                    "modules, which need full logits; set text_config.num_mtp_modules=0."
                )
            return {"hidden_states": hidden}

        # Use hidden states as output on non-final pipeline stages.
        logits = self.lm_head(hidden) if self.lm_head is not None else hidden

        # Return plain tensors between pipeline stages.
        if is_pp_stage:
            return logits

        if self.model.mtp is not None and self.training and input_ids is not None:
            mtp_logits = self.model.mtp_logits(
                hidden, input_ids, self.lm_head, position_ids=position_ids, attention_mask=attention_mask, **kwargs
            )
            return MiniMaxM3CausalLMOutput(logits=logits, mtp_per_depth_logits=mtp_logits)
        return logits

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            if self.lm_head is not None:
                final_out_std = self.config.text_config.hidden_size**-0.5
                nn.init.trunc_normal_(
                    self.lm_head.weight, mean=0.0, std=final_out_std, a=-3 * final_out_std, b=3 * final_out_std
                )
        cast_model_to_dtype(self, dtype)
        with buffer_device:
            self.model.rotary_emb.device = buffer_device


ModelClass = MiniMaxM3SparseForConditionalGeneration

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

"""DeepSeek V4.1 text backbone for AutoModel training.

Forward contract (reference ``Transformer.forward`` of ``inference/model.py``):

1. Embed tokens and expand the hidden state into ``hc_mult`` residual streams.
2. Run the 40 blocks with single-pass mHC: each block receives the input mix
   produced by the previous block's FFN site (a one-hot mix reads stream 0 at
   the start).  Engram modules write into the streams before their layer.
3. Collapse the streams with the last FFN-site mix, apply the final RMSNorm and
   the fp32 ``lm_head``.

Cross-layer CSA2 state (shared compressed KV, index keys, Top-K indices and the
hierarchical candidate pool) lives in per-layer snapshots of
:class:`~nemo_automodel.components.models.deepseek_v41.layers.DeepseekV41SharedState`.
Snapshots share tensors and preserve the state needed for activation recomputation.

Scope: text-only training of the released backbone.  The vision tower, the
DSpark draft layers (``mtp.*``) and inference-time KV caching / SWA bounded
replay are out of scope; their checkpoint tensors are dropped by the
state-dict adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.deepseek_v4.cp import build_packed_seq_ids
from nemo_automodel.components.models.deepseek_v4.model import _normalize_thd_packing_metadata
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41EngramHasher, EngramLayout
from nemo_automodel.components.models.deepseek_v41.layers import (
    DeepseekV41Block,
    DeepseekV41RotaryEmbedding,
    DeepseekV41SharedState,
    build_window_topk_indices,
    hc_collapse,
    make_identity_pre_mix,
)
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepSeekV41StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def document_relative_positions(seq_ids: torch.Tensor) -> torch.Tensor:
    """Positions that restart at ``0`` on every document boundary of ``seq_ids`` (``[B, S]``)."""
    seq_len = seq_ids.shape[1]
    idx = torch.arange(seq_len, device=seq_ids.device).unsqueeze(0).expand_as(seq_ids)
    is_start = torch.ones_like(seq_ids, dtype=torch.bool)
    is_start[:, 1:] = seq_ids[:, 1:] != seq_ids[:, :-1]
    starts = torch.where(is_start, idx, torch.zeros_like(idx)).cummax(dim=1).values
    return idx - starts


class DeepseekV41Model(nn.Module):
    """DeepSeek V4.1 decoder stack: embeddings, hyper-connected blocks, final norm."""

    def __init__(
        self,
        config: DeepseekV41Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
        engram_process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.config = config
        config.validate_layer_layout()

        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.moe_intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            # noaux_tc routing without group limits.
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=1e-3,
            score_func="sqrtsoftplus",
            router_weights_fp32=True,
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0,
            norm_topk_prob=config.norm_topk_prob,
            dtype=model_dtype,
            # Routed and shared experts use clamped SwiGLU in fp32 (reference ``Expert.forward``).
            swiglu_limit=float(config.swiglu_limit),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.engram_layout = EngramLayout.from_config(config)
        self.engram_hasher = (
            DeepseekV41EngramHasher(config, self.engram_layout) if self.engram_layout is not None else None
        )
        self.layers = nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = DeepseekV41Block(
                layer_id,
                config,
                self.moe_config,
                backend,
                engram_layout=self.engram_layout,
                engram_process_group=engram_process_group,
            )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        # Base rope (no YaRN) for pure sliding-window layers, compress rope (YaRN)
        # for every CSA2 layer and for the pooled latents (reference ``Attention.__init__``).
        partial_rotary_factor = float(config.qk_rope_head_dim) / float(config.head_dim)
        self.rotary_emb = DeepseekV41RotaryEmbedding(
            rope_theta=float(config.rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=None,
        )
        self.rotary_emb_compress = DeepseekV41RotaryEmbedding(
            rope_theta=float(config.compress_rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=getattr(config, "rope_scaling", None),
        )

    def set_engram_tokenizer(self, tokenizer) -> None:
        """Attach the tokenizer-derived compressed token map used by Engram hashing."""
        if self.engram_hasher is not None:
            self.engram_hasher.set_tokenizer(tokenizer)

    def _ensure_engram_token_map(self) -> None:
        if self.engram_hasher is None or self.engram_hasher.has_token_map:
            return
        path = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)
        if not path:
            raise RuntimeError(
                "Engram needs the tokenizer-derived token map: call model.set_engram_tokenizer(tokenizer) "
                "or construct the config with name_or_path pointing at the checkpoint."
            )
        from transformers import AutoTokenizer  # noqa: PLC0415

        self.set_engram_tokenizer(AutoTokenizer.from_pretrained(path))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run the backbone.

        Args:
            input_ids: ``[B, S]`` token ids, or ``[T]`` for packed THD batches.
            inputs_embeds: optional ``[B, S, hidden]`` embeddings replacing the lookup.
            position_ids: ``[B, S]`` document-relative positions; derived from the
                packing metadata / padding when omitted.
            attention_mask: ``[B, S]`` with ``1`` for valid tokens (HF convention).
            padding_mask: ``[B, S]`` bool with ``True`` at padding.

        Returns:
            Final hidden states ``[B, S, hidden]`` after the last norm.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("DeepseekV41Model requires input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if inputs_embeds.dim() == 2:
            # Packed THD inputs arrive with the batch axis collapsed.
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        batch, seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device

        if padding_mask is None and attention_mask is not None and attention_mask.dim() == 2:
            padding_mask = attention_mask.to(device).bool().logical_not()
        if padding_mask is not None and padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)

        _normalize_thd_packing_metadata(attn_kwargs)
        seq_ids = attn_kwargs.get("packed_seq_ids")
        if seq_ids is None and attn_kwargs.get("qkv_format") == "thd":
            packed_seq_lens = attn_kwargs.get("seq_lens_padded")
            if packed_seq_lens is None:
                packed_seq_lens = attn_kwargs.get("seq_lens")
            if packed_seq_lens is not None:
                seq_ids = build_packed_seq_ids(packed_seq_lens, seq_len=seq_len, device=device)
        if seq_ids is None:
            seq_ids = torch.ones(batch, seq_len, dtype=torch.long, device=device)
        else:
            seq_ids = seq_ids.to(device=device, dtype=torch.long)
            if seq_ids.dim() == 1:
                seq_ids = seq_ids.unsqueeze(0)
        if padding_mask is not None:
            seq_ids = seq_ids.masked_fill(padding_mask.to(device), 0)

        if position_ids is None:
            position_ids = document_relative_positions(seq_ids)
        else:
            position_ids = position_ids.to(device=device, dtype=torch.long)
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.shape[0] == 1 and batch > 1:
                position_ids = position_ids.expand(batch, -1)

        engram_hash_ids = None
        engram_mask = None
        if self.engram_hasher is not None:
            if input_ids is None:
                raise ValueError("Engram hashing needs input_ids; inputs_embeds alone is not enough")
            self._ensure_engram_token_map()
            engram_mask = seq_ids > 0
            engram_hash_ids = self.engram_hasher(input_ids.to(device), position_ids, engram_mask)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        position_embeddings_compress = self.rotary_emb_compress(inputs_embeds, position_ids)

        h = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre_mix = make_identity_pre_mix(h, self.config.hc_mult)
        state = DeepseekV41SharedState(window_topk_idxs=build_window_topk_indices(seq_ids, self.config.sliding_window))
        moe_padding_mask = padding_mask.to(device) if padding_mask is not None else None

        for layer in self.layers.values():
            layer_hash_ids = None
            if engram_hash_ids is not None and layer.engram is not None:
                layer_hash_ids = engram_hash_ids[:, :, layer.engram.layer_hash_index, :]
            h, pre_mix, state = layer(
                h,
                pre_mix,
                padding_mask=moe_padding_mask,
                engram_hash_ids=layer_hash_ids,
                engram_mask=engram_mask,
                position_embeddings=position_embeddings,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=self.rotary_emb_compress,
                position_ids=position_ids,
                seq_ids=seq_ids,
                state=state,
            )

        h = hc_collapse(h, pre_mix)
        return self.norm(h)

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.layers.values():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        init_std = float(self.config.initializer_range)
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight)
            self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device, init_std=init_std)


class DeepseekV41ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """DeepSeek V4.1 causal LM (text backbone + fp32 ``lm_head``).

    ``engram_process_group`` explicitly selects contiguous row owners for the
    Engram tables. By default, distributed models use WORLD; single-rank models
    retain local tables. FSDP's shard mesh must match the owner group exactly.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    # Reference-sensitive tensors that must stay fp32 regardless of the outer cast policy.
    _keep_in_fp32_modules_strict = [
        "attn_hc.fn",
        "attn_hc.base",
        "attn_hc.scale",
        "ffn_hc.fn",
        "ffn_hc.base",
        "ffn_hc.scale",
        "self_attn.sinks",
        # Compressors pool in fp32 with fp32 projections (the reference promotes the
        # ratio > 1 projections to fp32; ratio-1 projections cast their input to the
        # weight dtype, see ``DeepseekV41Compressor.forward``).
        "self_attn.compressor.wkv",
        "self_attn.compressor.wgate",
        "e_score_correction_bias",
        "lm_head",
        "rotary_emb",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = True

    @classmethod
    def from_config(
        cls,
        config: DeepseekV41Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
        **kwargs,
    ) -> DeepseekV41ForCausalLM:
        """Construct the model, forwarding the runtime Engram owner group."""
        return cls(config, moe_config, backend, engram_process_group=engram_process_group, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = DeepseekV41Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: DeepseekV41Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = DeepseekV41Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
            engram_process_group=engram_process_group,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=torch.float32
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepSeekV41StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_engram_tokenizer(self, tokenizer) -> None:
        """Attach the tokenizer used to derive the Engram compressed token map."""
        self.model.set_engram_tokenizer(tokenizer)

    def _nemo_prepare_model_owned_dtensors(self, fsdp_mesh: DeviceMesh) -> set[nn.Parameter]:
        """Register owner table DTensors before FSDP records ignored parameters.

        Args:
            fsdp_mesh: One-dimensional shard mesh whose ranks and ordering must
                match the Engram owner group.

        Returns:
            Exact registered parameter identities to exclude from FSDP. Each
            has global shape [padded_rows, head_dim] and placement Shard(0);
            local storage has shape [padded_rows / owner_world_size, head_dim].
        """
        parameters: set[nn.Parameter] = set()
        for layer in self.model.layers.values():
            if layer.engram is None or layer.engram.embed.process_group is None:
                continue
            parameters.add(layer.engram.embed.parallelize_weight(fsdp_mesh))
        return parameters

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run causal language modeling.

        Args:
            input_ids: ``[B, S]`` token ids (``[T]`` for packed THD).
            position_ids: ``[B, S]`` document-relative positions.
            attention_mask: ``[B, S]`` valid-token mask.
            padding_mask: ``[B, S]`` bool padding mask.
            logits_to_keep: Number or positions of logits to retain.
            output_hidden_states: Whether to expose the final hidden states.
        """
        if attn_kwargs.pop("_pre_embed_only", False):
            # Context parallelism is not supported; there is no model-owned CP batch prep.
            return {}
        if output_hidden_states is None:
            output_hidden_states = getattr(getattr(self, "config", None), "output_hidden_states", False)
        thd_mode = attn_kwargs.get("qkv_format") == "thd"

        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )
        return compute_lm_head_logits(
            self.lm_head,
            hidden_states,
            logits_to_keep,
            is_thd=thd_mode,
            fp32_lm_head=True,
            output_hidden_states=bool(output_hidden_states),
        )

    def update_moe_gate_bias(self) -> None:
        self.model.update_moe_gate_bias()

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            nn.init.trunc_normal_(
                self.lm_head.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
        # After FSDP2 wrapping, parameter dtypes must already be correct from
        # construction-time metadata; a blanket cast would downcast fp32 DTensors.
        if not _has_dtensor_params(self):
            cast_model_to_dtype(self, dtype)
        for layer in self.model.layers.values():
            if layer.engram is not None:
                layer.engram.embed.mark_sharding_contract()


ModelClass = DeepseekV41ForCausalLM

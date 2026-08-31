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

import copy
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    compute_lm_head_logits,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.mtp import (
    MTPConfig,
    MTPContextParallelInputs,
    prepare_mtp_context_parallel_inputs,
)
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.hy_v4.config import HyV4Config
from nemo_automodel.components.models.hy_v4.cp import shard_hy_v4_packed_cp_batch
from nemo_automodel.components.models.hy_v4.hc import HyV4HCHead, HyV4HCLayer
from nemo_automodel.components.models.hy_v4.layers import HyV4MLA
from nemo_automodel.components.models.hy_v4.optimized_kernels import prepare_cudnn_dsa_packed_metadata
from nemo_automodel.components.models.hy_v4.rope_utils import freqs_cis_from_position_ids, precompute_freqs_cis
from nemo_automodel.components.models.hy_v4.state_dict_adapter import HyV4StateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _uses_indexshare(config: HyV4Config) -> bool:
    """Return whether the model has layers that reuse another layer's DSA indices."""
    return "shared" in config.indexer_types


class _Fp32AccumulationMM(torch.autograd.Function):
    """CUDA BF16 matrix multiply with FP32 output and an explicit linear backward.

    ``torch.mm(..., out_dtype=torch.float32)`` reproduces vLLM's HY V4 logits
    without materializing an FP32 copy of the vocabulary projection, but PyTorch
    does not define its derivative.  The derivative is the ordinary pair of
    matrix multiplies, also accumulated in FP32 before casting to each input's
    gradient dtype.
    """

    @staticmethod
    def forward(ctx: Any, hidden: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        """Project BF16 hidden states into FP32 logits.

        Args:
            ctx: Autograd context used to retain the two matrix operands.
            hidden: Flattened hidden states shaped ``[tokens, hidden]``.
            weight_t: Transposed vocabulary weight shaped ``[hidden, vocab]``.

        Returns:
            New FP32 logits shaped ``[tokens, vocab]``.
        """
        ctx.save_for_backward(hidden, weight_t)
        return torch.mm(hidden, weight_t, out_dtype=torch.float32)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Differentiate the FP32-accumulating vocabulary projection.

        Args:
            ctx: Autograd context containing ``hidden`` and ``weight_t``.
            grad_output: Logit gradient shaped ``[tokens, vocab]``.

        Returns:
            Newly allocated gradients shaped ``[tokens, hidden]`` and
            ``[hidden, vocab]`` in their respective operand dtypes. An entry is
            ``None`` when that operand does not require gradients.
        """
        hidden, weight_t = ctx.saved_tensors
        grad_hidden = grad_weight_t = None
        if ctx.needs_input_grad[0]:
            grad_hidden = torch.mm(
                grad_output.to(weight_t.dtype).contiguous(),
                weight_t.t(),
                out_dtype=torch.float32,
            ).to(hidden.dtype)
        if ctx.needs_input_grad[1]:
            grad_weight_t = torch.mm(
                hidden.t(),
                grad_output.to(hidden.dtype).contiguous(),
                out_dtype=torch.float32,
            ).to(weight_t.dtype)
        return grad_hidden, grad_weight_t


class HyV4LMHead(nn.Linear):
    """HY V4 output projection with vLLM's FP32-accumulation contract.

    The checkpoint keeps the weight in the model dtype. On CUDA, ``torch.mm``
    accumulates directly into FP32 without materializing an FP32 copy of that
    weight; CPU/reference execution uses the equivalent explicit cast path.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project arbitrary token axes to FP32 vocabulary logits.

        Args:
            hidden_states: Activations shaped ``[..., hidden]``.

        Returns:
            Newly allocated FP32 logits shaped ``[..., vocab]``.
        """
        output_shape = (*hidden_states.shape[:-1], self.out_features)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        if flat.is_cuda and flat.dtype in (torch.float16, torch.bfloat16) and self.weight.dtype == flat.dtype:
            logits = _Fp32AccumulationMM.apply(flat, self.weight.t())
            if self.bias is not None:
                logits = logits + self.bias.float()
        else:
            logits = F.linear(
                flat.float(),
                self.weight.float(),
                self.bias.float() if self.bias is not None else None,
            )
        return logits.reshape(output_shape)


@dataclass
class HyV4CausalLMOutput(CausalLMOutputWithPast):
    """Causal-LM output extended with HY V4's auxiliary MTP states."""

    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


class Block(nn.Module):
    """One HY V4 decoder block with iHC-wrapped attention and MLP sites."""

    def __init__(self, layer_idx: int, config: HyV4Config, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        # IndexShare: per-layer indexer mode from `config.indexer_types`. A "shared" layer
        # owns no indexer and reuses the previous "full" layer's top-k selection. Absent the
        # field (e.g. HY V4-5.1, which runs a full indexer every layer), every layer is "full".
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.self_attn = HyV4MLA(config, backend, skip_topk=self.skip_topk)
        self.hc_attn_layer = HyV4HCLayer(config)
        self.hc_mlp_layer = HyV4HCLayer(config)

        # Thread dtype from config.torch_dtype so the block's own params stay
        # aligned with the rest of the model (fp32 under fp32 master weights).
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        is_moe_layer = config.mlp_layer_types[layer_idx] == "sparse"

        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
            # HY V4 clamps only routed experts; its dense MLP and shared expert
            # use ordinary SwiGLU.
            if self.mlp.shared_experts is not None:
                self.mlp.shared_experts.swiglu_limit = 0.0
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)

        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the block and return ``(hidden_states, topk_indices)``.

        Args:
            x: Packed iHC streams shaped ``[tokens, hc_mult, hidden]`` (or
                ``[tokens, hidden]`` for the no-iHC MTP copy).
            freqs_cis: Complex rotations ``[tokens, rope_head_dim / 2]``.
            attention_mask: Must be ``None`` for packed THD attention.
            padding_mask: Optional expert-dispatch mask ``[tokens]``.
            prev_topk_indices: Selection ``[tokens, 1, index_topk]`` from the
                previous full indexer for an IndexShare layer.
            **attn_kwargs: Packed cuDNN metadata including ``cu_seqlens``.

        Returns:
            Newly computed hidden streams and this layer's int32 top-k indices.
            A shared layer returns ``prev_topk_indices`` unchanged as the second
            item; a full layer returns a newly computed selection.
        """
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_input, post_gates, residual = self.hc_attn_layer(x)
        attn_out, topk_indices = self.self_attn(
            x=self.input_layernorm(attn_input),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            prev_topk_indices=prev_topk_indices,
            return_topk_indices=True,
            **attn_kwargs,
        )
        x = self.hc_attn_layer.post(attn_out, residual, post_gates)

        mlp_input, post_gates, residual = self.hc_mlp_layer(x)
        mlp_out = self._mlp(x=self.post_attention_layernorm(mlp_input), padding_mask=padding_mask)
        x = self.hc_mlp_layer.post(mlp_out, residual, post_gates)
        return x, topk_indices

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Run the layer's dense or routed feed-forward module.

        Args:
            x: Packed activations shaped ``[tokens, hidden]``.
            padding_mask: Optional boolean vector ``[tokens]`` where ``True``
                excludes a row from expert dispatch.

        Returns:
            Newly computed activations shaped ``[tokens, hidden]``.
        """
        if isinstance(self.mlp, MLP):
            return self.mlp(x)
        else:
            assert isinstance(self.mlp, MoE)
            return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.006):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std=init_std)
        self.mlp.init_weights(buffer_device, init_std=init_std)
        self.hc_attn_layer.init_weights(init_std)
        self.hc_mlp_layer.init_weights(init_std)


class HyV4MTPLayer(Block):
    """One no-iHC HY V4 decoder layer used by multi-token prediction."""

    def __init__(
        self,
        depth: int,
        config: HyV4Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ) -> None:
        mtp_config = copy.copy(config)
        mtp_config.enable_ihc = False
        mtp_config.num_hidden_layers = max(depth + 1, 1)
        mtp_config.mlp_layer_types = ["sparse"] * mtp_config.num_hidden_layers
        mtp_config.layer_types = ["deepseek_sparse_attention"] * mtp_config.num_hidden_layers
        mtp_config.indexer_types = ["full"] * mtp_config.num_hidden_layers
        super().__init__(depth, mtp_config, moe_config, backend)
        self.config = config

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.enorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.hnorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.eh_proj = initialize_linear_module(
            backend.linear,
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            dtype=dtype,
        )
        self.final_layernorm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )

    def forward(
        self,
        previous_hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Fuse the future embedding with the prior depth and predict its state.

        Args:
            previous_hidden_states: Prior-depth states ``[tokens, hidden]``.
            embed_input: Shifted future-token embeddings ``[tokens, hidden]``.
            freqs_cis: Complex rotations ``[tokens, rope_head_dim / 2]``.
            attention_mask: Must be ``None`` for packed THD execution.
            padding_mask: Optional expert-dispatch mask ``[tokens]``.
            **attn_kwargs: Packed cuDNN metadata including ``cu_seqlens``.

        Returns:
            Newly computed MTP states shaped ``[tokens, hidden]``.
        """
        hidden_states = self.eh_proj(torch.cat((self.enorm(embed_input), self.hnorm(previous_hidden_states)), dim=-1))
        hidden_states, _ = super().forward(
            hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            prev_topk_indices=None,
            **attn_kwargs,
        )
        return self.final_layernorm(hidden_states)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.006) -> None:
        super().init_weights(buffer_device, init_std=init_std)
        for norm in (self.enorm, self.hnorm, self.final_layernorm):
            norm.reset_parameters()
        with buffer_device:
            nn.init.trunc_normal_(self.eh_proj.weight, mean=0.0, std=init_std)


class HyV4Model(nn.Module):
    """HY V4 decoder backbone with independent residual streams."""

    def __init__(
        self,
        config: HyV4Config,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        # Resolve model dtype once; thread it explicitly to every sub-module
        # so fp32 master weights work even when construction is not wrapped in
        # local_torch_dtype().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=1e-3,
            score_func="sigmoid",
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            swiglu_limit=config.swiglu_limit,
            softmax_before_topk=False,
            router_weights_fp32=True,
            force_e_score_correction_bias=True,
            dtype=model_dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.hc_head = HyV4HCHead(config) if config.enable_ihc else None
        self.mtp_layers = nn.ModuleList(
            [
                HyV4MTPLayer(depth, config, self.moe_config, backend)
                for depth in range(int(config.num_nextn_predict_layers or 0))
            ]
        )

        self.max_seq_len = config.max_position_embeddings
        self.qk_rope_head_dim = config.qk_rope_head_dim

        self.freqs = precompute_freqs_cis(
            qk_rope_head_dim=self.qk_rope_head_dim,
            rope_theta=float(config.rope_parameters["rope_theta"]),
        )

    def prepare_packed_dsa_kwargs(
        self,
        token_states: torch.Tensor,
        padding_mask: torch.Tensor | None,
        attn_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize and cache cuDNN DSA metadata for a packed layer stack.

        Args:
            token_states: Local packed states shaped ``[tokens, ...]``.
            padding_mask: Optional local padding mask shaped ``[tokens]``.
            attn_kwargs: Packed metadata containing global cumulative sequence
                offsets and optional local CP query indices.

        Returns:
            A shallow copy containing contiguous device-local metadata. Existing
            tensors may alias inputs when they already have the requested
            device, dtype, and contiguous layout.
        """
        if self.backend.attn != "cudnn" or attn_kwargs.get("qkv_format") != "thd":
            return attn_kwargs

        cu_seqlens = attn_kwargs.get("cu_seqlens")
        if cu_seqlens is None:
            raise ValueError("cuDNN DSA requires 'cu_seqlens' for packed THD input.")
        cu_seqlens = cu_seqlens.flatten().to(device=token_states.device, dtype=torch.int32).contiguous()
        query_indices = attn_kwargs.get("hy_v4_cp_query_indices")
        if query_indices is not None:
            query_indices = query_indices.flatten().to(device=token_states.device, dtype=torch.int32).contiguous()
        cu_seqlens_padded = attn_kwargs.get("cu_seqlens_padded")
        if cu_seqlens_padded is not None:
            cu_seqlens_padded = (
                cu_seqlens_padded.flatten().to(device=token_states.device, dtype=torch.int32).contiguous()
            )
        cudnn_padding_mask = None
        if padding_mask is not None:
            cudnn_padding_mask = padding_mask.flatten().to(device=token_states.device, dtype=torch.bool).contiguous()
        cp_size = int(attn_kwargs.get("cp_size", 1))
        packed_metadata = prepare_cudnn_dsa_packed_metadata(
            cu_seqlens,
            token_states.shape[0] * cp_size,
            query_indices=query_indices,
            cu_seqlens_padded=cu_seqlens_padded,
            padding_mask=cudnn_padding_mask,
        )
        prepared = dict(attn_kwargs)
        prepared["cu_seqlens"] = cu_seqlens
        if query_indices is not None:
            prepared["hy_v4_cp_query_indices"] = query_indices
        if cu_seqlens_padded is not None:
            prepared["cu_seqlens_padded"] = cu_seqlens_padded
        prepared["_cudnn_dsa_packed_metadata"] = packed_metadata
        prepared["_cudnn_dsa_topk_length"] = packed_metadata.causal_lengths.clamp_max(
            int(self.config.index_topk)
        ).contiguous()
        prepared["_cudnn_dsa_all_rows_nonempty"] = packed_metadata.all_rows_nonempty
        prepared["_cudnn_dsa_valid_row_indices"] = packed_metadata.valid_row_indices
        return prepared

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run one vLLM-compatible packed HY V4 decoder stage.

        Args:
            input_ids: Packed token IDs with shape ``[tokens]`` when this stage
                owns ``embed_tokens``. Later pipeline stages receive packed iHC
                streams with shape ``[tokens, hc_mult, hidden]`` instead.
            position_ids: Packed integer positions with shape ``[tokens]``.
            attention_mask: Must be ``None``. Packed boundaries use ``cu_seqlens``.
            padding_mask: Optional packed boolean mask with shape ``[tokens]``;
                ``True`` marks padding for expert dispatch.
            prev_topk_indices: Optional previous-stage IndexShare selection with
                shape ``[tokens, 1, index_topk]`` and integer dtype.
            **attn_kwargs: Attention layout metadata. Packed THD uses ``cu_seqlens``
                and optional ``cu_seqlens_padded`` of shape ``[sequences + 1]``, plus
                optional ``hy_v4_cp_query_indices`` of shape ``[tokens]`` containing
                global padded-storage query coordinates.

        Returns:
            Newly computed packed hidden states and the latest IndexShare top-k
            selection. Non-final pipeline stages return hidden states with shape
            ``[tokens, hc_mult, hidden]``; the final stage returns
            ``[tokens, hidden]`` after ``hc_head`` and ``norm``.
        """
        if self.backend.attn != "cudnn" or attn_kwargs.get("qkv_format") != "thd":
            raise NotImplementedError(
                "HY V4 follows the vLLM sparse-attention forward and supports only packed THD "
                "execution with backend.attn='cudnn'."
            )
        if attention_mask is not None:
            raise ValueError("Packed HY V4 attention masks must be represented by cu_seqlens.")
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[0], device=input_ids.device)

        freqs_cis = freqs_cis_from_position_ids(
            position_ids,
            self.freqs.to(position_ids.device),
        )

        if self.embed_tokens is not None:
            if input_ids.dim() != 1 or input_ids.dtype not in (torch.int32, torch.int64, torch.long):
                raise ValueError(
                    "The first HY V4 pipeline stage requires packed integer input IDs with shape [tokens]."
                )
            h = self.embed_tokens(input_ids)
            if self.config.enable_ihc:
                h = h.unsqueeze(-2).expand(*h.shape[:-1], self.config.hc_mult, self.config.hidden_size)
        else:
            expected_rank = 3 if self.config.enable_ihc else 2
            if input_ids.dim() != expected_rank or input_ids.shape[-1] != self.config.hidden_size:
                raise ValueError(
                    "A non-first HY V4 pipeline stage requires upstream hidden states with shape "
                    f"[tokens, {'hc_mult, ' if self.config.enable_ihc else ''}hidden]."
                )
            h = input_ids

        attn_kwargs = self.prepare_packed_dsa_kwargs(h, padding_mask, attn_kwargs)

        # IndexShare: thread the most recent "full" layer's top-k selection forward so the
        # following "shared" layers can reuse it. Legacy HY V4 configs have no shared layers, so
        # avoid retaining and propagating their per-layer selections.
        uses_indexshare = _uses_indexshare(self.config)
        topk_indices = prev_topk_indices if uses_indexshare else None
        for layer in self.layers.values():
            h, layer_topk_indices = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                prev_topk_indices=topk_indices if uses_indexshare else None,
                **attn_kwargs,
            )
            if uses_indexshare:
                topk_indices = layer_topk_indices

        h = self.hc_head(h) if self.hc_head is not None else h
        h = self.norm(h) if self.norm else h
        return h, topk_indices

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        init_std = float(self.config.initializer_range)

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            if self.hc_head is not None:
                self.hc_head.init_weights(init_std)

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device, init_std=init_std)
        for layer in self.mtp_layers:
            layer.init_weights(buffer_device=buffer_device, init_std=init_std)

    def update_moe_gate_bias(self) -> None:
        """Update the noaux router correction bias of each local MoE layer; dense layers and disabled gates are skipped."""
        with torch.no_grad():
            for block in self.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()
            for block in self.mtp_layers:
                if block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()


class HyV4ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """AutoModel-owned HY V4 causal language model implementation."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _pp_keep_self_forward: bool = True
    _pp_return_hidden_states_supported: bool = True
    _pp_fused_linear_ce_mtp_supported: bool = True
    _pp_mtp_targets_in_output: bool = True
    _owns_cp_attention = True
    _packed_cp_attn_backends = ("cudnn",)
    _keep_in_fp32_modules_strict = [
        "hc_pre.hc_fn",
        "hc_pre.hc_base",
        "hc_pre.hc_scale",
        "hc_head.hc_head_fn",
        "hc_head.hc_head_base",
        "hc_head.hc_head_scale",
        "learnable_sink_param",
        "e_score_correction_bias",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True
        supports_mtp_cp: bool = True
        supports_mtp_cp_pp: bool = True

    @classmethod
    def from_config(
        cls,
        config: HyV4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = HyV4Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: HyV4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        resolved_backend = backend or BackendConfig()
        if resolved_backend.rope_fusion:
            raise NotImplementedError(
                "HY4-preview parity is validated only with vLLM's default interleaved RoPE; "
                "set backend.rope_fusion=false."
            )
        # vLLM computes the HY V4 router projection and selected mixture weights in fp32.
        if resolved_backend.gate_precision is None:
            resolved_backend = replace(resolved_backend, gate_precision=torch.float32)
        self.backend = resolved_backend
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = HyV4Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.mtp_config = MTPConfig(
            num_layers=int(config.num_nextn_predict_layers or 0),
            layer_pattern="*" if int(config.num_nextn_predict_layers or 0) > 0 else "",
            loss_scaling_factor=float(config.mtp_loss_factor),
        )
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = HyV4LMHead(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=model_dtype,
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = HyV4StateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=model_dtype
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def update_moe_gate_bias(self) -> None:
        """Delegate the noaux router correction-bias update to the inner model."""
        self.model.update_moe_gate_bias()

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep HY V4's post-backbone and MTP modules on the final PP stage.

        Args:
            module_names_per_stage: Per-stage fully qualified module names from
                AutoPipeline's default decoder-layer split.
            layers_prefix: Fully qualified prefix of the HY V4 decoder backbone.
            text_model: Decoder module selected by AutoPipeline; unused because
                HY V4 owns a fixed top-level layout.

        Returns:
            A copied stage-module mapping with ``hc_head`` and ``mtp_layers``
            attached to the final stage when those modules are enabled.
        """
        del text_model
        stage_modules = [list(modules) for modules in module_names_per_stage]
        if not stage_modules:
            return stage_modules

        final_stage = stage_modules[-1]
        if self.model.hc_head is not None:
            hc_head_fqn = f"{layers_prefix}hc_head"
            if hc_head_fqn not in final_stage:
                final_stage.append(hc_head_fqn)
        if len(self.model.mtp_layers) > 0:
            mtp_layers_fqn = f"{layers_prefix}mtp_layers"
            if mtp_layers_fqn not in final_stage:
                final_stage.append(mtp_layers_fqn)
        return stage_modules

    def _is_pipeline_parallel_stage(self) -> bool:
        """Return whether this module is a trimmed pipeline stage."""
        if self.lm_head is None or self.model.embed_tokens is None:
            return True
        return len(self.model.layers) != int(self.config.num_hidden_layers)

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Declare packed-THD pipeline tensors for iHC, IndexShare, and MTP.

        Args:
            is_first: Whether this is the first physical pipeline stage.
            microbatch_size: Number of packed records in one pipeline microbatch;
                HY V4 requires this to be one for packed THD execution.
            seq_len: Number of packed tokens in each microbatch.
            dtype: Backbone activation dtype.

        Returns:
            Input and output metadata tuples. The first stage consumes token IDs
            with shape ``[microbatch, tokens]``. Stage boundaries carry iHC states
            ``[tokens, hc_mult, hidden]``, float32 IndexShare coordinates
            ``[tokens, 1, index_topk]``, and one future-token embedding tensor
            ``[tokens, hidden]`` per MTP depth. The final stage emits either
            float32 logits ``[microbatch, tokens, vocab]`` or, when fused linear
            CE is enabled, model-dtype hidden states
            ``[microbatch, tokens, hidden]``. MTP states, an int32
            packed-document map, and one authoritative int64 MTP target tensor
            per depth follow the primary output.
        """
        if microbatch_size != 1:
            raise ValueError(
                "HY V4 pipeline parallelism requires pp_microbatch_size=1 for packed THD execution; "
                f"got microbatch_size={microbatch_size}."
            )

        hidden_size = int(self.config.hidden_size)
        mtp_depth = int(self.mtp_config.num_layers)
        uses_indexshare = _uses_indexshare(self.config)

        def meta(shape: tuple[int, ...], tensor_dtype: torch.dtype = dtype) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=tensor_dtype)

        token_ids_meta = meta((microbatch_size, seq_len), torch.long)
        ihc_hidden_meta = meta((seq_len, self.config.hc_mult, hidden_size))
        topk_meta = meta((seq_len, 1, int(self.config.index_topk)), torch.float32)
        mtp_embed_metas = tuple(meta((seq_len, hidden_size)) for _ in range(mtp_depth))

        if is_first:
            inputs_meta = (token_ids_meta,)
        else:
            inputs_meta = (ihc_hidden_meta, *((topk_meta,) if uses_indexshare else ()), *mtp_embed_metas)

        if self.lm_head is None:
            outputs_meta = (ihc_hidden_meta, *((topk_meta,) if uses_indexshare else ()), *mtp_embed_metas)
        else:
            if getattr(self, "_pp_return_hidden_states", False) is True:
                primary_output_meta = meta((microbatch_size, seq_len, hidden_size))
            else:
                primary_output_meta = meta((microbatch_size, seq_len, self.config.vocab_size), torch.float32)
            if mtp_depth > 0:
                mtp_hidden_metas = tuple(meta((microbatch_size, seq_len, hidden_size)) for _ in range(mtp_depth))
                seq_idx_meta = meta((microbatch_size, seq_len), torch.int32)
                mtp_target_metas = tuple(meta((microbatch_size, seq_len), torch.long) for _ in range(mtp_depth))
                outputs_meta = (primary_output_meta, *mtp_hidden_metas, seq_idx_meta, *mtp_target_metas)
            else:
                outputs_meta = (primary_output_meta,)
        return inputs_meta, outputs_meta

    def prepare_mtp_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        ignore_index: int = -100,
    ) -> MTPContextParallelInputs | None:
        """Prepare globally shifted, packed-boundary-safe MTP inputs for CP.

        Args:
            batch: Global packed mapping whose token fields use
                ``[batch, sequence]``.
            ignore_index: Label sentinel excluded from MTP loss.

        Returns:
            Per-depth global token/position inputs, or ``None`` when MTP is
            disabled. Returned tensors are newly prepared from ``batch``.
        """
        if not self.mtp_config.enabled:
            return None
        return prepare_mtp_context_parallel_inputs(
            batch,
            num_depths=self.mtp_config.num_layers,
            ignore_index=ignore_index,
        )

    def should_pack_validation_with_training(self) -> bool:
        """Return whether validation must use the optimized packed THD layout."""
        return self.backend.attn == "cudnn"

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Attach HY V4 DSA's packed THD context-parallel batch sharder.

        Args:
            batch: Global packed mapping whose token fields conventionally use
                ``[batch, sequence]``.
            num_chunks: Number of chunks for load-balanced CP sharding.

        Returns:
            A mapping containing the model-owned ``ContextParallelSharder``;
            no tensor in ``batch`` is mutated.
        """
        attn_backend = self.backend.attn
        if attn_backend not in self._packed_cp_attn_backends:
            raise NotImplementedError(
                "HY V4 DSA packed context parallelism requires backend.attn='cudnn'; "
                f"got backend.attn={attn_backend!r}."
            )

        cp_sharder = ContextParallelSharder(
            shard_batch=partial(
                shard_hy_v4_packed_cp_batch,
                num_chunks=int(num_chunks),
            ),
            # Contiguous over the packed THD token axis: rank r keeps
            # tokens [r * T/cp, (r + 1) * T/cp).
            local_token_global_indices=contiguous_local_indices,
        )
        return {"cp_sharder": cp_sharder}

    @staticmethod
    def _shift_packed_thd(
        tensor: torch.Tensor,
        *,
        depth: int,
        attn_kwargs: dict[str, Any],
        fill_value: int = 0,
    ) -> torch.Tensor:
        """Shift a THD tensor without crossing packed-document boundaries.

        Args:
            tensor: Packed values shaped ``[..., tokens]``.
            depth: Positive future-token offset.
            attn_kwargs: Mapping containing cumulative packed boundaries.
            fill_value: Scalar written where a shift crosses a document end.

        Returns:
            Newly allocated shifted values with the same shape and dtype as
            ``tensor``.
        """
        boundaries = attn_kwargs.get("cu_seqlens_padded")
        if boundaries is None:
            boundaries = attn_kwargs.get("cu_seqlens")
        if boundaries is None:
            raise ValueError("Packed HY V4 MTP shifting requires cu_seqlens or cu_seqlens_padded.")
        boundaries = boundaries.reshape(-1).to(device=tensor.device, dtype=torch.long)
        boundaries = boundaries[boundaries >= 0]
        token_count = tensor.shape[-1]
        positions = torch.arange(token_count, device=tensor.device)
        seq_idx = torch.searchsorted(boundaries[1:].contiguous(), positions, right=True)
        shifted = torch.roll(tensor, shifts=-depth, dims=-1)
        shifted_seq_idx = torch.roll(seq_idx, shifts=-depth, dims=0)
        valid = (positions + depth < token_count) & (shifted_seq_idx == seq_idx)
        return torch.where(valid, shifted, torch.as_tensor(fill_value, dtype=tensor.dtype, device=tensor.device))

    def _build_mtp_embed_inputs_for_pp(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        attn_kwargs: dict[str, Any],
        mtp_per_depth_input_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_position_ids: tuple[torch.LongTensor, ...] | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Build packed-boundary-safe MTP embeddings on the first PP stage.

        Args:
            input_ids: Packed integer token IDs with shape ``[tokens]``.
            position_ids: Optional packed positions with shape ``[tokens]``.
            attn_kwargs: Packed THD metadata containing ``cu_seqlens`` or
                ``cu_seqlens_padded`` with shape ``[sequences + 1]``.
            mtp_per_depth_input_ids: Optional globally shifted then CP-sharded
                future-token IDs, one ``[1, tokens]`` tensor per depth.
            mtp_per_depth_position_ids: Matching globally shifted then
                CP-sharded future positions. These preserve a future token that
                resides on the next CP rank.

        Returns:
            One newly computed future-token embedding tensor with shape
            ``[tokens, hidden]`` per MTP depth.
        """
        if self.model.embed_tokens is None:
            raise ValueError("The first HY V4 pipeline stage must own embed_tokens to build MTP embeddings.")
        if input_ids.dim() != 1 or input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("The first HY V4 pipeline stage requires packed integer input IDs with shape [tokens].")
        if (mtp_per_depth_input_ids is None) != (mtp_per_depth_position_ids is None):
            raise ValueError("MTP per-depth input IDs and position IDs must be provided together.")
        if mtp_per_depth_input_ids is not None and (
            len(mtp_per_depth_input_ids) != self.mtp_config.num_layers
            or len(mtp_per_depth_position_ids) != self.mtp_config.num_layers
        ):
            raise ValueError(f"HY V4 expected {self.mtp_config.num_layers} precomputed MTP input depths.")

        source_positions = (
            position_ids
            if position_ids is not None
            else torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long)
        )
        embeddings: list[torch.Tensor] = []
        for depth in range(1, self.mtp_config.num_layers + 1):
            if mtp_per_depth_input_ids is not None:
                depth_ids = mtp_per_depth_input_ids[depth - 1].squeeze(0)
                depth_positions = mtp_per_depth_position_ids[depth - 1].squeeze(0)
                if depth_ids.shape != input_ids.shape or depth_positions.shape != input_ids.shape:
                    raise ValueError(
                        "HY V4 precomputed PP MTP inputs must match the local packed token shape; "
                        f"got ids={tuple(depth_ids.shape)}, positions={tuple(depth_positions.shape)}, "
                        f"tokens={tuple(input_ids.shape)}."
                    )
            else:
                depth_ids = self._shift_packed_thd(input_ids, depth=depth, attn_kwargs=attn_kwargs)
                depth_positions = self._shift_packed_thd(source_positions, depth=depth, attn_kwargs=attn_kwargs)
            embed_input = self.model.embed_tokens(depth_ids)
            embeddings.append(torch.where((depth_positions == 0).unsqueeze(-1), 0, embed_input))
        return tuple(embeddings)

    @staticmethod
    def _packed_seq_idx_for_pp(token_states: torch.Tensor, attn_kwargs: dict[str, Any]) -> torch.Tensor:
        """Build the per-microbatch document map consumed by pipeline MTP loss.

        Args:
            token_states: Final-stage logits ``[batch, tokens, vocab]`` or hidden
                states ``[batch, tokens, hidden]``.
            attn_kwargs: Packed metadata with optional ``seq_idx`` of shape
                ``[batch, tokens]`` or cumulative boundaries of shape
                ``[sequences + 1]``.

        Returns:
            A contiguous int32 tensor with shape ``[batch, tokens]``. The result
            is newly allocated unless an already-compatible ``seq_idx`` is supplied.
        """
        batch_size, token_count = token_states.shape[:2]
        seq_idx = attn_kwargs.get("seq_idx")
        if isinstance(seq_idx, torch.Tensor):
            if seq_idx.dim() == 1:
                seq_idx = seq_idx.unsqueeze(0)
            if tuple(seq_idx.shape) != (batch_size, token_count):
                raise ValueError(
                    "HY V4 pipeline seq_idx must match the primary output token axes; "
                    f"got seq_idx={tuple(seq_idx.shape)}, token_states={tuple(token_states.shape)}."
                )
            return seq_idx.to(device=token_states.device, dtype=torch.int32).contiguous()

        boundaries = attn_kwargs.get("cu_seqlens_padded")
        if boundaries is None:
            boundaries = attn_kwargs.get("cu_seqlens")
        if isinstance(boundaries, torch.Tensor):
            boundaries = boundaries.reshape(-1).to(device=token_states.device, dtype=torch.long)
            boundaries = boundaries[boundaries >= 0]
            positions = torch.arange(token_count, device=token_states.device)
            seq_idx_1d = torch.searchsorted(boundaries[1:].contiguous(), positions, right=True).to(torch.int32)
            return seq_idx_1d.unsqueeze(0).expand(batch_size, -1).contiguous()

        return torch.ones((batch_size, token_count), device=token_states.device, dtype=torch.int32)

    def _mtp_target_tail_for_pp(
        self,
        token_states: torch.Tensor,
        mtp_per_depth_targets: tuple[torch.LongTensor, ...] | None,
    ) -> tuple[torch.LongTensor, ...]:
        """Validate authoritative local MTP targets carried to the PP loss.

        The targets are shifted from the global packed batch before CP
        sharding. Returning them from the last stage binds each target shard to
        the exact pipeline microbatch whose MTP states produced the loss.

        Args:
            token_states: Main last-stage output with shape ``[batch, tokens, ...]``.
            mtp_per_depth_targets: One CP-local ``[batch, tokens]`` target per depth.

        Returns:
            A contiguous tuple suitable for the non-differentiable output tail.
        """
        num_depths = int(self.mtp_config.num_layers)
        if mtp_per_depth_targets is None:
            raise ValueError(
                "HY V4 pipeline MTP requires globally shifted per-depth targets; "
                "prepare them before context-parallel sharding."
            )
        if len(mtp_per_depth_targets) != num_depths:
            raise ValueError(f"HY V4 expected {num_depths} precomputed MTP target depths.")

        expected_shape = tuple(token_states.shape[:2])
        targets: list[torch.LongTensor] = []
        for depth, target in enumerate(mtp_per_depth_targets, start=1):
            if target.dtype != torch.long or tuple(target.shape) != expected_shape:
                raise ValueError(
                    "HY V4 precomputed PP MTP targets must be int64 and match the local output token axes; "
                    f"depth={depth}, target={tuple(target.shape)}/{target.dtype}, expected={expected_shape}/int64."
                )
            targets.append(target.contiguous())
        return tuple(targets)

    def _run_mtp(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor | None,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        mtp_embed_inputs: tuple[torch.Tensor, ...] | None,
        mtp_per_depth_input_ids: tuple[torch.LongTensor, ...] | None,
        mtp_per_depth_position_ids: tuple[torch.LongTensor, ...] | None,
        attn_kwargs: dict[str, Any],
    ) -> list[torch.Tensor]:
        """Run checkpoint-native HY4 MTP layers over final decoder states.

        Args:
            hidden_states: Final backbone states ``[tokens, hidden]``.
            input_ids: Optional packed token IDs ``[tokens]``. The non-pipeline
                path uses these to build future-token embeddings locally.
            position_ids: Optional packed positions ``[tokens]``.
            attention_mask: Must be ``None`` for packed THD attention.
            padding_mask: Optional expert-dispatch mask ``[tokens]``.
            mtp_embed_inputs: Optional future-token embeddings propagated from
                the first pipeline stage, one ``[tokens, hidden]`` tensor per depth.
            mtp_per_depth_input_ids: Optional pre-shifted local IDs, each
                ``[1, tokens]`` before squeezing. Propagated embeddings replace
                these IDs on non-first PP stages.
            mtp_per_depth_position_ids: Matching pre-shifted positions. These
                remain authoritative on the final PP stage even when embeddings
                were propagated separately.
            attn_kwargs: Packed cuDNN and CP metadata.

        Returns:
            One newly computed ``[tokens, hidden]`` state tensor per MTP depth.
        """
        num_depths = len(self.model.mtp_layers)
        if num_depths == 0:
            return []
        if mtp_embed_inputs is not None and mtp_per_depth_input_ids is not None:
            raise ValueError("MTP propagated embeddings cannot be combined with per-depth input IDs.")
        if mtp_embed_inputs is None and (mtp_per_depth_input_ids is None) != (mtp_per_depth_position_ids is None):
            raise ValueError("MTP per-depth input IDs and position IDs must be provided together.")
        if mtp_embed_inputs is not None and len(mtp_embed_inputs) != num_depths:
            raise ValueError(f"HY V4 expected {num_depths} propagated MTP embedding depths.")
        if mtp_per_depth_input_ids is not None and len(mtp_per_depth_input_ids) != num_depths:
            raise ValueError(f"HY V4 expected {num_depths} precomputed MTP input depths.")
        if mtp_per_depth_position_ids is not None and len(mtp_per_depth_position_ids) != num_depths:
            raise ValueError(f"HY V4 expected {num_depths} precomputed MTP position depths.")

        if attn_kwargs.get("qkv_format") != "thd":
            raise NotImplementedError("HY V4 MTP follows the vLLM packed THD forward.")
        prepared_kwargs = self.model.prepare_packed_dsa_kwargs(hidden_states, padding_mask, attn_kwargs)
        outputs: list[torch.Tensor] = []
        if position_ids is None:
            position_ids = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        for depth, layer in enumerate(self.model.mtp_layers, start=1):
            if mtp_per_depth_position_ids is not None:
                depth_positions = mtp_per_depth_position_ids[depth - 1].squeeze(0)
            else:
                depth_positions = self._shift_packed_thd(
                    position_ids,
                    depth=depth,
                    attn_kwargs=attn_kwargs,
                )

            if mtp_embed_inputs is not None:
                embed_input = mtp_embed_inputs[depth - 1]
            else:
                if input_ids is None or self.model.embed_tokens is None:
                    raise ValueError("HY V4 MTP requires token IDs or propagated embeddings on this stage.")
                if mtp_per_depth_input_ids is not None:
                    depth_ids = mtp_per_depth_input_ids[depth - 1].squeeze(0)
                else:
                    depth_ids = self._shift_packed_thd(
                        input_ids,
                        depth=depth,
                        attn_kwargs=attn_kwargs,
                    )
                embed_input = self.model.embed_tokens(depth_ids)
            # vLLM's HY V4 MTP reference masks the wrapped token at each
            # document boundary after shifting. Position zero is the boundary
            # sentinel for both ordinary and packed sequences.
            embed_input = torch.where((depth_positions == 0).unsqueeze(-1), 0, embed_input)
            freqs_cis = freqs_cis_from_position_ids(
                depth_positions,
                self.model.freqs.to(depth_positions.device),
            )
            hidden_states = layer(
                hidden_states,
                embed_input=embed_input,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                **prepared_kwargs,
            )
            outputs.append(hidden_states)
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *pipeline_carries: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        mtp_per_depth_input_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_position_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_targets: tuple[torch.LongTensor, ...] | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...] | torch.Tensor:
        """Run packed THD HY V4 in single-model or pipeline-stage mode.

        Args:
            input_ids: Packed token IDs with shape ``[1, tokens]`` on the first
                stage. Later pipeline stages receive upstream iHC states with
                shape ``[tokens, hc_mult, hidden]`` in this slot.
            *pipeline_carries: On non-first stages, an optional float32 IndexShare
                tensor ``[tokens, 1, index_topk]`` followed by one propagated MTP
                embedding ``[tokens, hidden]`` per prediction depth.
            position_ids: Packed position IDs with shape ``[1, tokens]``.
            attention_mask: Must be ``None``. Document boundaries use ``cu_seqlens``.
            padding_mask: Optional packed padding mask with shape ``[1, tokens]``.
            mtp_per_depth_input_ids: Optional globally shifted then CP-sharded
                future-token IDs, one tensor per MTP depth.
            mtp_per_depth_position_ids: Optional globally shifted then CP-sharded
                future positions, one tensor per MTP depth.
            mtp_per_depth_targets: Optional globally shifted then CP-sharded
                MTP loss targets. Pipeline execution requires these on the
                final stage and returns them as a loss-only output tail.
            logits_to_keep: If ``0``, project all positions; else only the last ``logits_to_keep``.
                A tensor value contains the one-dimensional token indices to project.
            output_hidden_states: When set (single-process), carry final hidden states on the output.
            **attn_kwargs: Additional attention metadata forwarded to the base model. Packed THD
                uses an int32 ``cu_seqlens`` tensor of shape ``[sequences + 1]`` and
                ``qkv_format="thd"``.

        Returns:
            Single-model execution returns a causal-LM output whose logits have
            shape ``[1, tokens, vocab]``. A non-final pipeline stage returns iHC
            hidden states, the float32 IndexShare carry, and propagated MTP
            embeddings. The final stage returns float32 logits, or hidden states
            when fused linear CE is configured, followed by MTP states, an
            int32 packed-document map, and authoritative int64 MTP targets used
            by the pipeline loss.
        """
        if input_ids is None:
            raise ValueError("HY V4 requires input_ids.")
        if self.backend.attn != "cudnn" or attn_kwargs.get("qkv_format") != "thd":
            raise NotImplementedError(
                "HY V4 follows the vLLM sparse-attention forward and supports only packed THD "
                "execution with backend.attn='cudnn'."
            )
        if attention_mask is not None:
            raise ValueError("Packed HY V4 attention masks must be represented by cu_seqlens.")
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        is_pp_stage = self._is_pipeline_parallel_stage()
        is_first_stage = self.model.embed_tokens is not None
        has_lm_head = self.lm_head is not None
        uses_indexshare = _uses_indexshare(self.config)
        mtp_depth = int(self.mtp_config.num_layers)
        pp_mtp_enabled = is_pp_stage and self.mtp_config.enabled

        carry_in: torch.Tensor | None = None
        mtp_embed_inputs: tuple[torch.Tensor, ...] = ()
        if is_pp_stage and not is_first_stage:
            carry_offset = 0
            if uses_indexshare:
                if not pipeline_carries:
                    raise ValueError("A non-first HY V4 pipeline stage requires the IndexShare top-k carry.")
                carry_in = pipeline_carries[0]
                carry_offset = 1
            mtp_embed_inputs = tuple(pipeline_carries[carry_offset:])
            if len(mtp_embed_inputs) != mtp_depth:
                raise ValueError(
                    f"A non-first HY V4 pipeline stage expected {mtp_depth} MTP embedding carries, "
                    f"got {len(mtp_embed_inputs)}."
                )
        elif pipeline_carries:
            raise ValueError("The first or non-pipeline HY V4 stage does not accept pipeline carries.")

        if self.mtp_config.enabled and self.training and is_first_stage:
            if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
                raise ValueError("HY V4 MTP requires integer token input IDs.")
            cp_size = int(attn_kwargs.get("cp_size", 1) or 1)
            requires_global_mtp_inputs = cp_size > 1 or (
                pp_mtp_enabled and getattr(self, "_pp_mtp_targets_in_output", False)
            )
            if requires_global_mtp_inputs and (mtp_per_depth_input_ids is None or mtp_per_depth_position_ids is None):
                raise ValueError(
                    "HY V4 MTP with pipeline/context parallelism requires globally shifted "
                    "per-depth input IDs and position IDs"
                )

        # Work on a fresh dict so activation-checkpoint recomputation sees the
        # same unsqueezed metadata as the original forward.
        attn_kwargs = dict(attn_kwargs)
        if position_ids is None:
            token_count = input_ids.shape[-1] if is_first_stage else input_ids.shape[0]
            position_ids = torch.arange(token_count, device=input_ids.device).unsqueeze(0)
        input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
            input_ids, position_ids, padding_mask, attn_kwargs
        )

        prev_topk_indices = carry_in.to(torch.int32) if carry_in is not None else None
        hidden, topk_indices = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=None,
            padding_mask=padding_mask,
            prev_topk_indices=prev_topk_indices,
            **attn_kwargs,
        )

        # Build future-token embeddings only after the first root forward has
        # run, so FSDP2 has completed lazy initialization for embed_tokens.
        if pp_mtp_enabled and is_first_stage:
            mtp_embed_inputs = self._build_mtp_embed_inputs_for_pp(
                input_ids,
                position_ids=position_ids,
                attn_kwargs=attn_kwargs,
                mtp_per_depth_input_ids=mtp_per_depth_input_ids,
                mtp_per_depth_position_ids=mtp_per_depth_position_ids,
            )

        if is_pp_stage and not has_lm_head:
            outputs: list[torch.Tensor] = [hidden]
            if uses_indexshare:
                if topk_indices is None:
                    raise RuntimeError("HY V4 IndexShare pipeline stage did not produce a top-k carry.")
                if carry_in is not None:
                    hidden = hidden + (carry_in.float().sum() * 0.0).to(hidden.dtype)
                    outputs[0] = hidden
                zero_from_hidden = hidden.float().sum() * 0.0
                outputs.append(topk_indices.to(torch.float32) + zero_from_hidden)
            outputs.extend(mtp_embed_inputs)
            return tuple(outputs) if len(outputs) > 1 else outputs[0]

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp_config.enabled and self.training:
            mtp_per_depth_h = self._run_mtp(
                hidden,
                input_ids=input_ids if is_first_stage else None,
                position_ids=position_ids,
                attention_mask=None,
                padding_mask=padding_mask,
                mtp_embed_inputs=mtp_embed_inputs or None,
                mtp_per_depth_input_ids=None if mtp_embed_inputs else mtp_per_depth_input_ids,
                mtp_per_depth_position_ids=mtp_per_depth_position_ids,
                attn_kwargs=attn_kwargs,
            )
            mtp_per_depth_h = [h.unsqueeze(0) if h.dim() == 2 else h for h in mtp_per_depth_h]
        elif pp_mtp_enabled and has_lm_head:
            mtp_per_depth_h = [hidden.new_empty((1, hidden.shape[0], hidden.shape[-1])) for _ in range(mtp_depth)]

        emits_hidden_states = is_pp_stage and getattr(self, "_pp_return_hidden_states", False) is True
        if emits_hidden_states:
            primary_output = hidden.unsqueeze(0) if hidden.dim() == 2 else hidden
            if carry_in is not None:
                primary_output = primary_output + (carry_in.float().sum() * 0.0).to(primary_output.dtype)
            if pp_mtp_enabled:
                if mtp_per_depth_h is None:
                    raise RuntimeError("The final HY V4 pipeline stage did not produce MTP outputs.")
                seq_idx = self._packed_seq_idx_for_pp(primary_output, attn_kwargs)
                mtp_targets = self._mtp_target_tail_for_pp(primary_output, mtp_per_depth_targets)
                return (primary_output, *mtp_per_depth_h, seq_idx, *mtp_targets)
            return primary_output

        lm_output = compute_lm_head_logits(
            self.lm_head,
            hidden,
            logits_to_keep,
            is_thd=True,
            output_hidden_states=output_hidden_states,
        )
        logits = lm_output.logits
        if carry_in is not None:
            logits = logits + (carry_in.float().sum() * 0.0).to(logits.dtype)

        if is_pp_stage:
            if pp_mtp_enabled:
                if mtp_per_depth_h is None:
                    raise RuntimeError("The final HY V4 pipeline stage did not produce MTP outputs.")
                seq_idx = self._packed_seq_idx_for_pp(logits, attn_kwargs)
                mtp_targets = self._mtp_target_tail_for_pp(logits, mtp_per_depth_targets)
                return (logits, *mtp_per_depth_h, seq_idx, *mtp_targets)
            return logits

        return HyV4CausalLMOutput(
            logits=logits,
            hidden_states=lm_output.hidden_states,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=(self.mtp_config.loss_scaling_factor if mtp_per_depth_h is not None else None),
        )

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        # Construction already assigns model-dtype weights and FP32 iHC/router
        # parameters. A blanket ``to(dtype)`` here would corrupt the latter.


ModelClass = HyV4ForCausalLM

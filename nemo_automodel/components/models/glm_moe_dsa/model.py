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

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.checkpoint.utils import reject_unsupported_tied_word_embeddings
from nemo_automodel.components.models.common import (
    BackendConfig,
    compute_lm_head_logits,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
)
from nemo_automodel.components.models.glm_moe_dsa.cp import make_glm_dsa_packed_cp_batch_and_ctx
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaMLA
from nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter import GlmMoeDsaStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    def __init__(self, layer_idx: int, config: GlmMoeDsaConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        # IndexShare: per-layer indexer mode from `config.indexer_types`. A "shared" layer
        # owns no indexer and reuses the previous "full" layer's top-k selection. Absent the
        # field (e.g. GLM-5.1, which runs a full indexer every layer), every layer is "full".
        indexer_types = getattr(config, "indexer_types", None)
        self.skip_topk = indexer_types is not None and indexer_types[layer_idx] == "shared"
        self.self_attn = GlmMoeDsaMLA(config, backend, skip_topk=self.skip_topk)

        # Thread dtype from config.torch_dtype so the block's own params stay
        # aligned with the rest of the model (fp32 under fp32 master weights).
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        mlp_layer_types = getattr(config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            is_moe_layer = mlp_layer_types[layer_idx] == "sparse"
        else:
            first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
            is_moe_layer = layer_idx >= first_k_dense_replace

        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
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

        ``topk_indices`` is this layer's DSA selection — freshly computed on "full" layers,
        or ``prev_topk_indices`` passed through on "shared" layers — so the caller can thread
        it to subsequent shared layers (GLM IndexShare).
        """
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out, topk_indices = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            prev_topk_indices=prev_topk_indices,
            return_topk_indices=True,
            **attn_kwargs,
        )
        x = x + attn_out

        mlp_out = self._mlp(x=self.post_attention_layernorm(x), padding_mask=padding_mask)
        x = x + mlp_out
        return x, topk_indices

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if isinstance(self.mlp, MLP):
            return self.mlp(x)
        else:
            assert isinstance(self.mlp, MoE)
            return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class GlmMoeDsaModel(nn.Module):
    def __init__(
        self,
        config: GlmMoeDsaConfig,
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
            softmax_before_topk=False,
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

        self.max_seq_len = config.max_position_embeddings
        self.qk_rope_head_dim = config.qk_rope_head_dim

        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            rope_theta = config.rope_parameters["rope_theta"]
        else:
            rope_theta = config.rope_theta

        rope_scaling = getattr(config, "rope_scaling", None)

        self.freqs = precompute_freqs_cis(
            qk_rope_head_dim=self.qk_rope_head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )

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
        """Run the decoder stack, returning ``(hidden_states, topk_indices)``.

        ``prev_topk_indices`` seeds the IndexShare running selection (used under pipeline
        parallelism, where an earlier "full" layer lives on the previous stage); it is ``None``
        in the single-process path. The returned ``topk_indices`` is the running selection at the
        end of this stage's layers, so it can be carried to the next pipeline stage.
        """
        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        freqs_cis = freqs_cis_from_position_ids(
            position_ids,
            self.freqs.to(position_ids.device),
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        # IndexShare: thread the most recent "full" layer's top-k selection forward so the
        # following "shared" layers can reuse it. Seeded from `prev_topk_indices` (carried from
        # the previous pipeline stage); `None` on the first stage / all-"full" configs (GLM-5.1).
        topk_indices = prev_topk_indices
        for layer in self.layers.values():
            h, topk_indices = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                prev_topk_indices=topk_indices,
                **attn_kwargs,
            )

        h = self.norm(h) if self.norm else h
        return h, topk_indices

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class GlmMoeDsaForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        # Model-owned CP: GLM DSA shards its packed THD batch itself.
        cp_style: str = "model_owned"
        cp_layout: str = "packed_thd"

    @classmethod
    def from_config(
        cls,
        config: GlmMoeDsaConfig,
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
        config = GlmMoeDsaConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tied_word_embeddings(config, type(self).__name__)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = GlmMoeDsaModel(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = GlmMoeDsaStateDictAdapter(
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

    def should_pack_validation_with_training(self) -> bool:
        """GLM DSA TileLang kernels require validation to use the THD packed layout."""
        return getattr(self.backend, "attn", None) == "tilelang"

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Attach GLM DSA's packed THD context-parallel batch sharder.

        Args:
            batch: The batch dict.
            num_chunks: Number of chunks for load-balanced CP sharding.
        """
        from functools import partial  # noqa: PLC0415

        from nemo_automodel.components.distributed.cp_sharder import (  # noqa: PLC0415
            CPSharder,
            contiguous_local_indices,
        )

        if getattr(self.backend, "attn", None) != "tilelang":
            raise NotImplementedError("GLM DSA context parallelism is implemented only for backend.attn='tilelang'.")

        return {
            "cp_sharder": CPSharder(
                shard_batch=partial(
                    make_glm_dsa_packed_cp_batch_and_ctx,
                    num_chunks=int(num_chunks),
                ),
                # Contiguous over the packed THD token axis: rank r keeps
                # tokens [r * T/cp, (r + 1) * T/cp).
                local_token_global_indices=contiguous_local_indices,
                layout="packed_thd",
            ),
        }

    def _is_pipeline_parallel_stage(self) -> bool:
        """True when this module is a trimmed pipeline-parallel stage (not the whole model)."""
        if self.lm_head is None:
            return True
        if self.model.embed_tokens is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.num_hidden_layers)
        except TypeError:
            return False

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Declare PP inter-stage I/O metas, threading the IndexShare top-k as a carry tensor.

        Non-first stages additionally receive the previous "full" layer's top-k selection, and
        non-last stages emit the running selection, so a stage that begins with a "shared" layer
        has the top-k it needs (correct at any sequence length).
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        # TileLang's fused indexer always returns ``index_topk`` columns. Under
        # CP the query length is sharded (for example 4096 / cp8 = 512), while
        # K/V are gathered inside the model, so capping by the local query
        # length would under-declare the inter-stage carry shape.
        index_topk = int(self.config.index_topk)
        topk = index_topk if self.backend.attn == "tilelang" else min(index_topk, seq_len)

        def meta(shape: tuple[int, ...], dt: torch.dtype) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dt)

        # The inter-stage tensor RANK matches the attention backend's data format, so each stage's
        # forward emits its natural tensors and no per-boundary reshape is needed:
        #   * TileLang DSA runs in THD (packed; batch folded into the token axis) -> 2D hidden
        #     ``[T, H]`` and top-k ``[T, 1, topk]`` (the tilelang layout). tilelang implies THD.
        #   * sdpa/te/eager run dense bshd -> 3D hidden ``[B, S, H]`` and top-k ``[B, S, topk]``.
        # Top-k indices cross the boundary as float32: torch.distributed.pipelining calls
        # ``requires_grad_(True)`` on recv buffers and int dtypes can't require grad; float32 holds
        # the index values losslessly and ``forward`` casts back (int32 for tilelang, int64 dense).
        thd = self.backend.attn == "tilelang"
        if thd:
            hidden_meta = meta((seq_len, hidden_size), dtype)
            topk_meta = meta((seq_len, 1, topk), torch.float32)
        else:
            hidden_meta = meta((microbatch_size, seq_len, hidden_size), dtype)
            topk_meta = meta((microbatch_size, seq_len, topk), torch.float32)

        if is_first:
            inputs_meta = (meta((microbatch_size, seq_len), torch.long),)
        else:
            inputs_meta = (hidden_meta, topk_meta)

        if self.lm_head is not None:
            # The last stage emits logits; compute_lm_head_logits restores [1, T, V] under THD.
            outputs_meta = (meta((microbatch_size, seq_len, vocab_size), dtype),)
        else:
            outputs_meta = (hidden_meta, topk_meta)

        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.Tensor,
        *carry: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...] | torch.Tensor:
        """Forward pass.

        Single process (no pipeline parallelism): returns
        :class:`~transformers.modeling_outputs.CausalLMOutputWithPast`, threading the IndexShare
        top-k internally (seeded ``None``).

        Pipeline parallelism: ``input_ids`` is the upstream hidden state on non-first stages and
        ``*carry`` holds the previous stage's running top-k selection. Non-last stages return
        ``(hidden_states, topk_indices)`` and the last stage returns the ``logits`` tensor.

        Args:
            input_ids: Token IDs (BSHD ``[B, S]`` / THD ``[1, T]``) on the first stage, or the
                upstream hidden state on later pipeline stages.
            carry: Optional ``(topk_indices,)`` carried from the previous pipeline stage.
            position_ids / attention_mask / padding_mask: Optional masks / positions.
            logits_to_keep: If ``0``, project all positions; else only the last ``logits_to_keep``.
            output_hidden_states: When set (single-process), carry final hidden states on the output.
            **attn_kwargs: Additional arguments forwarded to the base model.
        """
        # Model-owned context-parallel input prep. The recipe routes the batch
        # through ``__call__(_pre_embed_only=True)`` before CP sharding so the model
        # can attach its own ``cp_sharder`` (see ``prepare_model_inputs_for_cp``).
        if attn_kwargs.pop("_pre_embed_only", False):
            return self.prepare_model_inputs_for_cp(
                {"input_ids": input_ids}, num_chunks=attn_kwargs.pop("num_chunks", 1)
            )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        # Carry-in arrives as float32 (see get_pipeline_stage_metas, where the pipeline recv
        # buffer must be a grad-capable dtype); restore the int64 index values.
        carry_in = carry[0] if carry else None
        is_thd = attn_kwargs.get("qkv_format") == "thd"

        prev_topk_indices = None
        if carry_in is not None:
            # The carry arrives in the backend's natural top-k layout (THD: [T, 1, topk]; bshd:
            # [B, S, topk]) as float32. tilelang SparseMLA requires int32 indices; the dense path
            # uses int64. Only the dtype differs -- no reshape (see get_pipeline_stage_metas).
            prev_topk_indices = carry_in.to(torch.int32) if is_thd else carry_in.to(torch.int64)

        # THD: squeeze the leading batch dim on EVERY stage. First stage ``input_ids`` is token ids
        # [1, T]; later stages receive the upstream hidden state [1, T, H] (the 3D pipeline meta).
        # squeeze_input_for_thd handles both (plain ``.squeeze(0)``) and also squeezes
        # position_ids / cu_seqlens so the 2D-THD DSA layers get consistent shapes on every stage.
        if is_thd:
            # squeeze_input_for_thd mutates attn_kwargs in place; under activation-checkpointing the
            # stage forward is recomputed for backward, and a mutated (already-squeezed) dict on the
            # second pass yields different input metadata -> CheckpointError. Pass a fresh copy so
            # the forward and its recompute are deterministic.
            attn_kwargs = dict(attn_kwargs)
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        hidden, topk_indices = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            prev_topk_indices=prev_topk_indices,
            **attn_kwargs,
        )

        if self._is_pipeline_parallel_stage():
            # The top-k carry is non-differentiable (integer indices transported as float32), but
            # torch.distributed.pipelining treats every float inter-stage tensor as an activation
            # and demands a gradient for it on backward. We add zero-weight autograd links so the
            # carry it RECEIVES and the carry it SENDS both get a DEFINED (zero) gradient instead of
            # ``None`` — values are unchanged (``+ x.sum() * 0.0``).
            if self.lm_head is not None:
                # Last stage: emit logits for the pipeline loss.
                logits = compute_lm_head_logits(self.lm_head, hidden, logits_to_keep, is_thd=is_thd).logits
                if carry_in is not None:
                    logits = logits + (carry_in.float().sum() * 0.0).to(logits.dtype)
                return logits
            # Non-last stage: emit (hidden, float32 top-k carry) to the next stage. The tensors are
            # already in the backend's natural pipeline shape (THD: [T, H] + [T, 1, topk]; bshd:
            # [B, S, H] + [B, S, topk]) per get_pipeline_stage_metas, so no reshape is needed.
            # (THD requires packed_sequence_size >= index_topk so the tilelang top-k width matches
            # the meta's min(index_topk, seq_len).)
            zero_from_hidden = hidden.float().sum() * 0.0  # connected to grad-bearing hidden
            carry_out = topk_indices.to(torch.float32) + zero_from_hidden  # requires grad, value unchanged
            if carry_in is not None:
                hidden = hidden + (carry_in.float().sum() * 0.0).to(hidden.dtype)  # defines grad(carry_in)
            return hidden, carry_out

        return compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, is_thd=is_thd, output_hidden_states=output_hidden_states
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

        self.to(dtype)
        for layer in self.model.layers.values():
            if isinstance(layer.mlp, MoE):
                layer.mlp.gate.e_score_correction_bias = torch.zeros(
                    (self.config.n_routed_experts), dtype=torch.float32
                ).to(buffer_device)


ModelClass = GlmMoeDsaForCausalLM

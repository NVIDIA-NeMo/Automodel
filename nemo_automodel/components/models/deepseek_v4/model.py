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

"""DeepSeek V4 Model.

Key architectural points (from official inference/model.py):

HC (Hyper-Connections):
  Every transformer block maintains hc_mult=4 copies of the hidden state.
  The embedding output is expanded: [B,S,dim] -> [B,S,hc_mult,dim].
  hc_pre  reduces [B,S,hc_mult,dim] -> [B,S,dim] before attn/ffn.
  hc_post expands [B,S,dim] -> [B,S,hc_mult,dim] after attn/ffn.
  Full HC requires the hc_split_sinkhorn CUDA kernel.
  Current fallback: mean-pooling for hc_pre, broadcast add for hc_post.

HC parameters (ALL layers, stored in float32):
  hc_attn_fn    : [mix_hc, hc_mult*dim]  where mix_hc = (2+hc_mult)*hc_mult = 24
  hc_attn_base  : [mix_hc]
  hc_attn_scale : [3]
  hc_ffn_fn     : [mix_hc, hc_mult*dim]
  hc_ffn_base   : [mix_hc]
  hc_ffn_scale  : [3]

Gate hash layers (layer_idx < num_hash_layers):
  Instead of score-based routing, the gate uses a fixed token-id -> expert-id
  lookup table (tid2eid: [vocab_size, n_activated_experts]).

All layers use MoE FFN (no dense layers).
Compress-ratio sliding-window attention is not yet implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RotaryEmbedding,
    build_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass
class DeepseekV4CausalLMOutput:
    """Output of DeepseekV4ForCausalLM.forward.

    Attributes:
        logits: ``[B, S, vocab_size]`` next-token prediction logits.
        mtp_per_depth_h: Per-depth MTP hidden states (training mode only).
            List of length ``num_nextn_predict_layers``, each ``[B, S, hidden]``.
            ``None`` when MTP is disabled or in eval mode.
        mtp_loss_scaling_factor: Coefficient for the MTP auxiliary loss.
    """

    logits: torch.Tensor
    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float = 0.1


class DeepseekV4Block(nn.Module):
    """Single transformer block for DeepSeek V4.

    Uses HuggingFace transformers PR 45616's HyperConnection decoder-layer
    pattern: two ``DeepseekV4HyperConnection`` modules own the collapse /
    expand mixer weights at the attention and FFN sites respectively.
    Checkpoint's flat ``hc_attn_*`` / ``hc_ffn_*`` keys are routed into
    ``attn_hc.*`` / ``ffn_hc.*`` by the state-dict adapter.
    """

    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV4Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult

        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.self_attn = DeepseekV4Attention(config, layer_idx=layer_idx, backend=backend)
        self.mlp = MoE(moe_config, backend)
        # Hash routing: the first ``num_hash_layers`` layers use a fixed
        # tid2eid lookup table instead of the score-based generic Gate.
        # Swap after MoE construction so the rest of MoE (experts, shared
        # experts, etc.) keeps its standard layout.
        self.is_hash_routing_layer = layer_idx < int(getattr(config, "num_hash_layers", 0) or 0)
        if self.is_hash_routing_layer:
            self.mlp.gate = DeepseekV4HashGate(config, moe_config)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        # HC (Hyper-Connection) mixers — one per sub-site (attention + FFN).
        # Each owns learnable ``fn`` (fp32 packed-linear), ``base`` (fp32 bias),
        # ``scale`` (fp32 per-head gain) parameters.  ``_keep_in_fp32_modules_strict``
        # on ``DeepseekV4ForCausalLM`` keeps all nine HC param tensors in fp32
        # at runtime via submodule-name matching.
        hc_kwargs = dict(
            hc_mult=config.hc_mult,
            hidden_size=config.hidden_size,
            hc_sinkhorn_iters=int(getattr(config, "hc_sinkhorn_iters", 20) or 20),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
        )
        self.attn_hc = DeepseekV4HyperConnection(**hc_kwargs)
        self.ffn_hc = DeepseekV4HyperConnection(**hc_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_compress: nn.Module | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        # x throughout this layer: [B, S, hc_mult, hidden] (HC multi-copy state)
        # padding_mask is only used by the MoE module; only derive it from a 2D
        # raw attention_mask (1=valid, 0=pad).  When attention_mask is the 4D
        # additive mask built upstream, the caller is expected to supply
        # padding_mask separately (or leave it None for the no-pad case).
        if attention_mask is not None and padding_mask is None and attention_mask.dim() == 2:
            padding_mask = attention_mask.bool().logical_not()

        # --- Attention site: collapse → norm → attn → expand ---
        pre, post, comb = self.attn_hc.compute_weights(x)
        collapsed = (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=rotary_compress,
        )
        dtype = x.dtype
        # Expand: native DSV4 uses comb[j, h] * residual[j], i.e. comb.T @ residual.
        x = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2).to(dtype), x)

        # --- MLP site: same pattern ---
        pre, post, comb = self.ffn_hc.compute_weights(x)
        collapsed = (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)
        # Hash-routing layers need the current batch's input_ids to do the
        # tid2eid lookup; stash it on the gate just before the MoE call.
        if self.is_hash_routing_layer and isinstance(self.mlp.gate, DeepseekV4HashGate):
            self.mlp.gate.set_input_ids(input_ids)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), padding_mask)
        dtype = x.dtype
        return post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2).to(dtype), x)

    def init_weights(self, buffer_device: torch.device) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)
        # HC mixer params stay at whatever the checkpoint provides (init.normal_
        # on ``fn``, init.zeros_ on ``base``, init.ones_ on ``scale`` for random
        # init — matches HF's _init_weights at modular_deepseek_v4.py:923-926).


class DeepseekV4HashGate(nn.Module):
    """Hash gate for first num_hash_layers: routes tokens via a fixed lookup table.

    Instead of computing routing scores, the gate uses tid2eid[token_id] to
    pre-assign expert indices.  The routing weight is still computed from the
    gate weight but the *selection* is deterministic per token id.

    tid2eid shape: [vocab_size, n_activated_experts]  (int32, non-trainable)

    Signature matches ``components.moe.layers.Gate`` — ``forward(x, token_mask,
    cp_mesh)`` returning ``(weights, indices, aux_loss)`` — so the generic MoE
    module can call it interchangeably.  The per-forward ``input_ids`` needed
    for the tid2eid lookup is stashed on the module by the enclosing Block via
    :meth:`set_input_ids` immediately before the MoE call.
    """

    def __init__(self, config: DeepseekV4Config, moe_config: MoEConfig):
        super().__init__()
        self.topk = moe_config.n_activated_experts
        self.n_experts = moe_config.n_routed_experts
        self.score_func = moe_config.score_func
        self.route_scale = moe_config.route_scale
        self.norm_topk_prob = moe_config.norm_topk_prob

        # Routing score weight (used to compute weights, not for selection)
        self.weight = nn.Parameter(torch.zeros(self.n_experts, config.hidden_size))
        # Token-id -> expert-id lookup table.  Registered as a persistent
        # buffer (not a Parameter) because FSDP's param-sharding path rejects
        # int tensors via .requires_grad_(), and the table is non-trainable
        # anyway.  Dtype matches the V4 Flash checkpoint on-disk layout (I64).
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.topk, dtype=torch.int64),
            persistent=True,
        )
        # Kept for API compat with the generic Gate (e.g. optimizer sync paths
        # that probe for .bias) — hash layers have no learnable bias.
        self.bias = None
        # Ephemeral per-forward input_ids set by the Block (not a parameter /
        # buffer; cleared after each forward to avoid holding references).
        self._pending_input_ids: torch.Tensor | None = None

    def set_input_ids(self, input_ids: torch.Tensor | None) -> None:
        """Stash the current batch's input_ids for the next ``forward`` call."""
        self._pending_input_ids = input_ids

    def update_bias(self) -> None:
        """No-op for compat with callers that walk MoE gates and call update_bias."""

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        nn.init.zeros_(self.weight)
        with torch.no_grad():
            self.tid2eid.zero_()

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        cp_mesh: "DeviceMesh | None" = None,  # noqa: F821 — MoE passes it but we do not need it
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        import torch.nn.functional as F

        input_ids = self._pending_input_ids
        # Clear immediately so a stale cached tensor cannot leak to a later
        # forward that forgets to set it.
        self._pending_input_ids = None

        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(dim=-1)

        if input_ids is not None:
            indices = self.tid2eid[input_ids.flatten().to(torch.int64)]
        else:
            # Fallback to score-based topk — keeps the module usable in tests or
            # PP stages where input_ids is not threaded through.
            indices = scores.topk(self.topk, dim=-1)[1]

        weights = scores.gather(1, indices.long())
        if self.score_func != "softmax":
            denom = weights.sum(dim=-1, keepdim=True) + 1e-20
            weights = weights / denom
        weights = weights * self.route_scale
        return weights.type_as(x), indices, None


class DeepseekV4Model(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
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

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.moe_intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            # V4 has no group-limited routing (noaux_tc with no n_group/topk_group)
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=1e-3,
            score_func="sqrtsoftplus",
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0,
            norm_topk_prob=config.norm_topk_prob,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            # V4 Flash routed experts use clamped SwiGLU (gate.max=limit,
            # up.±limit) in FP32 — see reference model.py Expert.forward.
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0) or 0.0),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.layers = nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = DeepseekV4Block(layer_id, config, self.moe_config, backend)

        # Final HC collapse: sigmoid-weighted sum across hc_mult streams before
        # the shared RMSNorm + lm_head.  Ported from HF PR 45616's
        # ``DeepseekV4HyperHead``.  Owns ``hc_fn`` / ``hc_base`` / ``hc_scale``
        # — all kept in fp32 via ``_keep_in_fp32_modules_strict`` (see
        # ``DeepseekV4ForCausalLM``).
        self.hc_head = DeepseekV4HyperHead(
            hc_mult=config.hc_mult,
            hidden_size=config.hidden_size,
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
        )

        self.norm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )

        self.max_seq_len = config.max_position_embeddings
        # Two rotary embeddings (HF PR 45616 pattern): main rope for the token
        # attention path, compressor rope for the long-range pooled KV branch.
        # HF partial_rotary_factor = qk_rope_head_dim / head_dim so cos/sin
        # come out sized to qk_rope_head_dim.
        partial_rotary_factor = float(config.qk_rope_head_dim) / float(config.head_dim)
        # Reference (``dsv4flash/inference/model.py:519-525``) only applies YaRN
        # to the compress-rope path: when compress_ratio>0 it uses
        # ``original_seq_len=args.original_seq_len`` and theta=compress_rope_theta;
        # otherwise ``original_seq_len=0`` (YaRN disabled) and theta=rope_theta.
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = DeepseekV4RotaryEmbedding(
            rope_theta=float(config.rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=None,
        )
        self.rotary_emb_compress = DeepseekV4RotaryEmbedding(
            rope_theta=float(getattr(config, "compress_rope_theta", 160000.0) or 160000.0),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        return_hc_hidden: bool = False,
        **attn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # PP-aware forward (same pattern as DeepseekV3Model.forward).
        # Stage 0 of pipeline parallelism owns ``embed_tokens`` and receives
        # raw token ids; subsequent stages have ``embed_tokens=None`` and
        # receive the previous stage's hidden state in the ``input_ids`` slot
        # (already 4D ``[B, S, hc_mult, hidden]`` because ``DeepseekV4Block``
        # preserves the HC stream axis).  Detect via ``self.embed_tokens is None``
        # rather than via dtype, since the stage trimming pass nulls the
        # attribute when the layer is dropped.
        on_first_stage = self.embed_tokens is not None

        if on_first_stage:
            if input_ids is None and inputs_embeds is None:
                raise ValueError("First PP stage requires input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            # Expand embeddings to hc_mult copies: [B,S,dim] -> [B,S,hc_mult,dim]
            h = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
            shape_ref = inputs_embeds  # 3D ref for rotary / mask sizing
        else:
            # Mid-stage: ``input_ids`` is actually the upstream activation.
            # Either positional (4D float) or via ``inputs_embeds=`` kwarg.
            h = input_ids if input_ids is not None else inputs_embeds
            if h is None:
                raise ValueError("Non-first PP stage expects an inter-stage activation")
            # h is [B, S, hc_mult, hidden]; shape_ref needs 3D [B, S, hidden].
            shape_ref = h.flatten(start_dim=2)[:, :, : self.config.hidden_size]

        if position_ids is None:
            seq_len = shape_ref.shape[1]
            position_ids = torch.arange(seq_len, device=shape_ref.device).unsqueeze(0).expand(shape_ref.shape[0], -1)

        # (cos, sin) pairs for the main attention path and the compressor path.
        # Rotary modules live on every stage (PP keep-list ensures it).
        position_embeddings = self.rotary_emb(shape_ref, position_ids)
        position_embeddings_compress = self.rotary_emb_compress(shape_ref, position_ids)

        # Build the 4D additive causal+padding+SWA mask.  Same band-diagonal
        # pattern HF's ``create_sliding_window_causal_mask`` produces; every
        # layer in the released DSV4-Flash was trained under it.
        sliding_window = int(getattr(self.config, "sliding_window", 0) or 0) or None
        attention_mask_4d = build_causal_padding_mask(
            attention_mask,
            seq_len=shape_ref.shape[1],
            dtype=shape_ref.dtype,
            device=shape_ref.device,
            batch_size=shape_ref.shape[0],
            sliding_window=sliding_window,
        )

        # ``input_ids`` is only meaningful for hash-routing layers, which live
        # on stage 0 (num_hash_layers <= layers per stage 0).  Mid-stages pass
        # None — hash layers shouldn't be present there.
        layer_input_ids = input_ids if on_first_stage else None

        for layer in self.layers.values():
            if layer is None:  # PP-trimmed slot
                continue
            h = layer(
                x=h,
                position_embeddings=position_embeddings,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=self.rotary_emb_compress,
                attention_mask=attention_mask_4d,
                padding_mask=padding_mask
                if padding_mask is not None
                else (
                    attention_mask.bool().logical_not()
                    if attention_mask is not None and attention_mask.dim() == 2
                    else None
                ),
                input_ids=layer_input_ids,
                **attn_kwargs,
            )

        mtp_hc_hidden = h if return_hc_hidden else None

        # Reduce hc_mult copies -> [B,S,dim] via the learned HC head, then
        # apply the shared RMSNorm.  Both modules live ONLY on the last PP
        # stage (intermediate stages keep h at 4D so the next stage can
        # consume it).  Matches HF PR 45616's ``DeepseekV4Model.forward``.
        if getattr(self, "hc_head", None) is not None:
            h = self.hc_head(h)
        if getattr(self, "norm", None) is not None:
            h = self.norm(h)
        if return_hc_hidden:
            if mtp_hc_hidden is None:
                raise ValueError("return_hc_hidden requested before HC stream was available")
            return h, mtp_hc_hidden
        return h

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.layers.values():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)


class DeepseekV4ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    # Keep HC mixers and the MoE gate's correction bias in fp32 regardless of
    # the outer cast policy.  Matches HF PR 45616's
    # ``DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict`` (lines 890-900
    # of modular_deepseek_v4.py) plus the existing ``e_score_correction_bias``
    # entry that is specific to KAutomodel's shared Gate buffer.
    _keep_in_fp32_modules_strict = [
        "attn_hc.fn",
        "attn_hc.base",
        "attn_hc.scale",
        "ffn_hc.fn",
        "ffn_hc.base",
        "ffn_hc.scale",
        "hc_head.hc_fn",
        "hc_head.hc_base",
        "hc_head.hc_scale",
        "e_score_correction_bias",
    ]

    @classmethod
    def from_config(
        cls,
        config: DeepseekV4Config,
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
        config = DeepseekV4Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: DeepseekV4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        mtp_loss_scaling_factor = kwargs.pop("mtp_loss_scaling_factor", 0.1)
        self.model = DeepseekV4Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepSeekV4StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )

        # MTP construction (import inside __init__ to avoid circular imports).
        from nemo_automodel.components.models.deepseek_v4.mtp import (  # noqa: PLC0415
            build_deepseek_v4_mtp,
            build_mtp_config_from_hf,
        )

        self.mtp_config = build_mtp_config_from_hf(config, loss_scaling_factor=mtp_loss_scaling_factor)
        if self.mtp_config.enabled:
            self.mtp = build_deepseek_v4_mtp(
                config=config,
                mtp_config=self.mtp_config,
                backend=self.backend,
                moe_config=self.model.moe_config,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
                rotary_emb=self.model.rotary_emb,
                rotary_emb_compress=self.model.rotary_emb_compress,
            )
        else:
            self.mtp = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep DSV4 non-layer PP dependencies with the stages that need them."""

        text_model = text_model or self.model
        stage_modules = [list(modules) for modules in module_names_per_stage]

        def append_once(modules: list[str], fqn: str) -> None:
            if fqn not in modules:
                modules.append(fqn)

        if getattr(text_model, "rotary_emb_compress", None) is not None:
            for modules in stage_modules:
                append_once(modules, f"{layers_prefix}rotary_emb_compress")
        if getattr(text_model, "hc_head", None) is not None:
            append_once(stage_modules[-1], f"{layers_prefix}hc_head")
        if self.mtp is not None:
            append_once(stage_modules[-1], "mtp")

        return stage_modules

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return PP input/output meta tensors for DSV4's HC and MTP contract."""

        hidden_shape = (microbatch_size, seq_len, self.config.hidden_size)
        hc_hidden_shape = (microbatch_size, seq_len, self.config.hc_mult, self.config.hidden_size)
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)

        def meta(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dtype)

        def append_mtp_metas(primary: torch.Tensor) -> tuple[torch.Tensor, ...]:
            mtp_metas = (meta(hidden_shape) for _ in range(mtp_depth))
            return (primary, *mtp_metas)

        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = append_mtp_metas(meta(hc_hidden_shape if self.config.hc_mult > 1 else hidden_shape))

        if self.lm_head is not None:
            output_meta = meta((microbatch_size, seq_len, self.config.vocab_size))
        elif getattr(self.model, "norm", None) is not None:
            output_meta = meta(hidden_shape)
        else:
            output_meta = meta(hc_hidden_shape if self.config.hc_mult > 1 else hidden_shape)

        return inputs_meta, append_mtp_metas(output_meta)

    def _is_pipeline_parallel_stage(self) -> bool:
        if self.lm_head is None:
            return True
        if getattr(self.model, "embed_tokens", None) is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.num_hidden_layers)
        except TypeError:
            return False

    def _build_mtp_embed_inputs_for_pp(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if getattr(self.model, "embed_tokens", None) is None:
            raise ValueError("First PP stage must own embed_tokens to build MTP embeddings")
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("First PP stage must receive token ids to build MTP embeddings")

        from nemo_automodel.components.models.common.mtp import roll_tensor  # noqa: PLC0415

        cur_input_ids = input_ids
        embeds = []
        for _ in range(self.mtp_config.num_layers):
            cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
            embeds.append(self.model.embed_tokens(cur_input_ids))
        return tuple(embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        *mtp_embed_inputs: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> "DeepseekV4CausalLMOutput" | tuple[torch.Tensor, ...] | torch.Tensor:
        is_pp_stage = self._is_pipeline_parallel_stage()
        pp_mtp_enabled = is_pp_stage and self.mtp_config.enabled

        thd_mode = "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd"
        if thd_mode:
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        use_mtp = self.mtp is not None and self.training
        model_out = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            return_hc_hidden=use_mtp,
            **attn_kwargs,
        )
        if use_mtp:
            hidden_states, mtp_hc_hidden = model_out
        else:
            hidden_states = model_out
            mtp_hc_hidden = None
        logits = self.lm_head(hidden_states) if self.lm_head else hidden_states
        if thd_mode:
            logits = logits.unsqueeze(0)

        if pp_mtp_enabled and self.lm_head is None:
            if not mtp_embed_inputs:
                mtp_embed_inputs = self._build_mtp_embed_inputs_for_pp(input_ids)
            return (logits, *mtp_embed_inputs)

        mtp_per_depth_h = None
        if use_mtp:
            if is_pp_stage and not mtp_embed_inputs:
                raise ValueError("Final PP stage requires propagated MTP embeddings")
            # MTP consumes the pre-final-head HC stream [B, S, hc_mult, hidden]
            # and returns collapsed per-depth [B, S, hidden] tensors for CE.
            seq_len = hidden_states.shape[1]
            batch_size = hidden_states.shape[0]
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            sliding_window = int(getattr(self.config, "sliding_window", 0) or 0) or None
            mtp_attn_mask = build_causal_padding_mask(
                attention_mask,
                seq_len=seq_len,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
                batch_size=batch_size,
                sliding_window=sliding_window,
            )
            mtp_kwargs = {
                "hidden_states": mtp_hc_hidden,
                "position_ids": position_ids,
                "attention_mask": mtp_attn_mask,
                "padding_mask": padding_mask,
            }
            if mtp_embed_inputs:
                mtp_kwargs["embed_inputs"] = tuple(mtp_embed_inputs)
            else:
                mtp_kwargs["input_ids"] = input_ids
                mtp_kwargs["embed_fn"] = self.model.embed_tokens
            mtp_per_depth_h = self.mtp(**mtp_kwargs)
        elif pp_mtp_enabled and self.lm_head is not None:
            mtp_per_depth_h = [hidden_states.new_empty(hidden_states.shape) for _ in range(self.mtp_config.num_layers)]

        if is_pp_stage:
            if pp_mtp_enabled:
                if self.training and self.mtp is None:
                    raise ValueError("Final PP stage has MTP enabled but does not own the MTP module")
                return (logits, *mtp_per_depth_h)
            return logits

        return DeepseekV4CausalLMOutput(
            logits=logits,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
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


ModelClass = DeepseekV4ForCausalLM

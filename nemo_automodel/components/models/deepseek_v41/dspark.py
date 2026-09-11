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

"""Native DeepSeek V4.1 DSpark draft backbone.

The released draft is stored under ``mtp.0`` through ``mtp.2`` but is not the
autoregressive MTP objective used by earlier DeepSeek models. It is a parallel
five-position drafter whose three blocks use sliding-window MLA, routed MoE and
single-pass mHC. This module implements the trainable, cache-free backbone; the
shared frozen embedding/LM head and anchor sampling remain owned by the generic
DSpark trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4VisionGate
from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection, DeepseekV41RMSNorm
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str


@dataclass(frozen=True)
class DeepseekV41DSparkBackboneOutput:
    """Native draft states consumed by the released output heads.

    Attributes:
        hidden_states: Collapsed pre-normalization states of shape [batch,
            draft_sequence, hidden]. The confidence head consumes these states.
        normalized_hidden_states: Final-normalized states of shape [batch,
            draft_sequence, hidden]. The frozen target LM head consumes these
            states to produce base token logits.
    """

    hidden_states: torch.Tensor
    normalized_hidden_states: torch.Tensor


class _DeepseekV41DSparkAttention(DeepseekV41Attention):
    """Cache-free DSpark attention over target context and parallel draft blocks."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        if backend.attn not in ("eager", "sdpa"):
            raise ValueError("DeepSeek V4.1 DSpark attention supports backend.attn='eager' or 'sdpa'")
        super().__init__(config, layer_idx, backend)
        if self.compress_ratio != 0 or self.compressor is not None or self.indexer is not None:
            raise ValueError("DeepSeek V4.1 DSpark layers must be SWA-only with compress_ratio=0")

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> DeepseekV41DSparkBackboneOutput:
        """Attend draft queries to target context and their own parallel block.

        Args:
            hidden_states: Draft tensor of shape [batch, draft_sequence, hidden].
            target_hidden_states: Projected target tensor of shape
                [batch, context_sequence, hidden].
            position_ids: Integer tensor of shape [batch, context_sequence +
                draft_sequence], containing absolute positions for both regions.
            attention_mask: Additive tensor broadcastable to shape [batch, heads,
                draft_sequence, context_sequence + draft_sequence], with zero for
                visible keys and negative infinity for masked keys.

        Returns:
            Tensor of shape [batch, draft_sequence, hidden]. Inputs are not mutated.
        """
        batch, draft_sequence, _ = hidden_states.shape
        context_sequence = target_hidden_states.shape[1]
        if target_hidden_states.shape[0] != batch:
            raise ValueError("DSpark target and draft hidden states must have the same batch size")
        if position_ids.shape != (batch, context_sequence + draft_sequence):
            raise ValueError("DSpark position_ids must cover the concatenated target and draft sequences")
        expected_mask_shape = (batch, 1, draft_sequence, context_sequence + draft_sequence)
        if attention_mask.shape != expected_mask_shape:
            raise ValueError(f"DSpark attention_mask must have shape {expected_mask_shape}")

        target_angles = self.rotary_emb(position_ids[:, :context_sequence])
        draft_angles = self.rotary_emb(position_ids[:, context_sequence:])
        query_latent = self.q_norm(self.wq_a(hidden_states))
        query = self.wq_b(query_latent).unflatten(-1, (self.num_heads, self.head_dim))
        query = self._apply_rotary(query, draft_angles)

        kv_source = torch.cat((target_hidden_states, hidden_states), dim=1)
        kv_angles = torch.cat((target_angles, draft_angles), dim=1)
        kv = self._apply_rotary(self.kv_norm(self.wkv(kv_source)), kv_angles)

        # The synthetic zero-valued key contributes only the learned sink logit
        # to the softmax denominator, matching the released sparse-attention op.
        kv = torch.cat((kv, kv.new_zeros(batch, 1, self.head_dim)), dim=1)
        bias = attention_mask.expand(-1, self.num_heads, -1, -1).float()
        sink = self.sinks_param(query).view(1, self.num_heads, 1, 1).expand(batch, -1, draft_sequence, -1)
        bias = torch.cat((bias, sink), dim=-1)
        if self.backend.attn == "sdpa":
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                kv.unsqueeze(1),
                kv.unsqueeze(1),
                attn_mask=bias,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.head_dim**-0.5,
            ).transpose(1, 2)
        else:
            logits = torch.einsum("bshd,btd->bhst", query.float(), kv.float()) * self.head_dim**-0.5
            probabilities = (logits + bias).softmax(dim=-1)
            probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
            attended = torch.einsum("bhst,btd->bshd", probabilities, kv.float()).to(query.dtype)
        valid_tokens = torch.ones(batch, draft_sequence, dtype=torch.bool, device=hidden_states.device)
        return self._project_output(attended, draft_angles, valid_tokens)


class _DeepseekV41DSparkMarkovHead(nn.Module):
    """Rank-factorized first-order token-transition bias."""

    def __init__(self, vocab_size: int, rank: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, rank, dtype=dtype)
        self.head = nn.Linear(rank, vocab_size, bias=False, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the transition bias and its conditioning embedding.

        Args:
            token_ids: Integer tensor of shape [...] containing preceding tokens.

        Returns:
            Transition logits of shape [..., vocab] and embeddings of shape
            [..., markov_rank].
        """
        embedding = self.embed(token_ids)
        return self.head(embedding), embedding


class _DeepseekV41DSparkConfidenceHead(nn.Module):
    """Predict conditional acceptance logits in FP32."""

    def __init__(self, hidden_size: int, markov_rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor, markov_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict a conditional acceptance logit for every draft position.

        Args:
            hidden_states: Tensor of shape [..., hidden].
            markov_embeddings: Tensor of shape [..., markov_rank] with matching
                leading dimensions.

        Returns:
            FP32 tensor of shape [...] containing uncalibrated confidence logits.
        """
        if hidden_states.shape[:-1] != markov_embeddings.shape[:-1]:
            raise ValueError("DSpark hidden states and Markov embeddings must have matching leading dimensions")
        return self.proj(torch.cat((hidden_states.float(), markov_embeddings.float()), dim=-1)).squeeze(-1)


class _DeepseekV41DSparkBlock(nn.Module):
    """One released DSpark stage with MLA, MoE, mHC and stage-owned heads."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        stage_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
    ) -> None:
        super().__init__()
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        layer_idx = config.num_hidden_layers + stage_idx
        self.attn = _DeepseekV41DSparkAttention(config, layer_idx, backend)
        self.ffn = MoE(moe_config, backend)
        self.ffn.gate = DeepseekV4VisionGate(
            DeepseekV4Config(vocab_size=config.vocab_size),
            moe_config,
            gate_precision=torch.float32,
            hash_routing=False,
        )
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.attn.wq_a.weight.device)
            if backend.rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.attn_norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.ffn_norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        sinkhorn_backend = "tilelang" if backend.attn == "tilelang" else "torch"
        self.attn_hc = DeepseekV41HyperConnection(config, sinkhorn_backend=sinkhorn_backend)
        self.ffn_hc = DeepseekV41HyperConnection(config, sinkhorn_backend=sinkhorn_backend)
        if stage_idx == 0:
            self.main_proj = nn.Linear(
                config.hidden_size * len(config.dspark_target_layer_ids), config.hidden_size, bias=False, dtype=dtype
            )
            self.main_norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        if stage_idx == config.num_nextn_predict_layers - 1:
            self.norm = norm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
            self.markov_head = _DeepseekV41DSparkMarkovHead(config.vocab_size, config.dspark_markov_rank, dtype)
            self.confidence_head = _DeepseekV41DSparkConfidenceHead(config.hidden_size, config.dspark_markov_rank)

    @property
    def mlp(self) -> MoE:
        """Expose the shared parallelizer's MoE interface without duplicate registration."""
        return self.ffn

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        target_hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one native DSpark stage.

        Args:
            hidden_states: Tensor of shape [batch, draft_sequence, streams, hidden].
            pre_mix: FP32 tensor of shape [batch, draft_sequence, streams].
            target_hidden_states: Tensor of shape [batch, context_sequence, hidden].
            position_ids: Integer tensor of shape [batch, context_sequence + draft_sequence].
            attention_mask: Additive tensor of shape [batch, 1, draft_sequence,
                context_sequence + draft_sequence].

        Returns:
            Updated streams of shape [batch, draft_sequence, streams, hidden] and
            the next FP32 pre-mix of shape [batch, draft_sequence, streams].
        """
        residual = hidden_states
        attn_mix = self.attn_hc(hidden_states)
        collapsed = self.attn_hc.collapse(hidden_states, pre_mix)
        attended = self.attn(
            self.attn_norm(collapsed),
            target_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = self.attn_hc.expand(attended, residual, attn_mix)
        residual = hidden_states
        ffn_mix = self.ffn_hc(hidden_states)
        collapsed = self.ffn_hc.collapse(hidden_states, attn_mix.pre)
        self.ffn.gate.set_routing_context(None, None)
        output = self.ffn(self.ffn_norm(collapsed), None)
        return self.ffn_hc.expand(output, residual, ffn_mix), ffn_mix.pre


class DeepseekV41DSparkBackbone(nn.Module):
    """Three-stage native drafter operating on prepared target and noise tensors."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        backend: BackendConfig,
        moe_config: MoEConfig | None = None,
    ) -> None:
        super().__init__()
        if config.num_nextn_predict_layers <= 0:
            raise ValueError("DeepSeek V4.1 DSpark requires at least one draft layer")
        required_schedule = config.num_hidden_layers + config.num_nextn_predict_layers
        if len(config.compress_ratios) < required_schedule:
            raise ValueError(
                f"DeepSeek V4.1 DSpark requires compress_ratios for {required_schedule} backbone and draft layers"
            )
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.config = config
        self.moe_config = moe_config or MoEConfig(
            dim=config.hidden_size,
            inter_dim=config.moe_intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.dspark_n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.dspark_num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sqrtsoftplus",
            route_scale=config.routed_scaling_factor,
            norm_topk_prob=config.norm_topk_prob,
            router_weights_fp32=True,
            force_e_score_correction_bias=True,
            swiglu_limit=config.swiglu_limit,
            dtype=dtype,
        )
        self.mtp = nn.ModuleList(
            _DeepseekV41DSparkBlock(config, stage_idx, backend, self.moe_config)
            for stage_idx in range(config.num_nextn_predict_layers)
        )

    def build_position_ids(self, anchor_positions: torch.Tensor, context_sequence: int) -> torch.Tensor:
        """Build official target-context and draft-query positions.

        Args:
            anchor_positions: Integer tensor [batch, num_anchors] containing the
                target token that seeds each draft block.
            context_sequence: Number of target-context tokens.

        Returns:
            Integer tensor [batch, context_sequence + num_anchors * block_size].
            Draft positions are ``anchor + 1`` through ``anchor + block_size``;
            the anchor token occupies the first draft input slot but predicts the
            following position in the released implementation.
        """
        if anchor_positions.ndim != 2:
            raise ValueError("DSpark anchor_positions must have shape [batch, num_anchors]")
        batch, num_anchors = anchor_positions.shape
        context = torch.arange(context_sequence, device=anchor_positions.device).view(1, -1).expand(batch, -1)
        offsets = torch.arange(
            1,
            self.config.dspark_block_size + 1,
            device=anchor_positions.device,
        ).view(1, 1, -1)
        draft = (anchor_positions.unsqueeze(-1) + offsets).reshape(batch, num_anchors * self.config.dspark_block_size)
        return torch.cat((context, draft), dim=1)

    def build_attention_mask(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        context_sequence: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the released SWA-128 multi-anchor training mask.

        Every query sees the target window ending at and including its anchor,
        plus every parallel input in its own draft block. It cannot see another
        anchor's block. Invalid padding blocks retain their own in-block keys so
        no attention row is fully masked; their losses are discarded later.

        Args:
            anchor_positions: Integer tensor [batch, num_anchors].
            block_keep_mask: Boolean tensor [batch, num_anchors].
            context_sequence: Number of target-context tokens.
            dtype: Floating dtype of the returned additive mask.

        Returns:
            Additive tensor [batch, 1, num_anchors * block_size,
            context_sequence + num_anchors * block_size].
        """
        if anchor_positions.ndim != 2 or block_keep_mask.shape != anchor_positions.shape:
            raise ValueError("DSpark anchors and block_keep_mask must have matching [batch, num_anchors] shapes")
        batch, num_anchors = anchor_positions.shape
        block_size = self.config.dspark_block_size
        draft_sequence = num_anchors * block_size
        kv_sequence = context_sequence + draft_sequence
        device = anchor_positions.device

        query_index = torch.arange(draft_sequence, device=device).view(1, 1, -1, 1)
        key_index = torch.arange(kv_sequence, device=device).view(1, 1, 1, -1)
        query_block = query_index // block_size
        anchor = anchor_positions.view(batch, 1, num_anchors, 1).repeat_interleave(block_size, dim=2)
        is_context = key_index < context_sequence
        context_visible = is_context & (key_index <= anchor) & (key_index > anchor - self.config.sliding_window)

        is_draft = key_index >= context_sequence
        key_block = (key_index - context_sequence) // block_size
        own_block_visible = is_draft & (query_block == key_block)
        keep = block_keep_mask.view(batch, 1, num_anchors, 1).repeat_interleave(block_size, dim=2)
        visible = (context_visible & keep) | own_block_visible
        return torch.where(
            visible,
            torch.tensor(0.0, device=device, dtype=dtype),
            torch.tensor(float("-inf"), device=device, dtype=dtype),
        )

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize every draft parameter for checkpoint-free training.

        Args:
            buffer_device: Device used by grouped expert initialization. Defaults
                to the first attention projection's device.
        """
        if buffer_device is None:
            buffer_device = self.mtp[0].attn.wq_a.weight.device
        std = self.config.initializer_range
        for layer in self.mtp:
            layer.ffn.init_weights(buffer_device, init_std=std)
            layer.attn_hc.reset_parameters(std)
            layer.ffn_hc.reset_parameters(std)
            nn.init.ones_(layer.attn_norm.weight)
            nn.init.ones_(layer.ffn_norm.weight)
            layer.ffn.gate.bias_vl.zero_()
            layer.attn.reset_parameters(std)
        first = self.mtp[0]
        nn.init.normal_(first.main_proj.weight, std=std)
        nn.init.ones_(first.main_norm.weight)
        last = self.mtp[-1]
        nn.init.ones_(last.norm.weight)
        nn.init.normal_(last.markov_head.embed.weight, std=std)
        nn.init.normal_(last.markov_head.head.weight, std=std)
        nn.init.normal_(last.confidence_head.proj.weight, std=std)

    def forward(
        self,
        noise_embeddings: torch.Tensor,
        target_hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the cache-free draft backbone for sampled anchors.

        Args:
            noise_embeddings: Tensor of shape [batch, draft_sequence, hidden],
                containing an anchor embedding followed by noise embeddings in
                each fixed-width block.
            target_hidden_states: Concatenated target features of shape [batch,
                context_sequence, target_layers * hidden].
            position_ids: Integer tensor of shape [batch, context_sequence + draft_sequence].
            attention_mask: Additive tensor of shape [batch, 1, draft_sequence,
                context_sequence + draft_sequence].

        Returns:
            Pre-normalization and normalized draft states. Both tensors have
            shape [batch, draft_sequence, hidden].
        """
        first = self.mtp[0]
        target_hidden_states = first.main_norm(first.main_proj(target_hidden_states))
        hidden_states = noise_embeddings.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1)
        pre_mix = torch.zeros(
            *noise_embeddings.shape[:2],
            self.config.hc_mult,
            device=noise_embeddings.device,
            dtype=torch.float32,
        )
        pre_mix[..., 0] = 1
        for layer in self.mtp:
            hidden_states, pre_mix = layer(
                hidden_states,
                pre_mix,
                target_hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        hidden_states = DeepseekV41HyperConnection.collapse(hidden_states, pre_mix)
        return DeepseekV41DSparkBackboneOutput(
            hidden_states=hidden_states,
            normalized_hidden_states=self.mtp[-1].norm(hidden_states),
        )


__all__ = ["DeepseekV41DSparkBackbone", "DeepseekV41DSparkBackboneOutput"]

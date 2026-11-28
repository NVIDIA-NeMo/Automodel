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

# Code provenance: the attentive correction's gated prefix read (``_prefix_read``) and its start from
# the last draft layer (``DeepseekV41DFly2Model.initialize_attentive_correction``) are adapted from
# Tencent/AngelSpec, https://github.com/Tencent/AngelSpec at revision e02ea36e65d44e6e165c4ba73c139d500e12f2d9,
# file angelspec/models/draft/dsv41_attentive_correction.py, Copyright (C) 2026 Tencent, licensed
# under the Apache License, Version 2.0. Modified by NVIDIA: rewritten for this module's layout.

"""DFly2 draft for DeepSeek-V4.1-Flash (arXiv:2611.34776).

Each draft layer is a V4.1 block (latent MQA with an attention sink, routed MoE and
single-pass mHC) paired with one of the target's ratio-1 sparse-attention layers. It owns
no context projection: its queries read the keys that target layer attends over, its own
sliding-window latents followed by the shared 1:1 global latents, at the slots that layer
attends from the row before the anchor (the last 128 tokens plus the indexer's top-512
picks), together with the draft keys of its own block. The released checkpoint pairs
layers 28, 32 and 36.

Around the backbone, the target's final hidden state at the row before the anchor is fused
into the anchor slot (MTP fusion), and a teacher-forced hidden correction, a SwiGLU plus
an attentive correction over the last layer's context and the block's own prefix, adjusts
every slot before the frozen LM head. Slot 0 holds the anchor and is not supervised, so a
block of ``block_size`` slots drafts ``block_size - 1`` tokens.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4FP32Parameter, DeepseekV4GroupedLinear
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4VisionGate
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention, _apply_rope, _RotaryEmbedding
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection, DeepseekV41RMSNorm
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.speculative.dfly2.common import build_block_key_indices, context_key_indices
from nemo_automodel.components.speculative.dspark.common import (
    DSparkForwardOutput,
    build_eval_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from nemo_automodel.shared.utils import dtype_from_str

# Width of the per-slot bias tables of the attentive correction's prefix read, so the longest
# supported block. Fixed by the shapes of the released draft's tensors.
MAX_BLOCK_SLOTS = 16
_SPARSE_BACKENDS = ("tilelang", "cudnn")


@dataclass(frozen=True)
class DeepseekV41DFly2Config:
    """Validated declarative configuration of the V4.1 DFly2 draft.

    Attributes:
        text_config: Target text configuration; the draft reuses its attention, mHC and
            MoE shapes and its compressed-layer rotary embedding.
        target_layer_ids: Target layers paired with the draft layers, in draft-layer
            order. Each must be a ratio-1 sparse-attention layer, whose context rows are
            raw tokens.
        block_size: Draft slots per block, the anchor slot included.
        num_anchors: Blocks sampled per sequence.
        mask_token_id: Token filling the non-anchor slots.
        enable_confidence_head: Build the acceptance head, absent from the released draft.
        confidence_head_stop_gradient: Feed the acceptance head detached states, so its
            loss trains only the head.
        n_routed_experts: Routed experts per draft layer.
        num_experts_per_tok: Experts each token is routed to.
    """

    text_config: DeepseekV41TextConfig
    target_layer_ids: tuple[int, ...]
    block_size: int
    num_anchors: int
    mask_token_id: int
    enable_confidence_head: bool = False
    confidence_head_stop_gradient: bool = True
    n_routed_experts: int = 128
    num_experts_per_tok: int = 3

    def build(self, backend: BackendConfig | None = None) -> DeepseekV41DFly2Model:
        """Validate the target pairing and build a freshly initialized draft.

        Args:
            backend: Kernel selection for the draft; defaults to :func:`default_dfly2_backend`.

        Returns:
            The draft, ready for device placement.

        Raises:
            ValueError: A paired layer is not a ratio-1 target layer, or a size is invalid.
        """
        text = self.text_config
        if not self.target_layer_ids:
            raise ValueError("DeepSeek V4.1 DFly2 needs at least one paired target layer")
        for layer_id in self.target_layer_ids:
            if not 0 <= layer_id < text.num_hidden_layers or text.compress_ratios[layer_id] != 1:
                raise ValueError(
                    f"DeepSeek V4.1 DFly2 pairs ratio-1 target layers, whose context rows are raw tokens; "
                    f"layer {layer_id} has compress_ratio "
                    f"{text.compress_ratios[layer_id] if 0 <= layer_id < len(text.compress_ratios) else None}"
                )
        if len(text.compress_ratios) < text.num_hidden_layers + len(self.target_layer_ids):
            raise ValueError("DeepSeek V4.1 DFly2 needs compress_ratios entries for the draft layers")
        if not 2 <= self.block_size <= MAX_BLOCK_SLOTS:
            raise ValueError(f"DeepSeek V4.1 DFly2 block_size must be in [2, {MAX_BLOCK_SLOTS}], got {self.block_size}")
        if self.num_anchors <= 0:
            raise ValueError(f"DeepSeek V4.1 DFly2 requires num_anchors > 0, got {self.num_anchors}")
        if not 0 <= self.mask_token_id < text.vocab_size:
            raise ValueError(f"mask_token_id={self.mask_token_id} is outside the vocabulary [0, {text.vocab_size})")
        return DeepseekV41DFly2Model(self, backend)


def default_dfly2_backend(attn: str = "sdpa") -> BackendConfig:
    """Return the draft's kernel selection.

    Args:
        attn: ``"sdpa"`` or ``"eager"`` scatter the key lists into a dense mask, which suits
            short rows; ``"tilelang"`` or ``"cudnn"`` run the shared sparse kernel, which long
            rows need.

    Returns:
        Backend with torch linears and FP32 RMSNorm. Like BackendConfig's own default, the
        experts use grouped GEMMs on CUDA.
    """
    return BackendConfig(
        attn=attn,
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch_mm" if torch.cuda.is_available() else "torch",
        dispatcher="torch",
    )


def _listed_latent_attention(
    query: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sinks: torch.Tensor,
    scale: float,
    *,
    backend: str,
) -> torch.Tensor:
    """Attend every query over its listed keys, one shared latent serving as key and value.

    Args:
        query: Tensor of shape [batch, queries, heads, head_dim].
        kv: Tensor of shape [batch, keys, head_dim].
        indices: Integer tensor of shape [batch, queries, slots] of ``kv`` rows, ``-1``
            unused. The listed keys of a row must be distinct.
        sinks: FP32 tensor of shape [heads], learned logits added to every softmax
            denominator only.
        scale: Query/key score multiplier.
        backend: ``"tilelang"`` or ``"cudnn"`` run the shared sparse kernel; ``"sdpa"`` and
            ``"eager"`` scatter the lists into a dense mask.

    Returns:
        Tensor of shape [batch, queries, heads, head_dim] in the dtype of ``query``; zero
        for a query without keys.
    """
    if backend in _SPARSE_BACKENDS:
        return dsv4_sparse_attention(query, kv, sinks, indices, scale, backend=backend)
    batch, queries, heads, head_dim = query.shape
    keys = kv.shape[1]
    # A separate sentinel column absorbs the -1 slots, as in the target's dense fallback.
    allowed = torch.zeros(batch, queries, keys + 1, dtype=torch.bool, device=query.device)
    allowed = allowed.scatter(-1, torch.where(indices >= 0, indices, keys).long(), True)[..., :keys]
    # A zero-valued extra key contributes exp(sink) only to the denominator; a query
    # without keys then puts all its weight on it and returns zero.
    kv = torch.cat((kv, kv.new_zeros(batch, 1, head_dim)), dim=1)
    bias = torch.zeros(batch, 1, queries, keys, device=query.device, dtype=torch.float32)
    bias = bias.masked_fill(~allowed.unsqueeze(1), -torch.inf).expand(-1, heads, -1, -1)
    bias = torch.cat((bias, sinks.float().view(1, heads, 1, 1).expand(batch, -1, queries, -1)), dim=-1)
    if backend == "sdpa":
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), kv.unsqueeze(1), kv.unsqueeze(1), attn_mask=bias, scale=scale
        ).transpose(1, 2)
    with torch.autocast(device_type=query.device.type, enabled=False):
        logits = torch.einsum("bshd,btd->bhst", query.float(), kv.float()) * scale
        probabilities = (logits + bias).softmax(dim=-1)
        return torch.einsum("bhst,btd->bshd", probabilities, kv.float()).to(query.dtype)


class _DeepseekV41DFly2Attention(DeepseekV41Attention):
    """Latent MQA over a paired target layer's keys and the draft's own block."""

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__(config, layer_idx, backend)
        if self.compress_ratio != 0 or self.compressor is not None or self.indexer is not None:
            raise ValueError("DeepSeek V4.1 DFly2 draft layers must own no compressor or indexer")
        # Queries rotate like the keys they read: those of the paired ratio-1 target layers.
        self.rotary_emb = _RotaryEmbedding(config, compressed=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_keys: torch.Tensor,
        key_indices: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Attend draft queries to their listed target keys and their own block.

        Args:
            hidden_states: Normalized draft tensor of shape [batch, draft_sequence, hidden].
            context_keys: Paired target layer's keys of shape [batch, context_keys, head_dim].
            key_indices: Integer tensor of shape [batch, draft_sequence, slots] of rows of
                ``cat(context_keys, draft_keys)``, ``-1`` unused.
            position_ids: Integer tensor of shape [batch, draft_sequence].

        Returns:
            Tensor of shape [batch, draft_sequence, hidden]. Inputs are not mutated.
        """
        batch, draft_sequence, _ = hidden_states.shape
        angles = self.rotary_emb(position_ids)
        query = self.wq_b(self.q_norm(self.wq_a(hidden_states))).unflatten(-1, (self.num_heads, self.head_dim))
        query = _apply_rope(query, angles)
        draft_kv = _apply_rope(self.kv_norm(self.wkv(hidden_states)), angles)
        kv = torch.cat((context_keys.to(draft_kv.dtype), draft_kv), dim=1)
        attended = _listed_latent_attention(
            query, kv, key_indices, self.sinks_param(query), self.head_dim**-0.5, backend=self.backend.attn
        )
        valid_tokens = torch.ones(batch, draft_sequence, dtype=torch.bool, device=hidden_states.device)
        return self._project_output(attended, angles, valid_tokens)


class _DeepseekV41DFly2Block(nn.Module):
    """One draft layer: latent MQA, routed MoE and single-pass mHC, as in the target's blocks."""

    def __init__(
        self, config: DeepseekV41TextConfig, layer_idx: int, backend: BackendConfig, moe_config: MoEConfig
    ) -> None:
        super().__init__()
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.attn = _DeepseekV41DFly2Attention(config, layer_idx, backend)
        self.ffn = MoE(moe_config, backend)
        # As in the backbone block, BackendConfig.fake_balanced_gate keeps the
        # FakeBalancedGate that MoE built.
        if not backend.fake_balanced_gate:
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

    @property
    def mlp(self) -> MoE:
        """Expose the shared parallelizer's MoE interface without duplicate registration."""
        return self.ffn

    def forward(
        self,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
        context_keys: torch.Tensor,
        key_indices: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply one draft layer.

        The attention input collapses the streams with the previous layer's FFN pre-mix
        and the FFN input with this layer's attention pre-mix, as the target's blocks do.

        Args:
            hidden_states: Tensor of shape [batch, draft_sequence, streams, hidden].
            pre_mix: FP32 tensor of shape [batch, draft_sequence, streams].
            context_keys: Tensor of shape [batch, context_keys, head_dim].
            key_indices: Integer tensor of shape [batch, draft_sequence, slots].
            position_ids: Integer tensor of shape [batch, draft_sequence].

        Returns:
            Updated streams of shape [batch, draft_sequence, streams, hidden] and the FP32
            pre-mix of shape [batch, draft_sequence, streams] for the next layer.
        """
        residual = hidden_states
        attn_mix = self.attn_hc(hidden_states)
        collapsed = self.attn_hc.collapse(hidden_states, pre_mix)
        attended = self.attn(self.attn_norm(collapsed), context_keys, key_indices, position_ids=position_ids)
        hidden_states = self.attn_hc.expand(attended, residual, attn_mix)
        residual = hidden_states
        ffn_mix = self.ffn_hc(hidden_states)
        collapsed = self.ffn_hc.collapse(hidden_states, attn_mix.pre)
        gate = self.ffn.gate
        if isinstance(gate, DeepseekV4VisionGate):
            gate.set_routing_context(None, None)
        output = self.ffn(self.ffn_norm(collapsed), None)
        return self.ffn_hc.expand(output, residual, ffn_mix), ffn_mix.pre


class _DeepseekV41DFly2MTPFusion(nn.Module):
    """Fuse the target's last hidden state into the anchor slot, as the target's MTP layer opens."""

    def __init__(self, config: DeepseekV41TextConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.enorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.hnorm = DeepseekV41RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        # Per-channel gate, zero at initialization so a fresh draft starts without fusion.
        self.gate = nn.Parameter(torch.zeros(config.hidden_size, dtype=dtype))

    def forward(self, anchor_embeddings: torch.Tensor, target_hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the residual added to every anchor slot.

        Args:
            anchor_embeddings: Tensor of shape [batch, num_anchors, hidden], the anchor
                tokens' embeddings.
            target_hidden_states: Tensor of shape [batch, num_anchors, hidden], the target's
                final normalized hidden state at the row before each anchor.

        Returns:
            Tensor of shape [batch, num_anchors, hidden].
        """
        fused = self.eh_proj(torch.cat((self.enorm(anchor_embeddings), self.hnorm(target_hidden_states)), dim=-1))
        return self.gate * fused


class _DeepseekV41DFly2AttentiveCorrection(nn.Module):
    """Per-slot latent MQA over the last draft layer's context plus a gated read of the block prefix.

    The query comes from the slot's normalized hidden state and the normalized embedding of
    the token before it. Slot ``k`` reads the context keys the last draft layer read, then,
    in a softmax of its own with a sink and a learned bias by distance, the keys of slots
    ``0..k`` of its block, added through a per-head ``tanh`` gate. ``wo_b`` and the gate
    start at zero, so a fresh correction is an exact no-op.
    """

    def __init__(self, config: DeepseekV41TextConfig, backend: BackendConfig, dtype: torch.dtype) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_groups = config.o_groups
        self.backend = backend
        self.q_in = nn.Linear(2 * hidden, config.q_lora_rank, bias=False, dtype=dtype)
        self.q_norm = DeepseekV41RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.kv_in = nn.Linear(2 * hidden, self.head_dim, bias=False, dtype=dtype)
        self.kv_norm = DeepseekV41RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // self.num_groups, self.num_groups * config.o_lora_rank, self.num_groups
        ).to(dtype=dtype)
        self.wo_b = nn.Linear(self.num_groups * config.o_lora_rank, hidden, bias=False, dtype=dtype)
        self.block_gate = nn.Linear(2 * hidden, self.num_heads, bias=False, dtype=dtype)
        self.attn_sink = DeepseekV4FP32Parameter(torch.zeros(self.num_heads))
        self.block_sink = DeepseekV4FP32Parameter(torch.zeros(self.num_heads))
        self.block_rel_bias = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, MAX_BLOCK_SLOTS))
        self.block_gate_bias = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, MAX_BLOCK_SLOTS))
        self.rotary_emb = _RotaryEmbedding(config, compressed=True)

    def _prefix_read(
        self, query: torch.Tensor, keys: torch.Tensor, inputs: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Gated read of each slot's own block prefix (slots ``0..k`` for slot ``k``).

        The prefix has a softmax of its own: a per-head learned bias indexed by how many slots
        back the key is, plus the ``block_sink`` logit in the denominator as an extra column.
        Each head scales its read by ``tanh`` of a gate computed from the slot's inputs.

        Args:
            query: Tensor of shape [batch, slots, heads, head_dim], whole blocks of ``block_size``.
            keys: Tensor of shape [batch, slots, head_dim], each slot's key, also its value.
            inputs: Tensor of shape [batch, slots, 2 * hidden], the query inputs.
            block_size: Slots per block.

        Returns:
            FP32 tensor of shape [batch, slots, heads, head_dim].
        """
        batch, slots, heads, head_dim = query.shape
        blocks = slots // block_size
        per_block_query = query.float().unflatten(1, (blocks, block_size))
        per_block_keys = keys.float().unflatten(1, (blocks, block_size))
        scores = torch.einsum("bnshd,bntd->bnhst", per_block_query, per_block_keys) * head_dim**-0.5
        # distance[s, t] = s - t: how far back key slot t lies from query slot s.
        positions = torch.arange(block_size, device=query.device)
        distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(min=0)
        scores = scores + self.block_rel_bias(query).float()[:, distance]
        prefix = torch.ones(block_size, block_size, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~prefix, float("-inf"))
        sink = self.block_sink(query).float().view(1, 1, heads, 1, 1).expand(*scores.shape[:-1], 1)
        weights = torch.cat((scores, sink), dim=-1).softmax(dim=-1)[..., :-1]
        read = torch.einsum("bnhst,bntd->bnshd", weights, per_block_keys).flatten(1, 2)
        slot_bias = self.block_gate_bias(query).float()[:, positions.repeat(blocks)]
        gate = torch.tanh(self.block_gate(inputs).float() + slot_bias.T)
        return read * gate.unsqueeze(-1)

    def forward(
        self,
        inputs: torch.Tensor,
        context_keys: torch.Tensor,
        context_indices: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Compute the correction of every slot.

        Args:
            inputs: Tensor of shape [batch, slots, 2 * hidden], the concatenated normalized
                hidden state and previous-token embedding of each slot; whole blocks.
            context_keys: Last draft layer's context keys of shape [batch, context_keys, head_dim].
            context_indices: Integer tensor of shape [batch, slots, key_slots] of
                ``context_keys`` rows, ``-1`` unused.
            position_ids: Integer tensor of shape [batch, slots].
            block_size: Slots per block.

        Returns:
            Tensor of shape [batch, slots, hidden].
        """
        angles = self.rotary_emb(position_ids)
        query = self.wq_b(self.q_norm(self.q_in(inputs))).unflatten(-1, (self.num_heads, self.head_dim))
        query = _apply_rope(query, angles)
        keys = _apply_rope(self.kv_norm(self.kv_in(inputs)), angles)
        context = _listed_latent_attention(
            query,
            context_keys.to(query.dtype),
            context_indices,
            self.attn_sink(query),
            self.head_dim**-0.5,
            backend=self.backend.attn,
        )
        read = (context.float() + self._prefix_read(query, keys, inputs, block_size)).to(query.dtype)
        read = _apply_rope(read, angles, inverse=True)
        return self.wo_b(self.wo_a(read.reshape(*read.shape[:2], self.num_groups, -1)).flatten(2))


class _DeepseekV41DFly2HiddenCorrection(nn.Module):
    """Teacher-forced residual from each slot's hidden state and the token before it."""

    def __init__(
        self, config: DeepseekV41TextConfig, intermediate_size: int, backend: BackendConfig, dtype: torch.dtype
    ) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.hidden_norm = DeepseekV41RMSNorm(hidden, eps=config.rms_norm_eps, dtype=dtype)
        self.embed_norm = DeepseekV41RMSNorm(hidden, eps=config.rms_norm_eps, dtype=dtype)
        self.gate_proj = nn.Linear(2 * hidden, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(2 * hidden, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden, bias=False, dtype=dtype)
        self.attention = _DeepseekV41DFly2AttentiveCorrection(config, backend, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_embeddings: torch.Tensor,
        context_keys: torch.Tensor,
        context_indices: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Return the corrected hidden states.

        Args:
            hidden_states: Final-normalized draft tensor of shape [batch, slots, hidden].
            previous_embeddings: Tensor of shape [batch, slots, hidden], the embedding of the
                token before each slot (the anchor for slots 0 and 1).
            context_keys: Tensor of shape [batch, context_keys, head_dim].
            context_indices: Integer tensor of shape [batch, slots, key_slots].
            position_ids: Integer tensor of shape [batch, slots].
            block_size: Slots per block.

        Returns:
            Tensor of shape [batch, slots, hidden].
        """
        inputs = torch.cat((self.hidden_norm(hidden_states), self.embed_norm(previous_embeddings)), dim=-1)
        delta = self.down_proj(F.silu(self.gate_proj(inputs)) * self.up_proj(inputs))
        delta = delta + self.attention(
            inputs, context_keys, context_indices, position_ids=position_ids, block_size=block_size
        )
        return hidden_states + delta


class _DeepseekV41DFly2ConfidenceHead(nn.Module):
    """FP32 acceptance head on the drafted slots' final-normalized states (absent from the released draft)."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        # Named so the strict FP32 entry also matches inside the head's own FSDP unit.
        self.confidence_proj = nn.Linear(hidden_size, 1, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict an acceptance logit per slot.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            FP32 tensor of shape [...].
        """
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return self.confidence_proj(hidden_states.float()).squeeze(-1)


class DeepseekV41DFly2Model(nn.Module):
    """Model-owned DFly2 draft emitting the shared DSpark loss inputs over its drafted slots."""

    _no_split_modules = ["_DeepseekV41DFly2Block"]
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "attn.sinks_param",
        "bias_vl",
        "attention.attn_sink",
        "attention.block_sink",
        "attention.block_rel_bias",
        "attention.block_gate_bias",
        "confidence_proj",
    ]
    # Same fp32 holders as DeepseekV41ForCausalLM: the draft's attention sinks share one FSDP
    # unit across the draft blocks (parallelizer_utils.fully_shard_across_blocks).
    _fsdp_shard_across_blocks = ["attn.sinks_param"]

    def __init__(self, config: DeepseekV41DFly2Config, backend: BackendConfig | None = None) -> None:
        backend = backend or default_dfly2_backend()
        super().__init__()
        text = config.text_config
        dtype = dtype_from_str(text.dtype, torch.bfloat16)
        self.dfly2_config = config
        # The exported config.json: the target's text config with the DFly2 settings in a
        # ``dfly_config`` block (block size, paired layers, mask token, correction layout).
        self.config = copy.deepcopy(text)
        self.config.architectures = ["DeepseekV41DFly2Model"]
        self.config.dfly_config = {
            "block_size": config.block_size,
            "causal_block": False,
            "target_kv_layers": list(config.target_layer_ids),
            "target_index_layers": list(config.target_layer_ids),
            "index_topk": text.index_topk,
            "mask_token_id": config.mask_token_id,
            "rope": {
                "theta": text.compress_rope_theta,
                "original_seq_len": (text.rope_scaling or {}).get("original_max_position_embeddings", 0),
                "factor": (text.rope_scaling or {}).get("factor", 1.0),
                "beta_fast": (text.rope_scaling or {}).get("beta_fast", 32),
                "beta_slow": (text.rope_scaling or {}).get("beta_slow", 1),
            },
            "mtp_fusion": "slot0",
            "hidden_correction": {"intermediate_size": text.hidden_size},
            "attentive_correction": {
                "source_layer": -1,
                "block_keys": True,
                "block_mode": "gated",
                "block_kv_in": "hidden_embed",
                "max_block_slots": MAX_BLOCK_SLOTS,
            },
            "dense_dtype": str(dtype).removeprefix("torch."),
        }
        self.moe_config = MoEConfig(
            dim=text.hidden_size,
            inter_dim=text.moe_intermediate_size,
            moe_inter_dim=text.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=text.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="sqrtsoftplus",
            route_scale=text.routed_scaling_factor,
            norm_topk_prob=text.norm_topk_prob,
            router_weights_fp32=True,
            force_e_score_correction_bias=True,
            swiglu_limit=text.swiglu_limit,
            dtype=dtype,
        )
        # Draft layers take the schedule slots after the target's, which carry no compressor or indexer.
        self.layers = nn.ModuleList(
            _DeepseekV41DFly2Block(text, text.num_hidden_layers + index, backend, self.moe_config)
            for index in range(len(config.target_layer_ids))
        )
        self.final_norm = DeepseekV41RMSNorm(text.hidden_size, eps=text.rms_norm_eps, dtype=dtype)
        self.mtp_fusion = _DeepseekV41DFly2MTPFusion(text, dtype)
        self.hidden_correction = _DeepseekV41DFly2HiddenCorrection(text, text.hidden_size, backend, dtype)
        self.confidence_head = (
            _DeepseekV41DFly2ConfidenceHead(text.hidden_size) if config.enable_confidence_head else None
        )
        self.embed_tokens = nn.Embedding(text.vocab_size, text.hidden_size, padding_idx=text.pad_token_id, dtype=dtype)
        self.lm_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False, dtype=dtype)
        from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import (
            DeepseekV41DFly2StateDictAdapter,
        )

        self.state_dict_adapter = DeepseekV41DFly2StateDictAdapter(text, self.moe_config, backend, dtype=dtype)
        # The released keys of the whole draft, recorded before sharding: under FSDP each rank
        # exports only its own experts, and the consolidated index must list all of them.
        self._pre_shard_hf_state_dict_keys = self.state_dict_adapter.get_hf_state_dict_keys(self.state_dict())
        self.initialize_weights(self.embed_tokens.weight.device)

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize every draft parameter for checkpoint-free training.

        The fusion gate, the hidden correction's down projection and the attentive
        correction's output start at zero, so a fresh draft is its backbone alone.

        Args:
            buffer_device: Device used by grouped expert initialization. Defaults to the
                first attention projection's device.
        """
        if buffer_device is None:
            buffer_device = self.layers[0].attn.wq_a.weight.device
        std = self.dfly2_config.text_config.initializer_range
        for layer in self.layers:
            layer.ffn.init_weights(buffer_device, init_std=std)
            layer.attn_hc.reset_parameters(std)
            layer.ffn_hc.reset_parameters(std)
            nn.init.ones_(layer.attn_norm.weight)
            nn.init.ones_(layer.ffn_norm.weight)
            if isinstance(layer.ffn.gate, DeepseekV4VisionGate):
                layer.ffn.gate.bias_vl.zero_()
            layer.attn.reset_parameters(std)
        nn.init.ones_(self.final_norm.weight)
        fusion = self.mtp_fusion
        nn.init.ones_(fusion.enorm.weight)
        nn.init.ones_(fusion.hnorm.weight)
        nn.init.normal_(fusion.eh_proj.weight, std=std)
        fusion.gate.zero_()
        correction = self.hidden_correction
        nn.init.ones_(correction.hidden_norm.weight)
        nn.init.ones_(correction.embed_norm.weight)
        nn.init.normal_(correction.gate_proj.weight, std=std)
        nn.init.normal_(correction.up_proj.weight, std=std)
        correction.down_proj.weight.zero_()
        self.initialize_attentive_correction()
        if self.confidence_head is not None:
            nn.init.normal_(self.confidence_head.confidence_proj.weight, std=std)
            nn.init.zeros_(self.confidence_head.confidence_proj.bias)

    @torch.no_grad()
    def initialize_attentive_correction(self) -> None:
        """Start the attentive correction from the last draft layer's attention, with its output closed.

        Its query and key input projections read ``[hidden; previous embedding]``: the hidden
        columns take the layer's ``wq_a`` / ``wkv`` and the embedding columns start at zero. The
        norms, ``wq_b``, ``wo_a`` and both sinks take the layer's values; ``wo_b``, the gate and
        the per-slot biases start at zero, so the correction contributes nothing until trained.
        """
        source = self.layers[-1].attn
        correction = self.hidden_correction.attention
        correction.q_norm.load_state_dict(source.q_norm.state_dict())
        correction.wq_b.load_state_dict(source.wq_b.state_dict())
        correction.kv_norm.load_state_dict(source.kv_norm.state_dict())
        correction.wo_a.load_state_dict(source.wo_a.state_dict())
        for widened, projection in ((correction.q_in, source.wq_a), (correction.kv_in, source.wkv)):
            hidden_columns = projection.weight
            widened.weight.copy_(torch.cat((hidden_columns, torch.zeros_like(hidden_columns)), dim=1))
        for sink in (correction.attn_sink, correction.block_sink):
            sink.weight.copy_(source.sinks_param.weight)
        for closed in (correction.wo_b, correction.block_gate, correction.block_rel_bias, correction.block_gate_bias):
            closed.weight.zero_()

    @torch.no_grad()
    def initialize_from_target(self, target: nn.Module) -> None:
        """Start every draft layer's attention, norms, mHC mixes and shared expert from its paired target layer.

        Collective when the target is sharded: every rank must call it, in the same order.
        The routed experts and the router keep their fresh initialization; warm-start them
        from an existing draft through ``draft_weights_path`` instead.

        Args:
            target: The frozen target ``DeepseekV41ForCausalLM``; its parameters may be
                DTensors, which are gathered one at a time.
        """
        target_layers = target.model.layers
        for layer, layer_id in zip(self.layers, self.dfly2_config.target_layer_ids):
            source = target_layers[str(layer_id)]
            pairs = [(layer.attn_norm, source.attn_norm), (layer.ffn_norm, source.ffn_norm)]
            pairs += [(getattr(layer.attn, name), getattr(source.attn, name)) for name in _ATTENTION_INIT_MODULES]
            pairs += [(layer.attn_hc, source.attn_hc), (layer.ffn_hc, source.ffn_hc)]
            pairs.append((layer.ffn.shared_experts, source.ffn.shared_experts))
            for draft_module, target_module in pairs:
                for name, parameter in draft_module.named_parameters():
                    value = target_module.get_parameter(name)
                    value = value.full_tensor() if isinstance(value, DTensor) else value
                    parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
        self.initialize_attentive_correction()

    def initialize_embeddings_and_head(
        self, *, embed_tokens: nn.Module, lm_head: nn.Module, freeze: bool = True
    ) -> None:
        """Copy the target's embedding and vocabulary projection.

        Args:
            embed_tokens: Target embedding with weight of shape [vocab, hidden].
            lm_head: Target output projection with weight of shape [vocab, hidden].
            freeze: Must be true: the draft shares the target's frozen embedding and head.
        """
        if not freeze:
            raise ValueError("DeepSeek V4.1 DFly2 requires frozen copied embedding and LM-head weights")
        if self.embed_tokens.weight.shape != embed_tokens.weight.shape:
            raise ValueError("DFly2 and target embedding shapes must match")
        if self.lm_head.weight.shape != lm_head.weight.shape:
            raise ValueError("DFly2 and target LM-head shapes must match")
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        self.embed_tokens.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states through the frozen vocabulary head in FP32.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            FP32 tensor of shape [..., vocab].
        """
        with torch.autocast(device_type=hidden_states.device.type, enabled=False):
            return F.linear(hidden_states.float(), self.lm_head.weight.float())

    def _backbone(
        self,
        noise_embeddings: torch.Tensor,
        context_keys: Sequence[torch.Tensor],
        key_indices: Sequence[torch.Tensor],
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run the draft layers on the block inputs.

        Args:
            noise_embeddings: Tensor of shape [batch, draft_sequence, hidden].
            context_keys: One tensor of shape [batch, context_keys, head_dim] per layer.
            key_indices: One integer tensor of shape [batch, draft_sequence, slots] per layer.
            position_ids: Integer tensor of shape [batch, draft_sequence].

        Returns:
            The collapsed, not yet normalized output of shape [batch, draft_sequence, hidden].
        """
        hc_mult = self.dfly2_config.text_config.hc_mult
        hidden_states = noise_embeddings.unsqueeze(2).expand(-1, -1, hc_mult, -1)
        pre_mix = noise_embeddings.new_zeros(*noise_embeddings.shape[:2], hc_mult, dtype=torch.float32)
        pre_mix[..., 0] = 1
        for layer, keys, indices in zip(self.layers, context_keys, key_indices):
            hidden_states, pre_mix = layer(hidden_states, pre_mix, keys, indices, position_ids=position_ids)
        return DeepseekV41HyperConnection.collapse(hidden_states, pre_mix)

    def forward(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        context_keys: Sequence[torch.Tensor],
        context_indices: Sequence[torch.Tensor],
        target_last_hidden_states: torch.Tensor,
    ) -> DSparkForwardOutput:
        """Run DFly2 training for sampled anchors.

        Slot ``k`` of the block at anchor ``a`` predicts token ``a + k``; slot 0 holds the
        anchor and is dropped from the outputs, so every output tensor covers the
        ``block_size - 1`` drafted slots and its teacher is the target's distribution at
        the token before the label.

        Args:
            input_ids: Target-token tensor of shape [batch, sequence].
            loss_mask: Supervision tensor of shape [batch, sequence].
            context_keys: One tensor per draft layer of shape [batch, context_keys,
                head_dim], as in ``DFly2TargetFeatures``.
            context_indices: One integer tensor per draft layer of shape [batch, sequence,
                slots], as in ``DFly2TargetFeatures``.
            target_last_hidden_states: Frozen target tensor of shape [batch, sequence,
                hidden], the input of the target's LM head.

        Returns:
            Shared DSpark loss inputs. Draft and teacher logits have shape [batch,
            num_anchors, block_size - 1, vocab]; token, mask, and confidence tensors have
            shape [batch, num_anchors, block_size - 1].
        """
        batch, sequence = input_ids.shape
        block_size = self.dfly2_config.block_size
        if len(context_keys) != len(self.layers) or len(context_indices) != len(self.layers):
            raise ValueError(f"DFly2 needs target features for {len(self.layers)} draft layers")
        if target_last_hidden_states.shape[:2] != (batch, sequence):
            raise ValueError("target_last_hidden_states must match input_ids [batch, sequence]")
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=sequence, loss_mask=loss_mask, num_anchors=self.dfly2_config.num_anchors, device=input_ids.device
        )
        num_blocks = anchor_positions.shape[1]
        hidden = target_last_hidden_states.shape[-1]
        target_last_hidden_states = target_last_hidden_states.detach()

        # Every slot's token: the anchor, then the labels a+1 .. a+block_size-1.
        slot_indices = anchor_positions.unsqueeze(-1) + torch.arange(block_size, device=input_ids.device)
        safe_slot_indices = torch.where(
            block_keep_mask.unsqueeze(-1), slot_indices.clamp(max=sequence - 1), torch.zeros_like(slot_indices)
        )
        slot_tokens = torch.gather(input_ids.unsqueeze(1).expand(-1, num_blocks, -1), 2, safe_slot_indices)

        noise_embeddings = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.dfly2_config.mask_token_id,
            block_size=block_size,
        ).view(batch, num_blocks, block_size, hidden)
        previous_rows = (anchor_positions - 1).clamp(min=0)
        previous_hidden = torch.gather(target_last_hidden_states, 1, previous_rows.unsqueeze(-1).expand(-1, -1, hidden))
        fusion = self.mtp_fusion(noise_embeddings[:, :, 0], previous_hidden.to(noise_embeddings.dtype))
        fused_blocks = (block_keep_mask & (anchor_positions > 0)).unsqueeze(-1)
        anchor_slot = noise_embeddings[:, :, :1] + torch.where(fused_blocks, fusion, 0).unsqueeze(2)
        noise_embeddings = torch.cat((anchor_slot, noise_embeddings[:, :, 1:]), dim=2).flatten(1, 2)

        position_ids = create_position_ids(anchor_positions, block_size)
        key_indices = [
            build_block_key_indices(
                indices,
                anchor_positions,
                block_keep_mask,
                num_context_keys=keys.shape[1],
                block_size=block_size,
                # The released draft attends bidirectionally within its block.
                causal=False,
            )
            for keys, indices in zip(context_keys, context_indices)
        ]
        normalized = self.final_norm(self._backbone(noise_embeddings, context_keys, key_indices, position_ids))

        # Teacher forcing: each slot's previous token is the anchor for slots 0 and 1, then the label before it.
        previous_tokens = torch.cat((slot_tokens[..., :1], slot_tokens[..., :-1]), dim=-1).flatten(1)
        corrected = self.hidden_correction(
            normalized,
            self.embed_tokens(previous_tokens),
            context_keys[-1],
            context_key_indices(key_indices[-1], context_keys[-1].shape[1]),
            position_ids=position_ids,
            block_size=block_size,
        )
        drafted = corrected.view(batch, num_blocks, block_size, hidden)[:, :, 1:]

        label_indices = slot_indices[..., 1:]
        safe_label_indices = safe_slot_indices[..., 1:]
        eval_mask = build_eval_mask(
            seq_len=sequence,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        teacher_rows = (safe_label_indices - 1).clamp(min=0)
        teacher_hidden = torch.gather(
            target_last_hidden_states.unsqueeze(1).expand(-1, num_blocks, -1, -1),
            2,
            teacher_rows.unsqueeze(-1).expand(-1, -1, -1, hidden),
        )
        confidence_pred = None
        if self.confidence_head is not None:
            # The final-normalized states, as the V4.1 DSpark draft feeds its head: the raw residual
            # stream of a trained draft is tens of units in RMS, which gives the head a loss and a
            # gradient that dominate the clipped norm of the whole draft.
            states = normalized.view(batch, num_blocks, block_size, hidden)[:, :, 1:]
            if self.dfly2_config.confidence_head_stop_gradient:
                states = states.detach()
            confidence_pred = self.confidence_head(states)
        return DSparkForwardOutput(
            draft_logits=self.compute_logits(drafted),
            target_ids=slot_tokens[..., 1:],
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=self.compute_logits(teacher_hidden),
        )


# Attention submodules a draft layer copies from its paired target layer.
_ATTENTION_INIT_MODULES = ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b", "sinks_param")

__all__ = ["DeepseekV41DFly2Config", "DeepseekV41DFly2Model", "default_dfly2_backend"]

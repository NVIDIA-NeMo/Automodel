# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""DSpark draft of Inkling 2 in the layout of the released ``mtp.safetensors``.

The checkpoint's ``mtp_config`` describes a five-layer dense drafter that reads
eight target features. Every consumer (each draft layer and the Markov head)
mixes the features with its own softmax weights (a row of ``fuse_weight``) and
normalizes the mix (its ``dflare_norm``). A layer projects its mix into context
keys and values with dedicated ``k_proj_ctx`` / ``v_proj_ctx`` / ``k_norm_ctx``
weights, while the block's own keys and values use ``k_proj`` / ``v_proj``.

A block has ``mtp_config.block_size`` slots at positions ``anchor .. anchor +
block_size - 1``: the target's normalized embedding of the anchor token, then
the learned ``mask_embed``. Rows ``1 .. block_size - 1`` propose the next
tokens, so a block drafts ``block_size - 1`` tokens; the anchor row's output is
unused. Local layers see the ``sliding_window_size`` context positions before
the anchor, the global layer the whole prefix, and every layer the whole block.

The ``rnn_hidden`` Markov head runs a gated recurrence over the proposal rows.
Its inputs are the recurrent state, the previous token's ``markov_w1`` row, the
row's hidden state and one block-wide feature, the Markov head's mix of the
target features at the position before the anchor. ``hidden_out`` maps the
recurrence output into the hidden space, and the shared LM head turns the sum
of the row's hidden state and that correction into logits, soft-capped like the
target's. Serving engines fold ``hidden_out`` into the LM head in FP64 to save
work per step; training computes the same linear map directly.

Native parameter names equal the released tensor names, so the 84 tensors of
``mtp.safetensors`` load 1:1 and the consolidated export writes the same names
and shapes. The embedding, its norm and the LM head are the frozen target's and
are not part of either file.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.attention.dflash_mask import create_dflash_sdpa_mask
from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.speculative.dspark.common import (
    DSparkForwardOutput,
    build_eval_mask,
    create_position_ids,
    sample_anchor_positions,
    validate_target_layer_ids,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

# The released draft's only Markov head variant.
_MARKOV_HEAD_TYPE = "rnn_hidden"
# Target modules the draft shares; the release does not store them.
_SHARED_KEYS = frozenset({"embed_tokens.weight", "lm_head.weight"})
_REQUIRED_MTP_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "max_position_embeddings",
    "rms_norm_eps",
    "rope_theta",
    "local_layer_ids",
    "sliding_window_size",
    "block_size",
    "markov_rank",
    "markov_head_type",
    "target_layer_ids",
)


class Inkling2DSparkOptions(Protocol):
    """Declarative DSpark settings supplied by the generic training recipe."""

    num_draft_layers: int
    target_layer_ids: list[int]
    block_size: int
    num_anchors: int
    mask_token_id: int
    markov_rank: int
    markov_head_type: str
    confidence_head_alpha: float
    confidence_head_with_markov: bool


class Inkling2DSparkTextFields(Protocol):
    """Target text-config fields the draft depends on (HF ``Inkling2TextConfig`` names)."""

    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    unpadded_vocab_size: int
    final_logit_softcapping: float


@dataclass(frozen=True)
class Inkling2DSparkReleasedConfig:
    """Draft contract read from the target's ``config.json`` ``mtp_config``.

    Attributes:
        hidden_size: Hidden width; equals the target's.
        num_hidden_layers: Number of draft decoder layers.
        num_attention_heads: Query heads per layer.
        num_key_value_heads: Key/value heads per layer.
        head_dim: Width of each attention head.
        intermediate_size: Hidden width of each layer's dense SwiGLU MLP.
        max_position_embeddings: Largest position the draft was trained for.
        rms_norm_eps: Epsilon of every RMSNorm.
        rope_theta: RoPE base of the draft attention.
        local_layer_ids: Draft layers that see only ``sliding_window_size`` context positions.
        sliding_window_size: Context positions before the anchor a local layer sees.
        block_size: Block slots, the anchor slot included.
        markov_rank: Rank of the recurrent Markov head.
        markov_head_type: Markov head variant; only ``rnn_hidden`` is released.
        target_layer_ids: Checkpoint labels of the target features: label ``L < num_target_layers``
            is the stream entering target layer ``L``, label ``num_target_layers`` the post-final-norm output.
        mtp_config: The released dictionary, copied for export.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    local_layer_ids: tuple[int, ...]
    sliding_window_size: int
    block_size: int
    markov_rank: int
    markov_head_type: str
    target_layer_ids: tuple[int, ...]
    mtp_config: dict[str, Any]

    @classmethod
    def from_mtp_config(cls, mtp_config: Mapping[str, Any]) -> Inkling2DSparkReleasedConfig:
        """Parse and validate a released ``mtp_config``.

        Args:
            mtp_config: The ``mtp_config`` mapping of the target's ``config.json``.

        Returns:
            The parsed contract.

        Raises:
            ValueError: If a field is missing or the drafter is not the released dense
                ``rnn_hidden`` design without attention biases.
        """
        missing = sorted(set(_REQUIRED_MTP_FIELDS) - set(mtp_config))
        if missing:
            raise ValueError(f"mtp_config is missing fields {missing}.")
        if mtp_config.get("q_bias", False) or mtp_config.get("o_bias", False):
            raise ValueError(
                "The Inkling 2 DSpark draft has no attention biases, but mtp_config enables q_bias/o_bias."
            )
        if mtp_config["markov_head_type"] != _MARKOV_HEAD_TYPE:
            raise ValueError(
                f"The Inkling 2 DSpark draft implements markov_head_type={_MARKOV_HEAD_TYPE!r}, "
                f"got {mtp_config['markov_head_type']!r}."
            )
        num_layers = int(mtp_config["num_hidden_layers"])
        local_layer_ids = tuple(int(i) for i in mtp_config["local_layer_ids"])
        if any(not 0 <= i < num_layers for i in local_layer_ids):
            raise ValueError(f"mtp_config.local_layer_ids {local_layer_ids} must lie in [0, {num_layers}).")
        if int(mtp_config["block_size"]) < 2:
            raise ValueError(
                f"mtp_config.block_size must count the anchor and at least one proposal, got "
                f"{mtp_config['block_size']}."
            )
        return cls(
            hidden_size=int(mtp_config["hidden_size"]),
            num_hidden_layers=num_layers,
            num_attention_heads=int(mtp_config["num_attention_heads"]),
            num_key_value_heads=int(mtp_config["num_key_value_heads"]),
            head_dim=int(mtp_config["head_dim"]),
            intermediate_size=int(mtp_config["intermediate_size"]),
            max_position_embeddings=int(mtp_config["max_position_embeddings"]),
            rms_norm_eps=float(mtp_config["rms_norm_eps"]),
            rope_theta=float(mtp_config["rope_theta"]),
            local_layer_ids=local_layer_ids,
            sliding_window_size=int(mtp_config["sliding_window_size"]),
            block_size=int(mtp_config["block_size"]),
            markov_rank=int(mtp_config["markov_rank"]),
            markov_head_type=str(mtp_config["markov_head_type"]),
            target_layer_ids=tuple(int(i) for i in mtp_config["target_layer_ids"]),
            mtp_config=copy.deepcopy(dict(mtp_config)),
        )

    def automodel_target_layer_ids(self, num_target_layers: int) -> tuple[int, ...]:
        """Return the target features in DSpark's ``target_layer_ids`` convention.

        DSpark's id ``i`` is the output of target layer ``i`` and its last id the post-final-norm
        output. The checkpoint labels the stream entering layer ``L`` and uses ``num_target_layers``
        for the post-final-norm output, so both map to ``label - 1``.

        Args:
            num_target_layers: Decoder depth of the target.

        Returns:
            Strictly increasing ids in ``[0, num_target_layers)``.

        Raises:
            ValueError: If a label is outside ``[1, num_target_layers]``.
        """
        if any(not 1 <= label <= num_target_layers for label in self.target_layer_ids):
            raise ValueError(
                f"mtp_config.target_layer_ids {list(self.target_layer_ids)} must lie in [1, {num_target_layers}]."
            )
        return tuple(label - 1 for label in self.target_layer_ids)


@dataclass(frozen=True)
class Inkling2DSparkConfig:
    """Validated settings of the Inkling 2 DSpark draft.

    Attributes:
        released: The checkpoint's draft contract.
        vocab_size: Rows of the target's embedding and LM head, padding included.
        unpadded_vocab_size: Real vocabulary; logits over the padding rows are dropped.
        final_logit_softcapping: Soft cap of the target's logits, also applied to the draft's.
        num_target_layers: Decoder depth of the target.
        num_anchors: Anchors sampled per training row.
    """

    released: Inkling2DSparkReleasedConfig
    vocab_size: int
    unpadded_vocab_size: int
    final_logit_softcapping: float
    num_target_layers: int
    num_anchors: int

    @classmethod
    def from_options(
        cls,
        text_config: Inkling2DSparkTextFields,
        mtp_config: Mapping[str, Any],
        options: Inkling2DSparkOptions,
    ) -> Inkling2DSparkConfig:
        """Validate the recipe's options against the released draft.

        The recipe's ``block_size`` counts drafted tokens, which is one less than the
        checkpoint's ``block_size`` (that one also counts the anchor slot). Its
        ``target_layer_ids`` use DSpark's convention (see
        ``Inkling2DSparkReleasedConfig.automodel_target_layer_ids``). ``mask_token_id`` is
        not read: non-anchor slots use the learned ``mask_embed``.

        Args:
            text_config: Target text configuration.
            mtp_config: The ``mtp_config`` mapping of the target's ``config.json``.
            options: Declarative draft settings supplied by the DSpark recipe.

        Returns:
            The validated settings.

        Raises:
            ValueError: If an option disagrees with the released draft or the target.
        """
        released = Inkling2DSparkReleasedConfig.from_mtp_config(mtp_config)
        num_target_layers = int(text_config.num_hidden_layers)
        expected = {
            "num_draft_layers": released.num_hidden_layers,
            "target_layer_ids": released.automodel_target_layer_ids(num_target_layers),
            "block_size": released.block_size - 1,
            "markov_rank": released.markov_rank,
            "markov_head_type": released.markov_head_type,
        }
        actual = {
            "num_draft_layers": int(options.num_draft_layers),
            "target_layer_ids": tuple(int(i) for i in options.target_layer_ids),
            "block_size": int(options.block_size),
            "markov_rank": int(options.markov_rank),
            "markov_head_type": str(options.markov_head_type),
        }
        for name, value in expected.items():
            if actual[name] != value:
                raise ValueError(f"The released Inkling 2 DSpark draft requires {name}={value}, got {actual[name]}.")
        validate_target_layer_ids(list(actual["target_layer_ids"]), num_target_layers)
        confidence_head_alpha = float(options.confidence_head_alpha)
        if confidence_head_alpha != 0.0:
            raise ValueError(
                "The released Inkling 2 DSpark draft has no confidence head; set confidence_head_alpha=0, "
                f"got {confidence_head_alpha}."
            )
        if int(text_config.hidden_size) != released.hidden_size:
            raise ValueError(
                f"The target's hidden_size={text_config.hidden_size} disagrees with mtp_config.hidden_size="
                f"{released.hidden_size}."
            )
        softcap = float(text_config.final_logit_softcapping)
        if not math.isfinite(softcap) or softcap <= 0:
            raise ValueError(f"final_logit_softcapping must be finite and positive, got {softcap}.")
        unpadded = int(text_config.unpadded_vocab_size)
        if not 0 < unpadded <= int(text_config.vocab_size):
            raise ValueError(f"unpadded_vocab_size={unpadded} must lie in (0, vocab_size={text_config.vocab_size}].")
        if int(options.num_anchors) < 1:
            raise ValueError(f"num_anchors must be >= 1, got {options.num_anchors}.")
        return cls(
            released=released,
            vocab_size=int(text_config.vocab_size),
            unpadded_vocab_size=unpadded,
            final_logit_softcapping=softcap,
            num_target_layers=num_target_layers,
            num_anchors=int(options.num_anchors),
        )

    def build(self) -> Inkling2DSparkModel:
        """Build a freshly initialized draft.

        Returns:
            The draft with FP32 parameters, ready for device placement.
        """
        return Inkling2DSparkModel(self)


@dataclass(frozen=True)
class _Rotary:
    """FP32 RoPE tables of one forward.

    Attributes:
        context_cos: Tensor of shape [batch, sequence, head_dim] for the context keys.
        context_sin: Tensor of shape [batch, sequence, head_dim] for the context keys.
        block_cos: Tensor of shape [batch, blocks * block_slots, head_dim] for the block rows.
        block_sin: Tensor of shape [batch, blocks * block_slots, head_dim] for the block rows.
    """

    context_cos: torch.Tensor
    context_sin: torch.Tensor
    block_cos: torch.Tensor
    block_sin: torch.Tensor


def _rope_tables(position_ids: torch.Tensor, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build NeoX-style RoPE tables over the full head width.

    The frequencies are computed in FP32 from the config on every call, so a model-wide
    dtype cast cannot round them.

    Args:
        position_ids: Integer tensor of shape [batch, length].
        head_dim: Rotated width; even.
        theta: RoPE base.

    Returns:
        ``cos`` and ``sin``, FP32 tensors of shape [batch, length, head_dim] whose two halves repeat.
    """
    exponent = torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim
    inv_freq = 1.0 / (theta**exponent)
    freqs = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate heads with NeoX-style RoPE, computed in FP32.

    Args:
        states: Tensor of shape [batch, heads, length, head_dim].
        cos: FP32 tensor of shape [batch, length, head_dim].
        sin: FP32 tensor of shape [batch, length, head_dim].

    Returns:
        Tensor of the input's shape and dtype.
    """
    x = states.float()
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)).to(states.dtype)


class Inkling2DSparkRMSNorm(nn.Module):
    """RMSNorm computed in FP32 with a plain (not ``1 + w``) gain."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the last axis.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of the input's shape and dtype.
        """
        return _rms_norm(hidden_states, self.weight, self.eps)


def _rms_norm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """FP32 RMSNorm of the last axis.

    Args:
        hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden], the gain.
        eps: Variance epsilon.

    Returns:
        Tensor of the input's shape and dtype.
    """
    normed = F.rms_norm(hidden_states.float(), (hidden_states.shape[-1],), weight.float(), eps)
    return normed.to(hidden_states.dtype)


class Inkling2DSparkAttention(nn.Module):
    """GQA block attention over ``[context | block]`` keys with separate context projections."""

    def __init__(self, released: Inkling2DSparkReleasedConfig) -> None:
        super().__init__()
        hidden = released.hidden_size
        self.num_heads = released.num_attention_heads
        self.num_key_value_heads = released.num_key_value_heads
        self.head_dim = released.head_dim
        q_width = self.num_heads * self.head_dim
        kv_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(hidden, q_width, bias=False)
        self.k_proj = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj = nn.Linear(hidden, kv_width, bias=False)
        self.o_proj = nn.Linear(q_width, hidden, bias=False)
        self.k_proj_ctx = nn.Linear(hidden, kv_width, bias=False)
        self.v_proj_ctx = nn.Linear(hidden, kv_width, bias=False)
        self.q_norm = Inkling2DSparkRMSNorm(self.head_dim, released.rms_norm_eps)
        self.k_norm = Inkling2DSparkRMSNorm(self.head_dim, released.rms_norm_eps)
        self.k_norm_ctx = Inkling2DSparkRMSNorm(self.head_dim, released.rms_norm_eps)

    def _heads(self, states: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Split heads.

        Args:
            states: Tensor of shape [batch, length, num_heads * head_dim].
            num_heads: Number of heads.

        Returns:
            Tensor of shape [batch, length, num_heads, head_dim] (a view).
        """
        return states.unflatten(-1, (num_heads, self.head_dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        rotary: _Rotary,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from the block rows to the context and the block.

        Args:
            hidden_states: Normalized block rows, tensor of shape [batch, blocks * block_slots, hidden].
            context: Normalized target-feature mix of this layer, tensor of shape [batch, sequence, hidden].
            rotary: RoPE tables of the context positions and the block positions.
            attention_mask: Additive mask of shape [batch, 1, blocks * block_slots, sequence + blocks *
                block_slots], zero where a row may attend and ``-inf`` elsewhere; keys are ordered
                ``[context | block_0 | block_1 | ...]``.

        Returns:
            Tensor of shape [batch, blocks * block_slots, hidden].
        """
        q = self.q_norm(self._heads(self.q_proj(hidden_states), self.num_heads)).transpose(1, 2)
        k_block = self.k_norm(self._heads(self.k_proj(hidden_states), self.num_key_value_heads)).transpose(1, 2)
        v_block = self._heads(self.v_proj(hidden_states), self.num_key_value_heads).transpose(1, 2)
        k_ctx = self.k_norm_ctx(self._heads(self.k_proj_ctx(context), self.num_key_value_heads)).transpose(1, 2)
        v_ctx = self._heads(self.v_proj_ctx(context), self.num_key_value_heads).transpose(1, 2)

        q = _apply_rope(q, rotary.block_cos, rotary.block_sin)
        k_block = _apply_rope(k_block, rotary.block_cos, rotary.block_sin)
        k_ctx = _apply_rope(k_ctx, rotary.context_cos, rotary.context_sin)
        k = self._repeat_kv(torch.cat((k_ctx, k_block), dim=2))
        v = self._repeat_kv(torch.cat((v_ctx, v_block), dim=2))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask.to(q.dtype), dropout_p=0.0, is_causal=False, scale=self.head_dim**-0.5
        )
        return self.o_proj(out.transpose(1, 2).flatten(2))

    def _repeat_kv(self, states: torch.Tensor) -> torch.Tensor:
        """Share each key/value head with its query group.

        An ``expand`` keeps the backward a plain sum (deterministic), and explicit heads keep
        the masked SDPA call on the memory-efficient kernel.

        Args:
            states: Tensor of shape [batch, key_value_heads, length, head_dim].

        Returns:
            Tensor of shape [batch, heads, length, head_dim]; query head ``h`` reads key/value
            head ``h // (heads / key_value_heads)``.
        """
        groups = self.num_heads // self.num_key_value_heads
        if groups == 1:
            return states
        batch, kv_heads, length, head_dim = states.shape
        expanded = states[:, :, None].expand(batch, kv_heads, groups, length, head_dim)
        return expanded.reshape(batch, kv_heads * groups, length, head_dim)


class Inkling2DSparkMLP(nn.Module):
    """Dense SwiGLU MLP (SiLU gate, no clamp)."""

    def __init__(self, released: Inkling2DSparkReleasedConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(released.hidden_size, released.intermediate_size, bias=False)
        self.up_proj = nn.Linear(released.hidden_size, released.intermediate_size, bias=False)
        self.down_proj = nn.Linear(released.intermediate_size, released.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP.

        Args:
            hidden_states: Tensor of shape [batch, length, hidden].

        Returns:
            Tensor of shape [batch, length, hidden].
        """
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Inkling2DSparkDecoderLayer(nn.Module):
    """Pre-norm block: attention over the layer's normalized feature mix, then the MLP."""

    def __init__(self, released: Inkling2DSparkReleasedConfig) -> None:
        super().__init__()
        hidden, eps = released.hidden_size, released.rms_norm_eps
        self.input_layernorm = Inkling2DSparkRMSNorm(hidden, eps)
        self.post_attention_layernorm = Inkling2DSparkRMSNorm(hidden, eps)
        self.dflare_norm = Inkling2DSparkRMSNorm(hidden, eps)
        self.self_attn = Inkling2DSparkAttention(released)
        self.mlp = Inkling2DSparkMLP(released)

    def forward(
        self,
        hidden_states: torch.Tensor,
        feature_mix: torch.Tensor,
        rotary: _Rotary,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one draft layer.

        Args:
            hidden_states: Block residual stream, tensor of shape [batch, blocks * block_slots, hidden].
            feature_mix: This layer's weighted sum of the target features, tensor of shape
                [batch, sequence, hidden], before ``dflare_norm``.
            rotary: RoPE tables of the context positions and the block positions.
            attention_mask: Additive mask, see ``Inkling2DSparkAttention.forward``.

        Returns:
            Tensor of shape [batch, blocks * block_slots, hidden].
        """
        context = self.dflare_norm(feature_mix)
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), context, rotary, attention_mask
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class Inkling2DSparkMarkovHead(nn.Module):
    """Gated recurrence over a block's proposal rows (``rnn_hidden``)."""

    def __init__(self, released: Inkling2DSparkReleasedConfig, vocab_size: int) -> None:
        super().__init__()
        hidden, rank = released.hidden_size, released.markov_rank
        self.markov_rank = rank
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        # Inputs ordered [state | markov_w1[previous] | row hidden | block feature];
        # outputs ordered [gate | candidate | output].
        self.rnn_proj = nn.Linear(2 * rank + 2 * hidden, 3 * rank, bias=True)
        self.hidden_out = nn.Linear(rank, hidden, bias=True)
        self.dflare_norm = Inkling2DSparkRMSNorm(hidden, released.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        previous_token_ids: torch.Tensor,
        feature_mix: torch.Tensor,
    ) -> torch.Tensor:
        """Run the recurrence teacher-forced over every proposal row.

        The state starts at zero in each block. The ``[row hidden | block feature]`` part of
        ``rnn_proj`` does not depend on the step, so it is projected for all rows at once.

        Args:
            hidden_states: Final-norm draft rows, tensor of shape [batch, blocks, proposals, hidden].
            previous_token_ids: Integer tensor of shape [batch, blocks, proposals]; entry ``j`` is the
                token before proposal ``j`` (the anchor for ``j = 0``).
            feature_mix: The head's weighted sum of the target features at the position before each
                anchor, tensor of shape [batch, blocks, hidden], before ``dflare_norm``.

        Returns:
            The hidden-space correction ``hidden_out(latent)`` of each row, tensor of shape
            [batch, blocks, proposals, hidden].
        """
        rank = self.markov_rank
        feature = self.dflare_norm(feature_mix).unsqueeze(2).expand_as(hidden_states)
        step_invariant = F.linear(
            torch.cat((hidden_states, feature), dim=-1),
            self.rnn_proj.weight[:, 2 * rank :],
            self.rnn_proj.bias,
        )
        recurrent_weight = self.rnn_proj.weight[:, : 2 * rank]
        previous = self.markov_w1(previous_token_ids.long()).to(hidden_states.dtype)
        state = hidden_states.new_zeros(*hidden_states.shape[:2], rank)
        latents = []
        for step in range(hidden_states.shape[2]):
            projected = step_invariant[:, :, step] + F.linear(
                torch.cat((state, previous[:, :, step]), dim=-1), recurrent_weight
            )
            gate, candidate, output = projected.chunk(3, dim=-1)
            gate = torch.sigmoid(gate)
            state = gate * state + (1.0 - gate) * torch.tanh(candidate)
            latents.append(torch.tanh(output))
        return self.hidden_out(torch.stack(latents, dim=2))


class Inkling2DSparkStateDictAdapter(StateDictAdapter):
    """Convert the draft between native keys and ``mtp.safetensors`` names.

    Native names already are the released names; the adapter drops the shared
    embedding and LM head on export and rejects tensors the release does not have.
    """

    def __init__(self, released_keys: frozenset[str]) -> None:
        self._released_keys = released_keys

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return the released tensors of a native state dict.

        Args:
            state_dict: Native mapping; every tensor keeps its parameter layout.
            exclude_key_regex: Optional pattern of names to drop.
            quantization: Must be false; the draft is released and exported in BF16/FP32.
            **kwargs: Additional checkpoint protocol arguments.

        Returns:
            Mapping from released names to the same tensor objects.
        """
        output: dict[str, Any] = {}
        for key, value in state_dict.items():
            output.update(
                self.convert_single_tensor_to_hf(
                    key, value, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
                )
            )
        return output

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return native tensors of a released state dict.

        Args:
            hf_state_dict: Mapping from released names to tensors in their parameter layout,
                such as the contents of ``mtp.safetensors``. The mapping is consumed.
            device_mesh: Unused; the draft has no expert-parallel tensors.
            **kwargs: Additional checkpoint protocol arguments.

        Returns:
            Mapping from native names to the same tensor objects.

        Raises:
            ValueError: If a key is not a released draft tensor.
        """
        unexpected = sorted(set(hf_state_dict) - self._released_keys)
        if unexpected:
            raise ValueError(f"Tensors {unexpected[:3]} are not part of the Inkling 2 DSpark draft.")
        return {key: hf_state_dict.pop(key) for key in list(hf_state_dict)}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Return the released name of one native tensor.

        Args:
            fqn: Native name of the tensor.
            tensor: Tensor in its parameter layout, returned as the same object.
            **kwargs: ``exclude_key_regex`` and ``quantization``.

        Returns:
            The released name and tensor, or an empty list for shared or excluded keys.

        Raises:
            ValueError: If quantized export is requested.
        """
        if kwargs.get("quantization", False):
            raise ValueError("The Inkling 2 DSpark draft exports unquantized weights.")
        if fqn in _SHARED_KEYS:
            return []
        exclude = kwargs.get("exclude_key_regex")
        return [] if exclude and re.match(exclude, fqn) else [(fqn, tensor)]


class _ReleasedDraftMetadataExporter:
    """Write the draft contract next to consolidated draft weights."""

    def __init__(self, mtp_config: dict[str, Any]) -> None:
        self._mtp_config = mtp_config

    def validate(
        self, *, tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None, original_model_path: str | None
    ) -> None:
        """Accept every input: the export only depends on the released config."""

    def save(
        self,
        *,
        hf_metadata_dir: str,
        tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None,
        original_model_path: str | None,
        v4_compatible: bool,
    ) -> None:
        """Write ``config.json`` holding the ``mtp_config`` the weights were trained for, and the tokenizer."""
        with open(os.path.join(hf_metadata_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"mtp_config": self._mtp_config}, f, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(hf_metadata_dir)


class Inkling2DSparkModel(nn.Module):
    """Inkling 2 DSpark draft with the released ``mtp.safetensors`` parameter names."""

    _no_split_modules = ["Inkling2DSparkDecoderLayer"]

    def __init__(self, config: Inkling2DSparkConfig) -> None:
        super().__init__()
        released = config.released
        self.config = config
        self.block_slots = released.block_size
        self.num_anchors = config.num_anchors
        self.local_layer_ids = frozenset(released.local_layer_ids)

        # Row c mixes the target features for consumer c: draft layers, then the Markov head.
        self.fuse_weight = nn.Parameter(torch.zeros(released.num_hidden_layers + 1, len(released.target_layer_ids)))
        # Non-anchor slots; unit scale like the normalized anchor embedding.
        self.mask_embed = nn.Parameter(torch.randn(released.hidden_size))
        self.layers = nn.ModuleList([Inkling2DSparkDecoderLayer(released) for _ in range(released.num_hidden_layers)])
        self.norm = Inkling2DSparkRMSNorm(released.hidden_size, released.rms_norm_eps)
        self.markov_head = Inkling2DSparkMarkovHead(released, config.vocab_size)

        # Shared with the frozen target and refreshed from it every run (initialize_embeddings_and_head).
        self.embed_tokens = nn.Embedding(config.vocab_size, released.hidden_size)
        self.register_buffer("embed_norm_weight", torch.ones(released.hidden_size), persistent=False)
        self.lm_head = nn.Linear(released.hidden_size, config.vocab_size, bias=False)

        # FSDP2 wants one dtype per flat-param group.
        self.to(torch.float32)
        released_keys = frozenset(self.state_dict()) - _SHARED_KEYS
        self.state_dict_adapter = Inkling2DSparkStateDictAdapter(released_keys)
        self._pre_shard_hf_state_dict_keys = self.state_dict_adapter.get_hf_state_dict_keys(self.state_dict())

    def _get_consolidated_hf_metadata_exporter(
        self, *, tokenizer: PreTrainedTokenizerBase | ProcessorMixin | None, original_model_path: str | None
    ) -> _ReleasedDraftMetadataExporter:
        """Return the writer of the draft's ``config.json``."""
        return _ReleasedDraftMetadataExporter(self.config.released.mtp_config)

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ) -> None:
        """Copy the target's normalized embedding and LM head.

        Args:
            embed_tokens: The target's input embedding: ``weight`` of shape [vocab, hidden] and
                ``embed_norm.weight`` of shape [hidden] (HF ``Inkling2NormedEmbedding`` layout), as
                modules or plain tensors.
            lm_head: The target's LM head with ``weight`` of shape [vocab, hidden].
            freeze: Must be true; the released draft does not own these weights.

        Raises:
            ValueError: If ``freeze`` is false, the embedding has no norm, or a shape differs.
        """
        if not freeze:
            raise ValueError(
                "The Inkling 2 DSpark draft shares the target's embedding and LM head and does not export them, "
                "so freeze_embeddings must be true."
            )
        embed_norm = getattr(embed_tokens, "embed_norm", None)
        if embed_norm is None:
            raise ValueError(
                "The Inkling 2 DSpark anchor uses the target's normalized embedding; the target's input "
                "embedding must expose embed_norm.weight."
            )
        sources = (
            (self.embed_tokens.weight, embed_tokens.weight),
            (self.embed_norm_weight, embed_norm.weight),
            (self.lm_head.weight, lm_head.weight),
        )
        for destination, source in sources:
            if destination.shape != source.shape:
                raise ValueError(
                    f"Shared target weight has shape {tuple(source.shape)}, expected {tuple(destination.shape)}."
                )
        with torch.no_grad():
            for destination, source in sources:
                destination.copy_(source.detach())
        self.embed_tokens.requires_grad_(False)
        self.lm_head.requires_grad_(False)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Soft-capped logits over the real vocabulary.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., unpadded_vocab] in the input's dtype; the soft cap is computed in FP32.
        """
        logits = F.linear(hidden_states, self.lm_head.weight[: self.config.unpadded_vocab_size])
        cap = self.config.final_logit_softcapping
        return (cap * torch.tanh(logits.float() / cap)).to(logits.dtype)

    def _block_embeddings(self, anchor_token_ids: torch.Tensor) -> torch.Tensor:
        """Embed every block: the anchor's normalized target embedding, then ``mask_embed``.

        Args:
            anchor_token_ids: Integer tensor of shape [batch, blocks].

        Returns:
            Tensor of shape [batch, blocks * block_slots, hidden].
        """
        anchor = _rms_norm(
            self.embed_tokens(anchor_token_ids), self.embed_norm_weight, self.config.released.rms_norm_eps
        )
        bsz, num_blocks, hidden = anchor.shape
        masks = self.mask_embed.to(anchor.dtype).expand(bsz, num_blocks, self.block_slots - 1, hidden)
        return torch.cat((anchor.unsqueeze(2), masks), dim=2).flatten(1, 2)

    def _attention_masks(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the global and the local additive masks.

        Both let a block row see its whole block and context positions before the anchor.
        The local mask also hides context positions more than ``sliding_window_size`` before
        the anchor; the window starts at the anchor for every row of the block.

        Args:
            anchor_positions: Integer tensor of shape [batch, blocks].
            block_keep_mask: Bool tensor of shape [batch, blocks]; padding blocks see only themselves.
            seq_len: Context length.
            dtype: Mask dtype.

        Returns:
            ``(global_mask, local_mask)``, each of shape [batch, 1, blocks * block_slots, seq_len +
            blocks * block_slots], zero where a row may attend and ``-inf`` elsewhere.
        """
        device = anchor_positions.device
        global_mask = create_dflash_sdpa_mask(
            anchor_positions, block_keep_mask, seq_len, self.block_slots, device, dtype
        )
        kv_idx = torch.arange(global_mask.shape[-1], device=device).view(1, 1, 1, -1)
        row_anchor = anchor_positions.repeat_interleave(self.block_slots, dim=1).view(
            anchor_positions.shape[0], 1, -1, 1
        )
        outside = (kv_idx < seq_len) & (kv_idx < row_anchor - self.config.released.sliding_window_size)
        return global_mask, global_mask.masked_fill(outside, float("-inf"))

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None = None,
    ) -> DSparkForwardOutput:
        """Run the draft on sampled anchor blocks.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            target_hidden_states: Tensor of shape [batch, sequence, features * hidden]: the target
                features concatenated in ``target_layer_ids`` order, in any floating dtype (cast to
                the parameters' dtype).
            loss_mask: Tensor of shape [batch, sequence], nonzero on supervised tokens.
            target_last_hidden_states: Optional tensor of shape [batch, sequence, hidden], the
                target's post-final-norm hidden states, for the teacher logits.

        Returns:
            Outputs over ``block_slots - 1`` proposals per block: ``draft_logits`` and
            ``aligned_target_logits`` of shape [batch, blocks, proposals, unpadded_vocab],
            ``target_ids`` and ``eval_mask`` of shape [batch, blocks, proposals],
            ``block_keep_mask`` of shape [batch, blocks], and no confidence predictions.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        released = self.config.released
        hidden = released.hidden_size
        num_features = len(released.target_layer_ids)
        if target_hidden_states.shape != (bsz, seq_len, num_features * hidden):
            raise ValueError(
                f"target_hidden_states has shape {tuple(target_hidden_states.shape)}, expected "
                f"{(bsz, seq_len, num_features * hidden)}."
            )

        # The Markov head reads the target row before the anchor, so position 0 cannot anchor.
        anchor_loss_mask = loss_mask.clone()
        anchor_loss_mask[:, 0] = 0
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len, loss_mask=anchor_loss_mask, num_anchors=self.num_anchors, device=device
        )
        num_blocks = anchor_positions.shape[1]
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)

        # Captured features can arrive in another dtype (the target's final norm runs in FP32);
        # the draft computes in its parameters' dtype.
        compute_dtype = self.mask_embed.dtype
        # [consumers, batch, sequence, hidden]: one softmax mix of the features per consumer.
        alpha = torch.softmax(self.fuse_weight.float(), dim=-1).to(compute_dtype)
        features = target_hidden_states.to(compute_dtype).unflatten(-1, (num_features, hidden))
        mixes = torch.einsum("cf,bsfh->cbsh", alpha, features)

        hidden_states = self._block_embeddings(anchor_token_ids).to(compute_dtype)
        context_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        context_cos, context_sin = _rope_tables(context_positions, released.head_dim, released.rope_theta)
        block_cos, block_sin = _rope_tables(
            create_position_ids(anchor_positions, self.block_slots), released.head_dim, released.rope_theta
        )
        rotary = _Rotary(context_cos, context_sin, block_cos, block_sin)
        global_mask, local_mask = self._attention_masks(anchor_positions, block_keep_mask, seq_len, hidden_states.dtype)
        for layer_idx, layer in enumerate(self.layers):
            mask = local_mask if layer_idx in self.local_layer_ids else global_mask
            hidden_states = layer(hidden_states, mixes[layer_idx], rotary, mask)
        hidden_states = self.norm(hidden_states)
        proposal_hidden = hidden_states.unflatten(1, (num_blocks, self.block_slots))[:, :, 1:]

        num_proposals = self.block_slots - 1
        label_offsets = torch.arange(1, num_proposals + 1, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1), safe_label_indices, torch.zeros_like(safe_label_indices)
        )
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, num_blocks, -1), 2, safe_label_indices)
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        previous_token_ids = torch.cat((anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]), dim=-1)

        before_anchor = (anchor_positions - 1).clamp(min=0)
        head_mix = torch.gather(mixes[-1], 1, before_anchor.unsqueeze(-1).expand(-1, -1, hidden))
        correction = self.markov_head(proposal_hidden, previous_token_ids, head_mix)
        draft_logits = self.compute_logits(proposal_hidden + correction)

        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_rows = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.to(compute_dtype).unsqueeze(1).expand(-1, num_blocks, -1, -1),
                2,
                target_rows.unsqueeze(-1).expand(-1, -1, -1, target_last_hidden_states.shape[-1]),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=None,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = [
    "Inkling2DSparkConfig",
    "Inkling2DSparkModel",
    "Inkling2DSparkReleasedConfig",
    "Inkling2DSparkStateDictAdapter",
]

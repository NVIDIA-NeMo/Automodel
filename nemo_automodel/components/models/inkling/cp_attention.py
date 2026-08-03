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

"""UPipe (Untied Ulysses) context-parallel attention for Inkling.

Inkling attention is walked one head-chunk at a time. Each stage projects ``cp_size``
query heads from the local sequence shard, all-to-alls them into head-sharded layout,
and then runs the rest of the chain -- short convolutions, per-head norms, the
relative-position logits, and attention itself -- on the *full* sequence for a single
head. A reverse all-to-all returns the stage's output to sequence-sharded layout.

Two properties of Inkling make this work:

* Everything before ``o_proj`` is head-independent. ``q/k/v/r_proj`` slice by weight
  rows, ``q_norm``/``k_norm`` are per-head RMSNorm over ``head_dim``, ``k_sconv`` and
  ``v_sconv`` are depthwise, and ``rel_logits_proj`` contracts only over ``d_rel``.
* The short convolutions run *after* the all-to-all, where the sequence is complete, so
  they need no halo exchange. This is why ``conv_mask`` must reach every rank at full
  length rather than being sharded.

The relative-position bias is expressed as a FlexAttention ``score_mod`` rather than
materialized: HF builds ``rel_logits`` densely at ``[B, H, S, rel_extent]``, which is
larger than Q, K and V combined at long context. Head chunking reduces the live slice to
``[B, 1, S, rel_extent]``.

Inkling-Small runs 42 layers at hidden 4096 with 32 query heads, 8 KV heads and
``head_dim`` 128 on *both* its global and sliding layers, so CP sizes 1 through 32 are
schedulable (KV replication starts at 16). Inkling-975B differs between layer types (64
query heads over 8 KV heads on global layers, 16 on sliding), which is why the schedule
is derived per layer rather than once per model.

Inference is not supported here (CP is a training-time layout); ``past_key_values`` must
be ``None``.
"""

from __future__ import annotations

import contextlib
import logging
import math
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.distributed.context_parallel.upipe import cp2hp, geometry_for_attention, hp2cp

logger = logging.getLogger(__name__)

_UPIPE_READY_LOGGED = False


# ---------------------------------------------------------------------------
# Block-mask cache
# ---------------------------------------------------------------------------
# create_block_mask costs milliseconds per call, and the mask depends only on the
# sequence geometry plus the padding map -- never on Q/K/V content. Every stage of every
# same-type layer in a step therefore builds an identical mask. Cache on the scalars and
# clear when the batch's padding tensor changes, holding that tensor as the generation
# token so its data_ptr cannot be recycled while it is still the key.
_BLOCK_MASK_CACHE: dict = {}
_BLOCK_MASK_GEN: list = [None, None]


def _block_mask_set_generation(gen_tensor: torch.Tensor | None) -> None:
    """Reset the per-step block-mask cache when a new batch arrives."""
    ptr = None if gen_tensor is None else gen_tensor.data_ptr()
    if ptr != _BLOCK_MASK_GEN[0]:
        _BLOCK_MASK_CACHE.clear()
        _BLOCK_MASK_GEN[0] = ptr
        _BLOCK_MASK_GEN[1] = gen_tensor


def _compiled_flex_attention():
    """Return the process-wide compiled ``flex_attention``."""
    compiled = globals().get("_COMPILED_FLEX")
    if compiled is None:
        from torch.nn.attention.flex_attention import flex_attention

        compiled = torch.compile(flex_attention, dynamic=True)
        globals()["_COMPILED_FLEX"] = compiled
    return compiled


@contextlib.contextmanager
def _duck_shape_disabled():
    """Locally disable flex duck-shape specialization for the wrapped flex call.

    Head chunking makes the head dimension 1, which duck-shaping happily unifies with any
    other incidental 1 in the shape, so the compiled kernel guards on coincidences and
    recompiles whenever they stop holding. Dynamo reads ``use_duck_shape`` at (re)trace
    time inside the flex call, so scoping it to the call window avoids mutating the
    process-global fx config.
    """
    from torch.fx.experimental import _config as _fx_config

    prev = _fx_config.use_duck_shape
    _fx_config.use_duck_shape = False
    try:
        yield
    finally:
        _fx_config.use_duck_shape = prev


def make_inkling_mask_mod(sliding_window: int | None, padding_mask: torch.Tensor | None):
    """Build the FlexAttention ``mask_mod`` for one Inkling layer.

    Args:
        sliding_window: Window size on ``hybrid_sliding`` layers, ``None`` on global ones.
        padding_mask: Optional ``[B, S]`` bool tensor, ``True`` at padded positions.

    Returns:
        Callable: A ``mask_mod(b, h, q_idx, kv_idx)`` predicate. Indices are global,
        because after the all-to-all every rank holds the whole sequence.

    Fully padded query rows would otherwise mask out every key and make softmax produce
    NaN, so such rows are pointed at key 0; their outputs are zeroed afterwards. Eager HF
    leaves those rows as NaN, so padded positions differ from the reference by design;
    they are dropped by the loss either way.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        allowed = kv_idx <= q_idx
        if sliding_window is not None:
            allowed = allowed & ((q_idx - kv_idx) < sliding_window)
        if padding_mask is not None:
            allowed = allowed & ~padding_mask[b, kv_idx]
            allowed = torch.where(padding_mask[b, q_idx], kv_idx == 0, allowed)
        return allowed

    return mask_mod


def make_rel_bias_score_mod(rel_logits: torch.Tensor, rel_extent: int):
    """Build the FlexAttention ``score_mod`` carrying Inkling's relative-position bias.

    Mirrors ``InklingRelativeLogits.forward``: the bias at ``(q, k)`` is
    ``rel_logits[b, h, q, clamp(q - k, 0, rel_extent - 1)]``, and is zero outside
    ``0 <= q - k < rel_extent``.

    Args:
        rel_logits: ``[B, H_local, S, rel_extent]`` bias bank for this stage's head. Any
            log-scaling factor must already be folded in.
        rel_extent: Maximum backward distance carrying a bias.

    Returns:
        Callable: A ``score_mod(score, b, h, q_idx, kv_idx)``.
    """

    def score_mod(score, b, h, q_idx, kv_idx):
        distance = q_idx - kv_idx
        in_range = (distance >= 0) & (distance < rel_extent)
        gather_idx = torch.where(in_range, distance, torch.zeros_like(distance))
        bias = rel_logits[b, h, q_idx, gather_idx]
        return score + torch.where(in_range, bias, torch.zeros_like(bias))

    return score_mod


def _build_block_mask(
    *,
    sliding_window: int | None,
    padding_mask: torch.Tensor | None,
    batch: int,
    seq_len: int,
    device: torch.device,
):
    """Build (and cache for the step) the causal/sliding/padding block mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    _block_mask_set_generation(padding_mask)
    mask_batch = batch if padding_mask is not None else None
    key = (sliding_window, mask_batch, seq_len, device.type, device.index)
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    mask = create_block_mask(
        make_inkling_mask_mod(sliding_window, padding_mask),
        B=mask_batch,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    if len(_BLOCK_MASK_CACHE) >= 64:
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[key] = mask
    return mask


# ---------------------------------------------------------------------------
# Short convolution on the head-sharded (full-sequence) layout
# ---------------------------------------------------------------------------


def _sconv_weight(sconv: torch.nn.Module) -> torch.Tensor:
    """Return the depthwise conv weight as ``[channels, kernel]``.

    Accepts both the raw HuggingFace ``InklingShortConvolution`` (``conv1d.weight``) and
    AutoModel's fp32-holder replacement (``_fp32_params.weight``).
    """
    holder = getattr(sconv, "_fp32_params", None)
    weight = holder.weight if holder is not None else sconv.conv1d.weight
    return weight.squeeze(1)


def apply_head_sconv(
    sconv: torch.nn.Module,
    hidden_states: torch.Tensor,
    head_index: int,
    head_dim: int,
    conv_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Apply one head's slice of a depthwise short convolution over the full sequence.

    Reproduces ``InklingShortConvolution.forward`` for the training path (no cache): fp32
    compute, padding states zeroed, causal depthwise conv, residual add, cast back. The
    weight is sliced in the forward pass so gradients still reach the full parameter.

    Args:
        sconv: The short-convolution module owning the full-width weight.
        hidden_states: ``[B, S_full, head_dim]`` for this rank's single head.
        head_index: Index of that head within the module's channel layout.
        head_dim: Channels per head.
        conv_mask: Optional full-length ``[B, S_full]`` padding mask.

    Returns:
        torch.Tensor: ``[B, S_full, head_dim]`` in the input dtype.
    """
    from transformers.models.inkling.modeling_inkling import (
        apply_mask_to_padding_states,
        causal_conv1d_fn,
    )

    input_dtype = hidden_states.dtype
    states = hidden_states.float()

    residual = states
    states = apply_mask_to_padding_states(states, conv_mask)
    states = states.transpose(1, 2)

    weight = _sconv_weight(sconv)[head_index * head_dim : (head_index + 1) * head_dim]
    states = causal_conv1d_fn(states, weight, None, seq_idx=None)

    states = states.transpose(1, 2)
    return (states + residual).to(dtype=input_dtype)


# ---------------------------------------------------------------------------
# Stage bodies
# ---------------------------------------------------------------------------


def _head_rows(weight: torch.Tensor, per_head: int, heads: torch.Tensor) -> torch.Tensor:
    """Gather whole-head row blocks out of a ``[num_heads * per_head, in_dim]`` weight.

    Selecting rows (rather than slicing a contiguous span) is what lets a stage own an
    arbitrary set of heads, and is also how a replicated KV head accumulates gradient
    from each of its replicas.
    """
    in_dim = weight.shape[-1]
    return weight.view(-1, per_head, in_dim).index_select(0, heads).reshape(-1, in_dim)


def _kv_stage(
    module: torch.nn.Module,
    stage: int,
    hidden_states: torch.Tensor,
    conv_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project, redistribute, convolve and normalize this stage's K and V.

    Returns ``(key, value)`` of shape ``[B, S_full, 1, head_dim]`` in head-sharded layout.
    """
    geometry = module._upipe_geometry
    group = module._upipe_group
    cp_rank = module._upipe_rank
    head_dim = geometry.head_dim
    batch, seq_local, _ = hidden_states.shape

    source_heads = geometry.stage_kv_source_heads(stage, device=hidden_states.device)
    key = F.linear(hidden_states, _head_rows(module.k_proj.weight, head_dim, source_heads))
    value = F.linear(hidden_states, _head_rows(module.v_proj.weight, head_dim, source_heads))
    key = key.view(batch, seq_local, geometry.cp_size, head_dim)
    value = value.view(batch, seq_local, geometry.cp_size, head_dim)

    key = cp2hp(key, group)
    value = cp2hp(value, group)

    # Full sequence is now local, so the causal convolutions need no halo.
    my_head = geometry.source_kv_head(stage, cp_rank)
    seq_full = key.shape[1]
    flat = (batch, seq_full, head_dim)
    per_head = (batch, seq_full, 1, head_dim)
    key = apply_head_sconv(module.k_sconv, key.reshape(flat), my_head, head_dim, conv_mask).view(per_head)
    value = apply_head_sconv(module.v_sconv, value.reshape(flat), my_head, head_dim, conv_mask).view(per_head)

    return module.k_norm(key), value


def _query_stage(
    module: torch.nn.Module,
    stage: int,
    hidden_states: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run attention for this stage's query head and return it sequence-sharded.

    Returns ``[B, S_local, cp_size, head_dim]``.
    """
    geometry = module._upipe_geometry
    group = module._upipe_group
    head_dim = geometry.head_dim
    d_rel = module.rel_logits_proj.proj.shape[0]
    batch, seq_local, _ = hidden_states.shape
    seq_full = key.shape[1]

    heads = geometry.stage_query_heads(stage, device=hidden_states.device)
    query = F.linear(hidden_states, _head_rows(module.q_proj.weight, head_dim, heads))
    relative = F.linear(hidden_states, _head_rows(module.r_proj.weight, d_rel, heads))
    query = query.view(batch, seq_local, geometry.cp_size, head_dim)
    relative = relative.view(batch, seq_local, geometry.cp_size, d_rel)

    query = module.q_norm(cp2hp(query, group))
    relative = cp2hp(relative, group)

    # [B, S_full, 1, d_rel] -> [B, 1, S_full, rel_extent]
    rel_logits = (relative @ module.rel_logits_proj.proj).transpose(1, 2)

    tau = _log_scaling_tau(module, seq_full, hidden_states.device)
    if tau is not None:
        query = (query.float() * tau.view(1, -1, 1, 1)).to(query.dtype)
        rel_logits = (rel_logits.float() * tau.view(1, 1, -1, 1)).to(rel_logits.dtype)

    block_mask = _build_block_mask(
        sliding_window=module.sliding_window,
        padding_mask=padding_mask,
        batch=batch,
        seq_len=seq_full,
        device=query.device,
    )
    with _duck_shape_disabled():
        attn_out = _compiled_flex_attention()(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            score_mod=make_rel_bias_score_mod(rel_logits, module.rel_extent),
            block_mask=block_mask,
            scale=module.scaling,
        )
    attn_out = attn_out.transpose(1, 2)
    if padding_mask is not None:
        attn_out = attn_out.masked_fill(padding_mask[:, :, None, None], 0.0)

    return hp2cp(attn_out.to(query.dtype), group)


def _log_scaling_tau(module: torch.nn.Module, seq_len: int, device: torch.device) -> torch.Tensor | None:
    """Per-position log-scaling factor, or ``None`` when the layer does not use it.

    Only global layers apply it. Positions are global without an offset because the
    all-to-all has already gathered the whole sequence onto every rank.
    """
    n_floor = getattr(module.config, "log_scaling_n_floor", None)
    if module.is_sliding or n_floor is None:
        return None
    effective_n = (torch.arange(seq_len, device=device) + 1).float()
    return 1.0 + module.config.log_scaling_alpha * torch.log((effective_n / n_floor).clamp(min=1.0))


# ---------------------------------------------------------------------------
# Forward override
# ---------------------------------------------------------------------------


def _padding_mask_from_conv_mask(conv_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Convert Inkling's keep-mask into a bool padding mask, or ``None`` if unpadded."""
    if conv_mask is None or conv_mask.dim() != 2:
        return None
    padding = ~conv_mask.bool()
    if not bool(padding.any()):
        return None
    return padding


def upipe_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    conv_mask: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Context-parallel replacement for ``InklingAttention.forward``.

    ``hidden_states`` is this rank's contiguous sequence shard, ``[B, S_local, D]``.
    ``attention_mask`` is ignored: causality, the sliding window and padding are all
    expressed through the FlexAttention block mask over global indices. ``conv_mask`` is
    expected at full length, since the short convolutions run after the all-to-all.
    """
    if past_key_values is not None:
        raise NotImplementedError("Inkling UPipe context parallelism is training-only; no KV cache support.")

    geometry = self._upipe_geometry
    batch, seq_local, _ = hidden_states.shape
    seq_full = seq_local * geometry.cp_size

    if conv_mask is not None and conv_mask.dim() == 2 and conv_mask.shape[1] not in (seq_full, 1):
        raise ValueError(
            f"Inkling UPipe CP expects a full-length conv_mask ({seq_full} positions), "
            f"got {conv_mask.shape[1]}. The CP sharder must leave it unsharded."
        )
    padding_mask = _padding_mask_from_conv_mask(conv_mask)

    global _UPIPE_READY_LOGGED
    if not _UPIPE_READY_LOGGED:
        logger.info(
            "Inkling UPipe CP active: cp_size=%d pipe_degree=%d gqa_ratio=%d kv_replication=%d "
            "heads=%d kv_heads=%d sliding_window=%s",
            geometry.cp_size,
            geometry.pipe_degree,
            geometry.gqa_ratio,
            geometry.kv_replication,
            geometry.num_heads,
            geometry.num_kv_heads,
            self.sliding_window,
        )
        _UPIPE_READY_LOGGED = True

    use_checkpoint = self.training and torch.is_grad_enabled()
    stage_outputs = []
    key = value = None
    for stage in range(geometry.pipe_degree):
        if geometry.recomputes_kv(stage):
            if use_checkpoint:
                key, value = checkpoint(_kv_stage, self, stage, hidden_states, conv_mask, use_reentrant=False)
            else:
                key, value = _kv_stage(self, stage, hidden_states, conv_mask)
        if use_checkpoint:
            stage_out = checkpoint(
                _query_stage, self, stage, hidden_states, key, value, padding_mask, use_reentrant=False
            )
        else:
            stage_out = _query_stage(self, stage, hidden_states, key, value, padding_mask)
        stage_outputs.append(stage_out)

    # Stages emit heads in schedule order; restore natural order before o_proj.
    attn_output = torch.cat(stage_outputs, dim=2)
    attn_output = attn_output.index_select(2, self._upipe_inverse_head_order.to(attn_output.device))
    attn_output = attn_output.reshape(batch, seq_local, -1)
    return self.o_proj(attn_output), None


def setup_cp_attention(self, cp_mesh) -> None:
    """Install the UPipe forward on this attention module for ``cp_mesh``.

    This is the model-owned CP seam the parallelizer calls. A CP size of 1 leaves the
    module untouched so the non-CP path stays byte-identical.
    """
    group = cp_mesh.get_group()
    cp_size = dist.get_world_size(group)
    if cp_size == 1:
        return

    if getattr(self, "attention_dropout", 0.0):
        raise NotImplementedError(
            f"Inkling UPipe CP does not support attention dropout (got {self.attention_dropout}); "
            "FlexAttention has no dropout hook, so enabling it would silently diverge from the "
            "single-device path."
        )

    geometry = geometry_for_attention(self, cp_size)
    self._upipe_cp_mesh = cp_mesh
    self._upipe_group = group
    self._upipe_rank = dist.get_rank(group)
    self._upipe_geometry = geometry
    self._upipe_inverse_head_order = geometry.inverse_head_order()
    self._cp_uses_attention_hook = True
    self.forward = MethodType(upipe_attention_forward, self)


def attach_inkling_upipe_attention(module: torch.nn.Module) -> None:
    """Expose the UPipe CP seam on an ``InklingAttention`` module.

    Binds ``setup_cp_attention`` so ``apply_cp`` can install the head-chunked forward once
    the CP mesh exists. Attaching is inert until then.
    """
    module.setup_cp_attention = MethodType(setup_cp_attention, module)


def reference_attention_with_rel_bias(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    rel_logits: torch.Tensor,
    rel_extent: int,
    scaling: float,
    sliding_window: int | None = None,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dense reference for the flex path, used by tests.

    Materializes the bias and the mask the way ``InklingRelativeLogits`` and
    ``eager_attention_forward`` do, so a mismatch isolates the ``score_mod`` / ``mask_mod``
    translation rather than the surrounding pipeline.

    Args:
        query: ``[B, H, S, D]``.
        key: ``[B, H, S, D]``.
        value: ``[B, H, S, D]``.
        rel_logits: ``[B, H, S, rel_extent]``.
        rel_extent: Maximum backward distance carrying a bias.
        scaling: Logit scale, ``1 / head_dim`` for Inkling.
        sliding_window: Window size, or ``None`` for global attention.
        padding_mask: Optional ``[B, S]`` bool tensor, ``True`` at padded positions.

    Returns:
        torch.Tensor: ``[B, H, S, D]`` attention output.
    """
    seq_len = query.shape[2]
    positions = torch.arange(seq_len, device=query.device)
    distance = (positions[:, None] - positions[None, :])[None, None, :, :]

    gather_index = distance.clamp(0, rel_extent - 1).expand(*rel_logits.shape[:2], -1, -1)
    position_bias = rel_logits.gather(-1, gather_index)
    position_bias = position_bias.masked_fill((distance < 0) | (distance >= rel_extent), 0.0)

    allowed = distance >= 0
    if sliding_window is not None:
        allowed = allowed & (distance < sliding_window)
    if padding_mask is not None:
        allowed = allowed & ~padding_mask[:, None, None, :]
        allowed = torch.where(padding_mask[:, None, :, None], positions.view(1, 1, 1, -1) == 0, allowed)

    scores = torch.matmul(query, key.transpose(2, 3)) * scaling + position_bias
    scores = scores.masked_fill(~allowed, -math.inf)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    out = torch.matmul(weights, value)
    if padding_mask is not None:
        out = out.masked_fill(padding_mask[:, None, :, None], 0.0)
    return out

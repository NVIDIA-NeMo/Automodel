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

"""Ulysses context parallelism for Inkling attention and short convolutions."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


class _AllToAll(torch.autograd.Function):
    """Autograd-aware equal-split all-to-all."""

    @staticmethod
    def forward(ctx: Any, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        """Exchange equal leading-axis chunks.

        Args:
            ctx: Autograd context storing the process group.
            tensor: Contiguous tensor of shape [cp, ...], where axis 0 selects
                the destination CP rank.
            group: One-dimensional context-parallel process group.

        Returns:
            Tensor of shape [cp, ...], with axis 0 ordered by source CP rank.
        """
        ctx.group = group
        output = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        dist.all_to_all_single(output, tensor.contiguous(), group=group)
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Route output gradients back to their source ranks.

        Args:
            ctx: Autograd context containing the forward process group.
            grad_output: Tensor of shape [cp, ...] in source-rank order.

        Returns:
            Gradient tensor of shape [cp, ...] in destination-rank order and
            no gradient for the process group.
        """
        grad_input = torch.empty(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device)
        dist.all_to_all_single(grad_input, grad_output.contiguous(), group=ctx.group)
        return grad_input, None


def _sequence_to_head(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Convert a contiguous sequence shard into a full-sequence head shard.

    Args:
        tensor: Tensor of shape [batch, heads, local_sequence, head_dim].
        group: Context-parallel process group of size ``cp``. ``heads`` must be
            divisible by ``cp``.

    Returns:
        Tensor of shape [batch, heads / cp, global_sequence, head_dim].
    """
    cp_size = dist.get_world_size(group)
    batch, heads, local_sequence, head_dim = tensor.shape
    if heads % cp_size:
        raise ValueError(f"Ulysses cp_size={cp_size} must divide the Inkling head count {heads}.")
    local_heads = heads // cp_size
    send = tensor.reshape(batch, cp_size, local_heads, local_sequence, head_dim)
    send = send.permute(1, 0, 2, 3, 4).contiguous()
    received = _AllToAll.apply(send, group)
    return received.permute(1, 2, 0, 3, 4).reshape(batch, local_heads, cp_size * local_sequence, head_dim)


def _head_to_sequence(tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Convert a full-sequence head shard into a contiguous sequence shard.

    Args:
        tensor: Tensor of shape [batch, local_heads, global_sequence, head_dim].
        group: Context-parallel process group of size ``cp``.

    Returns:
        Tensor of shape [batch, local_heads * cp, local_sequence, head_dim].
    """
    cp_size = dist.get_world_size(group)
    batch, local_heads, global_sequence, head_dim = tensor.shape
    if global_sequence % cp_size:
        raise ValueError(f"Ulysses global sequence length {global_sequence} must be divisible by cp_size={cp_size}.")
    local_sequence = global_sequence // cp_size
    send = tensor.reshape(batch, local_heads, cp_size, local_sequence, head_dim)
    send = send.permute(2, 0, 1, 3, 4).contiguous()
    received = _AllToAll.apply(send, group)
    return received.permute(1, 0, 2, 3, 4).reshape(batch, cp_size * local_heads, local_sequence, head_dim)


def gather_padding_mask(padding_mask: torch.Tensor | None, group: dist.ProcessGroup) -> torch.Tensor | None:
    """Gather contiguous padding-mask shards into global sequence order.

    Args:
        padding_mask: Optional boolean tensor of shape [batch, local_sequence],
            where True marks padding.
        group: Context-parallel process group.

    Returns:
        Optional boolean tensor of shape [batch, global_sequence].
    """
    if padding_mask is None:
        return None
    parts = [torch.empty_like(padding_mask) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, padding_mask.contiguous(), group=group)
    return torch.cat(parts, dim=1)


_BLOCK_MASK_CACHE: dict[tuple[Any, ...], Any] = {}
_BLOCK_MASK_GENERATION: list[Any] = [None, None]
_COMPILED_FLEX_ATTENTION: list[Any] = [None]


def _get_block_mask(
    padding_mask: torch.Tensor | None,
    *,
    sequence_length: int,
    sliding_window: int | None,
    device: torch.device,
) -> Any:
    """Build one causal FlexAttention mask for a full Inkling sequence.

    Args:
        padding_mask: Optional boolean tensor of shape
            ``[batch, global_sequence]`` where True marks padding.
        sequence_length: Global query and key sequence length.
        sliding_window: Optional causal sliding-window width.
        device: Device on which FlexAttention creates the block mask.

    Returns:
        FlexAttention block mask for the global sequence.
    """
    from torch.nn.attention.flex_attention import BlockMask

    pointer = padding_mask.data_ptr() if padding_mask is not None else None
    if pointer != _BLOCK_MASK_GENERATION[0]:
        _BLOCK_MASK_CACHE.clear()
        _BLOCK_MASK_GENERATION[:] = [pointer, padding_mask]

    batch_size = padding_mask.shape[0] if padding_mask is not None else None
    key = (pointer, batch_size, sequence_length, sliding_window, device.type, device.index)
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    block_size = 128
    num_blocks = (sequence_length + block_size - 1) // block_size
    block_ids = torch.arange(num_blocks, device=device)
    q_start = block_ids[:, None] * block_size
    q_end = torch.minimum(q_start + block_size - 1, q_start.new_tensor(sequence_length - 1))
    kv_start = block_ids[None, :] * block_size
    kv_end = torch.minimum(kv_start + block_size - 1, kv_start.new_tensor(sequence_length - 1))

    if padding_mask is None:
        valid_lengths = q_start.new_full((1,), sequence_length)
    else:
        valid = ~padding_mask
        valid_lengths = valid.sum(dim=1)
        positions = torch.arange(sequence_length, device=device)
        if not torch.equal(valid, positions[None, :] < valid_lengths[:, None]):
            raise NotImplementedError("Inkling Ulysses requires right padding.")

    valid_q_end = torch.minimum(q_end[None, :, :], valid_lengths[:, None, None] - 1)
    any_allowed = (q_start[None, :, :] < valid_lengths[:, None, None]) & (kv_start[None, :, :] <= valid_q_end)
    full_allowed = (q_end[None, :, :] < valid_lengths[:, None, None]) & (kv_end < q_start)
    if sliding_window is not None:
        min_distance = (q_start - kv_end).clamp(min=0)
        any_allowed &= min_distance[None, :, :] < sliding_window
        full_allowed &= (q_end - kv_start)[None, :, :] < sliding_window

    complete_q = q_end - q_start + 1 == block_size
    complete_kv = kv_end - kv_start + 1 == block_size
    full_allowed &= complete_q[None, :, :] & complete_kv[None, :, :]
    padding_queries = q_end[None, :, :] >= valid_lengths[:, None, None]
    padding_fallback = padding_queries & (block_ids[None, None, :] == 0)
    partial_allowed = (any_allowed | padding_fallback) & ~full_allowed

    partial_allowed = partial_allowed[:, None]
    full_allowed = full_allowed[:, None]
    kv_num_blocks = partial_allowed.sum(dim=-1, dtype=torch.int32)
    kv_indices = torch.argsort(partial_allowed.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
    full_kv_num_blocks = full_allowed.sum(dim=-1, dtype=torch.int32)
    full_kv_indices = torch.argsort(full_allowed.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)

    def mask_mod(
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Return whether one scalar query/key coordinate may attend.

        Args:
            batch_idx: Scalar batch-index tensor.
            head_idx: Scalar head-index tensor.
            q_idx: Scalar global query-position tensor.
            kv_idx: Scalar global key-position tensor.

        Returns:
            Scalar boolean tensor indicating whether the coordinate is valid.
        """
        del head_idx
        allowed = kv_idx <= q_idx
        if sliding_window is not None:
            allowed = allowed & ((q_idx - kv_idx) < sliding_window)
        if padding_mask is not None:
            query_is_padding = padding_mask[batch_idx, q_idx]
            key_is_padding = padding_mask[batch_idx, kv_idx]
            allowed = allowed & ~key_is_padding
            allowed = torch.where(query_is_padding, kv_idx == 0, allowed)
        return allowed

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(sequence_length, sequence_length),
    )
    _BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


def _compiled_flex_attention():
    """Return the shared dynamically compiled FlexAttention callable."""
    if _COMPILED_FLEX_ATTENTION[0] is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX_ATTENTION[0] = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION[0]


def inkling_ulysses_attention(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    conv_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    group: dist.ProcessGroup,
    past_key_values: Any | None = None,
) -> tuple[torch.Tensor, None]:
    """Run Inkling attention with Ulysses sequence/head redistribution.

    Args:
        module: Inkling attention module whose projections and learned relative
            bias are used.
        hidden_states: Local tensor of shape [batch, local_sequence, hidden].
        conv_mask: Optional boolean tensor of shape [batch, local_sequence],
            where True marks convolution inputs to retain.
        padding_mask: Optional boolean tensor of shape [batch, global_sequence],
            where True marks padding.
        group: Context-parallel process group.
        past_key_values: Optional inference cache. Cached decoding is unsupported
            with Ulysses training.

    Returns:
        Local attention output of shape [batch, local_sequence, hidden] and no
        attention-weight tensor.
    """
    if past_key_values is not None:
        raise NotImplementedError("Inkling Ulysses supports training without a KV cache.")
    if module.training and module.attention_dropout:
        raise NotImplementedError("Inkling Ulysses requires attention_dropout=0.")

    batch, local_sequence, _ = hidden_states.shape
    query = module.q_proj(hidden_states).view(batch, local_sequence, module.num_heads, module.head_dim)
    key = module.k_proj(hidden_states)
    value = module.v_proj(hidden_states)
    key = module.k_sconv(key, conv_mask=conv_mask, cp_group=group)
    value = module.v_sconv(value, conv_mask=conv_mask, cp_group=group)
    key = key.view(batch, local_sequence, module.num_key_value_heads, module.head_dim)
    value = value.view(batch, local_sequence, module.num_key_value_heads, module.head_dim)
    relative = module.r_proj(hidden_states).view(batch, local_sequence, module.num_heads, module.config.d_rel)

    query = module.q_norm(query).transpose(1, 2)
    key = module.k_norm(key).transpose(1, 2)
    value = value.transpose(1, 2)
    relative = relative.transpose(1, 2)

    key = key.repeat_interleave(module.num_key_value_groups, dim=1)
    value = value.repeat_interleave(module.num_key_value_groups, dim=1)
    query = _sequence_to_head(query, group)
    key = _sequence_to_head(key, group)
    value = _sequence_to_head(value, group)
    relative = _sequence_to_head(relative, group)

    global_sequence = query.shape[2]
    tau = None
    if not module.is_sliding and module.config.log_scaling_n_floor is not None:
        positions = torch.arange(global_sequence, device=query.device)
        tau = 1.0 + module.config.log_scaling_alpha * torch.log(
            ((positions + 1).float() / module.config.log_scaling_n_floor).clamp(min=1.0)
        )
        tau = tau.view(1, 1, global_sequence, 1)
        query = (query.float() * tau).to(query.dtype)

    relative_profiles = module.rel_logits_proj.proj
    relative_extent = module.rel_logits_proj.rel_extent
    relative_bias = torch.einsum("bhqd,de->bhqe", relative, relative_profiles)
    if tau is not None:
        relative_bias = (relative_bias.float() * tau).to(relative_bias.dtype)

    def score_mod(
        score: torch.Tensor,
        batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Add Inkling's learned relative bias to one attention score.

        Args:
            score: Scalar query/key dot-product tensor.
            batch_idx: Scalar batch-index tensor.
            head_idx: Scalar local-head-index tensor.
            q_idx: Scalar global query-position tensor.
            kv_idx: Scalar global key-position tensor.

        Returns:
            Scalar attention score including learned relative bias.
        """
        distance = q_idx - kv_idx
        profile_idx = distance.clamp(0, relative_extent - 1)
        bias = relative_bias[batch_idx, head_idx, q_idx, profile_idx]
        return score + torch.where((distance >= 0) & (distance < relative_extent), bias, 0.0)

    block_mask = _get_block_mask(
        padding_mask,
        sequence_length=global_sequence,
        sliding_window=module.sliding_window,
        device=query.device,
    )
    output = _compiled_flex_attention()(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        score_mod=score_mod,
        block_mask=block_mask,
        scale=module.scaling,
    )
    output = _head_to_sequence(output, group)
    output = output.transpose(1, 2).reshape(batch, local_sequence, -1).contiguous()
    return module.o_proj(output), None

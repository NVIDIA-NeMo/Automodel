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

"""Context-parallel and packed-sequence support for Kimi Linear.

Kimi Linear interleaves KDA linear-attention layers with MLA full-attention
layers, so context parallelism has to satisfy both at once:

* KDA carries a sequential recurrent state, so FLA's context-parallel kernels
  require every rank to own one **contiguous** slice of the global token stream
  (rank ``r`` owns ``[r * S / cp, (r + 1) * S / cp)``) and take document
  boundaries through ``cu_seqlens``. PyTorch's default load-balanced
  ``context_parallel`` layout (head/tail chunk swap) does not satisfy that, so
  Kimi Linear owns its batch sharding through ``_cp_make_batch_fn``.
* MLA attends globally. Under the contiguous layout each rank all-gathers the
  *compressed* KV latent (``kv_lora_rank + qk_rope_head_dim`` values per token,
  roughly an order of magnitude smaller than the expanded per-head K/V) and runs
  FlexAttention with a causal, per-document block mask against the full-sequence
  keys.

Everything is driven by one ``[batch, sequence]`` document-id map (``0`` marks
padding, ``1..n`` are 1-based document indices), which is also what makes packed
sequences work with and without CP.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.distributed.context_parallel.sharder import ShardLayout

_PAD_DOC_ID = 0


@dataclass
class KimiPackedContext:
    """Per-step document layout shared by the KDA and MLA layers.

    Attributes:
        doc_ids: Global document ids of shape [batch, sequence] covering the full
            (unsharded) sequence, where 0 marks padding.
        seq_start: Global sequence offset of the local shard; 0 without CP.
        cp_size: Number of context-parallel shards the global sequence was split into.
    """

    doc_ids: torch.Tensor
    seq_start: int = 0
    cp_size: int = 1

    def __post_init__(self) -> None:
        self._cu_seqlens: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._has_multiple_documents: bool | None = None

    @property
    def cp_enabled(self) -> bool:
        """Whether the batch was sharded across a context-parallel mesh."""
        return self.cp_size > 1

    @property
    def local_doc_ids(self) -> torch.Tensor:
        """Document ids of shape [batch, local_sequence] for this rank's shard."""
        if not self.cp_enabled:
            return self.doc_ids
        local_len = self.doc_ids.shape[1] // self.cp_size
        return self.doc_ids[:, self.seq_start : self.seq_start + local_len]

    @property
    def has_multiple_documents(self) -> bool:
        """Whether any batch row contains more than one non-padding document."""
        if self._has_multiple_documents is None:
            self._has_multiple_documents = bool((self.doc_ids > 1).any().item())
        return self._has_multiple_documents

    def row_cu_seqlens(self, row: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the global segment boundaries of one batch row.

        Computed on first use (and cached for the step) because the device-to-host
        copy is only needed by the context-parallel path.

        Args:
            row: Batch row to describe.

        Returns:
            The device and CPU copies of the row's cumulative segment lengths, each
            of shape [segments + 1].
        """
        cached = self._cu_seqlens.get(row)
        if cached is None:
            cu_seqlens = segment_cu_seqlens(self.doc_ids[row]).to(torch.long)
            cached = (cu_seqlens, cu_seqlens.cpu())
            self._cu_seqlens[row] = cached
        return cached


def doc_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Build document ids from a binary or indexed attention mask.

    Args:
        attention_mask: Tensor of shape [batch, sequence]. A binary mask marks
            valid tokens with 1; an Automodel packing mask marks document ``i``
            (1-based) with the value ``i`` and padding with 0.

    Returns:
        Tensor of shape [batch, sequence] with 1-based document ids and 0 for
        padding.
    """
    return attention_mask.to(torch.int32)


def doc_ids_from_seq_lens(seq_lens: torch.Tensor, seq_len: int, *, padding_value: int = -1000) -> torch.Tensor:
    """Build document ids from the packed-sequence collater's ``seq_lens``.

    Args:
        seq_lens: Tensor of shape [batch, packs] with per-pack token counts, using
            ``padding_value`` for unused pack slots.
        seq_len: Sequence length of the batch's token tensors.
        padding_value: Sentinel marking unused pack slots.

    Returns:
        Tensor of shape [batch, sequence] with 1-based document ids and 0 for the
        trailing positions not covered by any pack.
    """
    batch_size = seq_lens.shape[0]
    doc_ids = torch.zeros((batch_size, seq_len), dtype=torch.int32, device=seq_lens.device)
    for row in range(batch_size):
        lengths = seq_lens[row]
        lengths = lengths[lengths != padding_value]
        offset = 0
        for doc_index, length in enumerate(lengths.tolist()):
            if length <= 0:
                continue
            end = min(offset + length, seq_len)
            doc_ids[row, offset:end] = doc_index + 1
            offset = end
    return doc_ids


def doc_ids_from_cu_seqlens(cu_seqlens: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Build single-row document ids from cumulative sequence lengths.

    Args:
        cu_seqlens: Tensor of shape [segments + 1] with cumulative token counts.
            THD batches pad unused entries with a negative sentinel, which is
            dropped here.
        seq_len: Sequence length of the batch's token tensors.

    Returns:
        Tensor of shape [1, sequence] with 1-based document ids and 0 for the
        trailing positions not covered by ``cu_seqlens``.
    """
    boundaries = cu_seqlens.to(torch.long)
    boundaries = boundaries[boundaries >= 0]
    positions = torch.arange(seq_len, device=cu_seqlens.device)
    doc_ids = torch.bucketize(positions, boundaries[1:], right=True) + 1
    doc_ids = torch.where(positions < boundaries[-1], doc_ids, torch.zeros_like(doc_ids))
    return doc_ids.to(torch.int32).unsqueeze(0)


def segment_cu_seqlens(doc_ids_row: torch.Tensor) -> torch.Tensor:
    """Return segment boundaries for one row of document ids.

    Consecutive runs of the same id -- including runs of padding -- become their
    own segment so that the boundaries always tile the full row, which is what
    FLA's context-parallel partitioning expects.

    Args:
        doc_ids_row: Tensor of shape [sequence] with 1-based document ids.

    Returns:
        Tensor of shape [segments + 1] with cumulative segment lengths.
    """
    seq_len = doc_ids_row.shape[0]
    if seq_len == 0:
        return torch.zeros(1, dtype=torch.int32, device=doc_ids_row.device)
    changed = doc_ids_row[1:] != doc_ids_row[:-1]
    starts = torch.nonzero(changed, as_tuple=False).flatten() + 1
    zero = torch.zeros(1, dtype=starts.dtype, device=doc_ids_row.device)
    end = torch.full((1,), seq_len, dtype=starts.dtype, device=doc_ids_row.device)
    return torch.cat((zero, starts, end)).to(torch.int32)


def build_fla_cp_context(
    packed_context: KimiPackedContext,
    row: int,
    cp_group: Any,
    conv_kernel_size: int,
):
    """Build FLA's per-row context-parallel context for a KDA layer.

    Args:
        packed_context: Context describing the global document layout.
        row: Batch row the context is built for.
        cp_group: Context-parallel process group.
        conv_kernel_size: Short-convolution kernel size, used by FLA to exchange
            the conv boundary tokens between neighbouring ranks.

    Returns:
        The FLA ``FLACPContext`` for this row.
    """
    from fla.ops.cp import build_cp_context

    cu_seqlens, cu_seqlens_cpu = packed_context.row_cu_seqlens(row)
    return build_cp_context(
        cu_seqlens=cu_seqlens,
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )


class _AllGatherSequence(torch.autograd.Function):
    """Autograd-aware all-gather of equal-sized shards along the sequence axis."""

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: Any, dim: int) -> torch.Tensor:
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        world_size = dist.get_world_size(group)
        local_tensor = local_tensor.contiguous()
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor, group=group)
        ctx.group = group
        ctx.dim = dim
        ctx.rank = dist.get_rank(group)
        ctx.local_size = local_tensor.size(dim)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_full = grad_output.contiguous()
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_size
        return grad_full.narrow(ctx.dim, start, ctx.local_size).contiguous(), None, None


def all_gather_sequence(tensor: torch.Tensor, cp_group: Any, *, dim: int = 1) -> torch.Tensor:
    """All-gather a sequence-sharded tensor while keeping autograd connected.

    Args:
        tensor: Tensor of shape [..., local_sequence, ...] whose sequence axis is
            selected by ``dim``. Every rank must contribute the same shape.
        cp_group: Context-parallel process group.
        dim: Sequence axis.

    Returns:
        Tensor with the sequence axis expanded to the full global sequence.
    """
    return _AllGatherSequence.apply(tensor, cp_group, dim)


def build_document_causal_mask(
    q_doc_ids: torch.Tensor,
    kv_doc_ids: torch.Tensor,
    *,
    q_global_start: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the additive causal mask that also blocks cross-document attention.

    Args:
        q_doc_ids: Tensor of shape [batch, query_sequence] with 1-based document ids.
        kv_doc_ids: Tensor of shape [batch, key_sequence] with 1-based document ids.
        q_global_start: Global sequence offset of the first query token.
        dtype: Floating-point dtype used for the additive mask values.

    Returns:
        Additive mask tensor of shape [batch, 1, query_sequence, key_sequence].
    """
    device = q_doc_ids.device
    q_positions = torch.arange(q_doc_ids.shape[1], device=device) + q_global_start
    kv_positions = torch.arange(kv_doc_ids.shape[1], device=device)
    allowed = kv_positions[None, :] <= q_positions[:, None]
    allowed = allowed[None, :, :] & (q_doc_ids[:, :, None] == kv_doc_ids[:, None, :]) & (kv_doc_ids[:, None, :] > 0)
    # A padding query has no document to attend to; let it read position 0 so the
    # softmax stays finite. Its output is discarded by the loss mask.
    is_padding_query = (q_doc_ids <= _PAD_DOC_ID)[:, :, None]
    allowed = torch.where(is_padding_query, kv_positions[None, None, :] == 0, allowed)
    min_value = torch.finfo(dtype).min
    return torch.zeros(allowed.shape, dtype=dtype, device=device).masked_fill_(~allowed, min_value).unsqueeze(1)


_BLOCK_MASK_CACHE: dict[tuple, Any] = {}
_BLOCK_MASK_GENERATION: list[Any] = [None, None]


def _block_mask_cache_generation(doc_ids: torch.Tensor) -> None:
    """Drop cached block masks when a new batch (new document map) arrives."""
    pointer = doc_ids.data_ptr()
    if pointer != _BLOCK_MASK_GENERATION[0]:
        _BLOCK_MASK_CACHE.clear()
        _BLOCK_MASK_GENERATION[0] = pointer
        # Hold the tensor so the allocator cannot recycle the address mid-step.
        _BLOCK_MASK_GENERATION[1] = doc_ids


def _document_causal_block_mask(
    q_doc_ids: torch.Tensor,
    kv_doc_ids: torch.Tensor,
    *,
    q_global_start: int,
):
    """Build (and cache for the step) the FlexAttention document-causal block mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    _block_mask_cache_generation(q_doc_ids)
    key = (
        int(q_global_start),
        int(q_doc_ids.shape[0]),
        int(q_doc_ids.shape[1]),
        int(kv_doc_ids.shape[1]),
        q_doc_ids.device.type,
        q_doc_ids.device.index,
    )
    cached = _BLOCK_MASK_CACHE.get(key)
    if cached is not None:
        return cached

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        q_doc = q_doc_ids[batch_idx, q_idx]
        kv_doc = kv_doc_ids[batch_idx, kv_idx]
        allowed = (kv_idx <= q_idx + q_global_start) & (q_doc == kv_doc) & (kv_doc > _PAD_DOC_ID)
        return torch.where(q_doc <= _PAD_DOC_ID, kv_idx == 0, allowed)

    block_mask = create_block_mask(
        mask_mod,
        B=q_doc_ids.shape[0],
        H=None,
        Q_LEN=q_doc_ids.shape[1],
        KV_LEN=kv_doc_ids.shape[1],
        device=q_doc_ids.device,
    )
    if len(_BLOCK_MASK_CACHE) >= 64:
        _BLOCK_MASK_CACHE.pop(next(iter(_BLOCK_MASK_CACHE)))
    _BLOCK_MASK_CACHE[key] = block_mask
    return block_mask


_COMPILED_FLEX_ATTENTION: list[Any] = [None]


def _compiled_flex_attention():
    if _COMPILED_FLEX_ATTENTION[0] is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX_ATTENTION[0] = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION[0]


def document_causal_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    q_doc_ids: torch.Tensor,
    kv_doc_ids: torch.Tensor,
    q_global_start: int,
    scale: float,
) -> torch.Tensor:
    """Run causal, per-document attention of local queries against global keys.

    Args:
        query: Tensor of shape [batch, heads, query_sequence, qk_head_dim].
        key: Tensor of shape [batch, heads, key_sequence, qk_head_dim].
        value: Tensor of shape [batch, heads, key_sequence, v_head_dim].
        q_doc_ids: Tensor of shape [batch, query_sequence] with 1-based document ids.
        kv_doc_ids: Tensor of shape [batch, key_sequence] with 1-based document ids.
        q_global_start: Global sequence offset of the first query token.
        scale: Softmax scale applied to the query-key product.

    Returns:
        Tensor of shape [batch, heads, query_sequence, v_head_dim].
    """
    block_mask = _document_causal_block_mask(q_doc_ids, kv_doc_ids, q_global_start=q_global_start)

    qk_head_dim = query.shape[-1]
    v_head_dim = value.shape[-1]
    # FlexAttention kernels require power-of-two head dims; MLA's 192/128 split
    # is padded here and trimmed back off the output.
    padded_qk = 1 << (qk_head_dim - 1).bit_length()
    padded_v = 1 << (v_head_dim - 1).bit_length()
    if padded_qk != qk_head_dim:
        query = F.pad(query, (0, padded_qk - qk_head_dim))
        key = F.pad(key, (0, padded_qk - qk_head_dim))
    if padded_v != v_head_dim:
        value = F.pad(value, (0, padded_v - v_head_dim))

    output = _compiled_flex_attention()(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        block_mask=block_mask,
        scale=scale,
    )
    if padded_v != v_head_dim:
        output = output[..., :v_head_dim]
    return output.masked_fill((q_doc_ids <= _PAD_DOC_ID)[:, None, :, None], 0)


def _pad_sequence_dim(tensor: torch.Tensor, seq_dim: int, pad_len: int, value: float | int) -> torch.Tensor:
    if pad_len <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[seq_dim] = pad_len
    pad = torch.full(pad_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=seq_dim)


def _pad_position_ids(position_ids: torch.Tensor, seq_dim: int, pad_len: int) -> torch.Tensor:
    if pad_len <= 0:
        return position_ids
    last = position_ids.select(seq_dim, position_ids.shape[seq_dim] - 1).unsqueeze(seq_dim)
    increment_shape = [1] * position_ids.ndim
    increment_shape[seq_dim] = pad_len
    increments = torch.arange(1, pad_len + 1, device=position_ids.device, dtype=position_ids.dtype).view(
        increment_shape
    )
    return torch.cat((position_ids, last + increments), dim=seq_dim)


def _global_doc_ids_from_batch(batch: dict, seq_len: int, device: torch.device) -> torch.Tensor:
    """Resolve the global document-id map for a batch about to be CP-sharded."""
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None and attention_mask.ndim == 2:
        return doc_ids_from_attention_mask(attention_mask)

    seq_lens = batch.get("seq_lens_padded", batch.get("seq_lens"))
    if seq_lens is not None:
        return doc_ids_from_seq_lens(seq_lens, seq_len)

    batch_size = batch["input_ids"].shape[0]
    doc_ids = torch.ones((batch_size, seq_len), dtype=torch.int32, device=device)
    padding_mask = batch.get("padding_mask")
    if padding_mask is not None:
        doc_ids = doc_ids.masked_fill(padding_mask.bool(), _PAD_DOC_ID)
    return doc_ids


def shard_batch_for_kimi_cp(cp_mesh, tp_mesh, batch: dict, *, loss_mask=None, padding_token_id: int = 0):
    """Shard a batch contiguously across the context-parallel mesh for Kimi K3.

    Exposed through the :class:`ContextParallelSharder` returned by
    :meth:`KimiK3ForCausalLM.prepare_model_inputs_for_cp`. Every rank starts
    from the same full batch, keeps the ``[seq_start, seq_end)`` slice of each
    sequence-aligned tensor, and gets the (unsharded) global document-id map
    needed by the KDA and MLA layers.

    Args:
        cp_mesh: One-dimensional context-parallel mesh, or None.
        tp_mesh: Tensor-parallel mesh; unused, accepted for interface parity.
        batch: Batch mapping containing ``input_ids`` of shape [batch, sequence]
            plus ``labels`` and optional sequence-aligned tensors.
        loss_mask: Optional tensor of shape [batch, sequence] sharded with the labels.
        padding_token_id: Token id used when padding ``input_ids``.

    Returns:
        ``(context_factory, batch, layout)``; the context factory is a null
        context because transport lives in the Kimi K3 layers themselves.
    """
    del tp_mesh
    cp_size = 1 if cp_mesh is None else cp_mesh.size()
    input_ids = batch["input_ids"]
    seq_len = input_ids.shape[1]
    original_seq_len = seq_len
    device = input_ids.device

    doc_ids = _global_doc_ids_from_batch(batch, seq_len, device)
    # The 4D/indexed mask no longer matches the sharded sequence; document
    # boundaries travel through the document-id map instead.
    batch.pop("attention_mask", None)
    for key in ("seq_lens", "seq_lens_padded", "cu_seqlens", "cu_seqlens_padded", "max_seqlen", "qkv_format"):
        batch.pop(key, None)

    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(input_ids.shape[0], -1).contiguous()
        )
    elif batch["position_ids"].ndim == 2 and batch["position_ids"].shape[0] == 1 and input_ids.shape[0] > 1:
        batch["position_ids"] = batch["position_ids"].expand(input_ids.shape[0], -1).contiguous()

    pad_len = (-seq_len) % cp_size
    if pad_len:
        pad_values = {"input_ids": padding_token_id, "labels": -100, "padding_mask": True}
        for key, pad_value in pad_values.items():
            if key in batch:
                batch[key] = _pad_sequence_dim(batch[key], 1, pad_len, pad_value)
        batch["position_ids"] = _pad_position_ids(batch["position_ids"], 1, pad_len)
        doc_ids = _pad_sequence_dim(doc_ids, 1, pad_len, _PAD_DOC_ID)
        if loss_mask is not None:
            loss_mask = _pad_sequence_dim(loss_mask, 1, pad_len, 0)
        seq_len += pad_len

    # The MoE router needs to know which tokens are padding; the packed collater
    # encodes that only in the document map, so mirror it into ``padding_mask``.
    batch.setdefault("padding_mask", doc_ids <= _PAD_DOC_ID)

    seq_start = 0 if cp_mesh is None else cp_mesh.get_local_rank() * (seq_len // cp_size)
    # Keep CP metadata as ordinary batch fields until it reaches the model.
    # Pipeline schedules chunk tensor kwargs along batch dim but replicate
    # arbitrary dataclasses, so carrying KimiPackedContext itself would leave
    # every microbatch with the full batch's document map.
    batch["kimi_packed_doc_ids"] = doc_ids
    batch["kimi_packed_seq_start"] = seq_start
    batch["kimi_packed_cp_size"] = cp_size

    if cp_size > 1:
        local_seq_len = seq_len // cp_size
        seq_end = seq_start + local_seq_len
        for key in ("input_ids", "labels", "position_ids", "padding_mask"):
            if key in batch:
                batch[key] = batch[key][:, seq_start:seq_end].contiguous()
        if loss_mask is not None:
            batch["loss_mask"] = loss_mask[:, seq_start:seq_end].contiguous()
    elif loss_mask is not None:
        batch["loss_mask"] = loss_mask

    layout = ShardLayout(original_seq_len=original_seq_len, padded_seq_len=seq_len)
    return contextlib.nullcontext, batch, layout

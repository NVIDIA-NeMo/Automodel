# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.thd_utils import split_batch_into_thd_chunks


@contextlib.contextmanager
def cp_dispatcher_suspended(cp_mesh):
    """Temporarily suspend torch's ``context_parallel`` ring-SDPA monkeypatch.

    A model that embeds and sequence-shards its own primary stream in ``forward``
    (Megatron-style per-microbatch CP) runs any auxiliary non-CP attention -- a
    VLM vision tower over image patches -- inside the ring-SDPA
    ``context_parallel`` context while CP is active. That attention is
    bidirectional and unsharded, so torch's load-balanced ring SDPA all-gathers
    its Q/K/V and rejects it with "Load balancing requires ``is_causal=True``".
    The legacy buffer-API ``context_parallel`` installs the ring by monkeypatching
    ``F.scaled_dot_product_attention``; this restores the original SDPA for the
    wrapped vision forward, then re-installs the ring for the sharded text decoder.

    Args:
        cp_mesh: The context-parallel submesh. A no-op when None or size <= 1, so
            callers pass the same mesh they shard with; re-enabling needs it.
    """
    if cp_mesh is None or cp_mesh.size() <= 1:
        yield
        return
    import torch.nn.functional as F_module

    current_sdpa = F_module.scaled_dot_product_attention
    dispatcher_was_enabled = (
        getattr(current_sdpa, "__module__", None)
        == "torch.distributed.tensor.experimental._context_parallel._attention"
        and getattr(current_sdpa, "__name__", None) == "inner_fn"
    )
    if not dispatcher_was_enabled:
        yield
        return

    # torch-internal: the legacy context_parallel enables its SDPA monkeypatch via
    # these _impl toggles with the attention seq dim (2 for [B, heads, seq, dim]).
    from torch.distributed.tensor.experimental._context_parallel._attention import (  # noqa: PLC0415
        _disable_context_parallel_dispatcher_impl,
        _enable_context_parallel_dispatcher_impl,
    )

    _disable_context_parallel_dispatcher_impl()
    try:
        yield
    finally:
        _enable_context_parallel_dispatcher_impl(seq_dim=2, mesh=cp_mesh)


def attach_context_parallel_hooks(model: torch.nn.Module):
    """Attach forward pre-hooks to self_attn modules to fix attention masks for context parallelism.

    Context parallelism shards Q/K/V on the sequence dimension as DTensors,
    so explicit 4D attention masks would have mismatched shapes.  This function
    registers a hook on every ``self_attn`` sub-module that strips the
    ``attention_mask`` kwarg and sets ``is_causal=True`` instead, letting
    SDPA handle causal masking internally.

    Based on ``accelerate.big_modeling._attach_context_parallel_hooks``.
    """

    def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            module.register_forward_pre_hook(_self_attn_pre_forward_hook, with_kwargs=True, prepend=True)


def attach_cp_sdpa_hooks(model: torch.nn.Module, cp_mesh) -> None:
    """Inject CP-aware SDPA into self_attn modules for compile + CP>1 correctness.

    Problem: when per-layer torch.compile is active, Dynamo traces through the decoder
    layer including Q/K/V projections.  At the F.scaled_dot_product_attention call site,
    Q/K/V are already local tensors (DTensor metadata was never propagated through the
    compiled graph).  The DTensor SDPA dispatch — which triggers the CP allgather — never
    fires, so each rank silently attends only to its local sequence shard.

    Fix: swap F.scaled_dot_product_attention with a @torch._dynamo.disable wrapper for
    the duration of each self_attn forward.  Dynamo sees the disabled function and creates
    a graph break there, so:
      - Everything before (Q/K/V proj + RoPE) is compiled and fused.
      - The disabled wrapper runs eagerly: re-wraps local Q/K/V as DTensors with
        Shard(2) on the CP mesh so the DTensor SDPA dispatch fires the allgather.
      - Everything after (O proj + residual + MLP) is compiled and fused.

    Seq dim at the SDPA call is 2: tensors are [B, nH, S/cp_size, D] after HF reshape.
    """
    import torch.nn.functional as F_module
    from torch.distributed.tensor import DTensor, Shard

    _original_sdpa = F_module.scaled_dot_product_attention

    @torch._dynamo.disable
    def _cp_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        # Re-wrap local Q/K/V as DTensors so DTensor SDPA dispatch fires the CP allgather.
        # Seq dim is 2: [B, nH, S/cp_size, D].
        if not isinstance(query, DTensor):
            query = DTensor.from_local(query, device_mesh=cp_mesh, placements=[Shard(2)])
            key = DTensor.from_local(key, device_mesh=cp_mesh, placements=[Shard(2)])
            value = DTensor.from_local(value, device_mesh=cp_mesh, placements=[Shard(2)])
        out = _original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )
        # Unwrap back to local tensor for the compiled O-proj + MLP region.
        return out.to_local() if isinstance(out, DTensor) else out

    def _pre_hook(module, args, kwargs):
        F_module.scaled_dot_product_attention = _cp_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        F_module.scaled_dot_product_attention = _original_sdpa

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            # Hook on the inner attention module so the hook fires during both
            # the original forward AND gradient-checkpointing recompute.
            # CheckpointWrapper's recompute bypasses __call__ (and thus pre-hooks
            # on the wrapper itself), so we must hook on the wrapped module directly.
            target = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
            target.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            # always_call=True ensures _original_sdpa is restored even if the forward raises.
            target.register_forward_hook(_post_hook, always_call=True)


def unshard_context_parallel_tensor(
    cp_mesh: DeviceMesh,
    tensor: torch.Tensor,
    *,
    seq_dim: int,
) -> torch.Tensor:
    """Restore a tensor from PyTorch's load-balanced context-parallel layout.

    Args:
        cp_mesh: One-dimensional context-parallel mesh of size ``C``.
        tensor: Tensor of shape ``[..., local_sequence, ...]`` whose sequence
            axis is selected by ``seq_dim`` and uses PyTorch's load-balanced CP
            layout.
        seq_dim: Axis containing the local sequence extent.

    Returns:
        Replicated tensor of shape ``[..., sequence, ...]`` with the same axis
        order and full sequence extent on ``seq_dim``.
    """
    if cp_mesh.size() <= 1:
        return tensor
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard

    (unsharded,) = context_parallel_unshard(cp_mesh, [tensor], seq_dims=[seq_dim])
    return unsharded


def make_cp_batch_for_te(
    cp_mesh,
    batch,
    qkv_format="thd",
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
    return_local_indices: bool = False,
):
    """
    Build a CP batch for Transformer Engine using THD format.

    This function converts BSHD format batches to THD format and shards them across
    context parallel ranks for use with Transformer Engine. It processes the batch
    in chunks if num_chunks > 1, allowing for better memory efficiency with large
    sequences.

    The function performs three main steps:
    1. Converts BSHD format to THD format using split_batch_into_thd_chunks
    2. Optionally splits the batch into multiple chunks for memory efficiency
    3. Shards each chunk across CP ranks using Transformer Engine's partitioning

    Args:
        cp_mesh (DeviceMesh or None): The device mesh for context parallel. If None or
            size <= 1, returns the batch in THD format without sharding.
        batch (Dict[str, torch.Tensor]): The input batch in BSHD format containing:
            - input_ids: Input token IDs [batch_size, seq_len] or [batch_size, seq_len, hidden_dim]
            - labels: Label token IDs [batch_size, seq_len]
            - position_ids (optional): Position IDs [batch_size, seq_len]
            - seq_lens: Actual sequence lengths [batch_size, num_packs]
            - seq_lens_padded: Padded sequence lengths [batch_size, num_packs]
        qkv_format (str): Format for QKV tensors. Currently only "thd" is supported.
        padding_token_id (int): Token ID used for padding in input_ids (default: 0)
        num_chunks (int): Number of chunks to split the batch into. If > 1, the batch
            dimension is split and each chunk is processed separately (default: 1)
        seq_lens_padding_value (int): Sentinel value used to indicate padding in
            seq_lens/seq_lens_padded tensors (default: -1000)
        return_local_indices (bool): Also return this rank's local-token global
            index map (the ``thd_get_partitioned_indices`` partition; an
            identity arange when CP is inactive; None in chunked mode, where
            each chunk is its own token space). Used by the THD ContextParallelismSharder's
            token verbs.

    Returns:
        dict: Processed batch in THD format (or ``(dict, LongTensor | None)``
        when ``return_local_indices``) with the following keys:
            - input_ids: Sharded input token IDs [total_tokens] or [num_chunks, chunk_tokens]
            - labels: Sharded labels [total_tokens] or [num_chunks, chunk_tokens]
            - position_ids: Generated and sharded position IDs [total_tokens] or [num_chunks, chunk_tokens]
            - cu_seqlens: Cumulative sequence lengths [num_seqs+1] or [num_chunks, max_seqs+1]
            - cu_seqlens_padded: Cumulative padded sequence lengths [num_seqs+1] or [num_chunks, max_seqs+1]
            - max_seqlen: Maximum sequence length (int32 tensor)
            - qkv_format: Format string ("thd")
            - padding_mask: Boolean mask indicating padding tokens

    Raises:
        ValueError: If qkv_format is not "thd"
        KeyError: If required fields (seq_lens, seq_lens_padded) are missing from batch

    Example:
        >>> # Single chunk, no CP
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 4]]),
        ...     'labels': torch.tensor([[2, 3, 4, 5]]),
        ...     'seq_lens': torch.tensor([[4]]),
        ...     'seq_lens_padded': torch.tensor([[4]])
        ... }
        >>> result = make_cp_batch_for_te(None, batch)
        >>> result['input_ids'].shape  # [4] in THD format
        torch.Size([4])

        >>> # Multiple chunks with CP
        >>> batch = {
        ...     'input_ids': torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        ...     'labels': torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]]),
        ...     'seq_lens': torch.tensor([[4], [4]]),
        ...     'seq_lens_padded': torch.tensor([[4], [4]])
        ... }
        >>> result = make_cp_batch_for_te(cp_mesh, batch, num_chunks=2)
        >>> result['input_ids'].shape  # [2, chunk_tokens] - 2 chunks
        torch.Size([2, 2])  # Example: 2 chunks, 2 tokens each after sharding
    """
    if qkv_format != "thd":
        raise ValueError(f"Currently only 'thd' format is supported, got: {qkv_format}")

    batch = split_batch_into_thd_chunks(
        batch, num_chunks=num_chunks, seq_lens_padding_value=seq_lens_padding_value, padding_token_id=padding_token_id
    )

    if cp_mesh is None or cp_mesh.size() <= 1:
        if not return_local_indices:
            return batch
        # Unsharded THD stream: identity index map. Chunked streams are
        # per-chunk token spaces with no single step-wide map -> None.
        input_ids = batch["input_ids"]
        local_indices = (
            torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.long) if num_chunks <= 1 else None
        )
        return batch, local_indices

    if num_chunks <= 1:
        sharded, local_indices = _shard_thd_chunk_for_te(
            batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id
        )
        return (sharded, local_indices) if return_local_indices else sharded

    # Extract each chunk from the batched result and shard it
    chunks = []
    for i in range(num_chunks):
        chunk_batch = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        chunks.append(
            _shard_thd_chunk_for_te(chunk_batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)[0]
        )

    return_dict = {
        "input_ids": torch.stack([chunk["input_ids"] for chunk in chunks]),
        "labels": torch.stack([chunk["labels"] for chunk in chunks]),
        "position_ids": torch.stack([chunk["position_ids"] for chunk in chunks]),
        "cu_seqlens": torch.stack([chunk["cu_seqlens"] for chunk in chunks]),
        "max_seqlen": torch.stack([chunk["max_seqlen"] for chunk in chunks]),
        "qkv_format": qkv_format,
        "padding_mask": torch.stack([chunk["padding_mask"] for chunk in chunks]),
        "cp_size": cp_mesh.size() if cp_mesh is not None else 1,
        "cp_rank": torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0,
    }

    # Chunked mode: each chunk is its own token space, so there is no single
    # step-wide local-token index map to expose.
    return (return_dict, None) if return_local_indices else return_dict


def _shard_thd_chunk_for_te(
    batch,
    cp_mesh,
    qkv_format,
    seq_lens_padding_value,
    padding_token_id,
):
    import transformer_engine_torch as tex

    cu_seqlens = batch.get("cu_seqlens", None)
    cu_seqlens_padded = batch.get("cu_seqlens_padded", batch["cu_seqlens"])
    filtered_cu_seqlens_padded = cu_seqlens_padded[cu_seqlens_padded != seq_lens_padding_value]

    # Check for required fields - BSHD format is not supported
    if cu_seqlens is None or cu_seqlens_padded is None:
        raise ValueError(
            "BSHD format is not supported. Both 'cu_seqlens' and 'cu_seqlens_padded' must be present in the batch. "
            "Please use packed sequence format with cu_seqlens and cu_seqlens_padded."
        )

    cp_size = cp_mesh.size()

    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group()) if cp_mesh is not None else 0

    # The partition is the same for every token-aligned key; it is also this
    # rank's local-token global index map, returned so the caller can install
    # it on the THD sharder (ContextParallelismSharder token verbs).
    local_indices = tex.thd_get_partitioned_indices(
        filtered_cu_seqlens_padded, batch["input_ids"].size(0), cp_size, cp_rank
    )
    mask_keys = ["input_ids", "labels", "position_ids", "padding_mask"]
    for key in mask_keys:
        if key in batch:
            batch[key] = batch[key].index_select(0, local_indices)

    max_seqlen = (filtered_cu_seqlens_padded[1:] - filtered_cu_seqlens_padded[:-1]).max().item()
    output_batch = {
        "input_ids": batch["input_ids"].to(torch.int64).contiguous(),
        "labels": batch["labels"].to(torch.int64).contiguous(),
        "position_ids": batch["position_ids"].to(torch.int64).contiguous(),
        "cu_seqlens": cu_seqlens_padded.to(torch.int32).contiguous(),
        "max_seqlen": torch.tensor(max_seqlen).to(torch.int32).to(device=cu_seqlens_padded.device),
        "qkv_format": qkv_format,
        "padding_mask": (batch["input_ids"] == padding_token_id).bool().contiguous(),
        "cp_size": cp_size,
        "cp_rank": cp_rank,
    }

    return output_batch, local_indices

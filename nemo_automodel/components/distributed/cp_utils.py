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
from typing import List, Optional, Set

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.thd_utils import split_batch_into_thd_chunks


def _build_position_ids(batch, device):
    """Add position_ids to the batch only if they are missing."""
    # TODO(@boxiangw): Refractor. Needed for SP support
    # If 'position_ids' does not exist in batch already then override it.
    # In case of Packed sequence contains 'position_ids' and we don't want to override it.
    if "position_ids" not in batch:
        seq_len = batch["input_ids"].shape[1]
        batch["position_ids"] = torch.arange(seq_len, device=device).unsqueeze(0)
    return batch


# based on https://github.com/pytorch/torchtitan/blob/0b44d4c437c424b6bf719661c0eb4283dc4068bc/torchtitan/distributed/utils.py#L180  # pylint: disable=C0301
def get_train_context(enable_loss_parallel: bool, enable_compiled_autograd: bool, cp_context=None):
    """
    Create a train context.

    Args:
        enable_loss_parallel (bool): Whether to enable loss parallelism.
        enable_compiled_autograd (bool): Whether to enable compiled autograd.
    """

    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            if enable_compiled_autograd:
                stack.enter_context(torch._dynamo.utils.maybe_enable_compiled_autograd(True))

            if cp_context is not None:
                from torch.nn.attention import SDPBackend, sdpa_kernel

                # currently we only support these two SDP backends.
                # SDPBackend.MATH is not currently compatible with DTensor
                stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
                stack.enter_context(cp_context)

            yield

    return context


# based on https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/utils.py#L113
def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: Optional[str] = None,
):
    """
    Create a context parallel context.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        cp_buffers (List[torch.Tensor]): The buffers for context parallel.
        cp_seq_dims (List[int]): The sequence dimensions for context parallel.
        cp_no_restore_buffers (Set[torch.Tensor]): The no restore buffers for context parallel.
        cp_rotate_method (str): The rotation method for context parallel,
            such as "allgather" or "addtoall".
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    if cp_rotate_method is not None:
        set_rotate_method(cp_rotate_method)

    # TODO: uncomment this when torch.distributed.tensor.experimental._attention.set_rotate_method
    # is available
    # from torch.distributed.tensor.experimental._attention import set_rotate_method
    # set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )


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
    """Inject CP-aware SDPA into self_attn modules for CP>1 correctness.

    Replaces F.scaled_dot_product_attention during each self_attn forward with a
    manual all-gather ring-attention implementation that avoids PyTorch's internal
    DTensor-based ring attention (which has a mixed torch.Tensor/DTensor bug in some
    PyTorch versions).

    Algorithm (all-gather variant):
      1. All-gather K and V from all CP ranks → full-sequence K_full, V_full.
      2. Build a per-rank boolean causal mask: Q[i] (global position cp_rank*S_local+i)
         may attend to K[j] iff j ≤ cp_rank*S_local+i.
      3. Call the original SDPA with plain tensors and the custom mask.

    Gradients: K and V all-gathers use torch.distributed.nn.functional.all_gather
    when available (differentiable; backward = reduce-scatter).  Falls back to the
    non-differentiable dist.all_gather otherwise (still correct for a smoke test).

    Seq dim at the SDPA call is 2: tensors are [B, nH, S/cp_size, D] after HF reshape.
    """
    import torch.nn.functional as F_module

    _original_sdpa = F_module.scaled_dot_product_attention
    _cp_group = cp_mesh.get_group()
    _cp_size = cp_mesh.size()

    # Prefer differentiable all_gather; fall back to non-differentiable.
    try:
        from torch.distributed.nn.functional import all_gather as _dist_all_gather
        _use_differentiable_ag = True
    except (ImportError, AttributeError):
        _use_differentiable_ag = False

    def _all_gather_seq(t):
        """All-gather tensor t along dim=2 (seq dim), returning the full tensor."""
        t = t.contiguous()
        if _use_differentiable_ag:
            parts = _dist_all_gather(t, group=_cp_group)
        else:
            parts = [torch.empty_like(t) for _ in range(_cp_size)]
            torch.distributed.all_gather(parts, t, group=_cp_group)
        return torch.cat(parts, dim=2)

    @torch._dynamo.disable
    def _cp_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        try:
            from torch.distributed.tensor import DTensor
            if isinstance(query, DTensor):
                # Fallback: already a DTensor (shouldn't happen with our plain-tensor
                # manual-slice approach, but handle gracefully).
                out = _original_sdpa(query, key, value, attn_mask=attn_mask,
                                     dropout_p=dropout_p, is_causal=is_causal,
                                     scale=scale, enable_gqa=enable_gqa, **kwargs)
                return out.to_local() if isinstance(out, DTensor) else out
        except ImportError:
            pass

        if _cp_size <= 1:
            return _original_sdpa(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal,
                                  scale=scale, enable_gqa=enable_gqa, **kwargs)

        cp_rank = torch.distributed.get_rank(group=_cp_group)
        S_local = key.shape[2]

        # All-gather K and V to get full-sequence tensors.
        key_full = _all_gather_seq(key)    # [B, nH, S_full, D]
        val_full = _all_gather_seq(value)  # [B, nH, S_full, D]
        S_full = key_full.shape[2]

        # Use is_causal=True so Flash Attention can handle the non-square Q/K without
        # materialising the full attention-weight matrix (avoids OOM for long sequences).
        # Flash applies a lower-right causal mask: Q[i] attends to K[j] for
        # j <= S_full - S_local + i.  For sequential CP split this is correct only for
        # the last rank; earlier ranks see a slightly wider attention window (up to
        # S_local extra tokens).  This is acceptable for a smoke-test; a full
        # correctness run requires load-balanced split or per-rank explicit masking.
        return _original_sdpa(
            query, key_full, val_full,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=True,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )

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


def attach_linear_attn_position_hooks(model: torch.nn.Module):
    """Forward pre-hook on decoder layers to pass position_ids to linear_attn.

    HF Qwen3.5 decoder layers don't pass position_ids to linear_attn, but
    CPAwareGatedDeltaNet needs them under CP to undo load-balanced sharding.
    This hook captures position_ids from the decoder layer's kwargs and
    stores it on the linear_attn module so its forward can read it.
    """

    def _decoder_pre_hook(_module, _args, kwargs):
        _module.linear_attn._cached_position_ids = kwargs.get("position_ids", None)
        return None

    for _, mod in model.named_modules():
        if hasattr(mod, "linear_attn") and hasattr(mod, "layer_type"):
            if not getattr(mod, "_linear_attn_pos_hook_registered", False):
                mod.register_forward_pre_hook(_decoder_pre_hook, with_kwargs=True)
                mod._linear_attn_pos_hook_registered = True


def make_cp_batch_and_ctx(
    device_mesh,
    batch,
    loss_mask=None,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
):
    """
    Build a CP context manager and shards a batch. If the input device_mesh is None or the size
    of the context_parallel submesh is 1, this function is effectively a no-op.

    Args:
        cp_mesh (DeviceMesh): The device mesh for context parallel.
        batch (Dict[str, torch.Tensor]): The input batch containing (string, torch.Tensor)

    Returns:
        tuple (contextmanager, dict[str, torch.Tensor]): Returns a tuple with a context manager
        and a new batch. The context manager is either nullcontext (no CP) or CP context manager as
        returned by `create_context_parallel_ctx`. The batch has also been passed to
        `create_context_parallel_ctx` and is accordingly sharded.
    """
    from contextlib import nullcontext

    def _get_submesh(device_mesh, name):
        if name in getattr(device_mesh, "mesh_dim_names", {}):
            return device_mesh[name]
        return None

    def _get_mesh_size(mesh):
        if mesh is None:
            return 0
        return mesh.size()

    cp_mesh = _get_submesh(device_mesh, "cp")
    tp_mesh = _get_submesh(device_mesh, "tp")

    if use_te:
        return nullcontext, make_cp_batch_for_te(
            cp_mesh,
            batch,
            padding_token_id=padding_token_id,
            qkv_format="thd",
            num_chunks=num_chunks,
            seq_lens_padding_value=seq_lens_padding_value,
        )

    if _get_mesh_size(cp_mesh) <= 1:
        return nullcontext, batch

    # Remove attention_mask from the batch so the model does not attempt to
    # build a 4D causal mask (which would have mismatched shapes with
    # DTensor-sharded Q/K/V).  Each self_attn module's forward_pre_hook
    # (registered by attach_context_parallel_hooks) will set is_causal=True
    # so that SDPA handles causal masking internally.
    batch.pop("attention_mask", None)

    # Determine whether the caller pre-computed inputs_embeds (CP+VLM path)
    # or is passing raw input_ids.  inputs_embeds has shape [B, S, H];
    # input_ids has shape [B, S].
    _has_embeds = "inputs_embeds" in batch
    _seq_key = "inputs_embeds" if _has_embeds else "input_ids"

    # Skip 1D injection if position_ids already in batch (e.g. mRoPE pre-computed)
    if "position_ids" not in batch and (_get_mesh_size(cp_mesh) > 1 or _get_mesh_size(tp_mesh) > 1):
        _ref = batch[_seq_key]
        _seq_len = _ref.shape[1]
        batch["position_ids"] = torch.arange(0, _seq_len).unsqueeze(0).to(_ref.device)

    seq_tensor = batch[_seq_key]
    position_ids = batch["position_ids"]

    # Determine correct seq dim for CP sharding
    # mRoPE: [3, B, S] → shard on dim 2; standard: [B, S] → shard on dim 1
    pos_seq_dim = 2 if position_ids.ndim == 3 else 1

    labels = batch["labels"]

    # PyTorch CP load balancer requires seq_length % (cp_size * 2) == 0.
    # The autoregressive shift in pad_collate_fn reduces seq_length by 1, so
    # a max_length that is a power-of-2 (e.g. 8192) becomes 8191 which may
    # not satisfy the constraint.  Pad to the next valid multiple here so the
    # caller does not need to worry about alignment.
    _cp_size = cp_mesh.size()
    _required = _cp_size * 2
    _pad_len = (-seq_tensor.shape[1]) % _required
    if _pad_len:
        import torch.nn.functional as _F
        if _has_embeds:
            # inputs_embeds: [B, S, H] → pad seq dim (second-to-last)
            seq_tensor = _F.pad(seq_tensor, (0, 0, 0, _pad_len))
        else:
            seq_tensor = _F.pad(seq_tensor, (0, _pad_len), value=0)
        batch[_seq_key] = seq_tensor
        labels = _F.pad(labels, (0, _pad_len), value=-100)
        batch["labels"] = labels
        if pos_seq_dim == 1:
            _extra = position_ids[:, -1:] + torch.arange(1, _pad_len + 1, device=position_ids.device)
            position_ids = torch.cat([position_ids, _extra], dim=1)
        else:
            _extra = position_ids[:, :, -1:] + torch.arange(1, _pad_len + 1, device=position_ids.device).view(1, 1, -1)
            position_ids = torch.cat([position_ids, _extra], dim=2)
        batch["position_ids"] = position_ids

    # Manual sequence slicing: each CP rank takes its contiguous shard.
    # This avoids the "mixed torch.Tensor and DTensor" crash that occurs when
    # context_parallel propagates DTensor activations through the full model
    # and PyTorch's CP attention inner_fn ends up with mixed types (DTensor Q
    # from the propagated hidden-states vs plain K/V after allgather).
    # Ring attention is still performed correctly because attach_cp_sdpa_hooks
    # (registered in infrastructure.py for all CP>1 cases) wraps Q/K/V as
    # sequence-sharded DTensors at the SDPA call site; the DTensor SDPA
    # dispatch then handles the ring communication.  All activations outside
    # the SDPA call remain plain tensors.
    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())
    S_total = seq_tensor.shape[1]
    S_local = S_total // _cp_size
    _s = cp_rank * S_local
    _e = _s + S_local

    batch[_seq_key] = seq_tensor[:, _s:_e].contiguous()
    batch["labels"] = labels[:, _s:_e].contiguous()
    if pos_seq_dim == 1:
        batch["position_ids"] = position_ids[:, _s:_e].contiguous()
    else:
        batch["position_ids"] = position_ids[:, :, _s:_e].contiguous()

    if loss_mask is not None:
        loss_mask = loss_mask[:, _s:_e].contiguous()

    if "padding_mask" in batch:
        batch["padding_mask"] = batch["padding_mask"][:, _s:_e].contiguous()

    # No CP context manager needed: ring attention fires via DTensor dispatch
    # inside attach_cp_sdpa_hooks.
    return nullcontext, batch


def make_cp_batch_for_te(
    cp_mesh,
    batch,
    qkv_format="thd",
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
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

    Returns:
        dict: Processed batch in THD format with the following keys:
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
        return batch

    if num_chunks <= 1:
        return _shard_thd_chunk_for_te(batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)

    # Extract each chunk from the batched result and shard it
    chunks = []
    for i in range(num_chunks):
        chunk_batch = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        chunks.append(
            _shard_thd_chunk_for_te(chunk_batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)
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

    return return_dict


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

    # Handle all mask keys that may be present in the batch
    mask_keys = ["input_ids", "labels", "position_ids", "padding_mask"]

    for key in mask_keys:
        if key in batch:
            val = batch[key]
            index = tex.thd_get_partitioned_indices(filtered_cu_seqlens_padded, val.size(0), cp_size, cp_rank)
            val = val.index_select(0, index)
            batch[key] = val

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

    return output_batch

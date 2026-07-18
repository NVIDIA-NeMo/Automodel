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
from functools import partial
from typing import List, Optional, Set

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.cp_sharder import (
    ContextParallelismSharder,
    ShardLayout,
    identity_local_indices,
    round_robin_local_indices,
    shard_batch_identity,
    shard_batch_load_balanced,
)
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


def _shard_grad_buffer_for_cp(buffer: torch.Tensor, seq_dim: int, cp_mesh: DeviceMesh) -> torch.Tensor:
    """Shard a gradient-bearing buffer with CP's head-tail load-balancing order."""
    cp_size = cp_mesh.size()
    num_chunks = 2 * cp_size
    seq_len = buffer.shape[seq_dim]
    if seq_len % num_chunks != 0:
        raise ValueError(f"CP sequence length {seq_len} must be divisible by {num_chunks}")

    chunk_size = seq_len // num_chunks
    # ``cp_mesh`` is the 1D CP submesh selected from the full device mesh, so
    # this is the rank within the CP process group even when the root mesh also
    # has HSDP replicate/shard dimensions.
    cp_rank = cp_mesh.get_local_rank()
    head_chunk = buffer.narrow(seq_dim, cp_rank * chunk_size, chunk_size)
    tail_chunk = buffer.narrow(seq_dim, (num_chunks - cp_rank - 1) * chunk_size, chunk_size)
    return torch.cat((head_chunk, tail_chunk), dim=seq_dim)


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


def _get_submesh(device_mesh, dim: str):
    """The named submesh, or None when the mesh or dimension is absent."""
    if device_mesh is None or dim not in getattr(device_mesh, "mesh_dim_names", ()):
        return None
    return device_mesh[dim]


def _mesh_dim_size(device_mesh, dim: str) -> int:
    """Size of a named submesh, 0 when the mesh or dimension is absent."""
    submesh = _get_submesh(device_mesh, dim)
    return submesh.size() if submesh is not None else 0


def prepare_cp_forward(
    model,
    device_mesh,
    batch,
    *,
    magi=None,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    loss_mask=None,
    invoke_pre_embed: bool = True,
    extra_seq_buffers: Optional[dict[str, int]] = None,
):
    """Single CP dispatch for a training/eval forward: hook -> sharder -> (ctx, batch).

    Collapses the per-recipe CP branching into one call: the model hook may
    return a ContextParallelismSharder, ``_make_cp_batch_and_ctx`` resolves it against the
    framework-owned sharders (magi / TE / generic torch ``context_parallel``)
    and calls ``shard_batch``. When CP is active and the model exposes
    ``prepare_model_inputs_for_cp``, the hook is invoked uniformly through
    ``model.__call__(_pre_embed_only=True, ...)`` so FSDP2 pre-forward hooks
    unshard e.g. vision towers during the pre-embed.

    Args:
        model: The (first) model part, or None (e.g. no-model contexts).
        device_mesh: The full device mesh (``cp``/``tp`` submeshes are read).
        batch: The full-sequence batch; mutated and sharded in place.
        magi: Optional recipe MagiState, threaded to ``_make_cp_batch_and_ctx``
            where it occupies the same dispatch rung as the TE path. Its
            recipe domain is bound at ``setup_magi``; for llm-domain magi the
            model hook is skipped (mirrors the recipes' historical branching),
            while vlm-domain magi still runs the pre-embed first (vision stays
            on SDPA under magi).
        use_te: THD-packed collator is active (TE/THD sharding; also magi's
            ``is_thd``).
        padding_token_id: Pad sentinel for ``input_ids``.
        num_chunks: THD chunk count, forwarded to the hook and TE sharding.
        loss_mask: Optional per-token mask forwarded to the batch sharding.
        invoke_pre_embed: Invoke the model hook when CP is active or the model
            owns native THD preparation. Recipes pass False for PP stages
            without embeddings and for KD paths that never wired model-owned CP.
        extra_seq_buffers: Additional batch keys mapped to their sequence axis
            (e.g. ``{"teacher_logits": 1}``), padded and sharded alongside the
            batch on the generic torch path (rejected on the TE THD path;
            ignored by backends that own their transport).
    Returns:
        ``(ctx_factory, batch, sharder)`` — the resolved :class:`ContextParallelismSharder`
        (the identity sharder when no CP prep applies), whose token
        verbs keep per-token tensors aligned with the sharded inputs.
    """

    magi_enabled = magi is not None and getattr(magi, "enabled", False)
    cp_sharder = None
    has_hook = model is not None and hasattr(model, "prepare_model_inputs_for_cp")
    effective_cp_size = _mesh_dim_size(device_mesh, "cp")

    # llm-domain magi replaces the whole batch prep (no model has both a CP
    # hook and magi); vlm-domain magi composes with the vision pre-embed.
    magi_replaces_hook = magi_enabled and getattr(magi, "domain", "llm") == "llm"
    model_owns_thd = use_te and bool(getattr(model, "supports_thd", False))
    if (effective_cp_size > 1 or model_owns_thd) and has_hook and not magi_replaces_hook and invoke_pre_embed:
        # The whole batch rides through __call__ as an opaque kwarg; the model
        # reads the keys it needs and removes the raw inputs it consumed (its
        # returned entries are merged on top). Sharder-only hooks (DSV4/GLM)
        # consume nothing; their batch — including ``input_ids`` — stays intact.
        prepared = model(_pre_embed_only=True, _cp_batch=batch, num_chunks=num_chunks)
        cp_sharder = prepared.pop("cp_sharder", None)
        # Merge the hook's return: a None value marks a raw input the hook
        # consumed (e.g. into inputs_embeds) — remove it so it cannot reach
        # the sharded forward. The return channel is used (not in-place pops)
        # because FSDP2's forward-kwargs cast can hand the hook a copy of the
        # batch dict. The sharder itself is passed onward as an explicit
        # parameter, never through the batch.
        for key, value in prepared.items():
            if value is None:
                batch.pop(key, None)
            else:
                batch[key] = value

    return _make_cp_batch_and_ctx(
        device_mesh,
        batch,
        loss_mask,
        use_te=use_te,
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        magi=magi,
        model=model,
        cp_sharder=cp_sharder,
        extra_seq_buffers=extra_seq_buffers,
    )


def _resolve_cp_sharder(
    cp_mesh,
    model_sharder: Optional[ContextParallelismSharder],
    *,
    magi,
    use_te: bool,
    num_chunks: int,
    seq_lens_padding_value: int,
    model,
    extra_seq_buffers: Optional[dict[str, int]] = None,
) -> ContextParallelismSharder:
    """Resolve the ContextParallelismSharder for this forward: model-owned > magi > TE > generic > none.

    Always returns a sharder: when no CP prep applies, an identity sharder,
    so callers hold working token verbs at every cp_size and
    need no branches. The generic torch ``context_parallel`` path only shards
    at cp_size > 1; model-owned, magi, and TE sharders may also run at
    cp_size <= 1 for native THD packing conversion / mask-spec activation. The
    magi and THD token layouts depend on batch content (``cu_seqlens``
    partitioning / dispatch solver), not just
    ``(cp_mesh, seq_len)``, so their sharders construct without
    ``local_token_global_indices`` and install the index map computed during
    ``shard_batch`` (token verbs raise before the first shard).
    """
    cp_active = cp_mesh is not None and cp_mesh.size() > 1

    # A model that owns its CP attention returns a ContextParallelismSharder from its CP
    # input-prep hook. Honor it instead of any framework-owned path so the
    # implementation stays with the model.
    if model_sharder is not None:
        return model_sharder

    if magi is not None and getattr(magi, "enabled", False):
        # Backend-owned prep (MagiAttention): magi manages its own CP transport,
        # so like the TE path shard_batch returns (nullcontext, prepped_batch).
        # All magi internals (HF-vs-custom, recipe domain, cp group) stay in
        # magi_attn_utils. The dispatch-solver partition is data-dependent, so
        # shard_batch installs the index map it just computed (magi's
        # get_position_ids) on the sharder for the token verbs.
        def _shard_batch_magi(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
            input_ids = batch.get("input_ids")
            row_shape = tuple(input_ids.shape[:2]) if input_ids is not None and input_ids.dim() >= 2 else None
            prepped, local_indices = magi.make_cp_batch(
                cp_mesh,
                batch,
                padding_token_id=padding_token_id,
                num_chunks=num_chunks,
                is_thd=use_te,
                model=model,
                return_local_indices=True,
            )
            facts = None
            if local_indices is not None:
                padded = local_indices.numel() * max(getattr(magi, "cp_size", 1) or 1, 1)
                original, in_rows = None, None
                if row_shape is not None:
                    if padded == row_shape[0] * row_shape[1]:
                        # Flatten moved no tokens and dispatch added no pad: the
                        # pre-flatten rows are the caller's coordinate system.
                        in_rows = row_shape
                    elif row_shape[0] == 1 and padded >= row_shape[1]:
                        # Single-sequence HF path: dispatch pads at the tail of
                        # the global order, so trim restores the original length.
                        original = row_shape[1]
                facts = ShardLayout(
                    local_token_global_indices=local_indices,
                    original_seq_len=original,
                    padded_seq_len=padded,
                    input_row_shape=in_rows,
                )
            return contextlib.nullcontext, prepped, facts

        return ContextParallelismSharder(shard_batch=_shard_batch_magi, local_token_global_indices=None)

    if use_te:
        # The THD partition is data-dependent (cu_seqlens), so shard_batch
        # installs the index map it just computed on the sharder for the token
        # verbs (chunked streams carry none). The BSHD->THD flatten is a pure
        # reshape, so the pre-flatten row shape is the caller's coordinate
        # system and the stream length is rows x cols.
        def _shard_batch_te(cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
            input_ids = batch.get("input_ids")
            row_shape = tuple(input_ids.shape[:2]) if input_ids is not None and input_ids.dim() >= 2 else None
            prepped, local_indices = make_cp_batch_for_te(
                cp_mesh,
                batch,
                padding_token_id=padding_token_id,
                qkv_format="thd",
                num_chunks=num_chunks,
                seq_lens_padding_value=seq_lens_padding_value,
                return_local_indices=True,
            )
            facts = None
            if local_indices is not None:
                facts = ShardLayout(
                    local_token_global_indices=local_indices,
                    padded_seq_len=row_shape[0] * row_shape[1] if row_shape is not None else None,
                    input_row_shape=row_shape,
                )
            return contextlib.nullcontext, prepped, facts

        return ContextParallelismSharder(shard_batch=_shard_batch_te, local_token_global_indices=None)

    if cp_active:
        return ContextParallelismSharder(
            shard_batch=partial(shard_batch_load_balanced, extra_seq_buffers=extra_seq_buffers),
            local_token_global_indices=round_robin_local_indices,
        )

    # No CP prep applies: the identity sharder, so callers hold working token
    # verbs at every cp_size.
    return ContextParallelismSharder(
        shard_batch=shard_batch_identity,
        local_token_global_indices=identity_local_indices,
    )


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


def _make_cp_batch_and_ctx(
    device_mesh,
    batch,
    loss_mask=None,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
    magi=None,
    model=None,
    cp_sharder: Optional[ContextParallelismSharder] = None,
    extra_seq_buffers: Optional[dict[str, int]] = None,
):
    """
    Resolve a ContextParallelismSharder and shard the batch; a no-op when no CP prep applies.

    Every CP backend is a :class:`ContextParallelismSharder`. A model that owns its CP
    attention returns one from its ``prepare_model_inputs_for_cp`` hook
    (threaded here as ``cp_sharder`` — an explicit parameter, never a batch
    key, so the batch stays pure tensors); the framework constructs one for
    magi, TE/THD, and the default load-balanced torch ``context_parallel``
    path. Resolution order: model-owned > magi > TE > generic. magi and TE
    also run at cp_size <= 1 (packing conversion / mask-spec activation);
    ``model`` is passed through opaquely to magi for its per-step key/spec
    stamping.

    Args:
        device_mesh (DeviceMesh): The device mesh; its ``cp``/``tp`` submeshes are read.
        batch (Dict[str, torch.Tensor]): The input batch containing (string, torch.Tensor)

    Returns:
        tuple (contextmanager, dict[str, torch.Tensor], ContextParallelismSharder): The forward
        context factory (nullcontext when the backend owns its transport or CP
        is inactive), the prepared/sharded batch, and the resolved sharder —
        callers use its token verbs (``shard_token_tensor`` /
        ``gather_token_tensor``) to keep per-token tensors aligned with the
        sharded inputs.
    """

    cp_mesh = _get_submesh(device_mesh, "cp")
    tp_mesh = _get_submesh(device_mesh, "tp")

    if use_te and extra_seq_buffers:
        raise ValueError("extra_seq_buffers are not supported by the TE THD context-parallel path")

    sharder = _resolve_cp_sharder(
        cp_mesh,
        cp_sharder,
        magi=magi,
        use_te=use_te,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
        model=model,
        extra_seq_buffers=extra_seq_buffers,
    )
    ctx, batch, facts = sharder.shard_batch(
        cp_mesh, tp_mesh, batch, loss_mask=loss_mask, padding_token_id=padding_token_id
    )
    sharder.shard_layout = facts
    return ctx, batch, sharder


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

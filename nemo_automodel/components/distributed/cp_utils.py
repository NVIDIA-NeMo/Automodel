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

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, List, Set, cast

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.cp_sharder import shard_batch_load_balanced
from nemo_automodel.components.distributed.thd_utils import split_batch_into_thd_chunks
from nemo_automodel.shared.cp_contracts import (
    ContextFactory,
    CPBatch,
    CPBatchWithIndices,
    CPForwardResult,
    CPPreparedInputs,
    CPPrepareModel,
    CPSharder,
    captured_token_indices,
    identity_local_indices,
    round_robin_local_indices,
    shard_batch_identity,
)

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.magi_attn_utils import MagiState


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
    cp_rotate_method: str | None = None,
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


def make_target_cp_ctx(cp_mesh: DeviceMesh, input_ids, position_ids=None):
    """Build a context-parallel context for a frozen target forward.

    Shards ``input_ids`` (and ``position_ids``) along the sequence dim across
    ``cp_mesh`` so the target's self-attention runs as ring attention. Unlike
    :func:`make_cp_batch_and_ctx`, this does not require ``labels`` and is meant
    for the EAGLE-3 target wrapper, which gathers the aux/logits back to the full
    sequence (see :func:`gather_cp_seq`) before handing them to the draft.

    Load balancing is disabled (``_cp_options.enable_load_balance = False``) so
    each rank holds a contiguous sequence chunk and the gather is a plain ordered
    concat (no round-robin un-permute). The sharding is thrown away right after
    the forward, so load balancing buys nothing here, and the ordered shard makes
    the gather deterministic. This is a process-global torch flag; the EAGLE-3
    recipe is the only context-parallel user in its process.

    The sequence is right-padded to a multiple of ``cp_size``; the returned
    ``orig_len`` lets the caller slice the gathered outputs back down.

    Args:
        cp_mesh: The context-parallel device (sub)mesh.
        input_ids: ``[B, T]`` token ids.
        position_ids: Optional ``[B, T]`` (or ``[1, T]``) position ids; an arange
            is injected when omitted.

    Returns:
        ``(cp_ctx, sharded_input_ids, sharded_position_ids, orig_len)``. Enter
        ``cp_ctx`` to run the target forward on the sharded tensors.
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import _cp_options

    _cp_options.enable_load_balance = False

    cp_size = cp_mesh.size()
    batch_size, orig_len = input_ids.shape[0], input_ids.shape[1]
    if position_ids is None:
        position_ids = torch.arange(orig_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
    position_ids = position_ids.to(input_ids.device)
    if position_ids.shape[0] == 1 and batch_size > 1:
        position_ids = position_ids.expand(batch_size, -1)

    # ``context_parallel`` shards these buffers in place and (being in
    # ``no_restore_buffers``) does not restore them on exit, so they must be
    # fresh tensors -- otherwise the caller's ``input_ids``/``position_ids``,
    # which ``generate_batch`` still uses unsharded for the shifted outputs,
    # would be corrupted. ``pad`` already produces a new tensor; ``clone`` the
    # unpadded case.
    pad = (-orig_len) % cp_size
    ids_buf = torch.nn.functional.pad(input_ids, (0, pad)) if pad else input_ids.clone()
    pos_buf = torch.nn.functional.pad(position_ids, (0, pad)) if pad else position_ids.clone()
    ids_buf = ids_buf.contiguous()
    pos_buf = pos_buf.contiguous()

    cp_ctx = context_parallel(
        cp_mesh,
        buffers=[ids_buf, pos_buf],
        buffer_seq_dims=[1, 1],
        no_restore_buffers={ids_buf, pos_buf},
    )
    return cp_ctx, ids_buf, pos_buf, orig_len


def gather_cp_seq(cp_mesh: DeviceMesh, tensors: List[torch.Tensor], seq_dim: int, orig_len: int):
    """Gather context-parallel sharded ``tensors`` back to the full sequence.

    Inverse of the sharding done by :func:`make_target_cp_ctx`. Uses torch's
    ``context_parallel_unshard`` with ``load_balancer=None`` (matching the
    load-balancing-disabled sharding) and slices the right-pad back off.

    Args:
        cp_mesh: The context-parallel device (sub)mesh used to shard.
        tensors: Local-shard tensors (e.g. captured aux hidden states, logits),
            each sharded to ``T/cp`` along ``seq_dim``.
        seq_dim: The sequence dimension to gather along.
        orig_len: The pre-pad sequence length to slice back to.

    Returns:
        A list of full-sequence tensors of length ``orig_len`` along ``seq_dim``.
    """
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard

    local_tensors = [t.to_local() if isinstance(t, DTensor) else t for t in tensors]
    full = context_parallel_unshard(cp_mesh, local_tensors, [seq_dim] * len(local_tensors))
    return [t.narrow(seq_dim, 0, orig_len).contiguous() for t in full]


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


def _mesh_dim_size(device_mesh: DeviceMesh | None, dim: str) -> int:
    """Size of a named submesh, 0 when the mesh or dimension is absent."""
    if device_mesh is None or dim not in getattr(device_mesh, "mesh_dim_names", ()):
        return 0
    return device_mesh[dim].size()


def prepare_cp_forward(
    model: torch.nn.Module | None,
    device_mesh: DeviceMesh | None,
    batch: CPBatch,
    *,
    magi: MagiState | None = None,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    loss_mask: torch.Tensor | None = None,
    cp_size: int | None = None,
    invoke_pre_embed: bool = True,
) -> CPForwardResult:
    """Prepare one training/evaluation forward for context parallelism.

    Collapses the per-recipe CP branching into one call: the model hook may
    return a CPSharder, ``make_cp_batch_and_ctx`` resolves it against the
    framework-owned sharders (magi / TE / generic torch ``context_parallel``)
    and calls ``shard_batch``. When CP is active and the model exposes
    ``prepare_model_inputs_for_cp``, the hook is invoked uniformly through
    ``model.__call__(_pre_embed_only=True, ...)`` so FSDP2 pre-forward hooks
    unshard e.g. vision towers during the pre-embed.

    Args:
        model: First model part, or ``None`` in no-model contexts.
        device_mesh: The full device mesh (``cp``/``tp`` submeshes are read).
        batch: Full-sequence mapping. Token IDs/masks are ``[B, S]``;
            embeddings are ``[B, S, H]``; mRoPE positions may be
            ``[R, B, S]``. The mapping is mutated in place.
        magi: Optional recipe MagiState, threaded to ``make_cp_batch_and_ctx``
            where it occupies the same dispatch rung as the TE path. Its
            recipe domain is bound at ``setup_magi``; for llm-domain magi the
            model hook is skipped (mirrors the recipes' historical branching),
            while vlm-domain magi still runs the pre-embed first (vision stays
            on SDPA under magi).
        use_te: THD-packed collator is active (TE/THD sharding; also magi's
            ``is_thd``).
        padding_token_id: Pad sentinel for ``input_ids``.
        num_chunks: THD chunk count, forwarded to the hook and TE sharding.
        loss_mask: Optional token mask ``[B, S]`` forwarded to sharding.
        cp_size: Override for the hook gate (e.g. the config-declared CP size);
            derived from the ``cp`` submesh when None.
        invoke_pre_embed: Invoke the model hook when CP is active or the model
            owns native THD preparation. Recipes pass False for PP stages
            without embeddings and for KD paths that never wired model-owned CP.
    Returns:
        Named context factory, prepared batch, and resolved sharder. Backend-
        owned paths return local tensors; native torch CP mutates registered
        tensors to local shards when the context is entered.
    """

    magi_enabled = magi is not None and getattr(magi, "enabled", False)
    cp_sharder = None
    has_hook = isinstance(model, CPPrepareModel)
    effective_cp_size = cp_size if cp_size is not None else _mesh_dim_size(device_mesh, "cp")

    # llm-domain magi replaces the whole batch prep (no model has both a CP
    # hook and magi); vlm-domain magi composes with the vision pre-embed.
    magi_replaces_hook = magi_enabled and getattr(magi, "domain", "llm") == "llm"
    model_owns_thd = use_te and bool(getattr(model, "supports_thd", False))
    if (effective_cp_size > 1 or model_owns_thd) and has_hook and not magi_replaces_hook and invoke_pre_embed:
        # The whole batch rides through __call__ as an opaque kwarg; the model
        # reads the keys it needs and removes the raw inputs it consumed (its
        # returned entries are merged on top). Sharder-only hooks (DSV4/GLM)
        # consume nothing; their batch — including ``input_ids`` — stays intact.
        if model is None:
            raise RuntimeError("CP preparation requires a model when a model hook is active")
        prepared = cast(
            CPPreparedInputs,
            model(input_ids=None, _pre_embed_only=True, _cp_batch=batch, num_chunks=num_chunks),
        )
        cp_sharder_value = prepared.pop("cp_sharder", None)
        if cp_sharder_value is not None and not isinstance(cp_sharder_value, CPSharder):
            raise TypeError(f"model CP hook returned invalid cp_sharder {type(cp_sharder_value).__name__}")
        cp_sharder = cp_sharder_value
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

    return _make_cp_batch_and_sharder(
        device_mesh,
        batch,
        loss_mask,
        use_te=use_te,
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        magi=magi,
        model=model,
        cp_sharder=cp_sharder,
    )


def _resolve_cp_sharder(
    cp_mesh: DeviceMesh | None,
    model_sharder: CPSharder | None,
    *,
    magi: MagiState | None,
    use_te: bool,
    num_chunks: int,
    seq_lens_padding_value: int,
    model: torch.nn.Module | None,
) -> CPSharder:
    """Resolve the CPSharder for this forward: model-owned > magi > TE > generic > none.

    Always returns a sharder: when no CP prep applies, the ``layout="none"``
    identity sharder, so callers hold working token verbs at every cp_size and
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

    # A model that owns its CP attention returns a CPSharder from its CP
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
        def _shard_batch_magi(
            cp_mesh: DeviceMesh | None,
            tp_mesh: DeviceMesh | None,
            batch: CPBatch,
            *,
            loss_mask: torch.Tensor | None = None,
            padding_token_id: int = 0,
        ) -> tuple[ContextFactory, CPBatch]:
            del tp_mesh, loss_mask
            result = magi.make_cp_batch(
                cp_mesh,
                batch,
                padding_token_id=padding_token_id,
                num_chunks=num_chunks,
                is_thd=use_te,
                model=model,
            )
            if result.local_indices is not None:
                magi_sharder.local_token_global_indices = captured_token_indices(result.local_indices)
            return contextlib.nullcontext, result.batch

        magi_sharder = CPSharder(shard_batch=_shard_batch_magi, local_token_global_indices=None, layout="magi")
        return magi_sharder

    if use_te:
        # The THD partition is data-dependent (cu_seqlens), so shard_batch
        # installs the index map it just computed on the sharder for the token
        # verbs (chunked streams carry none).
        def _shard_batch_te(
            cp_mesh: DeviceMesh | None,
            tp_mesh: DeviceMesh | None,
            batch: CPBatch,
            *,
            loss_mask: torch.Tensor | None = None,
            padding_token_id: int = 0,
        ) -> tuple[ContextFactory, CPBatch]:
            del tp_mesh, loss_mask
            result = _make_cp_batch_for_te_with_indices(
                cp_mesh,
                batch,
                padding_token_id=padding_token_id,
                qkv_format="thd",
                num_chunks=num_chunks,
                seq_lens_padding_value=seq_lens_padding_value,
            )
            if result.local_indices is not None:
                te_sharder.local_token_global_indices = captured_token_indices(result.local_indices)
            return contextlib.nullcontext, result.batch

        te_sharder = CPSharder(shard_batch=_shard_batch_te, local_token_global_indices=None, layout="thd")
        return te_sharder

    if cp_active:
        return CPSharder(
            shard_batch=shard_batch_load_balanced,
            local_token_global_indices=round_robin_local_indices,
            layout="round_robin",
        )

    return CPSharder(
        shard_batch=shard_batch_identity,
        local_token_global_indices=identity_local_indices,
        layout="none",
    )


def make_cp_batch_and_ctx(
    device_mesh: DeviceMesh | None,
    batch: CPBatch,
    loss_mask: torch.Tensor | None = None,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
    magi: MagiState | None = None,
    model: torch.nn.Module | None = None,
    cp_sharder: CPSharder | None = None,
) -> tuple[ContextFactory, CPBatch]:
    """Resolve a CP backend and return the established two-value result.

    Every CP backend is a :class:`CPSharder`. A model that owns its CP
    attention returns one from its ``prepare_model_inputs_for_cp`` hook
    (threaded here as ``cp_sharder`` — an explicit parameter, never a batch
    key, so the batch stays pure tensors); the framework constructs one for
    magi, TE/THD, and the default load-balanced torch ``context_parallel``
    path. Resolution order: model-owned > magi > TE > generic. magi and TE
    also run at cp_size <= 1 (packing conversion / mask-spec activation);
    ``model`` is passed through opaquely to magi for its per-step key/spec
    stamping.

    Args:
        device_mesh: Full device mesh, or ``None``.
        batch: Full batch with IDs/masks ``[B, S]`` or embeddings
            ``[B, S, H]``. Here ``B`` is batch size, ``S`` is global sequence
            length, and ``H`` is hidden size. It is mutated during preparation.
        loss_mask: Optional token mask ``[B, S]``.
        use_te: Convert and shard a packed THD batch.
        padding_token_id: Token padding sentinel.
        num_chunks: Number of independent THD batch chunks.
        seq_lens_padding_value: Packed-length padding sentinel.
        magi: Optional MagiAttention state.
        model: Optional model passed to backend preparation.
        cp_sharder: Optional model-owned sharder.

    Returns:
        Context factory and prepared batch. This preserves the public return
        contract; callers needing token verbs use :func:`prepare_cp_forward`.
    """

    result = _make_cp_batch_and_sharder(
        device_mesh,
        batch,
        loss_mask,
        use_te=use_te,
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
        magi=magi,
        model=model,
        cp_sharder=cp_sharder,
    )
    return result.context_factory, result.batch


def _make_cp_batch_and_sharder(
    device_mesh: DeviceMesh | None,
    batch: CPBatch,
    loss_mask: torch.Tensor | None = None,
    *,
    use_te: bool = False,
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
    magi: MagiState | None = None,
    model: torch.nn.Module | None = None,
    cp_sharder: CPSharder | None = None,
) -> CPForwardResult:
    """Resolve a sharder and prepare one full token batch.

    Args:
        device_mesh: Full device mesh, or ``None``.
        batch: Full token mapping with IDs/masks ``[B, S]`` or embeddings
            ``[B, S, H]``, where ``B`` is batch size, ``S`` is global sequence
            length, and ``H`` is hidden size. It is mutated during sharding.
        loss_mask: Optional token mask ``[B, S]``.
        use_te: Convert and shard a packed THD batch.
        padding_token_id: Token padding sentinel.
        num_chunks: Number of independent THD chunks.
        seq_lens_padding_value: Packed-length padding sentinel.
        magi: Optional MagiAttention state.
        model: Optional model passed to backend preparation.
        cp_sharder: Optional model-owned sharder.

    Returns:
        Named context factory, prepared batch, and resolved sharder. Token
        tensors are local for backend-owned sharding and become local on native
        torch CP context entry.
    """

    def _get_submesh(device_mesh: DeviceMesh | None, name: str) -> DeviceMesh | None:
        if device_mesh is not None and name in getattr(device_mesh, "mesh_dim_names", {}):
            return device_mesh[name]
        return None

    cp_mesh = _get_submesh(device_mesh, "cp")
    tp_mesh = _get_submesh(device_mesh, "tp")

    sharder = _resolve_cp_sharder(
        cp_mesh,
        cp_sharder,
        magi=magi,
        use_te=use_te,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
        model=model,
    )
    ctx, batch = sharder.shard_batch(cp_mesh, tp_mesh, batch, loss_mask=loss_mask, padding_token_id=padding_token_id)
    return CPForwardResult(ctx, batch, sharder)


def make_cp_batch_for_te(
    cp_mesh: DeviceMesh | None,
    batch: CPBatch,
    qkv_format: str = "thd",
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
) -> CPBatch:
    """Build a Transformer Engine THD batch with a stable batch-only return.

    Args:
        cp_mesh: Context-parallel mesh, or ``None`` for conversion without
            sharding.
        batch: BSH input containing IDs ``[B, S]`` or embeddings ``[B, S, H]``,
            labels/positions ``[B, S]``, and packed lengths ``[B, P]`` where
            ``B`` is batch size, ``S`` is sequence length, ``H`` is hidden
            size, and ``P`` is the maximum pack count.
        qkv_format: Required output layout; only ``"thd"`` is supported.
        padding_token_id: Token padding sentinel.
        num_chunks: Number of independent batch chunks.
        seq_lens_padding_value: Sentinel for unused packed lengths.

    Returns:
        THD batch. IDs are ``[T_local]`` and embeddings ``[T_local, H]`` for
        one chunk; chunked tensors prepend a chunk axis. Here ``T_local`` is
        total local tokens after CP partitioning.
    """
    return _make_cp_batch_for_te_with_indices(
        cp_mesh,
        batch,
        qkv_format=qkv_format,
        padding_token_id=padding_token_id,
        num_chunks=num_chunks,
        seq_lens_padding_value=seq_lens_padding_value,
    ).batch


def _make_cp_batch_for_te_with_indices(
    cp_mesh: DeviceMesh | None,
    batch: CPBatch,
    qkv_format: str = "thd",
    padding_token_id: int = 0,
    num_chunks: int = 1,
    seq_lens_padding_value: int = -1000,
) -> CPBatchWithIndices:
    """Build a THD batch and capture its local-to-global token map.

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
        Named processed batch and local int64 index map ``[T_local]``. The map
        is ``None`` in chunked mode, where chunks have independent token spaces.
        Here ``T_local`` is per-rank packed-token count.
        The batch contains:
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

    batch = cast(
        CPBatch,
        split_batch_into_thd_chunks(
            cast(dict[str, torch.Tensor], batch),
            num_chunks=num_chunks,
            seq_lens_padding_value=seq_lens_padding_value,
            padding_token_id=padding_token_id,
        ),
    )

    if cp_mesh is None or cp_mesh.size() <= 1:
        # Unsharded THD stream: identity index map. Chunked streams are
        # per-chunk token spaces with no single step-wide map -> None.
        input_ids = cast(torch.Tensor, batch["input_ids"])
        local_indices = (
            torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.long) if num_chunks <= 1 else None
        )
        return CPBatchWithIndices(batch, local_indices)

    if num_chunks <= 1:
        sharded, local_indices = _shard_thd_chunk_for_te(
            batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id
        )
        return CPBatchWithIndices(sharded, local_indices)

    # Extract each chunk from the batched result and shard it
    chunks = []
    for i in range(num_chunks):
        chunk_batch = {k: v[i] if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        chunks.append(
            _shard_thd_chunk_for_te(chunk_batch, cp_mesh, qkv_format, seq_lens_padding_value, padding_token_id)[0]
        )

    return_dict: CPBatch = {
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
    return CPBatchWithIndices(return_dict, None)


def _shard_thd_chunk_for_te(
    batch: CPBatch,
    cp_mesh: DeviceMesh,
    qkv_format: str,
    seq_lens_padding_value: int,
    padding_token_id: int,
) -> tuple[CPBatch, torch.Tensor]:
    """Shard one flat THD stream and return local global-token positions.

    Token-aligned input tensors use flat ``[T]`` layout. The returned batch
    keeps local token tensors ``[T_local]`` and packed cumulative lengths
    ``[N_sequence + 1]``; the index map is int64 ``[T_local]``. Here ``T`` is
    global packed-token count, ``T_local`` is per-rank count, and
    ``N_sequence`` is packed sequence count.

    Args:
        batch: Flat THD mapping with token tensors ``[T]`` and cumulative
            lengths ``[N_sequence + 1]``. Token fields are replaced in place
            by local ``[T_local]`` shards.
        cp_mesh: Active context-parallel mesh.
        qkv_format: Output QKV layout marker; ``"thd"`` for this path.
        seq_lens_padding_value: Sentinel excluded from cumulative lengths.
        padding_token_id: Token value used to synthesize local padding mask.

    Returns:
        Local THD mapping and int64 global token positions ``[T_local]``.
    """
    import transformer_engine_torch as tex

    cu_seqlens = batch.get("cu_seqlens")
    cu_seqlens_padded = batch.get("cu_seqlens_padded", cu_seqlens)
    if not isinstance(cu_seqlens, torch.Tensor) or not isinstance(cu_seqlens_padded, torch.Tensor):
        raise ValueError("THD sharding requires tensor cu_seqlens and cu_seqlens_padded")
    filtered_cu_seqlens_padded = cu_seqlens_padded[cu_seqlens_padded != seq_lens_padding_value]

    cp_size = cp_mesh.size()
    cp_rank = torch.distributed.get_rank(group=cp_mesh.get_group())
    input_ids = batch.get("input_ids")
    if not isinstance(input_ids, torch.Tensor):
        raise ValueError("THD sharding requires tensor input_ids")

    # The partition is the same for every token-aligned key; it is also this
    # rank's local-token global index map, returned so the caller can install
    # it on the THD sharder (CPSharder token verbs).
    local_indices = tex.thd_get_partitioned_indices(filtered_cu_seqlens_padded, input_ids.size(0), cp_size, cp_rank)
    mask_keys = ["input_ids", "labels", "position_ids", "padding_mask"]
    for key in mask_keys:
        if key in batch:
            value = batch[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"THD token-aligned field {key!r} must be a tensor")
            batch[key] = value.index_select(0, local_indices)

    max_seqlen = (filtered_cu_seqlens_padded[1:] - filtered_cu_seqlens_padded[:-1]).max().item()
    input_ids = cast(torch.Tensor, batch["input_ids"])
    labels = cast(torch.Tensor, batch["labels"])
    position_ids = cast(torch.Tensor, batch["position_ids"])
    output_batch: CPBatch = {
        "input_ids": input_ids.to(torch.int64).contiguous(),
        "labels": labels.to(torch.int64).contiguous(),
        "position_ids": position_ids.to(torch.int64).contiguous(),
        "cu_seqlens": cu_seqlens_padded.to(torch.int32).contiguous(),
        "max_seqlen": torch.tensor(max_seqlen).to(torch.int32).to(device=cu_seqlens_padded.device),
        "qkv_format": qkv_format,
        "padding_mask": (input_ids == padding_token_id).bool().contiguous(),
        "cp_size": cp_size,
        "cp_rank": cp_rank,
    }

    return output_batch, local_indices

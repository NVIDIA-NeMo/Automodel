# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Target-side context parallelism for speculative-decoding draft training.

The draft trainers (EAGLE-3, DFlash, DSpark) shard the FROZEN target's forward
along the sequence and gather its captured hidden states / logits back to the
full sequence before handing them to the draft. That flow is specific to
speculative decoding -- it needs no gradients through the target and no
model-owned CP sharder -- so it lives here rather than in the shared
``components/distributed/context_parallel/utils`` surface.
"""

from typing import Callable, List, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh


def make_target_cp_ctx(cp_mesh: DeviceMesh, input_ids, position_ids=None):
    """Build a context-parallel context for a frozen target forward.

    Shards ``input_ids`` (and ``position_ids``) along the sequence dim across
    ``cp_mesh`` so the target's self-attention runs as ring attention. Unlike
    the CP dispatch, this does not require ``labels`` and is meant
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


def run_target_cp_forward_and_gather(
    cp_mesh: DeviceMesh,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    forward_kwargs: dict,
    collect: Callable[[object], List[torch.Tensor]],
    position_ids: Optional[torch.Tensor] = None,
    filter_kwargs: bool = False,
) -> tuple:
    """Run a frozen target under context parallelism and gather its outputs.

    Shards ``input_ids`` (and ``position_ids``) along the sequence via
    :func:`make_target_cp_ctx`, runs ``model`` as ring attention with
    ``attention_mask=None`` (the ``self_attn`` hooks force ``is_causal``), and --
    still inside the CP context, before any capture hooks are removed -- gathers
    the tensors returned by ``collect(outputs)`` back to the full sequence with
    :func:`gather_cp_seq` (``seq_dim=1``, un-padded to ``orig_len``).

    Centralizes the gather-inside-context invariant and the ``seq_dim``/``orig_len``
    contract shared by the eagle3/dflash/dspark target wrappers, so a fix lands in
    one place instead of drifting across three copies.

    Args:
        cp_mesh: The ``cp`` device submesh.
        model: The frozen target module.
        input_ids: Full (unsharded) ``[B, T]`` token ids.
        forward_kwargs: Extra kwargs for ``model.forward``. Must not include
            ``input_ids`` / ``attention_mask`` / ``position_ids`` -- those are
            injected here (CP forces ``attention_mask=None``).
        collect: ``callable(outputs) -> list[Tensor]`` selecting the tensors to
            gather; invoked inside the CP context after the forward, so it also
            sees any tensors captured by forward hooks.
        position_ids: Optional ``[B, T]`` / ``[1, T]`` positions; an arange is
            injected by :func:`make_target_cp_ctx` when omitted.
        filter_kwargs: Drop kwargs the model's forward does not accept (via
            :func:`filter_forward_kwargs`) -- needed for the VLM/MoE targets.

    Returns:
        ``(outputs, gathered)`` -- the raw model outputs and the list of
        full-sequence gathered tensors, in the order ``collect`` returned them.
    """
    import inspect

    # CP shards the sequence, so the target MUST honor per-shard position_ids;
    # a target whose forward ignores them would attend at the wrong positions.
    if "position_ids" not in inspect.signature(model.forward).parameters:
        raise ValueError("Context parallelism requires the target model's forward to accept `position_ids`.")

    cp_ctx, cp_input_ids, cp_position_ids, orig_len = make_target_cp_ctx(cp_mesh, input_ids, position_ids)
    call_kwargs = {
        "input_ids": cp_input_ids,
        "attention_mask": None,
        "position_ids": cp_position_ids,
        **forward_kwargs,
    }
    if filter_kwargs:
        from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

        call_kwargs = filter_forward_kwargs(model, call_kwargs)
    with cp_ctx:
        outputs = model(**call_kwargs)
        gathered = gather_cp_seq(cp_mesh, collect(outputs), seq_dim=1, orig_len=orig_len)
    return outputs, gathered


def attach_cp_kv_gather_hooks(model: torch.nn.Module, cp_mesh) -> None:
    """Context-parallel self-attention for a FROZEN (forward-only) target.

    Torch's ``context_parallel`` ring dispatch does not fire for a plain HuggingFace
    forward -- q/k/v reach ``F.scaled_dot_product_attention`` as ordinary local
    tensors, so each rank silently attends only to its own sequence shard and every
    position past the first shard is wrong. Because the target is frozen (no
    backward through attention), the correct fix is simple: all-gather K/V across the
    cp group and attend the local Q against the full K/V with a global causal mask.
    The O(S^2) attention matrix stays sharded ``[S/cp, S]`` per rank (the memory
    win); only the O(S) K/V is replicated.

    Assumes contiguous (non-load-balanced) sharding -- rank ``r`` holds global
    positions ``[r*S_local, (r+1)*S_local)`` -- which is what :func:`make_target_cp_ctx`
    produces (it disables load balancing). Q/K/V are ``[B, nH, S_local, D]`` at the
    SDPA call, so the sequence dim is 2.
    """
    import torch.nn.functional as F_module

    _original_sdpa = F_module.scaled_dot_product_attention
    # The global causal mask depends only on the shard geometry (seq_local, cp
    # rank/size), which is constant across layers and steps, so build the additive
    # ``[S_local, S]`` float mask once and reuse it rather than per layer.
    _mask_cache: dict = {}

    @torch._dynamo.disable
    def _cp_gather_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kwargs
    ):
        # The K/V all-gather needs a live cp process group. Without one (a unit test
        # exercising the surrounding logic; real CP always has one) fall back to plain
        # local SDPA -- mirrors the ``cp_mesh.size() <= 1`` graceful no-ops elsewhere.
        if not torch.distributed.is_initialized():
            return _original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
        # Resolve the cp group lazily (at forward time): attaching the hook must not
        # require a live process group, so a mesh mock with only ``.size()`` works.
        group = cp_mesh.get_group()
        cp_size = cp_mesh.size()
        cp_rank = torch.distributed.get_rank(group=group)
        seq_local = query.shape[2]
        key = key.contiguous()
        value = value.contiguous()
        k_parts = [torch.empty_like(key) for _ in range(cp_size)]
        v_parts = [torch.empty_like(value) for _ in range(cp_size)]
        torch.distributed.all_gather(k_parts, key, group=group)
        torch.distributed.all_gather(v_parts, value, group=group)
        k_full = torch.cat(k_parts, dim=2)
        v_full = torch.cat(v_parts, dim=2)
        # Global causal mask: local query at global position ``cp_rank*seq_local + i``
        # attends to every key position ``j`` up to and including it. Built as an
        # additive FLOAT (0 / -inf) mask, not a boolean one: a bool mask under GQA
        # rejects the fused kernels and lands on MATH, which materializes the full
        # ``[H, S/cp, S]`` fp32 score matrix (tens of GB at the long contexts CP is
        # meant for). Cached on the shard geometry (constant across layers/steps).
        mask_key = (seq_local, k_full.shape[2], query.dtype, query.device)
        mask = _mask_cache.get(mask_key)
        if mask is None:
            q_pos = torch.arange(seq_local, device=query.device) + cp_rank * seq_local
            k_pos = torch.arange(k_full.shape[2], device=query.device)
            mask = torch.zeros(seq_local, k_full.shape[2], dtype=query.dtype, device=query.device)
            mask.masked_fill_(q_pos[:, None] < k_pos[None, :], float("-inf"))
            _mask_cache[mask_key] = mask
        # Expand the gathered K/V heads to Q's head count so ``enable_gqa`` is off: the
        # memory-efficient kernel rejects GQA combined with an explicit mask, but takes
        # a float mask once the heads match (keeping the O(S^2) scores tiled, not
        # materialized). The extra K/V copy is only O(S).
        if enable_gqa and k_full.shape[1] != query.shape[1]:
            rep = query.shape[1] // k_full.shape[1]
            k_full = k_full.repeat_interleave(rep, dim=1)
            v_full = v_full.repeat_interleave(rep, dim=1)
        # Prefer the memory-efficient kernel; MATH stays only as a correctness fallback.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            return _original_sdpa(query, k_full, v_full, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale)

    def _pre_hook(module, args, kwargs):
        F_module.scaled_dot_product_attention = _cp_gather_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        F_module.scaled_dot_product_attention = _original_sdpa

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            target = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
            target.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            target.register_forward_hook(_post_hook, always_call=True)

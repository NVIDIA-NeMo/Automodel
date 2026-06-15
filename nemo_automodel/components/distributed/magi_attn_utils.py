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

"""MagiAttention integration for Automodel.

MagiAttention (https://github.com/SandAI-org/MagiAttention) is a distributed
(context-parallel) attention built on a Flex-Flash-Attention (FFA) kernel.  It
shards a single packed sequence across a CP process group with a load-balancing
dispatch solver and exchanges KV with zero-redundant GroupCast/GroupReduce
collectives.

This module wires MagiAttention into the HF-transformers-based LLM path used by
``recipes/llm/train_ft.py`` following MagiAttention's official
``examples/transformers`` recipe:

1. ``register_magi_attention()`` registers a ``"magi"`` entry in HF's
   ``ALL_ATTENTION_FUNCTIONS`` so that a model loaded with
   ``attn_implementation="magi"`` routes its attention through the FFA kernel.
2. ``magi_prepare_batch()`` builds the per-step dist-attn runtime key, dispatches
   ``input_ids``/``position_ids``/``labels`` across the CP group and stamps
   ``cp_group`` on every attention sub-module so the registered forward finds the key.
3. Each rank runs the model on its local shard and computes a per-shard loss; the
   recipe's cross-CP reduction sums the shards into the global loss (like TE-CP).
   Sharding labels (rather than undispatching logits) keeps the loss path identical
   for the HF and custom-model backends.

When ``cp_size == 1`` the dispatch is a no-op shard (identity + chunk padding),
so this path is also a clean way to swap *only* the attention kernel (FFA) in
place of eager/SDPA/flash for convergence-parity comparisons.
"""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx, make_cp_batch_for_te

logger = logging.getLogger(__name__)

_MAGI_REGISTERED = False
# Default dispatch chunk size used by the load-balancing solver.
DEFAULT_CHUNK_SIZE = 512
# Self-key path: id(cp_group) -> last sequence length a key was built for.
_MAGI_SELF_KEY_LEN: dict = {}
# Active CP group for the custom-model attention factory (set by the recipe).
# The custom factory's attn_func is a closure with no access to the attention
# module, so it reads the cp_group from here. One CP group is active per process.
_ACTIVE_CP_GROUP: Optional["dist.ProcessGroup"] = None


def set_active_cp_group(cp_group: Optional["dist.ProcessGroup"]) -> None:
    """Record the CP group the custom-model magi attn_func should use."""
    global _ACTIVE_CP_GROUP
    _ACTIVE_CP_GROUP = cp_group


def get_active_cp_group() -> Optional["dist.ProcessGroup"]:
    """Return the CP group set by :func:`set_active_cp_group` (may be None)."""
    return _ACTIVE_CP_GROUP


# ---------------------------------------------------------------------------
# Arbitrary attention masks (AttnSlice / flex key)
#
# MagiAttention's FFA expresses any mask as a set of rectangles: for each
# (q_range, k_range) pair, an AttnMaskType in {full, causal, ...}. This covers
# plain causal, varlen/block-diagonal (sequence packing), sliding-window, and the
# block-sparse prefix-tree mask needed for shared-prefix RL training (verl RFC
# #6401 / Automodel #2385): each tree node attends FULL to its ancestor chain and
# CAUSAL to itself. ``AttnMaskSpec`` is a backend-agnostic description of such a
# mask; ``build_flex_key`` turns it into a magi dist-attn key.
# ---------------------------------------------------------------------------

# Active mask spec for the custom-model attn_func (set per step by the caller;
# None -> plain causal self-key). Keyed implicitly by the single active cp_group.
_ACTIVE_ATTN_SPEC: Optional["AttnMaskSpec"] = None
# id(cp_group) -> (spec_fingerprint, built_key), so all layers in a step reuse one key.
_FLEX_KEY_CACHE: dict = {}


@dataclass
class AttnMaskSpec:
    """Backend-agnostic description of an attention mask as AttnSlice rectangles.

    Attributes:
        q_ranges: list of ``[start, end)`` query token ranges (one per rectangle).
        k_ranges: list of ``[start, end)`` key token ranges (one per rectangle).
        mask_types: per-rectangle mask type, ``"full"`` or ``"causal"``.
        total_seqlen: total number of tokens in the (flat) sequence.
    """

    q_ranges: list
    k_ranges: list
    mask_types: list
    total_seqlen: int

    def fingerprint(self) -> tuple:
        """Hashable identity used to cache the built key across layers."""
        return (
            self.total_seqlen,
            tuple(map(tuple, self.q_ranges)),
            tuple(map(tuple, self.k_ranges)),
            tuple(self.mask_types),
        )

    @classmethod
    def causal(cls, seqlen: int) -> "AttnMaskSpec":
        """A single full causal sequence."""
        return cls([[0, seqlen]], [[0, seqlen]], ["causal"], seqlen)

    @classmethod
    def varlen(cls, seqlens: list[int], causal: bool = True) -> "AttnMaskSpec":
        """Block-diagonal mask for packed sequences (one block per document)."""
        q_ranges, k_ranges, mask_types, off = [], [], [], 0
        for n in seqlens:
            q_ranges.append([off, off + n])
            k_ranges.append([off, off + n])
            mask_types.append("causal" if causal else "full")
            off += n
        return cls(q_ranges, k_ranges, mask_types, off)

    @classmethod
    def prefix_tree(cls, node_lengths: list[int], sample_paths: list[list[int]]):
        """Build a prefix-tree mask over a flat deduplicated token layout.

        Args:
            node_lengths: token count of each node, in flat layout order. The flat
                layout is ``[node_0 | node_1 | ...]`` with node ``i`` occupying
                ``[offset_i, offset_i + node_lengths[i])``.
            sample_paths: one list of node indices per sample, root -> leaf. Every
                sample is the causal concatenation of its nodes; a shared prefix
                node simply appears in multiple paths.

        Returns:
            (spec, sample_token_ranges): ``spec`` is the AttnMaskSpec; the second
            value lists, per sample, the flat ``[start, end)`` ranges of its nodes
            (in path order) for reconstructing per-sample outputs.

        Each node attends FULL to every ancestor node in its path and CAUSAL to
        itself; duplicate rectangles (shared nodes) are emitted once.
        """
        offsets, acc = [], 0
        for n in node_lengths:
            offsets.append(acc)
            acc += n
        total = acc
        q_ranges, k_ranges, mask_types, seen = [], [], [], set()
        for path in sample_paths:
            for pos, node in enumerate(path):
                q_rng = (offsets[node], offsets[node] + node_lengths[node])
                # self -> causal
                rect = (q_rng, q_rng, "causal")
                if rect not in seen:
                    seen.add(rect)
                    q_ranges.append(list(q_rng))
                    k_ranges.append(list(q_rng))
                    mask_types.append("causal")
                # ancestors -> full
                for anc in path[:pos]:
                    k_rng = (offsets[anc], offsets[anc] + node_lengths[anc])
                    rect = (q_rng, k_rng, "full")
                    if rect not in seen:
                        seen.add(rect)
                        q_ranges.append(list(q_rng))
                        k_ranges.append(list(k_rng))
                        mask_types.append("full")
        sample_token_ranges = [[[offsets[n], offsets[n] + node_lengths[n]] for n in path] for path in sample_paths]
        return cls(q_ranges, k_ranges, mask_types, total), sample_token_ranges


def set_active_attn_spec(spec: Optional["AttnMaskSpec"]) -> None:
    """Set the mask spec the custom-model magi attn_func should apply this step."""
    global _ACTIVE_ATTN_SPEC
    _ACTIVE_ATTN_SPEC = spec


def get_active_attn_spec() -> Optional["AttnMaskSpec"]:
    """Return the active mask spec (None -> plain causal self-key)."""
    return _ACTIVE_ATTN_SPEC


def build_flex_key(
    spec: "AttnMaskSpec", *, num_heads_q, num_heads_kv, head_dim, cp_group
):  # pragma: no cover - requires GPU + magi_attention
    """Build a magi dist-attn key for an arbitrary AttnSlice mask (no extra padding)."""
    from magi_attention.api import magi_attn_flex_key
    from magi_attention.common import AttnRanges

    return magi_attn_flex_key(
        q_ranges=AttnRanges.from_ranges([list(r) for r in spec.q_ranges]),
        k_ranges=AttnRanges.from_ranges([list(r) for r in spec.k_ranges]),
        attn_mask_type=list(spec.mask_types),
        total_seqlen_q=spec.total_seqlen,
        total_seqlen_k=spec.total_seqlen,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )


def _flex_key_for(
    cp_group, spec, *, num_heads_q, num_heads_kv, head_dim
):  # pragma: no cover - requires GPU + magi_attention
    """Return the flex key for ``spec``, rebuilding only when the mask changes."""
    fp = spec.fingerprint()
    cached = _FLEX_KEY_CACHE.get(id(cp_group))
    if cached is not None and cached[0] == fp:
        return cached[1]
    key = build_flex_key(
        spec,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        cp_group=cp_group,
    )
    _FLEX_KEY_CACHE[id(cp_group)] = (fp, key)
    return key


def _build_self_key(
    cp_group, *, seqlen, num_heads_q, num_heads_kv, head_dim, device
):  # pragma: no cover - requires GPU + magi_attention
    """Build a cp=1 causal varlen key matching the actual q length (no dispatch)."""
    from magi_attention.api import magi_attn_varlen_key

    cu = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    return magi_attn_varlen_key(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=0,
        cp_group_or_mesh=cp_group,
        causal=True,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )


def _self_key_for(
    cp_group, *, seqlen, num_heads_q, num_heads_kv, head_dim, device
):  # pragma: no cover - requires GPU + magi_attention
    """Return a causal self-key for ``seqlen``, (re)building only when it changes.

    All attention layers in one forward share the same sequence length, so the
    first layer builds the key and the rest reuse it via ``get_most_recent_key``.
    """
    from magi_attention.api import get_most_recent_key

    if _MAGI_SELF_KEY_LEN.get(id(cp_group)) == seqlen:
        try:
            return get_most_recent_key(cp_group)
        except Exception:
            pass
    key = _build_self_key(
        cp_group,
        seqlen=seqlen,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        device=device,
    )
    _MAGI_SELF_KEY_LEN[id(cp_group)] = seqlen
    return key


def make_magi_attn_func(softmax_scale: Optional[float] = None):  # pragma: no cover - requires GPU + magi_attention
    """Build the attn_func used by the custom-model attention factory.

    The returned callable accepts q/k/v in either THD ``[t, nh, hd]`` or BSHD
    ``[b, s, nh, hd]`` (b must be 1) layout — the same layouts the custom models
    feed to their backend attn_func — and runs the MagiAttention FFA kernel.

    Mask selection (no CP dispatch; cp=1 in-order tokens):
      * if an :class:`AttnMaskSpec` is active (``set_active_attn_spec``), use its
        flex key — this covers packing, sliding-window and prefix-tree masks;
      * otherwise build a plain causal self-key from the q length.
    If no CP group is active it falls back to causal SDPA so non-magi modules
    (e.g. a VLM vision tower routed through the same factory) keep working.

    Args:
        softmax_scale: attention softmax scale (defaults to 1/sqrt(head_dim) inside
            FFA when None); forwarded so non-default scales stay correct.
    """
    from einops import rearrange
    from magi_attention.api import calc_attn, get_most_recent_key

    def magi_attn_func(q, k, v, **call_kwargs):
        cp_group = get_active_cp_group()
        if cp_group is None:
            import torch.nn.functional as F

            qh, kh, vh = (e.transpose(1, 2) if e.dim() == 4 else e for e in (q, k, v))
            # Forward the configured scale (None -> SDPA default 1/sqrt(head_dim)) so a
            # non-default attention scale (e.g. Cohere) is honored on the fallback path too.
            return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=softmax_scale, enable_gqa=True)

        bshd = q.dim() == 4
        if bshd:
            b = q.shape[0]
            qf, kf, vf = (rearrange(e, "b s nh hd -> (b s) nh hd") for e in (q, k, v))
        else:
            qf, kf, vf = q, k, v
        dtype = qf.dtype
        qf, kf, vf = (e.to(torch.bfloat16).contiguous() for e in (qf, kf, vf))
        # Context-parallel (cp>1): q/k/v here are the LOCAL sequence shard and the
        # recipe has already built + dispatched the global dist key (the FFA kernel
        # does the cross-rank KV exchange). Use that pre-built key directly rather
        # than building one from the local shard.
        if cp_group.size() > 1:
            key = get_most_recent_key(cp_group)
            o = calc_attn(qf, kf, vf, key, softmax_scale=softmax_scale)[0].to(dtype)
            if bshd:
                o = rearrange(o, "(b s) nh hd -> b s nh hd", b=b)
            return o
        # Resolve the mask, preferring per-call args over module-global state:
        #   1. explicit ``magi_attn_spec`` (arbitrary AttnSlice mask, e.g. prefix tree)
        #   2. ``cu_seqlens`` -> varlen/block-diagonal (packed sequences)
        #   3. the active spec set out-of-band via set_active_attn_spec()
        #   4. fallback: a single causal sequence
        # Sliding-window attention is not wired into the magi key yet; fail loudly
        # rather than silently computing full causal. (-1, 0)/(-1, -1) == "no window".
        win = call_kwargs.get("window_size")
        if win is not None:
            left = win[0] if isinstance(win, (tuple, list)) else win
            right = win[1] if isinstance(win, (tuple, list)) and len(win) > 1 else 0
            if (left is not None and left >= 0) or (right is not None and right > 0):
                raise NotImplementedError(
                    "magi attention does not yet support sliding-window masks; "
                    "pass an explicit magi_attn_spec or use a non-windowed config."
                )
        spec = call_kwargs.get("magi_attn_spec")
        if spec is None:
            cu = call_kwargs.get("cu_seqlens", call_kwargs.get("cu_seqlens_q"))
            if cu is not None and int(cu.numel()) > 2:
                seqlens = (cu[1:] - cu[:-1]).tolist()
                spec = AttnMaskSpec.varlen(seqlens, causal=True)
        if spec is None:
            spec = get_active_attn_spec()
        if spec is not None:
            key = _flex_key_for(
                cp_group,
                spec,
                num_heads_q=qf.shape[1],
                num_heads_kv=kf.shape[1],
                head_dim=qf.shape[2],
            )
        else:
            key = _self_key_for(
                cp_group,
                seqlen=qf.shape[0],
                num_heads_q=qf.shape[1],
                num_heads_kv=kf.shape[1],
                head_dim=qf.shape[2],
                device=qf.device,
            )
        o = calc_attn(qf, kf, vf, key, softmax_scale=softmax_scale)[0].to(dtype)
        if bshd:
            o = rearrange(o, "(b s) nh hd -> b s nh hd", b=b)
        return o

    return magi_attn_func


def is_magi_available() -> bool:
    """Return True if the ``magi_attention`` package is importable."""
    try:
        import magi_attention  # noqa: F401

        return True
    except Exception as e:  # pragma: no cover - import guard
        logger.debug("magi_attention not importable: %s", e)
        return False


def register_magi_attention() -> None:  # pragma: no cover - requires GPU + magi_attention
    """Register the ``"magi"`` attention backend in HF transformers (idempotent)."""
    global _MAGI_REGISTERED
    if _MAGI_REGISTERED:
        return

    from einops import rearrange
    from magi_attention.api import calc_attn, get_most_recent_key
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def magi_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,  # (b, num_heads, seq, head_dim)
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs: Any,
    ):
        cp_group = getattr(module, "cp_group", None)
        if cp_group is None:
            # Not a magi-managed (causal LM) attention -> fall back to SDPA. This
            # happens for VLM vision-tower modules when the whole model is loaded
            # with attn_implementation="magi": only the language backbone gets a
            # cp_group stamped, so the (bidirectional, different head_dim) vision
            # tower transparently uses SDPA.
            from transformers.integrations.sdpa_attention import sdpa_attention_forward

            return sdpa_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout=dropout,
                scaling=scaling,
                is_causal=is_causal,
                **kwargs,
            )

        if getattr(module, "_magi_self_key", False):
            # VLM (cp_size==1) path: the post-image-merge LM sequence length is only
            # known here (query dim 2: [b, nh, s, hd]) and may differ from input_ids
            # length. Build a no-dispatch causal key matching the actual q length.
            magi_attn_key = _self_key_for(
                cp_group,
                seqlen=query.shape[2],
                num_heads_q=query.shape[1],
                num_heads_kv=key.shape[1],
                head_dim=query.shape[3],
                device=query.device,
            )
        else:
            magi_attn_key = get_most_recent_key(cp_group)

        b = query.shape[0]
        dtype = query.dtype
        # (b, nh, s, hd) -> ((b s), nh, hd); FFA only supports fp16/bf16.
        # ``.contiguous()`` is required: the rearrange is a permute, and the FFA
        # backward kernel reads the saved q/k/v assuming contiguous strides — a
        # non-contiguous tensor yields nan gradients (forward is unaffected).
        q, k, v = (
            rearrange(e, "b nh s hd -> (b s) nh hd").to(torch.bfloat16).contiguous() for e in (query, key, value)
        )
        # Forward the HF attention scale to FFA. For the standard 1/sqrt(head_dim)
        # scale (Llama, Qwen, ...) ``scaling`` matches FFA's default, but models with a
        # non-default scale (e.g. Cohere) would otherwise be computed with the wrong one.
        o = calc_attn(q, k, v, magi_attn_key, softmax_scale=scaling)[0]
        # ((b s), nh, hd) -> (b, s, nh*hd)
        o = rearrange(o, "(b s) nh hd -> b s (nh hd)", b=b).to(dtype)
        return o, None

    ALL_ATTENTION_FUNCTIONS.register("magi", magi_attention_forward)
    _MAGI_REGISTERED = True
    logger.info("Registered 'magi' attention backend (MagiAttention FFA) in transformers.")


def get_cp_group(device_mesh) -> Optional[dist.ProcessGroup]:
    """Return the CP process group from the device mesh (size-1 group is fine)."""
    if device_mesh is None:
        return None
    dim_names = getattr(device_mesh, "mesh_dim_names", None) or ()
    if "cp" in dim_names:
        return device_mesh["cp"].get_group()
    return None


def _get_head_config(model) -> tuple[int, int, int]:
    """Extract (num_heads_q, num_heads_kv, head_dim) from an HF model/config.

    For VLMs the text attention dims live under ``config.text_config``; prefer that
    sub-config when the top-level config does not expose ``num_attention_heads``."""
    cfg = getattr(model, "config", model)
    if not hasattr(cfg, "num_attention_heads") and hasattr(cfg, "text_config"):
        cfg = cfg.text_config
    num_heads_q = getattr(cfg, "num_attention_heads")
    num_heads_kv = getattr(cfg, "num_key_value_heads", None) or num_heads_q
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None:
        head_dim = cfg.hidden_size // num_heads_q
    return int(num_heads_q), int(num_heads_kv), int(head_dim)


def _set_cp_group_on_attention(model, cp_group) -> None:
    """Stamp ``cp_group`` on every attention sub-module so the FFA forward finds the key."""
    for module in model.modules():
        if "Attention" in type(module).__name__:
            module.cp_group = cp_group


def magi_prepare_batch(  # pragma: no cover - requires GPU + magi_attention
    model,
    batch: dict,
    cp_group: Optional[dist.ProcessGroup],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """Dispatch a (batch_size==1) sequence for MagiAttention on the HF path.

    Builds a causal varlen dist-attn key over the single sequence and dispatches
    ``input_ids``, ``position_ids`` *and* ``labels`` across ``cp_group`` (identity
    shard at cp_size==1; load-balanced sharding at cp_size>1). Labels are sharded the
    same way as the input so the loss is computed per-shard and summed across CP — no
    logit undispatch; ``MaskedCrossEntropy`` does not shift, so the dispatch
    permutation is harmless (logits[j] stay paired with labels[j]). The FFA kernel
    does the cross-rank KV exchange via the stamped ``cp_group`` + the dist key.

    Args:
        model: the (possibly FSDP-wrapped) HF causal-LM.
        batch: dict with at least ``input_ids`` of shape ``[1, S]``.
        cp_group: CP process group (size 1 -> identity shard).
        chunk_size: dispatch solver chunk size.

    Returns:
        (new_batch, key): ``new_batch`` has dispatched ``input_ids``/``position_ids``/
        ``labels`` (each ``[1, local_S]``); ``key`` is the dist-attn runtime key.
    """
    from magi_attention.api import dispatch, get_position_ids, magi_attn_varlen_key
    from magi_attention.api.functools import compute_pad_size

    input_ids = batch["input_ids"]
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"this integration maps magi to a single flat (varlen) sequence per micro-step "
            f"(local_batch_size=1), got input_ids shape {tuple(input_ids.shape)}. "
            "Set step_scheduler.local_batch_size=1 and pack the batch via packed_sequence "
            "(magi itself has no batch dim; an outer batch>1 would need a block-diagonal "
            "varlen pack over the batch elements, which is not wired yet)."
        )
    device = input_ids.device
    seqlen = input_ids.shape[1]
    cp_size = cp_group.size() if cp_group is not None else 1

    num_heads_q, num_heads_kv, head_dim = _get_head_config(model.module if hasattr(model, "module") else model)

    # Single full causal sequence -> cu_seqlens = [0, S].
    cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int32, device=device)
    pad_size = compute_pad_size(seqlen, cp_size, chunk_size=chunk_size)

    key = magi_attn_varlen_key(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        pad_size=pad_size,
        cp_group_or_mesh=cp_group,
        causal=True,
        chunk_size=chunk_size,
    )

    local_input = dispatch(input_ids.squeeze(0), key=key).unsqueeze(0)  # [1, local_S]
    position_ids = get_position_ids(key).unsqueeze(0).to(device)  # [1, local_S]

    _set_cp_group_on_attention(model.module if hasattr(model, "module") else model, cp_group)

    new_batch = dict(batch)
    new_batch["input_ids"] = local_input
    new_batch["position_ids"] = position_ids
    # Shard labels the same way as the input -> per-shard loss (like the custom path).
    # pad_value=-100 keeps the trailing dispatch padding out of the loss (ignore_index).
    if batch.get("labels") is not None:
        new_batch["labels"] = dispatch(batch["labels"].squeeze(0), key=key, pad_value=-100).unsqueeze(0)
    # Remove anything that no longer matches the dispatched layout.
    new_batch.pop("attention_mask", None)
    return new_batch, key


def _packed_cp_doc_seqlens(batch: dict, total_len: int) -> list:
    """Resolve per-document lengths spanning the *padded* THD layout of length ``total_len``.

    The TE collater pads each document for the THD layout, so ``cu_seqlens_padded``
    spans the full flat ``input_ids`` while ``cu_seqlens`` covers only the real
    tokens. magi dispatches the whole flat sequence, so the dist key must be built
    over the padded layout — otherwise the dispatched shard length (from
    ``input_ids``) won't match ``get_position_ids`` (built from the key), which
    surfaces downstream as a RoPE q vs cos/sin length mismatch. Causal masking keeps
    real tokens from attending the trailing per-document pad, and pad-token rows are
    dropped by the loss (labels == ignore_index), so this is numerically equivalent
    to attending only the real tokens.

    Raises:
        ValueError: if the resolved document layout does not span ``total_len``.
    """
    cu = batch.get("cu_seqlens_padded")
    if cu is None:
        cu = batch["cu_seqlens"]
    cu = cu.reshape(-1)
    cu = cu[cu >= 0]  # drop -1000 padding sentinels if present
    seqlens = (cu[1:] - cu[:-1]).tolist()
    if sum(seqlens) != total_len:
        raise ValueError(
            f"magi packed-CP: document layout total {sum(seqlens)} != flat input length {total_len}; "
            "expected cu_seqlens_padded to span the padded THD layout."
        )
    return seqlens


def magi_prepare_packed_cp(model, batch: dict, cp_group):  # pragma: no cover - requires GPU + magi_attention
    """Context-parallel prep for a packed (THD) batch on the custom-model path.

    Takes a *global* THD batch (flat ``input_ids``/``labels``/``position_ids`` plus
    ``cu_seqlens`` marking document boundaries), builds a per-document varlen dist
    key over ``cp_group`` and dispatches the sequence with MagiAttention's own
    load-balancing solver (not TE's THD sharding). Each rank then runs the model on
    its local shard; the attn_func uses this pre-built key so the FFA kernel does
    the cross-rank KV exchange. Labels are dispatched the same way as the input, so
    each rank computes a per-shard loss that the recipe's cross-CP reduction sums
    into the global loss (like TE-CP).

    Returns:
        (new_batch, key): ``new_batch`` has the local ``input_ids``/``position_ids``
        and local ``labels``; ``key`` is the dist-attn runtime key.
    """
    from magi_attention.api import dispatch, get_position_ids

    input_ids = batch["input_ids"].reshape(-1)  # global flat (padded THD) [T]
    seqlens = _packed_cp_doc_seqlens(batch, int(input_ids.numel()))
    spec = AttnMaskSpec.varlen(seqlens, causal=True)
    num_heads_q, num_heads_kv, head_dim = _get_head_config(model.module if hasattr(model, "module") else model)
    key = build_flex_key(
        spec,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        cp_group=cp_group,
    )
    local_input = dispatch(input_ids, key=key)
    local_pos = get_position_ids(key).to(local_input.device)
    # Shard labels the same way as the input (like TE-CP): each rank computes the
    # loss on its own shard and the recipe's cross-CP reduction sums the shards
    # into the global loss. (Undispatching logits to global instead would make
    # every CP rank compute the full loss redundantly -> the reduction would
    # double-count it by a factor of cp_size.)
    local_labels = dispatch(batch["labels"].reshape(-1), key=key)
    new_batch = {
        "input_ids": local_input,
        "position_ids": local_pos,
        "labels": local_labels,
        # THD layout: the model must use the thd RoPE path and our dispatched
        # (permuted) positions. Requires non-fused RoPE (backend.rope_fusion=false),
        # since the fused thd path rebuilds contiguous positions that ignore the
        # magi dispatch permutation.
        "qkv_format": "thd",
    }
    return new_batch, key


def _iter_language_model_attention(model):
    """Yield attention sub-modules belonging to the *language* backbone only.

    For VLMs we must leave the vision tower on its own (bidirectional) attention.
    HF VLMs nest the text stack under a ``language_model``/``model.language_model``
    attribute; we walk only that subtree. Falls back to the whole model for plain
    LLMs (no language_model attribute)."""
    root = model.module if hasattr(model, "module") else model
    lm = None
    for attr in ("language_model", "model"):
        sub = getattr(root, attr, None)
        if sub is not None and (hasattr(sub, "language_model") or attr == "language_model"):
            lm = getattr(sub, "language_model", sub)
            break
    if lm is None:
        lm = getattr(root, "language_model", root)
    for module in lm.modules():
        if "Attention" in type(module).__name__:
            yield module


def magi_prepare_vlm(
    model,
    batch: dict,
    cp_group: Optional[dist.ProcessGroup],
):
    """Prepare a VLM (bs==1) step for MagiAttention on the language backbone.

    Unlike the LLM path, the VLM merges image features into ``inputs_embeds``
    *inside* its forward, so we cannot dispatch ``input_ids`` (image-placeholder
    positions must stay put). For ``cp_size == 1`` we instead build a no-padding
    causal key (``chunk_size=1`` -> dispatch/undispatch are identity), stamp the
    cp_group on the language-model attention modules only, and let the FFA kernel
    run on the natural-length q/k/v. The vision tower falls back to SDPA.

    Returns the (unchanged) batch and ``None`` (the key is built lazily in the
    attention forward from the real query length).
    """
    input_ids = batch.get("input_ids")
    if input_ids is None or input_ids.shape[0] != 1:
        raise ValueError("magi VLM path requires input_ids of shape [1, S] (local_batch_size=1).")
    cp_size = cp_group.size() if cp_group is not None else 1
    if cp_size != 1:
        raise NotImplementedError("magi VLM path currently supports cp_size==1 (kernel-swap parity).")

    # Stamp cp_group + self-key flag on the language-backbone attention only; the
    # forward builds the causal key from the actual post-merge query length. The
    # vision tower keeps no cp_group and falls back to SDPA.
    for module in _iter_language_model_attention(model):
        module.cp_group = cp_group
        module._magi_self_key = True
    return batch, None


@dataclass
class MagiState:
    """Resolved MagiAttention wiring for a recipe, produced by :func:`setup_magi`.

    A single handle (stored as ``self.magi``) replacing the scattered
    ``magi_enabled``/``magi_custom``/``magi_cp_group``/``magi_cp_size`` recipe
    attributes. When MagiAttention is not configured, ``enabled`` is False and the
    per-step methods are no-ops, so recipes can call them unconditionally.
    """

    enabled: bool = False
    custom: bool = False  # custom-model factory backend (vs HF attn_implementation)
    cp_group: Optional["dist.ProcessGroup"] = None
    cp_size: int = 1

    @property
    def hf_dispatch(self) -> bool:
        """HF path: dispatch the sequence (input + labels) across CP for a per-shard loss.

        Distinguishes the HF ``attn_implementation=magi`` path (single causal sequence,
        :func:`magi_prepare_batch`) from the custom-model factory path; both shard labels
        and compute a per-shard loss at cp>1, so neither undispatches logits.
        """
        return self.enabled and not self.custom

    def prepare_llm_batch(
        self, model, batch, *, device_mesh, is_thd, pad_id, num_chunks
    ):  # pragma: no cover - requires GPU + magi_attention
        """Per-step batch prep for the LLM recipe (assumes ``enabled``).

        Returns ``(train_ctx, batch)``. magi does its own CP, so ``train_ctx`` is
        always ``nullcontext`` (no torch-native DTensor CP context).
        """
        # cp=1 arbitrary mask spec (e.g. prefix tree): activate the per-step spec the
        # collate attached, out-of-band, for the magi attn_func. Setting it every step
        # (the batch's spec or None) is self-clearing, so a stale spec never leaks into
        # the next batch; plain batches omit "magi_attn_spec".
        set_active_attn_spec(batch.pop("magi_attn_spec", None))
        if self.hf_dispatch:
            # HF path: dispatch the (single causal) sequence across the CP group.
            batch, _ = magi_prepare_batch(model, batch, self.cp_group)
        elif self.custom and self.cp_size > 1 and is_thd:
            # Custom-model CP packed path: build the *global* THD layout (no TE
            # sharding) then dispatch it with magi's own load-balancing solver.
            batch = make_cp_batch_for_te(None, batch, qkv_format="thd", padding_token_id=pad_id, num_chunks=1)
            batch, _ = magi_prepare_packed_cp(model, batch, self.cp_group)
        elif is_thd:
            # cp=1 packing: THD conversion (no sharding) so the batch carries
            # cu_seqlens -> the magi attn_func builds the per-document mask.
            _, batch = make_cp_batch_and_ctx(
                device_mesh, batch, use_te=True, padding_token_id=pad_id, num_chunks=num_chunks
            )
        return nullcontext, batch

    def prepare_vlm_batch(self, model, batch):
        """Per-step batch prep for the VLM recipe (assumes ``enabled``).

        HF VLMs stamp the cp_group on the language-backbone attention; custom VLMs
        use the factory attn_func with the active cp_group set in :func:`setup_magi`
        (the vision tower stays on SDPA either way). Returns ``(train_ctx, batch)``.
        """
        if not self.custom:
            batch, _ = magi_prepare_vlm(model, batch, self.cp_group)
        return nullcontext, batch


def setup_magi(cfg, device_mesh, *, label: str = "") -> MagiState:
    """Resolve MagiAttention from config: register the backend and CP group.

    Enabled when the model is configured with ``attn_implementation="magi"`` (HF) or
    ``backend.attn="magi"`` (custom models). Returns a :class:`MagiState`
    (``enabled=False`` when magi is not configured). ``label`` is an optional suffix
    for the log line (e.g. ``"VLM language backbone"``).
    """
    custom = str(cfg.get("model.backend.attn", "")) == "magi"
    enabled = custom or str(cfg.get("model.attn_implementation", "")) == "magi"
    if not enabled:
        return MagiState()

    if not is_magi_available():
        raise RuntimeError(
            "The 'magi' attention backend is configured (model.attn_implementation='magi' or "
            "model.backend.attn='magi') but the 'magi_attention' package is not importable. "
            "It is a source-only CUDA build (not published on PyPI): build it from source per "
            "https://github.com/SandAI-org/MagiAttention (see its install docs) and make it "
            "importable (a `pip install -e .` editable install, or on PYTHONPATH)."
        )

    register_magi_attention()
    cp_group = get_cp_group(device_mesh)
    cp_size = cp_group.size() if cp_group is not None else 1
    if custom:
        # the custom-model attn_func is a closure; it reads the cp_group from here.
        set_active_cp_group(cp_group)
    logger.info(
        "MagiAttention enabled%s (%s, cp_size=%d).",
        f" for {label}" if label else "",
        "custom-model factory" if custom else "HF backend",
        cp_size,
    )
    return MagiState(enabled=True, custom=custom, cp_group=cp_group, cp_size=cp_size)

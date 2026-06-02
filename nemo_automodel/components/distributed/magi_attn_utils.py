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
   ``input_ids``/``position_ids`` across the CP group and stamps ``cp_group`` on
   every attention sub-module so the registered forward can find the key.
3. ``magi_undispatch_logits()`` gathers the sharded logits back to the global
   (unpadded) sequence before the loss is computed, so the loss is numerically
   identical to a non-CP run.

When ``cp_size == 1`` the dispatch is a no-op shard (identity + chunk padding),
so this path is also a clean way to swap *only* the attention kernel (FFA) in
place of eager/SDPA/flash for convergence-parity comparisons.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist

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
                    q_ranges.append(list(q_rng)); k_ranges.append(list(q_rng)); mask_types.append("causal")
                # ancestors -> full
                for anc in path[:pos]:
                    k_rng = (offsets[anc], offsets[anc] + node_lengths[anc])
                    rect = (q_rng, k_rng, "full")
                    if rect not in seen:
                        seen.add(rect)
                        q_ranges.append(list(q_rng)); k_ranges.append(list(k_rng)); mask_types.append("full")
        sample_token_ranges = [
            [[offsets[n], offsets[n] + node_lengths[n]] for n in path] for path in sample_paths
        ]
        return cls(q_ranges, k_ranges, mask_types, total), sample_token_ranges


def set_active_attn_spec(spec: Optional["AttnMaskSpec"]) -> None:
    """Set the mask spec the custom-model magi attn_func should apply this step."""
    global _ACTIVE_ATTN_SPEC
    _ACTIVE_ATTN_SPEC = spec


def get_active_attn_spec() -> Optional["AttnMaskSpec"]:
    """Return the active mask spec (None -> plain causal self-key)."""
    return _ACTIVE_ATTN_SPEC


def build_flex_key(spec: "AttnMaskSpec", *, num_heads_q, num_heads_kv, head_dim, cp_group):
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


def _flex_key_for(cp_group, spec, *, num_heads_q, num_heads_kv, head_dim):
    """Return the flex key for ``spec``, rebuilding only when the mask changes."""
    fp = spec.fingerprint()
    cached = _FLEX_KEY_CACHE.get(id(cp_group))
    if cached is not None and cached[0] == fp:
        return cached[1]
    key = build_flex_key(
        spec, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
        head_dim=head_dim, cp_group=cp_group,
    )
    _FLEX_KEY_CACHE[id(cp_group)] = (fp, key)
    return key


def _build_self_key(cp_group, *, seqlen, num_heads_q, num_heads_kv, head_dim, device):
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


def _self_key_for(cp_group, *, seqlen, num_heads_q, num_heads_kv, head_dim, device):
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
        cp_group, seqlen=seqlen, num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv, head_dim=head_dim, device=device,
    )
    _MAGI_SELF_KEY_LEN[id(cp_group)] = seqlen
    return key


def make_magi_attn_func(softmax_scale: Optional[float] = None):
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

    from magi_attention.api import calc_attn

    def magi_attn_func(q, k, v, **call_kwargs):
        cp_group = get_active_cp_group()
        if cp_group is None:
            import torch.nn.functional as F

            qh, kh, vh = (e.transpose(1, 2) if e.dim() == 4 else e for e in (q, k, v))
            return F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)

        bshd = q.dim() == 4
        if bshd:
            b = q.shape[0]
            qf, kf, vf = (rearrange(e, "b s nh hd -> (b s) nh hd") for e in (q, k, v))
        else:
            qf, kf, vf = q, k, v
        dtype = qf.dtype
        qf, kf, vf = (e.to(torch.bfloat16).contiguous() for e in (qf, kf, vf))
        spec = get_active_attn_spec()
        if spec is not None:
            key = _flex_key_for(
                cp_group, spec, num_heads_q=qf.shape[1],
                num_heads_kv=kf.shape[1], head_dim=qf.shape[2],
            )
        else:
            key = _self_key_for(
                cp_group, seqlen=qf.shape[0], num_heads_q=qf.shape[1],
                num_heads_kv=kf.shape[1], head_dim=qf.shape[2], device=qf.device,
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


def register_magi_attention() -> None:
    """Register the ``"magi"`` attention backend in HF transformers (idempotent)."""
    global _MAGI_REGISTERED
    if _MAGI_REGISTERED:
        return

    from einops import rearrange
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from magi_attention.api import calc_attn, get_most_recent_key

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
                module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
            )

        if getattr(module, "_magi_self_key", False):
            # VLM (cp_size==1) path: the post-image-merge LM sequence length is only
            # known here (query dim 2: [b, nh, s, hd]) and may differ from input_ids
            # length. Build a no-dispatch causal key matching the actual q length.
            magi_attn_key = _self_key_for(
                cp_group, seqlen=query.shape[2], num_heads_q=query.shape[1],
                num_heads_kv=key.shape[1], head_dim=query.shape[3], device=query.device,
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
            rearrange(e, "b nh s hd -> (b s) nh hd").to(torch.bfloat16).contiguous()
            for e in (query, key, value)
        )
        o = calc_attn(q, k, v, magi_attn_key)[0]
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


def magi_prepare_batch(
    model,
    batch: dict,
    cp_group: Optional[dist.ProcessGroup],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """Dispatch a (batch_size==1) packed sequence for MagiAttention.

    Builds a causal varlen dist-attn key over the single sequence, dispatches
    ``input_ids`` and ``position_ids`` across ``cp_group`` and stamps the cp_group
    on the model's attention modules.  ``labels`` are left global; the caller
    undispatches the logits before the loss via :func:`magi_undispatch_logits`.

    Args:
        model: the (possibly FSDP-wrapped) HF causal-LM.
        batch: dict with at least ``input_ids`` of shape ``[1, S]``.
        cp_group: CP process group (size 1 allowed -> identity shard).
        chunk_size: dispatch solver chunk size.

    Returns:
        (new_batch, key): ``new_batch`` has dispatched ``input_ids`` ``[1, local_S]``
        and ``position_ids`` ``[1, local_S]``; ``key`` is the dist-attn runtime key.
    """
    from magi_attention.api import dispatch, get_position_ids, magi_attn_varlen_key
    from magi_attention.api.functools import compute_pad_size

    input_ids = batch["input_ids"]
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(
            f"magi attention requires batch_size==1 packed sequences, got input_ids shape {tuple(input_ids.shape)}. "
            "Set step_scheduler.local_batch_size=1 (and use packing / do_not_pad)."
        )
    device = input_ids.device
    seqlen = input_ids.shape[1]
    cp_size = cp_group.size() if cp_group is not None else 1

    num_heads_q, num_heads_kv, head_dim = _get_head_config(
        model.module if hasattr(model, "module") else model
    )

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
    # Remove anything that no longer matches the dispatched layout.
    new_batch.pop("attention_mask", None)
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


def magi_undispatch_logits(
    logits: torch.Tensor, cp_group: Optional[dist.ProcessGroup]
) -> torch.Tensor:
    """Gather sharded/padded logits back to the global, unpadded sequence.

    Args:
        logits: model logits ``[1, local_S, V]`` (or ``[local_S, V]``).
        cp_group: CP process group used for the matching dispatch.

    Returns:
        Global logits ``[1, S, V]``.
    """
    from magi_attention.api import get_most_recent_key, undispatch

    key = get_most_recent_key(cp_group)
    if key is None:
        return logits
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    logits = undispatch(logits, key)
    return logits.unsqueeze(0)

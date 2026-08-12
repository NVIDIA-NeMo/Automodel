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

"""Carrying two hidden-state streams through a decoder stack.

The expansion needs a second hidden state alongside the pretrained one. It is carried as
an ``(h_a, h_b)`` tuple passed from one decoder layer to the next, rather than by stacking
the streams on the batch axis. Stacking was ruled out by sequence packing and MoE: both
flatten the batch axis into a token stream, and MoE additionally permutes it per routing
decision, which destroys the token-to-token correspondence the lateral term needs.

With a tuple, each stream is handed to the *unmodified* decoder layer with the
*unmodified* ``attention_mask`` and ``position_ids``, so packing, ``cu_seqlens``, context
parallelism and MoE dispatch keep working exactly as they do without expansion.

Two per-layer modes:

``expand``
    The layer owns expansion weights. Its forward runs twice, once per stream. The A pass
    records each expanded linear's output; the B pass consumes it as the lateral term.
    For a MoE layer this is also what makes routing alignable: with
    ``RouterReplay`` replaying the A pass's expert selection, both passes dispatch tokens
    in the same order, so the lateral applies elementwise.

``skip``
    The layer owns no expansion weight, which is exactly the case ``W_b == 0``.
    Substituting zero into ``y_b = W_b x_b + y_a`` makes every stream-B intermediate --
    its norms, attention and MLP results -- multiply by zero and vanish, leaving only
    stream A's residual contributions::

        h_b' = h_b + (h_a' - h_a) = h_a' + (h_b - h_a)

    So the layer runs *once*, on stream A, and forwards its residual delta to stream B.
    This is exact, not an approximation, and costs one layer instead of two.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
import torch.nn as nn

__all__ = ["ExpansionMode", "patch_layer_for_expansion", "patch_model_for_pipeline", "patch_norm_for_merge"]

ExpansionMode = str  # "expand" | "skip"

# Dynamically created subclasses, keyed so a decoder-layer class is only subclassed once.
_PATCHED_CLASSES: dict[tuple[str, type], type] = {}


def _split_streams(hidden_states: torch.Tensor, hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(h_a, h_b)`` pair, seeding stream B on the first patched layer.

    Args:
        hidden_states: Either ``[batch, seq, hidden]`` -- the first patched layer in the
            stack receives the embedding output -- or ``[batch, seq, 2 * hidden]``, the
            carrier a previous patched layer produced. The width is what distinguishes
            them, which is why this needs to be told the model's hidden size.
        hidden_size: The model's hidden size, i.e. the width of a single stream.

    Returns:
        ``(h_a, h_b)``, each ``[batch, seq, hidden]``. Seeding stream B from stream A is
        why the two are identical at initialization, and hence why the merge collapses to
        the pretrained result whatever the merge weight.
    """
    if hidden_states.shape[-1] == 2 * hidden_size:
        h_a, h_b = hidden_states.chunk(2, dim=-1)
        return h_a, h_b
    return hidden_states, hidden_states


def _join_streams(h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
    """Pack a stream pair back into the single tensor passed between layers.

    Args:
        h_a: Stream A, ``[batch, seq, hidden]``.
        h_b: Stream B, same shape.

    Returns:
        ``[batch, seq, 2 * hidden]``, contiguous, owning its storage.
    """
    return torch.cat((h_a, h_b), dim=-1)


def _stream_b_kwargs(hidden_states: torch.Tensor, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Cache arguments for the stream-B pass of an expanded layer.

    Stream A owns the layer's KV cache and uses it normally. Stream B cannot share it: its
    hidden states differ once the expansion weights learn, so its keys and values differ,
    and appending both to one cache makes attention run against a mix of the two. Serving
    stream B without a cache is exact whenever the pass sees the whole sequence, which
    covers training and prefill; it is only decoding, where the layer is handed one token
    and the history lives in the cache, that stream B genuinely needs a cache of its own.

    Args:
        hidden_states: Stream B's input, of shape ``[batch, sequence, hidden]``. Only its
            sequence length is read, to tell a whole-sequence pass from a decode step.
        kwargs: Keyword arguments destined for the decoder layer's ``forward``.

    Returns:
        A copy with the cache arguments neutralized, so stream B recomputes rather than
        reading or writing stream A's cache.

    Raises:
        NotImplementedError: When the layer is decoding, which stream B cannot serve
            without a second cache. Refusing beats returning plausible wrong tokens.
    """
    cache = kwargs.get("past_key_values")
    cached_length = cache.get_seq_length() if cache is not None else 0
    if cached_length > hidden_states.shape[1]:
        raise NotImplementedError(
            "model expansion does not support generation with a KV cache: stream B would "
            f"need a cache of its own, and this layer was handed {hidden_states.shape[1]} "
            f"token(s) against {cached_length} cached. Generate with use_cache=False -- "
            "correct, and quadratic in the generated length."
        )
    stripped = dict(kwargs)
    stripped["past_key_values"] = None
    stripped["use_cache"] = False
    return stripped


def _dual_stream_forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    """Decoder-layer forward carrying two hidden-state streams.

    Args:
        hidden_states: ``[batch, seq, hidden]`` on the first patched layer, otherwise the
            ``(h_a, h_b)`` pair from the previous one.
        *args: Forwarded to the pretrained layer unchanged.
        **kwargs: Forwarded to the pretrained layer with cache arguments stripped.

    Returns:
        ``(h_a, h_b)``, each ``[batch, seq, hidden]``.
    """
    from nemo_automodel.components.expansion.expanded_linear import LateralBus, LateralBusMode
    from nemo_automodel.components.expansion.routing import aligned_routing, switch_to_replay

    h_a, h_b = _split_streams(hidden_states, self._expansion_hidden_size)
    # Not ``super(type(self), self)``: ``fully_shard`` also patches ``__class__``, and that
    # form would re-enter this function through the new subclass and recurse. Bind the
    # class captured at patch time so the target is fixed however many layers wrap us.
    pretrained_forward = partial(self._expansion_base_cls.forward, self)

    if self._expansion_mode == "skip":
        # One pass, on stream A, with the layer's arguments untouched -- including its KV
        # cache, which stream A owns and uses exactly as it would without expansion.
        h_a_out = pretrained_forward(h_a, *args, **kwargs)
        # Accumulated in fp32 so that h_b == h_a at initialization reproduces h_a_out
        # bit-exactly, and so the difference of two same-magnitude residual streams does
        # not lose precision once they diverge.
        h_b_out = (h_a_out.float() + (h_b.float() - h_a.float())).to(h_a_out.dtype)
        return _join_streams(h_a_out, h_b_out)

    # A MoE layer routes from the hidden state, so the two streams pick different experts
    # as soon as they diverge and the lateral term stops lining up. Pin stream B to stream
    # A's expert selection for the duration of the pair.
    b_kwargs = _stream_b_kwargs(h_b, kwargs)
    previous = LateralBus.set_mode(LateralBusMode.RECORD)
    try:
        with aligned_routing(self._expansion_router_replays):
            h_a_out = pretrained_forward(h_a, *args, **kwargs)
            LateralBus.set_mode(LateralBusMode.APPLY)
            switch_to_replay(self._expansion_router_replays)
            h_b_out = pretrained_forward(h_b, *args, **b_kwargs)
    finally:
        LateralBus.set_mode(previous)
    return _join_streams(h_a_out, h_b_out)


def _merge_then_norm(self, hidden_states: Any) -> torch.Tensor:
    """Collapse the stream pair, then apply the pretrained norm.

    Merging before the final norm rather than after is deliberate. Merging afterwards
    would average two already-normalized vectors, whose mean has a smaller norm than
    either: the logit scale would be capped at the pretrained level and would shrink
    monotonically as stream B diverges from stream A.

    Args:
        hidden_states: ``(h_a, h_b)`` from the last patched layer, or a plain
            ``[batch, seq, hidden]`` tensor when expansion is inactive.

    Returns:
        ``[batch, seq, hidden]``.
    """
    if hidden_states.shape[-1] == 2 * self._merge_hidden_size:
        h_a, h_b = hidden_states.chunk(2, dim=-1)
        # lerp(a, b, w) == a + w * (b - a); in fp32 so that h_b == h_a reproduces h_a.
        hidden_states = torch.lerp(h_a.float(), h_b.float(), self._merge_weight).to(h_a.dtype)
    return self._merge_base_cls.forward(self, hidden_states)


def _patch_class(module: nn.Module, kind: str, forward: Any, base_attr: str) -> nn.Module:
    """Replace ``module.__class__`` with a cached subclass overriding ``forward``."""
    return _patch_class_with(module, kind, {"forward": forward, base_attr: type(module)})


def _patch_class_with(module: nn.Module, kind: str, namespace: dict[str, Any]) -> nn.Module:
    """Replace ``module.__class__`` with a cached subclass carrying ``namespace``."""
    cls = type(module)
    key = (kind, cls)
    patched = _PATCHED_CLASSES.get(key)
    if patched is None:
        patched = type(f"{kind}{cls.__name__}", (cls,), namespace)
        _PATCHED_CLASSES[key] = patched
    module.__class__ = patched
    return module


def _pipeline_stage_metas(
    self, *, is_first: bool, microbatch_size: int, seq_len: int, dtype: torch.dtype
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Declare this pipeline stage's inter-stage tensor shapes.

    The pipeline precomputes stage metadata from ``config.hidden_size`` rather than
    inferring it, so it expects one stream's width where expansion sends two. This hook
    is the extension point it offers for exactly that: a model that moves something other
    than a plain hidden state between stages says so itself.

    Args:
        is_first: Whether this stage owns the embedding, and so takes token ids.
        microbatch_size: Rows per microbatch.
        seq_len: Sequence length.
        dtype: Dtype of the tensors crossing stage boundaries.

    Returns:
        ``(inputs_meta, outputs_meta)``, each a tuple of meta-device tensors. A stage takes
        ``[microbatch, seq]`` token ids when first and ``[microbatch, seq, 2 * hidden]``
        otherwise; it emits ``[microbatch, seq, vocab]`` when it owns the LM head -- the
        streams are merged before it -- and ``[microbatch, seq, 2 * hidden]`` otherwise.
    """
    hidden_size = self._expansion_stage_hidden_size
    carrier = torch.empty(microbatch_size, seq_len, 2 * hidden_size, device="meta", dtype=dtype)
    if is_first:
        inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
    else:
        inputs_meta = (carrier,)

    lm_head = getattr(self, "lm_head", None)
    if lm_head is not None:
        outputs_meta = (
            torch.empty(microbatch_size, seq_len, self._expansion_stage_vocab_size, device="meta", dtype=dtype),
        )
    else:
        outputs_meta = (carrier,)
    return inputs_meta, outputs_meta


def patch_model_for_pipeline(model: nn.Module, hidden_size: int, vocab_size: int) -> nn.Module:
    """Teach the pipeline that this model moves two streams between stages, in place.

    Args:
        model: The causal-LM being expanded.
        hidden_size: Width of a single stream.
        vocab_size: Width of the LM head's output, for the final stage.

    Returns:
        The same object, now answering ``get_pipeline_stage_metas``.
    """
    _patch_class_with(model, "PipelineMetas", {"get_pipeline_stage_metas": _pipeline_stage_metas})
    model._expansion_stage_hidden_size = hidden_size
    model._expansion_stage_vocab_size = vocab_size
    return model


def patch_layer_for_expansion(layer: nn.Module, mode: ExpansionMode, hidden_size: int) -> nn.Module:
    """Give a decoder layer a second hidden-state stream, in place.

    Args:
        layer: The decoder layer to patch. Modified in place so its module path is
            preserved; wrapping would insert a path segment and stop tensor-parallel plan
            patterns, which match segment-wise, from resolving.
        mode: ``"expand"`` if this layer owns expansion weights, ``"skip"`` otherwise.
        hidden_size: Width of a single stream, used to tell an incoming carrier from the
            embedding output.

    Returns:
        The same object, now carrying two streams.
    """
    from nemo_automodel.components.expansion.routing import (
        find_router_replays,
        requires_routing_replay,
    )

    if mode == "expand":
        problem = requires_routing_replay(layer)
        if problem is not None:
            raise ValueError(f"cannot expand this layer: {problem}")
    _patch_class(layer, "DualStream", _dual_stream_forward, "_expansion_base_cls")
    layer._expansion_mode = mode
    layer._expansion_hidden_size = hidden_size
    # Resolved once at patch time; the registry is ordered by construction and says
    # nothing about which handle belongs to this layer.
    layer._expansion_router_replays = find_router_replays(layer) if mode == "expand" else []
    return layer


def patch_norm_for_merge(norm: nn.Module, merge_weight: float, hidden_size: int) -> nn.Module:
    """Make the final norm merge the stream pair before normalizing, in place.

    Args:
        norm: The decoder stack's final norm.
        merge_weight: ``lambda`` in ``h = h_a + lambda * (h_b - h_a)``. ``1.0`` keeps only
            the expanded stream, ``0.5`` averages the two. The value does not affect the
            model at initialization, where the streams are identical.
        hidden_size: Width of a single stream, used to recognize the carrier.

    Returns:
        The same object, now merging before it normalizes.
    """
    _patch_class(norm, "Merging", _merge_then_norm, "_merge_base_cls")
    norm._merge_weight = merge_weight
    norm._merge_hidden_size = hidden_size
    return norm

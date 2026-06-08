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

"""Selective activation checkpointing core.

TorchTitan-style selective activation checkpointing: the policy decides, per op,
whether to save or recompute an activation, saving the expensive ops (attention,
half of the matmuls, comm collectives) while recomputing the cheap ones.

This module holds the parts of the AC implementation that do not depend on the
rest of ``parallelizer.py`` (notably the heavy, transformers-aware
``_extract_model_layers``). ``parallelizer.py`` imports from here -- never the
other way around -- so the dependency stays one-directional and the central
parallelizer file stays small.
"""

import logging
import os
from typing import List

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

logger = logging.getLogger(__name__)


def _resolve_torch_op(namespace: str, name: str, overload: str = "default"):
    """Resolve ``torch.ops.<namespace>.<name>.<overload>``, or ``None`` if absent."""
    ns = getattr(torch.ops, namespace, None)
    packet = getattr(ns, name, None) if ns is not None else None
    return getattr(packet, overload, None) if packet is not None else None


def _resolve_op_attr(root: object, dotted_path: str):
    """Resolve a dotted attribute path from ``root``, or ``None`` if any part is absent.

    Used for ops that live outside ``torch.ops`` (higher-order ops, optional
    custom backends such as DeepEP/HybridEP). Missing namespaces/ops raise
    ``AttributeError`` on access, so they are swallowed and reported as ``None``.
    """
    obj = root
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _existing_ops(*ops):
    return frozenset(op for op in ops if op is not None)


# Matmul ops whose activations alternate between save and recompute (every other
# one is saved). Following TorchTitan, plain ``mm``/``linear`` alternate;
# ``addmm``/``bmm`` stay in the always-save set built below. The grouped-GEMM
# variants are the dominant compute in expert-parallel MoE blocks (custom
# ``torch._grouped_mm`` kernels), so they alternate too -- otherwise selective AC
# would recompute every expert GEMM, matching full checkpointing and giving no
# speedup while still paying the policy overhead.
_SELECTIVE_AC_MATMUL_OPS = _existing_ops(
    _resolve_torch_op("aten", "mm"),
    _resolve_torch_op("aten", "linear"),
    _resolve_torch_op("aten", "_grouped_mm"),
    _resolve_torch_op("aten", "_scaled_grouped_mm"),
)


def _default_compute_intensive_ops() -> tuple:
    """Compute-intensive aten ops from PyTorch's partitioner, or ``()`` if unavailable.

    Mirrors TorchTitan: seeding from PyTorch's own ``compute_intensive_ops`` list
    keeps the save-set in sync with upstream rather than relying on a frozen,
    hand-maintained list. ``torch._functorch.partitioners`` is a private API, so
    any failure falls back to the curated supplement in
    :func:`_build_selective_ac_save_ops`.
    """
    try:
        from torch._functorch.partitioners import get_default_op_list

        return tuple(op.default for op in get_default_op_list().compute_intensive_ops)
    except (ImportError, AttributeError):
        return ()


def _build_selective_ac_save_ops() -> frozenset:
    """Build the set of ops whose activations are always saved under selective AC.

    The set is seeded from PyTorch's compute-intensive op list and supplemented
    with attention variants, low-precision/reduction ops, the compiled HOP, and
    communication collectives whose outputs are expensive to recompute.
    """
    save_ops = set(_default_compute_intensive_ops())

    # Compute ops the partitioner list may not classify as compute-intensive.
    compute_ops = _existing_ops(
        _resolve_torch_op("aten", "mm"),
        _resolve_torch_op("aten", "addmm"),
        _resolve_torch_op("aten", "bmm"),
        _resolve_torch_op("aten", "linear"),
        _resolve_torch_op("aten", "_scaled_mm"),
        _resolve_torch_op("aten", "_scaled_dot_product_cudnn_attention"),
        _resolve_torch_op("aten", "_scaled_dot_product_efficient_attention"),
        _resolve_torch_op("aten", "_scaled_dot_product_flash_attention"),
        _resolve_torch_op("aten", "_scaled_dot_product_flash_attention_for_cpu"),
        _resolve_torch_op("aten", "_scaled_dot_product_fused_attention_overrideable"),
        _resolve_torch_op("aten", "scaled_dot_product_attention"),
        _resolve_torch_op("aten", "_flex_attention"),
        # topk is saved to keep MoE expert assignments stable across recompute;
        # max is saved for low-precision scaling factors.
        _resolve_torch_op("aten", "topk"),
        _resolve_torch_op("aten", "max"),
        # FlexAttention HOP and the inductor compiled-graph HOP (present only when
        # torch.compile is used); custom torch_attn varlen backend.
        _resolve_op_attr(torch, "_higher_order_ops.flex_attention"),
        _resolve_op_attr(torch, "_higher_order_ops.inductor_compiled_code"),
        _resolve_op_attr(torch.ops, "torch_attn._varlen_attn.default"),
    )

    # Communication ops whose outputs should be saved to avoid re-communication.
    comm_ops = _existing_ops(
        _resolve_torch_op("aten", "all_to_all_single"),
        _resolve_torch_op("aten", "reduce_scatter_tensor"),
        _resolve_torch_op("_c10d_functional", "all_to_all_single"),
        _resolve_torch_op("_c10d_functional", "reduce_scatter_tensor"),
        # Optional expert-parallel comm backends.
        _resolve_op_attr(torch.ops, "deepep.dispatch.default"),
        _resolve_op_attr(torch.ops, "deepep.combine.default"),
        _resolve_op_attr(torch.ops, "hybridep.dispatch.default"),
        _resolve_op_attr(torch.ops, "hybridep.combine.default"),
    )

    save_ops.update(compute_ops)
    save_ops.update(comm_ops)
    return frozenset(save_ops)


_SELECTIVE_AC_MUST_SAVE_OPS = _build_selective_ac_save_ops()

_SELECTIVE_AC_TO_COPY_OP = _resolve_torch_op("aten", "_to_copy")


def is_selective_activation_checkpointing(activation_checkpointing: object) -> bool:
    """Return whether the config value selects selective activation checkpointing.

    Args:
        activation_checkpointing: The configured value (bool or string such as
            ``"selective"``/``"full"``).

    Returns:
        bool: ``True`` only for the string ``"selective"`` (case- and
        hyphen/underscore-insensitive).
    """
    return (
        isinstance(activation_checkpointing, str) and activation_checkpointing.lower().replace("-", "_") == "selective"
    )


def _is_cuda_to_cpu_copy(func, args, kwargs) -> bool:
    if func != _SELECTIVE_AC_TO_COPY_OP or not args:
        return False
    tensor = args[0]
    src_device = getattr(tensor, "device", None)
    target_device = kwargs.get("device")
    if target_device is None:
        return False
    try:
        target_device = torch.device(target_device)
    except (TypeError, RuntimeError):
        return False
    return getattr(src_device, "type", None) == "cuda" and target_device.type == "cpu"


# Opt-in diagnostics: set NEMO_SELECTIVE_AC_TRACE=1 to log, once per unique op,
# whether selective AC saves or recomputes it. Useful for confirming that a
# model's expensive ops (e.g. expert grouped-GEMMs, comm collectives) are
# actually saved rather than silently recomputed.
_SELECTIVE_AC_TRACE = os.environ.get("NEMO_SELECTIVE_AC_TRACE", "0").lower() not in ("0", "", "false", "no")
_SELECTIVE_AC_TRACE_SEEN: set[str] = set()


def _maybe_trace_selective_ac_decision(func, decision, is_alternating: bool, *, is_recompute: bool) -> None:
    """Log a selective-AC decision once per op (no-op unless tracing is enabled).

    Args:
        func: The op the policy was queried about.
        decision: The ``CheckpointPolicy`` the policy returned for ``func``.
        is_alternating: Whether ``func`` is an alternating-save matmul op.
        is_recompute: Whether the policy was queried during the recompute pass;
            decisions are only logged on the forward pass to avoid duplicates.
    """
    if not _SELECTIVE_AC_TRACE or is_recompute:
        return
    key = str(func)
    if key in _SELECTIVE_AC_TRACE_SEEN:
        return
    _SELECTIVE_AC_TRACE_SEEN.add(key)
    if is_alternating:
        verdict = "ALTERNATE (save/recompute every other call)"
    elif decision == CheckpointPolicy.MUST_SAVE:
        verdict = "SAVE"
    else:
        verdict = "RECOMPUTE"
    logger.info("[selective-ac] %s -> %s", key, verdict)


def make_selective_checkpoint_context_fn():
    """Build a TorchTitan-style selective activation checkpointing context."""

    def selective_checkpointing_context_fn():
        # Count matmuls separately for the forward and recompute passes. torch
        # calls ``context_fn`` once per checkpointed region, so a single shared
        # counter would continue from the forward count into recompute and flip
        # the save/recompute parity whenever the region has an odd number of
        # matmuls. Keying on ``ctx.is_recompute`` resets each pass to 0 so the
        # same matmul gets the same decision in both passes.
        mm_counts = {False: 0, True: 0}

        def selective_checkpointing_policy(ctx, func, *args, **kwargs):
            is_alternating = func in _SELECTIVE_AC_MATMUL_OPS
            if is_alternating:
                mm_counts[ctx.is_recompute] += 1
                decision = (
                    CheckpointPolicy.PREFER_RECOMPUTE
                    if mm_counts[ctx.is_recompute] % 2 == 0
                    else CheckpointPolicy.MUST_SAVE
                )
            elif func in _SELECTIVE_AC_MUST_SAVE_OPS or _is_cuda_to_cpu_copy(func, args, kwargs):
                decision = CheckpointPolicy.MUST_SAVE
            else:
                decision = CheckpointPolicy.PREFER_RECOMPUTE
            _maybe_trace_selective_ac_decision(func, decision, is_alternating, is_recompute=ctx.is_recompute)
            return decision

        return create_selective_checkpoint_contexts(selective_checkpointing_policy)

    return selective_checkpointing_context_fn


# Marker set on whole-block selective-AC wrappers so the per-layer compile step
# compiles the wrapper itself (compile OUTER, SAC INNER) instead of unwrapping
# to the inner decoder layer. Compiling outer lets AOT autograd's partitioner
# read the SAC recompute tags; compiling inner would hide every aten op behind a
# single compiled HOP and collapse selective recompute into full recompute.
SELECTIVE_AC_WRAPPER_FLAG = "_nemo_selective_ac"


def _disable_dynamo_lru_cache() -> None:
    """Best-effort disable of TorchDynamo's LRU cache for selective AC + compile.

    With multiple pipeline microbatches, dynamo may compile a second graph with
    dynamic shapes and then select it over the static graph whose compiled-HOP
    output SAC cached for microbatch 0, tripping a missing-symint assertion.
    Selecting graphs in insertion order avoids this. Mirrors TorchTitan. The
    underlying API is private, so failures are swallowed.
    """
    try:
        torch._C._dynamo.eval_frame._set_lru_cache(False)
    except (AttributeError, RuntimeError):
        logger.debug("Could not disable dynamo LRU cache for selective AC + compile.", exc_info=True)


def apply_submodule_checkpointing(layers: List[nn.Module], has_kv_sharing: bool) -> None:
    """Wrap a transformer block's sub-modules with ``checkpoint_wrapper``.

    This is the sub-module granularity path used both as the default
    (non-compile) behavior and as the fallback for selective activation
    checkpointing on KV-shared models, which cannot checkpoint the whole block.

    ``self_attn`` is skipped for KV-shared models: recomputing attention during
    backward would double-write to the ``DynamicCache``, corrupting the K/V
    entries that later shared layers depend on.

    Args:
        layers: Transformer decoder layers to wrap (mutated in place).
        has_kv_sharing: Whether the model reuses K/V across layers via the cache.
    """
    for layer in layers:
        if hasattr(layer, "mlp"):
            layer.mlp = checkpoint_wrapper(layer.mlp)  # type: ignore
        if hasattr(layer, "self_attn") and not has_kv_sharing:
            layer.self_attn = checkpoint_wrapper(layer.self_attn)  # type: ignore
        if hasattr(layer, "input_layernorm"):
            layer.input_layernorm = checkpoint_wrapper(layer.input_layernorm)  # type: ignore
        if hasattr(layer, "post_attention_layernorm"):
            layer.post_attention_layernorm = checkpoint_wrapper(layer.post_attention_layernorm)  # type: ignore

        # MoT (mixture-of-transformers) sibling submodules -- present in BAGEL's
        # Qwen2MoTDecoderLayer for the generation expert. mlp_moe_gen is a full
        # Qwen2MLP duplicate (same size as mlp), so omitting it from AC roughly
        # doubles per-layer activation memory in Stage-2 BAGEL training.
        if hasattr(layer, "mlp_moe_gen"):
            layer.mlp_moe_gen = checkpoint_wrapper(layer.mlp_moe_gen)  # type: ignore
        if hasattr(layer, "input_layernorm_moe_gen"):
            layer.input_layernorm_moe_gen = checkpoint_wrapper(layer.input_layernorm_moe_gen)  # type: ignore
        if hasattr(layer, "post_attention_layernorm_moe_gen"):
            layer.post_attention_layernorm_moe_gen = checkpoint_wrapper(layer.post_attention_layernorm_moe_gen)  # type: ignore


def _replace_child_module(root: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
    """Replace ``target`` with ``replacement`` in ``root``'s module tree."""
    for name, child in root.named_children():
        if child is target:
            if isinstance(root, nn.ModuleList):
                root[int(name)] = replacement
            elif isinstance(root, nn.ModuleDict):
                root[name] = replacement
            else:
                setattr(root, name, replacement)
            return True
        if _replace_child_module(child, target, replacement):
            return True
    return False


def detect_kv_sharing_and_maybe_disable_cache(model: nn.Module) -> bool:
    """Detect KV-sharing and disable ``use_cache`` for non-KV-shared models.

    Models with KV-shared layers (e.g. Gemma4 2B/4B) pass K/V from earlier
    layers to later layers through the ``DynamicCache``; disabling the cache
    breaks that dependency, so ``use_cache`` is left untouched for them.

    Returns:
        bool: Whether the model uses KV-sharing.
    """
    text_cfg = getattr(getattr(model, "config", None), "text_config", None) or getattr(model, "config", None)
    has_kv_sharing = getattr(text_cfg, "num_kv_shared_layers", 0) > 0
    if not has_kv_sharing:
        if hasattr(model, "config") and getattr(model.config, "use_cache", None) is not False:
            try:
                model.config.use_cache = False
            except Exception:
                pass
    return has_kv_sharing


def apply_selective_checkpointing_to_layers(
    model: nn.Module,
    layers: List[nn.Module],
    has_kv_sharing: bool,
    *,
    enable_compile: bool = False,
) -> None:
    """Wrap whole transformer blocks with the selective-AC policy.

    KV-shared models cannot checkpoint attention through the ``DynamicCache``,
    so they fall back to sub-module checkpointing. ``layers`` is mutated in
    place so callers that retain the list (e.g. for subsequent FSDP sharding)
    see the wrapped modules. Works without FSDP/distributed, so it is shared by
    the FSDP2 strategy and the single-GPU path.
    """
    if has_kv_sharing:
        logger.warning(
            "Selective activation checkpointing is not supported for KV-shared models; "
            "falling back to sub-module activation checkpointing."
        )
        apply_submodule_checkpointing(layers, has_kv_sharing)
        return

    # With compile, the per-layer compile step compiles these wrappers OUTER so
    # the SAC policy is traced and respected by the partitioner; disable dynamo's
    # LRU cache to keep graph selection stable across pipeline microbatches.
    if enable_compile:
        _disable_dynamo_lru_cache()
    context_fn = make_selective_checkpoint_context_fn()
    for i, layer in enumerate(layers):
        wrapped_layer = checkpoint_wrapper(
            layer,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=context_fn,
            preserve_rng_state=True,
        )
        setattr(wrapped_layer, SELECTIVE_AC_WRAPPER_FLAG, True)
        if not _replace_child_module(model, layer, wrapped_layer):
            logger.warning("Could not replace layer %d with selective activation checkpoint wrapper.", i)
        layers[i] = wrapped_layer

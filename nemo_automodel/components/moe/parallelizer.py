# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import functools
import logging

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor import Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from nemo_automodel.components.distributed.pipelining.hf_utils import get_text_module
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP, GroupedExpertsTE
from nemo_automodel.components.moe.layers import (
    MoE,
)
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)
_CP_STREAM = None


def _moe_shard_placement(param):
    """FSDP shard placement for grouped-expert params.

    Shard on dim=1 for the (>=2D) expert weights since there may be more shards than
    experts (dim=0). A 1D param (e.g. the per-expert bias of the experts="te"
    GroupedLinear path, shape [out_features]) has no dim 1, so shard it on dim 0
    instead. FSDP all-gathers before use, so the shard dim is a storage detail and does
    not change compute.
    """
    return Shard(0) if param.ndim < 2 else Shard(1)


def _is_selective_ac(activation_checkpointing: object) -> bool:
    """Return True when the AC mode requests selective checkpointing.

    Kept inline (rather than imported from the dense FSDP2 parallelizer) so that
    threading the mode does not pull the heavy ``distributed.parallelizer`` module
    into the lightweight call path.
    """
    return (
        isinstance(activation_checkpointing, str) and activation_checkpointing.lower().replace("-", "_") == "selective"
    )


def _is_deepseek_v4_model(model: torch.nn.Module) -> bool:
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "deepseek_v4":
        return True

    inner_model = getattr(model, "model", None)
    inner_config = getattr(inner_model, "config", None)
    return getattr(inner_config, "model_type", None) == "deepseek_v4"


def _get_cp_stream() -> torch.cuda.Stream:
    global _CP_STREAM
    if _CP_STREAM is None:
        _CP_STREAM = torch.cuda.Stream()
    return _CP_STREAM


def _iter_transformer_and_mtp_blocks(model: nn.Module):
    inner = model.model if hasattr(model, "model") and model.model is not None else model
    text_model = get_text_module(inner)

    layers = getattr(text_model, "layers", None)
    if layers is not None:
        for layer_id, block in layers.named_children():
            yield layers, layer_id, block

    mtp = getattr(model, "mtp", None)
    mtp_layers = getattr(mtp, "layers", None)
    if mtp_layers is not None:
        for layer_id, block in mtp_layers.named_children():
            yield mtp_layers, layer_id, block


def _get_moe_module(block: nn.Module) -> MoE | None:
    for name in ("moe", "mlp"):
        module = getattr(block, name, None)
        if isinstance(module, MoE):
            return module


def _get_model_moe_config(model: nn.Module):
    """Return the model-level MoE config exposed by custom MoE architectures."""
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
        text_model = get_text_module(inner)
        if text_model is not inner:
            candidates.append(text_model)
    candidates.append(model)

    for candidate in candidates:
        moe_config = getattr(candidate, "moe_config", None)
        if moe_config is not None:
            return moe_config

    raise AttributeError("MoE models must expose moe_config on the inner, text, or top-level model.")


def _module_weights_are_tied(left: nn.Module | None, right: nn.Module | None) -> bool:
    """Return True when two modules expose the same ``weight`` parameter object."""
    if left is None or right is None:
        return False
    left_weight = getattr(left, "weight", None)
    right_weight = getattr(right, "weight", None)
    return left_weight is not None and left_weight is right_weight


class ExpertParallel(ParallelStyle):
    """
    ExpertParallel class is used to shard the MoE parameters on the EP mesh.
    Dim `0` of each parameter is sharded since that is the expert dimension.
    """

    def _partition_fn(self, name, module, device_mesh):
        # shard on the expert dimension
        assert device_mesh.ndim == 1

        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            dist_param.requires_grad = param.requires_grad
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            module.init_token_dispatcher(ep_mesh=device_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


def _iter_moe_blocks(model_wrapper: nn.Module, backbone: nn.Module):
    """Yield decoder blocks that may contain MoE sublayers.

    Covers the main backbone (``backbone.layers``) plus an optional MTP
    auxiliary head (``model_wrapper.mtp.layers``) when present. MTP sublayers
    are not registered under ``backbone.layers`` but carry the same MoE
    structure and must receive the same EP / FSDP treatment so their
    state-dict round-trips cleanly.

    Args:
        model_wrapper: Outer model (e.g. ``NemotronHForCausalLM``) — the
            attribute that may carry the MTP head.
        backbone: Inner backbone (``model_wrapper.model``, possibly text-only
            after VLM unwrapping) whose ``.layers`` holds the main decoder
            stack.
    """
    yield from backbone.layers.children()
    mtp_module = getattr(model_wrapper, "mtp", None)
    if mtp_module is not None and hasattr(mtp_module, "layers"):
        yield from mtp_module.layers.children()


def apply_ep(model: nn.Module, ep_mesh: DeviceMesh, moe_mesh: DeviceMesh | None = None):
    """Applies EP to MoE module."""
    assert ep_mesh.size() > 1

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present
    _model = get_text_module(_model)

    for block in _iter_moe_blocks(model, _model):
        moe_module = _get_moe_module(block)
        if moe_module is None:
            continue
        # GroupedExpertsTEGroupedLinear uses TE's GroupedLinear which creates
        # local experts directly. It doesn't support DTensor wrapping, so we
        # skip distribute_module entirely and just initialize token dispatcher.
        if isinstance(moe_module.experts, GroupedExpertsTE):
            moe_module.experts.init_token_dispatcher(ep_mesh=ep_mesh, moe_mesh=moe_mesh)
        else:
            parallelize_module(
                module=moe_module.experts,
                device_mesh=ep_mesh,
                parallelize_plan=ExpertParallel(),
            )


_VISION_TOWER_ATTRS = ("visual", "vision_tower", "vision_model", "vit_model")

_VISION_CHECKPOINT_SPECS = ("submodule", "pinned_block")


def _has_trainable_vision_tower(model: nn.Module) -> bool:
    """Return whether the model (or its inner ``.model``) exposes a trainable vision tower.

    Deliberately a cheap duck-typed gate, not a second owner of the vision
    mapping: it only decides whether importing the heavy, transformers-aware
    dense parallelizer is worthwhile, while the dense parallelizer's per-model
    layer-group mapping remains the sole owner of which blocks get wrapped.
    Requiring a trainable, parameter-bearing tower (rather than mere attribute
    existence) keeps the import off text-only, frozen-tower, and duck-typed
    stub-model call paths.
    """
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        for attr in _VISION_TOWER_ATTRS:
            tower = getattr(owner, attr, None)
            if tower is None or not hasattr(tower, "parameters"):
                continue
            if any(param.requires_grad for param in tower.parameters()):
                return True
    return False


def _apply_vision_tower_ac(model: nn.Module, vision_checkpoint_spec: str = "submodule") -> None:
    """Checkpoint trainable VLM vision-tower blocks on the expert-parallel path.

    ``apply_ac`` iterates only the text/MTP decoder stack
    (``_iter_transformer_and_mtp_blocks``), and the generic FSDP2 scope
    handling does not run for expert-parallel configs, so a trainable vision
    tower would otherwise keep every activation. Reuses the per-model vision
    layer mapping from the dense parallelizer. The default ``"submodule"`` spec
    applies the same per-submodule wrapping (attention/MLP/norms) as the
    generic FSDP2/DDP path; ``"pinned_block"`` wraps whole blocks with the
    SDPA-backend-pinned spec for environments where ambient SDPA backend
    forcing does not span the checkpoint recompute. Fully frozen vision towers
    are left untouched, consistent with the generic path's frozen-tower
    behavior.
    """
    if not _has_trainable_vision_tower(model):
        return

    # Lazy imports keep the heavy, transformers-aware dense parallelizer module
    # off the text-only MoE call path.
    from nemo_automodel.components.distributed.activation_checkpointing import (
        apply_submodule_checkpointing,
        apply_vision_block_checkpointing,
    )
    from nemo_automodel.components.distributed.parallelizer import get_model_layer_groups

    vision_layers = [
        layer
        for layer in get_model_layer_groups(model).get("vision", [])
        if any(param.requires_grad for param in layer.parameters())
    ]
    if not vision_layers:
        logger.info("No trainable vision blocks found; skipping vision-tower activation checkpointing.")
        return
    if vision_checkpoint_spec == "pinned_block":
        apply_vision_block_checkpointing(model, vision_layers)
    else:
        # Vision towers have no KV cache, so KV-sharing (a text-decoder concern)
        # never applies here.
        apply_submodule_checkpointing(vision_layers, has_kv_sharing=False)


def apply_ac(
    model: nn.Module,
    ignore_router: bool = True,
    hidden_size: int | None = None,
    num_experts: int | None = None,
    selective: bool = False,
    vision_checkpoint_spec: str = "submodule",
):
    """Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to.
        ignore_router: If True (the default), saves the MoE router output so the dispatch
            is not recomputed under activation checkpointing (avoids a CheckpointError from
            non-deterministic re-routing on recompute). If False, a warning is emitted.
        hidden_size: Hidden dimension size. If None, derived from model.config.hidden_size.
        num_experts: Number of routed experts. If None, derived from moe_config.n_routed_experts
            first, then falls back to model.config attributes.
        selective: If True, applies TorchTitan-style per-op selective activation checkpointing
            (shared with the dense FSDP2 path) to each block. Takes precedence over
            ``ignore_router``; the shared policy already saves expert-parallel communication
            collectives and ``topk``, so it composes with expert parallelism.
        vision_checkpoint_spec: How trainable VLM vision-tower blocks are checkpointed.
            ``"submodule"`` (the default) wraps each block's attention/MLP/norms with plain
            ``checkpoint_wrapper``, matching the generic FSDP2/DDP path. ``"pinned_block"``
            wraps whole blocks and pins the SDPA backend set on both the checkpoint forward
            and its recompute, for environments where ambient SDPA backend forcing does not
            span the recompute.

    Trainable VLM vision-tower blocks are checkpointed as well (in both modes) per
    ``vision_checkpoint_spec``; frozen vision towers are left untouched.
    """
    if vision_checkpoint_spec not in _VISION_CHECKPOINT_SPECS:
        raise ValueError(
            f"Unknown vision_checkpoint_spec {vision_checkpoint_spec!r}; expected one of {_VISION_CHECKPOINT_SPECS}."
        )
    if not selective and not ignore_router:
        logger.warning(
            "Activation checkpointing is enabled with ignore_router_for_ac=False. The MoE "
            "router/dispatch will be recomputed in the backward pass, which can route a "
            "different number of tokens per expert than the forward pass and crash with "
            "torch.utils.checkpoint.CheckpointError ('Recomputed values ... have different "
            "metadata'). Set ignore_router_for_ac=True (the default) to save the router "
            "output and keep routing consistent across recompute."
        )

    if selective:
        # Reuse the dense FSDP2 selective policy so the save-op set (attention,
        # matmuls, comm collectives, topk, D2H copies) stays single-sourced.
        from nemo_automodel.components.distributed.activation_checkpointing import (
            SELECTIVE_AC_WRAPPER_FLAG,
            make_selective_checkpoint_context_fn,
        )

        selective_context_fn = make_selective_checkpoint_context_fn()
        for parent_layers, layer_id, block in _iter_transformer_and_mtp_blocks(model):
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=True, context_fn=selective_context_fn)
            # Tag so _apply_per_layer_compile compiles the wrapper OUTER (keeping the
            # selective policy visible to the partitioner) instead of unwrapping and
            # compiling the block inner, which would collapse selective AC into full
            # recompute. The flag is only read when per-layer torch.compile is
            # enabled, so it is a no-op for every other mode.
            setattr(block, SELECTIVE_AC_WRAPPER_FLAG, True)
            parent_layers.register_module(layer_id, block)
        _apply_vision_tower_ac(model, vision_checkpoint_spec)
        return

    # Derive hidden_size and num_experts from model.config if not provided
    if hidden_size is None:
        cfg = getattr(model, "config", None)
        # VLM models nest language model config under text_config or llm_config
        hidden_size = (
            getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(cfg, "llm_config", None), "hidden_size", None)
            or getattr(cfg, "hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError("hidden_size must be provided or model must have config.hidden_size attribute")

    if num_experts is None:
        _inner = getattr(model, "model", model)
        if hasattr(_inner, "moe_config") and hasattr(_inner.moe_config, "n_routed_experts"):
            num_experts = _inner.moe_config.n_routed_experts
        else:
            cfg = getattr(model, "config", None)
            text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or cfg
            for attr in ["num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts"]:
                if text_cfg is not None and hasattr(text_cfg, attr):
                    num_experts = getattr(text_cfg, attr)
                    break
            else:
                raise ValueError("num_experts must be provided or model must have config.num_experts attribute")

    def _is_router_projection(func, args) -> bool:
        aten = torch.ops.aten
        mm = getattr(getattr(aten, "mm", None), "default", None)
        addmm = getattr(getattr(aten, "addmm", None), "default", None)
        linear = getattr(getattr(aten, "linear", None), "default", None)
        if func == mm:
            return len(args) == 2 and args[1].shape == (hidden_size, num_experts)
        if func == addmm:
            return len(args) >= 3 and args[2].shape == (hidden_size, num_experts)
        if func == linear:
            return len(args) >= 2 and args[1].shape == (num_experts, hidden_size)
        return False

    def _custom_policy(ctx, func, *args, **kwargs):
        if _is_router_projection(func, args):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        return create_selective_checkpoint_contexts(_custom_policy)

    # Weight-tied (use_repeated_layer) MTP head blocks must NOT be activation
    # checkpointed: the single physical block is recomputed once per MTP depth in
    # backward, and FSDP2 cannot re-unshard the *shared* EP-sharded experts param
    # group on the 2nd+ recompute (the 1st recompute's post_backward reshards it, and
    # the 2nd recompute's pre_forward unshard does not re-gather it) -> the experts
    # weight is read in the resharded Shard(1) state and grouped_gemm raises
    # "Expected hidden_in == a.size(1)". The MTP head is tiny (1 physical block), so
    # skipping its recompute costs negligible activation memory. Non-tied MTP heads
    # (each physical block recomputed exactly once) are unaffected and keep AC.
    mtp_module = getattr(model, "mtp", None)
    mtp_block_ids: set[int] = set()
    mtp_repeated = False
    if mtp_module is not None and hasattr(mtp_module, "layers"):
        mtp_block_ids = {id(b) for b in mtp_module.layers.children()}
        mtp_repeated = bool(getattr(getattr(mtp_module, "mtp_config", None), "use_repeated_layer", False))
    if mtp_repeated and mtp_block_ids:
        logger.info("Skipping activation checkpointing on %d weight-tied MTP head block(s)", len(mtp_block_ids))

    for parent_layers, layer_id, block in _iter_transformer_and_mtp_blocks(model):
        if mtp_repeated and id(block) in mtp_block_ids:
            continue
        if ignore_router:
            block = ptd_checkpoint_wrapper(
                block,
                preserve_rng_state=True,
                context_fn=selective_checkpointing_context_fn,
            )
        else:
            block = ptd_checkpoint_wrapper(block, preserve_rng_state=True)

        parent_layers.register_module(layer_id, block)

    _apply_vision_tower_ac(model, vision_checkpoint_spec)


def _shard_fp32_param_holders(block, fsdp_mesh, reshard_after_forward, offload_policy):
    """Shard each ``_fp32_params`` holder in ``block`` as its own fp32 FSDP unit.

    Model implementations own the architecture-specific decision to create these
    holders (for example Qwen3.5/Qwen3-Next GatedDeltaNet ``A_log``/``dt_bias``).
    FSDP only treats the holder as a dtype-uniform fp32 unit and excludes its params
    from the block's bf16 FSDP unit.

    Returns the set of holder parameters to exclude from the block's FSDP wrap.
    Blocks that do not expose ``named_modules`` (e.g. non-``nn.Module`` test
    stubs) cannot hold fp32 holders, so an empty set is returned.
    """
    if not hasattr(block, "named_modules"):
        return set()
    fp32_mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=False,
    )
    ignored: set = set()
    for name, sub in block.named_modules():
        if not name.endswith("_fp32_params"):
            continue
        holder_params = list(sub.parameters(recurse=False))
        if not holder_params:
            continue
        fully_shard(
            sub,
            mesh=fsdp_mesh,
            reshard_after_forward=reshard_after_forward,
            mp_policy=fp32_mp_policy,
            offload_policy=offload_policy,
        )
        ignored.update(holder_params)
    return ignored


def apply_fsdp(
    model: torch.nn.Module,
    fsdp_mesh: DeviceMesh,
    ep_enabled: bool,
    ep_shard_enabled: bool,
    ep_shard_mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
):
    """Apply FSDP wrapping to MoE transformer blocks and model-level modules."""

    if isinstance(lm_head_precision, str):
        lm_head_precision = dtype_from_str(lm_head_precision, default=None)

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

    fully_shard_impl = fully_shard
    if _is_deepseek_v4_model(model):
        from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4

        fully_shard_impl = fully_shard_deepseek_v4

    fully_shard_default = functools.partial(
        fully_shard_impl,
        mesh=fsdp_mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present (VLM models)
    _model = get_text_module(_model)

    # MTP auxiliary-head blocks keep their EP-sharded experts un-resharded
    # (reshard_after_forward=False) even when the backbone reshards. The MTP head is
    # tiny (1-2 MoE sublayers) so keeping its experts gathered costs negligible resident
    # memory, while it removes FSDP2 unshard fragility for the weight-tied head whose
    # single physical experts param group is used multiple times per step (see apply_ac,
    # which skips AC on these blocks for the same reason). Bulk backbone experts keep the
    # configured reshard_after_forward.
    mtp_module = getattr(model, "mtp", None)
    mtp_block_ids = set()
    if mtp_module is not None and hasattr(mtp_module, "layers"):
        mtp_block_ids = {id(b) for b in mtp_module.layers.children()}

    for block in _iter_moe_blocks(model, _model):
        moe_module = _get_moe_module(block)
        experts_reshard_after_forward = False if id(block) in mtp_block_ids else reshard_after_forward
        if isinstance(moe_module, MoE) and ep_shard_enabled:
            # Apply FSDP on dim=1 for grouped experts since we may have more
            # shards than experts (dim=0).
            # Forward the same mp_policy used elsewhere so that when params are
            # kept in fp32 (e.g. for fp32 master weights under FSDP2) the
            # all-gathered expert weights are still cast to param_dtype for
            # forward compute (required by GMM / TE kernels that expect bf16).
            fully_shard(
                moe_module.experts,
                mesh=ep_shard_mesh,
                shard_placement_fn=_moe_shard_placement,
                reshard_after_forward=experts_reshard_after_forward,
                mp_policy=mp_policy,
            )
        # If FSDP is disabled for grouped experts because the parameters are already
        # fully sharded by PP and EP, then we need to explicitly remove the parameters
        # from FSDP for the transformer block.
        # If FSDP is enabled for grouped experts, the parameters are automatically
        # removed from the FSDP for the transformer block due to the rules of the
        # PyTorch FSDP implementation.
        ignored_params = None
        if isinstance(moe_module, MoE) and ep_enabled:
            ignored_params = set(moe_module.experts.parameters())

        # Shard model-owned fp32 holders on their own and exclude their params from
        # the block's FSDP unit to keep the block dtype-uniform.
        fp32_ignored = _shard_fp32_param_holders(block, fsdp_mesh, reshard_after_forward, offload_policy)
        if fp32_ignored:
            ignored_params = (ignored_params or set()) | fp32_ignored
        fully_shard_default(block, ignored_params=ignored_params)

    # Re-establish weight tying before detecting it: a device/dtype move during
    # from_pretrained (HF replaces param tensors) can silently break a tie set in
    # __init__ (lm_head.weight = embed_tokens.weight), so the identity check below
    # would miss it and shard embed and lm_head as two INDEPENDENT FSDP params.
    # For tie_word_embeddings models those two then drift apart during full
    # fine-tuning, diverging from the tied architecture and breaking checkpoint
    # resume (the checkpoint drops lm_head and reconstructs it from embed on load,
    # silently discarding the drifted lm_head). Re-tying here keeps them in one
    # FSDP root so they remain a single shared parameter.
    # NB: re-point output->input embedding directly (model.__init__'s own tie);
    # HF's tie_weights() is incompatible with this model's _tied_weights_keys
    # format ('list' object has no attribute 'keys').
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        _out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        _inp = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if _out is not None and _inp is not None and hasattr(_out, "weight") and hasattr(_inp, "weight"):
            _out.weight = _inp.weight

    embed_tokens = getattr(_model, "embed_tokens", None)
    inner_lm_head = getattr(_model, "lm_head", None)
    outer_lm_head = getattr(model, "lm_head", None) if model is not _model else None
    lm_head = inner_lm_head or outer_lm_head
    tied_input_output_embeddings = _module_weights_are_tied(embed_tokens, lm_head)
    tied_embeddings_cross_fsdp_roots = tied_input_output_embeddings and lm_head is outer_lm_head

    if embed_tokens is not None and not tied_input_output_embeddings:
        fully_shard_default(embed_tokens)

    if lm_head is not None and not tied_input_output_embeddings:
        # Use custom mixed precision policy for lm_head if lm_head_precision is specified
        if lm_head_precision == torch.float32:
            lm_head_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )
            fully_shard(
                lm_head,
                mesh=fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=lm_head_mp_policy,
                offload_policy=offload_policy,
            )
        else:
            fully_shard_default(lm_head)
    elif tied_input_output_embeddings and lm_head_precision is not None:
        logger.warning(
            "Skipping separate lm_head FSDP wrapping because lm_head.weight is tied to embed_tokens.weight; "
            "lm_head_precision=%s will not be applied independently.",
            lm_head_precision,
        )

    # TODO: properly handle all possible multimodal component names
    if hasattr(model, "audio_tower") and model.audio_tower is not None:
        if any(param.requires_grad for param in model.audio_tower.parameters()):
            fully_shard_default(model.audio_tower)
        else:
            logging.info("Skipping FSDP wrap for frozen audio tower")

    if hasattr(model, "visual") and model.visual is not None:
        if any(param.requires_grad for param in model.visual.parameters()):
            fully_shard_default(model.visual)
        else:
            logging.info("Skipping FSDP wrap for frozen visual tower")

    if tied_embeddings_cross_fsdp_roots and wrap_outer_model and model is not _model:
        logger.info(
            "Skipping separate inner-model FSDP root because lm_head.weight is tied to embed_tokens.weight "
            "across the outer model boundary; the outer FSDP root will own the tied parameter."
        )
    else:
        if tied_embeddings_cross_fsdp_roots and not wrap_outer_model:
            logger.warning(
                "lm_head.weight is tied to embed_tokens.weight across the outer model boundary, but "
                "wrap_outer_model=False prevents preserving the tie in one FSDP root."
            )
        fully_shard_default(_model)

    # If model has a nested structure (outer model wrapping inner _model), wrap the outer model if requested
    if wrap_outer_model and model is not _model:
        fully_shard_default(model)


def apply_cp(model: torch.nn.Module, cp_mesh: DeviceMesh, cp_comm_type: str = "p2p"):
    """Configure context parallelism for attention and MoE layers."""

    from transformer_engine.pytorch.attention import DotProductAttention

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present (VLM models)
    _model = get_text_module(_model)

    # Set CP flags so wrapper and text-module forward paths can prepare
    # full-sequence embeddings and null out local attention masks.
    # With CP the mask is not sharded along the sequence dim and TE asserts
    # "Padding mask not supported with context parallelism!".
    model._cp_enabled = True
    _model._cp_enabled = True

    # Route each attention block's CP setup by capability:
    #   * TE DotProductAttention -> TE's own context-parallel group;
    #   * a module exposing setup_cp_attention (e.g. Gemma4's p2p ring or MiniMax
    #     M3's block-sparse DSA) -> installs its own CP attention + mask handling
    #     (model-owned, like TE/DSV4).
    # Any other (non-TE, non-model-owned) attention is not supported under CP here.
    for _parent, _layer_id, block in _iter_transformer_and_mtp_blocks(model):
        layer_type = getattr(block, "layer_type", getattr(block, "attention_type", "full_attention"))

        if layer_type in ("full_attention", "sliding_attention"):
            self_attn = block.self_attn
            attn_module = getattr(self_attn, "attn_module", None)
            if isinstance(attn_module, DotProductAttention):
                attn_cp_comm_type = "all_gather" if layer_type == "sliding_attention" else cp_comm_type
                attn_module.set_context_parallel_group(
                    cp_mesh.get_group(),
                    torch.distributed.get_process_group_ranks(cp_mesh.get_group()),
                    _get_cp_stream(),
                    cp_comm_type=attn_cp_comm_type,
                )
            elif hasattr(self_attn, "setup_cp_attention"):
                # Model-owned CP attention (e.g. Gemma4's p2p ring): the model
                # installs its own SDPA hook + mask handling.
                self_attn.setup_cp_attention(cp_mesh)
            else:
                logger.warning(
                    "Skipping CP setup for block with unsupported attention module "
                    "(neither TE DotProductAttention nor model-owned setup_cp_attention): %s",
                    type(attn_module).__name__ if attn_module is not None else type(self_attn).__name__,
                )
        elif layer_type == "mamba":
            from nemo_automodel.components.distributed.mamba_cp import MambaContextParallel

            mixer = block.self_attn  # NemotronV3Block.self_attn aliases mixer
            mixer.cp = MambaContextParallel(
                cp_group=cp_mesh.get_group(),
                num_heads=mixer.num_heads,
                head_dim=mixer.head_dim,
                n_groups=mixer.n_groups,
                d_state=mixer.ssm_state_size,
                mixer=mixer,
            )
        elif layer_type == "linear_attention":
            # FLA-based CP: store the CP mesh on the linear attention module so it
            # can recover dense token order and build its CP context during forward.
            if hasattr(block, "linear_attn") and hasattr(block.linear_attn, "_cp_mesh"):
                block.linear_attn._cp_mesh = cp_mesh
            else:
                logger.warning(
                    "Block %s has linear_attention but no CP-aware linear_attn module; "
                    "skipping CP setup for this layer.",
                    getattr(block, "layer_idx", "?"),
                )

        moe_module = block.moe if hasattr(block, "moe") else block.mlp
        if isinstance(moe_module, MoE):
            moe_module.cp_mesh = cp_mesh


def parallelize_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    activation_checkpointing: bool | str = False,
    ignore_router_for_ac: bool = True,
    vision_checkpoint_spec: str = "submodule",
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
):
    """Apply context, expert, activation-checkpointing, and FSDP parallelism."""

    assert tp_axis_name is None or world_mesh[tp_axis_name].size() == 1, (
        "Tensor parallelism not supported for custom MoE models"
    )

    cp_enabled = cp_axis_name is not None and world_mesh[cp_axis_name].size() > 1
    if cp_enabled:
        apply_cp(model, world_mesh[cp_axis_name])

    ep_enabled = ep_axis_name is not None and moe_mesh is not None and moe_mesh[ep_axis_name].size() > 1
    if ep_enabled:
        moe_config = _get_model_moe_config(model)
        assert moe_config.n_routed_experts % moe_mesh[ep_axis_name].size() == 0, (
            f"n_routed_experts {moe_config.n_routed_experts} must be divisible by "
            f"expert_parallel_degree {moe_mesh[ep_axis_name].size()}"
        )

        apply_ep(model, moe_mesh[ep_axis_name], moe_mesh=moe_mesh)

    if activation_checkpointing:
        apply_ac(
            model,
            ignore_router=ignore_router_for_ac,
            selective=_is_selective_ac(activation_checkpointing),
            vision_checkpoint_spec=vision_checkpoint_spec,
        )

    if ep_shard_axis_names is not None:
        ep_shard_mesh = moe_mesh[ep_shard_axis_names]
    else:
        ep_shard_mesh = None

    from nemo_automodel.components.distributed.mesh_utils import get_submesh as _get_submesh

    fsdp_enabled = dp_axis_names is not None and _get_submesh(world_mesh, tuple(dp_axis_names)).size() > 1
    fsdp_mesh = _get_submesh(world_mesh, tuple(dp_axis_names)) if fsdp_enabled else None
    if fsdp_enabled:
        apply_fsdp(
            model,
            fsdp_mesh,
            ep_enabled=ep_enabled,
            ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
            ep_shard_mesh=ep_shard_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=reshard_after_forward,
            lm_head_precision=lm_head_precision,
            wrap_outer_model=wrap_outer_model,
        )

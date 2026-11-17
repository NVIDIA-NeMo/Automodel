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

import functools
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from nemo_automodel.shared.import_utils import safe_import_from, safe_import_te

HAS_TE, transformer_engine = safe_import_te()
HAS_FUSED_ADAM, FusedAdam = (
    safe_import_from("transformer_engine.pytorch.optimizers", "FusedAdam") if HAS_TE else (False, None)
)

# The Conflict:
# PyTorch DCP passes an _EXTRA_STATE sentinel for missing keys, but Transformer Engine (TE)
# throws a RuntimeError if it receives anything other than None or a Tensor.
#
# The Fix (Monkeypatch):
# Intercept set_extra_state calls. If the input is the _EXTRA_STATE sentinel, return early
# (doing nothing) to safely ignore the missing state without crashing TE.
if HAS_TE:
    import transformer_engine.pytorch.module.base as te_base
    import transformer_engine.pytorch.ops.op as te_ops

    _original_set_extra_state = te_base.TransformerEngineBaseModule.set_extra_state
    _original_op_set_extra_state = te_ops.BasicOperation.set_extra_state

    def _safe_set_extra_state(self, state):
        if state is not None and "EXTRA_STATE" in str(type(state)):
            return
        return _original_set_extra_state(self, state)

    def _safe_op_set_extra_state(self, state):
        if state is not None and "EXTRA_STATE" in str(type(state)):
            return
        return _original_op_set_extra_state(self, state)

    te_base.TransformerEngineBaseModule.set_extra_state = _safe_set_extra_state
    te_ops.BasicOperation.set_extra_state = _safe_op_set_extra_state

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.checkpoint.utils import (
    ensure_tied_lm_head,
    get_lm_head_weight_and_name,
    has_local_tied_lm_head,
    is_tied_word_embeddings,
    materialize_missing_tied_lm_head,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

_PREFIX = "model."
_OPTIMIZER_PARTS_KEY = "optimizer_parts"
_OPTIMIZER_PART_KEY_PREFIX = "stage_"


def _unwrap_ddp_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the module that owns model metadata and parameter names hidden by DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _dcp_parameter_name(name: str) -> str:
    """Normalize a ``named_parameters()`` name to the key DCP's state dict uses for it.

    DCP drops activation-checkpoint and ``torch.compile`` wrapper components; DDP's
    ``module.`` prefix is handled by unwrapping the model before naming parameters.
    """
    return canonical_parameter_fqn(name).replace("_orig_mod.", "")


def _trainable_non_lora_parameter_names(model_parts: list[torch.nn.Module]) -> set[str]:
    """Return the DCP state-dict names of trainable parameters that are not LoRA weights.

    Under PEFT these are parameters ``freeze_config`` made trainable next to the adapters
    (``unfreeze_modules``/``unfreeze_parameters``), which train in full and must be saved
    with the adapter.

    Args:
        model_parts: Model or pipeline-stage parts, optionally wrapped in DDP.

    Returns:
        Normalized parameter names, matching the keys of ``get_model_state_dict``. A shared
        parameter appears once, under its first registered name.
    """
    return {
        _dcp_parameter_name(name)
        for part in model_parts
        for name, param in _unwrap_ddp_model(part).named_parameters()
        if param.requires_grad and "lora_" not in name
    }


def _select_peft_tensors(
    model_state_dict: dict[str, Any], model_parts: list[torch.nn.Module], full_modules: Collection[str]
) -> dict[str, Any]:
    """Keep the entries of a PEFT model's DCP state dict that its checkpoint holds.

    ``get_model_state_dict(..., ignore_frozen_params=True)`` returns the trainable parameters
    together with every persistent buffer. A PEFT checkpoint holds the LoRA weights, every other
    trainable parameter, and the persistent buffers of the modules it stores whole, because HF
    PEFT restores a ``modules_to_save`` module from all of its state-dict entries. Other buffers
    are dropped.

    Args:
        model_state_dict: DCP model state dict mapping names to tensors of arbitrary shape. DCP
            materializes it on rank 0 only, so other ranks pass an empty dict.
        model_parts: The model parts ``model_state_dict`` was collected from.
        full_modules: Modules the checkpoint stores whole, from ``_fully_trained_modules``.

    Returns:
        A new dict holding the same tensor objects (not copies) for the selected entries.

    Raises:
        RuntimeError: If a trainable non-LoRA parameter is missing from a non-empty
            ``model_state_dict``. Saving would otherwise drop trained weights silently.
    """
    trainable = _trainable_non_lora_parameter_names(model_parts)
    full_module_prefixes = tuple(f"{module}." for module in full_modules)
    selected = {
        k: v
        for k, v in model_state_dict.items()
        if "lora_" in k or k in trainable or k.startswith(full_module_prefixes)
    }
    missing = trainable - selected.keys()
    if model_state_dict and missing:
        raise RuntimeError(
            "The PEFT checkpoint would drop trainable parameters that are missing from the model state dict: "
            f"{sorted(missing)[:10]}. Their names do not match the state-dict keys "
            f"(examples: {sorted(model_state_dict)[:5]})."
        )
    return selected


def _get_peft_full_module_buffers(model: torch.nn.Module, full_modules: Collection[str]) -> dict[str, Any]:
    """Collect the persistent buffers of the modules a PEFT checkpoint stores whole, bypassing DCP.

    The local counterpart of the buffer selection in ``_select_peft_tensors``. Names follow
    ``_get_peft_state_dict``, which collects the parameters on the same path.

    Args:
        model: One local model part.
        full_modules: Modules the checkpoint stores whole, from ``_fully_trained_modules``.

    Returns:
        CPU copies of the persistent buffers under ``full_modules``, keyed by name. Each buffer keeps
        its shape and dtype; a DTensor buffer is gathered to its full tensor first.
    """
    full_module_prefixes = tuple(f"{module}." for module in full_modules)
    buffers = {}
    for module_name, module in model.named_modules():
        for buffer_name, buffer in module.named_buffers(recurse=False):
            # nn.Module exposes buffer persistence only through state_dict(), which modules such as
            # grouped experts override with renamed keys, so read the flag state_dict() itself uses.
            if buffer_name in module._non_persistent_buffers_set:
                continue
            name = canonical_parameter_fqn(f"{module_name}.{buffer_name}" if module_name else buffer_name)
            if not _dcp_parameter_name(name).startswith(full_module_prefixes):
                continue
            buffer = buffer.full_tensor() if hasattr(buffer, "full_tensor") else buffer
            buffers[name] = buffer.detach().cpu()
    return buffers


@dataclass(frozen=True)
class _TrainableModuleLayout:
    """Where a PEFT model's trainable non-LoRA parameters sit in its module tree.

    ``freeze_config`` can make parameters trainable next to the LoRA adapters (for
    example ``unfreeze_modules``). Those train in full, so the adapter checkpoint
    stores them whole and names their modules in ``modules_to_save``, which tells
    Hugging Face PEFT to restore them. All names use the DCP state-dict normalization.

    Attributes:
        trainable: Names of trainable parameters that are not LoRA weights, one name
            per parameter.
        aliased: Name groups of trainable non-LoRA parameters that are registered
            under several names, such as tied input and output embeddings.
        adapted: Paths of the modules that own LoRA parameters.
        fully_trainable: For every module path that holds parameters, whether all of
            them are trainable non-LoRA parameters.
    """

    trainable: frozenset[str]
    aliased: tuple[tuple[str, ...], ...]
    adapted: frozenset[str]
    fully_trainable: dict[str, bool]

    def merge(self, other: "_TrainableModuleLayout") -> "_TrainableModuleLayout":
        """Combine the layouts of two pipeline stages that hold different parameters.

        A module path present on both stages is fully trainable only if it is on each.
        """
        fully_trainable = dict(self.fully_trainable)
        for path, value in other.fully_trainable.items():
            fully_trainable[path] = fully_trainable.get(path, True) and value
        return _TrainableModuleLayout(
            trainable=self.trainable | other.trainable,
            aliased=self.aliased + other.aliased,
            adapted=self.adapted | other.adapted,
            fully_trainable=fully_trainable,
        )


def _lora_adapted_module(name: str) -> str:
    """Return the path of the module that owns the LoRA parameter ``name``."""
    if ".lora_" in name:
        return name.split(".lora_", 1)[0]
    return name.rpartition(".")[0]


def _trainable_module_layout(model_parts: Sequence[torch.nn.Module]) -> _TrainableModuleLayout:
    """Describe the trainable non-LoRA parameters of local model parts.

    Args:
        model_parts: This rank's model or pipeline-stage parts, optionally wrapped in DDP.

    Returns:
        The layout of the parameters these parts hold.
    """
    trainable: set[str] = set()
    aliased: list[tuple[str, ...]] = []
    adapted: set[str] = set()
    fully_trainable: dict[str, bool] = {}
    for part in model_parts:
        names_by_parameter: dict[int, list[str]] = {}
        for raw_name, parameter in _unwrap_ddp_model(part).named_parameters(remove_duplicate=False):
            name = _dcp_parameter_name(raw_name)
            is_lora = "lora_" in name
            is_full_weight = parameter.requires_grad and not is_lora
            if is_lora:
                adapted.add(_lora_adapted_module(name))
            if is_full_weight:
                names_by_parameter.setdefault(id(parameter), []).append(name)
            components = name.split(".")
            for depth in range(1, len(components)):
                path = ".".join(components[:depth])
                fully_trainable[path] = fully_trainable.get(path, True) and is_full_weight
        for names in names_by_parameter.values():
            trainable.add(names[0])
            if len(names) > 1:
                aliased.append(tuple(names))
    return _TrainableModuleLayout(
        trainable=frozenset(trainable),
        aliased=tuple(aliased),
        adapted=frozenset(adapted),
        fully_trainable=fully_trainable,
    )


def _validate_trainable_module_layout(layout: _TrainableModuleLayout) -> None:
    """Reject trainable non-LoRA parameters that an HF PEFT adapter cannot restore.

    Args:
        layout: Layout of the whole model, or of the pipeline stages this rank holds.

    Raises:
        ValueError: If a trainable non-LoRA parameter has several names (``modules_to_save``
            would restore each alias as a separate copy, untying them), belongs to the root
            module (``modules_to_save`` names submodules), or shares a module path with a
            LoRA adapter (the adapter and a full copy of the same module conflict on load).
    """
    if layout.aliased:
        raise ValueError(
            "PEFT checkpoints cannot store a trainable parameter that is registered under several names, "
            f"such as tied input and output embeddings: {sorted(layout.aliased)[:5]}. Hugging Face PEFT "
            "would restore each name as a separate copy. Keep these parameters frozen, or untie them."
        )
    root_owned = sorted(name for name in layout.trainable if "." not in name)
    if root_owned:
        raise ValueError(
            f"PEFT checkpoints cannot store the trainable parameters {root_owned[:10]}: they belong to the "
            "root module, and modules_to_save can only restore submodules. Keep them frozen."
        )
    adapted_and_ancestors = set()
    for module in layout.adapted:
        components = module.split(".")
        adapted_and_ancestors.update(".".join(components[:depth]) for depth in range(1, len(components) + 1))
    overlapping = []
    for name in sorted(layout.trainable):
        components = name.split(".")[:-1]
        owner = ".".join(components)
        owner_and_ancestors = {".".join(components[:depth]) for depth in range(1, len(components) + 1)}
        if owner in adapted_and_ancestors or owner_and_ancestors & layout.adapted:
            overlapping.append(name)
    if overlapping:
        raise ValueError(
            "PEFT checkpoints cannot store trainable parameters whose module also holds LoRA adapters or "
            f"lies inside or around one: {overlapping[:10]}. Hugging Face PEFT would restore the adapter and "
            "a full copy of the same module, which conflict. Keep these modules out of the PEFT targets, or "
            "narrow the selection that trains them in full: freeze_config.unfreeze_modules or "
            "unfreeze_parameters, or the recipe's default when freeze_config is not set."
        )


def validate_peft_trainable_modules(model: torch.nn.Module) -> None:
    """Check that a PEFT model's trainable parameters fit an HF PEFT adapter checkpoint.

    Run this after the trainability policy is final and before training starts, so an
    unsupported ``freeze_config`` fails at startup instead of at the first checkpoint save.

    Args:
        model: A PEFT model or pipeline stage, optionally wrapped in DDP.

    Raises:
        ValueError: If a trainable non-LoRA parameter cannot be restored by Hugging Face
            PEFT; see ``_validate_trainable_module_layout``.
    """
    _validate_trainable_module_layout(_trainable_module_layout([model]))


def _combined_trainable_module_layout(
    model_parts: Sequence[torch.nn.Module], *, pp_group: "torch.distributed.ProcessGroup | None" = None
) -> _TrainableModuleLayout:
    """Return the trainable-module layout of the whole model, combining pipeline stages.

    Under pipeline parallelism each rank holds some stages, so the layouts of all stages are
    gathered and merged, and every rank returns the same layout. Every rank of ``pp_group``
    must call this together.

    Args:
        model_parts: This rank's model or pipeline-stage parts, optionally wrapped in DDP.
        pp_group: Pipeline-parallel process group, or None without pipeline parallelism.

    Returns:
        The layout of every parameter of the model.
    """
    layout = _trainable_module_layout(model_parts)
    if pp_group is not None and torch.distributed.get_world_size(group=pp_group) > 1:
        gathered: list[_TrainableModuleLayout] = [layout] * torch.distributed.get_world_size(group=pp_group)
        torch.distributed.all_gather_object(gathered, layout, group=pp_group)
        # The gathered list includes this rank's own layout, so merge it from scratch.
        layout = functools.reduce(_TrainableModuleLayout.merge, gathered)
    return layout


def _fully_trained_modules(layout: _TrainableModuleLayout) -> list[str]:
    """Return the modules a PEFT checkpoint stores whole, under their native names.

    Each entry is the largest module whose parameters are all trainable and none of them LoRA
    weights. HF PEFT wraps every module listed in ``modules_to_save`` and loads all of its
    state-dict entries from the adapter file. Naming a submodule of a fully trained module
    breaks loading when PEFT also wraps the parent; for sequence classification it always
    wraps ``classifier``/``score``.

    This only names the modules. Writers of ``adapter_config.json`` reject layouts HF PEFT
    cannot restore first, with ``_validate_trainable_module_layout``.

    Args:
        layout: Layout of the whole model, from ``_combined_trainable_module_layout``.

    Returns:
        Sorted module names. Empty when only LoRA weights train.
    """
    modules = set()
    for name in layout.trainable:
        module = name.rpartition(".")[0]
        while "." in module and layout.fully_trainable.get(module.rpartition(".")[0], False):
            module = module.rpartition(".")[0]
        if module:
            modules.add(module)
    return sorted(modules)


def _is_quantized_module(module: torch.nn.Module) -> bool:
    """Check if a module is a BitsAndBytes quantized type.

    Detects quantization by checking for `quant_state` attribute which is
    common across BitsAndBytes quantized module types (Params4bit, Int8Params, etc.).
    """
    return getattr(module, "quant_state", None) is not None


def _has_quantized_params(model: torch.nn.Module) -> bool:
    """Check if model has any BitsAndBytes quantized modules."""
    return any(map(_is_quantized_module, model.modules()))


def _has_expert_parallelism(model: torch.nn.Module) -> bool:
    """Check if any MoE expert module in the model has expert parallelism enabled.

    After EP initialization, expert modules (GroupedExpertsDeepEP, GroupedExpertsTE)
    store ``ep_size`` on themselves. A value > 1 signals that expert weights are
    sharded across EP ranks and DCP's state_dict APIs cannot handle them.
    """
    return any(getattr(m, "ep_size", 1) > 1 for m in model.modules())


def _zeros_like_optimizer_param(param: torch.Tensor) -> torch.Tensor:
    """Allocate zero optimizer state matching a parameter.

    Args:
        param: Tensor of arbitrary shape representing one optimizer parameter.

    Returns:
        Zero tensor with the same shape, dtype, device, and layout as ``param``.
    """
    try:
        return torch.zeros_like(param, memory_format=torch.preserve_format)
    except TypeError:
        return torch.zeros_like(param)


def _materialize_missing_adam_state(optimizer: torch.optim.Optimizer) -> None:
    """Create zero-valued Adam state for parameters that do not have state yet."""
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return

    for group in optimizer.param_groups:
        step_dtype = (
            torch.float32
            if group.get("fused", False)
            else torch.float64
            if torch.get_default_dtype() == torch.float64
            else torch.float32
        )
        for param in group["params"]:
            state = optimizer.state[param]
            if "step" not in state:
                if group.get("capturable", False) or group.get("fused", False):
                    state["step"] = torch.zeros((), dtype=step_dtype, device=param.device)
                else:
                    state["step"] = torch.tensor(0.0, dtype=step_dtype)
            if "exp_avg" not in state:
                state["exp_avg"] = _zeros_like_optimizer_param(param)
            if "exp_avg_sq" not in state:
                state["exp_avg_sq"] = _zeros_like_optimizer_param(param)
            if group.get("amsgrad", False) and "max_exp_avg_sq" not in state:
                state["max_exp_avg_sq"] = _zeros_like_optimizer_param(param)


def _materialize_native_fused_adam_state(optimizer: torch.optim.Optimizer) -> None:
    """Create the state a never-stepped TE FusedAdam would otherwise allocate in its first ``step()``.

    TE FusedAdam keeps the step counter in each parameter group and allocates the moments and master weights
    lazily inside ``step()``. A fresh optimizer therefore returns an empty ``state_dict()``, and DCP only loads
    keys present in that skeleton, so a resume on the native optimizer checkpoint path would restore nothing.
    This mirrors the initialization in ``FusedAdam.step``: a zero step counter for every group (an int32 tensor
    of shape [1] when capturable) and TE's own ``initialize_state`` for every parameter without state, which
    allocates zero moments plus the fp32 master copy, or the zero int16 remainder of a bf16 parameter. The first
    real update after materialization is unchanged.

    Args:
        optimizer: Optimizer to populate; every other optimizer type is left untouched. Each parameter keeps its
            shape, dtype, device and DTensor placements, and TE allocates matching per-parameter buffers.
    """
    if not HAS_FUSED_ADAM or not isinstance(optimizer, FusedAdam):
        return
    for group in optimizer.param_groups:
        if "step" not in group:
            if optimizer.capturable:
                # FusedAdam.step() keeps the counter on the group's first parameter and, for a group without
                # parameters, on the optimizer's own overflow buffer.
                step_device = group["params"][0].device if group["params"] else optimizer._dummy_overflow_buf.device
                group["step"] = torch.zeros(1, dtype=torch.int32, device=step_device)
            else:
                group["step"] = 0
        for param in group["params"]:
            if not optimizer.state[param]:
                optimizer.initialize_state(param, optimizer.store_param_remainders and param.dtype == torch.bfloat16)


def _get_peft_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """Extract only trainable PEFT adapter weights, bypassing DCP.

    This function directly iterates over model parameters to collect trainable weights,
    avoiding PyTorch DCP's state_dict traversal which fails on (1) BitsAndBytes quantized
    modules (Params4bit, Int8Params, etc.) and (2) MoE models with expert parallelism
    where expert weights are sharded across EP ranks.
    """
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Strip _checkpoint_wrapped_module. from FQNs to match DCP's normalization.
            # Without this, activation checkpointing causes key mismatches on reload.
            name = canonical_parameter_fqn(name)
            param = param.full_tensor() if hasattr(param, "full_tensor") else param
            state_dict[name] = param.detach().cpu()
    return state_dict


def _gather_peft_state_dict_across_pp(
    local_state_dict: dict[str, Any],
    pp_group: "torch.distributed.ProcessGroup",
) -> dict[str, Any]:
    """All-gather PEFT adapter tensors across a pipeline-parallel group.

    Pipeline parallelism partitions the model's layers across PP ranks: each
    rank's local module only contains its own stage's layers. The local
    collection in :meth:`ModelState.state_dict` gathers PEFT tensors solely from
    the local model parts, so under ``pp_size > 1`` the per-rank state dict is
    missing every layer owned by another stage. Saving that directly yields a
    truncated adapter (only ~1/pp of the layers), which silently degrades a
    merged model.

    This gathers the per-rank PEFT dicts over ``pp_group`` and merges them by FQN
    so every rank returns the complete adapter. Keys are globally unique across PP
    stages (layer indices never overlap between stages), so the union is exact and
    order-independent; on the rare chance the same key appears on two ranks (e.g.
    a replicated tied parameter) the lowest-rank value wins deterministically.

    Args:
        local_state_dict: This rank's PEFT tensors (already CPU, bf16/fp32).
        pp_group: The pipeline-parallel process group to gather over.

    Returns:
        The merged PEFT state dict containing every PP stage's adapter tensors.
    """
    world = torch.distributed.get_world_size(group=pp_group)
    if world == 1:
        return local_state_dict

    gathered: list[dict[str, Any]] = [None] * world
    torch.distributed.all_gather_object(gathered, local_state_dict, group=pp_group)

    merged: dict[str, Any] = {}
    # Iterate in rank order so a duplicate key resolves to the lowest rank.
    for rank_sd in gathered:
        if not rank_sd:
            continue
        for k, v in rank_sd.items():
            if k not in merged:
                merged[k] = v

    # Sanity check: when two or more ranks contribute adapter tensors, the merge
    # must add keys beyond any single rank's set; otherwise the gather silently
    # collapsed (e.g. a wrong/global group was passed in) and we would write a
    # truncated adapter -- the exact failure this fix exists to prevent. Skip the
    # check when only one rank is non-empty: a valid PP layout can place all
    # trainable adapters on a single stage (per_rank=[N, 0, ...]), where
    # merged_n == max(per_rank) is correct, not a collapse.
    local_n = len(local_state_dict)
    merged_n = len(merged)
    per_rank = [len(sd) if sd else 0 for sd in gathered]
    non_empty_ranks = sum(1 for n in per_rank if n > 0)
    logging.getLogger(__name__).info(
        "PEFT PP gather: pp_world=%d per_rank_tensors=%s merged_tensors=%d (local=%d)",
        world,
        per_rank,
        merged_n,
        local_n,
    )
    if non_empty_ranks > 1 and merged_n <= max(per_rank):
        logging.getLogger(__name__).warning(
            "PEFT PP gather produced no more tensors (%d) than the largest single "
            "rank (%d) despite %d non-empty ranks (pp_world=%d). The saved adapter "
            "may be INCOMPLETE -- verify the pipeline-parallel process group is correct.",
            merged_n,
            max(per_rank),
            non_empty_ranks,
            world,
        )
    return merged


def _set_peft_state_dict(model: torch.nn.Module, state_dict: dict[str, Any]) -> None:
    """Load trainable PEFT adapter weights into the model, bypassing DCP.

    Mirrors _get_peft_state_dict: directly assigns saved tensors to model parameters
    by name, handling DTensor re-sharding for EP-parallel weights. This avoids
    DCP's set_model_state_dict() which raises KeyError on expert-parallel FQNs.
    """
    from torch.distributed.tensor import DTensor, Replicate

    # Strip _checkpoint_wrapped_module. from FQNs to match DCP's normalization.
    # Without this, activation checkpointing causes key mismatches on reload.
    param_dict = {canonical_parameter_fqn(name): param for name, param in model.named_parameters()}
    loaded, skipped = 0, 0

    for name, saved_tensor in state_dict.items():
        if name not in param_dict:
            skipped += 1
            continue

        param = param_dict[name]
        if not param.requires_grad:
            skipped += 1
            continue

        if isinstance(param.data, DTensor):
            full_t = saved_tensor.to(param.data.to_local().device)
            full_dt = DTensor.from_local(
                full_t, device_mesh=param.data.device_mesh, placements=[Replicate()] * param.data.device_mesh.ndim
            )
            local_shard = full_dt.redistribute(placements=param.data.placements).to_local()
            param.data.to_local().copy_(local_shard)
        else:
            param.data.copy_(saved_tensor.to(param.data.device))
        loaded += 1

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        import logging

        logging.getLogger(__name__).info(f"_set_peft_state_dict: loaded {loaded} params, skipped {skipped} keys")


def _drop_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX) -> None:
    """
    Remove the *first* occurrence of `prefix` on every key in-place.
    """
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix) :]] = sd.pop(k)


def _add_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX, skip_keys: list[str] | None = None) -> None:
    """
    Prepend `prefix` once to every key in-place (inverse of `_drop_outer_prefix`).
    """
    skip_keys = [] if skip_keys is None else skip_keys
    for k in list(sd.keys()):
        if not k.startswith(prefix) and k not in skip_keys:
            sd[prefix + k] = sd.pop(k)


def _rename_dora_keys_to_hf(sd: dict[str, Any]) -> None:
    """
    Rename DoRA magnitude keys to match HF PEFT's saved checkpoint format in-place.

    HF PEFT's ``get_peft_model_state_dict`` strips the adapter name and the
    ``.weight`` suffix from ``lora_magnitude_vector.<adapter>.<weight>`` so the
    round-trip format on disk is simply ``<module>.lora_magnitude_vector``.
    When loading, ``set_peft_model_state_dict`` re-inserts the adapter name
    and the ``.weight`` suffix automatically, so we must NOT include them here.
    """
    for k in list(sd.keys()):
        if k.endswith(".lora_magnitude"):
            sd[k[: -len(".lora_magnitude")] + ".lora_magnitude_vector"] = sd.pop(k)


def _rename_dora_keys_from_hf(sd: dict[str, Any]) -> None:
    """
    Reverse of _rename_dora_keys_to_hf: convert HF PEFT key format back to internal names.

    Handles both the current on-disk format (``<module>.lora_magnitude_vector``)
    and the legacy format that included ``.default.weight`` for robustness.
    """
    for k in list(sd.keys()):
        if k.endswith(".lora_magnitude_vector.default.weight"):
            sd[k[: -len(".lora_magnitude_vector.default.weight")] + ".lora_magnitude"] = sd.pop(k)
        elif k.endswith(".lora_magnitude_vector"):
            sd[k[: -len(".lora_magnitude_vector")] + ".lora_magnitude"] = sd.pop(k)


def _get_lm_head_weight_and_name(model: torch.nn.Module) -> tuple[torch.Tensor, str] | None:
    return get_lm_head_weight_and_name(model)


# modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState:
    """
    Helper class for tracking model state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model to track.
    """

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        is_peft: bool = False,
        is_init_step: bool = False,
        skip_task_head_prefixes: list[str] | None = None,
        cpu_offload: bool = False,
        pp_group: "torch.distributed.ProcessGroup | None" = None,
        *,
        has_expert_parallelism: bool = False,
    ):
        """
        Initialize a ModelState instance for distributed checkpointing.

        The constructor records the model reference, detects whether the model
        ties its language-model head to the input embeddings, and stores the
        desired serialization backend so that DCP can correctly save and restore
        the model's parameters and buffers.

        Args:
            model (torch.nn.Module): The PyTorch model whose state should be
                captured during checkpointing.
            is_peft (bool): Whether the model is PEFT.
            is_init_step (bool): Whether the model is being initialized.
            skip_task_head_prefixes (list[str] | None): List of parameter name prefixes to skip when loading from base model. If None or empty, loads all parameters.
                Common examples:
                - ["classifier."] for sequence/token classification
                - ["qa_outputs."] for question answering
                - ["score."] for some classification heads
            cpu_offload: Whether DCP should move sharded tensors to CPU before saving.
            pp_group (ProcessGroup | None): Pipeline-parallel process group. When
                set and ``pp_size > 1``, PEFT adapter weights are all-gathered
                across this group at save time so the on-disk adapter contains
                every PP stage's layers (not just the local stage's). Required
                for correct PEFT saves under pipeline parallelism; ignored for
                non-PEFT models and no-op when ``pp_size == 1``.
            has_expert_parallelism: Whether the distributed topology uses expert
                parallelism. This runtime topology signal keeps PEFT loading on
                the same path across pipeline ranks, including stages without a
                local expert module.
        """
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.uses_tied_lm_head = is_tied_word_embeddings(self.model[0])
        self.has_local_tied_lm_head = has_local_tied_lm_head(self.model[0])

        if self.uses_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            self.lm_head_param_name = lm_head_param_name
        self.is_peft = is_peft
        self.is_init_step = is_init_step
        self.skip_task_head_prefixes = skip_task_head_prefixes or []
        self.cpu_offload = cpu_offload
        self.pp_group = pp_group
        self.has_expert_parallelism = has_expert_parallelism

    def _refresh_local_tied_lm_head(self) -> None:
        """Refresh tied-head metadata after DCP has normalized module state."""
        self.has_local_tied_lm_head = has_local_tied_lm_head(self.model[0])
        if self.uses_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            self.lm_head_param_name = lm_head_param_name

    def state_dict(self) -> dict[str, Any]:
        """
        Get the model's state dictionary.

        Returns:
            Dictionary containing the model state dict, optionally offloaded to CPU.
        """
        if self.is_init_step:
            return self._get_base_model_state_dict()

        # Decide how to collect the PEFT adapter:
        #   * Local per-rank collection: directly walk named_parameters and
        #     full_tensor() the local shards. Needed for (a) BnB-quantized or
        #     EP-sharded models DCP can't traverse, and (b) PIPELINE PARALLELISM
        #     -- see below.
        #   * DCP full_state_dict: consolidates across FSDP to rank 0.
        #
        # Under pp_size>1 we MUST use the local path even for a plain (non-EP,
        # non-quant) PEFT model: DCP's full_state_dict returns the dict only on
        # global rank 0 and an EMPTY dict on every other rank (PyTorch contract).
        # That empties PP ranks 1..N-1, so the cross-PP gather below would collect
        # nothing from them. The local collection keeps each PP rank's own stage
        # adapters, which the gather then unions into the complete adapter.
        use_local_peft_collection = self.is_peft and (
            self.pp_group is not None
            or any(_has_expert_parallelism(m) for m in self.model)
            or any(_has_quantized_params(m) for m in self.model)
        )
        # A PEFT checkpoint holds every trainable parameter (the LoRA weights plus any module
        # freeze_config made trainable next to them, e.g. unfreeze_modules) and the persistent
        # buffers of the modules it stores whole, which HF PEFT restores with their parameters.
        full_modules = (
            _fully_trained_modules(_combined_trainable_module_layout(self.model, pp_group=self.pp_group))
            if self.is_peft
            else []
        )
        if use_local_peft_collection:
            # Collects exactly the trainable parameters; add the full modules' buffers.
            model_state_dict = {k: v for sd in map(_get_peft_state_dict, self.model) for k, v in sd.items()}
            if full_modules:
                for part in self.model:
                    model_state_dict.update(_get_peft_full_module_buffers(part, full_modules))
        else:
            options = StateDictOptions(cpu_offload=True) if self.cpu_offload else None
            if self.is_peft:
                options = StateDictOptions(full_state_dict=True, cpu_offload=True, ignore_frozen_params=True)

            func = partial(get_model_state_dict, options=options)
            model_state_dict = {k: v for sd in map(func, self.model) for k, v in sd.items()}
            if self.is_peft:
                # ignore_frozen_params still returns every persistent buffer; keep those of full modules.
                model_state_dict = _select_peft_tensors(model_state_dict, self.model, full_modules)

        # Pipeline parallelism partitions layers across PP ranks, so each rank's
        # local adapter (collected above) only covers its own stages. Gather the
        # PEFT tensors across the PP group and union by FQN so every rank ends up
        # with the complete adapter. No-op when pp_group is None (pp_size==1).
        # Done after the selection above so only adapter tensors travel.
        if self.is_peft and self.pp_group is not None:
            model_state_dict = _gather_peft_state_dict_across_pp(model_state_dict, self.pp_group)

        self._refresh_local_tied_lm_head()
        if self.has_local_tied_lm_head:
            model_state_dict.pop(self.lm_head_param_name, None)

        if self.is_peft:
            # HF PEFT models are saved with a "base.model." prefix. This is so they can be loaded
            # correctly with the HF PEFT API. Quantized PEFT bypasses DCP above, but the collected
            # trainable tensors still need the same on-disk key normalization.
            _add_outer_prefix(model_state_dict, "base_model.model.")
            # DoRA: rename lora_magnitude to match HF PEFT's expected key format
            _rename_dora_keys_to_hf(model_state_dict)

        return model_state_dict

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        strict: bool = True,
        broadcast_from_rank0: bool = True,
    ) -> None:
        """
        Load the state dictionary into the model.

        Args:
            state_dict: Model state mapping whose tensor values may have arbitrary
                rank and axis order and retain each parameter or buffer's exact
                shape and DTensor placement.
            strict: Whether missing or unexpected keys should fail the load.
            broadcast_from_rank0: Whether rank 0 owns the full PEFT state dict.
                Set to ``False`` when every rank in a model-local process group
                loaded the adapter independently.
        """
        if self.is_init_step:
            self._set_base_model_state_dict(state_dict)
            if self.uses_tied_lm_head and not self.is_peft:
                for model_part in self.model:
                    ensure_tied_lm_head(model_part)
            return

        # Multi-stage PP models have different state dicts for each stage.
        options = StateDictOptions(strict=strict)
        if self.is_peft:
            _drop_outer_prefix(state_dict, "base_model.model.")
            # DoRA: reverse the HF PEFT key rename so DCP can match model params
            _rename_dora_keys_from_hf(state_dict)
            # For EP models, DCP's set_model_state_dict silently skips EP-sharded
            # LoRA params (strict=False hides the FQN mismatch caused by custom
            # expert state_dict() keys like gate_up_linear.weight0). Bypass DCP.
            # Use the global topology signal first: under PP, some ranks may not
            # own an expert layer, and choosing from local modules would make
            # ranks enter different DCP collectives. Inspect every local model
            # part as a fallback for callers that do not provide the topology.
            if self.has_expert_parallelism or any(_has_expert_parallelism(part) for part in self.model):
                for model_part in self.model:
                    _set_peft_state_dict(model_part, state_dict)
                return
            options = StateDictOptions(
                strict=False,
                broadcast_from_rank0=broadcast_from_rank0,
                full_state_dict=True,
            )

        # If we intentionally skipped saving "lm_head.weight" (tied embeddings)
        # PyTorch will complain during load even with strict=False.
        # To be fully compatible we inject a reference tensor so the key exists.
        if self.uses_tied_lm_head and not self.is_peft:
            materialize_missing_tied_lm_head(
                state_dict,
                self.model[0],
                allow_current_lm_head_fallback=True,
            )

        for model_part in self.model:
            set_model_state_dict(model_part, state_dict, options=options)

        if self.uses_tied_lm_head and not self.is_peft:
            for model_part in self.model:
                ensure_tied_lm_head(model_part)

    def _get_base_model_state_dict(self) -> dict[str, Any]:
        model_state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

        self._refresh_local_tied_lm_head()
        if self.has_local_tied_lm_head:
            model_state_dict.pop(self.lm_head_param_name, None)

        if self.is_peft:
            keys_to_remove = [k for k in model_state_dict.keys() if "lora" in k]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        if self.skip_task_head_prefixes:
            # Remove task-specific heads when loading base model for fine-tuning
            # These layers don't exist in base pretrained models and will be randomly initialized
            keys_to_remove = [
                k
                for k in model_state_dict.keys()
                if any(k.startswith(prefix) for prefix in self.skip_task_head_prefixes)
            ]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        return model_state_dict

    def _set_base_model_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = partial(set_model_state_dict, model_state_dict=state_dict, options=StateDictOptions(strict=False))
        list(map(func, self.model))


class OptimizerState:
    """
    Helper class for tracking optimizer state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model associated with the optimizer.
        optimizer: The optimizer to track.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
        scheduler: Any | None = None,
        is_peft: bool = False,
        cpu_offload: bool = False,
        *,
        has_expert_parallelism: bool = False,
        optimizer_part_ids: list[int] | None = None,
    ):
        """
        Initialize an OptimizerState instance.

        The constructor simply stores references to the model, optimizer, and
        (optionally) learning-rate scheduler so that their state can be captured
        and restored by the Distributed Checkpointing (DCP) framework.

        Args:
            model: Neural-network model or pipeline model parts whose parameters
                the optimizer updates. Keeping the references allows DCP to
                re-establish each model–optimizer relationship when loading a
                checkpoint.
            optimizer: Optimizer or per-model-part optimizers whose internal
                buffers (e.g., momentum, Adam moments, step counters) need to be
                saved and restored.
            scheduler (Optional[Any], optional): Learning-rate scheduler to track
                alongside the optimizer. Pass ``None`` if no scheduler is used.
            is_peft (bool): Whether the model uses PEFT adapters (e.g. LoRA/QLoRA).
            cpu_offload: Whether DCP should move sharded tensors to CPU before saving.
            has_expert_parallelism: Whether the distributed topology uses expert
                parallelism. This runtime topology signal avoids inferring global
                EP state from only one local pipeline part.
            optimizer_part_ids: Global pipeline-stage indices corresponding to
                ``optimizer``. These namespace native optimizer state across PP
                ranks so different stages cannot produce overlapping DCP keys.
        """
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.optimizer = [optimizer] if isinstance(optimizer, torch.optim.Optimizer) else optimizer
        self.scheduler = [scheduler] if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler) else scheduler
        self.is_peft = is_peft
        self.cpu_offload = cpu_offload
        self.optimizer_part_ids = optimizer_part_ids
        if self.optimizer_part_ids is not None:
            if len(self.optimizer_part_ids) != len(self.optimizer):
                raise ValueError(
                    "Optimizer part IDs must match the local optimizer layout: "
                    f"received {len(self.optimizer_part_ids)} IDs for {len(self.optimizer)} optimizers."
                )
            if len(set(self.optimizer_part_ids)) != len(self.optimizer_part_ids):
                raise ValueError(f"Optimizer part IDs must be unique, got {self.optimizer_part_ids}.")
        self._use_native_optimizer_state = self.is_peft and (
            has_expert_parallelism
            or any(_has_expert_parallelism(model_part) for model_part in self.model)
            or any(_has_quantized_params(model_part) for model_part in self.model)
        )

    def state_dict(self) -> dict[str, Any]:
        """
        Get the optimizer and scheduler state dictionaries.

        Returns:
            Dictionary containing the optimizer and scheduler state dicts, optionally offloaded to CPU.
        """
        # For PEFT models with quantized parameters or expert parallelism, bypass
        # PyTorch DCP's get_optimizer_state_dict() which fails because DCP cannot
        # build a consistent parameter-ID-to-FQN mapping when the model contains
        # quantized frozen params (Params4bit/Int8Params) alongside trainable LoRA
        # params, or when expert weights are sharded across EP ranks (MoE+EP) and
        # the optimizer only tracks trainable params. Use native state_dict instead.
        # Adam creates state lazily only after a parameter receives a gradient. Discrete routing/indexing parameters
        # can remain trainable yet unused for a step, so normalize missing state before both native and flattened DCP
        # serialization. This makes the save and subsequent load skeletons agree without changing a future first
        # update: the materialized step and moment tensors are all zero.
        # TE FusedAdam allocates all of its state, including the per-group step counter, in its first step(), so a
        # fresh optimizer would hand DCP an empty native load skeleton and a resume would silently keep fresh
        # optimizer state. The flattened path needs no help: torch's get_optimizer_state_dict() initializes a fresh
        # optimizer itself.
        for optimizer in self.optimizer:
            _materialize_missing_adam_state(optimizer)
            if self._use_native_optimizer_state:
                _materialize_native_fused_adam_state(optimizer)
        if self._use_native_optimizer_state:
            if self.optimizer_part_ids is None:
                if len(self.optimizer) != 1:
                    raise ValueError(
                        "Native optimizer checkpointing requires global optimizer part IDs "
                        f"when saving {len(self.optimizer)} local optimizer parts."
                    )
                optimizer_state_dict = self.optimizer[0].state_dict()
            else:
                optimizer_state_dict = {
                    _OPTIMIZER_PARTS_KEY: {
                        f"{_OPTIMIZER_PART_KEY_PREFIX}{part_id}": optimizer.state_dict()
                        for part_id, optimizer in zip(self.optimizer_part_ids, self.optimizer, strict=True)
                    }
                }
        else:
            # this line automatically manages FSDP FQN's, as well as sets the default state dict type
            # to FSDP.SHARDED_STATE_DICT
            func = partial(
                get_optimizer_state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True, cpu_offload=self.cpu_offload),
            )
            optimizer_state_dict = {k: v for sd in map(func, self.model, self.optimizer) for k, v in sd.items()}

        state_dict = {
            "optim": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler[0].state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state dictionaries into the optimizer and scheduler.

        Args:
            state_dict (dict): State dictionary containing optimizer and scheduler states to load.
        """
        # Mirror state_dict(): PEFT with quantized parameters or EP topology uses native optimizer state.
        if self._use_native_optimizer_state:
            optimizer_state_dict = state_dict["optim"]
            if self.optimizer_part_ids is None:
                if len(self.optimizer) != 1:
                    raise ValueError(
                        "Native optimizer checkpointing requires global optimizer part IDs "
                        f"when loading {len(self.optimizer)} local optimizer parts."
                    )
                self.optimizer[0].load_state_dict(optimizer_state_dict)
            else:
                optimizer_parts = optimizer_state_dict.get(_OPTIMIZER_PARTS_KEY)
                if not isinstance(optimizer_parts, dict):
                    raise ValueError(
                        f"Pipeline native optimizer checkpoint is missing the '{_OPTIMIZER_PARTS_KEY}' state mapping."
                    )
                expected_part_keys = [f"{_OPTIMIZER_PART_KEY_PREFIX}{part_id}" for part_id in self.optimizer_part_ids]
                if set(optimizer_parts) != set(expected_part_keys):
                    raise ValueError(
                        "Optimizer checkpoint parts do not match the current pipeline layout: "
                        f"checkpoint has {sorted(optimizer_parts)}, current rank expects {sorted(expected_part_keys)}."
                    )
                for optimizer, part_key in zip(self.optimizer, expected_part_keys, strict=True):
                    optimizer.load_state_dict(optimizer_parts[part_key])
        else:
            # sets our state dicts on the optimizer, now that we've loaded
            func = partial(
                set_optimizer_state_dict,
                optim_state_dict=state_dict["optim"],
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            list(map(func, self.model, self.optimizer))

        # load the scheduler state if it exists
        if "sched" in state_dict and self.scheduler is not None:
            list(map(lambda x: x.load_state_dict(state_dict["sched"]), self.scheduler))

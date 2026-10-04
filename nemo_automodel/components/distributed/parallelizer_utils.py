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

from collections.abc import Iterator
from dataclasses import fields, replace

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

from nemo_automodel.components.distributed.fsdp_patches import (
    patch_fsdp_uniform_reduce_dtype as _patch_fsdp_uniform_reduce_dtype,
)
from nemo_automodel.components.distributed.fsdp_patches import (
    patch_fsdp_unused_param_reduction as _patch_fsdp_unused_param_reduction,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

# PyTorch >= 2.15: MixedPrecisionPolicy.param_dtype_override_fn keeps selected parameters in
# their storage dtype inside one FSDP unit. Required only when a model has an fp32 contract.
_HAS_PARAM_DTYPE_OVERRIDE = "param_dtype_override_fn" in {field.name for field in fields(MixedPrecisionPolicy)}

__all__ = [
    "fsdp_unit_named_parameters",
    "get_internal_fsdp_mp_policy",
    "with_fp32_compute_override",
    "reject_unsupported_mtp_cp",
    "reject_unsupported_mtp_cp_pp",
]


def reject_unsupported_mtp_cp(model: nn.Module) -> None:
    """Reject enabled MTP when the model has not declared CP support."""
    if model.supports.mtp_enabled and not model.supports.supports_mtp_cp:
        raise RuntimeError(f"{type(model).__name__} does not support MTP with context parallelism")


def reject_unsupported_mtp_cp_pp(model: nn.Module) -> None:
    """Reject MTP+CP on every trimmed pipeline stage before CP collectives."""
    is_pp_stage_fn = getattr(model, "_is_pipeline_parallel_stage", None)
    if (
        model.supports.mtp_enabled
        and not model.supports.supports_mtp_cp_pp
        and callable(is_pp_stage_fn)
        and is_pp_stage_fn()
    ):
        raise NotImplementedError(
            "MTP with context and pipeline parallelism is not supported; use PP size 1 or CP size 1"
        )


def configure_fsdp_unused_param_reduction(module: nn.Module) -> int:
    """Reduce zero gradients for FSDP parameters unused on a local CP rank.

    Packed or modality-dependent context-parallel batches may execute a module
    on only a subset of ranks. FSDP must still issue the same reduce-scatter
    sequence everywhere; otherwise a rank with ``grad is None`` can omit a
    collective and discard peer contributions. PyTorch's public API fills the
    missing local contribution with zero, analogous to DDP unused-parameter
    handling. AutoModel keeps a compatibility fallback for supported PyTorch
    versions that predate that public API.

    Args:
        module: Root module containing the FSDP units to configure.

    Returns:
        Number of FSDP units configured.
    """
    fsdp_modules = [candidate for candidate in module.modules() if isinstance(candidate, FSDPModule)]
    if not fsdp_modules:
        return 0

    # Install first so the zero fill below wraps it: the filled zero is in param
    # dtype and must be aligned with the peers' reduce-dtype accumulations before
    # the group reaches ``foreach_reduce``.
    _patch_fsdp_uniform_reduce_dtype()
    if hasattr(fsdp_modules[0], "set_reduce_scatter_unused_params"):
        for fsdp_module in fsdp_modules:
            fsdp_module.set_reduce_scatter_unused_params(True, recurse=False)
    else:
        _patch_fsdp_unused_param_reduction()
    return len(fsdp_modules)


def get_internal_fsdp_mp_policy(
    mp_policy: MixedPrecisionPolicy | None,
) -> MixedPrecisionPolicy | None:
    """Clone an FSDP policy without imposing an external output dtype.

    Internal FSDP units are implementation details inside a parent module's
    forward. Their outputs may feed an unwrapped sibling before another FSDP
    input cast, so they preserve the wrapped module's natural output dtype.

    Args:
        mp_policy: Mixed-precision policy inherited from the enclosing FSDP
            boundary, or ``None`` when mixed precision is disabled.

    Returns:
        A cloned policy with ``output_dtype=None``, or ``None`` when no policy
        was provided. Parameter, reduction, and input-cast settings are unchanged.
    """
    if mp_policy is None:
        return None
    return replace(mp_policy, output_dtype=None)


def fsdp_unit_named_parameters(module: nn.Module) -> Iterator[tuple[str, nn.Parameter]]:
    """Yield the parameters FSDP2 assigns to ``module``'s own unit.

    Parameters below an already-sharded descendant belong to that descendant's
    unit and are excluded, exactly as ``fully_shard`` excludes them. A parameter
    reachable through several modules (tied weights) is yielded once.

    Args:
        module: Module about to become one FSDP unit.

    Yields:
        ``(name, parameter)`` pairs with names relative to ``module``.
    """
    seen: set[int] = set()

    def _walk(current: nn.Module, prefix: str) -> Iterator[tuple[str, nn.Parameter]]:
        for name, param in current.named_parameters(recurse=False):
            if id(param) not in seen:
                seen.add(id(param))
                yield f"{prefix}{name}", param
        for child_name, child in current.named_children():
            if not isinstance(child, FSDPModule):
                yield from _walk(child, f"{prefix}{child_name}.")

    return _walk(module, "")


def with_fp32_compute_override(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy | None,
    fp32_compute_module_names: tuple[str, ...],
    ignored_params: set[nn.Parameter] | None = None,
    *,
    module_name: str = "",
) -> MixedPrecisionPolicy | None:
    """Keep ``module``'s fp32-contract parameters in fp32 under a lower-precision policy.

    A parameter computes in fp32 when its canonical name contains one of
    ``fp32_compute_module_names`` (the model's ``_keep_in_fp32_modules_strict``).
    Every other parameter computes in ``mp_policy.param_dtype``.

    PyTorch's ``param_dtype_override_fn`` keeps a parameter in its *storage*
    dtype; it cannot upcast. fp32 compute therefore requires fp32 storage, which
    is what ``model.dtype: float32`` (fp32 master weights) provides.

    Only the parameters of ``module``'s own unit are considered (see
    :func:`fsdp_unit_named_parameters`), so the model root can be sharded after
    its layers without re-checking them.

    Args:
        module: Module about to be sharded as one FSDP unit.
        mp_policy: Policy of the enclosing FSDP boundary, or ``None``.
        fp32_compute_module_names: Parameter-name substrings that must compute in fp32.
        ignored_params: Parameters owned by another FSDP or parallelism unit.
        module_name: Model-level name of ``module``. When given, tokens are matched
            against ``module_name`` joined with each parameter's relative name, so
            model-level tokens such as ``lm_head`` resolve inside the ``lm_head`` unit.

    Returns:
        ``mp_policy`` itself when no parameter needs fp32 compute, otherwise a
        copy carrying ``param_dtype_override_fn``.

    Raises:
        ValueError: Trainable parameters mix storage dtypes, or a parameter must
            compute in fp32 but is not stored in fp32.
        RuntimeError: PyTorch lacks ``param_dtype_override_fn`` and fp32 compute is needed.
    """
    ignored_param_ids = {id(param) for param in ignored_params or ()}
    unit_params = [
        (name, param)
        for name, param in fsdp_unit_named_parameters(module)
        if id(param) not in ignored_param_ids and param.dtype.is_floating_point
    ]
    # FSDP2 packs one unit into one all-gather buffer, so trainable parameters must
    # share a storage dtype. Fail here with the offending names instead of at the
    # first forward with PyTorch's bare uniformity assertion.
    trainable_dtypes: dict[torch.dtype, list[str]] = {}
    for name, param in unit_params:
        if param.requires_grad:
            trainable_dtypes.setdefault(param.dtype, []).append(name)
    if len(trainable_dtypes) > 1:
        minority = min(trainable_dtypes.items(), key=lambda item: len(item[1]))
        raise ValueError(
            f"FSDP2 requires one storage dtype per unit but trainable parameters use {sorted(map(str, trainable_dtypes))}; "
            f"{minority[0]}: {', '.join(minority[1][:8])}. Set model.dtype to float32 (fp32 master weights) so fp32 "
            "parameters can share the block's unit."
        )
    if mp_policy is None or mp_policy.param_dtype in (None, torch.float32):
        return mp_policy
    fp32_param_ids: set[int] = set()
    for name, param in unit_params:
        fqn = canonical_parameter_fqn(f"{module_name}.{name}" if module_name else name)
        if not any(token in fqn for token in fp32_compute_module_names):
            continue
        if param.dtype != torch.float32:
            if not param.requires_grad:
                # A frozen lower-precision copy (e.g. a LoRA base) has no fp32 storage to keep.
                continue
            raise ValueError(
                f"{name} must compute in fp32 but is stored in {param.dtype}. FSDP2 keeps fp32 compute parameters "
                "in their storage dtype, so set model.dtype to float32 (fp32 master weights) for this model."
            )
        fp32_param_ids.add(id(param))
    if not fp32_param_ids:
        return mp_policy
    if not _HAS_PARAM_DTYPE_OVERRIDE:
        raise RuntimeError(
            "Keeping fp32 parameters in fp32 under FSDP2 mixed precision requires PyTorch >= 2.15 "
            "(MixedPrecisionPolicy.param_dtype_override_fn)."
        )
    return replace(
        mp_policy,
        param_dtype_override_fn=lambda param: torch.float32 if id(param) in fp32_param_ids else None,
    )

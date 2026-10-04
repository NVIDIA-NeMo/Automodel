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

from dataclasses import fields, replace
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)

from nemo_automodel.components.distributed.fsdp_patches import (
    patch_fsdp_uniform_reduce_dtype as _patch_fsdp_uniform_reduce_dtype,
)
from nemo_automodel.components.distributed.fsdp_patches import (
    patch_fsdp_unused_param_reduction as _patch_fsdp_unused_param_reduction,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.parallelizer import ModelParallelizer

# PyTorch >= 2.15: MixedPrecisionPolicy.param_dtype_override_fn keeps selected parameters in
# their storage dtype inside one FSDP unit. Required only when a model has an fp32 contract.
_HAS_PARAM_DTYPE_OVERRIDE = "param_dtype_override_fn" in {field.name for field in fields(MixedPrecisionPolicy)}

__all__ = [
    "fully_shard_by_dtype",
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


def with_fp32_compute_override(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy | None,
    fp32_compute_module_names: tuple[str, ...],
    ignored_params: set[nn.Parameter] | None = None,
) -> MixedPrecisionPolicy | None:
    """Keep ``module``'s fp32-contract parameters in fp32 under a lower-precision policy.

    A parameter computes in fp32 when its canonical name contains one of
    ``fp32_compute_module_names`` (the model's ``_keep_in_fp32_modules_strict``)
    or when the checkpoint loader recorded fp32 as its original dtype
    (``tensor._hf_compute_dtype``, see ``_restore_loaded_model_dtype``). Every
    other parameter computes in ``mp_policy.param_dtype``.

    PyTorch's ``param_dtype_override_fn`` keeps a parameter in its *storage*
    dtype; it cannot upcast. fp32 compute therefore requires fp32 storage, which
    is what ``model.dtype: float32`` (fp32 master weights) provides.

    Args:
        module: Module about to be sharded as one FSDP unit.
        mp_policy: Policy of the enclosing FSDP boundary, or ``None``.
        fp32_compute_module_names: Parameter-name substrings that must compute in fp32.
        ignored_params: Parameters owned by another FSDP or parallelism unit.

    Returns:
        ``mp_policy`` itself when no parameter needs fp32 compute, otherwise a
        copy carrying ``param_dtype_override_fn``.

    Raises:
        ValueError: A parameter must compute in fp32 but is not stored in fp32.
        RuntimeError: PyTorch lacks ``param_dtype_override_fn`` and fp32 compute is needed.
    """
    if mp_policy is None or mp_policy.param_dtype in (None, torch.float32):
        return mp_policy
    ignored_param_ids = {id(param) for param in ignored_params or ()}
    fp32_param_ids: set[int] = set()
    for name, param in module.named_parameters():
        if id(param) in ignored_param_ids or not param.dtype.is_floating_point:
            continue
        pinned = any(token in canonical_parameter_fqn(name) for token in fp32_compute_module_names)
        recorded = getattr(param, "_hf_compute_dtype", None)
        if not pinned and recorded != torch.float32:
            continue
        if param.dtype != torch.float32:
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


def fully_shard_by_dtype(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    fp32_compute_module_names: tuple[str, ...] = (),
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    model_parallelizer: "ModelParallelizer | None" = None,
) -> None:
    """Fully shard ``module`` as one FSDP unit whose fp32-contract parameters compute in fp32.

    Everything computes in ``mp_policy.param_dtype`` (e.g. bf16) except the
    parameters selected by :func:`with_fp32_compute_override`, which keep their
    fp32 storage dtype through all-gather. Modules owning such parameters cast
    their own inputs; the unit's input and output casting is unchanged.

    Args:
        module: Module to shard, typically one transformer block.
        mesh: Device mesh for FSDP sharding.
        mp_policy: Mixed-precision policy of the enclosing boundary.
        offload_policy: FSDP offload policy.
        fp32_compute_module_names: Parameter-name substrings that must compute in
            fp32, sourced from the model's ``_keep_in_fp32_modules_strict``.
        reshard_after_forward: Optional FSDP2 reshard override. ``None`` leaves
            the FSDP2 default unchanged.
        ignored_params: Parameters already owned by another FSDP or parallelism
            unit. They are excluded from the fp32 contract and forwarded to FSDP.
        model_parallelizer: Optional model sidecar that owns the FSDP primitive.
    """
    shard_module = fully_shard if model_parallelizer is None else model_parallelizer._fully_shard_module
    kwargs = {
        "mesh": mesh,
        "mp_policy": with_fp32_compute_override(module, mp_policy, fp32_compute_module_names, ignored_params),
        "offload_policy": offload_policy,
    }
    if reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = reshard_after_forward
    if ignored_params:
        module_param_ids = {id(param) for param in module.parameters()}
        module_ignored_params = {param for param in ignored_params if id(param) in module_param_ids}
        if module_ignored_params:
            kwargs["ignored_params"] = module_ignored_params
    shard_module(module, **kwargs)

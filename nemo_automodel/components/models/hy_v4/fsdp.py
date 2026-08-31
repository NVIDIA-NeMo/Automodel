# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""HY V4 FSDP policy for reference-sensitive FP32 parameter islands."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.models.hy_v4.hc import HyV4HCHead, HyV4HCLayer
from nemo_automodel.components.models.hy_v4.layers import HyV4FP32Parameter

_HY_V4_FP32_MODULE_TYPES = (HyV4FP32Parameter, HyV4HCHead, HyV4HCLayer)


def _has_fsdp_state(module: nn.Module) -> bool:
    """Return whether ``module`` is already an FSDP2 unit."""
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    except ImportError:
        return False
    return _get_module_fsdp_state(module) is not None


def _floating_param_dtypes(module: nn.Module) -> set[torch.dtype]:
    """Return storage dtypes for floating parameters owned below ``module``."""
    return {parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()}


def _fp32_mp_policy(mp_policy: MixedPrecisionPolicy | object) -> MixedPrecisionPolicy | object:
    """Pin a reference-sensitive HY4 unit to FP32 parameter and reduction math."""
    if not isinstance(mp_policy, MixedPrecisionPolicy):
        return mp_policy
    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        # HY V4's FP32 islands explicitly cast their mathematical result back
        # to the activation dtype where required. Preserve that module contract.
        output_dtype=None,
        cast_forward_inputs=mp_policy.cast_forward_inputs,
    )


def _filtered_fsdp_kwargs(module: nn.Module, fsdp_kwargs: dict) -> dict:
    """Keep only ignored parameters that belong to the requested FSDP unit."""
    ignored_params = fsdp_kwargs.get("ignored_params")
    if not ignored_params:
        return fsdp_kwargs
    module_param_ids = {id(parameter) for parameter in module.parameters()}
    filtered = {parameter for parameter in ignored_params if id(parameter) in module_param_ids}
    if len(filtered) == len(ignored_params):
        return fsdp_kwargs
    result = dict(fsdp_kwargs)
    if filtered:
        result["ignored_params"] = filtered
    else:
        result.pop("ignored_params", None)
    return result


def _fully_shard_once(
    module: nn.Module,
    *,
    mesh,
    mp_policy,
    offload_policy,
    fp32_policy: bool,
    **fsdp_kwargs,
) -> nn.Module:
    """Apply one HY4-aware FSDP boundary unless the module is already sharded."""
    if module is None or _has_fsdp_state(module):
        return module
    return fully_shard(
        module,
        mesh=mesh,
        mp_policy=_fp32_mp_policy(mp_policy) if fp32_policy else mp_policy,
        offload_policy=offload_policy,
        **_filtered_fsdp_kwargs(module, fsdp_kwargs),
    )


def _iter_fp32_modules(module: nn.Module):
    """Yield typed HY4 FP32 parameter islands below ``module`` once each."""
    seen: set[int] = set()
    for submodule in module.modules():
        if submodule is module or id(submodule) in seen:
            continue
        if not isinstance(submodule, _HY_V4_FP32_MODULE_TYPES):
            continue
        if _floating_param_dtypes(submodule) != {torch.float32}:
            continue
        seen.add(id(submodule))
        yield submodule


def _holder_fsdp_kwargs(module: nn.Module, fsdp_kwargs: dict) -> dict:
    """Keep a returned FP32 parameter materialized until its parent consumes it."""
    if not isinstance(module, HyV4FP32Parameter) or fsdp_kwargs.get("reshard_after_forward") is False:
        return fsdp_kwargs
    # The holder returns its parameter to its parent. Keep it materialized until
    # the parent's kernel has consumed the tensor.
    result = dict(fsdp_kwargs)
    result["reshard_after_forward"] = False
    return result


def fully_shard_hy_v4(module: nn.Module, mesh, mp_policy, offload_policy=None, **fsdp_kwargs) -> nn.Module:
    """Shard a HY4 module without mixing reference FP32 and BF16 parameters.

    Args:
        module: HY4 module selected by the model-owned sharding hook.
        mesh: FSDP device mesh.
        mp_policy: Caller mixed-precision policy for ordinary model weights.
        offload_policy: Optional FSDP offload policy.
        **fsdp_kwargs: Additional keyword arguments forwarded to ``fully_shard``.

    Returns:
        The FSDP-sharded module. FP32 iHC/sink holders become nested units and
        their mathematical outputs retain the layouts of their inputs.
    """
    policy_dtype = getattr(mp_policy, "param_dtype", None)
    all_fp32_with_fp32_policy = _floating_param_dtypes(module) == {torch.float32} and (
        isinstance(module, _HY_V4_FP32_MODULE_TYPES) or policy_dtype in (None, torch.float32)
    )
    if all_fp32_with_fp32_policy:
        return _fully_shard_once(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=True,
            **_holder_fsdp_kwargs(module, fsdp_kwargs),
        )
    for fp32_module in _iter_fp32_modules(module):
        _fully_shard_once(
            fp32_module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=True,
            **_holder_fsdp_kwargs(fp32_module, fsdp_kwargs),
        )
    return _fully_shard_once(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        fp32_policy=False,
        **fsdp_kwargs,
    )

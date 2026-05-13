# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

_DSV4_CLASS_NAMES = {
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4Block",
}

_DSV4_FP32_MODULE_SUFFIXES = (
    "attn_hc",
    "ffn_hc",
    "hc_head",
    "lm_head",
    "self_attn.compressor.wkv",
    "self_attn.compressor.wgate",
    "self_attn.compressor.indexer.wkv",
    "self_attn.compressor.indexer.wgate",
)

_DSV4_DIRECT_FP32_PARAM_SUFFIXES = (
    "self_attn.sinks",
    "self_attn.compressor.ape",
    "self_attn.compressor.indexer.ape",
)


def _matches_suffix(name: str, suffix: str) -> bool:
    return name == suffix or name.endswith(f".{suffix}")


def _has_fsdp_state(module: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    except ImportError:
        return False

    return _get_module_fsdp_state(module) is not None


def _module_config_model_type(module: nn.Module) -> str | None:
    return getattr(getattr(module, "config", None), "model_type", None)


def _is_deepseek_v4_module(module: nn.Module) -> bool:
    if module.__class__.__name__ in _DSV4_CLASS_NAMES or _module_config_model_type(module) == "deepseek_v4":
        return True

    wrapped = getattr(module, "_checkpoint_wrapped_module", None)
    if wrapped is not None and _is_deepseek_v4_module(wrapped):
        return True

    return any(
        sub.__class__.__name__ in _DSV4_CLASS_NAMES or _module_config_model_type(sub) == "deepseek_v4"
        for sub in module.modules()
        if sub is not module
    )


def _floating_param_dtypes(module: nn.Module) -> set[torch.dtype]:
    return {param.dtype for param in module.parameters() if torch.is_floating_point(param)}


def _fp32_mp_policy(mp_policy):
    if not isinstance(mp_policy, MixedPrecisionPolicy):
        return mp_policy

    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
        cast_forward_inputs=mp_policy.cast_forward_inputs,
    )


def _fsdp_kwargs_for_module(module: nn.Module, fsdp_kwargs: dict) -> dict:
    ignored_params = fsdp_kwargs.get("ignored_params")
    if not ignored_params:
        return fsdp_kwargs

    module_param_ids = {id(param) for param in module.parameters()}
    filtered_ignored_params = {param for param in ignored_params if id(param) in module_param_ids}
    if len(filtered_ignored_params) == len(ignored_params):
        return fsdp_kwargs

    filtered_kwargs = dict(fsdp_kwargs)
    if filtered_ignored_params:
        filtered_kwargs["ignored_params"] = filtered_ignored_params
    else:
        filtered_kwargs.pop("ignored_params", None)
    return filtered_kwargs


def _fully_shard_once(module: nn.Module, *, mesh, mp_policy, offload_policy, fp32_policy: bool, **fsdp_kwargs):
    if module is None or _has_fsdp_state(module):
        return module

    return fully_shard(
        module,
        mesh=mesh,
        mp_policy=_fp32_mp_policy(mp_policy) if fp32_policy else mp_policy,
        offload_policy=offload_policy,
        **_fsdp_kwargs_for_module(module, fsdp_kwargs),
    )


def _iter_dsv4_fp32_modules(module: nn.Module):
    seen: set[int] = set()
    for name, submodule in module.named_modules():
        if not name or id(submodule) in seen:
            continue
        if not any(_matches_suffix(name, suffix) for suffix in _DSV4_FP32_MODULE_SUFFIXES):
            continue
        if _floating_param_dtypes(submodule) != {torch.float32}:
            continue
        seen.add(id(submodule))
        yield submodule


def _nested_fsdp_param_ids(module: nn.Module) -> set[int]:
    ids: set[int] = set()
    for child in module.modules():
        if child is module:
            continue
        if _has_fsdp_state(child):
            ids.update(id(param) for param in child.parameters())
    return ids


def _direct_dsv4_fp32_params(module: nn.Module, skipped_param_ids: set[int]) -> set[nn.Parameter]:
    ignored_params: set[nn.Parameter] = set()
    for name, param in module.named_parameters():
        if id(param) in skipped_param_ids or param.dtype != torch.float32:
            continue
        if any(_matches_suffix(name, suffix) for suffix in _DSV4_DIRECT_FP32_PARAM_SUFFIXES):
            ignored_params.add(param)
    return ignored_params


def fully_shard_deepseek_v4(module: nn.Module, mesh, mp_policy, offload_policy=None, **fsdp_kwargs):
    """Apply FSDP2 to DeepSeek-V4 without mixing fp32 and bf16 params in one unit.

    This is intentionally model-specific.  DeepSeek-V4 keeps a small set of
    reference-sensitive tensors in fp32, while the existing DeepEP path expects
    the transformer block itself to remain the main FSDP unit.
    """
    if _floating_param_dtypes(module) == {torch.float32}:
        return _fully_shard_once(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=True,
            **fsdp_kwargs,
        )

    if not _is_deepseek_v4_module(module):
        return _fully_shard_once(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=False,
            **fsdp_kwargs,
        )

    wrapped_param_ids: set[int] = set()
    for fp32_module in _iter_dsv4_fp32_modules(module):
        _fully_shard_once(
            fp32_module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=True,
            **fsdp_kwargs,
        )
        wrapped_param_ids.update(id(param) for param in fp32_module.parameters())

    ignored_params = set(fsdp_kwargs.get("ignored_params") or ())
    skipped_param_ids = {id(param) for param in ignored_params}
    skipped_param_ids.update(_nested_fsdp_param_ids(module))
    skipped_param_ids.update(wrapped_param_ids)
    ignored_params.update(_direct_dsv4_fp32_params(module, skipped_param_ids))

    parent_kwargs = dict(fsdp_kwargs)
    if ignored_params:
        parent_kwargs["ignored_params"] = ignored_params

    return _fully_shard_once(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        fp32_policy=False,
        **parent_kwargs,
    )

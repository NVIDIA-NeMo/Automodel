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

import inspect
from typing import Callable, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    fully_shard,
)

UniformSubtreeItem = Union[Tuple[nn.Module, torch.dtype], Tuple[str, nn.Module, torch.dtype]]


def iter_maximal_uniform_dtype_subtrees(
    module: nn.Module,
    *,
    include_buffers: bool = True,
    tensor_pred: Optional[Callable[[torch.Tensor], bool]] = None,
    return_paths: bool = False,
) -> Iterator[UniformSubtreeItem]:
    """
    Traverse `module` and yield maximal submodules whose entire subtree has a unified dtype.

    - include_buffers: include buffers in dtype unification checks.
    - tensor_pred: predicate to choose which tensors to consider (default: all).
                   Example: tensor_pred=torch.is_floating_point  (to consider only FP tensors)
    - return_paths: if True, yields (qualified_name, module, dtype); else (module, dtype).

    Notes:
    - If a module subtree has no tensors passing `tensor_pred`, it is ignored.
    - Maximality ensures no yielded module is a strict child of another yielded module.
    """
    if tensor_pred is None:
        tensor_pred = lambda t: True

    def _local_dtype_set(m: nn.Module) -> Set[torch.dtype]:
        ds: Set[torch.dtype] = set()
        for p in m.parameters(recurse=False):
            if tensor_pred(p):
                ds.add(p.dtype)
        if include_buffers:
            for b in m.buffers(recurse=False):
                if tensor_pred(b):
                    ds.add(b.dtype)
        return ds

    def _visit(m: nn.Module, path: Tuple[str, ...]) -> Tuple[Set[torch.dtype], List[UniformSubtreeItem]]:
        local = _local_dtype_set(m)
        subtree_dtypes: Set[torch.dtype] = set(local)
        collected: List[UniformSubtreeItem] = []

        # Recurse into children
        for name, child in m.named_children():
            child_set, child_yields = _visit(child, path + (name,))
            subtree_dtypes |= child_set
            collected.extend(child_yields)

        # If entire subtree has exactly one dtype (and not empty), this node is maximal: override children yields
        if len(subtree_dtypes) == 1:
            if subtree_dtypes:
                dtype = next(iter(subtree_dtypes))
                if return_paths:
                    qname = ".".join(path)  # empty string at root
                    return subtree_dtypes, [(qname, m, dtype)]
                else:
                    return subtree_dtypes, [(m, dtype)]
            # else: no tensors in subtree -> ignore entirely
        # Not uniform -> keep whatever maximal sets children produced
        return subtree_dtypes, collected

    _, items = _visit(module, ())
    # Stream results
    for it in items:
        yield it


def _group_params_by_dtype(layer):
    ans = {}
    for name, param in layer.named_parameters():
        if param.dtype not in ans:
            ans[param.dtype] = []
        ans[param.dtype].append(param)
    return ans


def _get_module_from_path(layer, path):
    for name in path.split("."):
        layer = getattr(layer, name)
    return layer


def _call_fully_shard(
    module,
    mesh,
    mp_policy,
    offload_policy,
    reshard_after_forward=None,
    ignored_params: Optional[Set[torch.nn.Parameter]] = None,
):
    kwargs = {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": offload_policy}
    if reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = reshard_after_forward
    if ignored_params:
        module_params = set(module.parameters())
        local_ignored_params = {p for p in ignored_params if p in module_params}
        if local_ignored_params:
            kwargs["ignored_params"] = local_ignored_params
    fully_shard(module, **kwargs)


def _fully_shard(
    module,
    mesh,
    mp_policy,
    offload_policy,
    reshard_after_forward=None,
    ignored_params: Optional[Set[torch.nn.Parameter]] = None,
):
    if isinstance(module, nn.ModuleList):
        for layer in module:
            _fully_shard(
                layer,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward=reshard_after_forward,
                ignored_params=ignored_params,
            )
    else:
        _call_fully_shard(
            module,
            mesh,
            mp_policy,
            offload_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params,
        )


def _call_nested_fully_shard(
    module,
    mesh,
    mp_policy,
    offload_policy,
    reshard_after_forward=None,
    ignored_params: Optional[Set[torch.nn.Parameter]] = None,
):
    """Call _fully_shard while remaining backward-compatible with tests that monkeypatch its signature."""
    kwargs = {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": offload_policy}
    try:
        params = inspect.signature(_fully_shard).parameters
    except (TypeError, ValueError):
        params = {}
    if "reshard_after_forward" in params:
        kwargs["reshard_after_forward"] = reshard_after_forward
    if "ignored_params" in params:
        kwargs["ignored_params"] = ignored_params
    _fully_shard(module, **kwargs)


def fully_shard_by_dtype(
    module,
    mesh,
    mp_policy,
    offload_policy,
    reshard_after_forward=None,
    ignored_params: Optional[Set[torch.nn.Parameter]] = None,
):
    # calling _group_params_by_dtype is not optimal here, because we may
    # end up with two traversals over the module, but this code is not in the hot path.
    grouped_params = _group_params_by_dtype(module)
    if len(grouped_params) == 0:
        return
    elif len(grouped_params) == 1:
        _call_fully_shard(
            module,
            mesh,
            mp_policy,
            offload_policy,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params,
        )
    else:
        least_items_dtype = min(grouped_params.items(), key=lambda x: len(x[1]))[0]
        for path, mod, dtype in iter_maximal_uniform_dtype_subtrees(
            module,
            tensor_pred=torch.is_floating_point,
            return_paths=True,
        ):
            if (len(grouped_params) == 2 and dtype == least_items_dtype) or len(grouped_params) > 2:
                _call_nested_fully_shard(
                    _get_module_from_path(module, path),
                    mesh=mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    ignored_params=ignored_params,
                )
        if len(grouped_params) == 2:
            _call_fully_shard(
                module,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward=reshard_after_forward,
                ignored_params=ignored_params,
            )

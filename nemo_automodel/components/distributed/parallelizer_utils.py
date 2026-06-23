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

from copy import copy
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)

UniformSubtreeItem = Union[Tuple[nn.Module, torch.dtype], Tuple[str, nn.Module, torch.dtype]]


def iter_maximal_uniform_dtype_subtrees(
    module: nn.Module,
    *,
    include_buffers: bool = True,
    tensor_pred: Optional[Callable[[torch.Tensor], bool]] = None,
    dtype_of: Optional[Callable[[torch.Tensor], torch.dtype]] = None,
    return_paths: bool = False,
) -> Iterator[UniformSubtreeItem]:
    """
    Traverse `module` and yield maximal submodules whose entire subtree has a unified dtype.

    - include_buffers: include buffers in dtype unification checks.
    - tensor_pred: predicate to choose which tensors to consider (default: all).
                   Example: tensor_pred=torch.is_floating_point  (to consider only FP tensors)
    - dtype_of: maps a tensor to the dtype used for unification (default: its storage
                dtype ``t.dtype``). Pass a custom function to group by *compute* dtype
                rather than storage dtype.
    - return_paths: if True, yields (qualified_name, module, dtype); else (module, dtype).

    Notes:
    - If a module subtree has no tensors passing `tensor_pred`, it is ignored.
    - Maximality ensures no yielded module is a strict child of another yielded module.
    """
    if tensor_pred is None:
        tensor_pred = lambda t: True
    if dtype_of is None:
        dtype_of = lambda t: t.dtype

    def _local_dtype_set(m: nn.Module) -> Set[torch.dtype]:
        ds: Set[torch.dtype] = set()
        for p in m.parameters(recurse=False):
            if tensor_pred(p):
                ds.add(dtype_of(p))
        if include_buffers:
            for b in m.buffers(recurse=False):
                if tensor_pred(b):
                    ds.add(dtype_of(b))
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


def _group_params_by_dtype(
    layer: nn.Module,
    dtype_of: Optional[Callable[[torch.Tensor], torch.dtype]] = None,
) -> Dict[torch.dtype, List[nn.Parameter]]:
    if dtype_of is None:
        dtype_of = lambda t: t.dtype
    ans: Dict[torch.dtype, List[nn.Parameter]] = {}
    for name, param in layer.named_parameters():
        dtype = dtype_of(param)
        if dtype not in ans:
            ans[dtype] = []
        ans[dtype].append(param)
    return ans


def _get_module_from_path(layer: nn.Module, path: str) -> nn.Module:
    for name in path.split("."):
        layer = getattr(layer, name)
    return layer


def _fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[OffloadPolicy],
    reshard_after_forward: Optional[bool] = None,
) -> None:
    if isinstance(module, nn.ModuleList):
        for layer in module:
            _fully_shard(layer, mesh, mp_policy, offload_policy, reshard_after_forward)
    else:
        kwargs = {
            "mesh": mesh,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
        }
        if reshard_after_forward is not None:
            kwargs["reshard_after_forward"] = reshard_after_forward
        fully_shard(module, **kwargs)


def _mp_policy_with_param_dtype(
    mp_policy: Optional[MixedPrecisionPolicy],
    param_dtype: torch.dtype,
) -> Optional[MixedPrecisionPolicy]:
    if mp_policy is None:
        return None
    mp_policy_copy = copy(mp_policy)
    object.__setattr__(mp_policy_copy, "param_dtype", param_dtype)
    return mp_policy_copy


def _make_compute_dtype_fn(
    module: nn.Module,
    mp_policy: Optional[MixedPrecisionPolicy],
    fp32_compute_module_names: Tuple[str, ...],
) -> Callable[[torch.Tensor], torch.dtype]:
    """Build the per-parameter *compute* dtype resolver used to group FSDP units.

    The compute dtype of a floating tensor is resolved by precedence:

      1. Pinned fp32 -- the tensor's name matches ``fp32_compute_module_names``
         (from the model's ``_keep_in_fp32_modules_strict``). Authoritative, works
         even from-scratch / quantized where there is no checkpoint to read.
      2. HF-recorded dtype -- ``tensor._hf_compute_dtype``, the checkpoint's original
         dtype recorded at load time (see ``_restore_loaded_model_dtype``). This makes
         any checkpoint-loaded model keep its intrinsically-fp32 params in fp32 compute
         automatically, even after storage was upcast for fp32 master weights.
      3. Fallback -- when the tensor carries no compute hint, the result depends on
         whether the module's floating-point *storage* is uniform:
           * uniform storage -- ``mp_policy.param_dtype`` (the requested mixed-precision
             compute dtype, typically bf16). This is the fp32-master-weights case: the
             uniform-fp32 storage is artificially widened and should compute in the
             policy dtype. Falls back to the storage dtype when no policy is given.
           * mixed storage -- the tensor's own storage dtype. A param whose storage
             differs from its peers is intrinsically that dtype (not a master weight),
             so it must compute in it. Applying the policy here would force differently
             stored params into one compute dtype and re-introduce the mixed *original*
             dtype that stock FSDP2 rejects (``_init_mp_dtypes``).

    Non-floating tensors always keep their storage dtype.
    """
    policy_dtype = getattr(mp_policy, "param_dtype", None)

    # The policy fallback only represents "fp32 master weights -> compute in policy
    # dtype" when the module's floating storage is uniform. If storage is already
    # mixed, unhinted params keep their storage dtype instead (see precedence above).
    floating_storage_dtypes = {t.dtype for t in (*module.parameters(), *module.buffers()) if t.dtype.is_floating_point}
    storage_is_uniform = len(floating_storage_dtypes) <= 1

    pinned_ids: Set[int] = set()
    if fp32_compute_module_names:
        for name, tensor in (*module.named_parameters(), *module.named_buffers()):
            if any(token in name for token in fp32_compute_module_names):
                pinned_ids.add(id(tensor))

    def compute_dtype_of(t: torch.Tensor) -> torch.dtype:
        if not t.dtype.is_floating_point:
            return t.dtype
        if id(t) in pinned_ids:
            return torch.float32
        recorded = getattr(t, "_hf_compute_dtype", None)
        if recorded is not None and recorded.is_floating_point:
            return recorded
        if policy_dtype is not None and storage_is_uniform:
            return policy_dtype
        return t.dtype

    return compute_dtype_of


def fully_shard_by_dtype(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: Optional[MixedPrecisionPolicy],
    offload_policy: Optional[OffloadPolicy],
    fp32_compute_module_names: Tuple[str, ...] = (),
    reshard_after_forward: Optional[bool] = None,
) -> None:
    """Fully shard a module so every parameter computes in its required dtype.

    The intent is simple: compute everything in ``mp_policy.param_dtype`` (e.g. bf16)
    except parameters that must stay in fp32 -- their FSDP unit gets ``param_dtype=fp32``
    while the rest of the module computes in the policy dtype. A parameter "must stay
    fp32" if it is pinned via ``fp32_compute_module_names`` or HF stored it in fp32 (see
    ``_make_compute_dtype_fn`` for the full precedence). This decouples *compute* dtype
    from *storage* dtype, so fp32 master weights (uniform fp32 storage) still compute in
    bf16 for the bulk.

    Implementation: group the module's parameters by their resolved compute dtype and
    shard so each FSDP unit is compute-dtype-uniform. The three cases below differ only
    in sharding granularity:

      * 1 compute dtype  -> shard the whole module once.
      * 2 compute dtypes -> shard the minority-dtype subtrees on their own, then shard
        the parent with the majority dtype (keeps the bulk as one FSDP unit).
      * 3+ compute dtypes -> shard every maximal compute-dtype-uniform subtree on its own.

    Args:
        fp32_compute_module_names: Parameter/buffer name substrings that must compute in
            fp32 (e.g. ``("_fp32_params",)`` for Qwen3.5's GatedDeltaNet fp32 holder).
            Sourced from the model's ``_keep_in_fp32_modules_strict``.
        reshard_after_forward: Optional FSDP2 reshard override for this module.
            ``None`` leaves the caller's default FSDP2 behavior unchanged.
    """
    compute_dtype_of = _make_compute_dtype_fn(module, mp_policy, fp32_compute_module_names)

    # FSDP2 requires every param group to be uniform in *storage* (original) dtype
    # -- ``_init_mp_dtypes`` asserts ``{p.orig_dtype}`` is a singleton -- while a group's
    # ``param_dtype`` controls *compute* dtype. These are independent axes, so we group by
    # the (storage, compute) pair: this keeps each FSDP unit storage-uniform (satisfying the
    # assertion even when two different storage dtypes share one compute dtype, e.g. bf16 and
    # fp32 weights both computing in bf16) while still splitting params that need a different
    # compute dtype. ``key[1]`` is the compute dtype used as the unit's ``param_dtype``.
    group_key_of = lambda t: (t.dtype, compute_dtype_of(t))

    # calling _group_params_by_dtype is not optimal here, because we may
    # end up with two traversals over the module, but this code is not in the hot path.
    grouped_params = _group_params_by_dtype(module, dtype_of=group_key_of)
    if len(grouped_params) == 0:
        return
    elif len(grouped_params) == 1:
        key = next(iter(grouped_params))
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=_mp_policy_with_param_dtype(mp_policy, key[1]),
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
        )
    else:
        least_items_key = min(grouped_params.items(), key=lambda x: len(x[1]))[0]
        for path, mod, key in iter_maximal_uniform_dtype_subtrees(
            module,
            tensor_pred=torch.is_floating_point,
            dtype_of=group_key_of,
            return_paths=True,
        ):
            if (len(grouped_params) == 2 and key == least_items_key) or len(grouped_params) > 2:
                _fully_shard(
                    _get_module_from_path(module, path),
                    mesh=mesh,
                    mp_policy=_mp_policy_with_param_dtype(mp_policy, key[1]),
                    offload_policy=offload_policy,
                    reshard_after_forward=reshard_after_forward,
                )
        if len(grouped_params) == 2:
            parent_key = next(key for key in grouped_params if key != least_items_key)
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=_mp_policy_with_param_dtype(mp_policy, parent_key[1]),
                offload_policy=offload_policy,
                reshard_after_forward=reshard_after_forward,
            )

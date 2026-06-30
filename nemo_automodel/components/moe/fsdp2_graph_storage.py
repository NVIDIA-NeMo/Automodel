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

"""Retain eager FSDP2 expert all-gather storage for partial CUDA graphs.

PyTorch FSDP2 keeps an unsharded ``nn.Parameter`` object alive across
all-gathers, but normally resizes its aliased all-gather storage to zero when
resharding. A CUDA graph cannot tolerate the resulting device-address change.
This module provides a narrow, instance-local interception for
``GroupedExpertsTeOps``: logical resharding still exposes the DTensor shards,
while the all-gather allocation captured by the graph remains resident.

The implementation deliberately depends on a small private FSDP2 contract and
validates every part of that contract before installing an interception. It
does not support compiled autograd, tensor extensions, CPU offload, legacy TE
experts, or custom FSDP parameter/group implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.nn as nn

_SUPPORTED_TORCH_MAJOR_MINOR = (2, 10)


class FSDP2ExpertGraphStorageError(RuntimeError):
    """Raised when stable expert graph storage cannot be retained safely."""


@dataclass(frozen=True)
class _FSDP2Contract:
    fsdp_module_type: type
    fsdp_param_group_type: type
    fsdp_param_type: type
    sharded_state_type: type
    compiled_autograd_enabled: Callable[[], bool]
    state_for_module: Callable[[nn.Module], Any]


@dataclass(frozen=True)
class _StorageSnapshot:
    fsdp_param: Any
    parameter_name: str
    unsharded_param: nn.Parameter
    all_gather_output: torch.Tensor
    unsharded_data_ptr: int
    all_gather_data_ptr: int
    storage_data_ptr: int
    storage_nbytes: int


@dataclass(frozen=True)
class _MethodInterception:
    fsdp_param: Any
    original_method: Callable[[], None]
    replacement_method: Callable[[], None]


def _load_fsdp2_contract() -> _FSDP2Contract:
    """Load the private eager-FSDP2 types used by the pinned PyTorch build."""
    version_parts = torch.__version__.split("+", 1)[0].split(".")
    try:
        major_minor = (int(version_parts[0]), int(version_parts[1]))
    except (IndexError, ValueError) as error:
        raise FSDP2ExpertGraphStorageError(f"Cannot parse the PyTorch version {torch.__version__!r}.") from error
    if major_minor != _SUPPORTED_TORCH_MAJOR_MINOR:
        git_version = getattr(torch.version, "git_version", None)
        raise FSDP2ExpertGraphStorageError(
            "Partial expert CUDA graphs with ep_shard depend on the pinned PyTorch 2.10 eager-FSDP2 contract; "
            f"got torch {torch.__version__} (git {git_version or 'unknown'})."
        )

    try:
        from torch.distributed.fsdp import fully_shard
        from torch.distributed.fsdp._fully_shard import FSDPModule
        from torch.distributed.fsdp._fully_shard._fsdp_common import compiled_autograd_enabled
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except (ImportError, AttributeError) as error:
        raise FSDP2ExpertGraphStorageError(
            "Partial expert CUDA graphs with ep_shard require the pinned eager FSDP2 private API "
            "(FSDPModule, FSDPParamGroup, FSDPParam, and ShardedState)."
        ) from error

    state_for_module = getattr(fully_shard, "state", None)
    if not callable(state_for_module):
        raise FSDP2ExpertGraphStorageError(
            "The pinned FSDP2 contract changed: torch.distributed.fsdp.fully_shard.state is unavailable."
        )
    if not callable(getattr(FSDPParam, "free_unsharded_param", None)) or not callable(
        getattr(FSDPParamGroup, "_to_sharded", None)
    ):
        raise FSDP2ExpertGraphStorageError(
            "The pinned PyTorch 2.10 FSDP2 contract changed: storage-freeing or logical-reshard methods are missing."
        )
    return _FSDP2Contract(
        fsdp_module_type=FSDPModule,
        fsdp_param_group_type=FSDPParamGroup,
        fsdp_param_type=FSDPParam,
        sharded_state_type=ShardedState,
        compiled_autograd_enabled=compiled_autograd_enabled,
        state_for_module=state_for_module,
    )


class FSDP2ExpertGraphStorage:
    """Keep FSDP2 all-gather buffers at stable addresses for TE expert graphs.

    The intended ordering is::

        with FSDP2ExpertGraphStorage(experts) as storage:
            storage.prepare_before_capture()  # eagerly unshards and pins storage
            capture_expert_graphs()
            storage.finish_capture()          # exposes the sharded parameters
            train_with_graphs()
            destroy_expert_graphs()
        # __exit__ synchronizes, restores FSDP2 freeing, and releases storage

    ``prepare_before_capture()`` is also safe when called from an FSDP
    pre-forward window where the parameters are already unsharded. The handle
    never changes ``reshard_after_backward``. Calling ``reset()`` while FSDP is
    in a forward/backward transition fails closed; callers must first finish
    the step and destroy every CUDA graph that references the retained device
    addresses.

    Args:
        experts: A stock FSDP2-wrapped ``GroupedExpertsTeOps`` instance.
    """

    def __init__(self, experts: nn.Module) -> None:
        self._experts = experts
        self._contract = _load_fsdp2_contract()
        self._param_group = self._validate_wrapped_experts(experts)
        self._snapshots: tuple[_StorageSnapshot, ...] = ()
        self._interceptions: tuple[_MethodInterception, ...] = ()
        self._phase = "idle"

    @property
    def retained_bytes(self) -> int:
        """Number of resident all-gather bytes held by this handle."""
        return sum(snapshot.storage_nbytes for snapshot in self._snapshots)

    @property
    def is_active(self) -> bool:
        """Whether storage retention is installed for this expert module."""
        return self._phase in ("prepared", "active")

    def __enter__(self) -> FSDP2ExpertGraphStorage:
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.reset()

    def prepare_before_capture(self) -> None:
        """Unshard experts and install stable-storage retention before capture."""
        if self._phase != "idle":
            raise FSDP2ExpertGraphStorageError(
                f"prepare_before_capture() requires an idle handle, current phase is {self._phase!r}."
            )
        if self._contract.compiled_autograd_enabled():
            raise FSDP2ExpertGraphStorageError(
                "FSDP2 expert graph storage retention only supports eager autograd; compiled autograd "
                "recreates all_gather_outputs on every all-gather."
            )

        self._validate_param_group_contract()
        try:
            self._experts.unshard(async_op=False)
            self._validate_unsharded_group()
            self._snapshots = tuple(self._snapshot_param(fsdp_param) for fsdp_param in self._param_group.fsdp_params)
            for snapshot in self._snapshots:
                self._interceptions += (self._intercept_free(snapshot.fsdp_param),)
        except Exception:
            self._rollback_prepare()
            raise

        self._phase = "prepared"
        try:
            self.validate_stable()
        except Exception:
            self._rollback_prepare()
            raise

    def finish_capture(self) -> None:
        """Logically reshard experts while leaving captured storage allocated."""
        if self._phase != "prepared":
            raise FSDP2ExpertGraphStorageError(
                f"finish_capture() requires a prepared handle, current phase is {self._phase!r}."
            )
        self.validate_stable()
        # FSDPParamGroup.reshard() intentionally obeys reshard_after_forward and
        # may no-op inside a forward hook. The pinned private _to_sharded()
        # transition always performs the requested logical reshard and is
        # idempotent when the subsequent FSDP post-forward hook runs.
        self._param_group._to_sharded()
        self._validate_logically_sharded()
        self._phase = "active"
        self.validate_stable()

    def validate_stable(self) -> None:
        """Validate retained object identity, aliases, allocation, and address."""
        if self._phase == "idle" and not self._snapshots:
            raise FSDP2ExpertGraphStorageError("No retained FSDP2 expert graph storage is active.")
        for snapshot in self._snapshots:
            fsdp_param = snapshot.fsdp_param
            if getattr(fsdp_param, "_unsharded_param", None) is not snapshot.unsharded_param:
                self._raise_changed(snapshot, "_unsharded_param object identity changed")
            outputs = getattr(fsdp_param, "all_gather_outputs", None)
            if not isinstance(outputs, list) or len(outputs) != 1 or outputs[0] is not snapshot.all_gather_output:
                self._raise_changed(snapshot, "all_gather_outputs object identity changed")
            if snapshot.unsharded_param.data_ptr() != snapshot.unsharded_data_ptr:
                self._raise_changed(snapshot, "_unsharded_param data_ptr changed")
            if snapshot.all_gather_output.data_ptr() != snapshot.all_gather_data_ptr:
                self._raise_changed(snapshot, "all-gather output data_ptr changed")
            storage = snapshot.all_gather_output.untyped_storage()
            if storage.data_ptr() != snapshot.storage_data_ptr:
                self._raise_changed(snapshot, "all-gather storage data_ptr changed")
            if storage.size() != snapshot.storage_nbytes:
                self._raise_changed(
                    snapshot,
                    f"all-gather storage size changed from {snapshot.storage_nbytes} to {storage.size()} bytes",
                )
            if snapshot.unsharded_param.untyped_storage().data_ptr() != storage.data_ptr():
                self._raise_changed(snapshot, "_unsharded_param no longer aliases its all-gather output")

    def reset(self) -> None:
        """Restore normal FSDP2 freeing and release retained expert storage.

        Every CUDA graph that captured these parameter addresses must be
        destroyed before this method is called. The method synchronizes each
        retained CUDA device, logically reshards the module, restores the
        original per-instance methods, and only then invokes normal FSDP2
        storage freeing.
        """
        if self._phase == "idle":
            return
        training_state = getattr(self._param_group, "_training_state", None)
        if getattr(training_state, "name", None) != "IDLE":
            raise FSDP2ExpertGraphStorageError(
                "reset() requires FSDP2 to be idle after the complete forward/backward step; "
                f"current training state is {training_state!r}. Destroy expert graphs, finish the step, then reset."
            )
        if getattr(self._param_group, "_all_gather_result", None) is not None:
            raise FSDP2ExpertGraphStorageError(
                "reset() cannot release expert graph storage while an FSDP2 all-gather is pending."
            )

        self._synchronize_cuda_devices()
        self._param_group._to_sharded()
        self._validate_logically_sharded()
        self._restore_interceptions()
        for interception in self._interceptions:
            interception.original_method()
        for snapshot in self._snapshots:
            if snapshot.all_gather_output.untyped_storage().size() != 0:
                raise FSDP2ExpertGraphStorageError(
                    f"FSDP2 did not free retained storage for {snapshot.parameter_name!r} during reset."
                )
        self._snapshots = ()
        self._interceptions = ()
        self._phase = "idle"

    def _validate_wrapped_experts(self, experts: nn.Module) -> Any:
        from nemo_automodel.components.moe.experts import GroupedExpertsTeOps

        if not isinstance(experts, GroupedExpertsTeOps):
            raise FSDP2ExpertGraphStorageError(
                "Stable expert graph storage is restricted to experts='te_ops' GroupedExpertsTeOps; "
                f"got {type(experts).__name__}. Legacy experts='te' is intentionally unsupported."
            )
        if not isinstance(experts, self._contract.fsdp_module_type):
            raise FSDP2ExpertGraphStorageError(
                "GroupedExpertsTeOps must be wrapped directly by stock FSDP2 before retaining graph storage."
            )
        try:
            fsdp_state = self._contract.state_for_module(experts)
        except Exception as error:
            raise FSDP2ExpertGraphStorageError(
                "Could not resolve the stock FSDP2 state for GroupedExpertsTeOps."
            ) from error
        param_group = getattr(fsdp_state, "_fsdp_param_group", None)
        if type(param_group) is not self._contract.fsdp_param_group_type:
            raise FSDP2ExpertGraphStorageError(
                "GroupedExpertsTeOps must own one stock FSDPParamGroup; "
                f"got {type(param_group).__name__ if param_group is not None else None}."
            )
        return param_group

    def _validate_param_group_contract(self) -> None:
        group = self._param_group
        required_group_attributes = (
            "fsdp_params",
            "_sharded_state",
            "_training_state",
            "_all_gather_result",
            "_to_sharded",
        )
        missing_group_attributes = [name for name in required_group_attributes if not hasattr(group, name)]
        if missing_group_attributes:
            raise FSDP2ExpertGraphStorageError(
                "The pinned FSDP2 parameter-group contract changed; missing "
                + ", ".join(missing_group_attributes)
                + "."
            )
        if not group.fsdp_params:
            raise FSDP2ExpertGraphStorageError("GroupedExpertsTeOps has no FSDP2-managed parameters.")

        named_parameters = dict(self._experts.named_parameters())
        expected_parameter_ids = {id(parameter) for parameter in named_parameters.values()}
        sharded = self._contract.sharded_state_type.SHARDED
        unsharded = self._contract.sharded_state_type.UNSHARDED
        if group._sharded_state not in (sharded, unsharded):
            raise FSDP2ExpertGraphStorageError(
                "GroupedExpertsTeOps graph storage only supports fully SHARDED or UNSHARDED FSDP2 states; "
                f"got {group._sharded_state!r}."
            )
        managed_parameter_ids: set[int] = set()
        for fsdp_param in group.fsdp_params:
            if type(fsdp_param) is not self._contract.fsdp_param_type:
                raise FSDP2ExpertGraphStorageError(
                    "Only stock FSDPParam instances are supported for expert graph storage; "
                    f"got {type(fsdp_param).__name__}."
                )
            required_param_attributes = (
                "_module_info",
                "sharded_param",
                "sharded_state",
                "all_gather_outputs",
                "_unsharded_inner_tensors",
                "offload_to_cpu",
                "free_unsharded_param",
            )
            missing_param_attributes = [name for name in required_param_attributes if not hasattr(fsdp_param, name)]
            if missing_param_attributes:
                raise FSDP2ExpertGraphStorageError(
                    "The pinned FSDP2 parameter contract changed; missing " + ", ".join(missing_param_attributes) + "."
                )
            if "free_unsharded_param" in vars(fsdp_param):
                raise FSDP2ExpertGraphStorageError(
                    "FSDPParam.free_unsharded_param already has an instance override; refusing to stack interceptions."
                )
            if fsdp_param.offload_to_cpu:
                raise FSDP2ExpertGraphStorageError(
                    "CPU-offloaded FSDP2 expert parameters are unsupported by CUDA graphs."
                )
            parameter_name = fsdp_param._module_info.param_name
            if parameter_name not in ("_stacked_weight", "_stacked_bias"):
                raise FSDP2ExpertGraphStorageError(
                    "GroupedExpertsTeOps graph storage may only retain stacked owners, "
                    f"but FSDP2 manages parameter {parameter_name!r}."
                )
            current_param = (
                fsdp_param.sharded_param
                if group._sharded_state is sharded
                else getattr(fsdp_param, "_unsharded_param", None)
            )
            managed_parameter_ids.add(id(current_param))

        if managed_parameter_ids != expected_parameter_ids:
            raise FSDP2ExpertGraphStorageError(
                "The GroupedExpertsTeOps FSDPParamGroup does not exactly match the module's stacked parameters."
            )

    def _validate_unsharded_group(self) -> None:
        unsharded = self._contract.sharded_state_type.UNSHARDED
        if self._param_group._sharded_state is not unsharded:
            raise FSDP2ExpertGraphStorageError(
                "FSDP2 unshard did not transition GroupedExpertsTeOps to the expected UNSHARDED state."
            )
        if getattr(self._param_group, "_all_gather_result", None) is not None:
            raise FSDP2ExpertGraphStorageError("FSDP2 unshard returned before its expert all-gather completed.")
        for fsdp_param in self._param_group.fsdp_params:
            if fsdp_param.sharded_state is not unsharded:
                raise FSDP2ExpertGraphStorageError("An expert FSDPParam did not reach the UNSHARDED state.")
            if not hasattr(fsdp_param, "_unsharded_param"):
                raise FSDP2ExpertGraphStorageError(
                    "The pinned FSDP2 contract changed: _unsharded_param was not preserved after all-gather."
                )
            if len(fsdp_param.all_gather_outputs) != 1:
                raise FSDP2ExpertGraphStorageError(
                    "FSDP2 tensor extensions with multiple all-gather outputs are unsupported for TE expert graphs."
                )
            if fsdp_param._unsharded_inner_tensors:
                raise FSDP2ExpertGraphStorageError(
                    "FSDP2 post-all-gather tensor extensions are unsupported for TE expert graphs."
                )

    def _snapshot_param(self, fsdp_param: Any) -> _StorageSnapshot:
        unsharded_param = fsdp_param._unsharded_param
        all_gather_output = fsdp_param.all_gather_outputs[0]
        if type(unsharded_param) is not nn.Parameter:
            raise FSDP2ExpertGraphStorageError(
                "GroupedExpertsTeOps requires a plain nn.Parameter after FSDP2 all-gather; "
                f"got {type(unsharded_param).__name__}."
            )
        output_storage = all_gather_output.untyped_storage()
        parameter_storage = unsharded_param.untyped_storage()
        expected_nbytes = all_gather_output.numel() * all_gather_output.element_size()
        if output_storage.size() != expected_nbytes:
            raise FSDP2ExpertGraphStorageError(
                f"FSDP2 all-gather output for {fsdp_param._module_info.param_name!r} has "
                f"{output_storage.size()} allocated bytes, expected {expected_nbytes}."
            )
        if output_storage.data_ptr() == 0 or all_gather_output.data_ptr() == 0:
            raise FSDP2ExpertGraphStorageError("FSDP2 expert all-gather output is not allocated.")
        if parameter_storage.data_ptr() != output_storage.data_ptr():
            raise FSDP2ExpertGraphStorageError(
                "The pinned eager FSDP2 contract changed: _unsharded_param no longer aliases all_gather_outputs[0]."
            )
        if unsharded_param.data_ptr() != all_gather_output.data_ptr():
            raise FSDP2ExpertGraphStorageError(
                "The pinned eager FSDP2 contract changed: expert unsharded/all-gather tensor offsets differ."
            )
        return _StorageSnapshot(
            fsdp_param=fsdp_param,
            parameter_name=fsdp_param._module_info.param_name,
            unsharded_param=unsharded_param,
            all_gather_output=all_gather_output,
            unsharded_data_ptr=unsharded_param.data_ptr(),
            all_gather_data_ptr=all_gather_output.data_ptr(),
            storage_data_ptr=output_storage.data_ptr(),
            storage_nbytes=output_storage.size(),
        )

    def _intercept_free(self, fsdp_param: Any) -> _MethodInterception:
        original_method = fsdp_param.free_unsharded_param

        def retain_storage(intercepted_param: Any) -> None:
            if intercepted_param is not fsdp_param:
                raise FSDP2ExpertGraphStorageError("FSDP2 expert storage interception was rebound unexpectedly.")

        replacement_method = MethodType(retain_storage, fsdp_param)
        fsdp_param.free_unsharded_param = replacement_method
        return _MethodInterception(
            fsdp_param=fsdp_param,
            original_method=original_method,
            replacement_method=replacement_method,
        )

    def _validate_logically_sharded(self) -> None:
        sharded = self._contract.sharded_state_type.SHARDED
        if self._param_group._sharded_state is not sharded:
            raise FSDP2ExpertGraphStorageError("GroupedExpertsTeOps did not return to the FSDP2 SHARDED state.")
        for fsdp_param in self._param_group.fsdp_params:
            if fsdp_param.sharded_state is not sharded:
                raise FSDP2ExpertGraphStorageError("An expert FSDPParam did not return to the SHARDED state.")
            module_info = fsdp_param._module_info
            if module_info.module._parameters.get(module_info.param_name) is not fsdp_param.sharded_param:
                raise FSDP2ExpertGraphStorageError(
                    f"Logical reshard did not expose the sharded parameter {module_info.param_name!r}."
                )

    def _rollback_prepare(self) -> None:
        self._restore_interceptions()
        try:
            self._param_group._to_sharded()
        finally:
            self._snapshots = ()
            self._interceptions = ()
            self._phase = "idle"

    def _restore_interceptions(self) -> None:
        for interception in self._interceptions:
            current_method = vars(interception.fsdp_param).get("free_unsharded_param")
            if current_method is not interception.replacement_method:
                raise FSDP2ExpertGraphStorageError(
                    "FSDPParam.free_unsharded_param changed while expert graph storage retention was active."
                )
        for interception in self._interceptions:
            delattr(interception.fsdp_param, "free_unsharded_param")

    def _synchronize_cuda_devices(self) -> None:
        devices = {snapshot.all_gather_output.device for snapshot in self._snapshots}
        for device in devices:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    @staticmethod
    def _raise_changed(snapshot: _StorageSnapshot, reason: str) -> None:
        raise FSDP2ExpertGraphStorageError(
            f"Retained FSDP2 expert storage for {snapshot.parameter_name!r} is no longer graph-safe: {reason}."
        )


__all__ = ["FSDP2ExpertGraphStorage", "FSDP2ExpertGraphStorageError"]

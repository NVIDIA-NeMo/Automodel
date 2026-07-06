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

"""Scoped partial CUDA graphs for fixed-shape attention and MoE execution.

Attention, fixed-capacity post-router MoE execution, the parameterless router
core, and HybridEP metadata preprocessing may be captured.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from nemo_automodel.shared.import_utils import safe_import_from, safe_import_te

logger = logging.getLogger(__name__)

_Canonicalizer = Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]


@dataclass(frozen=True)
class _CudaMemoryState:
    """CUDA allocator usage split between normal and graph-private pools."""

    normal_allocated: int
    normal_reserved: int
    private_allocated: int
    private_reserved: int


def _cuda_memory_state() -> _CudaMemoryState | None:
    """Read allocator state without making CUDA a module-import requirement."""
    if not torch.cuda.is_available():
        return None

    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    private_allocated = 0
    private_reserved = 0
    for segment in torch.cuda.memory_snapshot():
        pool_id = segment.get("segment_pool_id")
        if pool_id is None or pool_id == (0, 0) or pool_id == [0, 0]:
            continue
        private_allocated += int(segment.get("allocated_size", 0))
        private_reserved += int(segment.get("total_size", 0))
    return _CudaMemoryState(
        normal_allocated=max(0, allocated - private_allocated),
        normal_reserved=max(0, reserved - private_reserved),
        private_allocated=private_allocated,
        private_reserved=private_reserved,
    )


def _format_memory_state(state: _CudaMemoryState | None) -> str:
    """Format allocator bytes as a stable one-line diagnostic."""
    if state is None:
        return "unavailable"
    mib = 1024 * 1024
    return (
        f"normal_allocated={state.normal_allocated / mib:.2f}MiB "
        f"normal_reserved={state.normal_reserved / mib:.2f}MiB "
        f"private_allocated={state.private_allocated / mib:.2f}MiB "
        f"private_reserved={state.private_reserved / mib:.2f}MiB"
    )


def _tensor_bytes(tensors: Sequence[torch.Tensor]) -> int:
    """Return storage bytes for a tensor surface, counting aliases once."""
    seen: set[tuple[torch.device, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storage_key = (tensor.device, storage.data_ptr())
        if storage_key in seen:
            continue
        seen.add(storage_key)
        total += storage.nbytes()
    return total


def _get_make_graphed_callables() -> Callable[..., Any]:
    """Load Transformer Engine's graph helper only when the feature is enabled."""
    has_te, _transformer_engine = safe_import_te()
    if not has_te:
        raise RuntimeError("Partial CUDA graphs require a working Transformer Engine PyTorch installation")
    has_graph_helper, make_graphed_callables = safe_import_from(
        "transformer_engine.pytorch.graph",
        "make_graphed_callables",
    )
    if not has_graph_helper:
        raise RuntimeError("Transformer Engine does not provide make_graphed_callables()")

    return make_graphed_callables


def _tensor_metadata(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return metadata that must stay invariant across graph replays.

    Args:
        tensor: Tensor of arbitrary shape whose type, layout, shape, strides,
            dtype, device, and autograd requirement define the replay contract.

    Returns:
        Tuple containing the tensor subclass and replay-critical metadata. The
        returned value does not retain or alias ``tensor`` storage.
    """
    return (
        type(tensor),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tuple(tensor.stride()),
        tensor.requires_grad,
    )


def _named_buffer_storage(target: nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Return buffer identities and storage properties captured by a graph."""
    signatures = []
    for name, buffer in target.named_buffers():
        try:
            data_ptr = buffer.data_ptr()
        except (RuntimeError, TypeError) as error:
            raise RuntimeError(f"CUDA graph buffer {name!r} has no stable local storage") from error
        if buffer.numel() > 0 and data_ptr == 0:
            raise RuntimeError(f"CUDA graph buffer {name!r} has no stable local storage")
        signatures.append((name, id(buffer), data_ptr, _tensor_metadata(buffer)))
    return tuple(signatures)


def _require_local_parameter_storage(name: str, parameter: nn.Parameter) -> None:
    """Reject sharded or otherwise unmaterialized parameters before graph capture."""
    try:
        data_ptr = parameter.data_ptr()
    except (RuntimeError, TypeError) as error:
        raise RuntimeError(
            "Parameterized CUDA graph capture requires materialized parameters with stable local storage; "
            f"{name!r} is not materialized. Nested FSDP ownership is not supported."
        ) from error
    if parameter.numel() > 0 and data_ptr == 0:
        raise RuntimeError(
            "Whole-attention CUDA graph capture requires materialized parameters with stable local storage; "
            f"{name!r} is not materialized. Nested FSDP attention ownership is not supported."
        )


def _alias_pattern(tensors: Sequence[torch.Tensor]) -> tuple[int, ...]:
    """Encode repeated tensor objects without depending on their absolute identities."""
    seen: dict[int, int] = {}
    pattern = []
    for tensor in tensors:
        tensor_id = id(tensor)
        if tensor_id not in seen:
            seen[tensor_id] = len(seen)
        pattern.append(seen[tensor_id])
    return tuple(pattern)


def _is_transformer_engine_pybind_enum(value: Any) -> bool:
    """Return whether ``value`` is an immutable enum exported by TE's extension."""
    value_type = type(value)
    members = getattr(value_type, "__members__", None)
    name = getattr(value, "name", None)
    return (
        value_type.__module__ == "transformer_engine_torch"
        and isinstance(members, dict)
        and isinstance(name, str)
        and name in members
    )


def _same_control_value(expected: Any, actual: Any) -> bool:
    """Compare non-tensor graph controls without invoking tensor-like equality."""
    if type(expected) is not type(actual):
        return False
    if expected is None or isinstance(expected, (bool, int, float, str, bytes, enum.Enum, torch.dtype, torch.device)):
        return bool(expected == actual)
    if isinstance(expected, (tuple, list)):
        return len(expected) == len(actual) and all(
            _same_control_value(expected_item, actual_item) for expected_item, actual_item in zip(expected, actual)
        )
    if isinstance(expected, dict):
        return expected.keys() == actual.keys() and all(
            _same_control_value(expected[key], actual[key]) for key in expected
        )
    if _is_transformer_engine_pybind_enum(expected) and _is_transformer_engine_pybind_enum(actual):
        # pybind11 materializes a fresh Python wrapper when a C++ function
        # returns an enum. TE therefore produces a value-equal but non-identical
        # fused-attention backend object on every DPA invocation.
        try:
            return int(expected) == int(actual)
        except (TypeError, ValueError):
            return False
    return expected is actual


def _describe_control_value(value: Any) -> str:
    """Return a bounded diagnostic for one non-tensor graph control."""
    type_name = f"{type(value).__module__}.{type(value).__qualname__}"
    try:
        value_repr = repr(value)
    except Exception:
        value_repr = "<repr failed>"
    if len(value_repr) > 160:
        value_repr = value_repr[:157] + "..."
    return f"{type_name}({value_repr})"


@dataclass(frozen=True)
class _CapturedCall:
    """Detached sample inputs and the invariants required for a safe replay."""

    tree_spec: Any
    template_leaves: tuple[Any, ...]
    tensor_positions: tuple[int, ...]
    tensor_input_indices: tuple[int, ...]
    sample_tensors: tuple[torch.Tensor, ...]
    tensor_metadata: tuple[tuple[Any, ...], ...]

    @classmethod
    def from_call(cls, args: tuple[Any, ...], kwargs: dict[str, Any]) -> _CapturedCall:
        leaves, tree_spec = tree_flatten((args, kwargs))
        tensor_positions = tuple(index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        tensors = tuple(leaves[index] for index in tensor_positions)

        input_index_by_identity: dict[int, int] = {}
        tensor_input_indices = []
        unique_tensors = []
        for tensor in tensors:
            tensor_id = id(tensor)
            input_index = input_index_by_identity.get(tensor_id)
            if input_index is None:
                input_index = len(unique_tensors)
                input_index_by_identity[tensor_id] = input_index
                unique_tensors.append(tensor)
            tensor_input_indices.append(input_index)

        sample_tensors = []
        for tensor in unique_tensors:
            clone = tensor.detach().clone(memory_format=torch.preserve_format)
            clone.requires_grad_(tensor.requires_grad)
            sample_tensors.append(clone)

        template_leaves = list(leaves)
        for index in tensor_positions:
            template_leaves[index] = None

        return cls(
            tree_spec=tree_spec,
            template_leaves=tuple(template_leaves),
            tensor_positions=tensor_positions,
            tensor_input_indices=tuple(tensor_input_indices),
            sample_tensors=tuple(sample_tensors),
            tensor_metadata=tuple(_tensor_metadata(tensor) for tensor in unique_tensors),
        )

    def rebuild(self, tensors: Sequence[torch.Tensor]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Reconstruct the target call and its aliases from unique tensor inputs."""
        if len(tensors) != len(self.sample_tensors):
            raise RuntimeError(f"Expected {len(self.sample_tensors)} unique tensor inputs, got {len(tensors)}")
        leaves = list(self.template_leaves)
        for position, input_index in zip(self.tensor_positions, self.tensor_input_indices):
            leaves[position] = tensors[input_index]
        args, kwargs = tree_unflatten(leaves, self.tree_spec)
        return args, kwargs

    def validate(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[bool, str, tuple[torch.Tensor, ...]]:
        """Validate a replay call and return its flattened tensor inputs."""
        leaves, tree_spec = tree_flatten((args, kwargs))
        if tree_spec != self.tree_spec:
            return False, "input pytree changed", ()

        tensor_positions = tuple(index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor))
        if tensor_positions != self.tensor_positions:
            return False, "tensor/control positions changed", ()

        tensors = tuple(leaves[index] for index in tensor_positions)
        if _alias_pattern(tensors) != self.tensor_input_indices:
            return False, "tensor aliasing changed", ()

        unique_tensors = []
        for tensor, input_index in zip(tensors, self.tensor_input_indices):
            if input_index == len(unique_tensors):
                unique_tensors.append(tensor)
        if len(unique_tensors) != len(self.sample_tensors):
            return False, "tensor aliasing changed", ()

        for input_index, (expected, tensor) in enumerate(zip(self.tensor_metadata, unique_tensors)):
            actual = _tensor_metadata(tensor)
            if actual != expected:
                return (
                    False,
                    f"tensor metadata changed at input {input_index}: expected {expected}, got {actual}",
                    (),
                )

        tensor_position_set = set(tensor_positions)
        for index, (expected, actual) in enumerate(zip(self.template_leaves, leaves)):
            if index in tensor_position_set:
                continue
            if not _same_control_value(expected, actual):
                return (
                    False,
                    f"non-tensor control changed at leaf {index}: "
                    f"expected {_describe_control_value(expected)}, got {_describe_control_value(actual)}",
                    (),
                )

        return True, "", tuple(unique_tensors)


class _TensorOnlyCallAdapter(nn.Module):
    """Expose a mixed tensor/control module call as a tensor-only module call."""

    def __init__(self, target: nn.Module, captured_call: _CapturedCall):
        super().__init__()
        self.target = target
        self.captured_call = captured_call

    def forward(self, *tensor_inputs: torch.Tensor) -> Any:
        """Rebuild and execute the captured target call."""
        args, kwargs = self.captured_call.rebuild(tensor_inputs)
        return self.target(*args, **kwargs)


class _ExplicitParameterCallAdapter(nn.Module):
    """Present module parameters as graph inputs instead of captured module state."""

    def __init__(self, target: nn.Module, captured_call: _CapturedCall) -> None:
        super().__init__()
        self.graph_target = target
        self.__dict__["target"] = target
        self.captured_call = captured_call
        self.dynamic_input_count = len(captured_call.sample_tensors)

        named_parameters = tuple(target.named_parameters())
        if not named_parameters:
            raise RuntimeError("Parameterized CUDA graph target has no parameters")
        all_parameter_names = tuple(name for name, _parameter in target.named_parameters(remove_duplicate=False))
        if len(all_parameter_names) != len(named_parameters):
            raise RuntimeError("Parameterized CUDA graphs do not support tied or aliased parameters")
        self.parameter_names = tuple(name for name, _parameter in named_parameters)
        self.parameter_metadata = tuple(_tensor_metadata(parameter) for _name, parameter in named_parameters)
        self.buffer_storage = _named_buffer_storage(target)
        capture_parameters = []
        for name, parameter in named_parameters:
            if not isinstance(parameter, nn.Parameter):
                raise RuntimeError(
                    "Parameterized CUDA graph capture requires materialized nn.Parameter values; "
                    f"{name!r} is {type(parameter).__name__}"
                )
            _require_local_parameter_storage(name, parameter)
            # Always use a fresh leaf, even when the target is not FSDP-owned.
            # Persistent model Parameters may already own AccumulateGrad nodes
            # created on the eager stream; capturing those nodes can make TE's
            # backward graph depend on the wrong stream. TE directly uses this
            # clone as its static input, so this is one staging allocation rather
            # than a clone followed by a second TE-owned parameter copy.
            capture_parameter = parameter.detach().clone(memory_format=torch.preserve_format)
            capture_parameter.requires_grad_(parameter.requires_grad)
            capture_parameters.append(capture_parameter)
        self.capture_parameters = tuple(capture_parameters)
        self.cloned_parameter_bytes = _tensor_bytes(self.capture_parameters)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Hide the registered target parameters from TE module-parameter discovery."""
        del recurse
        return iter(())

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Hide the registered target parameters from generic module utilities."""
        del prefix, recurse, remove_duplicate
        return iter(())

    @property
    def capture_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return dynamic samples followed by graph-owned parameter samples."""
        return self.captured_call.sample_tensors + self.capture_parameters

    def replay_inputs(self, dynamic_inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Append the currently materialized live parameters for graph replay."""
        named_parameters = tuple(self.target.named_parameters())
        parameter_names = tuple(name for name, _parameter in named_parameters)
        if parameter_names != self.parameter_names:
            raise RuntimeError(
                f"CUDA graph parameter names changed: expected {self.parameter_names}, got {parameter_names}"
            )
        for index, (name, parameter) in enumerate(named_parameters):
            if not isinstance(parameter, nn.Parameter):
                raise RuntimeError(
                    f"CUDA graph parameters are not materialized for replay: {name!r} is {type(parameter).__name__}"
                )
            _require_local_parameter_storage(name, parameter)
            actual_metadata = _tensor_metadata(parameter)
            expected_metadata = self.parameter_metadata[index]
            if actual_metadata != expected_metadata:
                raise RuntimeError(
                    f"CUDA graph parameter metadata changed for {name!r}: "
                    f"expected {expected_metadata}, got {actual_metadata}"
                )
        buffer_storage = _named_buffer_storage(self.target)
        if buffer_storage != self.buffer_storage:
            raise RuntimeError(
                f"CUDA graph buffer storage changed: expected {self.buffer_storage}, got {buffer_storage}"
            )
        return dynamic_inputs + tuple(parameter for _name, parameter in named_parameters)

    def forward(self, *tensor_inputs: torch.Tensor) -> Any:
        """Run the target with explicit parameter values."""
        expected_inputs = self.dynamic_input_count + len(self.parameter_names)
        if len(tensor_inputs) != expected_inputs:
            raise RuntimeError(f"Expected {expected_inputs} explicit graph inputs, got {len(tensor_inputs)}")
        dynamic_inputs = tensor_inputs[: self.dynamic_input_count]
        parameter_inputs = tensor_inputs[self.dynamic_input_count :]
        args, kwargs = self.captured_call.rebuild(dynamic_inputs)
        return torch.func.functional_call(
            self.graph_target,
            (
                dict(zip(self.parameter_names, parameter_inputs)),
                dict(self.graph_target.named_buffers()),
            ),
            args,
            kwargs,
            tie_weights=False,
            strict=True,
        )


class _PartialGraphEntry:
    """One graphable module invocation and its replay safety state."""

    def __init__(
        self,
        *,
        name: str,
        target: nn.Module,
        canonicalizer: _Canonicalizer | None = None,
        pool_group: str | None = None,
        capture_input_variants: int = 1,
        explicit_parameters: bool = False,
        capture_owner: nn.Module | None = None,
        retain_graph_in_backward: bool = False,
    ):
        # Target configuration.
        self.name = name
        self.target = target
        self.canonicalizer = canonicalizer
        self.pool_group = pool_group or name
        if capture_input_variants <= 0:
            raise ValueError("capture_input_variants must be positive")
        self.capture_input_variants = capture_input_variants
        self.explicit_parameters = explicit_parameters
        self.capture_owner = capture_owner
        self.retain_graph_in_backward = retain_graph_in_backward
        self.original_forward = target.forward

        # Recorded input contracts and captured graphs.
        self.captured_call: _CapturedCall | None = None
        self._captured_call_variants: list[_CapturedCall] = []
        self._adapters: tuple[nn.Module, ...] = ()

        # Runtime statistics.
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0

        # Lifecycle state.
        self._record_hook: Any = None
        self._captured_training: bool | None = None
        self._logged_replay = False
        self._logged_fallback = False
        self._capture_owner_unsharded = False
        self._closed = False

    def _canonicalize(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.canonicalizer is None:
            return args, kwargs
        return self.canonicalizer(args, kwargs)

    def start_recording(self) -> None:
        """Record the first eager call through a temporary pre-hook."""
        if self._closed:
            raise RuntimeError(f"Partial CUDA graph entry {self.name!r} is closed")

        def record_call(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            canonical_args, canonical_kwargs = self._canonicalize(args, kwargs)
            if self.captured_call is None:
                self.captured_call = _CapturedCall.from_call(canonical_args, canonical_kwargs)
                return

            captured_calls = self.captured_calls()
            if any(call.validate(canonical_args, canonical_kwargs)[0] for call in captured_calls):
                return
            if len(captured_calls) >= self.capture_input_variants:
                return
            self._captured_call_variants.append(_CapturedCall.from_call(canonical_args, canonical_kwargs))

        self._record_hook = self.target.register_forward_pre_hook(record_call, with_kwargs=True)

    def stop_recording(self) -> None:
        """Remove the temporary eager-call recorder."""
        if self._record_hook is not None:
            self._record_hook.remove()
            self._record_hook = None

    def close(self) -> None:
        """Destroy graph replay state and restore the eager target."""
        if self._closed:
            return

        self.stop_recording()

        # Stop new calls from entering a graph before destroying it. TE attaches
        # ``reset`` to the adapter returned by make_graphed_callables; that must
        # run while every captured tensor/parameter address is still alive.
        self.target.forward = self.original_forward
        adapters = self._adapters
        for adapter in adapters:
            reset = getattr(adapter, "reset", None)
            if self.capture_count and not callable(reset):
                raise RuntimeError(f"Captured partial CUDA graph adapter for {self.name!r} has no reset()")
            if callable(reset):
                reset()
            # TE installs graph closures directly on the adapter instance.
            adapter.__dict__.pop("forward", None)
            adapter.__dict__.pop("backward_dw", None)
            adapter.__dict__.pop("reset", None)
            del reset

        self._adapters = ()
        self.captured_call = None
        self._captured_call_variants.clear()
        self._captured_training = None
        del adapters
        self._closed = True

    def captured_calls(self) -> tuple[_CapturedCall, ...]:
        """Return the distinct iteration-0 input contracts in call order."""
        if self.captured_call is None:
            return ()
        return (self.captured_call, *self._captured_call_variants)

    def build_adapter(self, captured_call: _CapturedCall | None = None) -> nn.Module:
        """Build the tensor-only capture adapter after an eager sample was observed."""
        if captured_call is None:
            captured_call = self.captured_call
        if captured_call is None:
            raise RuntimeError(f"Partial CUDA graph target {self.name!r} was not called during iteration 0")
        if self.explicit_parameters:
            adapter = _ExplicitParameterCallAdapter(self.target, captured_call)
        else:
            adapter = _TensorOnlyCallAdapter(self.target, captured_call)
        adapter.train(self.target.training)
        return adapter

    def capture_inputs(self, adapter: nn.Module, captured_call: _CapturedCall) -> tuple[torch.Tensor, ...]:
        """Return the sample surface passed to Transformer Engine."""
        if isinstance(adapter, _ExplicitParameterCallAdapter):
            return adapter.capture_inputs
        return captured_call.sample_tensors

    def prepare_for_capture(self) -> None:
        """Materialize FSDP2 parameters before constructing explicit graph inputs."""
        if self.capture_owner is not None:
            self.capture_owner.unshard(async_op=False)
            self._capture_owner_unsharded = True
            for name, parameter in self.target.named_parameters():
                if not isinstance(parameter, nn.Parameter):
                    raise RuntimeError(
                        "Parameterized CUDA graph capture requires the selected FSDP owner to materialize "
                        f"every target parameter; {name!r} remains {type(parameter).__name__}. "
                        "Nested FSDP ownership is not supported."
                    )
                _require_local_parameter_storage(name, parameter)

    def finish_capture(self) -> None:
        """Return an FSDP2 owner to its normal sharded state after capture."""
        if self.capture_owner is not None and self._capture_owner_unsharded:
            try:
                self.capture_owner.reshard()
            finally:
                self._capture_owner_unsharded = False

    def install(
        self,
        graphed_adapter: nn.Module | Sequence[nn.Module],
        *,
        allow_eager_fallback: bool = True,
    ) -> None:
        """Install validated graph replay around the original target forward."""
        if self._closed:
            raise RuntimeError(f"Partial CUDA graph entry {self.name!r} is closed")
        if isinstance(graphed_adapter, nn.Module):
            adapters = (graphed_adapter,)
        else:
            adapters = tuple(graphed_adapter)
        captured_calls = self.captured_calls()
        if not adapters or len(adapters) != len(captured_calls):
            raise RuntimeError(
                f"Partial CUDA graph target {self.name!r} has {len(captured_calls)} input contracts "
                f"but {len(adapters)} graphed adapters"
            )
        self._adapters = adapters
        self.capture_count = len(adapters)
        self._captured_training = self.target.training

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            try:
                canonical_args, canonical_kwargs = self._canonicalize(args, kwargs)
                validation_results = tuple(
                    captured_call.validate(canonical_args, canonical_kwargs) for captured_call in captured_calls
                )
                match = next(
                    (
                        (index, tensors)
                        for index, (variant_valid, _variant_reason, tensors) in enumerate(validation_results)
                        if variant_valid
                    ),
                    None,
                )
                if match is None:
                    valid = False
                    tensors = ()
                    reason = "; ".join(
                        f"variant {index}: {variant_reason}"
                        for index, (_variant_valid, variant_reason, _variant_tensors) in enumerate(validation_results)
                    )
                else:
                    valid = True
                    variant_index, tensors = match
            except Exception as error:  # a changed dynamic call must stay correct via eager fallback
                valid, reason, tensors = False, f"call canonicalization failed: {error}", ()

            if valid and self.target.training != self._captured_training:
                valid, reason = False, "training mode changed"
            if valid and isinstance(adapters[variant_index], _ExplicitParameterCallAdapter):
                try:
                    tensors = adapters[variant_index].replay_inputs(tensors)
                except Exception as error:
                    valid, reason = False, f"explicit parameter inputs unavailable: {error}"
            if valid:
                self.replay_count += 1
                if not self._logged_replay:
                    logger.info("Partial CUDA graph replay active for %s", self.name)
                    self._logged_replay = True
                return adapters[variant_index](*tensors)

            if not allow_eager_fallback:
                raise RuntimeError(
                    f"Partial CUDA graph pooled entry {self.name!r} cannot fall back independently: {reason}"
                )

            self.fallback_count += 1
            if not self._logged_fallback:
                logger.warning("Partial CUDA graph eager fallback for %s: %s", self.name, reason)
                self._logged_fallback = True
            return self.original_forward(*args, **kwargs)

        self.target.forward = dispatch


def _canonicalize_bf16_fused_attention(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Remove ignored FP8 metadata while requiring the DPA compute itself to be BF16."""
    if kwargs.get("fp8", False):
        raise RuntimeError("partial attention CUDA graphs require BF16 dot-product attention")
    canonical_kwargs = dict(kwargs)
    if "fp8_meta" in canonical_kwargs:
        canonical_kwargs["fp8_meta"] = None
    if "quantizers" in canonical_kwargs:
        canonical_kwargs["quantizers"] = None
    return args, canonical_kwargs


def _unwrap_checkpoint_wrappers(module: nn.Module) -> nn.Module:
    """Return the module beneath any nested PyTorch checkpoint wrappers."""
    while isinstance(getattr(module, "_checkpoint_wrapped_module", None), nn.Module):
        module = module._checkpoint_wrapped_module
    return module


def _model_label(model: nn.Module) -> str:
    """Return a diagnostic model label without using it as a capability gate."""
    outer_config = getattr(model, "config", None)
    inner_model = getattr(model, "model", None)
    inner_config = getattr(inner_model, "config", None)
    for config in (
        outer_config,
        getattr(outer_config, "text_config", None),
        getattr(outer_config, "llm_config", None),
        inner_config,
        getattr(inner_config, "text_config", None),
        getattr(inner_config, "llm_config", None),
    ):
        model_type = getattr(config, "model_type", None)
        if isinstance(model_type, str) and model_type:
            return model_type
    return type(model).__name__


@dataclass(frozen=True)
class _DiscoveredBlock:
    """One structurally discovered transformer or MTP block."""

    name: str
    module: nn.Module
    capture_owner: nn.Module | None
    moe: nn.Module | None
    is_mtp: bool


def _find_moe_module(block: nn.Module) -> nn.Module | None:
    """Find an MoE sublayer by its gate-and-experts capability."""
    candidates: list[nn.Module] = []
    for attribute in ("moe", "mlp", "mixer"):
        candidate = getattr(block, attribute, None)
        if isinstance(candidate, nn.Module):
            candidates.append(candidate)
    candidates.extend(block.modules())

    seen: set[int] = set()
    for candidate in candidates:
        candidate = _unwrap_checkpoint_wrappers(candidate)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(getattr(candidate, "gate", None), nn.Module) and isinstance(
            getattr(candidate, "experts", None), nn.Module
        ):
            return candidate
    return None


def _find_attention(block: _DiscoveredBlock) -> nn.Module | None:
    """Find the whole attention module in a transformer block."""
    attention = getattr(block.module, "self_attn", None)
    return _unwrap_checkpoint_wrappers(attention) if isinstance(attention, nn.Module) else None


def _find_te_dpa(block: _DiscoveredBlock) -> nn.Module | None:
    """Find the parameterless TE fused-attention boundary in a transformer block."""
    attention = _find_attention(block)
    if isinstance(attention, nn.Module):
        dpa = getattr(attention, "attn_module", None)
        target = getattr(dpa, "fused_attention", None)
        if isinstance(target, nn.Module):
            return target

    for module in block.module.modules():
        target = getattr(module, "fused_attention", None)
        if isinstance(target, nn.Module):
            return target
    return None


def _find_moe_router(block: _DiscoveredBlock) -> nn.Module | None:
    """Find the parameterless routing core in an MoE transformer block."""
    gate = getattr(block.moe, "gate", None)
    target = getattr(gate, "routing_core", None)
    return target if isinstance(target, nn.Module) else None


def _find_moe_preprocess(block: _DiscoveredBlock) -> nn.Module | None:
    """Find graphable fused HybridEP metadata preprocessing."""
    experts = getattr(block.moe, "experts", None)
    dispatcher = getattr(experts, "token_dispatcher", None)
    target = getattr(dispatcher, "hybridep_metadata_processor", None)
    if isinstance(target, nn.Module) and getattr(target, "permute_fusion", False):
        return target
    return None


def _find_moe_execution(block: _DiscoveredBlock) -> nn.Module | None:
    """Find the dispatch, TE GroupedLinear, and combine boundary after routing."""
    experts = getattr(block.moe, "experts", None)
    return (
        experts if isinstance(experts, nn.Module) and getattr(experts, "_supports_full_moe_cuda_graph", False) else None
    )


def _require_parameterless_target(module_name: str, block: _DiscoveredBlock, target: nn.Module) -> None:
    """Reject graph boundaries that own parameters managed outside the graph."""
    if any(True for _ in target.parameters()):
        raise RuntimeError(f"The {module_name} CUDA graph boundary in {block.name} must be parameterless")


def _build_whole_attention_entry(
    block: _DiscoveredBlock,
    target: nn.Module,
    _activation_checkpointing: bool,
) -> _PartialGraphEntry:
    """Build one whole-attention graph entry with graph-safe explicit parameters."""
    from torch.distributed.fsdp import FSDPModule

    capture_owner = block.capture_owner if isinstance(block.capture_owner, FSDPModule) else None
    return _PartialGraphEntry(
        name=f"{block.name}.attention",
        target=target,
        pool_group="attn",
        capture_input_variants=1,
        # PyTorch parameters are explicit inputs even without FSDP. Letting TE
        # discover them as static module state can keep AccumulateGrad nodes on
        # the legacy stream and invalidate backward capture.
        explicit_parameters=True,
        capture_owner=capture_owner,
        # TE DPA backward may consume the upstream RoPE graph before the outer
        # callable backward finishes capturing it.
        retain_graph_in_backward=True,
    )


def _build_te_dpa_entry(
    block: _DiscoveredBlock,
    target: nn.Module,
    activation_checkpointing: bool,
) -> _PartialGraphEntry:
    """Build one TE fused-attention graph entry."""
    _require_parameterless_target("te_dpa", block, target)
    return _PartialGraphEntry(
        name=f"{block.name}.fused_attention",
        target=target,
        pool_group="te_dpa",
        canonicalizer=_canonicalize_bf16_fused_attention,
        # Reentrant checkpointing invokes attention once under no-grad and once
        # under grad-enabled recompute. Each contract needs its own TE graph
        # because requires_grad is a replay invariant.
        capture_input_variants=2 if activation_checkpointing else 1,
    )


def _build_moe_router_entry(
    block: _DiscoveredBlock,
    target: nn.Module,
    _activation_checkpointing: bool,
) -> _PartialGraphEntry:
    """Build one parameterless MoE router graph entry."""
    assert block.moe is not None
    gate = block.moe.gate
    if getattr(gate, "router_replay", None) is not None or getattr(gate, "e_score_correction_bias", None) is not None:
        raise RuntimeError("Partial MoE router graphs require routing replay and score-correction bias to be disabled")
    _require_parameterless_target("moe_router", block, target)
    return _PartialGraphEntry(name=f"{block.name}.moe_router", target=target, pool_group="moe_router")


def _build_moe_preprocess_entry(
    block: _DiscoveredBlock,
    target: nn.Module,
    _activation_checkpointing: bool,
) -> _PartialGraphEntry:
    """Build one fused HybridEP metadata-preprocessing graph entry."""
    _require_parameterless_target("moe_preprocess", block, target)
    return _PartialGraphEntry(name=f"{block.name}.moe_preprocess", target=target, pool_group="moe_preprocess")


def _build_moe_execution_entry(
    block: _DiscoveredBlock,
    target: nn.Module,
    _activation_checkpointing: bool,
) -> _PartialGraphEntry:
    """Build one fixed-capacity post-router MoE graph entry."""
    from torch.distributed.fsdp import FSDPModule

    if isinstance(target, FSDPModule):
        raise RuntimeError(
            "Full MoE CUDA graphs do not support nested FSDP expert sharding (ep_shard > 1); "
            "the expert parameters must be owned by the transformer block"
        )
    capture_owner = block.capture_owner if isinstance(block.capture_owner, FSDPModule) else None
    return _PartialGraphEntry(
        name=f"{block.name}.moe",
        target=target,
        pool_group="moe",
        explicit_parameters=True,
        capture_owner=capture_owner,
        retain_graph_in_backward=True,
    )


@dataclass(frozen=True)
class _GraphModuleSpec:
    """Discovery and entry-construction contract for one user-facing graph module."""

    find_target: Callable[[_DiscoveredBlock], nn.Module | None]
    build_entry: Callable[[_DiscoveredBlock, nn.Module, bool], _PartialGraphEntry]


_GRAPH_MODULE_SPECS = {
    "attn": _GraphModuleSpec(_find_attention, _build_whole_attention_entry),
    "te_dpa": _GraphModuleSpec(_find_te_dpa, _build_te_dpa_entry),
    "moe": _GraphModuleSpec(_find_moe_execution, _build_moe_execution_entry),
    "moe_router": _GraphModuleSpec(_find_moe_router, _build_moe_router_entry),
    "moe_preprocess": _GraphModuleSpec(_find_moe_preprocess, _build_moe_preprocess_entry),
}


def _discover_blocks(model: nn.Module) -> list[_DiscoveredBlock]:
    """Discover main-stack and MTP blocks using the shared MoE traversal."""
    from torch.distributed.fsdp import FSDPModule

    from nemo_automodel.components.moe.parallelizer import _iter_transformer_and_mtp_blocks

    label = _model_label(model)
    mtp_layers = getattr(getattr(model, "mtp", None), "layers", None)
    blocks = []
    for parent_layers, layer_id, wrapped_block in _iter_transformer_and_mtp_blocks(model):
        block = _unwrap_checkpoint_wrappers(wrapped_block)
        capture_owner = wrapped_block if isinstance(wrapped_block, FSDPModule) else None
        is_mtp = parent_layers is mtp_layers
        scope = "mtp.layers" if is_mtp else "layers"
        blocks.append(
            _DiscoveredBlock(
                name=f"{label}.{scope}.{layer_id}",
                module=block,
                capture_owner=capture_owner,
                moe=_find_moe_module(block),
                is_mtp=is_mtp,
            )
        )
    return blocks


def _uses_repeated_mtp_layer(model: nn.Module) -> bool:
    """Return whether one physical MTP layer is invoked repeatedly per forward."""
    mtp = getattr(model, "mtp", None)
    for config in (getattr(mtp, "mtp_config", None), getattr(model, "mtp_config", None)):
        if bool(getattr(config, "use_repeated_layer", False)):
            return True
    return False


def _select_graph_targets(
    *,
    module_name: str,
    candidates: list[tuple[_DiscoveredBlock, nn.Module]],
    repeated_mtp_layer: bool,
) -> dict[str, nn.Module]:
    """Select graph targets while rejecting shared/repeated physical call sites."""
    if not candidates:
        raise RuntimeError(
            f"cuda_graph_modules includes {module_name!r}, but no layer exposes a graphable {module_name} boundary"
        )
    if repeated_mtp_layer and any(block.is_mtp for block, _target in candidates):
        raise RuntimeError(
            f"cuda_graph_modules={module_name!r} cannot capture repeated-layer MTP: one physical "
            f"{module_name} target is invoked multiple times before backward and requires graph replicas or ring slots"
        )
    target_ids = [id(target) for _block, target in candidates]
    if len(target_ids) != len(set(target_ids)):
        raise RuntimeError(
            f"cuda_graph_modules={module_name!r} selected a shared physical target more than once; "
            "shared/repeated modules require independent graph replicas"
        )
    return {block.name: target for block, target in candidates}


class PartialCudaGraphManager:
    """Capture selected TE attention and MoE submodules after one eager iteration."""

    def __init__(self, entries: list[_PartialGraphEntry]):
        self.entries = entries
        self._captured = False
        self._closed = False

    @classmethod
    def from_model_parts(
        cls,
        model_parts: list[nn.Module],
        *,
        activation_checkpointing: bool = False,
        pipeline_parallel: bool = False,
    ) -> PartialCudaGraphManager | None:
        """Discover graph targets from an already-built training model."""
        enabled_parts = [
            part for part in model_parts if bool(getattr(getattr(part, "backend", None), "cuda_graph_modules", []))
        ]
        if not enabled_parts:
            return None
        if pipeline_parallel or len(model_parts) != 1:
            raise RuntimeError(
                "Partial CUDA graphs require one non-pipeline model root; pipeline stages and virtual "
                "pipeline chunks can overwrite static graph buffers with in-flight microbatches"
            )
        if activation_checkpointing:
            if any("attn" in part.backend.cuda_graph_modules for part in enabled_parts):
                raise RuntimeError(
                    "Whole-attention CUDA graphs do not support activation checkpointing; "
                    "use te_dpa or disable activation checkpointing"
                )
            if any("moe" in part.backend.cuda_graph_modules for part in enabled_parts):
                raise RuntimeError("Full MoE CUDA graphs do not support activation checkpointing")
            if any(
                "moe_router" in part.backend.cuda_graph_modules or "moe_preprocess" in part.backend.cuda_graph_modules
                for part in enabled_parts
            ):
                raise RuntimeError(
                    "PyTorch activation checkpointing cannot recompute across the partial MoE router/preprocess "
                    "CUDA graph scope; use attention-only graphs or disable router/preprocess graphs"
                )
            logger.info(
                "Partial CUDA graphs are enabled with PyTorch activation checkpointing; "
                "checkpoint recomputation will use the same guarded graph entry points"
            )
        model = model_parts[0]
        backend = model.backend
        cuda_graph_modules = set(backend.cuda_graph_modules)
        blocks = _discover_blocks(model)
        entries: list[_PartialGraphEntry] = []

        targets_by_module: dict[str, dict[str, nn.Module]] = {}
        repeated_mtp_layer = _uses_repeated_mtp_layer(model)
        for module_name, module_spec in _GRAPH_MODULE_SPECS.items():
            if module_name not in cuda_graph_modules:
                continue
            candidates = []
            for block in blocks:
                target = module_spec.find_target(block)
                if target is not None:
                    candidates.append((block, target))
            targets_by_module[module_name] = _select_graph_targets(
                module_name=module_name,
                candidates=candidates,
                repeated_mtp_layer=repeated_mtp_layer,
            )

        for block in blocks:
            for module_name, module_spec in _GRAPH_MODULE_SPECS.items():
                target = targets_by_module.get(module_name, {}).get(block.name)
                if target is not None:
                    entries.append(module_spec.build_entry(block, target, activation_checkpointing))

        if not entries:
            raise RuntimeError("Partial CUDA graphs were enabled but no graph targets were found")

        manager = cls(entries)
        manager.start_recording()
        return manager

    def start_recording(self) -> None:
        """Install first-call recorders on every selected target."""
        if self._closed:
            raise RuntimeError("Partial CUDA graph manager is closed")
        for entry in self.entries:
            entry.start_recording()

    def capture(self) -> None:
        """Batch-capture all observed targets in real forward order."""
        if self._closed:
            raise RuntimeError("Partial CUDA graph manager is closed")
        if self._captured:
            return
        for entry in self.entries:
            entry.stop_recording()

        missing_entries = tuple(entry.name for entry in self.entries if entry.captured_call is None)
        if missing_entries:
            raise RuntimeError(
                "Every partial CUDA graph target must have an iteration-0 sample; "
                f"missing samples for {missing_entries}"
            )

        staged_groups: dict[tuple[str, bool], list[tuple[_PartialGraphEntry, nn.Module, tuple[torch.Tensor, ...]]]] = {}
        variant_entries = []
        for entry in self.entries:
            captured_calls = entry.captured_calls()
            entry_before = _cuda_memory_state()
            staged_adapters = []
            try:
                entry.prepare_for_capture()
                for captured_call in captured_calls:
                    adapter = entry.build_adapter(captured_call)
                    capture_inputs = entry.capture_inputs(adapter, captured_call)
                    staged_adapters.append((adapter, capture_inputs))
            finally:
                entry.finish_capture()

            entry_after = _cuda_memory_state()
            cloned_parameter_bytes = sum(
                adapter.cloned_parameter_bytes
                for adapter, _capture_inputs in staged_adapters
                if isinstance(adapter, _ExplicitParameterCallAdapter)
            )
            logger.info(
                "Partial CUDA graph memory (entry staged): name=%s calls=%d static_input_bytes=%d "
                "cloned_parameter_bytes=%d before=[%s] after=[%s]",
                entry.name,
                len(staged_adapters),
                sum(_tensor_bytes(capture_inputs) for _adapter, capture_inputs in staged_adapters),
                cloned_parameter_bytes,
                _format_memory_state(entry_before),
                _format_memory_state(entry_after),
            )
            if len(staged_adapters) == 1:
                adapter, capture_inputs = staged_adapters[0]
                group_key = (entry.pool_group, entry.retain_graph_in_backward)
                staged_groups.setdefault(group_key, []).append((entry, adapter, capture_inputs))
            else:
                variant_entries.append((entry, staged_adapters))

        uninstalled_adapters: list[nn.Module] = []

        def capture_group(
            staged_group: Sequence[tuple[_PartialGraphEntry, nn.Module, tuple[torch.Tensor, ...]]],
        ) -> tuple[nn.Module, ...]:
            if not staged_group:
                return ()
            group_before = _cuda_memory_state()
            modules = tuple(adapter for _entry, adapter, _capture_inputs in staged_group)
            sample_args = tuple(capture_inputs for _entry, _adapter, capture_inputs in staged_group)
            graph_kwargs: dict[str, Any] = {
                # TE runs callable-local forward/backward warmups, separate from training-loop warmup steps.
                "num_warmup_iters": 3,
                # This disables TE FP8/FP4 quantization, not CUDA graph capture.
                "enabled": tuple(False for _ in modules),
            }
            if staged_group[0][0].retain_graph_in_backward:
                graph_kwargs["retain_graph_in_backward"] = True
            names = tuple(entry.name for entry, _adapter, _capture_inputs in staged_group)
            try:
                result = _get_make_graphed_callables()(
                    modules=modules,
                    sample_args=sample_args,
                    **graph_kwargs,
                )
            except Exception as error:
                raise RuntimeError(f"Partial CUDA graph capture failed for shared-pool entries {names}") from error
            if not isinstance(result, tuple):
                result = (result,)
            # Track returned graphs before validating the result shape so even a
            # malformed TE return is reset by the surrounding transaction.
            uninstalled_adapters.extend(result)
            if len(result) != len(staged_group) or not all(isinstance(adapter, nn.Module) for adapter in result):
                raise RuntimeError(
                    "Transformer Engine must return one nn.Module graph for every partial target; "
                    f"got {result!r} for {len(staged_group)} targets"
                )
            group_after = _cuda_memory_state()
            logger.info(
                "Partial CUDA graph memory (shared capture): entries=%d names=%s before=[%s] after=[%s]",
                len(staged_group),
                names,
                _format_memory_state(group_before),
                _format_memory_state(group_after),
            )
            return result

        captured_groups: list[
            tuple[
                Sequence[tuple[_PartialGraphEntry, nn.Module, tuple[torch.Tensor, ...]]],
                tuple[nn.Module, ...],
            ]
        ] = []
        captured_variants: list[tuple[_PartialGraphEntry, tuple[nn.Module, ...]]] = []
        try:
            # Each scope is captured in one TE invocation. TE captures forwards
            # in model order and backwards in reverse order, so every layer in
            # that scope can safely share one private pool.
            for staged_group in staged_groups.values():
                captured_groups.append((staged_group, capture_group(staged_group)))

            # Multiple contracts for one physical target are alternatives rather
            # than a fixed execution sequence, so they must not share a pool.
            for entry, staged_adapters in variant_entries:
                graphed_adapters = []
                for adapter, capture_inputs in staged_adapters:
                    graphed_adapters.extend(capture_group([(entry, adapter, capture_inputs)]))
                captured_variants.append((entry, tuple(graphed_adapters)))
        except Exception:
            # No graph has been installed yet. Release every successfully
            # captured TE graph so a later group failure cannot leak a private
            # pool while all user-visible forwards remain eager.
            for graphed_adapter in uninstalled_adapters:
                reset = getattr(graphed_adapter, "reset", None)
                if callable(reset):
                    try:
                        reset()
                    except Exception:
                        logger.exception("Failed to reset an uninstalled partial CUDA graph after capture failure")
            raise

        for staged_group, grouped_results in captured_groups:
            pooled = len(staged_group) > 1
            for (entry, _adapter, _capture_inputs), graphed_adapter in zip(staged_group, grouped_results):
                entry.install(graphed_adapter, allow_eager_fallback=not pooled)
        for entry, graphed_adapters in captured_variants:
            entry.install(graphed_adapters)

        self._captured = True
        self.log_stats("capture")

    def close(self) -> None:
        """Idempotently destroy every partial graph before distributed teardown."""
        if self._closed:
            return

        first_error: Exception | None = None
        for entry in self.entries:
            try:
                entry.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
                logger.exception("Failed to close partial CUDA graph entry %s", entry.name)
        if first_error is not None:
            # Successfully closed entries remain idempotent; retaining the
            # manager lets a caller retry the entry that failed closed.
            raise first_error

        self._captured = False
        self._closed = True

    def stats(self) -> dict[str, int]:
        """Return aggregate capture, replay, and eager-fallback counters."""
        return {
            "captured": sum(entry.capture_count for entry in self.entries),
            "replayed": sum(entry.replay_count for entry in self.entries),
            "fallback": sum(entry.fallback_count for entry in self.entries),
        }

    def log_stats(self, phase: str) -> None:
        """Log visible aggregate graph activity counters."""
        stats = self.stats()
        logger.info(
            "Partial CUDA graph stats (%s): captured=%d replayed=%d fallback=%d",
            phase,
            stats["captured"],
            stats["replayed"],
            stats["fallback"],
        )
        for entry in self.entries:
            logger.info(
                "Partial CUDA graph entry stats (%s): name=%s captured=%d replayed=%d fallback=%d",
                phase,
                entry.name,
                entry.capture_count,
                entry.replay_count,
                entry.fallback_count,
            )

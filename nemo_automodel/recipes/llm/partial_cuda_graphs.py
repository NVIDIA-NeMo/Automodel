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

"""Benchmark-only partial CUDA graphs for GPT-OSS attention and expert compute."""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)

_Canonicalizer = Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]


def _get_make_graphed_callables() -> Callable[..., Any]:
    """Load Transformer Engine's graph helper only when the feature is enabled."""
    from transformer_engine.pytorch.graph import make_graphed_callables

    return make_graphed_callables


def _tensor_metadata(tensor: torch.Tensor) -> tuple[Any, ...]:
    """Return metadata that must stay invariant across graph replays."""
    return (
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.layout,
        tuple(tensor.stride()),
        tensor.requires_grad,
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
    return expected is actual


@dataclass
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

        for expected, tensor in zip(self.tensor_metadata, unique_tensors):
            if _tensor_metadata(tensor) != expected:
                return False, "tensor metadata changed", ()

        tensor_position_set = set(tensor_positions)
        for index, (expected, actual) in enumerate(zip(self.template_leaves, leaves)):
            if index in tensor_position_set:
                continue
            if not _same_control_value(expected, actual):
                return False, "non-tensor control changed", ()

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


class _PartialGraphEntry:
    """One graphable module invocation and its replay safety state."""

    def __init__(
        self,
        *,
        name: str,
        target: nn.Module,
        fp8_enabled: bool,
        canonicalizer: _Canonicalizer | None = None,
    ):
        self.name = name
        self.target = target
        self.fp8_enabled = fp8_enabled
        self.canonicalizer = canonicalizer
        self.original_forward = target.forward
        self.captured_call: _CapturedCall | None = None
        self.adapter: _TensorOnlyCallAdapter | None = None
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0
        self._record_hook: Any = None
        self._captured_training: bool | None = None
        self._parameter_signature: tuple[tuple[Any, ...], ...] = ()
        self._logged_replay = False
        self._logged_fallback = False

    def _canonicalize(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.canonicalizer is None:
            return args, kwargs
        return self.canonicalizer(args, kwargs)

    def start_recording(self) -> None:
        """Record the first eager call through a temporary pre-hook."""

        def record_call(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self.captured_call is not None:
                return
            canonical_args, canonical_kwargs = self._canonicalize(args, kwargs)
            self.captured_call = _CapturedCall.from_call(canonical_args, canonical_kwargs)

        self._record_hook = self.target.register_forward_pre_hook(record_call, with_kwargs=True)

    def stop_recording(self) -> None:
        """Remove the temporary eager-call recorder."""
        if self._record_hook is not None:
            self._record_hook.remove()
            self._record_hook = None

    def build_adapter(self) -> _TensorOnlyCallAdapter:
        """Build the tensor-only capture adapter after an eager sample was observed."""
        if self.captured_call is None:
            raise RuntimeError(f"Partial CUDA graph target {self.name!r} was not called during iteration 0")
        self.adapter = _TensorOnlyCallAdapter(self.target, self.captured_call)
        self.adapter.train(self.target.training)
        return self.adapter

    @staticmethod
    def _parameter_state(parameter: nn.Parameter) -> tuple[Any, ...] | None:
        from torch.distributed.tensor import DTensor

        if isinstance(parameter, DTensor):
            return None
        try:
            data_ptr = parameter.data_ptr()
        except (AttributeError, RuntimeError):
            return None
        if data_ptr == 0:
            return None
        return (id(parameter), data_ptr, tuple(parameter.shape), parameter.dtype, parameter.device)

    def _parameters_match(self) -> bool:
        current = tuple(self._parameter_state(parameter) for parameter in self.target.parameters())
        return all(state is not None for state in current) and current == self._parameter_signature

    def install(self, graphed_adapter: _TensorOnlyCallAdapter) -> None:
        """Install validated graph replay around the original target forward."""
        self.adapter = graphed_adapter
        self.capture_count = 1
        self._captured_training = self.target.training
        parameter_signature = tuple(self._parameter_state(parameter) for parameter in self.target.parameters())
        if any(state is None for state in parameter_signature):
            raise RuntimeError(f"Partial CUDA graph target {self.name!r} has unstable parameter storage")
        self._parameter_signature = tuple(state for state in parameter_signature if state is not None)

        def dispatch(*args: Any, **kwargs: Any) -> Any:
            try:
                canonical_args, canonical_kwargs = self._canonicalize(args, kwargs)
                assert self.captured_call is not None
                valid, reason, tensors = self.captured_call.validate(canonical_args, canonical_kwargs)
            except Exception as error:  # a changed dynamic call must stay correct via eager fallback
                valid, reason, tensors = False, f"call canonicalization failed: {error}", ()

            if valid and self.target.training != self._captured_training:
                valid, reason = False, "training mode changed"
            if valid and not self._parameters_match():
                valid, reason = False, "parameter identity or storage changed"

            if valid:
                self.replay_count += 1
                if not self._logged_replay:
                    logger.info("Partial CUDA graph replay active for %s", self.name)
                    self._logged_replay = True
                assert self.adapter is not None
                return self.adapter(*tensors)

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


def _canonicalize_te_ops_experts(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Validate TE fused-MLP input aliases while keeping split values dynamic."""
    if kwargs:
        raise RuntimeError("TE-ops expert graph expects positional tensor inputs only")
    if len(args) != 5 or not all(isinstance(arg, torch.Tensor) for arg in args):
        raise RuntimeError("TE-ops expert graph expects (hidden, splits, probs, splits, probs)")
    if args[1] is not args[3] or args[2] is not args[4]:
        raise RuntimeError("TE fused grouped MLP requires repeated split/prob inputs to preserve tensor identity")
    return args, kwargs


class PartialCudaGraphManager:
    """Capture selected GPT-OSS submodules after one eager benchmark iteration."""

    def __init__(self, entries: list[_PartialGraphEntry], fp8_recipe: Any = None):
        self.entries = entries
        self.fp8_recipe = fp8_recipe
        self._captured = False

    @classmethod
    def from_model_parts(
        cls,
        model_parts: list[nn.Module],
        *,
        activation_checkpointing: bool = False,
    ) -> PartialCudaGraphManager | None:
        """Discover graph targets from an already-built GPT-OSS benchmark model."""
        enabled_parts = [
            part
            for part in model_parts
            if getattr(getattr(part, "backend", None), "partial_cuda_graph_attention", False) is True
            or getattr(getattr(part, "backend", None), "partial_cuda_graph_experts", False) is True
        ]
        if not enabled_parts:
            return None
        if activation_checkpointing:
            raise RuntimeError("Partial CUDA graphs require activation_checkpointing=false")
        if len(model_parts) != 1:
            raise RuntimeError("Partial GPT-OSS CUDA graphs currently require pipeline parallel size 1")

        model = model_parts[0]
        if getattr(getattr(model, "config", None), "model_type", None) != "gpt_oss":
            raise RuntimeError("Partial CUDA graph benchmark support is currently limited to GptOssForCausalLM")
        backend = model.backend
        limit = backend.partial_cuda_graph_layer_limit
        layers = list(model.model.layers.items())
        if limit > len(layers):
            raise RuntimeError(
                f"partial_cuda_graph_layer_limit={limit} exceeds the {len(layers)} discovered GPT-OSS layers"
            )
        selected_layers = layers[:limit]
        entries: list[_PartialGraphEntry] = []

        for layer_name, block in selected_layers:
            if backend.partial_cuda_graph_attention:
                dpa = getattr(block.self_attn, "attn_module", None)
                target = getattr(dpa, "fused_attention", None)
                if not isinstance(target, nn.Module):
                    raise RuntimeError(f"GPT-OSS layer {layer_name} does not expose TE FusedAttention")
                if any(True for _ in target.parameters()):
                    raise RuntimeError("The partial attention boundary must be parameterless")
                entries.append(
                    _PartialGraphEntry(
                        name=f"gpt_oss.layers.{layer_name}.fused_attention",
                        target=target,
                        fp8_enabled=False,
                        canonicalizer=_canonicalize_bf16_fused_attention,
                    )
                )

            if backend.partial_cuda_graph_experts:
                experts = getattr(getattr(block, "mlp", None), "experts", None)
                target = getattr(experts, "_te_grouped_mlp", None)
                if not getattr(experts, "use_te_ops", False) or not isinstance(target, nn.Module):
                    raise RuntimeError(f"GPT-OSS layer {layer_name} does not expose a TE-ops grouped MLP")
                unstable_parameters = [
                    name
                    for name, parameter in target.named_parameters()
                    if _PartialGraphEntry._parameter_state(parameter) is None
                ]
                if unstable_parameters:
                    raise RuntimeError(
                        "Partial expert CUDA graphs require plain parameters with stable allocated storage; "
                        f"found DTensor or unallocated parameters in layer {layer_name}: {unstable_parameters}"
                    )
                entries.append(
                    _PartialGraphEntry(
                        name=f"gpt_oss.layers.{layer_name}.te_ops_experts",
                        target=target,
                        fp8_enabled=backend.te_fp8 is not None,
                        canonicalizer=_canonicalize_te_ops_experts,
                    )
                )

        if not entries:
            raise RuntimeError("Partial CUDA graphs were enabled but no graph targets were found")

        recipe = backend.te_fp8.build_recipe() if backend.te_fp8 is not None else None
        manager = cls(entries, fp8_recipe=recipe)
        manager.start_recording()
        return manager

    def start_recording(self) -> None:
        """Install first-call recorders on every selected target."""
        for entry in self.entries:
            entry.start_recording()

    def capture(self) -> None:
        """Batch-capture all observed targets in real forward order."""
        if self._captured:
            return
        for entry in self.entries:
            entry.stop_recording()

        skipped_entries = tuple(entry for entry in self.entries if entry.captured_call is None)
        unexpected_skips = tuple(
            entry.name for entry in skipped_entries if entry.canonicalizer is not _canonicalize_te_ops_experts
        )
        if unexpected_skips:
            raise RuntimeError(
                "Every non-expert partial CUDA graph target must have an iteration-0 sample; "
                f"missing samples for {unexpected_skips}"
            )
        for entry in skipped_entries:
            logger.warning(
                "Skipping partial CUDA graph capture for %s because it received no iteration-0 expert tokens; "
                "the target will remain eager",
                entry.name,
            )

        captured_entries = tuple(entry for entry in self.entries if entry.captured_call is not None)
        if not captured_entries:
            self._captured = True
            self.log_stats("capture")
            return

        adapters = tuple(entry.build_adapter() for entry in captured_entries)
        captured_calls = tuple(entry.captured_call for entry in captured_entries)
        sample_args = tuple(call.sample_tensors for call in captured_calls if call is not None)
        enabled = tuple(entry.fp8_enabled for entry in captured_entries)
        if any(enabled) and self.fp8_recipe is None:
            raise RuntimeError("FP8 expert graph capture requires a Transformer Engine FP8 recipe")

        kwargs = {
            "num_warmup_iters": 3,
            "allow_unused_input": True,
            "sample_kwargs": tuple({} for _ in adapters),
            "enabled": enabled,
        }
        if self.fp8_recipe is not None:
            kwargs["recipe"] = self.fp8_recipe

        try:
            graphed_adapters = _get_make_graphed_callables()(adapters, sample_args, **kwargs)
        except Exception as error:
            raise RuntimeError("Explicit partial CUDA graph capture failed") from error

        if not isinstance(graphed_adapters, tuple):
            graphed_adapters = (graphed_adapters,)
        if len(graphed_adapters) != len(captured_entries):
            raise RuntimeError(
                f"Transformer Engine returned {len(graphed_adapters)} graphs for {len(captured_entries)} targets"
            )
        for entry, graphed_adapter in zip(captured_entries, graphed_adapters):
            entry.install(graphed_adapter)

        self._captured = True
        self.log_stats("capture")

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

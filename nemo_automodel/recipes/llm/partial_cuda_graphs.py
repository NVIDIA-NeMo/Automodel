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

"""Scoped partial CUDA graphs for GPT-OSS attention and MoE compute.

The convergence-safe dropless-MoE path mirrors Megatron-LM: graph attention,
the fixed-shape router, and fixed-shape HybridEP preprocessing while keeping
variable-size dispatch dynamic. Expert graphs can optionally use a fixed local
compute bucket after dispatch: inputs are padded for Transformer Engine, outputs
are sliced back to the exact routed-token count before HybridEP combine, and
overflow calls stay eager without dropping or retrying tokens.
"""

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
        expert_bucket_tokens: int | None = None,
    ):
        self.name = name
        self.target = target
        self.fp8_enabled = fp8_enabled
        self.canonicalizer = canonicalizer
        self.expert_bucket_tokens = expert_bucket_tokens
        self.original_forward = target.forward
        self.captured_call: _CapturedCall | None = None
        self.adapter: _TensorOnlyCallAdapter | None = None
        self.capture_count = 0
        self.replay_count = 0
        self.fallback_count = 0
        self.bucketed_replay_count = 0
        self.bucket_padding_tokens = 0
        self.bucket_overflow_fallback_count = 0
        self.bucket_empty_fallback_count = 0
        self.bucket_capture_overflow_skip_count = 0
        self.bucket_capture_empty_skip_count = 0
        self._record_hook: Any = None
        self._capture_skip_reason: str | None = None
        self._captured_training: bool | None = None
        self._parameter_signature: tuple[tuple[Any, ...], ...] = ()
        self._logged_replay = False
        self._logged_fallback = False

    def _canonicalize(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.canonicalizer is None:
            return args, kwargs
        return self.canonicalizer(args, kwargs)

    def _apply_expert_bucket(
        self,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any], int | None, str]:
        """Pad one post-dispatch expert call to its fixed compute capacity."""
        capacity = self.expert_bucket_tokens
        if capacity is None:
            return args, kwargs, None, ""

        hidden_states, split_sizes, probs, _, _ = args
        token_count = hidden_states.shape[0]
        if token_count == 0:
            return args, kwargs, token_count, "routed token count is zero"
        if split_sizes.numel() == 0:
            return args, kwargs, token_count, "expert split tensor has no local experts"
        if token_count > capacity:
            return (
                args,
                kwargs,
                token_count,
                f"routed token count {token_count} exceeds expert bucket capacity {capacity}",
            )

        padding = capacity - token_count
        if padding == 0:
            return args, kwargs, token_count, ""

        hidden_padding = hidden_states.new_zeros((padding, *hidden_states.shape[1:]))
        probs_padding = probs.new_zeros((padding, *probs.shape[1:]))
        padded_hidden_states = torch.cat((hidden_states, hidden_padding), dim=0)
        padded_probs = torch.cat((probs, probs_padding), dim=0)
        # TE's graph-safe GroupedTensor and fused grouped-MLP paths explicitly
        # support paged-stash buffers where physical B > sum(split_sizes). The
        # tail is unused capacity, so keep the real per-expert sizes unchanged;
        # assigning the tail to an expert would change the routing metadata and
        # force that expert to compute the padding. numel()/shape checks above
        # inspect tensor metadata only and do not synchronize with the device.
        return (
            (padded_hidden_states, split_sizes, padded_probs, split_sizes, padded_probs),
            kwargs,
            token_count,
            "",
        )

    def start_recording(self) -> None:
        """Record the first eager call through a temporary pre-hook."""

        def record_call(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if self._capture_skip_reason is not None:
                return
            if self.captured_call is not None and self.expert_bucket_tokens is None:
                return
            canonical_args, canonical_kwargs = self._canonicalize(args, kwargs)
            canonical_args, canonical_kwargs, bucket_token_count, bucket_reason = self._apply_expert_bucket(
                canonical_args,
                canonical_kwargs,
            )
            if bucket_reason:
                self.captured_call = None
                self._capture_skip_reason = f"iteration 0 {bucket_reason}"
                if self.expert_bucket_tokens is not None:
                    if bucket_token_count:
                        if bucket_token_count > self.expert_bucket_tokens:
                            self.bucket_capture_overflow_skip_count = 1
                    else:
                        self.bucket_capture_empty_skip_count = 1
                return
            if self.captured_call is None:
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
            bucket_token_count: int | None = None
            try:
                canonical_args, canonical_kwargs = self._canonicalize(args, kwargs)
                canonical_args, canonical_kwargs, bucket_token_count, reason = self._apply_expert_bucket(
                    canonical_args,
                    canonical_kwargs,
                )
                if reason:
                    valid, tensors = False, ()
                else:
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
                if bucket_token_count is not None:
                    self.bucketed_replay_count += 1
                    assert self.expert_bucket_tokens is not None
                    self.bucket_padding_tokens += self.expert_bucket_tokens - bucket_token_count
                if not self._logged_replay:
                    logger.info("Partial CUDA graph replay active for %s", self.name)
                    self._logged_replay = True
                assert self.adapter is not None
                output = self.adapter(*tensors)
                if bucket_token_count is not None:
                    if not isinstance(output, torch.Tensor):
                        raise RuntimeError("Partial expert CUDA graph bucket expects one tensor output")
                    output = output.narrow(0, 0, bucket_token_count)
                return output

            self.fallback_count += 1
            if self.expert_bucket_tokens is not None and bucket_token_count is not None:
                if bucket_token_count == 0:
                    self.bucket_empty_fallback_count += 1
                elif bucket_token_count > self.expert_bucket_tokens:
                    self.bucket_overflow_fallback_count += 1
            if not self._logged_fallback:
                logger.warning("Partial CUDA graph eager fallback for %s: %s", self.name, reason)
                self._logged_fallback = True
            return self.original_forward(*args, **kwargs)

        self.target.forward = dispatch

    def mark_unobserved_expert_bucket(self) -> None:
        """Record that iteration 0 never reached this bucketed expert target."""
        if self.expert_bucket_tokens is not None and self._capture_skip_reason is None:
            self.bucket_capture_empty_skip_count = 1


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
        pipeline_parallel: bool = False,
    ) -> PartialCudaGraphManager | None:
        """Discover graph targets from an already-built GPT-OSS benchmark model."""
        enabled_parts = [
            part
            for part in model_parts
            if getattr(getattr(part, "backend", None), "partial_cuda_graph_attention", False) is True
            or getattr(getattr(part, "backend", None), "partial_cuda_graph_moe_router", False) is True
            or getattr(getattr(part, "backend", None), "partial_cuda_graph_moe_preprocess", False) is True
            or getattr(getattr(part, "backend", None), "partial_cuda_graph_experts", False) is True
        ]
        if not enabled_parts:
            return None
        if pipeline_parallel:
            raise RuntimeError(
                "Partial GPT-OSS CUDA graphs require pipeline parallel size 1; multiple in-flight pipeline "
                "microbatches can overwrite a graph's static forward buffers before backward"
            )
        if activation_checkpointing:
            if any(
                getattr(part.backend, "partial_cuda_graph_moe_router", False)
                or getattr(part.backend, "partial_cuda_graph_moe_preprocess", False)
                for part in enabled_parts
            ):
                raise RuntimeError(
                    "PyTorch activation checkpointing cannot recompute across the partial MoE router/preprocess "
                    "CUDA graph scope; use attention/exact-shape expert graphs or disable router/preprocess graphs"
                )
            logger.info(
                "Partial CUDA graphs are enabled with PyTorch activation checkpointing; "
                "checkpoint recomputation will use the same guarded graph entry points"
            )
        if len(model_parts) != 1:
            raise RuntimeError("Partial GPT-OSS CUDA graphs currently require one local model part")

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

        for layer_name, wrapped_block in selected_layers:
            block = wrapped_block
            while hasattr(block, "_checkpoint_wrapped_module"):
                block = block._checkpoint_wrapped_module

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

            if backend.partial_cuda_graph_moe_router:
                gate = getattr(getattr(block, "mlp", None), "gate", None)
                target = getattr(gate, "routing_core", None)
                if not isinstance(target, nn.Module):
                    raise RuntimeError(f"GPT-OSS layer {layer_name} does not expose a graphable MoE routing core")
                if (
                    getattr(gate, "router_replay", None) is not None
                    or getattr(gate, "e_score_correction_bias", None) is not None
                ):
                    raise RuntimeError(
                        "Partial MoE router graphs require routing replay and score-correction bias to be disabled"
                    )
                if any(True for _ in target.parameters()):
                    raise RuntimeError("The partial MoE routing-core boundary must be parameterless")
                entries.append(
                    _PartialGraphEntry(
                        name=f"gpt_oss.layers.{layer_name}.moe_router",
                        target=target,
                        fp8_enabled=False,
                    )
                )

            if backend.partial_cuda_graph_moe_preprocess:
                experts = getattr(getattr(block, "mlp", None), "experts", None)
                dispatcher = getattr(experts, "token_dispatcher", None)
                target = getattr(dispatcher, "hybridep_metadata_processor", None)
                if not isinstance(target, nn.Module):
                    raise RuntimeError(
                        f"GPT-OSS layer {layer_name} does not expose graphable HybridEP metadata preprocessing"
                    )
                if not getattr(target, "permute_fusion", False):
                    raise RuntimeError(
                        "Partial HybridEP preprocess graphs require fused fixed-shape metadata conversion"
                    )
                entries.append(
                    _PartialGraphEntry(
                        name=f"gpt_oss.layers.{layer_name}.moe_preprocess",
                        target=target,
                        fp8_enabled=False,
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
                        expert_bucket_tokens=getattr(
                            backend,
                            "partial_cuda_graph_expert_bucket_tokens",
                            None,
                        ),
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
            entry.mark_unobserved_expert_bucket()
            if entry._capture_skip_reason is not None:
                logger.warning(
                    "Skipping partial CUDA graph capture for %s because %s; the target will remain eager",
                    entry.name,
                    entry._capture_skip_reason,
                )
            else:
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
        if any(entry.fp8_enabled for entry in captured_entries) and self.fp8_recipe is None:
            raise RuntimeError("FP8 expert graph capture requires a Transformer Engine FP8 recipe")

        # Give every guarded entry its own graph pool. Any entry may deliberately
        # fall back when metadata changes; sharing a pool would require replaying
        # all graphs in capture order and could corrupt another entry's buffers
        # when one dynamic expert call is skipped.
        graphed_adapters = []
        for entry, adapter in zip(captured_entries, adapters):
            assert entry.captured_call is not None
            kwargs = {
                "num_warmup_iters": 3,
                "allow_unused_input": True,
                "sample_kwargs": ({},),
                "enabled": (entry.fp8_enabled,),
            }
            if self.fp8_recipe is not None:
                kwargs["recipe"] = self.fp8_recipe
            try:
                result = _get_make_graphed_callables()(
                    (adapter,),
                    (entry.captured_call.sample_tensors,),
                    **kwargs,
                )
            except Exception as error:
                raise RuntimeError(f"Explicit partial CUDA graph capture failed for {entry.name}") from error
            if not isinstance(result, tuple):
                result = (result,)
            if len(result) != 1:
                raise RuntimeError(
                    f"Transformer Engine returned {len(result)} graphs for partial target {entry.name!r}"
                )
            graphed_adapters.append(result[0])

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

    def expert_bucket_stats(self) -> dict[str, int]:
        """Return counters specific to fixed-capacity post-dispatch expert graphs."""
        bucket_entries = tuple(entry for entry in self.entries if entry.expert_bucket_tokens is not None)
        return {
            "entries": len(bucket_entries),
            "capacity_tokens": sum(entry.expert_bucket_tokens or 0 for entry in bucket_entries),
            "bucketed_replay": sum(entry.bucketed_replay_count for entry in bucket_entries),
            "padding_tokens": sum(entry.bucket_padding_tokens for entry in bucket_entries),
            "overflow_fallback": sum(entry.bucket_overflow_fallback_count for entry in bucket_entries),
            "empty_fallback": sum(entry.bucket_empty_fallback_count for entry in bucket_entries),
            "capture_overflow_skip": sum(entry.bucket_capture_overflow_skip_count for entry in bucket_entries),
            "capture_empty_skip": sum(entry.bucket_capture_empty_skip_count for entry in bucket_entries),
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
        bucket_stats = self.expert_bucket_stats()
        if bucket_stats["entries"]:
            logger.info(
                "Partial expert bucket stats (%s): entries=%d capacity_tokens=%d bucketed_replay=%d "
                "padding_tokens=%d overflow_fallback=%d empty_fallback=%d capture_overflow_skip=%d "
                "capture_empty_skip=%d",
                phase,
                bucket_stats["entries"],
                bucket_stats["capacity_tokens"],
                bucket_stats["bucketed_replay"],
                bucket_stats["padding_tokens"],
                bucket_stats["overflow_fallback"],
                bucket_stats["empty_fallback"],
                bucket_stats["capture_overflow_skip"],
                bucket_stats["capture_empty_skip"],
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

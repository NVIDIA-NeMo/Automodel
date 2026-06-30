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

"""Scoped partial CUDA graphs for Transformer Engine attention and MoE compute.

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
from collections.abc import Callable, Iterator, Sequence
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
    """Capture module parameters as fresh leaf inputs instead of module inputs.

    PyTorch associates each persistent ``AccumulateGrad`` node with the stream
    on which the node was created. FSDP/DDP may keep nodes from an eager step
    alive, so letting TE discover the real module parameters during a later
    side-stream capture can make the legacy stream wait on a capturing stream.

    This adapter presents no module parameters to TE. Instead, capture uses
    detached aliases that share the real parameter storage but own fresh
    ``AccumulateGrad`` nodes. Replay passes the real parameters as explicit
    inputs to TE's graphed autograd function, which returns their captured
    gradients to the normal eager/FSDP autograd graph.
    """

    def __init__(
        self,
        target: nn.Module,
        graph_target: nn.Module,
        captured_call: _CapturedCall,
    ) -> None:
        super().__init__()
        # Register the graph-only target so TE can discover its BasicOperations
        # and FP8 state. ``parameters()`` below intentionally hides its owners
        # from make_graphed_callables.
        self.graph_target = graph_target
        self.__dict__["target"] = target
        self.captured_call = captured_call

        target_parameters = tuple(target.named_parameters())
        graph_parameter_names = tuple(name for name, _parameter in graph_target.named_parameters())
        target_parameter_names = tuple(name for name, _parameter in target_parameters)
        if graph_parameter_names != target_parameter_names:
            raise RuntimeError(
                "Partial expert CUDA graph parameter clone does not match the live TE-ops target; "
                f"clone={graph_parameter_names}, target={target_parameter_names}"
            )
        if not target_parameters:
            raise RuntimeError("Partial expert CUDA graph explicit-parameter adapter found no parameters")

        self.parameter_names = target_parameter_names
        self.dynamic_input_count = len(captured_call.sample_tensors)
        aliases = []
        for _name, parameter in target_parameters:
            if not isinstance(parameter, nn.Parameter):
                raise RuntimeError(
                    "Partial expert CUDA graph parameters must be plain nn.Parameter objects during capture; "
                    f"got {type(parameter).__name__}"
                )
            alias = parameter.detach()
            alias.requires_grad_(parameter.requires_grad)
            if alias.data_ptr() != parameter.data_ptr():
                raise RuntimeError("Partial expert CUDA graph parameter alias did not preserve storage")
            aliases.append(alias)
        self.capture_parameter_aliases = tuple(aliases)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Hide registered graph-target owners from TE parameter discovery."""
        del recurse
        return iter(())

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """Hide registered graph-target owners from generic module utilities."""
        del prefix, recurse, remove_duplicate
        return iter(())

    @property
    def capture_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return dynamic samples followed by fresh storage-sharing leaves."""
        return self.captured_call.sample_tensors + self.capture_parameter_aliases

    def forward(self, *tensor_inputs: torch.Tensor) -> Any:
        """Run the graph-only target with explicit parameter values."""
        expected_inputs = self.dynamic_input_count + len(self.parameter_names)
        if len(tensor_inputs) != expected_inputs:
            raise RuntimeError(f"Expected {expected_inputs} explicit graph inputs, got {len(tensor_inputs)}")
        dynamic_inputs = tensor_inputs[: self.dynamic_input_count]
        parameter_inputs = tensor_inputs[self.dynamic_input_count :]
        args, kwargs = self.captured_call.rebuild(dynamic_inputs)
        parameter_mapping = dict(zip(self.parameter_names, parameter_inputs))
        return torch.func.functional_call(
            self.graph_target,
            parameter_mapping,
            args,
            kwargs,
            tie_weights=False,
            strict=False,
        )


def _build_te_ops_explicit_parameter_adapter(
    target: nn.Module,
    captured_call: _CapturedCall,
) -> _ExplicitParameterCallAdapter | None:
    """Build a fresh TE Sequential fuser whose leaves are capture aliases."""
    try:
        from transformer_engine.pytorch import ops as te_ops
    except ImportError:
        return None
    if not isinstance(target, te_ops.Sequential):
        return None

    # GroupedExpertsTeOps deliberately owns one lazily-created Sequential.
    # Reusing its BasicOperations in a fresh Sequential gives capture a fresh
    # OperationFuser. During the first functional call that fuser caches the
    # explicit alias leaves instead of the real FSDP/optimizer Parameters.
    graph_target = te_ops.Sequential(dict(target._modules))
    return _ExplicitParameterCallAdapter(target, graph_target, captured_call)


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
        expert_bucket_uses_paged_capacity: bool = False,
        expert_graph_storage: Any = None,
    ):
        self.name = name
        self.target = target
        self.fp8_enabled = fp8_enabled
        self.canonicalizer = canonicalizer
        self.expert_bucket_tokens = expert_bucket_tokens
        self.expert_bucket_uses_paged_capacity = expert_bucket_uses_paged_capacity
        self.expert_graph_storage = expert_graph_storage
        self.original_forward = target.forward
        self.captured_call: _CapturedCall | None = None
        self.adapter: nn.Module | None = None
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
        self._closed = False
        self._explicit_parameter_names: tuple[str, ...] = ()

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

        hidden_states, split_sizes, probs = args[:3]
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
        # Host-split grouped ops require sum(split_sizes) to match the physical
        # compute shape, so assign their zero-probability tail to the final local
        # expert. TE's graph-safe GroupedTensor path supports paged capacity
        # (physical B > sum(split_sizes)) in BF16/FP16 and MXFP8 forward and
        # backward, including the full MXFP8 grouped-MLP fusion. Those paths
        # must retain the real dynamic splits. Dispatch itself always uses the
        # exact splits.
        if self.expert_bucket_uses_paged_capacity:
            padded_split_sizes = split_sizes
        else:
            padded_split_sizes = torch.cat((split_sizes[:-1], split_sizes[-1:] + padding))
        if len(args) == 4:
            padded_args = (padded_hidden_states, padded_split_sizes, padded_probs, padded_split_sizes)
        else:
            padded_args = (
                padded_hidden_states,
                padded_split_sizes,
                padded_probs,
                padded_split_sizes,
                padded_probs,
            )
        return (
            padded_args,
            kwargs,
            token_count,
            "",
        )

    def start_recording(self) -> None:
        """Record the first eager call through a temporary pre-hook."""
        if self._closed:
            raise RuntimeError(f"Partial CUDA graph entry {self.name!r} is closed")

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

    def close(self) -> None:
        """Destroy graph replay state and release retained expert storage safely."""
        if self._closed:
            return

        self.stop_recording()

        # Stop new calls from entering a graph before destroying it. TE attaches
        # ``reset`` to the adapter returned by make_graphed_callables; that must
        # run while every captured tensor/parameter address is still alive.
        self.target.forward = self.original_forward
        adapter = self.adapter
        if adapter is not None:
            reset = getattr(adapter, "reset", None)
            if self.capture_count and not callable(reset):
                raise RuntimeError(f"Captured partial CUDA graph adapter for {self.name!r} has no reset()")
            if callable(reset):
                reset()
            # TE installs graph closures directly on the adapter instance.
            # Remove them after reset so no Python attribute retains graph/static
            # buffers while expert all-gather storage is released below.
            adapter.__dict__.pop("forward", None)
            adapter.__dict__.pop("backward_dw", None)
            adapter.__dict__.pop("reset", None)
            del reset

        # Drop all graph/static-buffer references before allowing FSDP2 to free
        # the retained all-gather allocation captured by an expert graph.
        self.adapter = None
        self.captured_call = None
        self._parameter_signature = ()
        self._explicit_parameter_names = ()
        self._captured_training = None
        del adapter

        storage = self.expert_graph_storage
        if storage is not None:
            storage.reset()
            self.expert_graph_storage = None
        self._closed = True

    def build_adapter(self) -> nn.Module:
        """Build the tensor-only capture adapter after an eager sample was observed."""
        if self.captured_call is None:
            raise RuntimeError(f"Partial CUDA graph target {self.name!r} was not called during iteration 0")
        explicit_adapter = None
        if self.canonicalizer is _canonicalize_te_ops_experts:
            explicit_adapter = _build_te_ops_explicit_parameter_adapter(self.target, self.captured_call)
        if explicit_adapter is None:
            self.adapter = _TensorOnlyCallAdapter(self.target, self.captured_call)
            self._explicit_parameter_names = ()
        else:
            self.adapter = explicit_adapter
            self._explicit_parameter_names = explicit_adapter.parameter_names
        self.adapter.train(self.target.training)
        return self.adapter

    def capture_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return the sample surface passed to Transformer Engine."""
        if self.adapter is None or self.captured_call is None:
            raise RuntimeError(f"Partial CUDA graph adapter for {self.name!r} has not been built")
        if isinstance(self.adapter, _ExplicitParameterCallAdapter):
            return self.adapter.capture_inputs
        return self.captured_call.sample_tensors

    def replay_inputs(self, dynamic_inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        """Append live parameters when they are explicit graph inputs."""
        if not self._explicit_parameter_names:
            return dynamic_inputs
        named_parameters = tuple(self.target.named_parameters())
        parameter_names = tuple(name for name, _parameter in named_parameters)
        if parameter_names != self._explicit_parameter_names:
            raise RuntimeError(
                f"Partial CUDA graph parameter names changed for {self.name!r}: "
                f"expected {self._explicit_parameter_names}, got {parameter_names}"
            )
        return dynamic_inputs + tuple(parameter for _name, parameter in named_parameters)

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

    def install(self, graphed_adapter: nn.Module) -> None:
        """Install validated graph replay around the original target forward."""
        if self._closed:
            raise RuntimeError(f"Partial CUDA graph entry {self.name!r} is closed")
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
                output = self.adapter(*self.replay_inputs(tensors))
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
    if len(args) not in (4, 5) or not all(isinstance(arg, torch.Tensor) for arg in args):
        raise RuntimeError(
            "TE-ops expert graph expects (hidden, splits, probs, splits[, probs]) for biasless or biased experts"
        )
    if args[1] is not args[3]:
        raise RuntimeError("TE grouped MLP requires repeated split inputs to preserve tensor identity")
    if len(args) == 5 and args[2] is not args[4]:
        raise RuntimeError("TE grouped MLP with expert bias requires repeated probability inputs to preserve identity")
    return args, kwargs


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


def _find_te_ops_experts(block: nn.Module) -> nn.Module | None:
    """Find stacked TE-ops experts anywhere beneath a transformer block."""
    for _name, module in block.named_modules():
        if getattr(module, "use_te_ops", False) and isinstance(getattr(module, "_te_grouped_mlp", None), nn.Module):
            return module
    return None


def _find_fused_attention(block: nn.Module) -> nn.Module | None:
    """Find the parameterless TE fused-attention boundary in a transformer block."""
    attention = getattr(block, "self_attn", None)
    if isinstance(attention, nn.Module):
        dpa = getattr(attention, "attn_module", None)
        target = getattr(dpa, "fused_attention", None)
        if isinstance(target, nn.Module):
            return target

    for module in block.modules():
        target = getattr(module, "fused_attention", None)
        if isinstance(target, nn.Module):
            return target
    return None


def _discover_blocks(model: nn.Module) -> list[_DiscoveredBlock]:
    """Discover main-stack and MTP blocks using the shared MoE traversal."""
    from nemo_automodel.components.moe.parallelizer import _iter_transformer_and_mtp_blocks

    label = _model_label(model)
    mtp_layers = getattr(getattr(model, "mtp", None), "layers", None)
    blocks = []
    for parent_layers, layer_id, wrapped_block in _iter_transformer_and_mtp_blocks(model):
        block = _unwrap_checkpoint_wrappers(wrapped_block)
        is_mtp = parent_layers is mtp_layers
        scope = "mtp.layers" if is_mtp else "layers"
        blocks.append(
            _DiscoveredBlock(
                name=f"{label}.{scope}.{layer_id}",
                module=block,
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
    feature: str,
    candidates: list[tuple[int, nn.Module]],
    blocks: list[_DiscoveredBlock],
    limit: int,
    repeated_mtp_layer: bool,
) -> dict[int, nn.Module]:
    """Select graph targets while rejecting shared/repeated physical call sites."""
    if len(candidates) < limit:
        raise RuntimeError(
            f"partial_cuda_graph_{feature} requested {limit} layers, but only {len(candidates)} "
            f"layers expose a graphable {feature} boundary"
        )
    selected = candidates[:limit]
    if repeated_mtp_layer and any(blocks[index].is_mtp for index, _target in selected):
        raise RuntimeError(
            f"partial_cuda_graph_{feature} cannot capture repeated-layer MTP: one physical {feature} target "
            "is invoked multiple times before backward and requires graph replicas or ring slots"
        )
    target_ids = [id(target) for _index, target in selected]
    if len(target_ids) != len(set(target_ids)):
        raise RuntimeError(
            f"partial_cuda_graph_{feature} selected a shared physical target more than once; "
            "shared/repeated modules require independent graph replicas"
        )
    return dict(selected)


class PartialCudaGraphManager:
    """Capture selected TE attention and MoE submodules after one eager iteration."""

    def __init__(self, entries: list[_PartialGraphEntry], fp8_recipe: Any = None):
        self.entries = entries
        self.fp8_recipe = fp8_recipe
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
        """Discover graph targets from an already-built MoE benchmark model."""
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
                "Partial CUDA graphs require pipeline parallel size 1; multiple in-flight pipeline "
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
            raise RuntimeError("Partial CUDA graphs currently require one local model part")

        model = model_parts[0]
        backend = model.backend
        limit = backend.partial_cuda_graph_layer_limit
        blocks = _discover_blocks(model)
        if limit > len(blocks):
            raise RuntimeError(
                f"partial_cuda_graph_layer_limit={limit} exceeds the {len(blocks)} discovered transformer/MTP layers"
            )
        entries: list[_PartialGraphEntry] = []

        targets_by_feature: dict[str, dict[int, nn.Module]] = {}
        unsupported_expert_graphs: list[str] = []
        repeated_mtp_layer = _uses_repeated_mtp_layer(model)

        if backend.partial_cuda_graph_attention:
            candidates = [
                (index, target) for index, block in enumerate(blocks) if (target := _find_fused_attention(block.module))
            ]
            targets_by_feature["attention"] = _select_graph_targets(
                feature="attention",
                candidates=candidates,
                blocks=blocks,
                limit=limit,
                repeated_mtp_layer=repeated_mtp_layer,
            )

        if backend.partial_cuda_graph_moe_router:
            candidates = [
                (index, target)
                for index, block in enumerate(blocks)
                if block.moe is not None
                and isinstance((target := getattr(getattr(block.moe, "gate", None), "routing_core", None)), nn.Module)
            ]
            targets_by_feature["router"] = _select_graph_targets(
                feature="moe_router",
                candidates=candidates,
                blocks=blocks,
                limit=limit,
                repeated_mtp_layer=repeated_mtp_layer,
            )

        if backend.partial_cuda_graph_moe_preprocess:
            candidates = []
            for index, block in enumerate(blocks):
                experts = getattr(block.moe, "experts", None)
                dispatcher = getattr(experts, "token_dispatcher", None)
                target = getattr(dispatcher, "hybridep_metadata_processor", None)
                if isinstance(target, nn.Module) and getattr(target, "permute_fusion", False):
                    candidates.append((index, target))
            targets_by_feature["preprocess"] = _select_graph_targets(
                feature="moe_preprocess",
                candidates=candidates,
                blocks=blocks,
                limit=limit,
                repeated_mtp_layer=repeated_mtp_layer,
            )

        if backend.partial_cuda_graph_experts:
            candidates = []
            for index, block in enumerate(blocks):
                experts = _find_te_ops_experts(block.module)
                target = getattr(experts, "_te_grouped_mlp", None)
                if isinstance(target, nn.Module):
                    capability = getattr(experts, "_te_ops_dynamic_splits_graph_capability", None)
                    if callable(capability):
                        graph_safe, reason = capability()
                    else:
                        graph_safe = bool(getattr(experts, "_te_ops_dynamic_splits_graph_safe", False))
                        reason = "the expert backend did not advertise on-device dynamic split support"
                    if graph_safe:
                        candidates.append((index, target))
                    else:
                        unsupported_expert_graphs.append(f"{block.name}: {reason}")
            if len(candidates) >= limit:
                targets_by_feature["experts"] = _select_graph_targets(
                    feature="experts",
                    candidates=candidates,
                    blocks=blocks,
                    limit=limit,
                    repeated_mtp_layer=repeated_mtp_layer,
                )
            elif unsupported_expert_graphs:
                logger.warning(
                    "Leaving TE-ops experts eager because dynamic split replay is unsupported: %s",
                    "; ".join(unsupported_expert_graphs),
                )
                targets_by_feature["experts"] = {}
            else:
                targets_by_feature["experts"] = _select_graph_targets(
                    feature="experts",
                    candidates=candidates,
                    blocks=blocks,
                    limit=limit,
                    repeated_mtp_layer=repeated_mtp_layer,
                )

        for index, block in enumerate(blocks):
            target = targets_by_feature.get("attention", {}).get(index)
            if target is not None:
                if any(True for _ in target.parameters()):
                    raise RuntimeError(f"The partial attention boundary in {block.name} must be parameterless")
                entries.append(
                    _PartialGraphEntry(
                        name=f"{block.name}.fused_attention",
                        target=target,
                        fp8_enabled=False,
                        canonicalizer=_canonicalize_bf16_fused_attention,
                    )
                )

            target = targets_by_feature.get("router", {}).get(index)
            if target is not None:
                assert block.moe is not None
                gate = block.moe.gate
                if (
                    getattr(gate, "router_replay", None) is not None
                    or getattr(gate, "e_score_correction_bias", None) is not None
                ):
                    raise RuntimeError(
                        "Partial MoE router graphs require routing replay and score-correction bias to be disabled"
                    )
                if any(True for _ in target.parameters()):
                    raise RuntimeError(f"The partial MoE routing-core boundary in {block.name} must be parameterless")
                entries.append(
                    _PartialGraphEntry(
                        name=f"{block.name}.moe_router",
                        target=target,
                        fp8_enabled=False,
                    )
                )

            target = targets_by_feature.get("preprocess", {}).get(index)
            if target is not None:
                entries.append(
                    _PartialGraphEntry(
                        name=f"{block.name}.moe_preprocess",
                        target=target,
                        fp8_enabled=False,
                    )
                )

            target = targets_by_feature.get("experts", {}).get(index)
            if target is not None:
                from torch.distributed.tensor import DTensor

                experts = _find_te_ops_experts(block.module)
                assert experts is not None

                sharded_parameters = [
                    name for name, parameter in target.named_parameters() if isinstance(parameter, DTensor)
                ]
                parameter_names = [name for name, _parameter in target.named_parameters()]
                if sharded_parameters and len(sharded_parameters) != len(parameter_names):
                    raise RuntimeError(
                        "Partial expert CUDA graphs require either an entirely FSDP2-sharded or entirely plain "
                        f"TE-ops parameter set in {block.name}; sharded={sharded_parameters}, all={parameter_names}"
                    )
                expert_graph_storage = None
                if sharded_parameters:
                    from nemo_automodel.components.moe.fsdp2_graph_storage import FSDP2ExpertGraphStorage

                    expert_graph_storage = FSDP2ExpertGraphStorage(experts)
                else:
                    unallocated_parameters = [
                        name
                        for name, parameter in target.named_parameters()
                        if _PartialGraphEntry._parameter_state(parameter) is None
                    ]
                    if unallocated_parameters:
                        raise RuntimeError(
                            "Partial expert CUDA graphs require stable allocated parameter storage; "
                            f"found unallocated parameters in {block.name}: {unallocated_parameters}"
                        )
                entries.append(
                    _PartialGraphEntry(
                        name=f"{block.name}.te_ops_experts",
                        target=target,
                        fp8_enabled=backend.te_fp8 is not None,
                        canonicalizer=_canonicalize_te_ops_experts,
                        expert_bucket_tokens=getattr(
                            backend,
                            "partial_cuda_graph_expert_bucket_tokens",
                            None,
                        ),
                        expert_bucket_uses_paged_capacity=bool(
                            getattr(experts, "_te_ops_graph_uses_paged_capacity", False)
                        ),
                        expert_graph_storage=expert_graph_storage,
                    )
                )

        if not entries:
            if unsupported_expert_graphs:
                return None
            raise RuntimeError("Partial CUDA graphs were enabled but no graph targets were found")

        recipe = backend.te_fp8.build_recipe() if backend.te_fp8 is not None else None
        manager = cls(entries, fp8_recipe=recipe)
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
        if any(entry.fp8_enabled for entry in captured_entries) and self.fp8_recipe is None:
            raise RuntimeError("FP8 expert graph capture requires a Transformer Engine FP8 recipe")

        # Every rank in an expert FSDP group must issue the same unshard
        # collectives even when a rank received no iteration-0 expert tokens and
        # therefore has no local graph sample. Capture itself is local; skipped
        # ranks release their unused retained allocation after the synchronized
        # prepare/finish sequence.
        storage_entries = tuple(entry for entry in self.entries if entry.expert_graph_storage is not None)
        prepared_storage_entries = []
        try:
            for entry in storage_entries:
                entry.expert_graph_storage.prepare_before_capture()
                prepared_storage_entries.append(entry)

            # Give every guarded entry its own graph pool. Any entry may
            # deliberately fall back when metadata changes; sharing a pool would
            # require replaying all graphs in capture order and could corrupt
            # another entry's buffers when one dynamic expert call is skipped.
            for entry in captured_entries:
                adapter = entry.build_adapter()
                assert entry.captured_call is not None
                kwargs = {
                    "num_warmup_iters": 3,
                    # Every explicit expert parameter must produce a gradient.
                    # Fail capture instead of allowing TE to silently filter an
                    # alias and omit the corresponding live-parameter gradient.
                    "allow_unused_input": not isinstance(adapter, _ExplicitParameterCallAdapter),
                    "sample_kwargs": ({},),
                    "enabled": (entry.fp8_enabled,),
                }
                if self.fp8_recipe is not None:
                    kwargs["recipe"] = self.fp8_recipe
                try:
                    result = _get_make_graphed_callables()(
                        (adapter,),
                        (entry.capture_inputs(),),
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
                entry.install(result[0])
        finally:
            for entry in prepared_storage_entries:
                entry.expert_graph_storage.finish_capture()

        for entry in captured_entries:
            storage = entry.expert_graph_storage
            if storage is not None:
                logger.info(
                    "Retaining %.2f MiB of FSDP2 all-gather storage for partial expert graph %s",
                    storage.retained_bytes / (1024**2),
                    entry.name,
                )
        for entry in skipped_entries:
            storage = entry.expert_graph_storage
            if storage is not None:
                storage.reset()

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

    def expert_storage_stats(self) -> dict[str, int]:
        """Return resident FSDP2 all-gather storage retained by expert graphs."""
        storage_handles = tuple(
            entry.expert_graph_storage for entry in self.entries if entry.expert_graph_storage is not None
        )
        for storage in storage_handles:
            if storage.is_active:
                storage.validate_stable()
        return {
            "entries": len(storage_handles),
            "active": sum(int(storage.is_active) for storage in storage_handles),
            "retained_bytes": sum(storage.retained_bytes for storage in storage_handles),
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
        storage_stats = self.expert_storage_stats()
        if storage_stats["entries"]:
            logger.info(
                "Partial expert FSDP2 storage stats (%s): entries=%d active=%d retained_bytes=%d",
                phase,
                storage_stats["entries"],
                storage_stats["active"],
                storage_stats["retained_bytes"],
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

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

"""Rollout Routing Replay (R3) for MoE policy-gradient training.

In on-policy RL on a Mixture-of-Experts model, the rollout (inference) engine and
the training engine compute the router's top-k expert selection independently.
Numerical differences between the two backends flip a small fraction of routing
decisions per layer, which compounds across layers until most tokens are routed to
a different set of experts than they were during rollout. That mismatch breaks the
importance-sampling assumption behind GRPO/GSPO and destabilizes training.

Routing replay removes the mismatch by capturing the top-k expert *selection* during
one forward pass (the rollout-equivalent forward) and replaying that exact selection
during the training forward. Only the discrete selection is replayed: the router
logits and their softmax/sigmoid are still recomputed from the live router weights,
so the gradient continues to flow into the router. This mirrors Megatron-LM's
``moe_enable_routing_replay`` integration.

Usage::

    from nemo_automodel.components.moe.router_replay import RouterReplay

    # Capture the selection on the rollout-equivalent forward.
    with RouterReplay.record():
        model(batch)
    captured = RouterReplay.collect()  # one tensor per MoE layer, in layer order

    # Replay it on the training forward over the same tokens.
    with RouterReplay.replay(captured):
        loss = model(batch)
    loss.backward()

Each :class:`Gate` constructed with routing replay enabled owns one
:class:`RouterReplay` instance and registers it in a process-global list at
construction time. The global order is the construction order, which matches the
layer order, so ``collect()`` and ``replay()`` line the per-layer tensors up by
position. This assumes single-threaded model construction (the norm for recipe
training); call :meth:`RouterReplay.clear_registry` before building a second
model in the same process.

For rollout-provided routing, :class:`RouterReplayAdapter` maps global
decoder-layer ids without using the registry. Callers prepare routes in the
same token order as the model input, then keep the adapter's replay context
active through forward and backward.
"""

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from enum import Enum
from math import prod

import torch
from torch import nn

from nemo_automodel.shared.model_utils import iter_transformer_blocks

__all__ = ["RouterReplayMode", "RouterReplay", "RouterReplayAdapter", "replay_selection"]


class RouterReplayMode(Enum):
    """Active mode of a :class:`RouterReplay` instance."""

    RECORD = "record"  # Store the freshly computed top-k selection for later replay.
    REPLAY = "replay"  # Override the freshly computed selection with the stored one.


class RouterReplay:
    """Per-gate handle that records or replays a single MoE layer's top-k selection.

    Instances register themselves in a process-global list on construction. The
    static helpers drive every registered instance at once so a caller toggles
    record/replay for the whole model with a single call (or the ``record`` /
    ``replay`` context managers).
    """

    _registry: list["RouterReplay"] = []

    def __init__(self, *, register: bool = True) -> None:
        """Create a handle, optionally registering it for legacy global control."""
        self.mode: RouterReplayMode | None = None
        self.recorded_indices: torch.Tensor | None = None
        self.target_indices: torch.Tensor | None = None
        self._allow_trailing_live_tokens = False
        if register:
            RouterReplay._registry.append(self)

    def apply(self, indices: torch.Tensor) -> torch.Tensor:
        """Record or replay ``indices`` according to the current mode.

        Args:
            indices: The top-k expert indices the gate just selected, shape
                ``[num_tokens, topk]``.

        Returns:
            ``indices`` unchanged when no mode is active or while recording; the
            stored target indices (moved to ``indices.device``) while replaying.
            A target row containing ``-1`` keeps that token's complete live
            top-k selection, preserving unique expert ids.
        """
        if self.mode == RouterReplayMode.RECORD:
            # Indices are integer selection ids carrying no gradient; detach so the
            # capture never pins the forward graph.
            self.recorded_indices = indices.detach()
            return indices
        if self.mode == RouterReplayMode.REPLAY:
            if self.target_indices is None:
                raise RuntimeError(
                    "RouterReplay is in REPLAY mode but no target indices were set for this layer. "
                    "Call RouterReplay.replay(indices) / set_replay_indices(...) with one tensor per MoE layer."
                )
            if self.target_indices.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
                raise TypeError(
                    f"RouterReplay target indices must use a signed integer dtype, got {self.target_indices.dtype}"
                )
            target = self.target_indices.to(device=indices.device, dtype=indices.dtype)
            if target.shape != indices.shape:
                if (
                    self._allow_trailing_live_tokens
                    and target.ndim == 2
                    and indices.ndim == 2
                    and target.shape[1] == indices.shape[1]
                    and target.shape[0] < indices.shape[0]
                ):
                    trailing = target.new_full((indices.shape[0] - target.shape[0], target.shape[1]), -1)
                    target = torch.cat((target, trailing), dim=0)
                else:
                    raise ValueError(
                        f"Replay indices shape {tuple(target.shape)} does not match the current "
                        f"selection shape {tuple(indices.shape)}; replay must run on the same tokens and topk."
                    )
            keep_live = (target == -1).any(dim=-1, keepdim=True)
            return torch.where(keep_live, indices, target)
        return indices

    # -- per-instance state -------------------------------------------------

    def set_target(self, indices: torch.Tensor) -> None:
        """Set the selection to replay for this layer."""
        self.target_indices = indices

    def clear(self) -> None:
        """Drop both the recorded and the target selection for this layer."""
        self.recorded_indices = None
        self.target_indices = None

    # -- global control over every registered instance ---------------------

    @staticmethod
    def instances() -> list["RouterReplay"]:
        """Return the registered instances in construction (layer) order."""
        return RouterReplay._registry

    @staticmethod
    def set_mode(mode: RouterReplayMode | None) -> None:
        """Set the mode on every registered instance (``None`` disables replay)."""
        for inst in RouterReplay._registry:
            inst.mode = mode

    @staticmethod
    def set_replay_indices(all_layers_indices: list[torch.Tensor]) -> None:
        """Distribute one selection tensor per layer to the registered instances.

        Args:
            all_layers_indices: One ``[num_tokens, topk]`` tensor per MoE layer, in
                the same order the layers were constructed.

        Raises:
            ValueError: If the number of tensors does not match the number of
                registered instances.
        """
        instances = RouterReplay._registry
        if len(all_layers_indices) != len(instances):
            raise ValueError(
                f"Got {len(all_layers_indices)} replay tensors but there are {len(instances)} "
                "registered RouterReplay instances (one per MoE layer)."
            )
        for inst, indices in zip(instances, all_layers_indices):
            inst.set_target(indices)

    @staticmethod
    def collect() -> list[torch.Tensor]:
        """Collect the recorded selection from every registered instance, in layer order.

        Raises:
            RuntimeError: If any instance has no recorded selection (i.e. a forward
                pass was not run under :meth:`record`).
        """
        collected: list[torch.Tensor] = []
        for layer_idx, inst in enumerate(RouterReplay._registry):
            if inst.recorded_indices is None:
                raise RuntimeError(
                    f"RouterReplay instance for layer {layer_idx} has no recorded selection; "
                    "run a forward pass inside `with RouterReplay.record():` before collecting."
                )
            collected.append(inst.recorded_indices)
        return collected

    @staticmethod
    def clear_indices() -> None:
        """Drop recorded and target selections on every registered instance."""
        for inst in RouterReplay._registry:
            inst.clear()

    @staticmethod
    def clear_registry() -> None:
        """Forget every registered instance (use between independently built models)."""
        RouterReplay._registry.clear()

    # -- ergonomic context managers ----------------------------------------

    @classmethod
    @contextmanager
    def record(cls) -> Iterator[None]:
        """Record the top-k selection of every gate for the duration of the block."""
        cls.set_mode(RouterReplayMode.RECORD)
        try:
            yield
        finally:
            cls.set_mode(None)

    @classmethod
    @contextmanager
    def replay(cls, all_layers_indices: list[torch.Tensor]) -> Iterator[None]:
        """Replay ``all_layers_indices`` (one tensor per layer) for the duration of the block.

        Target selections are cleared on exit so a stale replay never leaks into a
        later forward pass.
        """
        cls.set_replay_indices(all_layers_indices)
        cls.set_mode(RouterReplayMode.REPLAY)
        try:
            yield
        finally:
            cls.set_mode(None)
            for inst in cls._registry:
                inst.target_indices = None


@dataclass(frozen=True)
class _RouterReplayBinding:
    """One decoder layer's model-scoped replay handle."""

    layer_idx: int
    replay: RouterReplay
    topk: int


class RouterReplayAdapter:
    """Bind rollout routes to model-scoped MoE gates.

    The adapter deliberately ignores the legacy process-global registry:
    decoder-layer ids determine the mapping, so sparse hybrid MoE stacks and
    multiple models in one process remain unambiguous. Do not nest the legacy
    process-global ``record``/``replay`` contexts around an active adapter
    context when the same gate handles are registered.

    Args:
        model: One complete model. Primary decoder blocks must expose numeric
            child ids or a consistent integer ``layer_idx``. A block may
            contain at most one module with a ``router_replay`` slot.
    """

    def __init__(self, model: nn.Module) -> None:
        # Descend through .module wrappers (e.g. an Engine) until decoder
        # blocks are visible.
        block_root = model
        while True:
            blocks = tuple(iter_transformer_blocks(block_root))
            if blocks:
                break
            wrapped = getattr(block_root, "module", None)
            if not isinstance(wrapped, nn.Module):
                break
            block_root = wrapped

        bindings: list[_RouterReplayBinding] = []
        seen_replays: set[int] = set()
        for _parent, child_name, block in blocks:
            slots = [module for module in block.modules() if hasattr(module, "router_replay")]
            if not slots:
                continue
            if len(slots) != 1:
                raise ValueError(
                    f"decoder block {child_name!r} has {len(slots)} router_replay slots; "
                    "RouterReplayAdapter requires one gate per routed layer"
                )

            declared_ids = {
                layer_idx
                for module in block.modules()
                if isinstance((layer_idx := getattr(module, "layer_idx", None)), int)
                and not isinstance(layer_idx, bool)
            }
            if len(declared_ids) > 1:
                raise ValueError(
                    f"decoder block {child_name!r} contains conflicting layer_idx values {sorted(declared_ids)}"
                )
            child_idx = int(child_name) if child_name.isdecimal() else None
            declared_idx = next(iter(declared_ids), None)
            if child_idx is not None and declared_idx is not None and child_idx != declared_idx:
                raise ValueError(f"decoder block key {child_idx} disagrees with its layer_idx {declared_idx}")
            layer_idx = declared_idx if declared_idx is not None else child_idx
            if layer_idx is None:
                raise ValueError(f"cannot resolve the global layer id for routed decoder block {child_name!r}")
            if layer_idx < 0:
                raise ValueError(f"routed decoder block {child_name!r} has negative layer_idx {layer_idx}")

            gate = slots[0]
            topk = getattr(gate, "topk", None)
            if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
                raise ValueError(f"decoder block {layer_idx} replay gate must expose a positive integer topk")
            if getattr(gate, "use_routing_core", False):
                raise RuntimeError(
                    "RouterReplayAdapter is incompatible with partial MoE router CUDA graphs; "
                    "disable the 'moe_router' graph module before enabling routing replay"
                )
            replay = gate.router_replay
            if replay is None:
                replay = RouterReplay(register=False)
                gate.router_replay = replay
            if not isinstance(replay, RouterReplay):
                raise TypeError(
                    f"decoder block {layer_idx} router_replay must be RouterReplay or None, "
                    f"got {type(replay).__name__}"
                )
            if id(replay) in seen_replays:
                raise ValueError("one RouterReplay handle is attached to more than one decoder block")
            seen_replays.add(id(replay))
            # Replayed routes cover only the live tokens the caller prepared;
            # trailing padded tokens keep live routing.
            replay._allow_trailing_live_tokens = True
            bindings.append(_RouterReplayBinding(layer_idx, replay, topk))

        if not bindings:
            raise ValueError("RouterReplayAdapter found no MoE gate with a router_replay slot in the primary decoder")
        bindings.sort(key=lambda binding: binding.layer_idx)
        layer_ids = [binding.layer_idx for binding in bindings]
        if len(set(layer_ids)) != len(layer_ids):
            raise ValueError(f"multiple replay gates map to the same global decoder layer: {layer_ids}")
        topks = {binding.topk for binding in bindings}
        if len(topks) != 1:
            raise ValueError(f"all replay gates must use one topk, got {sorted(topks)}")
        self._bindings = tuple(bindings)
        self._layer_ids = tuple(layer_ids)
        self._topk = next(iter(topks))

    @property
    def layer_ids(self) -> tuple[int, ...]:
        """Global decoder-layer ids, in the model's replay order."""
        return self._layer_ids

    def replay(self, prepared_routes: torch.Tensor | None) -> AbstractContextManager[None]:
        """Replay routes prepared in the model input's token order.

        Args:
            prepared_routes: Signed integer expert ids with arbitrary token
                axes followed by ``[global_layers, topk]``. Its flattened token
                order must match the model input after any padding, packing, or
                context-parallel sharding. ``-1`` keeps a token's complete live
                top-k selection. ``None`` selects live routing.

        Returns:
            A context that replays this model's layer targets through forward,
            backward, and activation-checkpoint recomputation.
        """
        if prepared_routes is None:
            return nullcontext()
        if not isinstance(prepared_routes, torch.Tensor):
            raise TypeError("prepared_routes must be a Tensor or None")
        if prepared_routes.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError(f"prepared_routes must use a signed integer dtype, got {prepared_routes.dtype}")
        if prepared_routes.ndim < 3:
            raise ValueError("prepared_routes must have token axes followed by [global_layers, topk]")

        route_tokens = prod(prepared_routes.shape[:-2])
        num_layers, route_topk = prepared_routes.shape[-2:]
        max_layer_idx = self._bindings[-1].layer_idx
        if num_layers <= max_layer_idx:
            raise ValueError(
                f"prepared_routes has {num_layers} global layers but this model requires layer {max_layer_idx}"
            )
        per_token = prepared_routes.reshape(route_tokens, num_layers, route_topk)
        if route_topk != self._topk:
            raise ValueError(f"prepared_routes topk {route_topk} does not match model topk {self._topk}")
        layer_indices = torch.tensor(self.layer_ids, device=per_token.device, dtype=torch.long)
        selected = per_token.index_select(1, layer_indices)
        targets = list(selected.unbind(dim=1))
        return self._activate(targets)

    @contextmanager
    def _activate(self, targets: list[torch.Tensor]) -> Iterator[None]:
        """Temporarily install one ``[tokens, topk]`` target per binding.

        Args:
            targets: Model-scoped replay targets in ``self._bindings`` order.
                Every tensor has shape ``[tokens, topk]``.

        Yields:
            ``None`` while replay is active. Previous handle state is restored
            on normal exit or exception.
        """
        previous = [(binding.replay.mode, binding.replay.target_indices) for binding in self._bindings]
        try:
            for binding, target in zip(self._bindings, targets):
                binding.replay.target_indices = target
                binding.replay.mode = RouterReplayMode.REPLAY
            yield
        finally:
            for binding, (mode, target) in zip(self._bindings, previous):
                binding.replay.mode = mode
                binding.replay.target_indices = target


def replay_selection(router_replay: RouterReplay | None, indices: torch.Tensor) -> torch.Tensor:
    """Route ``indices`` through ``router_replay`` when routing replay is enabled.

    Returns ``indices`` unchanged when ``router_replay`` is ``None`` (replay disabled)
    or when no mode is active, so the gate's default path is a true no-op.
    """
    if router_replay is None:
        return indices
    return router_replay.apply(indices)

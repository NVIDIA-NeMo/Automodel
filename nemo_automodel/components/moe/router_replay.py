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
"""

from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, List, Optional

import torch

__all__ = ["RouterReplayMode", "RouterReplay", "replay_selection"]


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

    _registry: List["RouterReplay"] = []

    def __init__(self) -> None:
        """Create a handle and register it in construction (i.e. layer) order."""
        self.mode: Optional[RouterReplayMode] = None
        self.recorded_indices: Optional[torch.Tensor] = None
        self.target_indices: Optional[torch.Tensor] = None
        self.native_indices: Optional[torch.Tensor] = None
        self.effective_indices: Optional[torch.Tensor] = None
        self.apply_count = 0
        self.selection_observer: Callable[[torch.Tensor, torch.Tensor], None] | None = None
        RouterReplay._registry.append(self)

    def apply(self, indices: torch.Tensor) -> torch.Tensor:
        """Record or replay ``indices`` according to the current mode.

        Args:
            indices: The top-k expert indices the gate just selected, shape
                ``[num_tokens, topk]``.

        Returns:
            ``indices`` unchanged when no mode is active or while recording; the
            stored target indices (moved to ``indices.device``) while replaying.
            A target row containing only ``-1`` uses the native selection from
            the first forward. The resulting effective selection is cached and
            reused by activation-checkpoint recomputation.
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
            target = self.target_indices.to(indices.device)
            if target.shape != indices.shape:
                raise ValueError(
                    f"Replay indices shape {tuple(target.shape)} does not match the current "
                    f"selection shape {tuple(indices.shape)}; replay must run on the same tokens and topk."
                )
            negative = target < 0
            missing = negative.all(dim=-1)
            partial_missing = negative.any(dim=-1) & ~missing
            if bool(partial_missing.any().item()):
                row = int(partial_missing.nonzero()[0].item())
                raise ValueError(
                    "Replay target rows must contain either all non-negative expert IDs or the all--1 "
                    f"missing-route sentinel; got {target[row].detach().cpu().tolist()} at row {row}."
                )
            invalid_missing = missing & ~target.eq(-1).all(dim=-1)
            if bool(invalid_missing.any().item()):
                row = int(invalid_missing.nonzero()[0].item())
                raise ValueError(
                    "Replay missing-route rows must use the all--1 sentinel; "
                    f"got {target[row].detach().cpu().tolist()} at row {row}."
                )

            if self.effective_indices is None:
                self.native_indices = indices.detach().clone()
                self.effective_indices = torch.where(missing.unsqueeze(-1), indices, target).detach().clone()
            self.apply_count += 1
            return self.effective_indices.to(indices.device)
        return indices

    # -- per-instance state -------------------------------------------------

    def set_target(self, indices: torch.Tensor) -> None:
        """Set the selection to replay for this layer.

        Args:
            indices: Integer tensor of shape [tokens, topk]. Each row contains
                global expert IDs, or all ``-1`` for native fallback.
        """
        self.target_indices = indices
        self.native_indices = None
        self.effective_indices = None
        self.apply_count = 0

    def clear(self) -> None:
        """Drop both the recorded and the target selection for this layer."""
        self.recorded_indices = None
        self.target_indices = None
        self.native_indices = None
        self.effective_indices = None
        self.apply_count = 0

    @contextmanager
    def replay_target(
        self,
        indices: torch.Tensor,
        selection_observer: Callable[[torch.Tensor, torch.Tensor], None] | None = None,
    ) -> Iterator[None]:
        """Replay one layer's target without changing other registered models.

        Args:
            indices: Integer tensor of shape [tokens, topk]. Each row contains
                global expert IDs, or all ``-1`` for native fallback.
            selection_observer: Optional observer of native and returned
                selections, both integer [tokens, topk] tensors. Called at the
                Gate boundary independently of the replay implementation.

        Yields:
            Control while this handle is in replay mode. The target and cached
            effective selection remain alive through backward recomputation.
        """
        previous = (
            self.mode,
            self.target_indices,
            self.native_indices,
            self.effective_indices,
            self.apply_count,
            self.selection_observer,
        )
        self.set_target(indices)
        self.mode = RouterReplayMode.REPLAY
        self.selection_observer = selection_observer
        try:
            yield
        finally:
            (
                self.mode,
                self.target_indices,
                self.native_indices,
                self.effective_indices,
                self.apply_count,
                self.selection_observer,
            ) = previous

    # -- global control over every registered instance ---------------------

    @staticmethod
    def instances() -> List["RouterReplay"]:
        """Return the registered instances in construction (layer) order."""
        return RouterReplay._registry

    @staticmethod
    def set_mode(mode: Optional[RouterReplayMode]) -> None:
        """Set the mode on every registered instance (``None`` disables replay)."""
        for inst in RouterReplay._registry:
            inst.mode = mode

    @staticmethod
    def set_replay_indices(all_layers_indices: List[torch.Tensor]) -> None:
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
    def collect() -> List[torch.Tensor]:
        """Collect the recorded selection from every registered instance, in layer order.

        Raises:
            RuntimeError: If any instance has no recorded selection (i.e. a forward
                pass was not run under :meth:`record`).
        """
        collected: List[torch.Tensor] = []
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
    def replay(cls, all_layers_indices: List[torch.Tensor]) -> Iterator[None]:
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
                inst.native_indices = None
                inst.effective_indices = None
                inst.apply_count = 0


def replay_selection(router_replay: Optional[RouterReplay], indices: torch.Tensor) -> torch.Tensor:
    """Route ``indices`` through ``router_replay`` when routing replay is enabled.

    Returns ``indices`` unchanged when ``router_replay`` is ``None`` (replay disabled)
    or when no mode is active, so the gate's default path is a true no-op.
    """
    if router_replay is None:
        return indices
    # Snapshot the independently selected route before replay can change it.
    observer = router_replay.selection_observer
    native = indices.detach().clone() if observer is not None else None
    effective = router_replay.apply(indices)
    if observer is not None:
        observer(native, effective)
    return effective

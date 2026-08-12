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

"""Keeping the two streams' MoE routing aligned.

The lateral term ``y_b = W_b x_b + y_a`` is per token, and an expanded linear inside a MoE
expert only ever sees the tokens routed to that expert. If the two streams route
differently, the expert receives a different number of tokens on each pass and the
recorded ``y_a`` cannot be added to ``y_b`` at all -- it fails loudly on a shape mismatch
rather than corrupting silently, but it fails.

The streams do route differently as soon as they diverge, because the router reads the
hidden state and the whole point of the expansion is that stream B's hidden state moves.

``RouterReplay`` already solves this, for a different reason: it exists so that on-policy
RL can replay the rollout's expert selection during the training forward. The same
mechanism pins stream B to stream A's selection, which makes the dispatch order identical
and the lateral applicable elementwise. Only the discrete selection is replayed -- each
stream still computes its own routing weights from its own hidden state -- so stream B is
not forced to be stream A, only to visit the same experts.

Coexistence with RL is the reason this is written as save/restore rather than as ownership.
When an outer ``RouterReplay.replay(...)`` is already active, both streams are pinned to
the rollout's selection, which satisfies RL and the expansion at once, so this code stays
out of the way.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

import torch.nn as nn

from nemo_automodel.shared.import_utils import safe_import

HAVE_ROUTER_REPLAY, _router_replay = safe_import("nemo_automodel.components.moe.router_replay")

__all__ = ["aligned_routing", "find_router_replays", "requires_routing_replay", "switch_to_replay"]


def find_router_replays(layer: nn.Module) -> list:
    """Collect the ``RouterReplay`` handles owned by a decoder layer's gates.

    Looked up by traversal rather than by index into ``RouterReplay``'s process-global
    registry: the registry is ordered by construction, which says nothing about which
    handle belongs to the layer being expanded.

    Args:
        layer: A decoder layer, possibly containing one or more  MoE gates.

    Returns:
        The layer's ``RouterReplay`` handles, empty if it has no MoE gate or if routing
        replay was not enabled when the model was built.
    """
    if not HAVE_ROUTER_REPLAY:
        return []
    handles = []
    for module in layer.modules():
        handle = getattr(module, "router_replay", None)
        if handle is not None:
            handles.append(handle)
    return handles


@contextmanager
def aligned_routing(handles: list) -> Iterator[None]:
    """Make the enclosed stream-B pass reuse the stream-A pass's expert selection.

    Enter this around the pair of passes. On entry the handles are put in record mode so
    the A pass captures its selection; :func:`switch_to_replay` promotes them to replay
    before the B pass; on exit the previous mode is restored.

    Args:
        handles: The layer's ``RouterReplay`` handles, from :func:`find_router_replays`.

    Yields:
        None.
    """
    previous = [handle.mode for handle in handles]
    externally_driven = _externally_driven(handles)
    if not externally_driven:
        for handle in handles:
            handle.mode = _router_replay.RouterReplayMode.RECORD
    try:
        yield
    finally:
        for handle, mode in zip(handles, previous):
            handle.mode = mode


def switch_to_replay(handles: list) -> None:
    """Promote recorded selections to replay targets, between the two passes.

    Args:
        handles: The layer's ``RouterReplay`` handles.
    """
    if _externally_driven(handles):
        # An outer RL replay already pins both streams to the rollout's selection, which
        # is exactly the alignment needed here. Overwriting its target would break it.
        return
    for handle in handles:
        handle.set_target(handle.recorded_indices)
        handle.mode = _router_replay.RouterReplayMode.REPLAY


def _externally_driven(handles: list) -> bool:
    """Whether an outer caller (RL routing replay) already owns these handles."""
    if not HAVE_ROUTER_REPLAY:
        return False
    return any(handle.mode is _router_replay.RouterReplayMode.REPLAY for handle in handles)


def requires_routing_replay(layer: nn.Module) -> Optional[str]:
    """Explain why a layer cannot be expanded, if it cannot.

    Args:
        layer: A decoder layer about to be expanded.

    Returns:
        A message naming the problem, or ``None`` if the layer can be expanded. A MoE layer
        built without routing replay cannot: its two passes would route independently and
        the lateral term would not line up.
    """
    has_gate = any(hasattr(module, "router_replay") for module in layer.modules())
    if not has_gate:
        return None
    if find_router_replays(layer):
        return None
    return (
        "this layer has a MoE gate but no RouterReplay handle, so the two expansion "
        "streams would route independently and their lateral connection would not line "
        "up. Build the model with enable_routing_replay=True to expand a MoE layer."
    )

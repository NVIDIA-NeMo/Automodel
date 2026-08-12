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

"""The linear layer that carries a dual-stream expansion weight.

Model expansion grows an already-pretrained model by giving selected linear layers a
second weight. The pretrained weight stays frozen and only the new one trains, so the
expanded model starts as an exact copy of its parent and departs from it only as the new
weights learn.

Within an expanded decoder layer two hidden-state streams are computed. Stream A is the
pretrained computation, untouched. Stream B is the expanded one, and at every expanded
linear it receives stream A's output as a lateral term::

    y_a = W_a @ x_a                 (frozen)
    y_b = W_b @ x_b + y_a           (trainable)

Adding the lateral on the *output* side is the only formulation that type-checks for the
non-square projections (fused QKV, gate/up), so it is used for all of them.

The two streams are computed by running the enclosing decoder layer twice -- see
``dual_stream.py`` -- so this module never sees both at once. It instead records ``y_a``
on the A pass and consumes it on the B pass, driven by :class:`LateralBus`. That mirrors
how ``nemo_automodel.components.moe.router_replay.RouterReplay`` records a routing
decision on one pass and replays it on the next.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LateralBusMode", "LateralBus", "ExpandedLinear", "patch_linear_for_expansion"]


class LateralBusMode(Enum):
    """Which stream the enclosing decoder layer is currently running."""

    RECORD = "record"  # stream A: compute with the frozen weight and stash the result
    APPLY = "apply"  # stream B: add the stashed result to the expansion weight's output


class LateralBus:
    """Process-global switch naming the stream currently being computed.

    A module-level switch rather than an argument because the signal has to reach every
    expanded linear nested anywhere inside a decoder layer whose ``forward`` this code
    does not own.
    """

    mode: LateralBusMode | None = None

    @classmethod
    def set_mode(cls, mode: LateralBusMode | None) -> LateralBusMode | None:
        """Set the active mode and return the previous one, for save/restore."""
        previous = cls.mode
        cls.mode = mode
        return previous


class ExpandedLinear(nn.Linear):
    """An ``nn.Linear`` carrying a second, trainable weight of the same shape.

    Subclassing ``nn.Linear`` in place rather than wrapping it is load-bearing, not a
    style choice:

    * tensor-parallel plans match on module *paths* (``model.layers.*.self_attn.q_proj``)
      and match segment-wise, so an inserted wrapper level stops the pattern matching;
    * ``to_hf`` / ``from_hf`` match on state-dict keys, so moving the pretrained weight to
      ``q_proj.base.weight`` breaks checkpoint conversion;
    * modules built under a meta device have no allocated weight to copy out of.

    ``nemo_automodel.components._peft.lora.patch_linear_module`` patches in place for the
    same reasons.

    Allocating the expansion weight and giving it a value are separate steps, because the
    two happen at different points in a training run. Allocation has to precede sharding,
    since the tensor-parallel plan and ``fully_shard`` are what distribute the new weight;
    initialization has to follow the checkpoint load, since it reads the pretrained weight
    it is copying. On a single process those two moments coincide and
    :func:`patch_linear_for_expansion` does both at once, which is why ``initialize``
    defaults to true.

    Attributes:
        expansion: The trainable weight, same shape as ``self.weight``, no bias. The bias
            reaches stream B through the lateral term, which is what keeps a zero
            expansion weight exactly equivalent to an unexpanded layer.
        zero_init: Whether :meth:`initialize_expansion` starts this weight at zero rather
            than as a copy of the pretrained one. Recorded at allocation because the
            decision is made there and acted on later.
    """

    expansion: nn.Linear
    zero_init: bool

    def allocate_expansion(self, zero_init: bool) -> None:
        """Give an already-constructed linear its expansion weight, with no value yet.

        Safe on a meta-device model: nothing is read from the pretrained weight here.

        Args:
            zero_init: Start the expansion weight at zero instead of copying the
                pretrained weight. Used for the projections that write into the residual
                stream (``o_proj``, ``down_proj``); it is what makes the expanded model
                reproduce its parent exactly before any optimizer step. Applied by
                :meth:`initialize_expansion`.
        """
        self.expansion = nn.Linear(self.in_features, self.out_features, bias=False)
        self.expansion.to(device=self.weight.device, dtype=self.weight.dtype)
        self.zero_init = zero_init
        self._lateral: torch.Tensor | None = None

    def initialize_expansion(self) -> None:
        """Set the expansion weight from the pretrained one.

        Call after the pretrained weights are materialized and loaded. Sharding in between
        is fine: once both weights are distributed the copy is a local operation on each
        rank's shard.

        Raises:
            RuntimeError: If the pretrained weight is still on the meta device, or if only
                one of the two weights is distributed, which means expansion was applied
                somewhere in the middle of sharding.
        """
        # DTensor is detected by class name so this module stays importable without
        # torch.distributed, the same test `checkpointing.py` uses.
        if self.weight.device.type == "meta":
            raise RuntimeError(
                "Cannot initialize an expansion weight from a pretrained weight that is "
                "still on the meta device: the copy would be a silent no-op and the "
                "expansion weight would later be filled with arbitrary memory. Nothing "
                "downstream would notice, because the zero-initialized output projections "
                "discard stream B until training starts. Initialize expansion after the "
                "pretrained weights are materialized and loaded."
            )
        if (type(self.weight).__name__ == "DTensor") != (type(self.expansion.weight).__name__ == "DTensor"):
            raise RuntimeError(
                "Cannot initialize an expansion weight when it and the pretrained weight "
                "disagree about being distributed: the pretrained weight is "
                f"{'already distributed' if type(self.weight).__name__ == 'DTensor' else 'local'} "
                f"and the expansion weight is "
                f"{'distributed' if type(self.expansion.weight).__name__ == 'DTensor' else 'local'}. "
                "Allocate the expansion weight before the model is parallelized so that "
                "both are distributed together."
            )
        with torch.no_grad():
            if self.zero_init:
                self.expansion.weight.zero_()
            else:
                self.expansion.weight.copy_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the pretrained weight, the expansion weight, or both.

        Args:
            input: ``[*, in_features]``. The leading dimensions are whatever the enclosing
                model uses -- ``[batch, seq, in_features]`` for a padded batch, or a
                flattened ``[tokens, in_features]`` inside a MoE expert. This layer is
                elementwise over those leading dimensions and does not interpret them.

        Returns:
            ``[*, out_features]``, with the same leading dimensions as ``input``.
        """
        if LateralBus.mode is LateralBusMode.RECORD:
            output = F.linear(input, self.weight, self.bias)
            self._lateral = output
            return output
        if LateralBus.mode is LateralBusMode.APPLY:
            if self._lateral is None:
                raise RuntimeError(
                    f"{type(self).__name__} ran a stream-B pass with no recorded stream-A "
                    "output. The two passes must alternate; check that nothing reset the "
                    "lateral bus between them."
                )
            lateral, self._lateral = self._lateral, None
            return self.expansion(input) + lateral
        return F.linear(input, self.weight, self.bias)


def patch_linear_for_expansion(linear: nn.Linear, zero_init: bool, initialize: bool = True) -> ExpandedLinear:
    """Give an existing ``nn.Linear`` an expansion weight, in place.

    Args:
        linear: The pretrained linear to expand. Modified in place; its identity, module
            path and ``weight`` are preserved.
        zero_init: See :meth:`ExpandedLinear.allocate_expansion`.
        initialize: Also give the expansion weight its value. Pass ``False`` when the
            pretrained weights are not loaded yet, then call
            :meth:`ExpandedLinear.initialize_expansion` once they are.

    Returns:
        The same object, now an :class:`ExpandedLinear`.
    """
    linear.__class__ = ExpandedLinear
    linear.allocate_expansion(zero_init)
    if initialize:
        linear.initialize_expansion()
    return linear

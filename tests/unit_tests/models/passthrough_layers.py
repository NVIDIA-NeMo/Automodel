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

"""Run a causal LM with identity decoder layers, to test packed (THD) plumbing without TE kernels."""

import contextlib
from collections.abc import Iterable, Iterator
from typing import Any
from unittest.mock import patch

import torch
from torch import nn


@contextlib.contextmanager
def passthrough_layers(layers: Iterable[nn.Module]) -> Iterator[list[tuple[int, ...]]]:
    """Make every one of ``layers`` return its input while the context is open.

    Args:
        layers: Decoder layers whose first argument, or keyword ``x``, holds their hidden states.

    Yields:
        The shape of the hidden states each layer received, in call order, filled in as the layers run.
    """
    shapes: list[tuple[int, ...]] = []

    def passthrough(*layer_args: Any, **layer_kwargs: Any) -> torch.Tensor:
        """Record and return a layer's hidden states.

        Args:
            *layer_args: The layer's positional arguments; the first is its hidden states, a tensor of shape
                [batch, sequence, hidden] or packed [tokens, hidden].
            **layer_kwargs: The layer's keyword arguments; ``x`` holds those hidden states when none is positional.

        Returns:
            The hidden states: Tensor of shape [batch, sequence, hidden], or packed [tokens, hidden].
        """
        hidden = layer_args[0] if layer_args else layer_kwargs["x"]
        shapes.append(tuple(hidden.shape))
        return hidden

    with contextlib.ExitStack() as stack:
        for layer in layers:
            stack.enter_context(patch.object(layer, "forward", side_effect=passthrough))
        yield shapes


def run_with_passthrough_layers(model: nn.Module, *args: Any, **kwargs: Any) -> tuple[Any, list[tuple[int, ...]]]:
    """Call ``model`` with every layer of its decoder returning its input.

    Args:
        model: Causal LM whose decoder ``model.model`` keeps its layers in the ``layers`` ModuleDict.
        *args: Positional arguments of the model's forward.
        **kwargs: Keyword arguments of the model's forward.

    Returns:
        The model's output, and the shape of the hidden states each layer received, in call order.
    """
    with passthrough_layers(model.model.layers.values()) as shapes:
        return model(*args, **kwargs), shapes

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

"""Explicit FP32-scoring reference for the HF Mistral4 router."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from functools import wraps
from typing import Any

import torch
from torch.overrides import TorchFunctionMode

from nemo_automodel.shared.import_utils import safe_import


class _FP32RouterSoftmax(TorchFunctionMode):
    """Change only the dtype argument of the router's native HF softmax."""

    def __init__(self) -> None:
        self.calls = 0

    def __torch_function__(
        self, func: Callable, types: tuple[type, ...], args: tuple = (), kwargs: dict | None = None
    ) -> Any:
        """Dispatch native operations, promoting the router softmax.

        Args:
            func: Original torch operation.
            types: Tensor types supplied by torch's dispatch protocol.
            args: Framework operands of arbitrary layouts. The intercepted softmax
                receives router logits of shape [tokens, experts] and its axis.
            kwargs: Original operation keywords, including an optional dtype.

        Returns:
            The native operation's result with its original layout. Router softmax
            returns FP32 probabilities of shape [tokens, experts]; all other
            operations retain their native dispatch and dtype behavior.
        """
        kwargs = dict(kwargs or {})
        if func is torch.Tensor.softmax:
            kwargs["dtype"] = torch.float32
            self.calls += 1
        return func(*args, **kwargs)


def _with_fp32_scores(forward: Callable) -> Callable:
    @wraps(forward)
    def wrapped(hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the original HF router with FP32 score arithmetic.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading
                dimensions and the native router projection's input dtype.

        Returns:
            Native projection logits of shape [tokens, experts], FP32 normalized
            routing weights of shape [tokens, top_k], and integer expert indices
            of shape [tokens, top_k]. Tokens flatten the input's leading axes.
        """
        with _FP32RouterSoftmax() as mode:
            result = forward(hidden_states)
        if mode.calls != 1:
            raise RuntimeError(f"HF Mistral4 router softmax contract changed: expected one call, got {mode.calls}")
        return result

    return wrapped


@contextmanager
def fp32_router_scores(model: torch.nn.Module) -> Iterator[None]:
    """Temporarily use FP32 scoring in HF Mistral4 without changing its weights.

    This is an explicitly modified reference, not vanilla HF. Projection retains
    its native dtype; softmax, selection, normalization, and returned weights use
    FP32. The original HF algorithm is executed rather than copied. Only this
    model's router instances are wrapped, including any existing device-map
    dispatch wrappers. All instance overrides are restored on exit or failure.

    Args:
        model: Loaded HF model containing Mistral4TopkRouter modules.

    Yields:
        Control to reference forwards using the selected scoring precision.
    """
    available, hf_module = safe_import("transformers.models.mistral4.modeling_mistral4")
    if not available:
        raise ImportError("FP32 Mistral4 reference scoring requires Transformers with Mistral4 support")
    routers = [module for module in model.modules() if isinstance(module, hf_module.Mistral4TopkRouter)]
    if not routers:
        raise ValueError("FP32 Mistral4 reference scoring found no HF Mistral4 routers")
    with ExitStack() as stack:
        for router in routers:
            original = router.forward
            if "forward" in router.__dict__:
                stack.callback(setattr, router, "forward", original)
            else:
                stack.callback(delattr, router, "forward")
            router.forward = _with_fp32_scores(original)
        yield

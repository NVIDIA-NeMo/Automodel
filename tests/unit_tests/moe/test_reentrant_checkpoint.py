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

import warnings

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
)

from nemo_automodel.components.moe.parallelizer import apply_ac


class _InnerExpert(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, events):
        ctx.events = events
        ctx.save_for_backward(value)
        events.append("inner-forward")
        return value.square()

    @staticmethod
    def backward(ctx, grad_output):
        (value,) = ctx.saved_tensors
        ctx.events.append("inner-backward")
        return grad_output * 2 * value, None


class _ZeroTokenDummy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, events):
        ctx.events = events
        events.append("zero-token-dummy-forward")
        return value * 0

    @staticmethod
    def backward(ctx, grad_output):
        ctx.events.append("zero-token-dummy-backward")
        return grad_output * 0, None


class _OrderedBlock(nn.Module):
    def __init__(self, events, rng_samples):
        super().__init__()
        self.events = events
        self.rng_samples = rng_samples

    def forward(self, value, *, offset, scale, empty_expert_tokens):
        self.events.append(("router-dispatch", torch.is_grad_enabled()))
        sample = torch.rand_like(value)
        self.rng_samples.append(sample.detach().clone())
        expert = _InnerExpert.apply(value * sample + offset, self.events)
        dummy = _ZeroTokenDummy.apply(empty_expert_tokens.sum() + value.sum() * 0, self.events)
        self.events.append(("block-tail", torch.is_grad_enabled()))
        return expert * scale + dummy


class _OneBlockModel(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.layers = nn.ModuleList([block])


def _count_grad_nodes(root, node_name):
    pending = [root]
    seen = set()
    count = 0
    while pending:
        node = pending.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        count += type(node).__name__ == node_name
        pending.extend(next_node for next_node, _ in node.next_functions)
    return count


def test_reentrant_full_block_packs_kwargs_preserves_rng_and_finishes_recompute_before_inner_backward():
    events = []
    rng_samples = []
    model = _OneBlockModel(_OrderedBlock(events, rng_samples))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        apply_ac(model)

    wrapped = model.layers[0]
    assert isinstance(wrapped, CheckpointWrapper)
    assert wrapped.checkpoint_impl is CheckpointImpl.REENTRANT

    torch.manual_seed(17)
    value = torch.randn(4, requires_grad=True)
    offset = torch.randn(4)
    empty_expert_tokens = torch.empty(0, requires_grad=True)
    output = wrapped(
        value,
        offset=offset,
        scale=1.75,
        empty_expert_tokens=empty_expert_tokens,
    )

    assert type(output.grad_fn).__name__ == "CheckpointFunctionBackward"
    assert _count_grad_nodes(output.grad_fn, "CheckpointFunctionBackward") == 1
    assert [event for event in events if isinstance(event, tuple) and event[0] == "block-tail"] == [
        ("block-tail", False)
    ]

    output.sum().backward()

    dispatch_events = [event for event in events if isinstance(event, tuple) and event[0] == "router-dispatch"]
    tail_events = [event for event in events if isinstance(event, tuple) and event[0] == "block-tail"]
    assert dispatch_events == [("router-dispatch", False), ("router-dispatch", True)]
    assert tail_events == [("block-tail", False), ("block-tail", True)]
    torch.testing.assert_close(rng_samples[0], rng_samples[1], rtol=0, atol=0)

    recompute_tail = max(index for index, event in enumerate(events) if event == ("block-tail", True))
    inner_backward = events.index("inner-backward")
    dummy_backward = events.index("zero-token-dummy-backward")
    assert recompute_tail < inner_backward
    assert recompute_tail < dummy_backward

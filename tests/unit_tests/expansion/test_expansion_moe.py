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

"""Unit tests for expanding a MoE decoder layer.

The lateral term ``y_b = W_b x_b + y_a`` is per token, and an expanded linear inside an
expert only sees the tokens routed to that expert. Once the streams diverge they route
differently, the expert receives a different token count on each pass, and the recorded
``y_a`` cannot be added. ``RouterReplay`` fixes this by pinning stream B to stream A's
expert selection.

The MoE block here is a stand-in that mirrors ``Gate``'s call pattern
(``replay_selection`` then ``scores.gather``); it uses the real ``RouterReplay``. A test
against ``nemo_automodel.components.moe.layers`` would be a functional test -- it pulls in
kernels this CPU-only tier cannot run.

The divergence trap is the thing to remember when editing these: expanding only layer 1
leaves layer 1's input identical on both streams, because layer 0 runs in skip mode. Tests
that need genuinely different routing per stream must expand an upstream layer too.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.expansion import (
    ExpansionConfig,
    apply_expansion,
    expansion_parameters,
)
from nemo_automodel.components.moe.router_replay import RouterReplay, replay_selection

DIM, N_EXPERTS, TOPK, LAYERS, BATCH, SEQ = 16, 4, 2, 3, 2, 8


class _Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(DIM, DIM, bias=False)
        self.down_proj = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``[tokens, DIM]``. Returns: ``[tokens, DIM]``."""
        return self.down_proj(F.silu(self.gate_proj(x)))


class _MoEBlock(nn.Module):
    """Gate plus experts, dispatching in the flattened token layout AutoModel's MoE uses."""

    def __init__(self, enable_routing_replay: bool) -> None:
        super().__init__()
        self.router = nn.Linear(DIM, N_EXPERTS, bias=False)
        self.experts = nn.ModuleList(_Expert() for _ in range(N_EXPERTS))
        self.router_replay = RouterReplay() if enable_routing_replay else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``[batch, seq, DIM]``. Returns: ``[batch, seq, DIM]``."""
        flat = x.reshape(-1, DIM)
        scores = self.router(flat).softmax(-1)
        _, indices = scores.topk(TOPK, dim=-1)
        indices = replay_selection(self.router_replay, indices)
        weights = scores.gather(1, indices)

        out = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            hit = indices == expert_id
            token_mask = hit.any(-1)
            if not token_mask.any():
                continue
            weight = (weights * hit).sum(-1)[token_mask].unsqueeze(-1)
            out[token_mask] += expert(flat[token_mask]) * weight
        return out.view_as(x)


class _MoEDecoderLayer(nn.Module):
    def __init__(self, enable_routing_replay: bool) -> None:
        super().__init__()
        self.mlp = _MoEBlock(enable_routing_replay)

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Args: hidden_states ``[batch, seq, DIM]``. Returns: ``[batch, seq, DIM]``."""
        return hidden_states + self.mlp(hidden_states)


class _Decoder(nn.Module):
    def __init__(self, enable_routing_replay: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_MoEDecoderLayer(enable_routing_replay) for _ in range(LAYERS))
        self.norm = nn.LayerNorm(DIM)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Args: hidden_states ``[batch, seq, DIM]``. Returns: ``[batch, seq, DIM]``."""
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)


class _TinyMoECausalLM(nn.Module):
    """Minimal model exposing the ``.model.layers`` / ``.model.norm`` shape expansion needs."""

    def __init__(self, enable_routing_replay: bool = True) -> None:
        super().__init__()
        self.model = _Decoder(enable_routing_replay)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Args: hidden_states ``[batch, seq, DIM]``. Returns: ``[batch, seq, DIM]``."""
        return self.model(hidden_states)


def _build(layers, enable_routing_replay=True, diverge=0.0):
    RouterReplay.clear_registry()
    torch.manual_seed(0)
    model = _TinyMoECausalLM(enable_routing_replay).eval()
    if layers is not None:
        apply_expansion(
            model,
            ExpansionConfig(
                enabled=True,
                layers=layers,
                target_modules=["gate_proj", "down_proj"],
                zero_init_modules=["down_proj"],
            ),
        )
    if diverge:
        with torch.no_grad():
            for _, param in expansion_parameters(model):
                param.add_(torch.randn_like(param) * diverge)
    return model


@pytest.fixture
def hidden_states():
    torch.manual_seed(1)
    return torch.randn(BATCH, SEQ, DIM)


def test_expanded_moe_reproduces_parent_bit_exactly(hidden_states):
    """At initialization the streams are identical, so routing aligns trivially."""
    parent = _build(None)
    with torch.no_grad():
        expected = parent(hidden_states)
    expanded = _build([1])
    with torch.no_grad():
        assert torch.equal(expanded(hidden_states), expected)


def test_diverged_streams_keep_their_routing_aligned(hidden_states):
    """Layer 0 is expanded too, so layer 1 genuinely sees a different hidden state per stream.

    Without routing replay this raises a shape mismatch, because an expert receives a
    different number of tokens on each pass.
    """
    model = _build([0, 1], diverge=0.5)
    with torch.no_grad():
        model(hidden_states)


def test_moe_without_routing_replay_is_refused():
    """Refusing beats a shape error thrown from somewhere inside the expert dispatch."""
    with pytest.raises(ValueError, match="RouterReplay"):
        _build([0, 1], enable_routing_replay=False)


def test_outer_routing_replay_is_not_clobbered(hidden_states):
    """Expansion must compose with the RL routing replay it borrows the mechanism from.

    While an outer ``RouterReplay.replay(...)`` is active both streams are already pinned
    to the rollout's selection, which is exactly the alignment the lateral term needs, so
    the expansion leaves the handles alone and restores whatever mode it found.
    """
    model = _build([0, 1], diverge=0.5)
    handles = RouterReplay.instances()
    rollout_selection = torch.zeros(BATCH * SEQ, TOPK, dtype=torch.long)
    rollout_selection[:, 1] = 1

    with RouterReplay.replay([rollout_selection] * len(handles)):
        modes_before = [handle.mode for handle in handles]
        with torch.no_grad():
            model(hidden_states)
        assert [handle.mode for handle in handles] == modes_before
        assert all(torch.equal(handle.target_indices, rollout_selection) for handle in handles)

    assert all(handle.mode is None for handle in handles)


class _GroupedExperts(nn.Module):
    """Experts as one stacked parameter, the layout this repository's MoE blocks use.

    ``nemo_automodel.components.moe.experts.GroupedExperts`` keeps every expert's
    projection in a single ``[experts, in, out]`` parameter rather than in per-expert
    ``nn.Linear`` modules, so there is nothing for expansion to patch.
    """

    def __init__(self) -> None:
        super().__init__()
        self.gate_and_up_projs = nn.Parameter(torch.empty(N_EXPERTS, DIM, DIM))
        self.down_projs = nn.Parameter(torch.empty(N_EXPERTS, DIM, DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x ``[tokens, DIM]``. Returns: ``[tokens, DIM]``."""
        return x


class _GroupedMoEDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Linear(DIM, DIM, bias=False)
        self.mlp = _GroupedExperts()

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Args: hidden_states ``[batch, seq, DIM]``. Returns: ``[batch, seq, DIM]``."""
        return hidden_states + self.self_attn(hidden_states)


class _GroupedMoECausalLM(nn.Module):
    """Minimal model whose experts live in stacked parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(_GroupedMoEDecoderLayer() for _ in range(LAYERS))
        self.model.norm = nn.LayerNorm(DIM)


def test_stacked_expert_weights_are_refused_rather_than_skipped():
    """Half-expanding a MoE layer must fail, not pass quietly.

    Expansion patches ``nn.Linear``. Applied to a block whose experts are one stacked
    parameter, it reaches the attention projection beside them and nothing else, and the
    resulting model looks expanded while its experts are untouched.
    """
    with pytest.raises(NotImplementedError, match="stacked expert weights"):
        apply_expansion(
            _GroupedMoECausalLM(),
            # zero_init_modules must stay a subset of target_modules; the stand-in has no
            # residual-stream projection to zero, so it names none.
            ExpansionConfig(enabled=True, layers=[1], target_modules=["self_attn"], zero_init_modules=[]),
        )


def test_dense_layers_are_unaffected_by_the_stacked_weight_check():
    """The check must not fire on a model that has no stacked parameters at all."""
    model = _build([1])
    assert list(expansion_parameters(model))

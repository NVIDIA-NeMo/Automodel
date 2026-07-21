# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components.moe.parallelizer import _apply_attention_only_ac


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        self.dropout = nn.Dropout(p=0.25)
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply projection and dropout while counting executions.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        self.calls += 1
        return self.dropout(self.projection(hidden_states))


class _MLP(nn.Linear):
    def __init__(self) -> None:
        super().__init__(4, 4, bias=False)
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply the MLP projection while counting executions.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        self.calls += 1
        return super().forward(hidden_states)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = _MLP()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply residual attention and MLP transformations.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        hidden_states = hidden_states + self.self_attn(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class _Inner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run decoder blocks.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Inner()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the decoder.

        Args:
            hidden_states: Tensor of shape [batch, hidden].

        Returns:
            Tensor of shape [batch, hidden].
        """
        return self.model(hidden_states)


def test_attention_only_checkpointing_matches_eager_forward_and_backward() -> None:
    torch.manual_seed(1234)
    eager = _Model()
    checkpointed = copy.deepcopy(eager)
    eager_input = torch.randn(3, 4, requires_grad=True)
    checkpointed_input = eager_input.detach().clone().requires_grad_(True)

    torch.manual_seed(5678)
    eager_output = eager(eager_input)
    eager_output.sum().backward()
    eager_rng_state = torch.get_rng_state()
    _apply_attention_only_ac(checkpointed, ("language",))
    torch.manual_seed(5678)
    checkpointed_output = checkpointed(checkpointed_input)
    checkpointed_output.sum().backward()
    checkpointed_rng_state = torch.get_rng_state()

    torch.testing.assert_close(checkpointed_output, eager_output)
    torch.testing.assert_close(checkpointed_input.grad, eager_input.grad)
    torch.testing.assert_close(checkpointed_rng_state, eager_rng_state)
    for checkpointed_parameter, eager_parameter in zip(checkpointed.parameters(), eager.parameters(), strict=True):
        torch.testing.assert_close(checkpointed_parameter.grad, eager_parameter.grad)
    for block in checkpointed.model.layers:
        assert hasattr(block.self_attn, "_checkpoint_wrapped_module")
        assert not hasattr(block.mlp, "_checkpoint_wrapped_module")
        assert block.self_attn._checkpoint_wrapped_module.calls == 2
        assert block.mlp.calls == 1


def test_attention_only_checkpointing_rejects_kv_sharing() -> None:
    model = _Model()
    model.config = SimpleNamespace(num_kv_shared_layers=1)

    with pytest.raises(RuntimeError, match="does not support KV-sharing"):
        _apply_attention_only_ac(model, ("language",))

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

import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.moe.experts import GroupedExpertsTE

_ROUTE_SEQUENCE = (True, False, True)
_NUM_EXPERTS = 2
_HIDDEN_SIZE = 4


class _NoFP8GlobalStateManager:
    @staticmethod
    def is_fp8_enabled() -> bool:
        return False


class _TinyGroupedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_groups = _NUM_EXPERTS
        for expert_index in range(_NUM_EXPERTS):
            self.register_parameter(
                f"weight{expert_index}",
                nn.Parameter(torch.randn(_HIDDEN_SIZE, _HIDDEN_SIZE, dtype=torch.float64) * 0.1),
            )
            self.register_parameter(
                f"bias{expert_index}",
                nn.Parameter(torch.randn(_HIDDEN_SIZE, dtype=torch.float64) * 0.1),
            )


class _TinyExpertPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_linear = _TinyGroupedLinear()
        self.down_linear = _TinyGroupedLinear()


class _TrackedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, token_probs, empty, dispatcher):
        ctx.hidden_shape = hidden_states.shape
        ctx.prob_shape = token_probs.shape
        ctx.empty = empty
        ctx.dispatcher = dispatcher
        ctx.step = dispatcher.current_step
        dispatcher.forward_steps.append((ctx.step, empty))
        if empty:
            return hidden_states[:0], token_probs.reshape(-1)[:0]
        return hidden_states.clone(), token_probs.reshape(-1).clone()

    @staticmethod
    def backward(ctx, grad_hidden_states, grad_token_probs):
        ctx.dispatcher.backward_steps.append((ctx.step, ctx.empty))
        if ctx.empty:
            assert grad_hidden_states.shape == (0, _HIDDEN_SIZE)
            assert grad_token_probs.shape == (0,)
            return (
                grad_hidden_states.new_zeros(ctx.hidden_shape),
                grad_token_probs.new_zeros(ctx.prob_shape),
                None,
                None,
            )
        return grad_hidden_states, grad_token_probs.reshape(ctx.prob_shape), None, None


class _ScriptedHybridEPDispatcher:
    def __init__(self) -> None:
        self.current_step = -1
        self.forward_steps = []
        self.backward_steps = []
        self.unpermutation_steps = []

    def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
        del num_local_tokens, token_indices
        empty = _ROUTE_SEQUENCE[self.current_step]
        permuted_hidden, permuted_probs = _TrackedDispatch.apply(hidden_states, token_probs, empty, self)
        if empty:
            tokens_per_expert = torch.zeros(_NUM_EXPERTS, dtype=torch.int64, device=hidden_states.device)
        else:
            tokens_per_expert = torch.tensor([2, 2], dtype=torch.int64, device=hidden_states.device)
        return permuted_hidden, tokens_per_expert, permuted_probs

    def token_unpermutation(self, hidden_states):
        self.unpermutation_steps.append(self.current_step)
        return hidden_states


def _grouped_mlp(owner, hidden_states, tokens_per_expert, token_probs):
    outputs = []
    offset = 0
    for expert_index, token_count_tensor in enumerate(tokens_per_expert):
        token_count = int(token_count_tensor.item())
        expert_hidden = hidden_states[offset : offset + token_count]
        expert_probs = token_probs[offset : offset + token_count].unsqueeze(-1)

        gate_weight = getattr(owner.gate_up_linear, f"weight{expert_index}")
        gate_bias = getattr(owner.gate_up_linear, f"bias{expert_index}")
        down_weight = getattr(owner.down_linear, f"weight{expert_index}")
        down_bias = getattr(owner.down_linear, f"bias{expert_index}")

        activated = torch.tanh(F.linear(expert_hidden, gate_weight, gate_bias))
        outputs.append(F.linear(activated, down_weight, down_bias) * expert_probs)
        offset += token_count

    assert offset == hidden_states.shape[0]
    return torch.cat(outputs, dim=0)


def _build_experts_under_test():
    experts = GroupedExpertsTE.__new__(GroupedExpertsTE)
    nn.Module.__init__(experts)
    experts.config = SimpleNamespace(n_routed_experts=_NUM_EXPERTS)
    experts.ep_size = 1
    experts.use_te_ops = True
    experts.expert_bias = True
    experts.gate_up_linear = _TinyGroupedLinear()
    experts.down_linear = _TinyGroupedLinear()
    experts._te_ops_fusion_checked = False
    experts.token_dispatcher = _ScriptedHybridEPDispatcher()

    grouped_mlp_calls = []

    def grouped_mlp(hidden_states, tokens_per_expert, token_probs, *fc2_extra_inputs):
        assert len(fc2_extra_inputs) == 2
        torch.testing.assert_close(fc2_extra_inputs[0], tokens_per_expert)
        torch.testing.assert_close(fc2_extra_inputs[1], token_probs)
        grouped_mlp_calls.append(experts.token_dispatcher.current_step)
        return _grouped_mlp(experts, hidden_states, tokens_per_expert, token_probs)

    experts.__dict__["_te_grouped_mlp"] = grouped_mlp
    return experts, grouped_mlp_calls


def _copy_parameters(source, destination) -> None:
    source_parameters = dict(source.named_parameters())
    destination_parameters = dict(destination.named_parameters())
    assert source_parameters.keys() == destination_parameters.keys()
    with torch.no_grad():
        for name, parameter in source_parameters.items():
            destination_parameters[name].copy_(parameter)


def _eager_explicit_zero_reference(experts, hidden_states, token_probs, empty):
    if not empty:
        splits = torch.tensor([2, 2], dtype=torch.int64, device=hidden_states.device)
        return _grouped_mlp(experts, hidden_states, splits, token_probs.reshape(-1))

    zero = hidden_states.sum() * 0 + token_probs.sum() * 0
    for parameter in experts.parameters():
        zero = zero + parameter.reshape(-1)[0] * 0
    return hidden_states[:0] * 0 + zero


def _assert_optimizer_parity(optimizer, reference_optimizer, parameters, reference_parameters) -> None:
    for name, parameter in parameters.items():
        reference_parameter = reference_parameters[name]
        state = optimizer.state[parameter]
        reference_state = reference_optimizer.state[reference_parameter]
        assert state.keys() == reference_state.keys()
        for key, value in state.items():
            reference_value = reference_state[key]
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, reference_value, rtol=0, atol=0)
            else:
                assert value == reference_value


def test_te_ops_empty_nonempty_empty_matches_explicit_zero_adamw_reference(monkeypatch):
    """Dynamic empty routing keeps communication, gradients, parameters, and AdamW state convergent."""
    quantization = types.ModuleType("transformer_engine.pytorch.quantization")
    quantization.FP8GlobalStateManager = _NoFP8GlobalStateManager
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.quantization", quantization)

    torch.manual_seed(1234)
    experts, grouped_mlp_calls = _build_experts_under_test()
    reference = _TinyExpertPair()
    _copy_parameters(experts, reference)

    parameters = dict(experts.named_parameters())
    reference_parameters = dict(reference.named_parameters())
    assert parameters.keys() == reference_parameters.keys()
    assert len(parameters) == 8
    parameter_ids = {name: id(parameter) for name, parameter in parameters.items()}

    learning_rate = 0.03
    weight_decay = 0.1
    optimizer = torch.optim.AdamW(
        parameters.values(),
        lr=learning_rate,
        betas=(0.8, 0.95),
        eps=1e-9,
        weight_decay=weight_decay,
    )
    reference_optimizer = torch.optim.AdamW(
        reference_parameters.values(),
        lr=learning_rate,
        betas=(0.8, 0.95),
        eps=1e-9,
        weight_decay=weight_decay,
    )

    token_mask = torch.ones(4, dtype=torch.bool)
    token_indices = torch.tensor([[0], [0], [1], [1]], dtype=torch.int64)

    for step, empty in enumerate(_ROUTE_SEQUENCE):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)

        generator = torch.Generator().manual_seed(8000 + step)
        hidden_base = torch.randn(4, _HIDDEN_SIZE, dtype=torch.float64, generator=generator)
        router_base = torch.randn(4, 1, dtype=torch.float64, generator=generator)

        hidden = hidden_base.clone().requires_grad_()
        reference_hidden = hidden_base.clone().requires_grad_()
        router_logits = router_base.clone().requires_grad_()
        reference_router_logits = router_base.clone().requires_grad_()
        token_probs = router_logits.sigmoid()
        reference_token_probs = reference_router_logits.sigmoid()
        token_probs.retain_grad()
        reference_token_probs.retain_grad()

        experts.token_dispatcher.current_step = step
        output = experts(hidden, token_mask, token_probs, token_indices)
        reference_output = _eager_explicit_zero_reference(
            reference,
            reference_hidden,
            reference_token_probs,
            empty,
        )
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)

        target = torch.linspace(-0.3, 0.4, output.numel(), dtype=output.dtype).reshape_as(output)
        reference_target = target.clone()
        loss = (output - target).square().sum() + output.sum() * 0.17
        reference_loss = (reference_output - reference_target).square().sum() + reference_output.sum() * 0.17
        loss.backward()
        reference_loss.backward()

        assert experts.token_dispatcher.backward_steps == [
            (completed_step, _ROUTE_SEQUENCE[completed_step]) for completed_step in range(step + 1)
        ]
        torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(token_probs.grad, reference_token_probs.grad, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(router_logits.grad, reference_router_logits.grad, rtol=1e-12, atol=1e-12)

        if empty:
            assert torch.count_nonzero(hidden.grad) == 0
            assert torch.count_nonzero(token_probs.grad) == 0
            assert torch.count_nonzero(router_logits.grad) == 0
            for name, parameter in parameters.items():
                assert parameter.grad is not None, f"{name} must have an explicit zero gradient"
                assert torch.count_nonzero(parameter.grad) == 0
        else:
            assert torch.count_nonzero(hidden.grad) > 0
            assert torch.count_nonzero(token_probs.grad) > 0
            assert torch.count_nonzero(router_logits.grad) > 0

        for name, parameter in parameters.items():
            torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, rtol=1e-12, atol=1e-12)

        parameters_before_step = {name: parameter.detach().clone() for name, parameter in parameters.items()}
        optimizer.step()
        reference_optimizer.step()

        assert {name: id(parameter) for name, parameter in experts.named_parameters()} == parameter_ids
        for name, parameter in parameters.items():
            torch.testing.assert_close(parameter, reference_parameters[name], rtol=0, atol=0)
            if step == 0:
                expected_after_decay = parameters_before_step[name] * (1 - learning_rate * weight_decay)
                torch.testing.assert_close(parameter, expected_after_decay, rtol=1e-12, atol=1e-12)
        _assert_optimizer_parity(optimizer, reference_optimizer, parameters, reference_parameters)

    assert experts.token_dispatcher.forward_steps == list(enumerate(_ROUTE_SEQUENCE))
    assert experts.token_dispatcher.unpermutation_steps == [0, 1, 2]
    assert grouped_mlp_calls == [1]
    for parameter in parameters.values():
        assert optimizer.state[parameter]["step"].item() == 3


@pytest.mark.run_only_on("GPU")
@pytest.mark.skip(
    reason=(
        "HybridEP convergence integration skeleton: requires two GB200 ranks, Transformer Engine, "
        "and the HybridEP runtime under torchrun"
    )
)
def test_hybridep_two_rank_te_ops_empty_nonempty_empty_convergence():
    """Run the three-step contract above with rank 0 alternating empty/nonempty HybridEP receives."""

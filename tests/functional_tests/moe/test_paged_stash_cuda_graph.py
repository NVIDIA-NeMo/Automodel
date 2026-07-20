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

"""CUDA parity guard for paged TE expert activations under real graph replay."""

import pytest
import torch
from torch import nn

from nemo_automodel.components.moe.paged_stash import PagedStashManager
from nemo_automodel.components.moe.paged_stash_ops import HAVE_TRITON
from nemo_automodel.recipes.llm.partial_cuda_graphs import PartialCudaGraphManager, _PartialGraphEntry


class _GraphableTEExperts(nn.Module):
    """Tiny fixed-capacity TE expert boundary with production-style stash hooks."""

    def __init__(self, stash_manager: PagedStashManager, grouped_linear: type[nn.Module]) -> None:
        super().__init__()
        self.__dict__["stash_manager"] = stash_manager
        self.grouped_linear = grouped_linear(
            num_gemms=2,
            in_features=4,
            out_features=8,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_mask: torch.Tensor,
        routing_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Run two equal-capacity experts while paging live saved rows.

        Args:
            hidden_states: BF16 expert inputs with shape ``[6, 4]``.
            token_mask: Boolean valid-token mask with shape ``[6]``.
            routing_weights: Differentiable routing weights with shape ``[6, 1]``.
            expert_indices: Expert ids with shape ``[6, 1]``; this guard uses expert zero for every live row.

        Returns:
            Grouped expert output with shape ``[6, 8]``.
        """
        weighted_states = hidden_states * routing_weights.to(dtype=hidden_states.dtype)
        live_token_mask = token_mask & routing_weights[:, 0].ne(0) & expert_indices[:, 0].eq(0)
        group = self.stash_manager.group(
            name="functional_te_grouped_linear",
            max_num_tokens=hidden_states.shape[0],
            live_token_mask=live_token_mask,
        )
        weighted_states = group.start(weighted_states)
        with group:
            output = self.grouped_linear(group.mark_activation(weighted_states), [3, 3])
        return group.commit(output)


def _inputs(live_token_mask: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, ...]:
    """Create one fixed-shape expert call whose padding rows carry zero routing weight."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    hidden_states = torch.randn(6, 4, generator=generator, device="cuda", dtype=torch.bfloat16)
    routing_weights = torch.rand(6, 1, generator=generator, device="cuda", dtype=torch.float32)
    routing_weights[~live_token_mask] = 0
    return (
        hidden_states.requires_grad_(),
        live_token_mask,
        routing_weights.requires_grad_(),
        torch.zeros(6, 1, device="cuda", dtype=torch.int64),
    )


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA and Triton are required")
def test_te_grouped_linear_paged_stash_matches_eager_across_graph_replays():
    """Compare BF16 graph output/gradients/update with eager and force page reuse."""
    te = pytest.importorskip("transformer_engine.pytorch")
    stash_manager = PagedStashManager()
    stash_manager.configure(enabled=True, page_size=2, buffer_size_factor=1.0)
    target = _GraphableTEExperts(stash_manager, te.GroupedLinear)
    optimizer = torch.optim.SGD(target.parameters(), lr=0.01)
    graph_manager = PartialCudaGraphManager(
        [
            _PartialGraphEntry(
                name="functional.te_experts",
                target=target,
                pool_group="moe",
                explicit_parameters=True,
                retain_graph_in_backward=True,
            )
        ]
    )
    graph_manager.start_recording()

    try:
        warmup_mask = torch.ones(6, dtype=torch.bool, device="cuda")
        warmup_inputs = _inputs(warmup_mask, seed=123)
        warmup_output = target(*warmup_inputs)
        warmup_output.sum().backward()
        assert stash_manager.diagnostics()["recorded_peak_tokens"]
        optimizer.zero_grad(set_to_none=True)

        stash_manager.prepare()
        graph_manager.capture()
        assert stash_manager.check_overflow().item() == 0
        stash_manager.finish_iteration()
        optimizer.zero_grad(set_to_none=True)

        replay_masks = (
            torch.tensor([True, False, True, True, False, True], device="cuda"),
            torch.tensor([False, True, True, False, True, True], device="cuda"),
        )
        for replay_index, live_token_mask in enumerate(replay_masks):
            eager_inputs = _inputs(live_token_mask, seed=456 + replay_index)
            with graph_manager.eager_execution(), stash_manager.disabled():
                eager_output = target(*eager_inputs)
            output_gradient = torch.linspace(
                0.25,
                2.0,
                eager_output.numel(),
                device="cuda",
                dtype=eager_output.dtype,
            ).reshape_as(eager_output)
            (eager_output * output_gradient).sum().backward()
            eager_input_gradients = (eager_inputs[0].grad.detach().clone(), eager_inputs[2].grad.detach().clone())
            eager_parameter_gradients = {
                name: parameter.grad.detach().clone() for name, parameter in target.named_parameters()
            }
            parameters_before_step = {
                name: parameter.detach().clone() for name, parameter in target.named_parameters()
            }
            optimizer.zero_grad(set_to_none=True)

            graph_inputs = tuple(value.detach().clone().requires_grad_(value.requires_grad) for value in eager_inputs)
            graph_output = target(*graph_inputs)
            (graph_output * output_gradient).sum().backward()

            torch.testing.assert_close(graph_output, eager_output)
            torch.testing.assert_close(graph_inputs[0].grad, eager_input_gradients[0])
            torch.testing.assert_close(graph_inputs[2].grad, eager_input_gradients[1])
            for name, parameter in target.named_parameters():
                torch.testing.assert_close(parameter.grad, eager_parameter_gradients[name])

            assert stash_manager.check_overflow().item() == 0
            stash_manager.finish_iteration()
            if replay_index == 0:
                optimizer.step()
                for name, parameter in target.named_parameters():
                    expected = parameters_before_step[name] - 0.01 * eager_parameter_gradients[name]
                    torch.testing.assert_close(parameter, expected)
            optimizer.zero_grad(set_to_none=True)
    finally:
        graph_manager.close()
        stash_manager.close()

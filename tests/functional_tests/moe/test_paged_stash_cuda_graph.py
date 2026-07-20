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

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp
from torch import nn

import nemo_automodel.components.moe.paged_stash as paged_stash
from nemo_automodel.components.moe.paged_stash import PagedStashManager
from nemo_automodel.components.moe.paged_stash_ops import HAVE_TRITON
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes.llm.partial_cuda_graphs import PartialCudaGraphManager, _PartialGraphEntry
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class _GatedActivation(torch.autograd.Function):
    """Tiny gated activation that saves the complete marked pre-activation surface."""

    @staticmethod
    def forward(ctx, gate_up_output: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU while saving the complete marked activation.

        Args:
            ctx: Autograd context for the gated activation.
            gate_up_output: Contiguous pre-activation with shape ``[tokens, 2 * hidden]``.

        Returns:
            Activated tensor with shape ``[tokens, hidden]``.
        """
        ctx.save_for_backward(gate_up_output)
        gate, up = gate_up_output.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """Differentiate the gated activation.

        Args:
            ctx: Autograd context populated by :meth:`forward`.
            grad_output: Activation gradient with shape ``[tokens, hidden]``.

        Returns:
            Pre-activation gradient with shape ``[tokens, 2 * hidden]``.
        """
        (gate_up_output,) = ctx.saved_tensors
        gate, up = gate_up_output.chunk(2, dim=-1)
        sigmoid = torch.sigmoid(gate)
        silu = gate * sigmoid
        gate_gradient = grad_output * up * sigmoid * (1 + gate * (1 - sigmoid))
        up_gradient = grad_output * silu
        return (torch.cat((gate_gradient, up_gradient), dim=-1),)


class _SegmentedSquare(torch.autograd.Function):
    """Simple saved-tensor consumer used to verify sparse row restoration."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        """Save and square a tensor with shape ``[tokens, hidden]``."""
        ctx.save_for_backward(tensor)
        return tensor.square()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        """Return the elementwise gradient with shape ``[tokens, hidden]``."""
        (tensor,) = ctx.saved_tensors
        return (2 * tensor * grad_output,)


class _GraphableTEExperts(nn.Module):
    """Tiny fixed-capacity TE expert boundary with production-style stash hooks."""

    def __init__(self, stash_manager: PagedStashManager, grouped_linear: type[nn.Module]) -> None:
        super().__init__()
        self.__dict__["stash_manager"] = stash_manager
        self.gate_up_linear = grouped_linear(
            num_gemms=2,
            in_features=4,
            out_features=8,
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
        self.down_linear = grouped_linear(
            num_gemms=2,
            in_features=4,
            out_features=4,
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
            Grouped expert output with shape ``[6, 4]``.
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
            gate_up_output = self.gate_up_linear(weighted_states, [3, 3])
            gate_up_output = group.mark_activation(gate_up_output)
            activated = _GatedActivation.apply(gate_up_output)
            activated = group.mark_activation(activated)
            output = self.down_linear(activated, [3, 3])
        del gate_up_output, activated
        return group.commit(output)


def _inputs(live_token_mask: torch.Tensor, *, seed: int) -> tuple[torch.Tensor, ...]:
    """Create one fixed-shape expert call whose padding rows carry zero routing weight.

    Args:
        live_token_mask: Boolean CUDA tensor with shape ``[6]``.
        seed: Seed for the local CUDA generator that creates inputs.

    Returns:
        Hidden states ``[6, 4]``, the unchanged token mask ``[6]``, differentiable routing weights ``[6, 1]``,
        and integer expert indices ``[6, 1]``.
    """
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
def test_active_stash_restores_sparse_rows_and_zeros_padding():
    """Validate Triton page copy/pop semantics before composing them with TE graphs."""
    manager = PagedStashManager()
    manager.configure(enabled=True, page_size=2, buffer_size_factor=1.5)

    warmup = torch.randn(8, 4, device="cuda", requires_grad=True)
    warmup_mask = torch.tensor([True, False, True, True, False, True, False, True], device="cuda")
    warmup_group = manager.group(name="warmup", max_num_tokens=8, live_token_mask=warmup_mask)
    warmup_for_compute = warmup_group.start(warmup)
    with warmup_group:
        warmup_output = _SegmentedSquare.apply(warmup_group.mark_activation(warmup_for_compute))
    warmup_group.commit(warmup_output).sum().backward()
    manager.prepare()

    tensor = torch.randn(8, 4, device="cuda", requires_grad=True)
    live_mask = torch.tensor([True, False, True, False, False, True, False, False], device="cuda")
    group = manager.group(name="active", max_num_tokens=8, live_token_mask=live_mask)
    tensor_for_compute = group.start(tensor)
    with group:
        output = _SegmentedSquare.apply(group.mark_activation(tensor_for_compute))
    group.commit(output).sum().backward()

    expected = torch.zeros_like(tensor)
    expected[live_mask] = 2 * tensor.detach()[live_mask]
    torch.testing.assert_close(tensor.grad, expected)
    assert manager.check_overflow() is not None
    assert manager.check_overflow().item() == 0
    manager.finish_iteration()
    manager.close()


def _asymmetric_overflow_worker(rank: int, world_size: int, init_method: str) -> None:
    """Verify a single-rank overflow makes every rank discard and retry."""
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
    )
    events = []
    stash_manager = SimpleNamespace(
        is_active=True,
        check_overflow=lambda: torch.tensor([rank == 0], dtype=torch.int64, device="cuda"),
        close=lambda: events.append("stash-close"),
    )
    paged_stash._PAGED_STASH_MANAGER = stash_manager
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    recipe.partial_cuda_graph_manager = SimpleNamespace(close=lambda: events.append("graph-close"))
    recipe._partial_cuda_graph_capture_pending = False
    recipe._partial_cuda_graph_paged_stash_enabled = True
    recipe._partial_cuda_graph_paged_stash_reruns = 0
    recipe.optimizer = [SimpleNamespace(zero_grad=lambda: events.append("zero-grad"))]
    stateful_rng = StatefulRNG(seed=1234 + rank)
    rng_state = stateful_rng.state_dict()
    expected_cuda_draw = torch.rand(3, device="cuda")
    torch.rand(3, device="cuda")

    def restore_rng(state) -> None:
        events.append(("restore-rng", state))
        stateful_rng.load_state_dict(state)

    actual_cuda_draws = []

    def rerun(_batches, num_label_tokens):
        events.append(("rerun", num_label_tokens))
        actual_cuda_draws.append(torch.rand(3, device="cuda"))
        return [torch.tensor(2.0, device="cuda")]

    recipe.rng = SimpleNamespace(load_state_dict=restore_rng)
    recipe._run_forward_backward_batches = rerun

    try:
        result = recipe._rerun_after_paged_stash_overflow(
            ["same-batch"],
            num_label_tokens=7,
            loss_buffer=[torch.tensor(1.0, device="cuda")],
            rng_state=rng_state,
        )
        assert result[0].item() == 2.0
        assert recipe._partial_cuda_graph_paged_stash_reruns == 1
        assert events[:3] == ["graph-close", "stash-close", "zero-grad"]
        assert events[3][0] == "restore-rng"
        assert events[3][1] is rng_state
        assert events[4] == ("rerun", 7)
        torch.testing.assert_close(actual_cuda_draws[0], expected_cuda_draw)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
def test_asymmetric_overflow_retries_on_every_rank():
    """Exercise the real NCCL overflow reduction with only rank zero set."""
    with tempfile.NamedTemporaryFile(delete=False) as rendezvous:
        rendezvous_path = rendezvous.name
    try:
        mp.spawn(
            _asymmetric_overflow_worker,
            args=(2, f"file://{rendezvous_path}"),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(rendezvous_path):
            os.unlink(rendezvous_path)


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
        assert set(stash_manager.diagnostics()["recorded_peak_tokens"]) == {
            (torch.bfloat16, 4),
            (torch.bfloat16, 8),
        }
        optimizer.zero_grad(set_to_none=True)
        # The manager owns detached sample clones after recording, so these local
        # warmup references are no longer needed.
        del warmup_output, warmup_inputs

        stash_manager.prepare()
        graph_manager.capture()
        assert stash_manager.check_overflow().item() == 0
        stash_manager.finish_iteration()
        optimizer.zero_grad(set_to_none=True)

        replay_masks = (
            torch.tensor([True, False, True, True, False, True], device="cuda"),
            torch.tensor([False, True, True, False, True, True], device="cuda"),
            torch.tensor([True, True, False, True, True, False], device="cuda"),
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
            parameters_before_step = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}
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
        assert graph_manager.stats() == {"captured": 1, "replayed": 3, "fallback": 0}
    finally:
        graph_manager.close()
        stash_manager.close()

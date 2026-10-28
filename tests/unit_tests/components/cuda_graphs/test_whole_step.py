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

import pytest
import torch
from torch import nn

from nemo_automodel.components.cuda_graphs import WholeStepGraph, WholeStepGraphValidation

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _model_and_step(device: torch.device, num_microbatches: int):
    """Return a tiny linear model and a step function summing squared errors over microbatches."""
    torch.manual_seed(0)
    model = nn.Linear(4, 3).to(device)

    def step_fn(batches):
        losses = []
        for batch in batches:
            loss = (model(batch["x"]) - batch["y"]).square().sum() / num_microbatches
            loss.backward()
            losses.append(loss.detach().clone())
        return losses

    return model, step_fn


def _batches(num_microbatches: int, seed: int, device: str = "cpu"):
    generator = torch.Generator().manual_seed(seed)
    return [
        {
            "x": torch.randn(2, 4, generator=generator).to(device),
            "y": torch.randn(2, 3, generator=generator).to(device),
        }
        for _ in range(num_microbatches)
    ]


def _eager_grads(model: nn.Module, step_fn, batches, device: torch.device):
    for parameter in model.parameters():
        parameter.grad = None
    losses = step_fn([{k: v.to(device) for k, v in b.items()} for b in batches])
    grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = None
    return losses, grads


class TestWholeStepGraphCpuContracts:
    def test_replay_before_capture_is_rejected(self):
        model, step_fn = _model_and_step(torch.device("cpu"), 1)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=torch.device("cpu"))
        assert not graph.captured
        with pytest.raises(RuntimeError, match="has not been captured"):
            graph.replay(_batches(1, seed=0))
        with pytest.raises(RuntimeError, match="has not been captured"):
            graph.validate(_batches(1, seed=0))

    def test_warmup_iterations_must_be_positive(self):
        model, step_fn = _model_and_step(torch.device("cpu"), 1)
        with pytest.raises(ValueError, match="warmup_iterations"):
            WholeStepGraph(step_fn, list(model.named_parameters()), device=torch.device("cpu"), warmup_iterations=0)

    def test_capture_rejects_non_tensor_batch_entries(self):
        model, step_fn = _model_and_step(torch.device("cpu"), 1)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=torch.device("cpu"))
        batch = _batches(1, seed=0)[0]
        batch["qkv_format"] = "thd"
        with pytest.raises(ValueError, match="flat tensor batch"):
            graph.capture([batch])
        assert not graph.captured

    def test_capture_rejects_empty_step(self):
        model, step_fn = _model_and_step(torch.device("cpu"), 1)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=torch.device("cpu"))
        with pytest.raises(ValueError, match="at least one microbatch"):
            graph.capture([])

    def test_reset_before_capture_is_a_no_op(self):
        model, step_fn = _model_and_step(torch.device("cpu"), 1)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=torch.device("cpu"))
        graph.reset()
        assert not graph.captured


@requires_cuda
class TestWholeStepGraphCuda:
    @pytest.mark.parametrize("num_microbatches", [1, 3])
    def test_replay_matches_eager_gradients_for_new_inputs(self, num_microbatches):
        device = torch.device("cuda")
        model, step_fn = _model_and_step(device, num_microbatches)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=device)
        try:
            graph.capture(_batches(num_microbatches, seed=1))
            assert graph.captured
            new_batches = _batches(num_microbatches, seed=2)
            losses = graph.replay(new_batches)
            torch.cuda.synchronize()
            graph_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters()}
            graph_losses = torch.stack(losses).clone()

            expected_losses, expected_grads = _eager_grads(model, step_fn, new_batches, device)
            torch.testing.assert_close(graph_losses, torch.stack(expected_losses))
            for name, expected in expected_grads.items():
                torch.testing.assert_close(graph_grads[name], expected)
        finally:
            graph.reset()
        assert not graph.captured

    def test_replay_rebinds_gradients_after_optimizer_zero_grad(self):
        device = torch.device("cuda")
        model, step_fn = _model_and_step(device, 1)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=device)
        try:
            graph.capture(_batches(1, seed=1))
            graph.replay(_batches(1, seed=2))
            captured = {name: p.grad for name, p in model.named_parameters()}
            for parameter in model.parameters():
                parameter.grad = None
            graph.replay(_batches(1, seed=3))
            for name, parameter in model.named_parameters():
                assert parameter.grad is captured[name]
        finally:
            graph.reset()

    def test_replay_rejects_changed_shapes_keys_and_microbatch_count(self):
        device = torch.device("cuda")
        model, step_fn = _model_and_step(device, 2)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=device)
        try:
            graph.capture(_batches(2, seed=1))
            with pytest.raises(ValueError, match="microbatches per step"):
                graph.replay(_batches(1, seed=2))
            wrong_shape = _batches(2, seed=2)
            wrong_shape[1]["x"] = torch.randn(3, 4)
            with pytest.raises(ValueError, match="microbatch 1 entry 'x' changed"):
                graph.replay(wrong_shape)
            wrong_keys = _batches(2, seed=2)
            wrong_keys[0]["mask"] = torch.ones(2, 4)
            with pytest.raises(ValueError, match="microbatch 0 keys changed"):
                graph.replay(wrong_keys)
        finally:
            graph.reset()

    def test_validate_reports_agreement_and_restores_graph_gradients(self):
        device = torch.device("cuda")
        model, step_fn = _model_and_step(device, 2)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=device)
        try:
            graph.capture(_batches(2, seed=1))
            batches = _batches(2, seed=2)
            graph.replay(batches)
            captured = {name: p.grad for name, p in model.named_parameters()}
            report = graph.validate(batches)
            assert isinstance(report, WholeStepGraphValidation)
            assert report.num_gradients == 2
            assert report.all_finite
            assert report.max_relative_rmse < 1e-5
            assert report.max_eager_repeat_relative_rmse < 1e-5
            assert report.max_abs_loss_difference < 1e-5
            for name, parameter in model.named_parameters():
                assert parameter.grad is captured[name]
        finally:
            graph.reset()

    def test_parameter_storage_change_is_detected(self):
        device = torch.device("cuda")
        model, step_fn = _model_and_step(device, 1)
        graph = WholeStepGraph(step_fn, list(model.named_parameters()), device=device)
        try:
            graph.capture(_batches(1, seed=1))
            model.weight.data = model.weight.data.clone()
            with pytest.raises(RuntimeError, match="Parameter storage changed"):
                graph.replay(_batches(1, seed=2))
        finally:
            graph.reset()

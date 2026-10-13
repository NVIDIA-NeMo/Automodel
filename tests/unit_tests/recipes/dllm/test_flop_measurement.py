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

from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe


def test_failed_measured_training_step_restores_compilation() -> None:
    torch.compiler.reset()
    normal_backend = CompileCounterWithBackend("inductor")

    @torch.compile(fullgraph=True, backend=normal_backend)
    def forward(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Multiply x [tokens, hidden] by weight [hidden, output]."""
        return x @ weight

    x = torch.randn(2, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)

    def failing_step(recipe, index, batch, **kwargs):
        forward(x, weight).sum().backward()
        raise ValueError("training step failed")

    recipe = SimpleNamespace(
        cfg={"measure_flops": True, "measure_flops_after": 0},
        step_scheduler=SimpleNamespace(step=0),
        dllm_strategy=SimpleNamespace(pre_step=lambda recipe, batches: (1, 1), forward_backward=failing_step),
        _dp_allreduce=lambda tensor: tensor,
        _compute_loss_denominators=lambda batches, noise, supervised: (noise, supervised),
        model_parts=[],
        pp_enabled=False,
    )
    with pytest.raises(ValueError, match="training step failed"):
        DiffusionLMSFTRecipe._run_train_optim_step(recipe, [{"input_ids": torch.ones(1, 2, dtype=torch.long)}])
    assert recipe._flop_measure_attempted
    assert x.grad is not None and weight.grad is not None
    assert normal_backend.frame_count == 0
    forward(x, weight).sum().backward()
    assert normal_backend.frame_count == 1

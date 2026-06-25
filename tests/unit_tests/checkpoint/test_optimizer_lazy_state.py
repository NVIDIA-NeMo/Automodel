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

from pathlib import Path

import torch

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig


class _SparseOptimizerStateModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.active = torch.nn.Linear(2, 2, bias=False)
        self.inactive = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.active(x).sum()


def _make_checkpointer(path: Path) -> Checkpointer:
    config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(path),
        model_save_format="safetensors",
        model_cache_dir=str(path / "cache"),
        model_repo_id="test/model",
        save_consolidated=False,
        is_peft=False,
    )
    return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)


def _step_active_only(model: _SparseOptimizerStateModel, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 2)).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _prime_all_optimizer_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.grad = torch.zeros_like(param)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_optimizer_save_materializes_missing_adam_state_for_dcp_load(tmp_path: Path) -> None:
    source_model = _SparseOptimizerStateModel()
    source_optimizer = torch.optim.AdamW(source_model.parameters(), lr=1e-3)
    _step_active_only(source_model, source_optimizer)
    assert source_model.inactive.weight not in source_optimizer.state

    checkpointer = _make_checkpointer(tmp_path)
    checkpointer.save_optimizer(source_optimizer, source_model, str(tmp_path))
    assert "step" in source_optimizer.state[source_model.inactive.weight]

    target_model = _SparseOptimizerStateModel()
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=1e-3)
    _prime_all_optimizer_state(target_model, target_optimizer)

    checkpointer.load_optimizer(target_optimizer, target_model, str(tmp_path))

    inactive_state = target_optimizer.state[target_model.inactive.weight]
    assert inactive_state["step"].item() == 0
    torch.testing.assert_close(inactive_state["exp_avg"], torch.zeros_like(target_model.inactive.weight))
    torch.testing.assert_close(inactive_state["exp_avg_sq"], torch.zeros_like(target_model.inactive.weight))

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from torch import nn

from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.recipes.vlm import kd as vlm_kd


class _Cfg:
    def get(self, key, default=None):
        if key == "fp8":
            return None
        return default


class _Optimizer:
    def __init__(self):
        self.param_groups = [{"lr": 0.1}]
        self.step_called = False
        self.zero_grad_set_to_none = None

    def step(self):
        self.step_called = True

    def zero_grad(self, set_to_none=False):
        self.zero_grad_set_to_none = set_to_none


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.update_moe_gate_bias_called = False

    def update_moe_gate_bias(self):
        self.update_moe_gate_bias_called = True


@pytest.mark.cuda(False)
@pytest.mark.parametrize(
    ("pp_enabled", "expected_aux_scale"),
    [
        (False, 4.0),
        (True, 3.0),
    ],
)
def test_vlm_kd_train_step_uses_distributed_step_helpers(monkeypatch, pp_enabled, expected_aux_scale):
    calls = {
        "prepare": [],
        "final": [],
        "scale": [],
        "forward": [],
    }

    monkeypatch.setattr(vlm_kd.time, "perf_counter", lambda: 2.0)
    monkeypatch.setattr(
        vlm_kd,
        "prepare_for_grad_accumulation",
        lambda model_parts, pp_enabled: calls["prepare"].append((model_parts, pp_enabled)),
    )
    monkeypatch.setattr(
        vlm_kd,
        "prepare_for_final_backward",
        lambda model_parts, pp_enabled: calls["final"].append((model_parts, pp_enabled)),
    )

    def _fake_scale_grads_and_clip_grad_norm(**kwargs):
        calls["scale"].append(kwargs)
        return 2.5

    monkeypatch.setattr(vlm_kd, "scale_grads_and_clip_grad_norm", _fake_scale_grads_and_clip_grad_norm)

    recipe = vlm_kd.KnowledgeDistillationRecipeForVLM.__new__(vlm_kd.KnowledgeDistillationRecipeForVLM)
    model = _Model()
    optimizer = _Optimizer()
    recipe.model_parts = [model]
    recipe.pp_enabled = pp_enabled
    recipe.device_mesh = None
    recipe.moe_mesh = SimpleNamespace(mesh_dim_names=("ep",))
    recipe.optimizer = [optimizer]
    recipe.lr_scheduler = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.cfg = _Cfg()
    recipe.timestamp = 1.0
    recipe.step_scheduler = SimpleNamespace(step=7, epoch=1)
    recipe.kd_ratio = 0.5
    recipe.kd_loss_fn = SimpleNamespace(temperature=1.0)
    recipe._ce_loss_buffer = []
    recipe._kd_loss_buffer = []
    recipe._dp_allreduce = lambda tensor, include_cp=False: tensor
    recipe._get_dp_group_size = lambda include_cp=False: 4
    recipe._get_cp_group_size = lambda: 2

    def _fake_forward_backward_step(idx, batch, *, loss_buffer, num_label_tokens, num_batches, is_train=True):
        calls["forward"].append((idx, num_label_tokens, num_batches, is_train))
        loss_buffer.append(torch.tensor(1.0))
        recipe._ce_loss_buffer.append(torch.tensor(0.25))
        recipe._kd_loss_buffer.append(torch.tensor(0.75))

    recipe._forward_backward_step = _fake_forward_backward_step

    batches = [
        {"labels": torch.tensor([[1, -100, 2]])},
        {"labels": torch.tensor([[-100, 3, -100]])},
    ]

    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    metrics = recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert isinstance(metrics, MetricsSample)
    assert calls["prepare"] == [([model], pp_enabled)]
    assert calls["final"] == [([model], pp_enabled)]
    assert calls["forward"] == [(0, 3, 2, True), (1, 3, 2, True)]
    assert calls["scale"] == [
        {
            "max_grad_norm": 1.0,
            "model_parts": [model],
            "norm_type": 2.0,
            "pp_enabled": pp_enabled,
            "device_mesh": None,
            "moe_mesh": recipe.moe_mesh,
            "ep_axis_name": "ep",
            "pp_axis_name": "pp" if pp_enabled else None,
            "foreach": True,
            "num_label_tokens": 3,
            "dp_group_size": 4,
        }
    ]
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(expected_aux_scale)
    assert optimizer.step_called
    assert optimizer.zero_grad_set_to_none is True
    assert model.update_moe_gate_bias_called
    assert metrics.metrics["grad_norm"] == 2.5
    assert metrics.metrics["loss"] == 2.0
    assert metrics.metrics["ce_loss"] == pytest.approx(0.5)
    assert metrics.metrics["kd_loss"] == pytest.approx(1.5)
    assert metrics.metrics["tps"] == 5.0
    assert metrics.metrics["tps_per_gpu"] == pytest.approx(5.0 / 2 / 4)
    assert metrics.metrics["num_label_tokens"] == 3

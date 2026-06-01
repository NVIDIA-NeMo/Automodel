# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.diffusion.train import (
    _build_optimizer,
    _resolve_model_dtypes,
    _validate_precision_configuration,
)


class RecordingOptimizer(torch.optim.Optimizer):
    """Tiny optimizer used to verify config-driven optimizer construction."""

    def __init__(
        self,
        params,
        lr,
        master_weight_dtype=None,
        exp_avg_dtype=None,
        exp_avg_sq_dtype=None,
        marker=None,
    ):
        super().__init__(params, {"lr": lr})
        self.master_weight_dtype = master_weight_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_sq_dtype
        self.marker = marker

    def step(self, closure=None):  # pragma: no cover - not needed for construction tests
        if closure is not None:
            return closure()
        return None


def _param():
    return torch.nn.Parameter(torch.ones(1))


def test_build_optimizer_keeps_legacy_adamw_config_behavior():
    cfg = ConfigNode({"weight_decay": 0.02, "betas": [0.8, 0.9]})

    optimizer = _build_optimizer([_param()], cfg, learning_rate=1.0e-4)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1.0e-4
    assert optimizer.param_groups[0]["weight_decay"] == 0.02
    assert tuple(optimizer.param_groups[0]["betas"]) == (0.8, 0.9)


def test_build_optimizer_instantiates_target_config_and_resolves_dtype_strings():
    cfg = ConfigNode(
        {
            "_target_": "tests.unit_tests.recipes.test_diffusion_train.RecordingOptimizer",
            "master_weight_dtype": "torch.float32",
            "exp_avg_dtype": "bfloat16",
            "exp_avg_sq_dtype": "float16",
            "marker": "from-config",
        }
    )

    optimizer = _build_optimizer([_param()], cfg, learning_rate=2.0e-4)

    # ConfigNode imports the target from its dotted path, which can create a
    # second module object under some pytest import modes. Check behavior rather
    # than class identity.
    assert optimizer.__class__.__name__ == "RecordingOptimizer"
    assert optimizer.param_groups[0]["lr"] == 2.0e-4
    assert optimizer.master_weight_dtype is torch.float32
    assert optimizer.exp_avg_dtype is torch.bfloat16
    assert optimizer.exp_avg_sq_dtype is torch.float16
    assert optimizer.marker == "from-config"
    assert cfg.master_weight_dtype is torch.float32
    assert cfg.exp_avg_dtype is torch.bfloat16
    assert cfg.exp_avg_sq_dtype is torch.float16


def test_resolve_model_dtypes_preserves_existing_bf16_defaults():
    cfg = ConfigNode({"model": {}})

    model_dtype, compute_dtype = _resolve_model_dtypes(cfg)

    assert model_dtype is torch.bfloat16
    assert compute_dtype is torch.bfloat16


def test_resolve_model_dtypes_allows_split_storage_and_compute_dtype():
    cfg = ConfigNode({"model": {"torch_dtype": "float32", "compute_dtype": "bfloat16"}})

    model_dtype, compute_dtype = _resolve_model_dtypes(cfg)

    assert model_dtype is torch.float32
    assert compute_dtype is torch.bfloat16


def test_validate_precision_configuration_allows_split_dtype_for_fsdp_full_training():
    _validate_precision_configuration(
        torch.float32,
        torch.bfloat16,
        ddp_cfg=None,
        peft_cfg=None,
    )


@pytest.mark.parametrize(
    ("ddp_cfg", "peft_cfg", "expected_mode"),
    [
        ({"world_size": 1}, None, "DDP"),
        (None, object(), "PEFT/LoRA"),
    ],
)
def test_validate_precision_configuration_rejects_split_dtype_without_fsdp_param_cast(
    ddp_cfg,
    peft_cfg,
    expected_mode,
):
    with pytest.raises(ValueError, match=expected_mode):
        _validate_precision_configuration(
            torch.float32,
            torch.bfloat16,
            ddp_cfg=ddp_cfg,
            peft_cfg=peft_cfg,
        )


def test_validate_precision_configuration_allows_matching_dtype_for_ddp_or_peft():
    _validate_precision_configuration(
        torch.bfloat16,
        torch.bfloat16,
        ddp_cfg={"world_size": 1},
        peft_cfg=object(),
    )

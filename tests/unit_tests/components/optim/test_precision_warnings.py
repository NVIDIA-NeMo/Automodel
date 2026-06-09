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

import logging
from types import SimpleNamespace

import torch

from nemo_automodel.components.optim import precision_warnings
from nemo_automodel.components.optim.precision_warnings import (
    resolve_storage_dtype,
    warn_if_torch_adam_with_bf16_params,
)

_WARNING_PREFIX = "Detected torch.optim.Adam/AdamW with trainable bf16 model parameters"
_RESOLVE_PREFIX = "Defaulting model.torch_dtype=float32"


def setup_function():
    precision_warnings._WARNED_CONTEXTS.clear()
    precision_warnings._DTYPE_RESOLVED_CONTEXTS.clear()


def test_warns_once_for_full_bf16_training_with_torch_adamw(caplog):
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([param], lr=1.0e-4)

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(optimizer=optimizer, context="unit-test")
        warn_if_torch_adam_with_bf16_params(optimizer=optimizer, context="unit-test")

    assert caplog.text.count(_WARNING_PREFIX) == 1
    assert "docs/guides/mixed-precision-training.md" in caplog.text


def test_skips_peft_bf16_training(caplog):
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([param], lr=1.0e-4)

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(optimizer=optimizer, is_peft=True, context="unit-test")

    assert _WARNING_PREFIX not in caplog.text


def test_skips_full_fp32_training(caplog):
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1.0e-4)

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(optimizer=optimizer, context="unit-test")

    assert _WARNING_PREFIX not in caplog.text


def test_warns_from_optimizer_config_target_and_parameters(caplog):
    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer_cfg = SimpleNamespace(_target_="torch.optim.AdamW")

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(
            optimizer_cfg=optimizer_cfg,
            parameters=[param],
            context="unit-test",
        )

    assert _WARNING_PREFIX in caplog.text


def test_resolve_defaults_fp32_for_full_param_torch_adamw(caplog):
    cfg_model = SimpleNamespace()
    cfg_opt = SimpleNamespace(_target_="torch.optim.AdamW")

    with caplog.at_level(logging.INFO):
        resolve_storage_dtype(cfg_model, cfg_opt, is_peft=False, context="unit-test")

    assert cfg_model.torch_dtype == "float32"
    assert _RESOLVE_PREFIX in caplog.text


def test_resolve_treats_auto_as_unset(caplog):
    cfg_model = SimpleNamespace(torch_dtype="auto")
    cfg_opt = SimpleNamespace(_target_="torch.optim.AdamW")

    resolve_storage_dtype(cfg_model, cfg_opt, is_peft=False, context="unit-test")

    assert cfg_model.torch_dtype == "float32"


def test_resolve_honors_explicit_dtype():
    cfg_model = SimpleNamespace(torch_dtype="bfloat16")
    cfg_opt = SimpleNamespace(_target_="torch.optim.AdamW")

    resolve_storage_dtype(cfg_model, cfg_opt, is_peft=False, context="unit-test")

    assert cfg_model.torch_dtype == "bfloat16"


def test_resolve_skips_peft():
    cfg_model = SimpleNamespace()
    cfg_opt = SimpleNamespace(_target_="torch.optim.AdamW")

    resolve_storage_dtype(cfg_model, cfg_opt, is_peft=True, context="unit-test")

    assert getattr(cfg_model, "torch_dtype", None) is None


def test_resolve_defaults_fp32_for_full_param_torch_sgd(caplog):
    # The fp32-master rationale generalizes to any in-place torch.optim optimizer,
    # not just Adam/AdamW.
    cfg_model = SimpleNamespace()
    cfg_opt = SimpleNamespace(_target_="torch.optim.SGD")

    with caplog.at_level(logging.INFO):
        resolve_storage_dtype(cfg_model, cfg_opt, is_peft=False, context="unit-test")

    assert cfg_model.torch_dtype == "float32"
    assert _RESOLVE_PREFIX in caplog.text


def test_resolve_skips_non_torch_optimizer():
    # Optimizers outside the torch.optim namespace (TE FusedAdam, DeepSpeed,
    # bitsandbytes, ...) manage their own master / state precision.
    cfg_model = SimpleNamespace()
    cfg_opt = SimpleNamespace(_target_="transformer_engine.pytorch.optimizers.FusedAdam")

    resolve_storage_dtype(cfg_model, cfg_opt, is_peft=False, context="unit-test")

    assert getattr(cfg_model, "torch_dtype", None) is None


def test_resolve_works_with_dict_configs():
    cfg_model = {}
    cfg_opt = {"_target_": "torch.optim.Adam"}

    resolve_storage_dtype(cfg_model, cfg_opt, is_peft=False, context="unit-test")

    assert cfg_model["torch_dtype"] == "float32"


def test_resolve_logs_once_per_context(caplog):
    cfg_opt = SimpleNamespace(_target_="torch.optim.AdamW")

    with caplog.at_level(logging.INFO):
        resolve_storage_dtype(SimpleNamespace(), cfg_opt, is_peft=False, context="unit-test")
        resolve_storage_dtype(SimpleNamespace(), cfg_opt, is_peft=False, context="unit-test")

    assert caplog.text.count(_RESOLVE_PREFIX) == 1

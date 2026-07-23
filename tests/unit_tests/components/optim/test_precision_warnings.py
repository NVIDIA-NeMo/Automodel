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
from nemo_automodel.components.optim.optimizer import build_optimizer_config
from nemo_automodel.components.optim.precision_warnings import (
    resolve_storage_dtype,
    warn_if_torch_adam_with_bf16_params,
)

_WARNING_PREFIX = "Detected torch.optim.Adam/AdamW with trainable bf16 model parameters"
_RESOLVE_PREFIX = "Defaulting the effective model.torch_dtype to float32"


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


def test_resolve_defaults_fp32_for_optimizer_without_separate_master_weights(caplog):
    with caplog.at_level(logging.INFO):
        resolved = resolve_storage_dtype(
            None,
            uses_model_params_as_master_weights=True,
            is_peft=False,
            context="unit-test",
        )

    assert resolved == "float32"
    assert _RESOLVE_PREFIX in caplog.text


def test_resolve_treats_auto_as_unset(caplog):
    resolved = resolve_storage_dtype(
        "auto",
        uses_model_params_as_master_weights=True,
        is_peft=False,
        context="unit-test",
    )

    assert resolved == "float32"


def test_resolve_honors_explicit_dtype():
    resolved = resolve_storage_dtype(
        "bfloat16",
        uses_model_params_as_master_weights=True,
        is_peft=False,
        context="unit-test",
    )

    assert resolved == "bfloat16"


def test_resolve_skips_peft():
    resolved = resolve_storage_dtype(
        None,
        uses_model_params_as_master_weights=True,
        is_peft=True,
        context="unit-test",
    )

    assert resolved is None


def test_resolve_skips_optimizer_with_separate_master_weights():
    resolved = resolve_storage_dtype(
        None,
        uses_model_params_as_master_weights=False,
        is_peft=False,
        context="unit-test",
    )

    assert resolved is None


def test_optimizer_configs_expose_master_weight_ownership():
    assert build_optimizer_config("adam").uses_model_params_as_master_weights()
    assert build_optimizer_config("adamw").uses_model_params_as_master_weights()
    assert build_optimizer_config("torch.optim.SGD").uses_model_params_as_master_weights()

    assert not build_optimizer_config("fused_adam").uses_model_params_as_master_weights()
    assert not build_optimizer_config("flash_adamw").uses_model_params_as_master_weights()


def test_resolve_logs_once_per_context(caplog):
    with caplog.at_level(logging.INFO):
        resolve_storage_dtype(None, uses_model_params_as_master_weights=True, context="unit-test")
        resolve_storage_dtype(None, uses_model_params_as_master_weights=True, context="unit-test")

    assert caplog.text.count(_RESOLVE_PREFIX) == 1

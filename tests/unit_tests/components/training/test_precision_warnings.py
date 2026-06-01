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

import torch

from nemo_automodel.components.training import precision_warnings
from nemo_automodel.components.training.precision_warnings import warn_if_torch_adam_with_bf16_params

_WARNING_PREFIX = "Detected torch.optim.Adam/AdamW with trainable bf16 model parameters"


def setup_function():
    precision_warnings._WARNED_CONTEXTS.clear()


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


def test_warns_from_optimizer_from_factory_config(caplog):
    from nemo_automodel.components.optim.optimizer import OptimizerFromFactoryConfig

    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer_cfg = OptimizerFromFactoryConfig(factory=torch.optim.AdamW, kwargs={"lr": 1.0e-4})

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(
            optimizer_cfg=optimizer_cfg,
            parameters=[param],
            context="unit-test",
        )

    assert _WARNING_PREFIX in caplog.text


def test_warns_from_typed_adamw_config(caplog):
    from nemo_automodel.components.optim.optimizer import AdamWConfig

    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(
            optimizer_cfg=AdamWConfig(lr=1.0e-4),
            parameters=[param],
            context="unit-test",
        )

    assert _WARNING_PREFIX in caplog.text


def test_skips_non_torch_adam_factory_config(caplog):
    from nemo_automodel.components.optim.optimizer import OptimizerFromFactoryConfig

    param = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer_cfg = OptimizerFromFactoryConfig(factory=torch.optim.SGD, kwargs={"lr": 1.0e-4})

    with caplog.at_level(logging.WARNING):
        warn_if_torch_adam_with_bf16_params(
            optimizer_cfg=optimizer_cfg,
            parameters=[param],
            context="unit-test",
        )

    assert _WARNING_PREFIX not in caplog.text

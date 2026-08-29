#!/usr/bin/env python
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

"""Validate dtype-aware Transformer Engine FusedAdam master ownership."""

import torch
import torch.nn as nn
from transformer_engine.pytorch.optimizers import FusedAdam

from nemo_automodel.components.optim.optimizer import FusedAdamConfig, OptimizerFromFactoryConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TE FusedAdam master-ownership validation requires CUDA")

    model = nn.Module().cuda()
    model.fp32_weight = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.float32))
    model.bf16_weight = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    optimizer = FusedAdamConfig(lr=1e-3, master_weights=True).build(model)[0]

    if set(optimizer.state[model.fp32_weight]) != {"exp_avg", "exp_avg_sq"}:
        raise AssertionError("resident FP32 parameter should own moments without a redundant master_param")
    if optimizer.state[model.bf16_weight]:
        raise AssertionError("BF16 optimizer state should remain lazy before the first step")

    (model.fp32_weight.sum() + model.bf16_weight.float().sum()).backward()
    optimizer.step()
    if "master_param" in optimizer.state[model.fp32_weight]:
        raise AssertionError("TE created a redundant master_param for a resident FP32 parameter")
    if "master_param" not in optimizer.state[model.bf16_weight]:
        raise AssertionError("TE must retain an FP32 master_param for a BF16 resident parameter")

    optimizer.load_state_dict(optimizer.state_dict())
    if "master_param" in optimizer.state[model.fp32_weight]:
        raise AssertionError("optimizer resume restored a redundant FP32 master_param")
    if "master_param" not in optimizer.state[model.bf16_weight]:
        raise AssertionError("optimizer resume dropped the BF16 parameter's FP32 master_param")

    # Hydra/factory configs may materialize an unset optional dtype as None.
    # Construction must preserve TE's omission-based default instead of passing
    # None, which TE rejects.
    factory_model = nn.Linear(2, 2, device="cuda", dtype=torch.bfloat16)
    OptimizerFromFactoryConfig(
        factory=FusedAdam,
        kwargs={"lr": 1e-3, "master_weights": True, "master_weight_dtype": None},
    ).build(factory_model)

    print("PASS: TE dtype-aware master ownership, resume, and factory defaults")


if __name__ == "__main__":
    main()

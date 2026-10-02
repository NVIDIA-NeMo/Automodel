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

import tempfile

import torch
import torch.nn as nn
from transformer_engine.pytorch.optimizers import FusedAdam

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.optim.optimizer import FusedAdamConfig, OptimizerFromFactoryConfig


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("TE FusedAdam master-ownership validation requires CUDA")

    model = nn.Module().cuda()
    model.fp32_weight = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.float32))
    model.bf16_weight = nn.Parameter(torch.ones(8, device="cuda", dtype=torch.bfloat16))
    optimizer = FusedAdamConfig(lr=1e-3, master_weights=True).build(model)[0]

    if optimizer.state:
        raise AssertionError("fresh TE optimizer must remain empty for DCP read-template initialization")

    (model.fp32_weight.sum() + model.bf16_weight.float().sum()).backward()
    optimizer.step()
    if "master_param" in optimizer.state[model.fp32_weight]:
        raise AssertionError("TE created a redundant master_param for a resident FP32 parameter")
    if "master_param" not in optimizer.state[model.bf16_weight]:
        raise AssertionError("TE must retain an FP32 master_param for a BF16 resident parameter")

    checkpoint = optimizer.state_dict()
    for saved_group, live_group in zip(checkpoint["param_groups"], optimizer.param_groups, strict=True):
        for saved_id, parameter in zip(saved_group["params"], live_group["params"], strict=True):
            if parameter is model.fp32_weight:
                checkpoint["state"][saved_id]["master_param"] = model.fp32_weight.detach().clone()
    optimizer.load_state_dict(checkpoint)
    if "master_param" in optimizer.state[model.fp32_weight]:
        raise AssertionError("optimizer resume restored a redundant FP32 master_param")
    if "master_param" not in optimizer.state[model.bf16_weight]:
        raise AssertionError("optimizer resume dropped the BF16 parameter's FP32 master_param")

    # Resume through the real Checkpointer/DCP path into a fresh optimizer.
    # The saved FP32 master contains bits absent from BF16 resident weights.
    resumed_model = nn.Module().cuda()
    resumed_model.fp32_weight = nn.Parameter(model.fp32_weight.detach().clone())
    resumed_model.bf16_weight = nn.Parameter(model.bf16_weight.detach().clone())
    resumed_optimizer = FusedAdamConfig(lr=1e-3, master_weights=True).build(resumed_model)[0]
    with tempfile.TemporaryDirectory(prefix="te-master-resume-") as checkpoint_dir:
        checkpointer = Checkpointer(
            CheckpointingConfig(checkpoint_dir=checkpoint_dir, is_async=False), dp_rank=0, tp_rank=0, pp_rank=0
        )
        try:
            checkpointer.save_optimizer(optimizer, model, checkpoint_dir)
            checkpointer.load_optimizer(resumed_optimizer, resumed_model, checkpoint_dir)
        finally:
            checkpointer.close()
    for name, parameter in model.named_parameters():
        resumed_parameter = dict(resumed_model.named_parameters())[name]
        for state_name, expected in optimizer.state[parameter].items():
            torch.testing.assert_close(resumed_optimizer.state[resumed_parameter][state_name], expected, rtol=0, atol=0)
    for current_model, current_optimizer in ((model, optimizer), (resumed_model, resumed_optimizer)):
        current_optimizer.zero_grad(set_to_none=True)
        (current_model.fp32_weight.sum() + current_model.bf16_weight.float().sum()).backward()
        current_optimizer.step()
    for name, parameter in model.named_parameters():
        resumed_parameter = dict(resumed_model.named_parameters())[name]
        torch.testing.assert_close(resumed_parameter, parameter, rtol=0, atol=0)
        for state_name, expected in optimizer.state[parameter].items():
            torch.testing.assert_close(resumed_optimizer.state[resumed_parameter][state_name], expected, rtol=0, atol=0)

    # Hydra/factory configs may materialize an unset optional dtype as None.
    # Construction must preserve TE's omission-based default instead of passing
    # None, which TE rejects.
    factory_model = nn.Linear(2, 2, device="cuda", dtype=torch.bfloat16)
    OptimizerFromFactoryConfig(
        factory=FusedAdam,
        kwargs={"lr": 1e-3, "master_weights": True, "master_weight_dtype": None},
    ).build(factory_model)

    print("PASS: TE dtype-aware master ownership, fresh DCP resume/next-step parity, and factory defaults")


if __name__ == "__main__":
    main()

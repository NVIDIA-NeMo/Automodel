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

from __future__ import annotations

import logging
from typing import Any

from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler

logger = logging.getLogger(__name__)


def build_lr_scheduler(cfg: Any, optimizer: Any, step_scheduler: Any) -> list[OptimizerParamScheduler] | None:
    """Build the learning rate scheduler.

    Args:
        cfg: Configuration for the OptimizerParamScheduler.
        optimizer: The optimizer to be scheduled.
        step_scheduler: The step scheduler to extract training parameters.

    Returns:
        Configured optimizer parameter schedulers, or None if not configured.
    """
    if cfg is None:
        return None

    # Calculate total steps for the training run
    total_epochs = step_scheduler.num_epochs
    epoch_len = len(step_scheduler.dataloader)
    grad_acc_steps = step_scheduler.grad_acc_steps

    # Total optimizer steps (accounting for gradient accumulation)
    total_steps = (total_epochs * epoch_len) // grad_acc_steps
    if step_scheduler.max_steps is not None:
        total_steps = min(total_steps, step_scheduler.max_steps)

    optimizer_param_schedulers = []
    user_kwargs = cfg.to_dict()
    default_kwargs = dict(
        lr_warmup_steps=min(1000, total_steps // 10),  # 10% warmup or max 1000 steps
        lr_decay_steps=total_steps,
        lr_decay_style="cosine",
        wd_incr_steps=total_steps,
        wd_incr_style="constant",
    )

    if not isinstance(optimizer, list):
        optimizer = [optimizer]

    for opt in optimizer:
        base_lr = opt.param_groups[0]["lr"]
        default_kwargs.update(
            dict(
                optimizer=opt,
                init_lr=base_lr * 0.1,  # Start warmup at 10% of base LR
                max_lr=base_lr,
                min_lr=base_lr * 0.01,  # End at 1% of base LR
                start_wd=opt.param_groups[0].get("weight_decay", 0.0),
                end_wd=opt.param_groups[0].get("weight_decay", 0.0),
            )
        )
        default_kwargs.update(user_kwargs)
        optimizer_param_schedulers.append(OptimizerParamScheduler(**default_kwargs))

    logger.info(
        f"Building LR scheduler with total_steps={total_steps}, "
        f"warmup_steps={default_kwargs['lr_warmup_steps']}, "
        f"decay_style={default_kwargs['lr_decay_style']}"
    )

    return optimizer_param_schedulers

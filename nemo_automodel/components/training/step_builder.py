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

from typing import Any

from nemo_automodel.components.training.step_scheduler import StepScheduler


def build_step_scheduler(cfg: Any, dataloader: Any, dp_group_size: int, local_batch_size: int) -> StepScheduler:
    """Build the step scheduler.

    Args:
        cfg: Configuration for the StepScheduler class.
        dataloader: The training dataloader, used for extracting the epoch_len in batches.
        dp_group_size: The size of the data parallel group.
        local_batch_size: The size of the local batch.

    Returns:
        Configured StepScheduler.
    """
    assert "_target_" not in cfg, "_target_ not permitted in step scheduler"
    default_kwargs = dict(
        num_epochs=10,
        global_batch_size=32,
        local_batch_size=local_batch_size,
        dp_size=dp_group_size,
        ckpt_every_steps=100,
        dataloader=dataloader,
    )
    if cfg is not None:
        default_kwargs |= cfg.to_dict()
    return StepScheduler(**default_kwargs)

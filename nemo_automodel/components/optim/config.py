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

"""Public config surface for the optim component.

Look here for the typed parameters that drive optimizer and LR scheduling.
Look at ``api.py`` for the builder functions that consume these configs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LRSchedulerConfig:
    """User-facing LR scheduler configuration.

    All fields are optional — the builder in ``api.py`` computes sensible
    defaults from the training schedule (total steps, optimizer base LR, etc.)
    for any field left as ``None``.

    Attributes:
        lr_warmup_steps: Number of linear warmup steps.  Default: min(1000, 10% of total steps).
        lr_decay_steps: Total steps over which the LR decays.  Default: total training steps.
        lr_decay_style: Decay curve — ``"cosine"``, ``"linear"``, ``"constant"``, ``"WSD"``,
            or ``"inverse-square-root"``.
        init_lr: LR at the start of warmup.  Default: 10% of base LR.
        max_lr: Peak LR after warmup.  Default: optimizer base LR.
        min_lr: Floor LR at end of decay.  Default: 1% of base LR.
        start_wd: Initial weight decay.  Default: optimizer weight_decay.
        end_wd: Final weight decay.  Default: same as ``start_wd``.
        wd_incr_steps: Steps over which WD ramps.  Default: ``lr_decay_steps``.
        wd_incr_style: WD ramp curve — ``"constant"``, ``"linear"``, or ``"cosine"``.
        use_checkpoint_opt_param_scheduler: Use checkpoint values when resuming.
        override_opt_param_scheduler: Force class values over checkpoint values.
        wsd_decay_steps: Decay steps for the WSD schedule tail.  Required when
            ``lr_decay_style="WSD"``.
        lr_wsd_decay_style: Sub-curve for the WSD tail — ``"linear"``, ``"cosine"``,
            ``"exponential"``, or ``"minus_sqrt"``.
    """

    lr_warmup_steps: int | None = None
    lr_decay_steps: int | None = None
    lr_decay_style: str = "cosine"
    init_lr: float | None = None
    max_lr: float | None = None
    min_lr: float | None = None
    start_wd: float | None = None
    end_wd: float | None = None
    wd_incr_steps: int | None = None
    wd_incr_style: str = "constant"
    use_checkpoint_opt_param_scheduler: bool = True
    override_opt_param_scheduler: bool = False
    wsd_decay_steps: int | None = None
    lr_wsd_decay_style: str | None = None


__all__ = ["LRSchedulerConfig"]

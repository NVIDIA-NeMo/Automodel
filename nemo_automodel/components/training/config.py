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

"""Public config surface for the training component.

Look here for the typed parameters that drive step scheduling.
Look at ``api.py`` for the builder functions that consume these configs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepSchedulerConfig:
    """User-facing step scheduler configuration.

    These fields correspond to the YAML-configurable parameters of the
    training loop.  Runtime-only values (``dataloader``, ``dp_size``,
    ``local_batch_size``) are passed separately to ``build_step_scheduler``
    in ``api.py``.

    Attributes:
        global_batch_size: Total samples per optimizer step across all GPUs.
        num_epochs: Number of training epochs.  When ``None`` the builder
            derives it from ``max_steps``.  Default: 10.
        max_steps: Hard cap on optimizer steps.  ``None`` means derive from
            ``num_epochs * epoch_len``.
        ckpt_every_steps: Save a checkpoint every N optimizer steps.
            ``None`` defaults to once per epoch.
        save_checkpoint_every_epoch: Also checkpoint at every epoch boundary.
        val_every_steps: Run validation every N optimizer steps.
            ``None`` disables periodic validation.
        log_remote_every_steps: Log to WandB / MLflow every N steps.
        gc_every_steps: Force ``gc.collect()`` every N steps.
            ``None`` disables manual GC.
        start_step: Initial global step (for checkpoint resume).
        start_epoch: Initial epoch (for checkpoint resume).
    """

    global_batch_size: int = 32
    num_epochs: int | None = 10
    max_steps: int | None = None
    ckpt_every_steps: int | None = 100
    save_checkpoint_every_epoch: bool = True
    val_every_steps: int | None = None
    log_remote_every_steps: int = 1
    gc_every_steps: int | None = None
    start_step: int = 0
    start_epoch: int = 0


__all__ = ["StepSchedulerConfig"]

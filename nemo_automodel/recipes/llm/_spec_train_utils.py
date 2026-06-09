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

"""Shared training-loop utilities for the speculative-decoding recipes.

EAGLE-1/2, EAGLE-3, and DFlash all hand-roll the same gradient-accumulation
bookkeeping (ceil optimizer-steps-per-epoch and the DDP ``no_sync`` skip) and the
same warmup + cosine LR schedule. Centralizing them here keeps the recipes from
drifting apart when one is fixed and the others are missed.
"""

from __future__ import annotations

import math
from collections.abc import Callable


def optim_steps_per_epoch(num_batches_per_epoch: int, grad_accumulation_steps: int) -> int:
    """Return ceil(num_batches / accum), the actual number of optimizer steps per epoch.

    Floor division silently drops the trailing partial accumulation window
    (up to ``grad_accumulation_steps - 1`` micro-batches) from the LR
    scheduler's view of training, even though the trainer flushes those
    gradients with an explicit step. Ceil keeps the scheduler aligned with
    the actual number of ``optimizer.step()`` calls.
    """
    if num_batches_per_epoch <= 0 or grad_accumulation_steps <= 0:
        return 0
    return -(-num_batches_per_epoch // grad_accumulation_steps)


def should_sync_grads(
    *,
    pending_micro_batches: int,
    grad_accumulation_steps: int,
    batch_idx: int,
    batches_per_epoch: int | None,
    is_ddp: bool,
) -> bool:
    """Return True when this micro-batch's backward should all-reduce gradients.

    Under DDP with gradient accumulation only the micro-batch immediately
    followed by an ``optimizer.step()`` needs to synchronize: at that point the
    locally-accumulated ``.grad`` already holds the whole window's contribution,
    so a single all-reduce averages the complete window and the intervening
    micro-batches can run under ``no_sync()`` -- saving ``grad_accumulation_steps - 1``
    all-reduces per window. That step is either the window closer
    (``pending_micro_batches + 1 == grad_accumulation_steps``) or the epoch's
    final batch (which the trailing-flush step consumes). When the dataloader
    length is unknown we cannot identify the final batch, so we sync every step
    (correct, just no speedup). With a single process (no DDP) there is nothing
    to synchronize, so this is always True.
    """
    if not is_ddp or batches_per_epoch is None:
        return True
    closes_window = pending_micro_batches + 1 == grad_accumulation_steps
    is_last_batch = batch_idx == batches_per_epoch - 1
    return closes_window or is_last_batch


def make_warmup_cosine_schedule(
    warmup_steps: int, total_optim_steps: int, min_lr_ratio: float
) -> Callable[[int], float]:
    """Build the ``LambdaLR`` multiplier: linear warmup then cosine decay to ``min_lr_ratio``.

    Linear from 0 to 1 over the first ``warmup_steps`` optimizer steps, then a
    cosine from 1 down to ``min_lr_ratio`` over the remaining steps. Shared by the
    EAGLE and DFlash recipes, which train the draft from scratch and diverge under
    a flat LR after the first epoch.
    """

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _lr_lambda

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

"""Unit tests for the shared speculative-recipe training utilities.

These pin the grad-accumulation bookkeeping and the warmup+cosine LR schedule
that EAGLE-1/2, EAGLE-3, DFlash, and DSpark share, plus the draft
activation-checkpointing wiring that DFlash and DSpark share (EAGLE-1/2 and
EAGLE-3 do not call it yet), so a change is caught for all current callers at
once (the drift these utilities exist to prevent). Recipe-level integration of
the same logic lives in each recipe's own tests.
"""

import math

import pytest
import torch

from nemo_automodel.recipes.llm._spec_train_utils import (
    apply_draft_activation_checkpointing,
    make_warmup_cosine_schedule,
    optim_steps_per_epoch,
    should_sync_grads,
)

# ---------------------------------------------------------------------------
# optim_steps_per_epoch (ceil division)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "num_batches,accum,expected",
    [
        (10, 1, 10),
        (10, 2, 5),
        (10, 3, 4),  # 3 full windows + 1 trailing micro-batch -> 4 steps
        (10, 4, 3),  # 2 full windows + 2 trailing -> 3 steps
        (1, 4, 1),  # entire epoch is one trailing flush
        (4, 4, 1),
        (5, 4, 2),
        (0, 4, 0),  # iterable dataloader / no length
    ],
)
def test_optim_steps_per_epoch_uses_ceil_division(num_batches, accum, expected):
    assert optim_steps_per_epoch(num_batches, accum) == expected


def test_optim_steps_per_epoch_handles_invalid_inputs():
    assert optim_steps_per_epoch(0, 1) == 0
    assert optim_steps_per_epoch(-1, 4) == 0
    assert optim_steps_per_epoch(10, 0) == 0
    assert optim_steps_per_epoch(10, -1) == 0


# ---------------------------------------------------------------------------
# should_sync_grads (DDP no_sync decision)
# ---------------------------------------------------------------------------


def _sync(pending, batch_idx, *, accum=4, batches_per_epoch=10, is_ddp=True):
    return should_sync_grads(
        pending_micro_batches=pending,
        grad_accumulation_steps=accum,
        batch_idx=batch_idx,
        batches_per_epoch=batches_per_epoch,
        is_ddp=is_ddp,
    )


def test_should_sync_always_true_without_ddp():
    # Single process: nothing to all-reduce, so every step "syncs" (no_sync is
    # never entered) regardless of window position or batch index.
    for pending in range(4):
        assert _sync(pending, batch_idx=0, is_ddp=False) is True


def test_should_sync_only_on_window_close_under_ddp():
    # accum=4: interior micro-batches defer the all-reduce, the 4th closes it.
    assert _sync(0, batch_idx=0) is False
    assert _sync(1, batch_idx=1) is False
    assert _sync(2, batch_idx=2) is False
    assert _sync(3, batch_idx=3) is True  # pending+1 == accum -> window closer


def test_should_sync_on_epoch_final_batch_even_mid_window():
    # The trailing-flush step consumes the last batch's grads, so it must sync
    # even though the window is not full (batch_idx == batches_per_epoch - 1).
    assert _sync(0, batch_idx=9, batches_per_epoch=10) is True
    assert _sync(1, batch_idx=9, batches_per_epoch=10) is True


def test_should_sync_every_step_when_length_unknown():
    # IterableDataset (len unknown): we cannot identify the final batch, so a
    # trailing window could step on un-synced grads -- sync every step instead.
    for pending in range(4):
        assert _sync(pending, batch_idx=pending, batches_per_epoch=None) is True


def test_should_sync_every_step_when_accum_is_one():
    # No accumulation: each batch closes its own window -> always sync.
    for batch_idx in range(5):
        assert _sync(0, batch_idx=batch_idx, accum=1) is True


# ---------------------------------------------------------------------------
# make_warmup_cosine_schedule (linear warmup -> cosine decay)
# ---------------------------------------------------------------------------


def test_warmup_cosine_schedule_warmup_is_linear():
    sched = make_warmup_cosine_schedule(warmup_steps=4, total_optim_steps=100, min_lr_ratio=0.1)
    assert sched(0) == pytest.approx(0.25)  # (0 + 1) / 4
    assert sched(3) == pytest.approx(1.0)  # (3 + 1) / 4, end of warmup


def test_warmup_cosine_schedule_decays_to_min_ratio():
    warmup, total, min_ratio = 4, 100, 0.1
    sched = make_warmup_cosine_schedule(warmup, total, min_ratio)
    # First post-warmup step is at the cosine peak (multiplier 1.0).
    assert sched(warmup) == pytest.approx(1.0)
    # The final step decays exactly to min_lr_ratio.
    assert sched(total) == pytest.approx(min_ratio)
    # Mid-decay lies strictly between the peak and the floor.
    mid = sched((warmup + total) // 2)
    assert min_ratio < mid < 1.0


def test_warmup_cosine_schedule_clamps_past_the_end():
    sched = make_warmup_cosine_schedule(warmup_steps=2, total_optim_steps=10, min_lr_ratio=0.2)
    # progress is clamped to [0, 1], so steps past total stay at min_lr_ratio.
    assert sched(50) == pytest.approx(0.2)
    assert not math.isnan(sched(50))


# ---------------------------------------------------------------------------
# apply_draft_activation_checkpointing (recipes that DDP/fully_shard the draft
# directly bypass FSDP2Manager/DDPManager's own AC wiring, so this must be
# called explicitly)
# ---------------------------------------------------------------------------


class _DraftLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Linear(4, 4)
        self.input_layernorm = torch.nn.LayerNorm(4)
        self.post_attention_layernorm = torch.nn.LayerNorm(4)


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_DraftLayer(), _DraftLayer()])


def test_draft_activation_checkpointing_wraps_trainable_submodules():
    draft = _Draft()
    apply_draft_activation_checkpointing(draft, True)
    for layer in draft.layers:
        assert hasattr(layer.self_attn, "_checkpoint_wrapped_module")
        assert hasattr(layer.mlp, "_checkpoint_wrapped_module")
        assert hasattr(layer.input_layernorm, "_checkpoint_wrapped_module")
        assert hasattr(layer.post_attention_layernorm, "_checkpoint_wrapped_module")


def test_draft_activation_checkpointing_false_is_noop():
    draft = _Draft()
    original_attention = draft.layers[0].self_attn
    apply_draft_activation_checkpointing(draft, False)
    assert draft.layers[0].self_attn is original_attention


@pytest.mark.parametrize("spelling", ["off", "none", "disabled", "no", "OFF"])
def test_draft_activation_checkpointing_disabled_spellings_are_noop(spelling):
    """Must recognize the same disabled spellings ``_normalize_activation_checkpointing``
    does for the target, or ``distributed: {activation_checkpointing: off}`` leaves the
    target uncheckpointed while fully checkpointing the draft."""
    draft = _Draft()
    original_attention = draft.layers[0].self_attn
    apply_draft_activation_checkpointing(draft, spelling)
    assert draft.layers[0].self_attn is original_attention


def test_draft_activation_checkpointing_rejects_an_unrecognized_string():
    with pytest.raises(ValueError, match="activation_checkpointing"):
        apply_draft_activation_checkpointing(_Draft(), "sometimes")


def test_draft_activation_checkpointing_warns_when_draft_has_no_layers():
    draft = torch.nn.Module()
    # Must not raise -- a draft with an unexpected shape should be a loud warning,
    # not a crash, since fp8/compile before it already succeeded.
    apply_draft_activation_checkpointing(draft, True)

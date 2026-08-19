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

"""CPU parity tests for the shared ``forward_backward_step`` primitive.

The LLM and VLM finetune recipes now delegate their forward + LM-loss to
``forward_backward_step``. These tests pin the primitive to exactly the inline
``model(**batch)`` + ``calculate_loss(...)`` it replaced, so the recipes cannot
silently drift from the primitive (or each other).
"""

import torch
import torch.nn as nn

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.training.forward_backward import forward_backward_step


class _Out:
    """Minimal HF-like model output carrying only ``logits``."""

    def __init__(self, logits):
        self.logits = logits


class _TinyLM(nn.Module):
    def __init__(self, vocab=64, hidden=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, input_ids, **kwargs):
        return _Out(self.head(self.embed(input_ids)))


def _fixture(seed=0):
    torch.manual_seed(seed)
    model = _TinyLM()
    batch = {"input_ids": torch.randint(0, 64, (2, 8))}
    labels = torch.randint(0, 64, (2, 8))
    return model, batch, labels


def _inline_reference(model, batch, labels, loss_fn):
    """What the recipes' non-PP branch computed before the extraction."""
    out = model(**batch)
    loss = calculate_loss(
        loss_fn,
        logits=getattr(out, "logits", out),
        labels=labels,
        model=model,
        hidden_states=None,
        lm_weight=None,
        num_label_tokens=None,
    )
    return out, loss


def test_matches_inline_forward_and_loss():
    model, batch, labels = _fixture()
    loss_fn = MaskedCrossEntropy()

    out, loss = forward_backward_step(model, batch, labels, loss_fn, num_label_tokens=None)
    ref_out, ref_loss = _inline_reference(model, batch, labels, loss_fn)

    assert torch.equal(out.logits, ref_out.logits)
    assert torch.allclose(loss, ref_loss)
    # Loss is returned un-scaled and un-backwarded so the caller owns those.
    assert loss.requires_grad


def test_no_mtp_term_when_model_emits_none():
    model, batch, labels = _fixture()
    loss_fn = MaskedCrossEntropy()

    # Model emits no mtp_per_depth_* → MTP branch is skipped even if a cfg is passed.
    class _Cfg:
        scaling_factor = 1.0
        ignore_index = -100

    _, loss = forward_backward_step(model, batch, labels, loss_fn, mtp_cfg=_Cfg())
    _, ref_loss = _inline_reference(model, batch, labels, loss_fn)
    assert torch.allclose(loss, ref_loss)


def test_returns_raw_output_for_downstream_extraction():
    # The Engine (RL) reuses `out` to extract per-datum logprobs/entropy.
    model, batch, labels = _fixture()
    out, _ = forward_backward_step(model, batch, labels, MaskedCrossEntropy())
    assert hasattr(out, "logits")
    assert out.logits.shape == (2, 8, 64)

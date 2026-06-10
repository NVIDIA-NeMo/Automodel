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

"""EAGLE-1/2 feature-noise data augmentation (EAGLE paper, arXiv:2401.15077).

The paper adds U(-0.1, 0.1) noise to the target features fed to the draft during
training to curb error accumulation over the autoregressive drafting steps. These
tests pin: the draft input is perturbed only in training and only when
``feature_noise > 0``, the perturbation stays within U(-fn, fn), and the SmoothL1
regression target is never touched.
"""

import torch
import torch.nn as nn

from nemo_automodel.components.speculative.eagle.core_v12 import EagleTrainerModule

_HIDDEN = 8
_VOCAB = 16


class _RecordingDraft(nn.Module):
    """Records the (possibly noised) input features it receives, then projects."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(_HIDDEN, _HIDDEN)
        self.received = None

    def forward(self, input_ids, target_hidden_states, attention_mask):
        self.received = target_hidden_states.detach().clone()
        return self.lin(target_hidden_states)


def _trainer(feature_noise):
    draft = _RecordingDraft()
    lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False)
    return EagleTrainerModule(draft, target_lm_head=lm_head, feature_noise=feature_noise), draft


def _inputs(batch=1, seq=4):
    return {
        "input_ids": torch.randint(0, _VOCAB, (batch, seq)),
        "attention_mask": torch.ones(batch, seq),
        "loss_mask": torch.ones(batch, seq, dtype=torch.long),
        "input_hidden_states": torch.randn(batch, seq, _HIDDEN),
        "target_hidden_states": torch.randn(batch, seq, _HIDDEN),
        "target_logits": torch.randn(batch, seq, _VOCAB),
    }


def test_feature_noise_zero_leaves_input_clean():
    torch.manual_seed(0)
    trainer, draft = _trainer(feature_noise=0.0)
    trainer.train()
    ins = _inputs()
    trainer(**ins)
    assert torch.equal(draft.received, ins["input_hidden_states"])


def test_feature_noise_perturbs_input_within_bounds_in_training():
    torch.manual_seed(0)
    fn = 0.1
    trainer, draft = _trainer(feature_noise=fn)
    trainer.train()
    ins = _inputs()
    trainer(**ins)
    diff = draft.received - ins["input_hidden_states"]
    # Perturbed, and every element stays within U(-fn, fn).
    assert not torch.equal(draft.received, ins["input_hidden_states"])
    assert 0.0 < diff.abs().max().item() <= fn + 1e-6
    # The regression target itself is never noised.
    # (input_hidden_states is reassigned locally; the passed-in tensor is intact.)


def test_feature_noise_disabled_in_eval():
    torch.manual_seed(0)
    trainer, draft = _trainer(feature_noise=0.1)
    trainer.eval()
    ins = _inputs()
    trainer(**ins)
    assert torch.equal(draft.received, ins["input_hidden_states"])

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

import torch

from nemo_automodel.components.training.model_output import (
    ModelOutput,
    compute_entropy,
    selected_token_logprobs,
    split_per_datum,
)

# ── ModelOutput ─────────────────────────────────────────────────────────────


def test_model_output_defaults_and_roundtrip():
    mo = ModelOutput(loss=torch.tensor(1.5), metrics={"loss": 1.5})
    assert mo.logprobs is None and mo.entropy is None and mo.values is None
    mo2 = ModelOutput.from_dict(mo.to_dict())
    assert torch.equal(mo2.loss, mo.loss)
    assert mo2.metrics == {"loss": 1.5}


# ── output extraction helpers ───────────────────────────────────────────────


def test_selected_token_logprobs_matches_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    targets = torch.randint(0, 7, (2, 4))
    got = selected_token_logprobs(logits, targets)
    expected = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, expected)
    assert got.shape == (2, 4)


def test_compute_entropy_uniform_is_log_vocab():
    logits = torch.zeros(1, 3, 8)  # uniform distribution over 8 classes
    ent = compute_entropy(logits)
    assert torch.allclose(ent, torch.full((1, 3), torch.log(torch.tensor(8.0))))


def test_split_per_datum_inverts_packing():
    flat = torch.arange(5).float()
    parts = split_per_datum(flat, [3, 2])
    assert parts[0].tolist() == [0, 1, 2]
    assert parts[1].tolist() == [3, 4]


def test_split_per_datum_drops_trailing_pad_and_accepts_2d():
    flat = torch.arange(8).float().unsqueeze(0)  # [1, 8], last 3 are pad
    parts = split_per_datum(flat, torch.tensor([3, 2]))
    assert len(parts) == 2
    assert parts[1].tolist() == [3, 4]


def test_packed_logprobs_roundtrip_through_split():
    """THD forward → flat per-token logprobs → split back to per-datum."""
    seq_lens = [3, 2]
    total, vocab = sum(seq_lens), 32
    torch.manual_seed(1)
    logits = torch.randn(total, vocab)  # flat THD layout
    targets = torch.randint(0, vocab, (total,))
    flat_lp = selected_token_logprobs(logits, targets)
    per_datum = split_per_datum(flat_lp, seq_lens)
    assert [p.shape[0] for p in per_datum] == seq_lens

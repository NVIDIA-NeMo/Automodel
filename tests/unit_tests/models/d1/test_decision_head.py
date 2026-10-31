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

"""Answer-row gradients, marker masking, and checkpoint behavior."""

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.d1.decision_head import D1OmniHeadConfig, D1TokenReadoutConfig


def test_answer_row_adapter_preserves_tied_base_and_alias_max(tmp_path):
    torch.manual_seed(7)
    vocabulary = torch.nn.Embedding(13, 8)
    original = vocabulary.weight.detach().clone()
    config = D1TokenReadoutConfig(8, (2, 7, 4))
    adapter = config.build(vocabulary.weight)
    states = torch.randn(4, 8)
    groups = ((0, 1), (2,))
    full_logits = F.linear(states, vocabulary.weight)
    expected = torch.stack((full_logits[:, [2, 7]].amax(-1), full_logits[:, 4]), -1)
    torch.testing.assert_close(adapter(states, groups), expected)
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    F.cross_entropy(adapter(states, groups), torch.tensor([0, 1, 0, 1])).backward()
    assert adapter.delta_weight.grad.abs().sum() > 0
    assert vocabulary.weight.grad is None
    optimizer.step()
    torch.testing.assert_close(vocabulary.weight, original, rtol=0, atol=0)
    assert not torch.allclose(adapter(states, groups), expected)
    path = tmp_path / "readout.pt"
    torch.save(adapter.state_dict(), path)
    restored = config.build(vocabulary.weight)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored(states, groups), adapter(states, groups), rtol=0, atol=0)


def test_omni_padding_does_not_change_real_marker_logits_and_gradients():
    torch.manual_seed(17)
    head = D1OmniHeadConfig(64).build().eval()
    states = torch.randn(2, 5, 64, requires_grad=True)
    valid = torch.ones(2, 5, dtype=torch.bool)
    markers = torch.tensor([[1, 3], [2, 4]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    types = torch.tensor([0, 2])
    original = head(states, valid, markers, mask, types)
    padded = torch.cat((states, torch.randn(2, 3, 64) * 100), dim=1)
    padded_valid = torch.cat((valid, torch.zeros(2, 3, dtype=torch.bool)), dim=1)
    padded_logits = head(padded, padded_valid, markers, mask, types)
    torch.testing.assert_close(original, padded_logits, atol=2e-6, rtol=2e-5)
    F.cross_entropy(original, torch.tensor([1, 0])).backward()
    assert torch.isfinite(states.grad).all()
    assert states.grad.abs().sum() > 0
    assert head.type_emb.weight.grad[2].abs().sum() > 0
    masked = head(states.detach(), valid, markers, torch.tensor([[True, False], [True, True]]), types)
    assert masked[0, 1] == -10000


def test_omni_checkpoint_restores_internal_false_true_order(tmp_path):
    head = D1OmniHeadConfig(64).build().eval()
    assert "head.layers.0.self_attn.in_proj_weight" in head.state_dict()
    assert "head.layers.1.norm2.weight" in head.state_dict()
    inputs = (
        torch.randn(1, 4, 64),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[1, 2]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([2]),
    )
    path = tmp_path / "head.pt"
    torch.save(head.state_dict(), path)
    restored = D1OmniHeadConfig(64).build().eval()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    torch.testing.assert_close(restored(*inputs), head(*inputs), rtol=0, atol=0)

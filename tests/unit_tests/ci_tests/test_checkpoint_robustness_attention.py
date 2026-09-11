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

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.attention.flex_attention import FlexAttention
from tests.functional_tests.checkpoint_robustness import test_checkpoint_robustness_llm as harness


class _StubFlexAttention(FlexAttention):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Exercise callable selection without compiling a GPU kernel.

        Args:
            x: Tensor of shape [batch, sequence, vocab].

        Returns:
            Tensor of shape [batch, sequence, vocab].
        """
        return FlexAttention.flex_attn(x)


class _Model(torch.nn.Module):
    def __init__(self, attention: torch.nn.Module) -> None:
        super().__init__()
        self.attention = attention

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None, use_cache: bool) -> torch.Tensor:
        """Return toy logits through the selected attention module.

        Args:
            input_ids: Integer tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence]; unused.
            use_cache: Unused cache setting.

        Returns:
            Tensor of shape [batch, sequence, 1].
        """
        return self.attention(input_ids.float().unsqueeze(-1))


@pytest.mark.parametrize("fail_forward", [False, True])
def test_parity_attention_is_scoped_to_forward(monkeypatch, fail_forward):
    original = Mock(side_effect=lambda x: x)
    parity = Mock(side_effect=RuntimeError("forward failed") if fail_forward else lambda x: x + 1)
    monkeypatch.setattr(FlexAttention, "flex_attn", original)
    monkeypatch.setattr(harness, "_parity_flex_attention", lambda: parity)
    model = _Model(_StubFlexAttention())

    if fail_forward:
        with pytest.raises(RuntimeError, match="forward failed"):
            harness._get_logits(model, [1, 2], torch.device("cpu"))
    else:
        logits = harness._get_logits(model, [1, 2], torch.device("cpu"))
        torch.testing.assert_close(logits, torch.tensor([[[2.0], [3.0]]]), rtol=0, atol=0)

    assert FlexAttention.flex_attn is original
    original.assert_not_called()
    parity.assert_called_once()
    # A subsequent ordinary forward must use the original execution policy.
    output = model(torch.tensor([[1, 2]]), None, False)
    torch.testing.assert_close(output, torch.tensor([[[1.0], [2.0]]]), rtol=0, atol=0)
    original.assert_called_once()


def test_other_attention_backends_do_not_initialize_parity_kernel(monkeypatch):
    factory = Mock(side_effect=AssertionError("unexpected FlexAttention compilation"))
    monkeypatch.setattr(harness, "_parity_flex_attention", factory)
    model = _Model(torch.nn.Identity())

    logits = harness._get_logits(model, [1, 2], torch.device("cpu"))

    torch.testing.assert_close(logits, torch.tensor([[[1.0], [2.0]]]), rtol=0, atol=0)
    factory.assert_not_called()


def test_parity_attention_covers_later_local_pipeline_parts(monkeypatch):
    original = FlexAttention.flex_attn
    parity = Mock()
    monkeypatch.setattr(harness, "_parity_flex_attention", lambda: parity)
    first = torch.nn.Identity()
    trainer = SimpleNamespace(pp_enabled=True, model_parts=[first, _Model(_StubFlexAttention())])

    def pipeline_forward(trainer, input_ids, device):
        assert FlexAttention.flex_attn is parity
        return torch.ones(1, len(input_ids), 1)

    monkeypatch.setattr(harness, "_get_logits_pp", pipeline_forward)
    logits = harness._get_logits(first, [1, 2], torch.device("cpu"), trainer=trainer)

    torch.testing.assert_close(logits, torch.ones(1, 2, 1), rtol=0, atol=0)
    assert FlexAttention.flex_attn is original

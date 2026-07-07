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

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.components.models.gpt_oss.model import Block as GptOssBlock
from nemo_automodel.components.models.qwen3_moe.model import Block as Qwen3MoeBlock


class _RecordingMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *args):
        self.calls.append(args)
        return torch.zeros_like(args[0])


@pytest.mark.parametrize("block_cls", [GptOssBlock, Qwen3MoeBlock])
def test_moe_dispatch_uses_layer_role_after_checkpoint_wrapping(block_cls):
    x = torch.randn(2, 3, 8)
    padding_mask = torch.ones(2, 3, dtype=torch.bool)
    mlp = _RecordingMlp()
    block = SimpleNamespace(is_moe_layer=True, mlp=checkpoint_wrapper(mlp))

    output = block_cls._mlp(block, x, padding_mask)

    assert output.shape == x.shape
    assert mlp.calls == [(x, padding_mask)]


def test_qwen3_dense_dispatch_uses_layer_role_after_checkpoint_wrapping():
    x = torch.randn(2, 3, 8)
    mlp = _RecordingMlp()
    block = SimpleNamespace(is_moe_layer=False, mlp=checkpoint_wrapper(mlp))

    output = Qwen3MoeBlock._mlp(block, x, padding_mask=None)

    assert output.shape == x.shape
    assert mlp.calls == [(x,)]

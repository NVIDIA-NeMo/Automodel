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
"""Kimi-K3 ``init_weights`` must work when the embedding is already an FSDP2/TP DTensor."""

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.models.kimi_k3.model import KimiK3ForCausalLM
from tests.unit_tests.models.kimi_k3.test_pipeline_parallel import _tiny_config, _torch_backend


@pytest.fixture
def single_rank_pg():
    if dist.is_initialized():
        pytest.skip("a process group is already initialized")
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    yield
    dist.destroy_process_group()


def test_init_weights_zeroes_padding_row_on_dtensor_embedding(single_rank_pg):
    model = KimiK3ForCausalLM(_tiny_config(), backend=_torch_backend())
    text = model.model
    assert text.padding_idx == 0  # pad_token_id default of the released config
    mesh = init_device_mesh("cpu", (1,))
    text.embed_tokens.weight = nn.Parameter(distribute_tensor(text.embed_tokens.weight.detach(), mesh, [Shard(0)]))

    text.init_weights(torch.device("cpu"))

    full = text.embed_tokens.weight.full_tensor()
    assert torch.all(full[0] == 0)
    assert full[1:].abs().sum() > 0
    assert torch.all(text.norm.weight == 1)


def test_init_weights_plain_embedding_still_zeroes_padding_row():
    model = KimiK3ForCausalLM(_tiny_config(), backend=_torch_backend())
    text = model.model
    text.init_weights(torch.device("cpu"))
    assert torch.all(text.embed_tokens.weight[0] == 0)
    assert text.embed_tokens.weight[1:].abs().sum() > 0

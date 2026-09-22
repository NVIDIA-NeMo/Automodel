# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn as nn

from nemo_automodel.components.distributed.parallelizer import (
    PARALLELIZATION_STRATEGIES,
    DefaultParallelizationStrategy,
)
from nemo_automodel.components.models.mimo_v2_flash.cp import (
    MiMoCPContext,
    build_cp_attention_mask,
    shard_batch_for_mimo_cp,
)


class _FakeCPMesh:
    def __init__(self, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class _FakeWorldMesh:
    mesh_dim_names = ("cp",)

    def __init__(self, cp_mesh):
        self.cp_mesh = cp_mesh

    def __getitem__(self, name):
        assert name == "cp"
        return self.cp_mesh


class _CPAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.cp_mesh = None

    def setup_cp_attention(self, cp_mesh):
        self.cp_mesh = cp_mesh


def test_full_attention_mask_uses_global_query_offsets():
    context = MiMoCPContext(doc_ids=torch.ones(1, 8, dtype=torch.int32), seq_start=4, cp_size=2)
    mask = build_cp_attention_mask(context, dtype=torch.float32, sliding_window=None)

    assert mask.shape == (1, 1, 4, 8)
    minimum = torch.finfo(torch.float32).min
    for local_query, global_query in enumerate(range(4, 8)):
        assert torch.all(mask[0, 0, local_query, : global_query + 1] == 0)
        assert torch.all(mask[0, 0, local_query, global_query + 1 :] == minimum)


def test_sliding_attention_mask_limits_global_history():
    context = MiMoCPContext(doc_ids=torch.ones(1, 8, dtype=torch.int32), seq_start=4, cp_size=2)
    mask = build_cp_attention_mask(context, dtype=torch.float32, sliding_window=3)

    minimum = torch.finfo(torch.float32).min
    assert torch.all(mask[0, 0, 0, 2:5] == 0)
    assert torch.all(mask[0, 0, 0, :2] == minimum)
    assert torch.all(mask[0, 0, 0, 5:] == minimum)


def test_attention_mask_blocks_other_documents_and_padding():
    doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 0, 0]], dtype=torch.int32)
    context = MiMoCPContext(doc_ids=doc_ids, seq_start=4, cp_size=2)
    mask = build_cp_attention_mask(context, dtype=torch.float32, sliding_window=None)

    minimum = torch.finfo(torch.float32).min
    assert torch.all(mask[0, 0, 0, :4] == minimum)
    assert mask[0, 0, 0, 4] == 0
    assert torch.all(mask[0, 0, 1, 4:6] == 0)
    assert mask[0, 0, 2, 0] == 0
    assert torch.all(mask[0, 0, 2, 1:] == minimum)


def test_contiguous_sharder_keeps_global_doc_ids_and_slices_sequence():
    batch = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.ones(1, 8, dtype=torch.int64),
    }
    _, local, layout = shard_batch_for_mimo_cp(_FakeCPMesh(2, 1), None, batch)

    torch.testing.assert_close(local["input_ids"], torch.tensor([[4, 5, 6, 7]]))
    torch.testing.assert_close(local["labels"], torch.tensor([[4, 5, 6, 7]]))
    torch.testing.assert_close(local["position_ids"], torch.tensor([[4, 5, 6, 7]]))
    assert local["mimo_cp_doc_ids"].shape == (1, 8)
    assert local["mimo_cp_seq_start"] == 4
    assert local["mimo_cp_size"] == 2
    assert layout.original_seq_len == 8
    assert layout.padded_seq_len == 8


def test_ep1_default_strategy_attaches_cp_mesh_to_model_owned_attention():
    strategy = PARALLELIZATION_STRATEGIES["MiMoV2ForCausalLM"]
    model = nn.Module()
    model.attention = _CPAwareModule()
    cp_mesh = _FakeCPMesh(2, 0)

    with patch.object(DefaultParallelizationStrategy, "parallelize", return_value=model):
        result = strategy.parallelize(model, _FakeWorldMesh(cp_mesh))

    assert result is model
    assert model.cp_mesh is cp_mesh
    assert model.attention.cp_mesh is cp_mesh

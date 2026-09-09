# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.kimi_k3.cp import shard_batch_for_kimi_cp
from nemo_automodel.components.models.kimi_k3.model import (
    KimiDeltaAttention,
    KimiK3ForCausalLM,
    KimiMLAAttention,
)


class _FakeCPMesh:
    def size(self):
        return 2

    def get_local_rank(self):
        return 1


def test_prepare_model_inputs_returns_current_cp_sharder_api():
    model = object.__new__(KimiK3ForCausalLM)
    torch.nn.Module.__init__(model)
    model.cp_mesh = _FakeCPMesh()
    mla = object.__new__(KimiMLAAttention)
    torch.nn.Module.__init__(mla)
    mla._cp_mesh = None
    kda = object.__new__(KimiDeltaAttention)
    torch.nn.Module.__init__(kda)
    kda._cp_mesh = None
    model.add_module("mla", mla)
    model.add_module("kda", kda)

    prepared = model.prepare_model_inputs_for_cp({}, num_chunks=1)

    assert set(prepared) == {"cp_sharder"}
    assert isinstance(prepared["cp_sharder"], ContextParallelSharder)
    assert prepared["cp_sharder"].shard_batch is shard_batch_for_kimi_cp
    assert mla._cp_mesh is model.cp_mesh
    assert kda._cp_mesh is model.cp_mesh


def test_kimi_cp_sharder_keeps_contiguous_tokens_and_global_document_map():
    batch = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
    }

    _, local_batch, layout = shard_batch_for_kimi_cp(
        _FakeCPMesh(),
        None,
        batch,
        padding_token_id=99,
    )

    assert local_batch["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert local_batch["labels"].tolist() == [[4, 5, 6, 7]]
    assert local_batch["position_ids"].tolist() == [[4, 5, 6, 7]]
    assert local_batch["padding_mask"].tolist() == [[False, False, True, True]]
    assert "attention_mask" not in local_batch
    assert local_batch["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert local_batch["kimi_packed_seq_start"] == 4
    assert local_batch["kimi_packed_cp_size"] == 2
    assert layout.original_seq_len == 8
    assert layout.padded_seq_len == 8


def test_pipeline_microbatches_chunk_kimi_document_map():
    batch = {
        "input_ids": torch.arange(16).reshape(2, 8),
        "labels": torch.arange(16).reshape(2, 8),
        "attention_mask": torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0, 0, 0, 0],
            ]
        ),
    }
    _, local_batch, _ = shard_batch_for_kimi_cp(_FakeCPMesh(), None, batch)

    _, microbatches = split_args_kwargs_into_chunks((), local_batch, chunks=2)

    assert len(microbatches) == 2
    assert [microbatch["input_ids"].shape for microbatch in microbatches] == [(1, 4), (1, 4)]
    assert [microbatch["kimi_packed_doc_ids"].shape for microbatch in microbatches] == [(1, 8), (1, 8)]
    assert microbatches[0]["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 1, 1, 0, 0]]
    assert microbatches[1]["kimi_packed_doc_ids"].tolist() == [[1, 1, 1, 1, 0, 0, 0, 0]]

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

"""Model input transfer retains only the deferred attention map on the host."""

import torch

from nemo_automodel.shared.batch import move_batch_to_device


def test_deferred_ids_remain_cpu_while_nested_model_tensors_move():
    ids = torch.tensor([[1, 1, 2, 2]])
    tokens = torch.ones(1, 4, dtype=torch.long)
    batch = {
        "input_ids": tokens,
        "_packed_seq_ids": ids,
        "media": [{"pixels": torch.ones(1, 3, 4, 4)}, (tokens, None)],
    }
    moved = move_batch_to_device(batch, torch.device("meta"))
    assert moved["_packed_seq_ids"] is ids
    assert ids.device.type == "cpu"
    assert moved["input_ids"].device.type == "meta"
    assert moved["media"][0]["pixels"].device.type == "meta"
    assert moved["media"][1][0].device.type == "meta"
    assert moved["media"][1][1] is None
    assert isinstance(moved["media"][1], tuple)
    assert tokens.device.type == "cpu"


def test_prepared_attention_and_thd_move_effective_ids_to_compute_device():
    ids = torch.tensor([[1, 1, 2, 2]])
    for prepared in ({"attention_mask": ids}, {"cu_seq_lens_q": torch.tensor([0, 2, 4])}, {"qkv_format": "thd"}):
        moved = move_batch_to_device({"_packed_seq_ids": ids, **prepared}, torch.device("meta"))
        assert moved["_packed_seq_ids"].device.type == "meta"
        assert ids.device.type == "cpu"

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

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.mimo_v25.state_dict_adapter import MiMoV2StateDictAdapter


@pytest.mark.parametrize("layer_idx", [0, 1])
def test_dequantize_restores_canonical_qkv_layout(layer_idx):
    config = SimpleNamespace(
        hybrid_layer_pattern=[0, 1],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        v_head_dim=64,
        swa_num_attention_heads=6,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        swa_v_head_dim=32,
    )
    adapter = MiMoV2StateDictAdapter.__new__(MiMoV2StateDictAdapter)
    adapter.config = config
    adapter.dtype = torch.float32

    if config.hybrid_layer_pattern[layer_idx]:
        num_heads = config.swa_num_attention_heads
        num_kv_heads = config.swa_num_key_value_heads
        head_dim = config.swa_head_dim
        v_head_dim = config.swa_v_head_dim
    else:
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        v_head_dim = config.v_head_dim

    checkpoint_tp = config.num_key_value_heads
    q_rows = (num_heads // checkpoint_tp) * head_dim
    k_rows = (num_kv_heads // checkpoint_tp) * head_dim
    v_rows = (num_kv_heads // checkpoint_tp) * v_head_dim
    rows_per_shard = q_rows + k_rows + v_rows
    scale_rows_per_shard = (rows_per_shard + 127) // 128

    raw_shards = []
    scale_shards = []
    expected_q = []
    expected_k = []
    expected_v = []
    for shard_idx in range(checkpoint_tp):
        raw_shard = torch.ones(rows_per_shard, 128).to(torch.float8_e4m3fn)
        scale_shard = torch.arange(
            1 + shard_idx * scale_rows_per_shard,
            1 + (shard_idx + 1) * scale_rows_per_shard,
            dtype=torch.float32,
        ).unsqueeze(1)
        dequantized_shard = raw_shard.float() * scale_shard.repeat_interleave(128, dim=0)[:rows_per_shard]

        raw_shards.append(raw_shard)
        scale_shards.append(scale_shard)
        expected_q.append(dequantized_shard[:q_rows])
        expected_k.append(dequantized_shard[q_rows : q_rows + k_rows])
        expected_v.append(dequantized_shard[q_rows + k_rows :])

    key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
    scale_key = key + "_scale_inv"
    state_dict = {
        key: torch.cat(raw_shards),
        scale_key: torch.cat(scale_shards),
    }
    expected = torch.cat(expected_q + expected_k + expected_v)

    result = adapter._dequantize(state_dict)

    assert scale_key not in result
    torch.testing.assert_close(result[key], expected, rtol=0, atol=0)

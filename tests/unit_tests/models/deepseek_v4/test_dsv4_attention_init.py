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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Attention


@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("from_meta", [False, True])
def test_attention_init_restores_all_norm_scales(ratio: int, dtype: str, from_meta: bool) -> None:
    """Materialized attention norms start as identity scales, including both compressed pools."""
    config = DeepseekV4Config(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=2,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=4,
        compress_ratios=[ratio],
        torch_dtype=dtype,
    )
    with torch.device("meta" if from_meta else "cpu"):
        attention = DeepseekV4Attention(config, layer_idx=0, backend=BackendConfig())
    if from_meta:
        attention.to_empty(device="cpu")
    # Poison every norm so the test cannot pass because an allocator returned ones.
    norm_weights = {
        name: parameter for name, parameter in attention.named_parameters() if name.endswith("_norm.weight")
    }
    expected_count = {0: 2, 4: 4, 128: 3}[ratio]
    assert len(norm_weights) == expected_count
    with torch.no_grad():
        for parameter in norm_weights.values():
            parameter.fill_(float("nan"))
    attention.init_weights(buffer_device=torch.device("cpu"))
    for name, parameter in norm_weights.items():
        torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0, msg=name)

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
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from nemo_automodel.components.models.llama.state_dict_adapter import LlamaStateDictAdapter
from nemo_automodel.components.models.qwen2.state_dict_adapter import Qwen2StateDictAdapter
from nemo_automodel.components.models.qwen3.state_dict_adapter import Qwen3StateDictAdapter


@pytest.mark.parametrize(
    "adapter",
    [
        LlamaStateDictAdapter(LlamaConfig()),
        Qwen2StateDictAdapter(Qwen2Config()),
        Qwen3StateDictAdapter(Qwen3Config()),
    ],
)
def test_passthrough_adapter_streams_one_hf_tensor(adapter):
    tensor = torch.randn(2, 3)

    converted = adapter.convert_single_tensor_to_hf("model.weight", tensor)

    assert len(converted) == 1
    assert converted[0][0] == "model.weight"
    assert converted[0][1] is tensor
    assert (
        adapter.convert_single_tensor_to_hf(
            "model.weight",
            tensor,
            exclude_key_regex=r"model\..*",
        )
        == []
    )

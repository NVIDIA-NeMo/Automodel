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

"""Gemma4 packing must remain accepted when its backend configuration names TE."""

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing_for_models
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration


@pytest.mark.parametrize("enable_moe", [False, True])
@pytest.mark.parametrize("hf_attention", ["sdpa", "eager"])
def test_neat_packing_accepts_gemma4_with_te_backend_config(enable_moe: bool, hf_attention: str) -> None:
    config = Gemma4Config(
        text_config=Gemma4TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=16,
            enable_moe_block=enable_moe,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=32,
        ),
    )
    config._attn_implementation = hf_attention
    backend = BackendConfig(
        attn="te",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    with torch.device("meta"):
        model = Gemma4ForConditionalGeneration(config, backend=backend)

    contract = configure_packing_for_models([model])

    assert contract.packed_mask_type == "block_causal"

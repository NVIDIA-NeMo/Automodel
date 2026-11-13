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

"""Inkling 2's activation on Transformer Engine grouped experts (needs TE and a CUDA device)."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.layers import Inkling2MoE, inkling2_softcapped_swiglu
from nemo_automodel.components.moe import layers as moe_layers
from nemo_automodel.components.moe.experts import GroupedExpertsTE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TE grouped experts are built on a CUDA device")
def test_expert_parallel_te_experts_get_the_softcapped_activation(monkeypatch):
    """TE grouped experts keep the adapter's stacked keys and take Inkling 2's activation."""
    pytest.importorskip("transformer_engine.pytorch")
    monkeypatch.setattr(moe_layers, "get_world_size_safe", lambda: 2)
    text = {
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "local_layer_ids": [0],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "vocab_size": 64,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 16,
        "eos_token_id": 1,
    }
    config = Inkling2Config(text_config=text, vision_config=None, eos_token_id=1).text_config
    moe = Inkling2MoE(config, BackendConfig(attn="sdpa", experts="te", dispatcher="hybridep"))
    assert isinstance(moe.experts, GroupedExpertsTE)
    assert moe.experts.expert_activation.func is inkling2_softcapped_swiglu
    assert set(moe.experts.state_dict()) == {"gate_and_up_projs", "down_projs"}

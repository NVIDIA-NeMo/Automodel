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

from copy import deepcopy

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v25.config import MiMoV2Config
from nemo_automodel.components.models.mimo_v25.model import MiMoV2ForCausalLM
from nemo_automodel.components.moe.parallelizer import _get_model_moe_config


@pytest.fixture
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # This test compares eager stage math; the distributed smoke covers the
        # compiled MoE path without charging CPU compiler startup to each test.
        with torch.compiler.set_stance("force_eager"):
            yield
    finally:
        torch.set_num_threads(previous)


def test_pipeline_stages_match_unsplit_forward_and_gradients(single_torch_thread):
    config = MiMoV2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        swa_v_head_dim=4,
        hybrid_layer_pattern=[0, 1, 0, 1],
        moe_layer_freq=[0, 1, 1, 1],
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        sliding_window=4,
        partial_rotary_factor=0.5,
        add_swa_attention_sink_bias=True,
        attention_projection_layout="fused_qkv",
        torch_dtype="float32",
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    reference = MiMoV2ForCausalLM.from_config(config, backend=backend)
    sink_name = "model.layers.1.self_attn.attention_sink_bias"
    assert sink_name not in dict(reference.named_parameters())
    assert dict(reference.named_buffers())[sink_name].dtype == torch.float32
    assert sink_name in reference.state_dict()
    # Supply deterministic weights, as the supported checkpoint-loading path does.
    torch.manual_seed(1234)
    with torch.no_grad():
        for name, param in reference.named_parameters():
            if name.endswith("norm.weight"):
                param.fill_(1)
            elif "bias" in name:
                param.zero_()
            else:
                param.normal_(0, 0.02)

    stages = []
    for stage_idx in range(4):
        stage = deepcopy(reference)
        for layer_idx in list(stage.model.layers):
            if layer_idx != str(stage_idx):
                del stage.model.layers[layer_idx]
        if stage_idx != 0:
            stage.model.embed_tokens = None
        if stage_idx != 3:
            stage.model.norm = None
            stage.lm_head = None
        assert _get_model_moe_config(stage) is stage.model.moe_config
        stages.append(stage)

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected = reference(input_ids).logits
    actual = input_ids
    for stage in stages:
        actual = stage(actual).logits
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    expected.square().mean().backward()
    actual.square().mean().backward()
    reference_params = dict(reference.named_parameters())
    for stage in stages:
        for name, param in stage.named_parameters():
            if not param.requires_grad:
                continue
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad, reference_params[name].grad, rtol=1e-5, atol=1e-6)

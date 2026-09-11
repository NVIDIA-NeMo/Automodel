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
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.dspark import DeepseekV41DSparkBackbone


def _config() -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        compress_ratios=[0, 0, 0, 0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        engram_layer_ids=[],
        num_nextn_predict_layers=3,
        dspark_noise_token_id=31,
        dspark_target_layer_ids=[0, 1],
        dspark_markov_rank=4,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dtype="float32",
    )


def _model(attn: str = "eager") -> DeepseekV41DSparkBackbone:
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")
    model = DeepseekV41DSparkBackbone(_config(), backend)
    model.initialize_weights(torch.device("cpu"))
    return model


def test_released_stage_ownership_and_draft_moe_shape() -> None:
    model = _model()
    keys = set(model.state_dict())
    assert "mtp.0.main_proj.weight" in keys
    assert "mtp.0.main_norm.weight" in keys
    assert not any(key.startswith("mtp.1.main_") for key in keys)
    assert "mtp.2.norm.weight" in keys
    assert "mtp.2.markov_head.embed.weight" in keys
    assert "mtp.2.markov_head.head.weight" in keys
    assert "mtp.2.confidence_head.proj.weight" in keys
    assert model.moe_config.n_routed_experts == 4
    assert model.moe_config.n_activated_experts == 2
    assert model.mtp[-1].confidence_head.proj.weight.dtype == torch.float32
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


def test_cache_free_backbone_forward_and_backward() -> None:
    torch.manual_seed(17)
    model = _model()
    context_sequence = 6
    draft_sequence = 5
    noise_embeddings = torch.randn(1, draft_sequence, 16, requires_grad=True)
    target_hidden_states = torch.randn(1, context_sequence, 32)
    position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6]])
    attention_mask = torch.zeros(1, 1, draft_sequence, context_sequence + draft_sequence)

    output = model(
        noise_embeddings,
        target_hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    assert output.hidden_states.shape == (1, draft_sequence, 16)
    assert output.normalized_hidden_states.shape == (1, draft_sequence, 16)
    assert torch.isfinite(output.hidden_states).all()
    assert torch.isfinite(output.normalized_hidden_states).all()
    output.normalized_hidden_states.square().mean().backward()
    assert noise_embeddings.grad is not None and torch.isfinite(noise_embeddings.grad).all()
    assert model.mtp[0].main_proj.weight.grad is not None
    assert model.mtp[0].ffn.experts.gate_and_up_projs.grad is not None


def test_sdpa_matches_eager_attention() -> None:
    torch.manual_seed(23)
    eager = _model("eager").eval()
    sdpa = _model("sdpa").eval()
    sdpa.load_state_dict(eager.state_dict())
    context_sequence = 4
    draft_sequence = 5
    noise_embeddings = torch.randn(1, draft_sequence, 16)
    target_hidden_states = torch.randn(1, context_sequence, 32)
    position_ids = torch.tensor([[0, 1, 2, 3, 2, 3, 4, 5, 6]])
    attention_mask = torch.zeros(1, 1, draft_sequence, context_sequence + draft_sequence)
    attention_mask[..., 0] = -torch.inf

    eager_output = eager(
        noise_embeddings,
        target_hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    sdpa_output = sdpa(
        noise_embeddings,
        target_hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    torch.testing.assert_close(
        sdpa_output.hidden_states,
        eager_output.hidden_states,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        sdpa_output.normalized_hidden_states,
        eager_output.normalized_hidden_states,
        rtol=1e-5,
        atol=1e-6,
    )


def test_markov_and_confidence_heads_follow_released_shapes() -> None:
    model = _model()
    final = model.mtp[-1]
    token_ids = torch.tensor([[1, 2, 3, 4, 5]])
    transition_logits, markov_embeddings = final.markov_head(token_ids)
    confidence = final.confidence_head(torch.randn(1, 5, 16), markov_embeddings)
    assert transition_logits.shape == (1, 5, 32)
    assert markov_embeddings.shape == (1, 5, 4)
    assert confidence.shape == (1, 5)
    assert confidence.dtype == torch.float32


def test_official_positions_and_swa_mask_include_anchor_context() -> None:
    model = _model()
    anchors = torch.tensor([[1, 5]])
    keep = torch.tensor([[True, True]])
    positions = model.build_position_ids(anchors, context_sequence=8)
    assert positions.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 6, 7, 8, 9, 10]]

    mask = model.build_attention_mask(anchors, keep, context_sequence=8, dtype=torch.float32)
    visible = mask[:, 0] == 0
    # Both blocks see their anchor in target context and every slot in their
    # own parallel draft block, but cannot see the other block.
    assert visible[0, 0, 1]
    assert visible[0, 0, 8:13].all()
    assert not visible[0, 0, 13:].any()
    assert visible[0, 5, 5]
    assert visible[0, 5, 13:18].all()
    assert not visible[0, 5, 8:13].any()


def test_draft_schedule_must_cover_all_native_stages() -> None:
    config = _config()
    config.compress_ratios = [0, 0]
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32")
    with pytest.raises(ValueError, match="compress_ratios"):
        DeepseekV41DSparkBackbone(config, backend)

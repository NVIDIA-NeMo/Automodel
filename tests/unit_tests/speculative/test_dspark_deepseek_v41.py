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
from torch import nn

from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.speculative.dspark.config import build_deepseek_v41_draft_config
from nemo_automodel.components.speculative.dspark.draft_deepseek_v41 import DeepseekV41DSparkModel
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss


class _Args(dict):
    def __getattr__(self, name: str):
        return self[name]


def _target_config() -> DeepseekV41Config:
    text = DeepseekV41TextConfig(
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
        dspark_block_size=5,
        dspark_noise_token_id=31,
        dspark_target_layer_ids=[0, 1],
        dspark_markov_rank=4,
        dspark_n_routed_experts=4,
        dspark_num_experts_per_tok=2,
        dtype="float32",
    )
    return DeepseekV41Config(text_config=text, vision_config=DeepseekV41VisionConfig(num_hidden_layers=0))


def _args(**overrides) -> _Args:
    values = {
        "num_draft_layers": 3,
        "target_layer_ids": [0, 1],
        "block_size": 5,
        "num_anchors": 2,
        "mask_token_id": 31,
        "markov_rank": 4,
        "confidence_head_alpha": 1.0,
    }
    values.update(overrides)
    return _Args(values)


def test_builder_preserves_released_native_contract() -> None:
    config = build_deepseek_v41_draft_config(_target_config(), _args())
    assert config.architectures == ["DeepseekV41DSparkModel"]
    assert config.num_nextn_predict_layers == 3
    assert config.dspark_block_size == 5
    assert config.dspark_noise_token_id == 31
    assert config.dspark_target_layer_ids == [0, 1]
    assert config.dspark_num_anchors == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"num_draft_layers": 2}, "num_draft_layers=3"),
        ({"block_size": 4}, "block_size=5"),
        ({"mask_token_id": 30}, "mask_token_id=31"),
        ({"markov_rank": 3}, "markov_rank=4"),
        ({"target_layer_ids": [0]}, "target_layer_ids must match"),
    ],
)
def test_builder_rejects_recipe_checkpoint_mismatch(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_deepseek_v41_draft_config(_target_config(), _args(**override))


def test_training_forward_uses_shared_loss_and_stops_target_gradients() -> None:
    torch.manual_seed(31)
    config = build_deepseek_v41_draft_config(_target_config(), _args())
    model = DeepseekV41DSparkModel(config)
    embedding = nn.Embedding(32, 16)
    head = nn.Linear(16, 32, bias=False)
    model.initialize_embeddings_and_head(embed_tokens=embedding, lm_head=head)

    input_ids = torch.randint(0, 31, (1, 9))
    loss_mask = torch.ones_like(input_ids, dtype=torch.float32)
    target_features = torch.randn(1, 9, 32, requires_grad=True)
    target_last_hidden = torch.randn(1, 9, 16, requires_grad=True)
    output = model(input_ids, target_features, loss_mask, target_last_hidden)
    assert output.draft_logits.shape == (1, 2, 5, 32)
    assert output.target_ids.shape == (1, 2, 5)
    assert output.confidence_pred is not None
    assert output.confidence_pred.shape == (1, 2, 5)
    assert output.aligned_target_logits is not None

    loss = compute_dspark_loss(
        outputs=output,
        loss_decay_gamma=None,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    )
    loss.backward()
    assert model.mtp[0].main_proj.weight.grad is not None
    assert model.mtp[-1].markov_head.head.weight.grad is not None
    assert model.mtp[-1].confidence_head.proj.weight.grad is not None
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None
    assert target_features.grad is None
    assert target_last_hidden.grad is None

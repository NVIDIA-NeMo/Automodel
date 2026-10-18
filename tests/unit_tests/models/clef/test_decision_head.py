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

import torch

from nemo_automodel.components.models.clef.joint_schema_head import ClefHeadConfig, EncodedQuestion, EncodedRecord


def test_clef_head_forward_backward_and_checkpoint_keys():
    config = ClefHeadConfig(
        hidden_size=8,
        width=8,
        routing_layers=1,
        layers=1,
        heads=2,
        feedforward=16,
    )
    head = config.build()
    record = EncodedRecord(
        input_ids=(0, 1, 2, 3, 4, 5),
        questions=(
            EncodedQuestion(
                question_id="disposition",
                question_type=1,
                question_span=(1, 2),
                option_spans=((2, 3), (3, 4), (4, 5)),
                option_ids=("benign", "suspicious", "malicious"),
            ),
        ),
        record_id="incident-1",
    )
    hidden_states = torch.randn(1, 6, 8, requires_grad=True)
    input_ids = torch.arange(6).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    output_embedding_weight = torch.randn(16, 8)

    logits = head(hidden_states, input_ids, attention_mask, [record], output_embedding_weight)
    loss = torch.nn.functional.cross_entropy(logits[0][0].unsqueeze(0), torch.tensor([2]))
    loss.backward()

    assert logits[0][0].shape == (3,)
    assert torch.isfinite(logits[0][0]).all()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert "evidence_layers.0.attention.in_proj_weight" in head.state_dict()
    assert "residual_scorer.3.weight" in head.state_dict()


def test_clef_config_rejects_invalid_attention_width():
    try:
        ClefHeadConfig(
            hidden_size=8,
            width=7,
            routing_layers=1,
            layers=1,
            heads=2,
            feedforward=16,
        )
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("non-divisible attention width should fail")

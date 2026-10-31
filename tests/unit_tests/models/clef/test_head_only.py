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

"""Frozen CLEF backbone features train and reload the external head."""

from types import SimpleNamespace

import torch
from torch import nn

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.clef.joint_schema_head import (
    ClefHeadConfig,
    EncodedQuestion,
    EncodedRecord,
    LexicalTable,
)
from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe


class FrozenBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.requires_grad_(False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        use_cache: bool,
        logits_to_keep: int,
        output_hidden_states: bool,
    ):
        """Return final states through the model output API.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            attention_mask: Long tensor of shape [batch, sequence].
            use_cache: Whether decoding cache is requested.
            logits_to_keep: Requested language-model output slice.
            output_hidden_states: Whether hidden states are requested.

        Returns:
            Object with hidden_states containing a tensor [batch, sequence, hidden].
        """
        assert output_hidden_states
        return SimpleNamespace(hidden_states=(self.embedding(input_ids),))


def recipe(output):
    torch.manual_seed(7)
    result = TrainClefDecisionRecipe.__new__(TrainClefDecisionRecipe)
    result.model = FrozenBackbone()
    result.head = ClefHeadConfig(hidden_size=8, width=8, routing_layers=1, layers=1, heads=2, feedforward=16).build()
    result.lexical = LexicalTable(list(range(16)), result.model.embedding.weight.detach().clone())
    result.pad_token_id = 0
    result.device = torch.device("cpu")
    result.peft_config = None
    result.dist_env = SimpleNamespace(is_main=True)
    result.tag = "sft"
    result.cfg = {}
    result.checkpointer = CheckpointingConfig(checkpoint_dir=output, is_peft=False, save_consolidated=False).build(
        dp_rank=0, tp_rank=0, pp_rank=0
    )
    return result


def test_frozen_backbone_head_forward_update_and_stage_reload(tmp_path):
    run = recipe(tmp_path)
    record = EncodedRecord(
        input_ids=(1, 2, 3, 4),
        record_id="example",
        questions=(EncodedQuestion("pick", 1, (0, 1), ((1, 2), (2, 3)), ("a", "b")),),
    )
    before_backbone = {name: value.clone() for name, value in run.model.state_dict().items()}
    before_head = {name: value.clone() for name, value in run.head.state_dict().items()}
    logits = run._forward([({"id": "example"}, record)])[0][0]
    loss = nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([1]))
    loss.backward()
    assert all(parameter.grad is None for parameter in run.model.parameters())
    optimizer = torch.optim.SGD(run.head.parameters(), lr=0.1)
    optimizer.step()
    assert any(not torch.equal(value, before_head[name]) for name, value in run.head.state_dict().items())
    for name, value in run.model.state_dict().items():
        torch.testing.assert_close(value, before_backbone[name], rtol=0, atol=0)
    stage = tmp_path / "stage"
    try:
        run._save_stage(stage)
        assert not (stage / "model").exists()
        assert (stage / "head" / ".metadata").exists()
        saved = {name: value.clone() for name, value in run.head.state_dict().items()}
        with torch.no_grad():
            for parameter in run.head.parameters():
                parameter.add_(1)
        run._load_stage(stage)
        for name, value in run.head.state_dict().items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
        for name, value in run.model.state_dict().items():
            torch.testing.assert_close(value, before_backbone[name], rtol=0, atol=0)
    finally:
        run.checkpointer.close()

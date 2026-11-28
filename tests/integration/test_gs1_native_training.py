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

import os

import pytest
import torch
from transformers import AutoModel, AutoTokenizer

from nemo_automodel.components.models.gs1.training import (
    GS1DecisionExample,
    GS1DecisionObjective,
    GS1TrainingConfig,
    collate_gs1_examples,
    ensure_gs1_transformers_compat,
)


@pytest.mark.skipif(
    not os.getenv("GS1_TEST_MODEL"),
    reason="set GS1_TEST_MODEL to run the native checkpoint smoke test",
)
def test_native_checkpoint_compiles_and_runs_forward_backward() -> None:
    ensure_gs1_transformers_compat()
    model_id = os.environ["GS1_TEST_MODEL"]
    revision = os.getenv("GS1_TEST_REVISION")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, revision=revision, code_revision=revision, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=revision,
        code_revision=revision,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    compiler, request_type = GS1TrainingConfig(model_id, revision, revision, max_context=4096).build(model, tokenizer)
    request = request_type.model_validate(
        {
            "model": model_id,
            "state": "A refund was requested for order 123.",
            "questions": {
                "refund": {
                    "type": "choice",
                    "instructions": "Should this request be refunded?",
                    "criteria": {"yes": "Approve refund", "no": "Reject refund"},
                },
                "priority": {
                    "type": "choice",
                    "instructions": "What is the request priority?",
                    "criteria": {"low": "Low", "medium": "Medium", "high": "High"},
                },
            },
        }
    )
    compiled = compiler.compile(request)
    targets = {"refund": (0.7, 0.3), "priority": (0.2, 0.5, 0.3)}
    examples = [
        GS1DecisionExample(
            "smoke",
            branch.question_id,
            (*compiled.state, *branch.suffix),
            branch,
            len(compiled.state),
            targets[branch.question_id],
        )
        for branch in compiled.branches
    ]
    device = next(model.parameters()).device
    batch = collate_gs1_examples(examples, pad_token_id=tokenizer.pad_token_id or 0, device=device)

    def forward(current_batch):
        return model(
            input_ids=current_batch.input_ids,
            attention_mask=current_batch.attention_mask,
            position_ids=current_batch.position_ids,
            branches=list(current_batch.branches),
            state_lengths=list(current_batch.state_lengths),
        )

    model.eval()
    with torch.no_grad():
        output = forward(batch)
        for row, example in enumerate(examples):
            single = collate_gs1_examples([example], pad_token_id=tokenizer.pad_token_id or 0, device=device)
            single_output = forward(single)
            width = len(example.target_probs)
            torch.testing.assert_close(output.logits[row, :width], single_output.logits[0, :width])
    model.train()
    output = forward(batch)
    assert output.logits is not None and torch.isfinite(output.logits[batch.option_mask]).all()
    loss, _ = GS1DecisionObjective()(output.logits, batch.target_probs, batch.option_mask)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

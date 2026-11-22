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

"""Offline CLEF parity against independently generated pinned publisher outputs."""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.clef.joint_schema_head import (
    ClefHeadConfig,
    EncodedQuestion,
    EncodedRecord,
    encode_record,
)


class _CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens):
        return SimpleNamespace(input_ids=[ord(character) for character in text])


@pytest.fixture(scope="module")
def reference():
    return json.loads(Path(__file__).with_name("fixtures").joinpath("reference.json").read_text())


def test_publisher_prompt_tokens_spans_types_and_option_order(reference):
    for case in reference["encodings"]:
        encoded = encode_record(
            _CharacterTokenizer(), reference["prompt_record"], max_state_tokens=case["max_state_tokens"]
        )
        # JSON normalization only converts the native contract's immutable tuples to arrays.
        assert json.loads(json.dumps(asdict(encoded))) == case["expected"]


def test_publisher_head_logits_and_input_parameter_gradients(reference):
    head = ClefHeadConfig(**reference["head_config"]).build().eval()
    head.load_state_dict({name: torch.tensor(value) for name, value in reference["state_dict"].items()}, strict=True)
    hidden = torch.tensor(reference["hidden_states"], requires_grad=True)
    lexical = torch.tensor(reference["lexical_rows"], requires_grad=True)
    records = [
        EncodedRecord(
            tuple(record["input_ids"]),
            tuple(
                EncodedQuestion(
                    question["question_id"],
                    question["question_type"],
                    tuple(question["question_span"]),
                    tuple(tuple(span) for span in question["option_spans"]),
                    tuple(question["option_ids"]),
                )
                for question in record["questions"]
            ),
            record["record_id"],
        )
        for record in reference["records"]
    ]
    logits = head(
        hidden, torch.tensor(reference["input_ids"]), torch.tensor(reference["attention_mask"]), records, lexical
    )
    for actual_record, expected_record in zip(logits, reference["logits"], strict=True):
        for actual, expected in zip(actual_record, expected_record, strict=True):
            torch.testing.assert_close(actual, torch.tensor(expected), atol=2e-6, rtol=2e-5)
    flat = [row for record in logits for row in record]
    loss = torch.stack(
        [
            torch.nn.functional.cross_entropy(row.unsqueeze(0), torch.tensor([target]))
            for row, target in zip(flat, reference["targets"], strict=True)
        ]
    ).mean()
    torch.testing.assert_close(loss, torch.tensor(reference["loss"]), atol=2e-6, rtol=2e-5)
    loss.backward()
    # fp32 attention/LayerNorm backward may differ slightly in reduction order.
    torch.testing.assert_close(hidden.grad, torch.tensor(reference["hidden_gradient"]), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(lexical.grad, torch.tensor(reference["lexical_gradient"]), atol=3e-6, rtol=3e-5)
    for name, parameter in head.named_parameters():
        torch.testing.assert_close(
            parameter.grad, torch.tensor(reference["parameter_gradients"][name]), atol=3e-6, rtol=3e-5, msg=name
        )

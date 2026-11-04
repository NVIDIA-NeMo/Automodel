# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Regression checks for D1's opposite noul orders and text calibration."""

from types import ModuleType, SimpleNamespace

import pytest

from nemo_automodel.components.datasets.decision.records import question_targets, validate_record
from nemo_automodel.components.models.d1.records import encode_d1_record


def inputs():
    record = {
        "id": "soc-1",
        "state": {"observation": "Endpoint needs analyst review"},
        "questions": {"review": {"type": "noul", "instructions": "Needs analyst review?"}},
        "answers": {"review": "true"},
        "rewards": {"review": {"true": 0.75, "false": -0.5}},
    }
    prompt = ModuleType("published_prompt_fixture")
    prompt.as_question = lambda schema: SimpleNamespace(**schema)
    prompt.QTYPES = {"choice": 0, "score": 1, "noul": 2}
    prompt.encode = lambda *args: ([1, 7, 8, 9], [2, 3])
    prompt.render = lambda *args, **kwargs: "a complete causal prompt"
    prompt.readout_ids = lambda *args: [[42, 43], [50]]
    prompt.temperature_key = lambda question: "noul:2"
    tokenizer = SimpleNamespace(bos_token="<bos>", encode=lambda *args, **kwargs: [1, 2, 3, 4])
    return record, prompt, tokenizer


def test_omni_marker_order_preserves_gold_reward_and_temperature():
    record, prompt, tokenizer = inputs()
    validate_record(record)
    (example,) = encode_d1_record(
        record,
        tokenizer=tokenizer,
        prompt=prompt,
        omni=True,
        max_length=64,
        temperatures={"noul": 1.1, "noul:2": 1.666},
    )
    (target,) = question_targets(record, {example.question_id: example.option_ids})
    assert example.option_ids == ("false", "true")
    assert target.option_ids[target.label] == record["answers"]["review"]
    assert target.rewards == (-0.5, 0.75)
    assert target.question_type == 0  # Existing loss contract: noul=0.
    assert example.head_question_type == 2  # Published Omni head: noul=2.
    assert example.marker_positions == (2, 3)
    assert example.alias_token_ids == ()
    assert example.temperature == 1.666


def test_causal_alias_order_preserves_gold_and_rewards():
    record, prompt, tokenizer = inputs()
    validate_record(record)
    (example,) = encode_d1_record(record, tokenizer=tokenizer, prompt=prompt, omni=False, max_length=64)
    (target,) = question_targets(record, {example.question_id: example.option_ids})
    assert example.option_ids == ("true", "false")
    assert target.label == 0
    assert target.rewards == (0.75, -0.5)
    assert example.alias_token_ids == ((42, 43), (50,))
    assert example.marker_positions == ()
    assert example.temperature == 1.0
    with pytest.raises(ValueError, match="exceeds max_length"):
        encode_d1_record(record, tokenizer=tokenizer, prompt=prompt, omni=False, max_length=3)


def test_omni_rejects_invalid_calibration_and_missing_markers():
    record, prompt, tokenizer = inputs()
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="positive"):
            encode_d1_record(
                record, tokenizer=tokenizer, prompt=prompt, omni=True, max_length=64, temperatures={"noul:2": invalid}
            )
    prompt.encode = lambda *args: ([1, 2, 3], [2])
    with pytest.raises(ValueError, match="option counts"):
        encode_d1_record(record, tokenizer=tokenizer, prompt=prompt, omni=True, max_length=64)


@pytest.mark.parametrize("omni", [False, True])
def test_encoding_is_independent_of_supervision(omni):
    record, prompt, tokenizer = inputs()
    validate_record(record)
    labelled = encode_d1_record(record, tokenizer=tokenizer, prompt=prompt, omni=omni, max_length=64)
    del record["answers"]
    del record["rewards"]
    unlabelled = encode_d1_record(record, tokenizer=tokenizer, prompt=prompt, omni=omni, max_length=64)
    assert unlabelled == labelled

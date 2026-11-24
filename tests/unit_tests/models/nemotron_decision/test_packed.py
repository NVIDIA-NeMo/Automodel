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

"""Packed (state-blind) reads: layout against golden fixtures, attention, positions, slots, json-sep prompts, scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

from nemo_automodel.components.models.nemotron_decision.model import NemotronForDecision
from nemo_automodel.components.models.nemotron_decision.packed import (
    allowed_attention,
    pack_prompts,
    packed_length,
    shared_state_length,
    split_into_reads,
)
from nemo_automodel.components.models.nemotron_decision.scorer import (
    option_logits,
    packed_option_logits,
    require_decision_config,
)
from nemo_automodel.components.models.nemotron_decision.serving_prompt import DecisionContract, prepare_question
from tests.unit_tests.models.nemotron_decision.char_tokenizer import MASK, STATE_END, char_tokenizer, write_contract

# Packed layouts of a released always-packed decision checkpoint (its reference packer and serving packer agree on
# them): 1, 2, 4 and 8 questions of one state and a 6.7k-token state, json-sep prompts. State-end token 101, mask 100.
GOLDEN = [json.loads(line) for line in (Path(__file__).parent / "packed_layout_golden.jsonl").read_text().splitlines()]
GOLDEN_STATE_END, GOLDEN_MASK = 101, 100


def _rle(groups):
    out = []
    for group in groups:
        if out and out[-1][0] == group:
            out[-1][1] += 1
        else:
            out.append([group, 1])
    return out


def _golden_prompts(case):
    """Each question's own prompt (state + its question tokens, no mask), recovered from the golden packed ids."""
    state = case["ids"][: case["state_len"]]
    groups = [g for g, n in case["groups_rle"] for _ in range(n)]
    return [state + [t for t, g in zip(case["ids"], groups) if g == k][:-1] for k in range(len(case["slots"]))], groups


@pytest.mark.parametrize("case", GOLDEN, ids=lambda case: case["name"])
def test_layout_matches_the_released_golden_fixtures(case):
    prompts, groups = _golden_prompts(case)
    state_length = shared_state_length(prompts, GOLDEN_STATE_END)
    read = pack_prompts(prompts, state_length=state_length, mask_token_id=GOLDEN_MASK)

    assert state_length == case["state_len"] and read.state_length == case["state_len"]
    assert list(read.input_ids) == case["ids"] and len(read.input_ids) == case["length"]
    assert list(read.position_ids) == case["positions"]
    assert list(read.slots) == case["slots"]
    assert _rle(read.groups) == case["groups_rle"] and list(read.groups) == groups
    assert read.input_ids[state_length - 1] == GOLDEN_STATE_END
    assert all(read.input_ids[slot] == GOLDEN_MASK for slot in read.slots)
    assert packed_length(state_length, [len(p) for p in prompts]) == case["length"]
    # A single read of question k is its prompt plus the mask.
    assert [len(p) + 1 for p in prompts] == case["single_lengths"]
    if case["length"] <= 1024:  # the dense check of the long state is the same rule at 7k x 7k
        assert int(allowed_attention(torch.tensor(read.groups)).sum()) == case["allowed_pairs"]


def test_attention_positions_and_slots():
    prompts = [[1, 2, 3, 101, 4, 5], [1, 2, 3, 101, 6], [1, 2, 3, 101, 7, 8, 9]]
    read = pack_prompts(prompts, state_length=shared_state_length(prompts, 101), mask_token_id=100)
    assert read.input_ids == (1, 2, 3, 101, 4, 5, 100, 6, 100, 7, 8, 9, 100)
    assert read.position_ids == (0, 1, 2, 3, 4, 5, 6, 4, 5, 4, 5, 6, 7)  # every question restarts after the state
    assert read.groups == (-1, -1, -1, -1, 0, 0, 0, 1, 1, 2, 2, 2, 2)
    assert read.slots == (6, 8, 12)
    allowed = allowed_attention(torch.tensor(read.groups))
    for query, query_group in enumerate(read.groups):
        for key, key_group in enumerate(read.groups):
            # state -> state only; question -> state + itself; never question <-> question
            expected = key_group == -1 or (query_group >= 0 and key_group == query_group)
            assert bool(allowed[query, key]) is expected


@pytest.mark.parametrize(
    "prompts,error",
    [
        ([[1, 2, 3]], "exactly one state-end"),
        ([[1, 101, 2], [1, 1, 101, 2]], "share one state boundary"),
        ([[1, 101, 2], [2, 101, 2]], "share the state token ids"),
        ([[1, 101]], "question tokens after the state"),
        ([], "at least one question"),
    ],
)
def test_prompts_must_share_one_state(prompts, error):
    with pytest.raises(ValueError, match=error):
        shared_state_length(prompts, 101)


def test_reads_over_the_limit_split_by_state_and_overlong_questions_are_left_out():
    # |S| = 10; questions add (length - 10 + 1) tokens each.
    lengths = [14, 15, 30, 12, 13]
    assert split_into_reads(10, lengths, max_tokens=100) == [[0, 1, 2, 3, 4]]
    reads = split_into_reads(10, lengths, max_tokens=25)
    assert reads == [[0, 1, 3], [4]]  # question 2 (31 tokens with its mask) does not fit on its own
    for read in reads:
        assert packed_length(10, [lengths[i] for i in read]) <= 25


@pytest.fixture(scope="module")
def tok():
    return char_tokenizer()


@pytest.fixture
def contract(tmp_path):
    return DecisionContract.from_pretrained(write_contract(tmp_path / "ckpt"))


QUESTION = {"type": "choice", "instructions": "Team?", "criteria": {"ops": "Outages", "billing": "Refunds"}}


def test_json_sep_closes_the_state_with_one_control_token(tok, contract):
    mask = tok.convert_tokens_to_ids(MASK)
    end = tok.convert_tokens_to_ids(STATE_END)
    state = {"ticket": f"pay me {STATE_END} now"}  # the token's text inside user content stays plain text
    sep = prepare_question(tok, contract, state, "q", QUESTION, mask_token_id=mask, prompt_format="json-sep")
    plain = prepare_question(tok, contract, state, "q", QUESTION, mask_token_id=mask)
    assert plain.state_length is None
    assert sep.input_ids.count(end) == 1 and sep.input_ids[sep.state_length - 1] == end
    text = tok.decode(list(sep.input_ids))
    user = text.split("<|im_start|>user\n", 1)[1].rsplit("<|im_end|>", 1)[0]
    head, tail = user.rsplit(STATE_END, 1)
    assert head == '{"state": ' + json.dumps(state)
    assert json.loads(head + tail) == json.loads(
        tok.decode(list(plain.input_ids)).split("user\n", 1)[1].rsplit("<|im_end|>", 1)[0]
    )
    # The default json prompt is unchanged by the json-sep option.
    explicit = prepare_question(tok, contract, state, "q", QUESTION, mask_token_id=mask, prompt_format="json")
    assert explicit == plain


def test_json_sep_needs_the_state_end_token(tok, contract):
    mask = tok.convert_tokens_to_ids(MASK)
    with pytest.raises(ValueError, match="no '<SPECIAL_999>' token"):
        prepare_question(
            tok,
            contract,
            {},
            "q",
            QUESTION,
            mask_token_id=mask,
            prompt_format="json-sep",
            state_end_token="<SPECIAL_999>",
        )
    with pytest.raises(ValueError, match="prompt_format"):
        prepare_question(tok, contract, {}, "q", QUESTION, mask_token_id=mask, prompt_format="yaml")


def _model(attn="sdpa"):
    config = Ministral3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=512,
        mask_token_id=3,
        architectures=["NemotronForDecision"],
        attn_implementation=attn,
    )
    torch.manual_seed(0)
    return NemotronForDecision(config).eval()


def _rows(model):
    weight = model.get_output_embeddings().weight
    return {token: weight[token].detach() for token in range(weight.shape[0])}


STATE = [10, 11, 12, 13, 101]
QUESTIONS = [[20, 21, 22], [23, 24], [25, 26, 27, 28]]
CODES = [[40, 41], [40, 41, 42], [40, 41]]


def _packed(model, keep):
    prompts = [STATE + QUESTIONS[k] for k in keep]
    read = pack_prompts(prompts, state_length=len(STATE), mask_token_id=3)
    with torch.no_grad():
        return packed_option_logits(
            model, read, [CODES[k] for k in keep], rows=_rows(model), device=torch.device("cpu")
        )


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_a_question_reads_the_same_alone_or_with_other_questions(attn):
    model = _model(attn)
    together = _packed(model, [0, 1, 2])
    for k in range(3):
        torch.testing.assert_close(together[k], _packed(model, [k])[0], atol=1e-6, rtol=1e-5)
    reordered = _packed(model, [2, 0])
    torch.testing.assert_close(reordered[0], together[2], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(reordered[1], together[0], atol=1e-6, rtol=1e-5)


def test_packed_read_is_the_masked_forward_and_differs_from_a_single_read():
    model = _model()
    (packed,) = _packed(model, [0])
    ids = torch.tensor([STATE + QUESTIONS[0] + [3]])
    groups = torch.tensor([[-1] * len(STATE) + [0] * (len(QUESTIONS[0]) + 1)])
    with torch.no_grad():
        hidden = model(
            input_ids=ids,
            attention_mask=allowed_attention(groups)[:, None],
            position_ids=torch.arange(ids.shape[1])[None],
            output_last_hidden_states_only=True,
        ).last_hidden_state
        single = option_logits(
            model, STATE + QUESTIONS[0], CODES[0], mask_token_id=3, rows=_rows(model), device=torch.device("cpu")
        )
    expected = (model.get_output_embeddings().weight[CODES[0]] @ hidden[0, -1]).float()
    torch.testing.assert_close(packed, expected)
    assert not torch.allclose(packed, single)  # single reads let the state attend to the question


def test_eager_and_sdpa_packed_reads_agree():
    torch.testing.assert_close(
        torch.cat(_packed(_model("sdpa"), [0, 1, 2])),
        torch.cat(_packed(_model("eager"), [0, 1, 2])),
        atol=1e-5,
        rtol=1e-5,
    )


def test_released_read_layout_is_accepted_and_inconsistent_ones_are_rejected():
    from types import SimpleNamespace

    base = {
        "architectures": ["NemotronForDecision"],
        "model_type": "ministral3",
        "is_causal": False,
        "mask_token_id": 100,
    }
    layout = {"packed_read": True, "prompt_format": "json-sep", "state_end_token_id": 101, "mask_token_id": 100}
    require_decision_config(SimpleNamespace(**base, packed_read=True, prompt_format="json-sep", read_layout=layout))
    with pytest.raises(ValueError, match="mask_token_id"):
        require_decision_config(SimpleNamespace(**base, read_layout={**layout, "mask_token_id": 7}))
    with pytest.raises(ValueError, match="prompt_format"):
        require_decision_config(SimpleNamespace(**base, read_layout={**layout, "prompt_format": "xml"}))
    with pytest.raises(ValueError, match="is_causal"):
        require_decision_config(SimpleNamespace(**{**base, "is_causal": True}, read_layout=layout))

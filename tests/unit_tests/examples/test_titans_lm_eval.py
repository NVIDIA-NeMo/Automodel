# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).parents[3] / "examples" / "llm_pretrain" / "titans_lm_eval.py"
SPEC = importlib.util.spec_from_file_location("titans_lm_eval", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
titans_lm_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(titans_lm_eval)


class BoundaryTokenizer:
    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens
        assert return_offsets_mapping
        assert text == "A bright moon"
        # The final token includes the boundary space and the final word.
        return {
            "input_ids": [0, 10, 11, 12],
            "offset_mapping": [(0, 0), (0, 1), (1, 8), (8, 13)],
        }

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens
        return {12: " moon", 13: " sun"}.get(token_ids[-1], "")


def test_causal_nll_and_token_weighted_metrics():
    input_ids = torch.tensor([[0, 1, 0]])
    logits = torch.tensor(
        [
            [
                [[0.0, math.log(3.0)]],
                [[math.log(7.0), 0.0]],
                [[0.0, 0.0]],
            ]
        ]
    ).squeeze(2)

    nll_sum, token_count = titans_lm_eval.causal_nll(logits, input_ids)
    assert nll_sum == pytest.approx(-math.log(3 / 4) - math.log(7 / 8))
    assert token_count == 2

    metrics = titans_lm_eval.metric_totals(
        [
            {"nll_sum": nll_sum, "token_count": 2, "byte_count": 4},
            {"nll_sum": math.log(2), "token_count": 1, "byte_count": 1},
        ]
    )
    expected_loss = (nll_sum + math.log(2)) / 3
    assert metrics["loss"] == pytest.approx(expected_loss)
    assert metrics["perplexity"] == pytest.approx(math.exp(expected_loss))
    assert metrics["bits_per_byte"] == pytest.approx((nll_sum + math.log(2)) / (5 * math.log(2)))


def test_lambada_boundary_and_last_word_exact_are_transparent():
    tokenizer = BoundaryTokenizer()
    boundary = titans_lm_eval.lambada_token_boundary(tokenizer, "A bright moon")

    assert boundary["prompt_ids"] == [0, 10, 11]
    assert boundary["target_ids"] == [12]
    assert boundary["boundary_offset"] == (8, 13)
    assert boundary["answer"] == "moon"
    assert titans_lm_eval.lambada_exact_result(tokenizer, [12], [12], "moon") == {
        "last_word_exact": True,
        "last_word_token_exact": True,
        "generated_continuation": " moon",
        "expected_last_word": "moon",
    }
    assert titans_lm_eval.lambada_exact_result(tokenizer, [13], [12], "moon")["last_word_exact"] is False


def test_local_jsonl_fixture_does_not_load_remote_dataset(tmp_path):
    fixture = tmp_path / "tiny.jsonl"
    fixture.write_text('{"text": "zero"}\n{"text": "one"}\n{"text": "two"}\n')

    rows = list(
        titans_lm_eval.iter_source(
            "fineweb_edu",
            fixture,
            start=1,
            end=3,
            text_field="text",
        )
    )

    assert rows == [(1, "one"), (2, "two")]


def _write_partial(path, metadata, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps({"metadata": metadata, "metadata_hash": "test", **row}) + "\n")


def test_partial_names_resume_ranges_and_deterministic_merge(tmp_path):
    metadata = {
        "schema_version": 1,
        "benchmark": "wikitext103",
        "checkpoint": "/checkpoint",
        "tokenizer": "tokenizer",
        "dataset": {"revision": "fixed"},
        "range": {"start": 0, "end": 2},
    }
    first = titans_lm_eval.partial_path(tmp_path, "wikitext103", 0, 2)
    second = titans_lm_eval.partial_path(tmp_path, "wikitext103", 2, 4)
    assert first != second
    _write_partial(
        first,
        metadata,
        [
            {"index": 1, "nll_sum": 2.0, "token_count": 2, "byte_count": 2},
            {"index": 0, "nll_sum": 1.0, "token_count": 1, "byte_count": 1},
        ],
    )
    metadata = {**metadata, "range": {"start": 2, "end": 4}}
    _write_partial(second, metadata, [{"index": 2, "nll_sum": 3.0, "token_count": 3, "byte_count": 3}])

    summary_path, manifest_path = titans_lm_eval.merge_partials(tmp_path, "wikitext103")
    first_summary = summary_path.read_bytes()
    first_manifest = manifest_path.read_bytes()
    summary = json.loads(first_summary)
    assert summary["source_indices"] == [0, 1, 2]
    assert summary["metrics"]["loss"] == 1.0
    assert summary["metrics"]["token_count"] == 6

    titans_lm_eval.merge_partials(tmp_path, "wikitext103")
    assert summary_path.read_bytes() == first_summary
    assert manifest_path.read_bytes() == first_manifest


def test_merge_rejects_overlapping_ranges(tmp_path):
    metadata = {"benchmark": "fineweb_edu", "range": {"start": 0, "end": 2}}
    row = {"index": 1, "nll_sum": 1.0, "token_count": 1, "byte_count": 1}
    _write_partial(titans_lm_eval.partial_path(tmp_path, "fineweb_edu", 0, 2), metadata, [row])
    metadata = {**metadata, "range": {"start": 1, "end": 3}}
    _write_partial(titans_lm_eval.partial_path(tmp_path, "fineweb_edu", 1, 3), metadata, [row])

    with pytest.raises(ValueError, match="Duplicate source index 1"):
        titans_lm_eval.merge_partials(tmp_path, "fineweb_edu")

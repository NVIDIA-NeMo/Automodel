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

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from pydantic import BaseModel, JsonValue

from nemo_automodel.components.models.gs1.training import (
    GS1CompiledRequest,
    GS1DecisionExample,
    GS1DecisionObjective,
    GS1InputRecord,
    GS1TrainingConfig,
    _contains_media,
    collate_gs1_examples,
    ensure_gs1_transformers_compat,
    load_gs1_examples,
)

pytestmark = pytest.mark.cpu_only


@dataclass(frozen=True)
class _Branch:
    suffix: tuple[int, ...]
    token_roles: tuple[str, ...]
    candidates: tuple[str, ...]
    decision_position: int = 0
    option_spans: tuple[tuple[int, int], ...] = ()
    option_starts: tuple[int, ...] = ()
    question_end_position: int = 0
    question_start_position: int = 0
    question_span: tuple[int, int] = (0, 0)
    state_evidence_start: int = 0


@dataclass(frozen=True)
class _Compiled:
    state: tuple[int, ...]
    branches: tuple[_Branch, ...]


class _Request(BaseModel):
    model: str
    state: JsonValue
    questions: dict[str, dict[str, JsonValue]]


class _Compiler:
    def compile(self, request: BaseModel) -> GS1CompiledRequest:
        value = _Request.model_validate(request.model_dump())
        question = next(iter(value.questions.values()))
        criteria = question["criteria"]
        candidates = tuple(criteria) if isinstance(criteria, dict) else ("false", "true")
        branch = _Branch((8, 9), ("question_context", "decision_head_query"), candidates)
        return _Compiled((1, 2, 3), (branch,))


def _record(state: JsonValue = "state") -> GS1InputRecord:
    return {
        "request": {
            "state": state,
            "questions": {
                "decision": {
                    "type": "choice",
                    "instructions": "Choose one.",
                    "criteria": {"a": "Option A", "b": "Option B"},
                }
            },
        },
        "targets": {"decision": {"candidate_ids": ["b", "a"], "distribution": [0.25, 0.75]}},
    }


def test_native_compiler_orders_soft_targets_by_candidate_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_record()) + "\n")
    examples = load_gs1_examples(
        source,
        compiler=_Compiler(),
        request_type=_Request,
        model_id="bigai/GS1-0.8B",
    )
    assert len(examples) == 1
    assert examples[0].branch.candidates == ("a", "b")
    assert examples[0].target_probs == (0.75, 0.25)
    assert examples[0].input_ids == (1, 2, 3, 8, 9)


def test_loader_rejects_media_states(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_record([{"type": "image", "path": "x.png"}])) + "\n")
    with pytest.raises(ValueError, match="No eligible text-only"):
        load_gs1_examples(
            source,
            compiler=_Compiler(),
            request_type=_Request,
            model_id="bigai/GS1-0.8B",
        )


def test_media_filter_only_matches_top_level_state_blocks() -> None:
    assert _contains_media([{"type": "image", "path": "x.png"}])
    assert _contains_media([{"type": "video", "path": "x.mp4"}])
    assert not _contains_media({"metadata": {"type": "image"}})
    assert not _contains_media([{"type": ["task", "detail"]}])


@pytest.mark.parametrize(
    "state",
    [{"attachment_metadata": {"type": "image"}}, [{"type": ["billing", "refund"]}]],
)
def test_loader_keeps_structured_text_states(tmp_path: Path, state: JsonValue) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(json.dumps(_record(state)) + "\n")
    examples = load_gs1_examples(
        source,
        compiler=_Compiler(),
        request_type=_Request,
        model_id="bigai/GS1-0.8B",
    )
    assert len(examples) == 1


def test_training_config_builds_native_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformers import dynamic_module_utils

    calls = {}
    compiler = _Compiler()

    class Model:
        def decision_compiler(self, tokenizer, *, max_context):
            calls["context"] = max_context
            return compiler

    def load_type(class_name, model_id, *, revision, code_revision):
        calls.update(
            class_name=class_name,
            model_id=model_id,
            revision=revision,
            code_revision=code_revision,
        )
        return _Request

    monkeypatch.setattr(dynamic_module_utils, "get_class_from_dynamic_module", load_type)
    result = GS1TrainingConfig("bigai/GS1-0.8B", "rev", "code-rev", 4096).build(Model(), object())
    assert result == (compiler, _Request)
    assert calls == {
        "context": 4096,
        "class_name": "schema.DecisionRequest",
        "model_id": "bigai/GS1-0.8B",
        "revision": "rev",
        "code_revision": "code-rev",
    }


def test_transformers_cache_compatibility() -> None:
    ensure_gs1_transformers_compat()
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache

    assert Qwen3_5DynamicCache(Qwen3_5Config()).layers


def test_collator_left_pads_and_keeps_branch_rows_aligned() -> None:
    branch_a = _Branch((8, 9), ("question_context", "decision_head_query"), ("a", "b"))
    branch_b = _Branch((7,), ("decision_head_query",), ("x", "y", "z"))
    examples = [
        GS1DecisionExample("r1", "q1", (1, 2, 3, 8, 9), branch_a, 3, (0.75, 0.25)),
        GS1DecisionExample("r2", "q2", (4, 7), branch_b, 1, (0.2, 0.3, 0.5)),
    ]
    batch = collate_gs1_examples(examples, pad_token_id=0, device=torch.device("cpu"))
    assert batch.input_ids.tolist() == [[1, 2, 3, 8, 9], [0, 0, 0, 4, 7]]
    assert batch.attention_mask.tolist() == [
        [True] * 5,
        [False, False, False, True, True],
    ]
    assert batch.position_ids.tolist() == [[0, 1, 2, 3, 4], [0, 0, 0, 0, 1]]
    assert batch.branches == (branch_a, branch_b)
    assert batch.option_mask.tolist() == [[True, True, False], [True, True, True]]


@pytest.mark.parametrize("weight", [0.0, 0.4, 1.0])
def test_objective_matches_distribution_nll_expected_brier_and_gradients(
    weight: float,
) -> None:
    logits = torch.tensor([[1.0, -0.5, 20.0]], requires_grad=True)
    target = torch.tensor([[0.25, 0.75, 0.0]])
    mask = torch.tensor([[True, True, False]])
    loss, metrics = GS1DecisionObjective(weight)(logits, target, mask)
    probabilities = torch.softmax(logits.detach()[0, :2].float(), dim=-1)
    expected_nll = F.cross_entropy(logits.detach()[0, :2].float()[None], target[0, :2][None])
    expected_brier = probabilities.square().sum() + 1 - 2 * (probabilities * target[0, :2]).sum()
    torch.testing.assert_close(loss, expected_nll + weight * expected_brier)
    torch.testing.assert_close(metrics["nll"], expected_nll)
    torch.testing.assert_close(metrics["expected_brier"], expected_brier)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 2] == 0
    reference_logits = logits.detach()[0, :2].clone().requires_grad_()
    reference_probabilities = torch.softmax(reference_logits.float(), dim=-1)
    reference_nll = F.cross_entropy(reference_logits.float()[None], target[0, :2][None])
    reference_brier = reference_probabilities.square().sum() + 1 - 2 * (reference_probabilities * target[0, :2]).sum()
    (reference_nll + weight * reference_brier).backward()
    torch.testing.assert_close(logits.grad[0, :2], reference_logits.grad)


def test_objective_rejects_invalid_weight_and_target_distribution() -> None:
    with pytest.raises(ValueError, match="brier_weight"):
        GS1DecisionObjective(float("nan"))
    with pytest.raises(ValueError, match="sum to one"):
        GS1DecisionObjective()(
            torch.zeros((1, 2)),
            torch.tensor([[0.2, 0.2]]),
            torch.ones((1, 2), dtype=torch.bool),
        )

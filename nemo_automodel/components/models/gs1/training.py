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

"""Text-record compilation and the native GS1 supervised objective."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, cast

import torch
import torch.nn.functional as F
from pydantic import BaseModel, JsonValue
from transformers import PreTrainedTokenizerBase
from transformers.configuration_utils import PreTrainedConfig


class GS1Target(TypedDict):
    """Publisher target indexed by stable candidate identity."""

    candidate_ids: list[str]
    distribution: list[float]


class GS1InputRecord(TypedDict):
    """Minimum native record fields used by the training adapter."""

    request: dict[str, JsonValue]
    targets: dict[str, GS1Target]


class GS1Branch(Protocol):
    """Compiler metadata consumed by ``GS1Model.forward`` and its readout."""

    suffix: tuple[int, ...]
    token_roles: tuple[str, ...]
    candidates: tuple[str, ...]
    decision_position: int
    option_spans: tuple[tuple[int, int], ...]
    option_starts: tuple[int, ...]
    question_end_position: int
    question_start_position: int
    question_span: tuple[int, int]
    state_evidence_start: int


class GS1CompiledRequest(Protocol):
    """Compiled shared state and native decision branches."""

    state: tuple[int, ...]
    branches: tuple[GS1Branch, ...]


class GS1Compiler(Protocol):
    """Public compiler interface exposed by GS1 checkpoints."""

    def compile(self, request: BaseModel) -> GS1CompiledRequest: ...


class GS1DecisionModel(Protocol):
    """Native checkpoint methods needed to build the training adapter."""

    def decision_compiler(self, tokenizer: PreTrainedTokenizerBase, *, max_context: int) -> GS1Compiler: ...


@dataclass(frozen=True)
class GS1TrainingConfig:
    """Build the compiler and request schema from a native GS1 checkpoint."""

    model_id: str
    revision: str | None = None
    code_revision: str | None = None
    max_context: int = 16384

    def build(self, model: GS1DecisionModel, tokenizer: PreTrainedTokenizerBase) -> tuple[GS1Compiler, type[BaseModel]]:
        """Create native GS1 components using the checkpoint's pinned code."""
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        compiler = model.decision_compiler(tokenizer, max_context=self.max_context)
        request_type = cast(
            type[BaseModel],
            get_class_from_dynamic_module(
                "schema.DecisionRequest",
                self.model_id,
                revision=self.revision,
                code_revision=self.code_revision,
            ),
        )
        return compiler, request_type


def ensure_gs1_transformers_compat() -> None:
    """Keep the pinned GS1 code compatible with Transformers' cache API."""
    from transformers.cache_utils import DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5

    if not hasattr(modeling_qwen3_5, "Qwen3_5DynamicCache"):

        class Qwen3_5DynamicCache(DynamicCache):
            def __init__(self, config: PreTrainedConfig) -> None:
                super().__init__(config=config)

        modeling_qwen3_5.Qwen3_5DynamicCache = Qwen3_5DynamicCache


@dataclass(frozen=True)
class GS1DecisionExample:
    """One compiled text branch and its option-probability target."""

    record_id: str
    question_id: str
    input_ids: tuple[int, ...]
    branch: GS1Branch
    state_length: int
    target_probs: tuple[float, ...]


@dataclass(frozen=True)
class GS1DecisionBatch:
    """Padded input and target tensors aligned with native compiler branches.

    Attributes:
        input_ids: Long Tensor [batch, sequence], left-padded.
        attention_mask: Boolean Tensor [batch, sequence], true for real tokens.
        position_ids: Long Tensor [batch, sequence], zero-based on real tokens.
        branches: Native branch objects in input row order.
        state_lengths: Shared-state length for each branch.
        target_probs: Float Tensor [batch, max_options], padded with zero.
        option_mask: Boolean Tensor [batch, max_options], true for real options.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    branches: tuple[GS1Branch, ...]
    state_lengths: tuple[int, ...]
    target_probs: torch.Tensor
    option_mask: torch.Tensor


@dataclass(frozen=True)
class GS1DecisionObjective:
    """Single-stage distribution NLL plus weighted expected multiclass Brier."""

    brier_weight: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.brier_weight) or self.brier_weight < 0:
            raise ValueError("brier_weight must be finite and nonnegative")

    def __call__(
        self,
        logits: torch.Tensor,
        target_probs: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss over variable-size candidate distributions.

        Args:
            logits: Tensor [questions, max_options] of native candidate scores.
            target_probs: Tensor [questions, max_options] of normalized targets.
            option_mask: Boolean Tensor [questions, max_options] for real options.

        Returns:
            Mean scalar objective and detached NLL/Brier metrics.
        """
        if logits.ndim != 2 or target_probs.shape != logits.shape or option_mask.shape != logits.shape:
            raise ValueError("logits, target_probs, and option_mask must share [questions, options] shape")
        if option_mask.dtype != torch.bool or not bool(option_mask.any(dim=-1).all()):
            raise ValueError("option_mask must be boolean with at least one option per question")
        if not bool(torch.isfinite(target_probs).all()) or bool((target_probs < 0).any()):
            raise ValueError("target_probs must be finite and nonnegative")
        target_sums = (target_probs * option_mask).sum(dim=-1)
        if not bool(torch.isclose(target_sums, torch.ones_like(target_sums), atol=1e-6, rtol=0).all()):
            raise ValueError("target probabilities must sum to one over valid options")
        if bool((target_probs.masked_select(~option_mask) != 0).any()):
            raise ValueError("padded target probabilities must be zero")

        log_probs = F.log_softmax(logits.float().masked_fill(~option_mask, float("-inf")), dim=-1)
        probs = log_probs.exp().masked_fill(~option_mask, 0.0)
        target = target_probs.float()
        nll = -(target * log_probs.masked_fill(~option_mask, 0.0)).sum(dim=-1).mean()
        brier = (((probs - target).square() * option_mask).sum(dim=-1) + 1.0 - target.square().sum(dim=-1)).mean()
        loss = nll + self.brier_weight * brier
        return loss, {"nll": nll.detach(), "expected_brier": brier.detach()}


def _contains_media(state: JsonValue) -> bool:
    return isinstance(state, list) and any(
        isinstance(item, dict) and item.get("type") in ("image", "video") for item in state
    )


def _candidate_ids(question: dict[str, JsonValue]) -> list[str]:
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice" and isinstance(criteria, dict):
        return list(criteria)
    if kind == "noul":
        return ["false", "true"]
    if kind == "score" and isinstance(criteria, list):
        return [str(index) for index in range(len(criteria))]
    raise ValueError(f"Unsupported or malformed GS1 question type: {kind!r}")


def load_gs1_examples(
    path: str | Path,
    *,
    compiler: GS1Compiler,
    request_type: type[BaseModel],
    model_id: str,
    max_examples: int = 3000,
    seed: int = 0,
) -> list[GS1DecisionExample]:
    """Sample and compile text-only questions from native GS1 JSONL.

    Args:
        path: JSONL file containing native ``request`` and ``targets`` records.
        compiler: Native compiler created by the loaded GS1 checkpoint.
        request_type: Dynamically loaded GS1 ``DecisionRequest`` class.
        model_id: Model identifier included in compiler requests.
        max_examples: Maximum questions selected by reservoir sampling.
        seed: Seed for reproducible sample selection.

    Returns:
        Compiled questions with probability targets in each branch's candidate order.
    """
    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    rng = random.Random(seed)
    selected: list[tuple[str, str, JsonValue, dict[str, JsonValue], GS1Target]] = []
    seen = 0
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            raw = cast(GS1InputRecord, json.loads(line))
            request_data = raw["request"]
            state = request_data.get("state")
            questions = request_data.get("questions")
            if state is None or not isinstance(questions, dict) or _contains_media(state):
                continue
            targets = raw["targets"]
            if set(questions) != set(targets):
                raise ValueError("GS1 question IDs must match target IDs")
            record_id = str(request_data.get("id", seen))
            for question_id, raw_question in questions.items():
                question = cast(dict[str, JsonValue], raw_question)
                candidates = _candidate_ids(question)
                target = targets[question_id]
                target_ids = target["candidate_ids"]
                distribution = target["distribution"]
                if len(candidates) < 2 or len(candidates) > 255:
                    continue
                if len(target_ids) != len(candidates) or set(target_ids) != set(candidates):
                    raise ValueError(f"Target candidate identities differ for question {question_id!r}")
                if len(distribution) != len(candidates):
                    raise ValueError(f"Target distribution length differs for question {question_id!r}")
                values = [float(value) for value in distribution]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError(f"Invalid target distribution for question {question_id!r}")
                if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f"Target distribution must sum to one for question {question_id!r}")
                row = (record_id, question_id, state, question, target)
                seen += 1
                if len(selected) < max_examples:
                    selected.append(row)
                else:
                    index = rng.randrange(seen)
                    if index < max_examples:
                        selected[index] = row

    examples = []
    for record_id, question_id, state, question, target in selected:
        request = request_type.model_validate({"model": model_id, "state": state, "questions": {question_id: question}})
        compiled = compiler.compile(request)
        branch = compiled.branches[0]
        target_by_id = dict(zip(target["candidate_ids"], target["distribution"], strict=True))
        target_probs = tuple(float(target_by_id[candidate]) for candidate in branch.candidates)
        examples.append(
            GS1DecisionExample(
                record_id=record_id,
                question_id=question_id,
                input_ids=(*compiled.state, *branch.suffix),
                branch=branch,
                state_length=len(compiled.state),
                target_probs=target_probs,
            )
        )
    if not examples:
        raise ValueError("No eligible text-only GS1 questions were found")
    return examples


def collate_gs1_examples(
    examples: list[GS1DecisionExample], *, pad_token_id: int, device: torch.device
) -> GS1DecisionBatch:
    """Left-pad native branches and align target distributions.

    Args:
        examples: Nonempty compiled questions.
        pad_token_id: Tokenizer pad ID, or EOS ID if no pad ID exists.
        device: Destination device for all input and target tensors.

    Returns:
        Batch tensors and native branch metadata in matching row order.
    """
    if not examples:
        raise ValueError("Cannot collate an empty GS1 batch")
    if any(len(item.input_ids) != item.state_length + len(item.branch.suffix) for item in examples):
        raise ValueError("Compiled branch lengths do not match input token sequences")
    batch_size = len(examples)
    sequence_length = max(len(item.input_ids) for item in examples)
    max_options = max(len(item.target_probs) for item in examples)
    input_ids = torch.full((batch_size, sequence_length), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.bool, device=device)
    position_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long, device=device)
    target_probs = torch.zeros((batch_size, max_options), dtype=torch.float32, device=device)
    option_mask = torch.zeros((batch_size, max_options), dtype=torch.bool, device=device)
    for row, item in enumerate(examples):
        length = len(item.input_ids)
        start = sequence_length - length
        input_ids[row, start:] = torch.tensor(item.input_ids, dtype=torch.long, device=device)
        attention_mask[row, start:] = True
        position_ids[row, start:] = torch.arange(length, dtype=torch.long, device=device)
        count = len(item.target_probs)
        target_probs[row, :count] = torch.tensor(item.target_probs, dtype=torch.float32, device=device)
        option_mask[row, :count] = True
    return GS1DecisionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        branches=tuple(item.branch for item in examples),
        state_lengths=tuple(item.state_length for item in examples),
        target_probs=target_probs,
        option_mask=option_mask,
    )

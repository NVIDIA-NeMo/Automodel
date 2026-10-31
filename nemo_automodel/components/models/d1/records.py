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

"""Published D1 text prompts with explicit question and option identities."""

import math
from dataclasses import dataclass
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class D1TextExample:
    """One decision in the publisher's internal option order.

    Marker positions and alias groups are mutually exclusive model readouts.
    Temperature applies to Omni text logits before training and evaluation.
    Recipes align labels and rewards separately using the question and option IDs.
    """

    record_id: str
    input_ids: tuple[int, ...]
    question_id: str
    option_ids: tuple[str, ...]
    marker_positions: tuple[int, ...]
    alias_token_ids: tuple[tuple[int, ...], ...]
    head_question_type: int
    temperature: float


def encode_d1_record(
    record: dict[str, Any],
    *,
    tokenizer: Any,
    prompt: ModuleType,
    omni: bool,
    max_length: int,
    temperatures: dict[str, float] | None = None,
) -> list[D1TextExample]:
    """Encode unchanged SOC records using reviewed pinned publisher prompt code.

    Args:
        record: Validated state/questions record. Supervision is not consumed.
        tokenizer: Tokenizer from the same immutable release as the prompt code.
        prompt: Imported publisher prompt module, pinned by the recipe.
        omni: Select the marker encoder instead of the causal token-alias prompt.
        max_length: Context limit; causal inputs are rejected rather than truncated.
        temperatures: Released Omni text calibration, unchanged across stages.

    Returns:
        Encoded questions with their model-output option identities. The caller
        aligns supervision with these identities separately.
    """
    result = []
    for qid, schema in record["questions"].items():
        question = prompt.as_question(schema)
        if schema["type"] == "noul":
            order = ("false", "true") if omni else ("true", "false")
        elif schema["type"] == "choice":
            order = tuple(str(key) for key in schema["criteria"])
        else:
            order = tuple(str(index) for index in range(len(schema["criteria"])))
        if omni:
            ids, markers = prompt.encode(tokenizer, record["state"], question, max_length)
            aliases = ()
            head_type = prompt.QTYPES[schema["type"]]
            calibration = temperatures or {}
            temperature = float(calibration.get(prompt.temperature_key(question), calibration.get(schema["type"], 1.0)))
        else:
            text = prompt.render(tokenizer, record["state"], question, bos=tokenizer.bos_token or "")
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > max_length:
                raise ValueError(f"{record['id']}/{qid}: causal prompt exceeds max_length")
            markers = ()
            aliases = tuple(tuple(group) for group in prompt.readout_ids(tokenizer, question))
            head_type = ("noul", "choice", "score").index(schema["type"])
            temperature = 1.0
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Released calibration temperatures must be finite and positive")
        if len(markers if omni else aliases) != len(order):
            raise ValueError("Published readout and question schema have different option counts")
        result.append(
            D1TextExample(str(record["id"]), tuple(ids), qid, order, tuple(markers), aliases, head_type, temperature)
        )
    return result

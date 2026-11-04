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

"""CLM prompt layout: how a SystemOne record becomes state texts and candidate texts.

This mirrors ``clm.schema`` from the ``contrastive-lm`` package (Apache-2.0) exactly, because the released CLM heads
were trained on this layout: the state head sees ``context + question``, the action head sees each option's own text.
"""

# Source layout: https://github.com/Contrastive-LM/CLM (Apache-2.0).
# License: https://github.com/Contrastive-LM/CLM/blob/main/LICENSE

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

CLM_NOUL_KEYS = ("false", "true")


def prepare_tokenizer(*, name_or_path: str, revision: str | None) -> PreTrainedTokenizerBase:
    """Load the tokenizer with the publisher's CLM serving input contract.

    Args:
        name_or_path: Tokenizer repository ID or local directory.
        revision: Tokenizer revision recorded in the frozen-feature cache identity.

    Returns:
        Tokenizer using right padding and left truncation, retaining the final
        prompt tokens as in clm-serve. Missing padding uses the EOS token.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def to_text(x: Any, indent: int = 0) -> str:
    """Render a JSON value as prose: objects become ``key: value`` lines, arrays become ``- item`` lines."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float)):
        return str(x)
    pad = " " * indent
    if isinstance(x, dict):
        parts = []
        for k, v in x.items():
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}{k}:\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}{k}: {to_text(v)}")
        return ("\n\n" if indent == 0 else "\n").join(parts)
    if isinstance(x, (list, tuple)):
        parts = []
        for v in x:
            if isinstance(v, (dict, list)) and v:
                parts.append(f"{pad}-\n{to_text(v, indent + 2)}")
            else:
                parts.append(f"{pad}- {to_text(v)}")
        return "\n".join(parts)
    return json.dumps(x, ensure_ascii=False)


def clm_state_text(state: Any, instructions: Any) -> str:
    """Context first, question last: the layout CLM's state head was trained on."""
    s, i = to_text(state).strip(), to_text(instructions).strip()
    return f"{s}\n\n{i}" if s and i else (s or i)


def clm_candidates(question: dict[str, Any]) -> tuple[list[str], list[str]]:
    """``(option ids in CLM order, candidate text per option)`` for one question."""
    qtype, crit = question["type"], question.get("criteria")
    instructions = to_text(question.get("instructions")).strip()
    if qtype == "choice":
        keys = [str(k) for k in crit]
        return keys, [to_text(crit[k]) if crit[k] not in (None, "") else str(k) for k in crit]
    if qtype == "score":
        return [str(i) for i in range(len(crit))], [to_text(c) for c in crit]
    crit = crit or {}
    texts = []
    for key in CLM_NOUL_KEYS:
        desc = crit.get(key) if isinstance(crit, dict) else None
        if desc in (None, ""):
            if instructions:
                desc = f"Yes. This is true: {instructions}" if key == "true" else f"No. This is false: {instructions}"
            else:
                desc = key
        texts.append(f"{key}: {to_text(desc)}")
    return list(CLM_NOUL_KEYS), texts


def clm_pairs(record: dict[str, Any]) -> dict[str, tuple[str, list[str], list[str]]]:
    """``{question_id: (state_text, option_ids, candidate_texts)}`` for every question of a record."""
    return {
        qid: (clm_state_text(record["state"], q.get("instructions")), *clm_candidates(q))
        for qid, q in record["questions"].items()
    }

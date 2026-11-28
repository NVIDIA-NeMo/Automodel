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

# Code provenance: control-token sanitizing (``_control_tokens``, ``_template_literal_text``, ``_mark``,
# ``_encode_marked``) is adapted from Hugging Face Transformers v5.23.0
# (https://github.com/huggingface/transformers/tree/v5.23.0, src/transformers/utils/chat_template_utils.py:
# ``_get_template_literal_text``, ``mark_control_tokens``, ``encode_marked_chat``, ``_encode_chat_piece``;
# src/transformers/tokenization_utils_base.py: ``_get_control_tokens``), Apache-2.0, Copyright The HuggingFace Team.
# It reproduces ``apply_chat_template(..., sanitize_control_tokens=True)`` on the Transformers version pinned here.

"""The decision prompt of a Nemotron 3.5 Decision checkpoint: one question per read, one trailing mask token.

The checkpoint owns its prompt contract, and training reads each question exactly as the checkpoint does:

* ``decision.json`` gives the option codes per question type and the read limits; ``decision_template.jinja`` renders
  the prompt around ``decision_content``, the JSON ``{"state", "question", "options": [{"code", "label", "meaning"}]}``;
* options keep the question's order (``choice``: criteria order; ``noul``: ``false``, ``true``; ``score``: level 0
  upwards) and take the type's codes by position;
* control tokens inside the user content are encoded as plain text, so only the template's own control tokens are
  special;
* the template ends with the mask token, and the answer is the head's distribution at that mask restricted to the
  option codes.

An optional, explicit option shuffle (training only) permutes the options and re-codes them by position, so the model
learns option meanings rather than codes. The permutation seed is ``crc32(f"{seed}|perm")``, independent of the coin
that decides whether a row is shuffled.

Prompt formats (``prompt_format``): ``json`` (default) writes the JSON above unchanged. ``json-sep`` writes the same
JSON with the state-end token (``<SPECIAL_101>`` by default) right after the state value, closing the state:
``{"state": ...<SPECIAL_101>, "question": ..., "options": [...]}``. Only that one token is a control token; the same
text inside the user content is still encoded as plain text. Everything up to and including the state-end token is
the state prefix that packed reads share (see ``packed.py``).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DECISION_CONFIG_FILE = "decision.json"
DECISION_TEMPLATE_FILE = "decision_template.jinja"
QUESTION_TYPES = ("choice", "noul", "score")
PROMPT_FORMATS = ("json", "json-sep")
DEFAULT_STATE_END_TOKEN = "<SPECIAL_101>"
NOUL_OPTIONS = ("false", "true")
NOUL_DEFAULT_MEANINGS = {"false": "No, the condition does not hold.", "true": "Yes, the condition holds."}
_QUESTION_FIELDS = frozenset({"type", "instructions", "criteria"})
_MARK_START, _MARK_END = "\U000f0000", "\U000f0001"


@dataclass(frozen=True)
class DecisionContract:
    """A checkpoint's decision prompt contract (``decision.json`` and ``decision_template.jinja``).

    Attributes:
        template: The Jinja decision template; it renders ``decision_content`` and ends with the mask token.
        labels: Option codes per question type, in position order.
        max_input_tokens: Longest read the checkpoint accepts, mask included.
    """

    template: str
    labels: dict[str, tuple[str, ...]]
    max_input_tokens: int

    @classmethod
    def from_pretrained(cls, path: str | Path, *, revision: str | None = None) -> DecisionContract:
        """Load and validate the contract of a local checkpoint directory or Hugging Face repository.

        Raises:
            ValueError: If a file is missing or the contract is malformed.
        """
        config = json.loads(_read_checkpoint_file(path, DECISION_CONFIG_FILE, revision))
        template = _read_checkpoint_file(path, DECISION_TEMPLATE_FILE, revision)
        if config.get("noul_options", list(NOUL_OPTIONS)) != list(NOUL_OPTIONS):
            raise ValueError(f"{DECISION_CONFIG_FILE}: noul_options must be {list(NOUL_OPTIONS)}")
        labels = {}
        for kind in QUESTION_TYPES:
            codes = (config.get("labels") or {}).get(kind) or []
            if not codes or len(set(codes)) != len(codes) or not all(isinstance(c, str) and c for c in codes):
                raise ValueError(f"{DECISION_CONFIG_FILE}: labels.{kind} must be distinct non-empty strings")
            labels[kind] = tuple(codes)
        if len(labels["noul"]) < len(NOUL_OPTIONS):
            raise ValueError(f"{DECISION_CONFIG_FILE}: labels.noul needs {len(NOUL_OPTIONS)} codes")
        max_input_tokens = config.get("max_input_tokens", 131072)
        if type(max_input_tokens) is not int or max_input_tokens < 1:
            raise ValueError(f"{DECISION_CONFIG_FILE}: max_input_tokens must be a positive integer")
        return cls(template=template, labels=labels, max_input_tokens=max_input_tokens)

    @property
    def sha256(self) -> str:
        """Identity of the contract (template, codes and limit)."""
        payload = {"template": self.template, "labels": self.labels, "max_input_tokens": self.max_input_tokens}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_checkpoint_file(path: str | Path, name: str, revision: str | None) -> str:
    local = Path(path) / name
    if Path(path).is_dir():
        if not local.is_file():
            raise ValueError(f"{path} has no {name}; decision post-training needs a Nemotron 3.5 Decision checkpoint")
        return local.read_text()
    from transformers.utils import cached_file

    resolved = cached_file(str(path), name, revision=revision, _raise_exceptions_for_missing_entries=False)
    if resolved is None:
        raise ValueError(f"{path} has no {name}; decision post-training needs a Nemotron 3.5 Decision checkpoint")
    return Path(resolved).read_text()


@dataclass(frozen=True)
class PreparedQuestion:
    """One tokenized decision prompt (without the trailing mask) and its option readout."""

    question_id: str
    kind: str
    labels: tuple[str, ...]  # API option labels in prompt order
    code_token_ids: tuple[int, ...]  # single-token code of each option, same order as ``labels``
    input_ids: tuple[int, ...]  # prompt tokens; the mask token is appended by the caller
    state_length: int | None = None  # json-sep: tokens up to and including the state-end token; json: None


def perm_seed(shuffle_seed: int) -> int:
    """Option-permutation seed, independent of the shuffle coin drawn from ``random.Random(shuffle_seed)``."""
    return zlib.crc32(f"{shuffle_seed}|perm".encode())


def question_options(question: dict[str, Any]) -> tuple[list[str], list[Any]]:
    """``(labels, meanings)`` of a question in checkpoint order (noul: false, true; score: level indices).

    Raises:
        ValueError: If the question has unsupported fields, an unknown type or criteria that do not fit the type.
    """
    if not isinstance(question, dict) or set(question) - _QUESTION_FIELDS:
        raise ValueError(f"a question may only have the fields {sorted(_QUESTION_FIELDS)}")
    if "instructions" not in question:
        raise ValueError("a question needs instructions")
    kind = question.get("type")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2 or not all(isinstance(k, str) and k for k in criteria):
            raise ValueError("a choice question needs criteria {label: meaning} with at least two named options")
        return list(criteria), list(criteria.values())
    if kind == "noul":
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) - set(NOUL_OPTIONS)):
            raise ValueError("a noul question's criteria may only contain false and true")
        criteria = criteria or {}
        return list(NOUL_OPTIONS), [criteria.get(option, NOUL_DEFAULT_MEANINGS[option]) for option in NOUL_OPTIONS]
    if kind == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("a score question needs criteria [level 0, level 1, ...] with at least two levels")
        return [str(i) for i in range(len(criteria))], list(criteria)
    raise ValueError(f"question type must be one of {QUESTION_TYPES}, got {kind!r}")


def label_token_ids(tokenizer: Any, codes: tuple[str, ...]) -> tuple[int, ...]:
    """Token ids of option codes, which must be distinct single known tokens.

    Raises:
        ValueError: If a code is not exactly one known token or two codes share a token.
    """
    ids = tokenizer(list(codes), add_special_tokens=False)["input_ids"]
    found = [t[0] for t in ids if len(t) == 1 and t[0] != tokenizer.unk_token_id]
    if len(set(found)) != len(codes):
        raise ValueError(f"the codes {list(codes)} are not distinct single tokens for this tokenizer")
    return tuple(int(t) for t in found)


def readout_token_ids(tokenizer: Any, contract: DecisionContract) -> tuple[int, ...]:
    """Every token id a question can read at its mask (all option codes of every type), sorted."""
    codes = sorted({code for kind in QUESTION_TYPES for code in contract.labels[kind]})
    return tuple(sorted(set(label_token_ids(tokenizer, tuple(codes)))))


def prepare_question(
    tokenizer: Any,
    contract: DecisionContract,
    state: Any,
    question_id: str,
    question: dict[str, Any],
    *,
    mask_token_id: int,
    shuffle_seed: int | None = None,
    prompt_format: str = "json",
    state_end_token: str = DEFAULT_STATE_END_TOKEN,
) -> PreparedQuestion:
    """Render one question with the checkpoint's decision template, optionally with shuffled, re-coded options.

    Args:
        tokenizer: Tokenizer of the checkpoint.
        contract: The checkpoint's decision contract.
        state: JSON-serializable evidence shared by the question.
        question_id: Key of the question (used in errors).
        question: ``{"type", "instructions", "criteria"}`` of a choice, noul or score question.
        mask_token_id: The checkpoint's mask token, which the template must write exactly once, at the end.
        shuffle_seed: When set, permute the options with ``perm_seed(shuffle_seed)`` and re-code them by position;
            ``None`` keeps the checkpoint's option order.
        prompt_format: ``json`` (the user content is the JSON object) or ``json-sep`` (the same JSON with
            ``state_end_token`` right after the state value).
        state_end_token: The control token that closes the state in ``json-sep`` prompts.

    Returns:
        The prepared prompt; token ids exclude the trailing mask. For ``json-sep``, ``state_length`` counts the tokens
        up to and including the state-end token.

    Raises:
        ValueError: If the question is malformed, has more options than codes, the template does not end in exactly
            one mask token, or a ``json-sep`` prompt does not hold exactly one state-end token before its questions.
    """
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(f"prompt_format must be one of {PROMPT_FORMATS}, got {prompt_format!r}")
    try:
        labels, meanings = question_options(question)
    except ValueError as error:
        raise ValueError(f"question {question_id!r}: {error}") from None
    kind = question["type"]
    if len(labels) > len(contract.labels[kind]):
        raise ValueError(f"question {question_id!r}: at most {len(contract.labels[kind])} {kind} options")
    codes = contract.labels[kind][: len(labels)]
    order = list(range(len(labels)))
    if shuffle_seed is not None:
        random.Random(perm_seed(shuffle_seed)).shuffle(order)
    options = [{"code": codes[i], "label": labels[j], "meaning": meanings[j]} for i, j in enumerate(order)]
    state_length = None
    if prompt_format == "json":
        content = json.dumps(
            {"state": state, "question": question["instructions"], "options": options},
            ensure_ascii=False,
            allow_nan=False,
        )
        ids = render_decision_ids(tokenizer, contract.template, content)
    else:
        # The same JSON split after the state value: '{"state": <state>' + state-end token + ', "question": ...}'.
        head = json.dumps({"state": state}, ensure_ascii=False, allow_nan=False)[:-1]
        tail = (
            ", "
            + json.dumps(
                {"question": question["instructions"], "options": options}, ensure_ascii=False, allow_nan=False
            )[1:]
        )
        ids = render_decision_ids(tokenizer, contract.template, head, tail=tail, state_end_token=state_end_token)
        state_end_id = tokenizer.convert_tokens_to_ids(state_end_token)
        if ids.count(state_end_id) != 1:
            raise ValueError(f"a json-sep prompt must hold exactly one {state_end_token!r} token")
        state_length = ids.index(state_end_id) + 1
    if not ids or ids[-1] != mask_token_id or ids.count(mask_token_id) != 1:
        raise ValueError("the decision template must produce exactly one trailing mask token")
    return PreparedQuestion(
        question_id=str(question_id),
        kind=kind,
        labels=tuple(labels[j] for j in order),
        code_token_ids=label_token_ids(tokenizer, codes),
        input_ids=tuple(int(t) for t in ids[:-1]),
        state_length=state_length,
    )


def render_decision_ids(
    tokenizer: Any,
    template: str,
    content: str,
    *,
    tail: str | None = None,
    state_end_token: str | None = None,
) -> list[int]:
    """Token ids of ``template`` rendered around ``content``, with control tokens in ``content`` encoded as text.

    With ``state_end_token``, the user content is ``content + state_end_token + tail``: only the inserted token is a
    control token; control-token text inside ``content`` and ``tail`` is still encoded as text.
    """
    control = _pattern(_control_tokens(tokenizer, template))
    decision_content = _mark(content, control)
    if state_end_token is not None:
        if state_end_token not in tokenizer.added_tokens_encoder:
            raise ValueError(f"the tokenizer has no {state_end_token!r} token for json-sep prompts")
        decision_content += state_end_token + _mark(tail or "", control)
    elif tail is not None:
        raise ValueError("tail needs a state_end_token")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        chat_template=template,
        decision_content=decision_content,
        tokenize=False,
        add_generation_prompt=False,
    )
    return _encode_marked(tokenizer, rendered)


def _control_tokens(tokenizer: Any, template: str) -> tuple[str, ...]:
    """Special tokens plus the declared (or, if none are declared, template-literal) added tokens."""
    added = list(tokenizer.added_tokens_decoder.values())
    declared = getattr(tokenizer, "chat_control_tokens", None)
    if declared is None:
        declared = (getattr(tokenizer, "init_kwargs", None) or {}).get("chat_control_tokens")
    if declared is not None:
        declared = set(declared)
        return tuple(t.content for t in added if t.special or t.content in declared)
    literal = _template_literal_text(template)
    return tuple(t.content for t in added if t.special or (t.content in literal and not t.content.isspace()))


@lru_cache(maxsize=8)
def _template_literal_text(template: str) -> str:
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    ast = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).parse(template)
    literals = {node.data for node in ast.find_all(jinja2.nodes.TemplateData)}
    literals |= {node.value for node in ast.find_all(jinja2.nodes.Const) if isinstance(node.value, str)}
    return "\0".join(literals)


@lru_cache(maxsize=8)
def _pattern(tokens: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(map(re.escape, sorted(tokens, key=len, reverse=True))) or "(?!)")


def _mark(text: str, pattern: re.Pattern[str]) -> str:
    text = text.replace(_MARK_START, "").replace(_MARK_END, "")
    return pattern.sub(f"{_MARK_START}\\g<0>{_MARK_END}", text)


def _encode_marked(tokenizer: Any, chat: str) -> list[int]:
    """Encode a rendered chat: marked control tokens as plain text, everything else normally."""
    tokens = _pattern(tuple(tokenizer.added_tokens_encoder)).pattern
    pieces = re.split(f"{_MARK_START}({tokens}){_MARK_END}|[{_MARK_START}{_MARK_END}]", chat)
    newline = tokenizer.encode("\n", add_special_tokens=False)
    ids = tokenizer.encode(pieces[0], add_special_tokens=False)
    for i, piece in enumerate(pieces[1:], start=1):
        if not piece:
            continue
        as_text = i % 2 == 1  # odd pieces are marked control tokens
        piece_ids = _encode_piece(tokenizer, piece, as_text, newline)
        if as_text and tokenizer.convert_tokens_to_ids(piece) in piece_ids:
            raise ValueError(f"cannot encode {piece!r} as text: the tokenizer produces it from plain text")
        ids += piece_ids
    return ids


def _encode_piece(tokenizer: Any, piece: str, as_text: bool, newline: list[int]) -> list[int]:
    def encode(text: str) -> list[int]:
        return (
            _encode_without_added_tokens(tokenizer, text)
            if as_text
            else tokenizer.encode(text, add_special_tokens=False)
        )

    # Encode after a newline so a SentencePiece-style model adds no prefix space, then drop the newline.
    ids = encode("\n" + piece)
    if ids[: len(newline)] == newline:
        return ids[len(newline) :]
    return encode(piece)


def _encode_without_added_tokens(tokenizer: Any, text: str) -> list[int]:
    """Encode ``text`` through the backend's normalizer, pre-tokenizer and model, matching no added token."""
    backend = tokenizer.backend_tokenizer
    if backend.normalizer is not None:
        text = backend.normalizer.normalize_str(text)
    splits = [text]
    if backend.pre_tokenizer is not None:
        splits = [split for split, _ in backend.pre_tokenizer.pre_tokenize_str(text)]
    return [token.id for split in splits for token in backend.model.tokenize(split)]

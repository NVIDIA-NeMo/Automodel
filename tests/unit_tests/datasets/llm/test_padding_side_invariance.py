#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Padding-side invariance for the shared SFT formatting path.

``_package_tokenized_example`` builds the attention mask by stripping
*trailing* pads. A left-padding tokenizer has no trailing pad run, so the
mask came out all-ones and every real token attended to the leading pad run
-- silently, because labels are still ``-100`` over the pad positions and the
loss curve looks normal.

Every LLM text dataset (squad, chat_dataset, xlam, agent_chat,
column_mapped_text_instruction, ...) reaches that function through exactly
two callers, :func:`format_prompt_completion` and
:func:`format_chat_template`, so the invariants below are asserted once at
that chokepoint rather than per dataset. ``test_all_formatter_call_sites_covered``
fails if a third caller appears and bypasses them.

Data prep only ever sees a tokenizer, never a model, so "does this hold for
every model" reduces to "does this hold for every tokenizer *shape*". The
parametrization enumerates those shapes: pad unset, pad == eos, pad id 0
(falsy), and pad != eos.
"""

import pytest

from nemo_automodel.components.datasets.llm import formatting_utils

# Tokenizer shapes that data prep actually distinguishes. Each maps to real
# checkpoints; the model architecture behind them is irrelevant here.
#   pad unset  -> meta-llama/Llama-3.1-8B, mistralai/Ministral-8B-Instruct-2410
#   pad == eos -> what _add_pad_token falls back to for the above
#   pad id 0   -> google/gemma-2-9b-it (falsy pad id)
#   pad != eos -> meta-llama/Llama-3.3-70B-Instruct (<|finetune_right_pad_id|>)
TOKENIZER_SHAPES = {
    "pad_unset": {"pad_token_id": None, "eos_token_id": 9},
    "pad_eq_eos": {"pad_token_id": 9, "eos_token_id": 9},
    "pad_id_zero": {"pad_token_id": 0, "eos_token_id": 9},
    "pad_ne_eos": {"pad_token_id": 4, "eos_token_id": 9},
}

CHAT_TEMPLATE = "{% for m in messages %}{{ m['content'] }}{% endfor %}"


class _PaddingTokenizer:
    """Tokenizer stub that honors ``padding_side`` the way a real HF one does."""

    def __init__(self, pad_token_id, eos_token_id, padding_side="right", chat_template=None):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.padding_side = padding_side
        self.chat_template = chat_template
        self.bos_token_id = 1
        self.add_bos_token = False
        self.name_or_path = "stub"
        self.observed_padding_sides = []

    @staticmethod
    def _encode(text):
        # Deterministic, vocab-free: one token per non-space char, ids >= 100 so
        # they can never collide with the pad/eos ids used above.
        return [100 + (ord(c) % 50) for c in text if not c.isspace()]

    def _pad(self, ids, padding, max_length):
        mask = [1] * len(ids)
        if padding != "max_length" or max_length is None or len(ids) >= max_length:
            # Unpadded calls (prompt-length probes, prefix re-tokenization for the
            # multiturn mask) are padding-side agnostic; only record real padding.
            return {"input_ids": ids, "attention_mask": mask}
        self.observed_padding_sides.append(self.padding_side)
        pad_id = self.pad_token_id if self.pad_token_id is not None else 0
        n = max_length - len(ids)
        if self.padding_side == "left":
            return {"input_ids": [pad_id] * n + ids, "attention_mask": [0] * n + mask}
        return {"input_ids": ids + [pad_id] * n, "attention_mask": mask + [0] * n}

    def __call__(self, text, add_special_tokens=True, padding=False, truncation=None, max_length=None):
        return self._pad(self._encode(text), padding, max_length)

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=None,
        max_length=None,
    ):
        ids = self._encode("".join(m["content"] for m in messages))
        return self._pad(ids, padding, max_length)


def _make(shape, side, chat):
    return _PaddingTokenizer(
        padding_side=side,
        chat_template=CHAT_TEMPLATE if chat else None,
        **TOKENIZER_SHAPES[shape],
    )


def _format(tok, chat, seq_length=64):
    kwargs = dict(
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        seq_length=seq_length,
        padding="max_length",
        truncation=True,
    )
    if chat:
        return formatting_utils.format_chat_template(
            tokenizer=tok,
            formatted_text=[
                {"role": "user", "content": "what is the capital of France"},
                {"role": "assistant", "content": "Paris"},
            ],
            **kwargs,
        )
    return formatting_utils.format_prompt_completion(
        tokenizer=tok, prompt="what is the capital of France", answer="Paris", **kwargs
    )


@pytest.mark.parametrize("shape", sorted(TOKENIZER_SHAPES))
@pytest.mark.parametrize("chat", [False, True], ids=["prompt_completion", "chat_template"])
def test_padding_side_does_not_change_the_example(shape, chat):
    """left and right padding must produce byte-identical training examples."""
    right = _format(_make(shape, "right", chat), chat)
    left = _format(_make(shape, "left", chat), chat)
    assert left == right


@pytest.mark.parametrize("shape", sorted(TOKENIZER_SHAPES))
@pytest.mark.parametrize("chat", [False, True], ids=["prompt_completion", "chat_template"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_no_padding_is_attended(shape, chat, side):
    """The attention mask must never cover the pad run (the original bug)."""
    tok = _make(shape, side, chat)
    out = _format(tok, chat)
    ids, mask = out["input_ids"], out["attention_mask"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    assert len(mask) == len(ids)
    # The example is padded, so the mask must not be all ones.
    assert 0 < sum(mask) < len(mask)
    # No leading pad run may survive, and nothing attended may be trailing pad.
    assert ids[0] != pad_id or mask[0] == 0
    attended_tail = [i for i, m in enumerate(mask) if m and ids[i] == pad_id]
    # pad_id may legitimately equal eos and appear as real content; what must not
    # happen is an attended *contiguous run* at either end.
    assert all(i < sum(mask) for i in attended_tail)


@pytest.mark.parametrize("shape", sorted(TOKENIZER_SHAPES))
@pytest.mark.parametrize("chat", [False, True], ids=["prompt_completion", "chat_template"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_no_supervision_on_unattended_positions(shape, chat, side):
    """A position the model cannot attend to must not carry a loss target."""
    out = _format(_make(shape, side, chat), chat)
    mask = out["attention_mask"]
    targets = out["labels"] if "labels" in out else out["loss_mask"]
    ignore = -100 if "labels" in out else 0
    for i, m in enumerate(mask):
        if not m:
            assert targets[i] == ignore, f"supervised target at unattended position {i}"


@pytest.mark.parametrize("chat", [False, True], ids=["prompt_completion", "chat_template"])
def test_tokenizer_is_pinned_to_right_padding(chat):
    """Both formatters must tokenize under right padding, and restore after."""
    tok = _make("pad_ne_eos", "left", chat)
    _format(tok, chat)
    assert tok.observed_padding_sides, "tokenizer was never invoked"
    assert set(tok.observed_padding_sides) == {"right"}
    assert tok.padding_side == "left", "caller's padding_side must be restored"


def test_left_padded_input_is_normalized_at_the_chokepoint():
    """Pre-tokenized left-padded ids reaching the packer directly are rotated."""
    pad, eos = 4, 9
    content = [101, 102, 103, 104]
    ids = [pad] * 3 + content
    out = formatting_utils._package_tokenized_example(
        tokenizer=_make("pad_ne_eos", "left", False),
        input_ids=list(ids),
        assistant_masks=[0] * 3 + [0, 0, 1, 1],
        eos_token_id=eos,
        pad_token_id=pad,
        seq_length=None,
        attention_mask=[0] * 3 + [1] * 4,
    )
    # pad run rotated to the end, so no leading pad is attended
    assert out["input_ids"][0] != pad
    assert out["attention_mask"][0] == 1
    assert sum(out["attention_mask"]) < len(out["attention_mask"])


def test_all_formatter_call_sites_covered():
    """Guard: _package_tokenized_example must stay reachable only via the two
    pinned formatters. A new caller needs its own padding-side guarantee."""
    import inspect
    from pathlib import Path

    root = Path(inspect.getfile(formatting_utils)).parents[3]
    callers = set()
    for path in root.rglob("*.py"):
        for num, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "_package_tokenized_example(" in line and not line.lstrip().startswith(("#", "``", '"')):
                if "``" in line:  # docstring reference
                    continue
                callers.add(f"{path.relative_to(root)}:{num}")
    assert all("formatting_utils.py" in c for c in callers), (
        f"_package_tokenized_example called outside formatting_utils.py: {sorted(callers)}. "
        "New callers must pin padding_side or pass the tokenizer's attention_mask."
    )

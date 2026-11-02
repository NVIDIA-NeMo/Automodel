# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""``chat_template_kwargs``: extra chat template variables applied to every render.

The tokenizer below renders the real Jinja template and maps each character to one
token id, so the tests see exactly what the loss masks are built from. A mask builder
that rendered its conversation prefixes without the kwargs would frame them differently
from the full conversation, which these tests would catch as a wrong supervised span or
an inconsistent-prefix error.
"""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

import nemo_automodel.components.datasets.llm.chat_dataset as chat_dataset
import nemo_automodel.components.datasets.llm.eagle3 as eagle3
from nemo_automodel.components.datasets.llm.formatting_utils import format_chat_template

# Thinking-off renders an empty reasoning block, as Qwen3's template does; no generation
# block, so answer-only masking goes through the prefix-rendering fallback.
THINKING_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<u>{{ m.content }}"
    "{%- elif m.role == 'assistant' -%}<a>"
    "{%- if enable_thinking is defined and not enable_thinking -%}<t></t>{%- endif -%}"
    "{{ m.content }}</a>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<a>"
    "{%- if enable_thinking is defined and not enable_thinking -%}<t></t>{%- endif -%}"
    "{%- endif -%}"
)
# Reasoning is rendered only when the template is asked to show it.
REASONING_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<u>{{ m.content }}"
    "{%- elif m.role == 'assistant' -%}<a>"
    "{%- if show_reasoning -%}<r>{{ m.reasoning_content }}</r>{%- endif -%}"
    "{{ m.content }}</a>"
    "{%- endif -%}"
    "{%- endfor -%}"
)
# Iterates a nested kwarg, so the mapping's order is part of the rendered text.
OPTIONS_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m.role == 'user' -%}<u>{%- for k, v in options.items() -%}{{ k }}={{ v }};{%- endfor -%}{{ m.content }}"
    "{%- elif m.role == 'assistant' -%}<a>{{ m.content }}</a>"
    "{%- endif -%}"
    "{%- endfor -%}"
)
TWO_TURNS = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2"},
]
V41_TEMPLATE = (
    Path(__file__).parents[4] / "examples/llm_finetune/deepseek_v41/deepseek_v41_flash_training_chat_template.jinja"
).read_text()


class _JinjaCharTokenizer:
    """Renders a real Jinja chat template and maps every character to one token id."""

    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def __init__(self, chat_template: str) -> None:
        self.chat_template = chat_template

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        return_dict=False,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=False,
        max_length=None,
        add_generation_prompt=False,
        **template_kwargs,
    ):
        (text,), (spans,) = render_jinja_template(
            conversations=[messages],
            tools=tools,
            chat_template=self.chat_template,
            return_assistant_tokens_mask=True,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        ids = [ord(ch) for ch in text]
        if not return_dict:
            return ids
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_assistant_tokens_mask:
            mask = [0] * len(ids)
            for start, end in spans:
                mask[start:end] = [1] * (end - start)
            out["assistant_masks"] = mask
        return out


def _format(template, messages, **kwargs):
    """Return (rendered text, supervised text) for one conversation."""
    sample = format_chat_template(
        _JinjaCharTokenizer(template), messages, eos_token_id=0, pad_token_id=0, unshifted=True, **kwargs
    )
    text = "".join(chr(i) for i in sample["input_ids"])
    supervised = "".join(chr(i) for i, keep in zip(sample["input_ids"], sample["loss_mask"]) if keep)
    return text, supervised


def test_kwargs_reach_the_render_and_the_answer_only_mask():
    text, supervised = _format(THINKING_TEMPLATE, TWO_TURNS, chat_template_kwargs={"enable_thinking": False})
    assert text == "<u>Q1<a><t></t>A1</a><u>Q2<a><t></t>A2</a>"
    # The prefix renders carry the kwargs too, so each assistant span is located exactly.
    assert supervised == "<a><t></t>A1</a><a><t></t>A2</a>"
    assert _format(THINKING_TEMPLATE, TWO_TURNS) == ("<u>Q1<a>A1</a><u>Q2<a>A2</a>", "<a>A1</a><a>A2</a>")


def test_pinned_enable_thinking_drives_the_generation_prompt_mask():
    # The builder would otherwise render the prompt with enable_thinking=True and False itself.
    # The prompt it renders, <a><t></t>, is the template-supplied part of each turn.
    _, supervised = _format(
        THINKING_TEMPLATE, TWO_TURNS, chat_template_kwargs={"enable_thinking": False}, mask_generation_prompt=True
    )
    assert supervised == "A1</a>A2</a>"


def test_kwargs_reach_the_reasoning_mask():
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A", "reasoning_content": "R"},
    ]
    text, supervised = _format(
        REASONING_TEMPLATE, messages, chat_template_kwargs={"show_reasoning": True}, mask_reasoning_content=True
    )
    assert text == "<u>Q<a><r>R</r>A</a>"
    # Only the reasoning text is removed: the cleared render still shows the empty <r></r> block.
    assert supervised == "<a><r></r>A</a>"


def test_nested_kwargs_order_reaches_the_render():
    # Equal mappings under Python ==, rendered differently, so caches must record the order.
    def render(options):
        return _format(OPTIONS_TEMPLATE, TWO_TURNS[:2], chat_template_kwargs={"options": options})[0]

    assert render({"b": 2, "a": 1}) == "<u>b=2;a=1;Q1<a>A1</a>"
    assert render({"a": 1, "b": 2}) == "<u>a=1;b=2;Q1<a>A1</a>"


def test_v41_training_template_takes_thinking_mode_from_kwargs():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
    ]
    _, supervised = _format(V41_TEMPLATE, messages, chat_template_kwargs={"thinking_mode": "thinking"})
    eos = "<｜end▁of▁sentence｜>"
    assert supervised == "A1" + eos + "R2</think>A2" + eos


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"tools": []}, ValueError, "may not set tools"),
        ({"add_generation_prompt": True, "truncation": True}, ValueError, "add_generation_prompt, truncation"),
        # Bound by the Jinja render itself, or silently changes it.
        ({"messages": [], "continue_final_message": True}, ValueError, "continue_final_message, messages"),
        ("enable_thinking=false", TypeError, "must be a mapping"),
    ],
)
def test_chat_dataset_rejects_kwargs_it_cannot_forward(kwargs, error, match):
    with pytest.raises(error, match=match):
        chat_dataset.ChatDataset("unused.jsonl", _JinjaCharTokenizer(THINKING_TEMPLATE), chat_template_kwargs=kwargs)


def test_format_chat_template_rejects_reserved_kwargs_for_direct_callers():
    with pytest.raises(ValueError, match="may not set documents"):
        _format(THINKING_TEMPLATE, TWO_TURNS, chat_template_kwargs={"documents": []})


def test_chat_dataset_forwards_kwargs_to_every_sample(monkeypatch):
    monkeypatch.setattr(chat_dataset, "_load_openai_messages", lambda *_a, **_k: [{"messages": TWO_TURNS}])
    kwargs = {"enable_thinking": False}
    dataset = chat_dataset.ChatDatasetConfig(
        path_or_dataset_id="unused.jsonl", chat_template_kwargs=kwargs, unshifted=True
    ).build(tokenizer=_JinjaCharTokenizer(THINKING_TEMPLATE))
    kwargs["enable_thinking"] = True  # the dataset keeps its own copy
    assert "".join(chr(i) for i in dataset[0]["input_ids"]) == "<u>Q1<a><t></t>A1</a><u>Q2<a><t></t>A2</a>"


@pytest.mark.parametrize("packed_sequence_size", [0, 64])
def test_eagle3_dataloader_forwards_kwargs(monkeypatch, packed_sequence_size):
    captured = {}

    class _Dataset(list):
        pad_token_id = 0

        def __init__(self, *_args, **kwargs):
            super().__init__([{"input_ids": [1, 2], "loss_mask": [0, 1], "attention_mask": [1, 1]}])
            captured.update(kwargs)

    monkeypatch.setattr(eagle3, "ChatDataset", _Dataset)
    eagle3.build_eagle3_dataloader(
        data_path="unused.jsonl",
        tokenizer=None,
        seq_length=2,
        batch_size=1,
        shuffle=False,
        packed_sequence_size=packed_sequence_size,
        chat_template_kwargs={"thinking_mode": "thinking"},
    )
    assert captured["chat_template_kwargs"] == {"thinking_mode": "thinking"}

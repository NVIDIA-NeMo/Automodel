# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import importlib
import sys
import types

import pytest
import torch


CONVERSATION = [
    {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
]


class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self.eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def decode(self, token):
        if isinstance(token, torch.Tensor):
            token = token.item()
        return str(token)


class DummyQwen25Processor:
    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)

    def apply_chat_template(self, conversation, *, tokenize=False, **kwargs):
        assert tokenize is False
        return "dummy chat string"

    def __call__(self, *, text, images, padding, return_tensors):
        batch_size = len(text)
        input_ids = torch.arange(1, 6).unsqueeze(0).repeat(batch_size, 1)
        return {
            "input_ids": input_ids,
            "pixel_values": torch.zeros(batch_size, 3, 224, 224, dtype=torch.float32),
        }


class DummyDefaultProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)

    def apply_chat_template(
        self,
        conv_list,
        *,
        tokenize,
        add_generation_prompt=True,
        padding=False,
        truncation=False,
        return_tensors,
        return_dict=True,
    ):
        assert tokenize and return_tensors == "pt" and return_dict
        batch_size = len(conv_list)
        input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
        pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
        return {"input_ids": input_ids, "pixel_values": pixel_values}


class DummyQwen3OmniProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)
        self.call_kwargs = []

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt is False
        assert tokenize is False
        return "chat:" + conversation[0]["content"][0]["text"]

    def __call__(self, *, text, return_tensors, padding, **kwargs):
        assert return_tensors == "pt"
        assert padding is True
        self.call_kwargs.append(dict(kwargs))
        batch_size = len(text)
        input_ids = torch.arange(1, 6).unsqueeze(0).repeat(batch_size, 1)
        return {"input_ids": input_ids}


class DummyPhi4Processor:
    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)
        self.chat_calls = []
        self.forward_calls = []
        self.produced_input_ids = None

    def apply_chat_template(self, conversation, *, tokenize, **kwargs):
        assert tokenize is False
        self.chat_calls.append({"conversation": conversation, "kwargs": kwargs})
        return "chat::" + conversation[0]["content"][0]["text"]

    def __call__(
        self,
        *,
        text,
        audios,
        return_tensors,
        padding,
        truncation,
        max_length,
    ):
        self.forward_calls.append(
            {
                "text": list(text),
                "audios": list(audios),
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
            },
        )
        batch_size = len(text)
        base = torch.arange(1, batch_size * 3 + 1, dtype=torch.long).reshape(batch_size, 3)
        attention_mask = torch.ones_like(base)
        extra = torch.arange(batch_size, dtype=torch.long)
        self.produced_input_ids = base.clone()
        return {"input_ids": base, "attention_mask": attention_mask, "extra": extra}


def test_build_labels_includes_stop_token(collate_mod, monkeypatch):
    """
    Ensure `build_labels` copies the trailing stop token when it matches the configured set.
    """

    class StubTokenizer:
        def __call__(self, text, add_special_tokens, return_tensors):
            assert text == "assistant text"
            assert add_special_tokens is False
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[5, 6]])}

        def decode(self, token):
            if isinstance(token, torch.Tensor):
                token = token.item()
            return "STOP" if token == 7 else str(token)

    class StubProcessor:
        def __init__(self):
            self.tokenizer = StubTokenizer()

    monkeypatch.setattr(collate_mod, "default_stop_tokens", lambda processor: ("STOP",), raising=True)

    input_ids_batch = torch.tensor([[1, 5, 6, 7]])
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "assistant text"}]},
    ]

    labels = collate_mod.build_labels(input_ids_batch, [conversation], StubProcessor())
    assert labels.shape == input_ids_batch.shape
    assert labels.tolist()[0] == [-100, 5, 6, 7]


def test_phi4_mm_collate_fn_handles_audio_and_trimming(collate_mod, monkeypatch):
    processor = DummyPhi4Processor()
    examples = [
        {
            "conversation": CONVERSATION,
            "audio": {"array": [0.1, 0.2], "sampling_rate": 16000},
        },
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "Hola"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Adios"}]},
            ],
            "audio": ([0.3, -0.4], 8000),
        },
    ]

    captured = {}
    labels_stub = torch.tensor([[101, 102, 103], [201, 202, 203]], dtype=torch.long)

    def fake_build_labels(input_ids, conversations, processor_arg):
        captured["input_ids"] = input_ids.clone()
        captured["conversations"] = conversations
        captured["processor"] = processor_arg
        return labels_stub

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    batch = collate_mod.phi4_mm_collate_fn(examples, processor)

    assert len(processor.chat_calls) == len(examples)
    for call, example in zip(processor.chat_calls, examples, strict=True):
        assert call["conversation"] is example["conversation"]

    assert len(processor.forward_calls) == 1
    forward_call = processor.forward_calls[0]
    assert forward_call["return_tensors"] == "pt"
    assert forward_call["padding"] is True
    assert forward_call["truncation"] is True
    assert forward_call["max_length"] == 1024
    assert forward_call["text"] == ["chat::Hi", "chat::Hola"]

    expected_audio0 = (examples[0]["audio"]["array"], examples[0]["audio"]["sampling_rate"])
    assert forward_call["audios"][0] == expected_audio0
    assert forward_call["audios"][1] == examples[1]["audio"]

    assert torch.equal(captured["input_ids"], processor.produced_input_ids)
    assert captured["conversations"] == [example["conversation"] for example in examples]
    assert captured["processor"] is processor

    trimmed_input = processor.produced_input_ids[:, :-1]
    assert torch.equal(batch["input_ids"], trimmed_input)
    assert torch.equal(batch["attention_mask"], torch.ones_like(trimmed_input))
    assert torch.equal(batch["extra"], torch.arange(len(examples), dtype=torch.long))
    assert torch.equal(batch["labels"], labels_stub)
@pytest.fixture()
def collate_mod():
    import nemo_automodel.components.datasets.vlm.collate_fns as _m

    return importlib.reload(_m)


@pytest.fixture()
def fake_qwen_utils(monkeypatch):
    vision_utils = types.ModuleType("qwen_vl_utils")

    def _fake_process_vision_info(conversation):
        return torch.zeros(3, 224, 224), None

    vision_utils.process_vision_info = _fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", vision_utils)

    omni_utils = types.ModuleType("qwen_omni_utils")

    def _fake_process_mm_info(conversation, use_audio_in_video=False):
        return None, [], []

    omni_utils.process_mm_info = _fake_process_mm_info
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", omni_utils)


def test_dispatch_table(collate_mod):
    assert collate_mod.COLLATE_FNS["Qwen2_5_VLProcessor"] is collate_mod.qwen2_5_collate_fn
    assert collate_mod.COLLATE_FNS["default"] is collate_mod.default_collate_fn


def test_qwen25_collate_shapes(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyQwen25Processor()
    batch = collate_mod.qwen2_5_collate_fn([{"conversation": CONVERSATION}], processor)

    assert batch["input_ids"].shape == (1, 4)
    assert batch["labels"].shape == (1, 4)
    assert torch.all(batch["labels"][:, -1] == -100)


def test_default_collate_shapes(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()
    batch = collate_mod.default_collate_fn([{"conversation": CONVERSATION} for _ in range(2)], processor)

    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"].shape == (2, 3)
    assert batch["pixel_values"].dtype == torch.bfloat16


def test_qwen3_omni_collate_shapes(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_OMNI_UTILS", True, raising=True)

    processor = DummyQwen3OmniProcessor()
    batch = collate_mod.qwen3_omni_collate_fn([{"conversation": CONVERSATION} for _ in range(3)], processor)

    assert batch["input_ids"].shape == (3, 4)
    assert batch["labels"].shape == (3, 4)


@pytest.mark.parametrize("fn_name", ["qwen2_5_collate_fn", "default_collate_fn", "qwen3_omni_collate_fn"])
def test_import_error_when_qwen_utils_missing(collate_mod, fn_name, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", False, raising=True)
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_OMNI_UTILS", False, raising=True)
    func = getattr(collate_mod, fn_name)

    with pytest.raises(ImportError):
        func([], None)

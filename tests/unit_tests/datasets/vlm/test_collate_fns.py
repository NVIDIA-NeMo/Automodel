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

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [1, 2, 3]}


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

    assert batch["input_ids"].shape == (1, 5)
    assert batch["labels"].shape == (1, 5)
    assert torch.all(batch["labels"][:, -1] == -100)


def test_default_collate_shapes(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()
    batch = collate_mod.default_collate_fn([{"conversation": CONVERSATION} for _ in range(2)], processor)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    assert batch["pixel_values"].dtype == torch.bfloat16


def test_qwen3_omni_collate_shapes(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_OMNI_UTILS", True, raising=True)

    processor = DummyQwen3OmniProcessor()
    batch = collate_mod.qwen3_omni_collate_fn([{"conversation": CONVERSATION} for _ in range(3)], processor)

    assert batch["input_ids"].shape == (3, 5)
    assert batch["labels"].shape == (3, 5)


@pytest.mark.parametrize("fn_name", ["qwen2_5_collate_fn", "default_collate_fn", "qwen3_omni_collate_fn"])
def test_import_error_when_qwen_utils_missing(collate_mod, fn_name, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", False, raising=True)
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_OMNI_UTILS", False, raising=True)
    func = getattr(collate_mod, fn_name)

    with pytest.raises(ImportError):
        func([], None)

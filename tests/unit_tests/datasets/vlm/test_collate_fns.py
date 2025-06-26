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
import types
from typing import List
import sys
import pytest
import torch

SKIP_TOKEN = 99  # token that must be masked with -100 in labels


class DummyQwen25Processor:
    """
    Mimics the public API used by qwen2_5_collate_fn:
      • apply_chat_template(tokenize=bool)
      • __call__(text, images, padding, return_tensors="pt")
    """

    def __init__(self):
        self.call_counter = 0  # handy for assertions if you like

    # Called with tokenize=False (single example) in qwen2_5_collate_fn
    def apply_chat_template(self, conversation, *, tokenize=False, **kwargs):
        if tokenize:
            raise RuntimeError("This fake is only used with tokenize=False in qwen2_5_collate_fn")
        return "dummy chat string"

    # Called by processor(...) in qwen2_5_collate_fn
    def __call__(self, *, text: List[str], images: List[torch.Tensor], padding: bool, return_tensors: str):
        self.call_counter += 1
        bs = len(text)
        # Produce a deterministic fake sequence that includes the SKIP_TOKEN so
        # we can validate the masking logic.
        seq = torch.tensor([0, 1, SKIP_TOKEN, 2, 3])
        input_ids = seq.unsqueeze(0).repeat(bs, 1)

        return {
            "input_ids": input_ids,
            "pixel_values": torch.zeros(bs, 3, 224, 224, dtype=torch.float32),
        }


class DummyDefaultProcessor:
    """
    Mimics the public API used by default_collate_fn:
      • apply_chat_template(tokenize=True, return_tensors="pt", return_dict=True)
    """

    def apply_chat_template(
        self,
        conv_list,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
    ):
        assert tokenize and return_tensors == "pt" and return_dict
        bs = len(conv_list)
        seq = torch.tensor([5, 6, SKIP_TOKEN, 7])
        input_ids = seq.unsqueeze(0).repeat(bs, 1)
        pixel_values = torch.ones(bs, 3, 64, 64, dtype=torch.float32)

        return {"input_ids": input_ids, "pixel_values": pixel_values}


@pytest.fixture()
def collate_mod():
    """
    Import the module under test fresh for every test so monkey-patching of
    module-level variables does not leak between tests.
    """
    import nemo_automodel.datasets.vlm.collate_fns as _m

    # Always reload so each test starts from a clean module object.
    return importlib.reload(_m)


@pytest.fixture()
def patch_skipped(monkeypatch):
    """
    Patch extract_skipped_token_ids to return our fixed SKIP_TOKEN.
    """
    def _fake_skip_fn(processor):
        return torch.tensor([SKIP_TOKEN])

    monkeypatch.setattr(
        "nemo_automodel.datasets.vlm.collate_fns.extract_skipped_token_ids",
        _fake_skip_fn,
        raising=True,
    )


@pytest.fixture()
def fake_qwen_utils(monkeypatch):
    """
    Provide a fake qwen_vl_utils.process_vision_info so the import inside the
    collate module succeeds.
    """
    fake_utils = types.ModuleType("qwen_vl_utils")

    def _fake_process_vision_info(conversation):
        # Return tuple expected by qwen2_5_collate_fn; only first element is used.
        return torch.zeros(3, 224, 224), None

    fake_utils.process_vision_info = _fake_process_vision_info
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_utils)

def test_dispatch_table(collate_mod):
    assert collate_mod.COLLATE_FNS["Qwen2_5_VLProcessor"] is collate_mod.qwen2_5_collate_fn
    assert collate_mod.COLLATE_FNS["default"] is collate_mod.default_collate_fn


def _fake_process_vision_info(conv):
    # qwen2_5_collate_fn only uses the first return value
    return (torch.zeros(3, 224, 224),)          # 1-tuple

def test_qwen25_collate_happy_path(
    collate_mod, patch_skipped, monkeypatch
):
    # Patch the *imported symbol* inside collate_mod, not the module.
    monkeypatch.setattr(collate_mod, "process_vision_info",
                        _fake_process_vision_info, raising=True)

    # Ensure code path that requires the utils is enabled
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyQwen25Processor()
    examples = [{"conversation": "a"}, {"conversation": "b"}]

    batch = collate_mod.qwen2_5_collate_fn(examples, processor)

    assert batch["input_ids"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)

    # last column is -100
    assert torch.all(batch["labels"][:, -1] == -100)

    # the SKIP_TOKEN (99) should be masked out in labels
    # Original seq: [0, 1, 99, 2, 3]
    # Shifted   -> [1, 99, 2, 3, -100]  then 99 → -100
    expected = torch.tensor([1, -100, 2, 3, -100])
    assert torch.equal(batch["labels"][0], expected)


def test_default_collate_happy_path(collate_mod, patch_skipped, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()
    examples = [{"conversation": "hello"}, {"conversation": "world"}]

    batch = collate_mod.default_collate_fn(examples, processor)

    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    # pixel_values should have been cast to bfloat16
    assert batch["pixel_values"].dtype == torch.bfloat16
    # last column -100
    assert torch.all(batch["labels"][:, -1] == -100)
    # SKIP_TOKEN location masked
    # seq = [5,6,SKIP_TOKEN,7]  -> after shift SKIP_TOKEN is at col 1
    assert torch.all(batch["labels"][:, 1] == -100)


@pytest.mark.parametrize("fn_name", ["qwen2_5_collate_fn", "default_collate_fn"])
def test_import_error_when_qwen_utils_missing(collate_mod, fn_name, monkeypatch):
    # Simulate missing qwen_vl_utils
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", False, raising=True)
    func = getattr(collate_mod, fn_name)

    with pytest.raises(ImportError):
        func([], None)

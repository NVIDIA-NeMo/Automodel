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
import importlib
import sys
import types

import pytest
import torch

CONVERSATION = [
    {"role": "user", "content": [{"type": "text", "text": "Hi"}, {"type": "image", "image": "dummy.png"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
]


class _DummyTokenizer:
    pad_token_id = 0


class _DummyProcessor:
    """Mirrors ``DummyDefaultProcessor`` in test_collate_fns.py: returns fixed
    ``input_ids``/``pixel_values``, recording the ``apply_chat_template`` kwargs
    it was called with."""

    def __init__(self):
        self.tokenizer = _DummyTokenizer()
        self.captured_kwargs = {}

    def apply_chat_template(self, conv_list, **kwargs):
        self.captured_kwargs.update(kwargs)
        batch_size = len(conv_list)
        input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)  # length 4
        pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
        return {"input_ids": input_ids, "pixel_values": pixel_values}


@pytest.fixture()
def collate_mod():
    import nemo_automodel.components.datasets.vlm.dspark_collate as _m

    return importlib.reload(_m)


@pytest.fixture()
def fake_qwen_utils(monkeypatch):
    vision_utils = types.ModuleType("qwen_vl_utils")
    vision_utils.process_vision_info = lambda conversation, **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", vision_utils)


def test_dspark_vlm_collate_fn_unshifted_full_length(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    # input_ids is length 4; mark position 0 (prompt) and position 3 (image
    # placeholder tail) as unsupervised, positions 1-2 as the assistant response.
    labels_stub = torch.tensor([[-100, 11, 12, -100]], dtype=torch.long)

    def fake_build_labels(input_ids_batch, conversations, processor_arg):
        assert input_ids_batch.shape == (1, 4)
        return labels_stub

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    processor = _DummyProcessor()
    batch = collate_mod.dspark_vlm_collate_fn([{"conversation": CONVERSATION}], processor, max_length=4)

    # Full length T=4, NOT T-1=3: this is the regression the whole module exists
    # to prevent (default_collate_fn would have shifted/truncated to length 3).
    assert batch["input_ids"].shape == (1, 4)
    assert batch["loss_mask"].shape == (1, 4)
    assert torch.equal(batch["loss_mask"], torch.tensor([[0, 1, 1, 0]], dtype=torch.long))
    assert "labels" not in batch
    assert batch["pixel_values"].dtype == torch.bfloat16
    assert batch["pixel_values"].shape == (1, 3, 64, 64)


def test_dspark_vlm_collate_fn_one_token_longer_than_default_collate_fn(monkeypatch, fake_qwen_utils):
    import nemo_automodel.components.datasets.vlm.collate_fns as collate_fns_mod
    import nemo_automodel.components.datasets.vlm.dspark_collate as dspark_mod

    monkeypatch.setattr(collate_fns_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)
    monkeypatch.setattr(dspark_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    labels_stub = torch.tensor([[-100, 11, 12, 13]], dtype=torch.long)
    monkeypatch.setattr(collate_fns_mod, "build_labels_from_template", lambda *a, **k: labels_stub, raising=True)
    monkeypatch.setattr(dspark_mod, "build_labels_from_template", lambda *a, **k: labels_stub, raising=True)

    default_out = collate_fns_mod.default_collate_fn([{"conversation": CONVERSATION}], _DummyProcessor())
    dspark_out = dspark_mod.dspark_vlm_collate_fn([{"conversation": CONVERSATION}], _DummyProcessor(), max_length=4)

    # default_collate_fn shifts labels[1:] and truncates input_ids[:-1] for the
    # standard next-token-prediction loss; DSpark must NOT inherit that shift.
    assert dspark_out["input_ids"].shape[1] == default_out["input_ids"].shape[1] + 1


def test_dspark_vlm_collate_fn_forces_max_length_padding(collate_mod, fake_qwen_utils, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)
    monkeypatch.setattr(
        collate_mod, "build_labels_from_template", lambda *a, **k: torch.zeros(1, 4, dtype=torch.long), raising=True
    )

    processor = _DummyProcessor()
    collate_mod.dspark_vlm_collate_fn([{"conversation": CONVERSATION}], processor, max_length=512)

    proc_kwargs = processor.captured_kwargs.get("processor_kwargs", {})
    assert proc_kwargs.get("max_length") == 512
    assert proc_kwargs.get("padding") == "max_length"
    assert proc_kwargs.get("truncation") is True


def test_dspark_vlm_collate_fn_raises_without_qwen_vl_utils(collate_mod, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", False, raising=True)
    with pytest.raises(ImportError):
        collate_mod.dspark_vlm_collate_fn([], None, max_length=4)

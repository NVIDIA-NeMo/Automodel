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
from PIL import Image as PILImage

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

    def convert_tokens_to_ids(self, token):
        return None  # Return None to trigger default fallback

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

    def __call__(self, *, text, images=None, videos=None, padding, return_tensors, **kwargs):
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
        # Find first text content (may be preceded by an injected fake image).
        for item in conversation[0]["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                return "chat:" + item["text"]
        return "chat:default"

    def __call__(self, *, text, return_tensors, padding, **kwargs):
        assert return_tensors == "pt"
        assert padding is True
        self.call_kwargs.append(dict(kwargs))
        batch_size = len(text)
        input_ids = torch.arange(1, 6).unsqueeze(0).repeat(batch_size, 1)
        return {"input_ids": input_ids}


class _DummyPhi4Tokenizer(DummyTokenizer):
    """Tokenizer with apply_chat_template for Phi4 tests."""

    def apply_chat_template(self, conversation, *, tokenize, **kwargs):
        assert tokenize is False
        self._chat_calls = getattr(self, "_chat_calls", [])
        self._chat_calls.append({"conversation": conversation, "kwargs": kwargs})
        return "chat::" + conversation[0]["content"][0]["text"]


class DummyPhi4Processor:
    def __init__(self):
        self.tokenizer = _DummyPhi4Tokenizer(pad_token_id=0)
        self.forward_calls = []
        self.produced_input_ids = None

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


class DummyNemotronParseProcessor:
    def __init__(self):
        self.tokenizer = types.SimpleNamespace(
            pad_token_id=0,
            decoder_start_token_id=5,
            bos_token_id=6,
            eos_token_id=7,
        )

    def __call__(self, *, images, text, padding, return_tensors):
        assert padding is True and return_tensors == "pt"
        batch_size = len(text)
        input_ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long).repeat(batch_size, 1)
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.ones(batch_size, 3, 2, 2, dtype=torch.float32)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}


class DummyKimiVLProcessor:
    """Dummy processor for KimiVL collate function tests."""

    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)
        self.chat_calls = []
        self.forward_calls = []

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt is False
        assert tokenize is False
        self.chat_calls.append({"conversation": conversation, "kwargs": kwargs})
        # Extract first text content from conversation
        for item in conversation[0]["content"]:
            if isinstance(item, dict) and item.get("type") == "text":
                return "chat:" + item["text"]
        return "chat:default"

    def __call__(self, *, text, return_tensors, padding, truncation, **kwargs):
        assert return_tensors == "pt"
        assert padding is True or padding == "max_length"
        self.forward_calls.append(
            {
                "text": list(text),
                "return_tensors": return_tensors,
                "padding": padding,
                "truncation": truncation,
                **kwargs,
            }
        )
        batch_size = len(text)
        input_ids = torch.arange(1, 6).unsqueeze(0).repeat(batch_size, 1)
        return {"input_ids": input_ids}


def test_build_labels_retries_with_stripped_whitespace(collate_mod, monkeypatch):
    """When a tokenizer produces different tokens for leading-whitespace text,
    build_labels should retry with lstripped text and still find the answer."""

    class WhitespaceTokenizer:
        """Tokenizer that produces different tokens for ' Hello' vs 'Hello'."""

        def __call__(self, text, add_special_tokens, return_tensors):
            assert add_special_tokens is False
            assert return_tensors == "pt"
            if text == " Hello":
                return {"input_ids": torch.tensor([[90, 91]])}
            if text == "Hello":
                return {"input_ids": torch.tensor([[10, 11]])}
            return {"input_ids": torch.tensor([[99]])}

        def decode(self, token):
            return ""

    class StubProcessor:
        def __init__(self):
            self.tokenizer = WhitespaceTokenizer()

    monkeypatch.setattr(collate_mod, "default_stop_tokens", lambda processor: (), raising=True)

    # Encoded sequence contains stripped tokens [10, 11] but NOT whitespace tokens [90, 91]
    input_ids_batch = torch.tensor([[1, 2, 10, 11, 3]])
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": " Hello"}]},
    ]

    labels = collate_mod.build_labels(input_ids_batch, [conversation], StubProcessor())
    assert labels.shape == input_ids_batch.shape
    # Tokens at positions 2,3 (the answer) should be unmasked; rest stays -100
    assert labels.tolist()[0] == [-100, -100, 10, 11, -100]


def test_build_labels_no_retry_when_no_leading_whitespace(collate_mod, monkeypatch):
    """When assistant text has no leading whitespace and tokens are not found,
    build_labels should NOT retry and should warn (answer_start stays -1)."""

    call_count = [0]

    class NoRetryTokenizer:
        def __call__(self, text, add_special_tokens, return_tensors):
            call_count[0] += 1
            return {"input_ids": torch.tensor([[90, 91]])}

        def decode(self, token):
            return ""

    class StubProcessor:
        def __init__(self):
            self.tokenizer = NoRetryTokenizer()

    monkeypatch.setattr(collate_mod, "default_stop_tokens", lambda processor: (), raising=True)

    input_ids_batch = torch.tensor([[1, 2, 3, 4, 5]])
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]

    labels = collate_mod.build_labels(input_ids_batch, [conversation], StubProcessor())
    # No match found, all labels stay -100
    assert labels.tolist()[0] == [-100, -100, -100, -100, -100]
    # Tokenizer called only once (no retry since text has no leading whitespace)
    assert call_count[0] == 1


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
            if isinstance(token, list):
                token = token[0]
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

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    batch = collate_mod.phi4_mm_collate_fn(examples, processor)

    # Chat template is called on the tokenizer, not the processor
    assert len(processor.tokenizer._chat_calls) == len(examples)
    for call, example in zip(processor.tokenizer._chat_calls, examples, strict=True):
        assert call["conversation"] is example["conversation"]

    assert len(processor.forward_calls) == 1
    forward_call = processor.forward_calls[0]
    assert forward_call["return_tensors"] == "pt"
    assert forward_call["padding"] is True
    assert forward_call["truncation"] is True
    assert forward_call["max_length"] == 1024
    assert forward_call["text"] == ["chat::Hi", "chat::Hola"]

    # Audio inputs are converted to (array, sampling_rate) tuples
    expected_audio0 = (examples[0]["audio"]["array"], examples[0]["audio"]["sampling_rate"])
    assert forward_call["audios"][0] == expected_audio0
    assert forward_call["audios"][1] == tuple(examples[1]["audio"])

    assert torch.equal(captured["input_ids"], processor.produced_input_ids)
    assert captured["conversations"] == [example["conversation"] for example in examples]
    assert captured["processor"] is processor

    trimmed_input = processor.produced_input_ids[:, :-1]
    assert torch.equal(batch["input_ids"], trimmed_input)
    assert torch.equal(batch["attention_mask"], torch.ones_like(trimmed_input))
    assert torch.equal(batch["extra"], torch.arange(len(examples), dtype=torch.long))
    # Labels are shifted by [:, 1:] — not overwritten with full labels
    assert torch.equal(batch["labels"], labels_stub[:, 1:])


def test_phi4_mm_collate_fn_input_mode_from_processor(collate_mod, monkeypatch):
    """When the processor already sets input_mode, the collate fn should not override it."""

    class Phi4ProcessorWithInputMode:
        def __init__(self):
            self.tokenizer = _DummyPhi4Tokenizer(pad_token_id=0)

        def __call__(self, *, text, audios, return_tensors, padding, truncation, max_length):
            bs = len(text)
            ids = torch.arange(1, bs * 3 + 1, dtype=torch.long).reshape(bs, 3)
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "input_mode": torch.tensor([2])}

    examples = [{"conversation": CONVERSATION, "audio": {"array": [0.1], "sampling_rate": 16000}}]
    monkeypatch.setattr(
        collate_mod,
        "build_labels_from_template",
        lambda *a, **kw: torch.tensor([[1, 2, 3]], dtype=torch.long),
        raising=True,
    )
    batch = collate_mod.phi4_mm_collate_fn(examples, Phi4ProcessorWithInputMode())
    assert torch.equal(batch["input_mode"], torch.tensor([2]))


def test_phi4_mm_collate_fn_input_mode_fallback(collate_mod, monkeypatch):
    """When processor doesn't set input_mode, collate fn computes it from batch keys."""

    class Phi4ProcessorWithAudioEmbeds:
        def __init__(self):
            self.tokenizer = _DummyPhi4Tokenizer(pad_token_id=0)

        def __call__(self, *, text, audios, return_tensors, padding, truncation, max_length):
            bs = len(text)
            ids = torch.arange(1, bs * 3 + 1, dtype=torch.long).reshape(bs, 3)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "input_audio_embeds": torch.randn(bs, 4, 80),
            }

    examples = [{"conversation": CONVERSATION, "audio": {"array": [0.1], "sampling_rate": 16000}}]
    monkeypatch.setattr(
        collate_mod,
        "build_labels_from_template",
        lambda *a, **kw: torch.tensor([[1, 2, 3]], dtype=torch.long),
        raising=True,
    )
    batch = collate_mod.phi4_mm_collate_fn(examples, Phi4ProcessorWithAudioEmbeds())
    assert batch["input_mode"] == 2  # SPEECH


def test_phi4_mm_collate_fn_raw_audio_passthrough(collate_mod, monkeypatch):
    """Audio that is neither a dict nor a tuple/list should pass through as-is."""
    import numpy as np

    processor = DummyPhi4Processor()
    raw_array = np.array([0.5, -0.5])
    examples = [
        {"conversation": CONVERSATION, "audio": raw_array},
    ]
    monkeypatch.setattr(
        collate_mod,
        "build_labels_from_template",
        lambda *a, **kw: torch.tensor([[1, 2, 3]], dtype=torch.long),
        raising=True,
    )
    collate_mod.phi4_mm_collate_fn(examples, processor)
    # The raw array should be wrapped as a single-element tuple by the collate fn
    forward_call = processor.forward_calls[0]
    assert forward_call["audios"][0] is raw_array


@pytest.fixture()
def collate_mod():
    import nemo_automodel.components.datasets.vlm.collate_fns as _m

    return importlib.reload(_m)


@pytest.fixture()
def fake_qwen_utils(monkeypatch):
    vision_utils = types.ModuleType("qwen_vl_utils")

    def _fake_process_vision_info(conversation, **kwargs):
        return None, None

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


def test_nemotron_omni_dispatch_registered(collate_mod):
    """The v3 NemotronOmni processor key must dispatch to its collate fn."""
    assert collate_mod.COLLATE_FNS["NemotronH_Nano_Omni_Reasoning_V3Processor"] is collate_mod.nemotron_omni_collate_fn


class DummyNemotronOmniProcessor:
    """Minimal stub that mirrors what ``nemotron_omni_collate_fn`` consumes.

    Returns deterministic per-sample input_ids of varying length so the collate
    has something interesting to right-pad.
    """

    image_token = "<image>"
    video_token = "<video>"

    def __init__(self):
        self.tokenizer = self  # collate uses getattr(processor, "tokenizer", processor)
        self.pad_token_id = 0
        self.sample_lens = [3, 5]
        self._call_idx = 0

    def apply_chat_template(self, text_conversation, *, tokenize=False, **kwargs):
        assert tokenize is False
        return "chat:" + str(len(text_conversation))

    def __call__(self, *, text, return_tensors, **kwargs):
        assert return_tensors == "pt"
        n = self.sample_lens[self._call_idx]
        self._call_idx += 1
        ids = torch.arange(1, n + 1, dtype=torch.long).unsqueeze(0)
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }


def test_nemotron_omni_collate_text_only_pads_and_shifts(collate_mod, monkeypatch):
    """Two text-only samples of differing length right-pad to the longer one,
    then the per-token shift drops the trailing position so labels are aligned."""
    processor = DummyNemotronOmniProcessor()

    # build_labels is exercised in its own tests; stub it to return a fixed tensor.
    def fake_build_labels(input_ids, conversations, processor_arg):
        assert processor_arg is processor
        return torch.full(input_ids.shape, -100, dtype=torch.long)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    examples = [
        {"conversation": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]},
        {"conversation": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]},
    ]
    batch = collate_mod.nemotron_omni_collate_fn(examples, processor=processor)

    # Sample lens are 3 and 5; max is 5; trailing-shift drops one position -> 4.
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2, 4)
    # Right-padding of the shorter sample uses pad_token_id=0 in the tail.
    assert batch["input_ids"][0, -1].item() == 0
    assert batch["attention_mask"][0, -1].item() == 0
    # No multimodal kwargs were produced for an all-text batch.
    for k in ("pixel_values", "pixel_values_videos", "sound_features", "image_flags"):
        assert k not in batch


class DummyNemotronOmniImageProcessor(DummyNemotronOmniProcessor):
    """Variant that records call kwargs and emits ``pixel_values`` when images are passed."""

    def __init__(self):
        super().__init__()
        self.seen_kwargs: list = []

    def __call__(self, *, text, return_tensors, **kwargs):
        assert return_tensors == "pt"
        self.seen_kwargs.append(kwargs)
        ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if "images" in kwargs:
            out["pixel_values"] = torch.zeros(len(kwargs["images"]), 3, 4, 4, dtype=torch.float32)
        return out


def test_nemotron_omni_collate_extracts_images(collate_mod, monkeypatch):
    """List-content with ``type=='image'`` items should be collected into pixel_values
    and an ``<image>`` token spliced into the text content."""
    processor = DummyNemotronOmniImageProcessor()

    monkeypatch.setattr(
        collate_mod,
        "build_labels",
        lambda ids, conv, p: torch.zeros_like(ids),
        raising=True,
    )

    from PIL import Image as PILImage

    img = PILImage.new("RGB", (4, 4))
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "what is this?"},
                    ],
                },
                {"role": "assistant", "content": "an image"},
            ]
        }
    ]
    batch = collate_mod.nemotron_omni_collate_fn(examples, processor=processor)

    # The image was extracted and forwarded to the processor, then surfaced in the batch.
    assert processor.seen_kwargs and processor.seen_kwargs[0].get("images") == [img]
    assert "pixel_values" in batch
    assert batch["pixel_values"].dtype == torch.bfloat16  # collate casts to bf16
    assert "image_flags" in batch and batch["image_flags"].shape == (1, 1)


def test_qwen25_collate_shapes(collate_mod, monkeypatch):
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


def test_nemotron_parse_collate_shifts_and_casts(collate_mod, monkeypatch):
    processor = DummyNemotronParseProcessor()

    # Return deterministic labels to bypass tokenizer-heavy logic.
    labels_stub = torch.tensor([[20, 21, 22, 23]], dtype=torch.long)

    def fake_build_labels(input_ids, conversations, processor_arg):
        assert processor_arg is processor
        assert input_ids.shape == (1, 4)
        return labels_stub

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "please parse"},
                        {"type": "image", "image": "dummy.png"},
                    ],
                },
                {"role": "assistant", "content": "ok"},
            ]
        }
    ]

    batch = collate_mod.nemotron_parse_collate_fn(
        examples,
        processor=processor,
        task_prompt="</s><s><predict_bbox>",
    )

    assert batch["pixel_values"].dtype == torch.bfloat16
    assert torch.equal(batch["input_ids"], torch.tensor([[10, 11, 12]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 1, 1]]))
    assert torch.equal(batch["labels"], torch.tensor([[21, 22, 23]]))
    assert torch.equal(batch["decoder_input_ids"], torch.tensor([[10, 11, 12]]))
    assert torch.equal(batch["decoder_attention_mask"], torch.tensor([[1, 1, 1]]))


@pytest.mark.parametrize("fn_name", ["default_collate_fn", "qwen3_omni_collate_fn"])
def test_import_error_when_qwen_utils_missing(collate_mod, fn_name, monkeypatch):
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", False, raising=True)
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_OMNI_UTILS", False, raising=True)
    func = getattr(collate_mod, fn_name)

    with pytest.raises(ImportError):
        func([], None)


def test_default_collate_fn_with_max_length(collate_mod, fake_qwen_utils, monkeypatch):
    """Test that default_collate_fn passes max_length and sets padding to 'max_length'."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    captured_kwargs = {}

    class MaxLengthProcessor:
        tokenizer = DummyTokenizer()

        def apply_chat_template(self, conv_list, **kwargs):
            captured_kwargs.update(kwargs)
            batch_size = len(conv_list)
            input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
            pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
            return {"input_ids": input_ids, "pixel_values": pixel_values}

    processor = MaxLengthProcessor()
    collate_mod.default_collate_fn([{"conversation": CONVERSATION}], processor, max_length=512)

    assert captured_kwargs.get("max_length") == 512
    assert captured_kwargs.get("padding") == "max_length"
    assert captured_kwargs.get("truncation") is True


def test_default_collate_fn_without_max_length(collate_mod, fake_qwen_utils, monkeypatch):
    """Test that default_collate_fn uses padding=True when max_length is not provided."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    captured_kwargs = {}

    class NoMaxLengthProcessor:
        tokenizer = DummyTokenizer()

        def apply_chat_template(self, conv_list, **kwargs):
            captured_kwargs.update(kwargs)
            batch_size = len(conv_list)
            input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
            pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
            return {"input_ids": input_ids, "pixel_values": pixel_values}

    processor = NoMaxLengthProcessor()
    collate_mod.default_collate_fn([{"conversation": CONVERSATION}], processor)

    assert "max_length" not in captured_kwargs
    assert captured_kwargs.get("padding") is True


def test_kimi_vl_collate_fn_registered(collate_mod):
    """Test that kimi_vl_collate_fn is registered in COLLATE_FNS."""
    assert "KimiVLProcessor" in collate_mod.COLLATE_FNS
    assert collate_mod.COLLATE_FNS["KimiVLProcessor"] is collate_mod.kimi_vl_collate_fn


def test_kimi_vl_collate_fn_shapes(collate_mod, monkeypatch):
    """Test kimi_vl_collate_fn produces correct output shapes."""
    processor = DummyKimiVLProcessor()

    # Stub build_labels_from_template to return deterministic labels
    # The collate fn does labels[:, 1:] so we need 5 elements to get 4 after shift
    labels_stub = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    def fake_build_labels(input_ids, conversations, processor_arg):
        assert processor_arg is processor
        return labels_stub

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    examples = [{"conversation": CONVERSATION}]
    batch = collate_mod.kimi_vl_collate_fn(examples, processor)

    # Input starts at [1, 5], trimmed by [:, :-1] to [1, 4]
    assert batch["input_ids"].shape == (1, 4)
    # Labels start at [1, 5], shifted by [:, 1:] to [1, 4]
    assert batch["labels"].shape == (1, 4)


def test_kimi_vl_collate_fn_with_max_length(collate_mod, monkeypatch):
    """Test kimi_vl_collate_fn passes max_length correctly."""
    processor = DummyKimiVLProcessor()

    labels_stub = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    monkeypatch.setattr(collate_mod, "build_labels_from_template", lambda *args, **kwargs: labels_stub, raising=True)

    examples = [{"conversation": CONVERSATION}]
    collate_mod.kimi_vl_collate_fn(examples, processor, max_length=2048)

    assert len(processor.forward_calls) == 1
    forward_call = processor.forward_calls[0]
    assert forward_call["max_length"] == 2048
    assert forward_call["padding"] == "max_length"


def test_kimi_vl_collate_fn_extracts_images(collate_mod, monkeypatch):
    """Test kimi_vl_collate_fn extracts images from conversation content."""
    processor = DummyKimiVLProcessor()

    labels_stub = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    monkeypatch.setattr(collate_mod, "build_labels_from_template", lambda *args, **kwargs: labels_stub, raising=True)

    conversation_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "test_image.jpg"},
                {"type": "text", "text": "What is this?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "A test image"}]},
    ]

    examples = [{"conversation": conversation_with_image}]
    collate_mod.kimi_vl_collate_fn(examples, processor)

    assert len(processor.forward_calls) == 1
    forward_call = processor.forward_calls[0]
    assert "images" in forward_call
    assert forward_call["images"] == ["test_image.jpg"]


def test_kimi_vl_collate_fn_passes_add_special_tokens_false(collate_mod, monkeypatch):
    """Test that kimi_vl_collate_fn passes add_special_tokens=False to processor."""
    processor = DummyKimiVLProcessor()

    labels_stub = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)
    monkeypatch.setattr(collate_mod, "build_labels_from_template", lambda *args, **kwargs: labels_stub, raising=True)

    examples = [{"conversation": CONVERSATION}]
    collate_mod.kimi_vl_collate_fn(examples, processor)

    assert len(processor.forward_calls) == 1
    forward_call = processor.forward_calls[0]
    assert "add_special_tokens" in forward_call
    assert forward_call["add_special_tokens"] is False


def test_kimi_vl_collate_fn_multiple_examples(collate_mod, monkeypatch):
    """Test kimi_vl_collate_fn handles multiple examples."""
    processor = DummyKimiVLProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size = input_ids.shape[0]
        return torch.arange(1, 6).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    examples = [{"conversation": CONVERSATION} for _ in range(3)]
    batch = collate_mod.kimi_vl_collate_fn(examples, processor)

    assert batch["input_ids"].shape[0] == 3
    assert batch["labels"].shape[0] == 3
    assert len(processor.chat_calls) == 3


# =============================================================================
# Tests for _decode_single_token
# =============================================================================


class TestDecodeSingleToken:
    """Tests for _decode_single_token helper function."""

    def test_decode_single_token_with_int(self, collate_mod):
        """Test _decode_single_token with tokenizer accepting int."""

        class IntTokenizer:
            def decode(self, token_id):
                return f"token_{token_id}"

        result = collate_mod._decode_single_token(IntTokenizer(), 42)
        assert result == "token_42"

    def test_decode_single_token_with_list(self, collate_mod):
        """Test _decode_single_token with tokenizer requiring list."""

        class ListTokenizer:
            def decode(self, token_ids):
                if isinstance(token_ids, int):
                    raise TypeError("Expected list")
                return f"token_{token_ids[0]}"

        result = collate_mod._decode_single_token(ListTokenizer(), 42)
        assert result == "token_42"

    def test_decode_single_token_with_tensor(self, collate_mod):
        """Test _decode_single_token with tokenizer requiring tensor."""

        class TensorTokenizer:
            def decode(self, token_ids):
                if isinstance(token_ids, int):
                    raise TypeError("Expected tensor")
                if isinstance(token_ids, list):
                    raise TypeError("Expected tensor")
                # Expects torch.Tensor
                return f"token_{token_ids[0].item()}"

        result = collate_mod._decode_single_token(TensorTokenizer(), 42)
        assert result == "token_42"

    def test_decode_single_token_fallback(self, collate_mod):
        """Test _decode_single_token falls back to str when all methods fail."""

        class FailingTokenizer:
            def decode(self, token_ids):
                raise RuntimeError("Cannot decode")

        result = collate_mod._decode_single_token(FailingTokenizer(), 42)
        assert result == "42"


# =============================================================================
# Tests for _expand_image_tokens
# =============================================================================


class TestExpandImageTokens:
    """Tests for _expand_image_tokens function."""

    def test_expand_image_tokens_basic(self, collate_mod):
        """Test basic expansion of image placeholder tokens."""
        # Input with 1 placeholder at position 2
        media_token_id = 163605
        input_ids = torch.tensor([1, 2, media_token_id, 3, 4])
        attention_mask = torch.ones(5, dtype=torch.long)

        # grid_thws: [1, 28, 28] -> (28//2) * (28//2) = 196 tokens
        grid_thws = torch.tensor([[1, 28, 28]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # Original: 5 tokens, placeholder expanded to 196, so 5 - 1 + 196 = 200
        assert expanded_ids.shape[0] == 200
        assert expanded_mask.shape[0] == 200

        # Check structure: [1, 2, media_token_id*196, 3, 4]
        assert expanded_ids[0] == 1
        assert expanded_ids[1] == 2
        assert (expanded_ids[2:198] == media_token_id).all()
        assert expanded_ids[198] == 3
        assert expanded_ids[199] == 4

    def test_expand_image_tokens_smaller_grid(self, collate_mod):
        """Test expansion with smaller grid."""
        media_token_id = 163605
        input_ids = torch.tensor([1, media_token_id, 2])
        attention_mask = torch.ones(3, dtype=torch.long)

        # grid_thws: [1, 4, 4] -> (4//2) * (4//2) = 4 tokens
        grid_thws = torch.tensor([[1, 4, 4]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # Original: 3 tokens, placeholder expanded to 4, so 3 - 1 + 4 = 6
        assert expanded_ids.shape[0] == 6
        assert expanded_mask.shape[0] == 6

        assert expanded_ids[0] == 1
        assert (expanded_ids[1:5] == media_token_id).all()
        assert expanded_ids[5] == 2

    def test_expand_image_tokens_no_placeholder(self, collate_mod):
        """Test expansion when no placeholder exists."""
        media_token_id = 163605
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        attention_mask = torch.ones(5, dtype=torch.long)
        grid_thws = torch.tensor([[1, 28, 28]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # No expansion should occur
        assert torch.equal(expanded_ids, input_ids)
        assert torch.equal(expanded_mask, attention_mask)

    def test_expand_image_tokens_attention_mask_values(self, collate_mod):
        """Test that expanded attention mask has correct values."""
        media_token_id = 163605
        input_ids = torch.tensor([1, media_token_id, 2])
        attention_mask = torch.tensor([1, 1, 0], dtype=torch.long)  # Last token is padding

        grid_thws = torch.tensor([[1, 4, 4]])  # 4 tokens

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # [1, 1111, 0] -> [1] + [1,1,1,1] + [0] = [1, 1, 1, 1, 1, 0]
        assert expanded_mask[0] == 1
        assert (expanded_mask[1:5] == 1).all()  # Image tokens should have attention
        assert expanded_mask[5] == 0

    def test_expand_image_tokens_custom_merge_kernel(self, collate_mod):
        """Test expansion with custom merge kernel size."""
        media_token_id = 163605
        input_ids = torch.tensor([1, media_token_id, 2])
        attention_mask = torch.ones(3, dtype=torch.long)

        # grid_thws: [1, 8, 8] with merge (4, 4) -> (8//4) * (8//4) = 4 tokens
        grid_thws = torch.tensor([[1, 8, 8]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id, merge_kernel_size=(4, 4)
        )

        # Original: 3 tokens, placeholder expanded to 4, so 3 - 1 + 4 = 6
        assert expanded_ids.shape[0] == 6

    def test_expand_image_tokens_preserves_dtype(self, collate_mod):
        """Test that expansion preserves input tensor dtypes."""
        media_token_id = 163605
        input_ids = torch.tensor([1, media_token_id, 2], dtype=torch.int32)
        attention_mask = torch.tensor([1, 1, 1], dtype=torch.int64)
        grid_thws = torch.tensor([[1, 4, 4]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        assert expanded_ids.dtype == torch.int32
        assert expanded_mask.dtype == torch.int64

    def test_expand_image_tokens_multi_image_different_sizes(self, collate_mod):
        """Multi-image: two placeholders with different grid sizes."""
        media_token_id = 163605
        # [BOS, PH1, TEXT, PH2, EOS]
        input_ids = torch.tensor([1, media_token_id, 99, media_token_id, 2])
        attention_mask = torch.ones(5, dtype=torch.long)

        # First image: [1,2,2] -> (2//2)*(2//2) = 1 token
        # Second image: [1,4,4] -> (4//2)*(4//2) = 4 tokens
        grid_thws = torch.tensor([[1, 2, 2], [1, 4, 4]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # 5 - 2 placeholders + 1 + 4 = 8 tokens
        assert expanded_ids.shape[0] == 8
        assert expanded_mask.shape[0] == 8

        # [1, media*1, 99, media*4, 2]
        assert expanded_ids[0] == 1
        assert expanded_ids[1] == media_token_id
        assert expanded_ids[2] == 99
        assert (expanded_ids[3:7] == media_token_id).all()
        assert expanded_ids[7] == 2

    def test_expand_image_tokens_multi_image_same_size(self, collate_mod):
        """Multi-image: two placeholders with identical grid sizes."""
        media_token_id = 163605
        input_ids = torch.tensor([media_token_id, 50, media_token_id])
        attention_mask = torch.ones(3, dtype=torch.long)

        # Both images: [1,4,4] -> 4 tokens each
        grid_thws = torch.tensor([[1, 4, 4], [1, 4, 4]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # 3 - 2 + 4 + 4 = 9 tokens
        assert expanded_ids.shape[0] == 9
        assert (expanded_ids[0:4] == media_token_id).all()
        assert expanded_ids[4] == 50
        assert (expanded_ids[5:9] == media_token_id).all()

    def test_expand_image_tokens_three_images(self, collate_mod):
        """Multi-image: three placeholders each with 4-token expansion."""
        media_token_id = 163605
        input_ids = torch.tensor([1, media_token_id, 2, media_token_id, 3, media_token_id, 4])
        attention_mask = torch.ones(7, dtype=torch.long)

        # All three: [1,4,4] -> 4 tokens each
        grid_thws = torch.tensor([[1, 4, 4], [1, 4, 4], [1, 4, 4]])

        expanded_ids, expanded_mask = collate_mod._expand_image_tokens(
            input_ids, attention_mask, grid_thws, media_token_id
        )

        # 7 - 3 + 4*3 = 16 tokens
        assert expanded_ids.shape[0] == 16
        assert expanded_mask.shape[0] == 16

        # Spot-check non-image tokens
        assert expanded_ids[0] == 1
        assert expanded_ids[5] == 2
        assert expanded_ids[10] == 3
        assert expanded_ids[15] == 4

    def test_expand_image_tokens_mismatch_raises(self, collate_mod):
        """ValueError when placeholder count does not match grid_thws rows."""
        media_token_id = 163605
        # Two placeholders but only one grid entry
        input_ids = torch.tensor([media_token_id, 5, media_token_id])
        attention_mask = torch.ones(3, dtype=torch.long)
        grid_thws = torch.tensor([[1, 4, 4]])

        with pytest.raises(ValueError, match="placeholder"):
            collate_mod._expand_image_tokens(input_ids, attention_mask, grid_thws, media_token_id)


# =============================================================================
# Tests for kimi_k25_vl_collate_fn
# =============================================================================


class DummyKimiK25VLProcessor:
    """Dummy processor for Kimi K2.5 VL collate function tests."""

    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)
        self.media_placeholder_token_id = 163605
        self.chat_calls = []
        self.forward_calls = []

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize, **kwargs):
        assert add_generation_prompt is False
        assert tokenize is False
        self.chat_calls.append({"conversation": conversation, "kwargs": kwargs})
        return "chat:processed"

    def __call__(self, *, text, return_tensors, medias=None, **kwargs):
        assert return_tensors == "pt"
        self.forward_calls.append({"text": text, "return_tensors": return_tensors, "medias": medias, **kwargs})

        # Simulate processor output with single placeholder
        input_ids = torch.tensor([[1, 2, self.media_placeholder_token_id, 3, 4]])
        attention_mask = torch.ones_like(input_ids)

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if medias:
            result["pixel_values"] = torch.randn(1, 3, 14, 14)
            result["grid_thws"] = torch.tensor([[1, 4, 4]])  # 4 image tokens

        return result


def test_kimi_k25_vl_collate_fn_registered(collate_mod):
    """Test that kimi_k25_vl_collate_fn is registered in COLLATE_FNS."""
    assert "KimiK25Processor" in collate_mod.COLLATE_FNS
    assert collate_mod.COLLATE_FNS["KimiK25Processor"] is collate_mod.kimi_k25_vl_collate_fn


def test_kimi_k25_vl_collate_fn_basic(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn basic functionality."""
    processor = DummyKimiK25VLProcessor()

    # Stub build_labels
    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch


def test_kimi_k25_vl_collate_fn_with_image(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn with image content."""
    processor = DummyKimiK25VLProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "test.jpg"},
                {"type": "text", "text": "What is this?"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "An image"}]},
    ]

    examples = [{"conversation": conversation_with_image}]
    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    # Should have pixel_values and grid_thws from image processing
    assert "pixel_values" in batch
    assert "grid_thws" in batch
    assert "image_grid_hws" in batch

    # image_grid_hws should be [N, 2] (H, W only)
    assert batch["image_grid_hws"].shape[-1] == 2


def test_kimi_k25_vl_collate_fn_image_token_expansion(collate_mod, monkeypatch):
    """Test that kimi_k25_vl_collate_fn expands image tokens correctly."""
    processor = DummyKimiK25VLProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation_with_image = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "test.jpg"},
                {"type": "text", "text": "Describe"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Description"}]},
    ]

    examples = [{"conversation": conversation_with_image}]
    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    # With grid_thws [1, 4, 4], expansion yields 4 image tokens
    # Original: [1, 2, placeholder, 3, 4] = 5 tokens
    # Expanded: [1, 2, placeholder*4, 3, 4] = 8 tokens
    # After :-1 shift: 7 tokens
    assert batch["input_ids"].shape[1] == 7


def test_kimi_k25_vl_collate_fn_with_max_length(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn with max_length padding."""
    processor = DummyKimiK25VLProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    # Set max_length larger than natural sequence
    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor, max_length=100)

    # After :-1 shift, should be max_length - 1 = 99
    assert batch["input_ids"].shape[1] == 99


def test_kimi_k25_vl_collate_fn_drops_overlong(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn drops samples when drop_overlong=True."""

    # Custom processor that produces longer sequences
    class LongSequenceProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = 163605

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            # Produce a 50-token sequence
            input_ids = torch.arange(1, 51).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = LongSequenceProcessor()

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    # All samples exceed max_length=20 with drop_overlong=True → ValueError
    with pytest.raises(ValueError, match="All samples in batch exceed max_length"):
        collate_mod.kimi_k25_vl_collate_fn(examples, processor, max_length=20, drop_overlong=True)


def test_kimi_k25_vl_collate_fn_truncates_by_default(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn passes truncation to processor by default (not drop)."""

    captured_kwargs = {}

    class TruncatingProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = 163605

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            captured_kwargs.update(kwargs)
            max_len = kwargs.get("max_length", 50)
            # Respect truncation like a real processor would
            length = min(50, max_len) if kwargs.get("truncation") else 50
            input_ids = torch.arange(1, length + 1).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = TruncatingProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor, max_length=20)

    # Processor should receive truncation=True and max_length=20
    assert captured_kwargs.get("truncation") is True
    assert captured_kwargs.get("max_length") == 20
    # After processor truncation to 20 and autoregressive shift (:-1), seq_len = 19
    assert batch["input_ids"].shape == (1, 19)
    assert batch["attention_mask"].shape == (1, 19)
    assert batch["labels"].shape == (1, 19)


def test_kimi_k25_vl_collate_fn_no_drop_preserves_batch_size(collate_mod, monkeypatch):
    """Test that default (no drop) preserves all samples in batch for PP compatibility."""
    call_count = [0]

    class TruncatingProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = 163605

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            call_count[0] += 1
            max_len = kwargs.get("max_length", 50)
            # First sample: 50 tokens, second: 10 tokens
            base_length = 50 if call_count[0] == 1 else 10
            length = min(base_length, max_len) if kwargs.get("truncation") else base_length
            input_ids = torch.arange(1, length + 1).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = TruncatingProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}, {"conversation": conversation}]

    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor, max_length=20)

    # Both samples preserved (truncated by processor, not dropped). After shift: 19 tokens.
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == 19


def test_kimi_k25_vl_collate_fn_truncation_drops_image_data(collate_mod, monkeypatch):
    """Test that truncation into image region drops pixel_values/grid_thws and replaces orphaned tokens."""
    MEDIA_TOKEN_ID = 163605
    PAD_TOKEN_ID = 0

    class ImageProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=PAD_TOKEN_ID)
            self.media_placeholder_token_id = MEDIA_TOKEN_ID

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            # 5 text tokens + 1 image placeholder = 6 tokens pre-expansion
            input_ids = torch.tensor([[1, 2, MEDIA_TOKEN_ID, 3, 4, 5]])
            attention_mask = torch.ones_like(input_ids)
            # grid_thws: t=1, h=8, w=8 → (8//2)*(8//2) = 16 expanded image tokens
            # Post-expansion: 5 text + 16 image = 21 tokens
            grid_thws = torch.tensor([[1, 8, 8]])
            pixel_values = torch.randn(1, 3, 64, 64)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "grid_thws": grid_thws,
                "pixel_values": pixel_values,
            }

    processor = ImageProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels_from_template", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "image", "image": "test.jpg"}, {"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    # max_length=15 < 21 post-expansion tokens → truncation cuts into image region
    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor, max_length=15)

    # Sample should be kept (not dropped) but image data excluded
    assert batch["input_ids"].shape[0] == 1
    # No pixel_values or grid_thws since image was partially truncated
    assert "pixel_values" not in batch
    assert "grid_thws" not in batch
    assert "image_grid_hws" not in batch
    # Orphaned image tokens should be replaced with pad_token_id
    assert (batch["input_ids"] == MEDIA_TOKEN_ID).sum().item() == 0


def test_kimi_k25_vl_collate_fn_multiple_examples(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn handles multiple examples with padding."""
    # Processor that produces variable length sequences
    call_count = [0]

    class VariableLengthProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = 163605

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            call_count[0] += 1
            # First call: 5 tokens, second call: 8 tokens
            length = 5 if call_count[0] == 1 else 8
            input_ids = torch.arange(1, length + 1).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = VariableLengthProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}, {"conversation": conversation}]

    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    # Both should be padded to same length (max is 8, after :-1 shift = 7)
    assert batch["input_ids"].shape == (2, 7)
    assert batch["attention_mask"].shape == (2, 7)
    assert batch["labels"].shape == (2, 7)


def test_kimi_k25_vl_collate_fn_default_media_token_id(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn uses default media_token_id when not in processor."""

    class ProcessorWithoutMediaToken:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = ProcessorWithoutMediaToken()

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    # Should not raise, uses default 163605
    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor)
    assert "input_ids" in batch


def test_kimi_k25_vl_collate_fn_labels_shifted(collate_mod, monkeypatch):
    """Test that labels are shifted by [:, 1:]."""

    class SimpleProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = 163605

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            input_ids = torch.tensor([[1, 2, 3, 4, 5]])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = SimpleProcessor()

    def fake_build_labels(input_ids, conversations, processor_arg):
        # Return labels [10, 20, 30, 40, 50]
        return torch.tensor([[10, 20, 30, 40, 50]])

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    # Labels should be shifted: [10, 20, 30, 40, 50][:, 1:] = [20, 30, 40, 50]
    # Then input_ids[:, :-1] means labels also become [:, :-1] from the shape matching
    # Final: [20, 30, 40]
    assert batch["labels"].shape[1] == 4  # 5 - 1 = 4


def test_kimi_k25_vl_collate_fn_fake_image_mask(collate_mod, monkeypatch):
    """mask_fake_vision_tokens_batch must be called with the injected sample index."""
    media_token_id = 163605

    class FakeImageProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = media_token_id

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            input_ids = torch.tensor([[1, media_token_id, media_token_id, 2]])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = FakeImageProcessor()
    monkeypatch.setattr(
        collate_mod,
        "build_labels_from_template",
        lambda ids, convs, proc: torch.full_like(ids, -100),
        raising=True,
    )

    mask_calls = []

    def fake_mask(batch, proc, sample_indices):
        mask_calls.append(list(sample_indices))

    monkeypatch.setattr(collate_mod, "mask_fake_vision_tokens_batch", fake_mask, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]
    examples = [{"conversation": conversation, "_injected_fake": True}]

    collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    assert len(mask_calls) == 1, "mask_fake_vision_tokens_batch should be called once"
    assert mask_calls[0] == [0], "injected sample is at batch index 0"


def test_kimi_k25_vl_collate_fn_non_fake_not_masked(collate_mod, monkeypatch):
    """mask_fake_vision_tokens_batch must NOT be called for non-injected samples."""
    media_token_id = 163605

    class RealImageProcessor:
        def __init__(self):
            self.tokenizer = DummyTokenizer(pad_token_id=0)
            self.media_placeholder_token_id = media_token_id

        def apply_chat_template(self, conversation, **kwargs):
            return "chat:processed"

        def __call__(self, **kwargs):
            input_ids = torch.tensor([[1, media_token_id, 2]])
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    processor = RealImageProcessor()
    monkeypatch.setattr(
        collate_mod,
        "build_labels_from_template",
        lambda ids, convs, proc: torch.full_like(ids, -100),
        raising=True,
    )

    mask_calls = []
    monkeypatch.setattr(
        collate_mod,
        "mask_fake_vision_tokens_batch",
        lambda batch, proc, indices: mask_calls.append(indices),
        raising=True,
    )

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
    ]
    # _injected_fake is NOT set
    examples = [{"conversation": conversation}]

    collate_mod.kimi_k25_vl_collate_fn(examples, processor)

    assert len(mask_calls) == 0, "mask_fake_vision_tokens_batch should not be called for non-injected samples"


# =============================================================================
# Tests for _ensure_rgb
# =============================================================================


class TestEnsureRgb:
    """Tests for _ensure_rgb helper that converts PIL images to RGB."""

    def test_rgba_image_converted_to_rgb(self, collate_mod):
        img = PILImage.new("RGBA", (4, 4), (255, 0, 0, 128))
        conversations = [
            [
                {"role": "user", "content": [{"image": img}]},
            ]
        ]
        collate_mod._ensure_rgb(conversations)
        assert conversations[0][0]["content"][0]["image"].mode == "RGB"

    def test_grayscale_image_converted_to_rgb(self, collate_mod):
        img = PILImage.new("L", (4, 4), 128)
        conversations = [
            [
                {"role": "user", "content": [{"image": img}]},
            ]
        ]
        collate_mod._ensure_rgb(conversations)
        assert conversations[0][0]["content"][0]["image"].mode == "RGB"

    def test_palette_image_converted_to_rgb(self, collate_mod):
        img = PILImage.new("P", (4, 4))
        conversations = [
            [
                {"role": "user", "content": [{"image": img}]},
            ]
        ]
        collate_mod._ensure_rgb(conversations)
        assert conversations[0][0]["content"][0]["image"].mode == "RGB"

    def test_rgb_image_unchanged(self, collate_mod):
        img = PILImage.new("RGB", (4, 4), (255, 0, 0))
        conversations = [
            [
                {"role": "user", "content": [{"image": img}]},
            ]
        ]
        collate_mod._ensure_rgb(conversations)
        result = conversations[0][0]["content"][0]["image"]
        assert result.mode == "RGB"

    def test_no_images_passthrough(self, collate_mod):
        conversations = [
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
            ]
        ]
        result = collate_mod._ensure_rgb(conversations)
        assert result == conversations

    def test_string_content_skipped(self, collate_mod):
        conversations = [
            [
                {"role": "assistant", "content": "plain string"},
            ]
        ]
        result = collate_mod._ensure_rgb(conversations)
        assert result[0][0]["content"] == "plain string"

    def test_empty_conversations(self, collate_mod):
        assert collate_mod._ensure_rgb([]) == []

    def test_multiple_images_in_one_turn(self, collate_mod):
        rgba = PILImage.new("RGBA", (4, 4))
        gray = PILImage.new("L", (4, 4))
        rgb = PILImage.new("RGB", (4, 4))
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"image": rgba},
                        {"type": "text", "text": "describe these"},
                        {"image": gray},
                        {"image": rgb},
                    ],
                },
            ]
        ]
        collate_mod._ensure_rgb(conversations)
        items = conversations[0][0]["content"]
        assert items[0]["image"].mode == "RGB"
        assert items[1] == {"type": "text", "text": "describe these"}
        assert items[2]["image"].mode == "RGB"
        assert items[3]["image"].mode == "RGB"

    def test_multiple_conversations(self, collate_mod):
        img1 = PILImage.new("RGBA", (4, 4))
        img2 = PILImage.new("L", (4, 4))
        conversations = [
            [{"role": "user", "content": [{"image": img1}]}],
            [{"role": "user", "content": [{"image": img2}]}],
        ]
        collate_mod._ensure_rgb(conversations)
        assert conversations[0][0]["content"][0]["image"].mode == "RGB"
        assert conversations[1][0]["content"][0]["image"].mode == "RGB"

    def test_non_image_dict_items_untouched(self, collate_mod):
        conversations = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "video", "video": "clip.mp4"},
                    ],
                },
            ]
        ]
        result = collate_mod._ensure_rgb(conversations)
        items = result[0][0]["content"]
        assert items[0] == {"type": "text", "text": "hi"}
        assert items[1] == {"type": "video", "video": "clip.mp4"}

    def test_returns_same_list_object(self, collate_mod):
        conversations = [[{"role": "user", "content": [{"type": "text", "text": "x"}]}]]
        result = collate_mod._ensure_rgb(conversations)
        assert result is conversations


class TestDefaultCollateFnEnsureRgb:
    """Test that default_collate_fn integrates _ensure_rgb correctly."""

    def test_rgba_image_converted_before_processing(self, collate_mod, fake_qwen_utils, monkeypatch):
        monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

        captured_conversations = []

        class CapturingProcessor:
            tokenizer = DummyTokenizer()

            def apply_chat_template(self, conv_list, **kwargs):
                for conv in conv_list:
                    for turn in conv:
                        content = turn.get("content")
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and isinstance(item.get("image"), PILImage.Image):
                                    captured_conversations.append(item["image"].mode)
                batch_size = len(conv_list)
                input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
                pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
                return {"input_ids": input_ids, "pixel_values": pixel_values}

        rgba_img = PILImage.new("RGBA", (4, 4), (255, 0, 0, 128))
        conversation = [
            {"role": "user", "content": [{"image": rgba_img}, {"type": "text", "text": "describe"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "red"}]},
        ]

        processor = CapturingProcessor()
        collate_mod.default_collate_fn([{"conversation": conversation}], processor)

        assert captured_conversations == ["RGB"]


# =============================================================================
# Tests for fake image fallback (FSDP / Zero3 hang prevention)
# =============================================================================

# -- Pure-text conversation without any media content items. -----------------
TEXT_ONLY_CONVERSATION = [
    {"role": "user", "content": [{"type": "text", "text": "What is 1+1?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "2"}]},
]

IMAGE_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": PILImage.new("RGB", (4, 4))},
            {"type": "text", "text": "Describe this"},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "A small image"}]},
]

VIDEO_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "Describe this video"},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "A video"}]},
]


class TestBatchHasMedia:
    """Tests for _batch_has_media helper."""

    def test_no_media(self, collate_mod):
        assert collate_mod._batch_has_media([TEXT_ONLY_CONVERSATION]) is False

    def test_with_image(self, collate_mod):
        assert collate_mod._batch_has_media([IMAGE_CONVERSATION]) is True

    def test_with_video(self, collate_mod):
        assert collate_mod._batch_has_media([VIDEO_CONVERSATION]) is True

    def test_mixed_batch_has_media(self, collate_mod):
        """At least one conversation with media → True."""
        assert collate_mod._batch_has_media([TEXT_ONLY_CONVERSATION, IMAGE_CONVERSATION]) is True

    def test_empty_batch(self, collate_mod):
        assert collate_mod._batch_has_media([]) is False

    def test_string_content_no_media(self, collate_mod):
        conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert collate_mod._batch_has_media([conv]) is False


class TestInjectFakeImage:
    """Tests for inject_fake_image_into_conversation helper."""

    def test_injects_into_first_user_list_content(self, collate_mod):
        import copy

        conversation = copy.deepcopy(TEXT_ONLY_CONVERSATION)
        result = collate_mod.inject_fake_image_into_conversation(conversation)

        user_content = result[0]["content"]
        assert user_content[0]["type"] == "image"
        assert isinstance(user_content[0]["image"], PILImage.Image)
        # Original text should still be present.
        assert user_content[1] == {"type": "text", "text": "What is 1+1?"}

    def test_does_not_mutate_original(self, collate_mod):
        import copy

        original = copy.deepcopy(TEXT_ONLY_CONVERSATION)
        result = collate_mod.inject_fake_image_into_conversation(original)

        # Result should be a deep copy, not the same object.
        assert result is not original

    def test_injects_into_string_content(self, collate_mod):
        conv = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        result = collate_mod.inject_fake_image_into_conversation(conv)

        user_content = result[0]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "image"
        assert user_content[1] == {"type": "text", "text": "hello"}

    def test_injects_when_no_user_message(self, collate_mod):
        conv = [{"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}]
        result = collate_mod.inject_fake_image_into_conversation(conv)

        # Should prepend a user message with the fake image.
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "image"


class TestMaskFakeVisionTokensBatch:
    """Tests for mask_fake_vision_tokens_batch helper."""

    def test_masks_known_vision_token_ids(self, collate_mod):
        IMAGE_PAD_ID = 151655

        class FakeProcessor:
            image_token_id = IMAGE_PAD_ID
            tokenizer = DummyTokenizer()

        batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, IMAGE_PAD_ID, IMAGE_PAD_ID, 3],
                    [1, 2, 3, 4, 5],
                ]
            ),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
        }

        collate_mod.mask_fake_vision_tokens_batch(batch, FakeProcessor(), [0])

        # Only sample 0 should be masked at positions with IMAGE_PAD_ID (positions 2, 3).
        assert batch["attention_mask"][0].tolist() == [1, 1, 0, 0, 1]
        # Sample 1 should be untouched (not in sample_indices).
        assert batch["attention_mask"][1].tolist() == [1, 1, 1, 1, 1]

    def test_masks_multiple_samples(self, collate_mod):
        IMAGE_PAD_ID = 151655

        class FakeProcessor:
            image_token_id = IMAGE_PAD_ID
            tokenizer = DummyTokenizer()

        batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, IMAGE_PAD_ID, 3, 4],
                    [1, IMAGE_PAD_ID, IMAGE_PAD_ID, 4, 5],
                ]
            ),
            "attention_mask": torch.ones(2, 5, dtype=torch.long),
        }

        collate_mod.mask_fake_vision_tokens_batch(batch, FakeProcessor(), [0, 1])

        assert batch["attention_mask"][0].tolist() == [1, 1, 0, 1, 1]
        assert batch["attention_mask"][1].tolist() == [1, 0, 0, 1, 1]

    def test_masks_via_tokenizer_convert(self, collate_mod):
        """Processor has no image_token_id attr but tokenizer resolves special tokens."""
        VISION_START = 100
        IMAGE_PAD = 101

        class ResolverTokenizer:
            unk_token_id = 0
            pad_token_id = 0
            eos_token = "<eos>"

            def convert_tokens_to_ids(self, token):
                return {
                    "<|vision_start|>": VISION_START,
                    "<|image_pad|>": IMAGE_PAD,
                }.get(token)

            def __call__(self, *a, **kw):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def decode(self, t):
                return str(t)

        class FakeProcessor:
            tokenizer = ResolverTokenizer()

        batch = {
            "input_ids": torch.tensor([[VISION_START, IMAGE_PAD, IMAGE_PAD, 5]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

        collate_mod.mask_fake_vision_tokens_batch(batch, FakeProcessor(), [0])
        assert batch["attention_mask"][0].tolist() == [0, 0, 0, 1]

    def test_noop_when_no_vision_tokens_found(self, collate_mod):
        class FakeProcessor:
            tokenizer = DummyTokenizer()

        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }

        collate_mod.mask_fake_vision_tokens_batch(batch, FakeProcessor(), [0])
        assert batch["attention_mask"][0].tolist() == [1, 1, 1]

    def test_noop_when_empty_indices(self, collate_mod):
        IMAGE_PAD_ID = 151655

        class FakeProcessor:
            image_token_id = IMAGE_PAD_ID
            tokenizer = DummyTokenizer()

        batch = {
            "input_ids": torch.tensor([[1, 2, IMAGE_PAD_ID, 3]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

        collate_mod.mask_fake_vision_tokens_batch(batch, FakeProcessor(), [])
        assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1]


class TestDefaultCollateFnFakeImage:
    """Integration tests: default_collate_fn masks fake-image samples via _injected_fake flag."""

    def test_fake_image_masked_for_flagged_sample(self, collate_mod, fake_qwen_utils, monkeypatch):
        monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

        IMAGE_PAD_ID = 151655

        # Simulate a sample that was injected with a fake image at dataset level.
        import copy

        fake_conv = copy.deepcopy(TEXT_ONLY_CONVERSATION)
        fake_conv[0]["content"].insert(0, {"type": "image", "image": PILImage.new("RGB", (56, 56), (255, 255, 255))})

        class FakeImageProcessor:
            image_token_id = IMAGE_PAD_ID
            tokenizer = DummyTokenizer()

            def apply_chat_template(self, conv_list, **kwargs):
                batch_size = len(conv_list)
                input_ids = torch.tensor([[1, IMAGE_PAD_ID, 2, 3]]).repeat(batch_size, 1)
                attention_mask = torch.ones_like(input_ids)
                pixel_values = torch.ones(batch_size, 3, 56, 56, dtype=torch.float32)
                return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

        processor = FakeImageProcessor()
        batch = collate_mod.default_collate_fn([{"conversation": fake_conv, "_injected_fake": True}], processor)

        # The image_pad token should be masked out (position 1 has IMAGE_PAD_ID).
        assert batch["attention_mask"][0, 1].item() == 0

    def test_no_masking_when_batch_has_real_media(self, collate_mod, fake_qwen_utils, monkeypatch):
        monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

        captured = {}

        class TrackingProcessor:
            tokenizer = DummyTokenizer()

            def apply_chat_template(self, conv_list, **kwargs):
                captured["conv_list"] = conv_list
                batch_size = len(conv_list)
                input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
                pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
                return {"input_ids": input_ids, "pixel_values": pixel_values}

        processor = TrackingProcessor()
        collate_mod.default_collate_fn([{"conversation": IMAGE_CONVERSATION}], processor)

        # No _injected_fake flag — sample has real media.
        first_user_content = captured["conv_list"][0][0]["content"]
        # Only the original image + text, no extra fake image prepended.
        assert len(first_user_content) == 2


class TestQwen25CollateFnFakeImage:
    """Integration tests: qwen2_5_collate_fn masks fake-image samples via _injected_fake flag."""

    def test_fake_image_extracted_and_masked(self, collate_mod, monkeypatch):
        """When a sample has _injected_fake, the fake image should be extracted and masked."""
        IMAGE_PAD_ID = 151655

        import copy

        fake_conv = copy.deepcopy(TEXT_ONLY_CONVERSATION)
        fake_conv[0]["content"].insert(0, {"type": "image", "image": PILImage.new("RGB", (56, 56), (255, 255, 255))})

        captured = {}

        class FakeProcessor:
            image_token_id = IMAGE_PAD_ID
            tokenizer = DummyTokenizer()

            def apply_chat_template(self, conversation, *, tokenize=False, **kwargs):
                return "dummy text"

            def __call__(self, *, text, images=None, videos=None, **kwargs):
                captured["images"] = images
                batch_size = len(text)
                input_ids = torch.tensor([[1, IMAGE_PAD_ID, 2, 3, 4]]).repeat(batch_size, 1)
                attention_mask = torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

        processor = FakeProcessor()

        def fake_build_labels(input_ids, conversations, proc):
            return torch.full_like(input_ids, -100)

        monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

        batch = collate_mod.qwen2_5_collate_fn([{"conversation": fake_conv, "_injected_fake": True}], processor)

        # The fake image should have been extracted.
        assert captured["images"] is not None
        assert len(captured["images"]) == 1
        assert isinstance(captured["images"][0], PILImage.Image)
        # The image_pad token should be masked (position 1 has IMAGE_PAD_ID).
        assert batch["attention_mask"][0, 1].item() == 0


# =============================================================================
# Tests for build_labels_from_template (template-based label builder)
# =============================================================================

# Token IDs mimicking Qwen's <|im_start|>/<|im_end|> convention.
_IM_START = 151644
_IM_END = 151645
_ASSISTANT_ID = 77091
_NEWLINE_ID = 198


class QwenStyleTokenizer:
    """Tokenizer stub that supports Qwen-style ``<|im_start|>`` / ``<|im_end|>``."""

    unk_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"

    _SPECIAL = {
        "<|im_start|>": _IM_START,
        "<|im_end|>": _IM_END,
        "<|image_pad|>": None,
        "<|video_pad|>": None,
        "<|vision_start|>": None,
        "<|vision_end|>": None,
    }

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL.get(token)

    def encode(self, text, add_special_tokens=False):
        if text == "assistant\n":
            return [_ASSISTANT_ID, _NEWLINE_ID]
        return []

    def __call__(self, text, add_special_tokens=True, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def decode(self, token):
        return str(token)


class Qwen3VLProcessor:
    def __init__(self):
        self.tokenizer = QwenStyleTokenizer()


def _make_qwen_input_ids(*turns):
    """Build an input_ids tensor from (role, content_ids) turn pairs.

    Each turn becomes: <|im_start|> role_id \\n  content...  <|im_end|> \\n
    """
    USER_ID = 872  # "user" token
    ids = []
    for role, content_ids in turns:
        role_id = _ASSISTANT_ID if role == "assistant" else USER_ID
        ids += [_IM_START, role_id, _NEWLINE_ID] + content_ids + [_IM_END, _NEWLINE_ID]
    return torch.tensor([ids], dtype=torch.long)


class TestBuildLabelsFromTemplate:
    """Tests for the template-based label builder."""

    def test_single_turn(self, collate_mod):
        """Labels are set only for assistant content + <|im_end|>."""
        # Layout: [IM_START,USER,NL, 10,11,12, IM_END,NL,  IM_START,ASST,NL, 20,21, IM_END,NL]
        #  pos:      0      1   2   3  4  5    6      7      8      9  10  11 12   13    14
        input_ids = _make_qwen_input_ids(
            ("user", [10, 11, 12]),
            ("assistant", [20, 21]),
        )
        conv = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
        labels = collate_mod.build_labels_from_template(input_ids, [conv], Qwen3VLProcessor())

        expected = torch.full_like(input_ids, -100)
        expected[0, 11] = 20  # assistant content token 1
        expected[0, 12] = 21  # assistant content token 2
        expected[0, 13] = _IM_END  # stop token
        assert torch.equal(labels, expected)

    def test_multi_turn(self, collate_mod):
        """Both assistant turns get labels."""
        input_ids = _make_qwen_input_ids(
            ("user", [10]),
            ("assistant", [20]),
            ("user", [30]),
            ("assistant", [40, 41]),
        )
        labels = collate_mod.build_labels_from_template(input_ids, [[]], Qwen3VLProcessor())

        labeled_positions = (labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
        # Turn 1: content [20] + im_end
        # Turn 2: content [40, 41] + im_end
        assert len(labeled_positions) == 5  # 1+1 + 2+1

    def test_user_text_never_labeled(self, collate_mod):
        """Even if user text matches assistant text, user tokens stay -100."""
        # Layout: [IM_START,USER,NL, 20, IM_END,NL,  IM_START,ASST,NL, 20, IM_END,NL]
        #  pos:      0      1   2   3    4      5      6      7   8   9   10     11
        input_ids = _make_qwen_input_ids(
            ("user", [20]),  # same content as assistant
            ("assistant", [20]),  # should be labeled
        )
        labels = collate_mod.build_labels_from_template(input_ids, [[]], Qwen3VLProcessor())

        # Only the SECOND occurrence (in assistant) should be labeled.
        assert labels[0, 3].item() == -100  # user content [20]
        assert labels[0, 9].item() == 20  # assistant content [20]

    def test_fallback_for_non_qwen_processor(self, collate_mod):
        """Non-Qwen processor types fall back to old build_labels."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        conv = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]
        # Processor type name "P" is not in _IMSTART_TEMPLATE_PROCESSORS → fallback
        processor = type("P", (), {"tokenizer": DummyTokenizer()})()

        # Should not raise; falls back to build_labels.
        labels = collate_mod.build_labels_from_template(input_ids, [conv], processor)
        assert labels.shape == input_ids.shape

    def test_empty_assistant_content(self, collate_mod):
        """Assistant turn with no content tokens → only <|im_end|> is labeled."""
        input_ids = _make_qwen_input_ids(
            ("user", [10]),
            ("assistant", []),  # empty content
        )
        labels = collate_mod.build_labels_from_template(input_ids, [[]], Qwen3VLProcessor())

        # The only labeled token should be <|im_end|> right after the marker.
        labeled = (labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
        assert len(labeled) == 1
        assert input_ids[0, labeled[0]].item() == _IM_END

    def test_padding_ignored(self, collate_mod):
        """Padding tokens (0) at the end are never labeled."""
        base = _make_qwen_input_ids(
            ("user", [10]),
            ("assistant", [20]),
        )
        padded = torch.cat([base, torch.zeros(1, 5, dtype=torch.long)], dim=1)
        labels = collate_mod.build_labels_from_template(padded, [[]], Qwen3VLProcessor())

        # Last 5 positions (padding) must be -100.
        assert (labels[0, -5:] == -100).all()

    def test_batch_processing(self, collate_mod):
        """Multiple samples in a batch are handled independently."""
        ids1 = _make_qwen_input_ids(("user", [10]), ("assistant", [20]))
        ids2 = _make_qwen_input_ids(("user", [30]), ("assistant", [40, 41]))

        # Pad to same length.
        max_len = max(ids1.shape[1], ids2.shape[1])
        ids1 = torch.cat([ids1, torch.zeros(1, max_len - ids1.shape[1], dtype=torch.long)], dim=1)
        ids2 = torch.cat([ids2, torch.zeros(1, max_len - ids2.shape[1], dtype=torch.long)], dim=1)
        batch = torch.cat([ids1, ids2], dim=0)

        labels = collate_mod.build_labels_from_template(batch, [[], []], Qwen3VLProcessor())
        assert labels.shape == batch.shape

        count0 = (labels[0] != -100).sum().item()
        count1 = (labels[1] != -100).sum().item()
        assert count0 == 2  # [20, im_end]
        assert count1 == 3  # [40, 41, im_end]


# ---------------------------------------------------------------------------
# Tests for _derive_turn_markers (Gemma4-style general path)
# ---------------------------------------------------------------------------

# Synthetic token IDs for a Gemma4-style tokenizer.
_SOT = 2  # <start_of_turn>
_USER_TK = 1645  # "user"
_MODEL_TK = 2516  # "model"
_NL = 108  # "\n"
_EOT = 107  # <end_of_turn>
_U_CONTENT = 506  # "u"
# sentinel encoded as two distinct ids
_SEN_A = 999
_SEN_B = 888


class _Gemma4StyleTokenizer:
    """Minimal Gemma4-like tokenizer stub for _derive_turn_markers tests.

    Template layout (ids):
      user turn:      [SOT, USER, NL, <content>, EOT, NL]
      assistant turn: [SOT, MODEL, NL, <content>, EOT]
    """

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        ids = []
        for msg in messages:
            if msg["role"] == "user":
                ids += [_SOT, _USER_TK, _NL, _U_CONTENT, _EOT, _NL]
            else:
                # assistant
                content = msg["content"]
                if content == "XSENTINELMARKERX":
                    content_ids = [_SEN_A, _SEN_B]
                else:
                    content_ids = [42]
                ids += [_SOT, _MODEL_TK, _NL] + content_ids + [_EOT]
        return ids

    def encode(self, text, add_special_tokens=False):
        if text == "XSENTINELMARKERX":
            return [_SEN_A, _SEN_B]
        return []


class TestDeriveTurnMarkers:
    """Unit tests for _derive_turn_markers."""

    def test_extracts_correct_marker_and_eot(self, collate_mod):
        """_derive_turn_markers returns the assistant prefix and EOT id."""
        tokenizer = _Gemma4StyleTokenizer()
        marker, eot = collate_mod._derive_turn_markers(tokenizer)

        # assistant marker should be [SOT, MODEL, NL]
        assert marker == [_SOT, _MODEL_TK, _NL]
        # end-of-turn token should be EOT (token right after sentinel)
        assert eot == _EOT

    def test_raises_when_sentinel_absent(self, collate_mod):
        """ValueError raised when the sentinel does not appear in template output."""

        class _NoSentinelTokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
                # Never includes sentinel
                return [1, 2, 3, 4, 5]

            def encode(self, text, add_special_tokens=False):
                return [999, 888]  # sentinel ids that won't be found above

        with pytest.raises(ValueError, match="not found"):
            collate_mod._derive_turn_markers(_NoSentinelTokenizer())

    def test_raises_when_marker_is_empty(self, collate_mod):
        """ValueError raised when user and sentinel positions are adjacent (empty marker)."""

        class _EmptyMarkerTokenizer:
            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
                # user_ids == all_ids[:6], sentinel starts at position 6 → marker empty
                user_ids = [1, 2, 3, 4, 5, 6]
                sentinel_ids = [999, 888]
                eot = [107]
                msgs = messages
                if len(msgs) == 1:
                    return user_ids
                return user_ids + sentinel_ids + eot

            def encode(self, text, add_special_tokens=False):
                return [999, 888]

        with pytest.raises(ValueError, match="empty"):
            collate_mod._derive_turn_markers(_EmptyMarkerTokenizer())


class TestBuildLabelsFromTemplateGeneralPath:
    """Tests for the general (non-Qwen) path of build_labels_from_template."""

    def _make_gemma4_input_ids(self, *turns):
        """Build input_ids matching _Gemma4StyleTokenizer layout."""
        ids = []
        for role, content_ids in turns:
            if role == "user":
                ids += [_SOT, _USER_TK, _NL, _U_CONTENT, _EOT, _NL]
            else:
                ids += [_SOT, _MODEL_TK, _NL] + content_ids + [_EOT]
        return torch.tensor([ids], dtype=torch.long)

    def test_general_path_labels_assistant_tokens(self, collate_mod):
        """Non-Qwen processor with apply_chat_template uses general path and labels correctly."""
        input_ids = self._make_gemma4_input_ids(
            ("user", []),
            ("assistant", [42, 43]),
        )

        class _Proc:
            tokenizer = _Gemma4StyleTokenizer()

        labels = collate_mod.build_labels_from_template(input_ids, [[]], _Proc())

        # Labeled positions: content tokens [42, 43] + EOT
        labeled = (labels[0] != -100).nonzero(as_tuple=True)[0].tolist()
        labeled_vals = [input_ids[0, p].item() for p in labeled]
        assert 42 in labeled_vals
        assert 43 in labeled_vals
        assert _EOT in labeled_vals
        # User tokens must NOT be labeled
        assert labels[0, 0].item() == -100  # SOT
        assert labels[0, 1].item() == -100  # USER

    def test_general_path_fallback_on_derive_failure(self, collate_mod):
        """If _derive_turn_markers raises, falls back to build_labels without error."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        conv = [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ]

        class _BrokenTokenizer:
            """Has apply_chat_template but sentinel never appears → derive fails."""

            def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
                return [1, 2, 3]

            def encode(self, text, add_special_tokens=False):
                return [999, 888]

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

            def convert_tokens_to_ids(self, token):
                return None

            def decode(self, token):
                return str(token)

            pad_token_id = 0
            eos_token = "<eos>"

        class _BrokenProc:
            tokenizer = _BrokenTokenizer()

        # Must not raise; should return a tensor of correct shape
        labels = collate_mod.build_labels_from_template(input_ids, [conv], _BrokenProc())
        assert labels.shape == input_ids.shape


# ---------------------------------------------------------------------------
# Tests for _drop_overlong_samples / _estimate_media_tokens
# ---------------------------------------------------------------------------


class _DropTestTokenizer:
    """Tokenizer whose encode returns 1 token per character."""

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text)))


class _DropTestProcessor:
    """Processor for _drop_overlong_samples tests.

    ``apply_chat_template`` concatenates all text content in the conversation,
    so the token count equals the total character count of the text.
    """

    def __init__(self):
        self.tokenizer = _DropTestTokenizer()

    def apply_chat_template(self, convs, tokenize=False, **kwargs):
        results = []
        for conv in convs:
            text = ""
            for msg in conv:
                c = msg.get("content", "")
                if isinstance(c, str):
                    text += c
                elif isinstance(c, list):
                    for item in c:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text += item.get("text", "")
            results.append(text)
        return results


def test_drop_overlong_samples_filters_long(collate_mod):
    """Verify that samples exceeding max_length are dropped."""
    short_conv = [{"role": "user", "content": [{"type": "text", "text": "x" * 10}]}]
    long_conv = [{"role": "user", "content": [{"type": "text", "text": "x" * 100}]}]

    result, kept = collate_mod._drop_overlong_samples(
        [short_conv, long_conv],
        _DropTestProcessor(),
        max_length=50,
    )
    assert len(result) == 1
    assert result[0] is short_conv
    assert kept == [0]


def test_drop_overlong_samples_keeps_short(collate_mod):
    """Verify that samples within max_length are kept."""
    conv1 = [{"role": "user", "content": [{"type": "text", "text": "x" * 10}]}]
    conv2 = [{"role": "user", "content": [{"type": "text", "text": "x" * 20}]}]

    result, kept = collate_mod._drop_overlong_samples(
        [conv1, conv2],
        _DropTestProcessor(),
        max_length=50,
    )
    assert len(result) == 2
    assert result[0] is conv1
    assert result[1] is conv2
    assert kept == [0, 1]


def test_drop_overlong_samples_all_long_raises(collate_mod):
    """Verify ValueError when all samples exceed max_length."""
    long1 = [{"role": "user", "content": [{"type": "text", "text": "x" * 100}]}]
    long2 = [{"role": "user", "content": [{"type": "text", "text": "x" * 200}]}]

    with pytest.raises(ValueError, match="All 2 samples"):
        collate_mod._drop_overlong_samples(
            [long1, long2],
            _DropTestProcessor(),
            max_length=50,
        )


def test_drop_overlong_samples_none_max_length_noop(collate_mod):
    """With max_length=None, all samples are returned unchanged."""
    convs = [
        [{"role": "user", "content": [{"type": "text", "text": "x" * 9999}]}],
    ]
    result, kept = collate_mod._drop_overlong_samples(convs, _DropTestProcessor(), max_length=None)
    assert result is convs
    assert kept == [0]


def test_default_collate_fn_truncation_by_default(collate_mod, fake_qwen_utils, monkeypatch):
    """Verify that with max_length set and drop_overlong=False (default), truncation=True."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    captured_kwargs = {}

    class CapturingProcessor:
        tokenizer = DummyTokenizer()

        def apply_chat_template(self, conv_list, **kwargs):
            if kwargs.get("tokenize", False):
                captured_kwargs.update(kwargs)
                batch_size = len(conv_list)
                input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
                pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
                return {"input_ids": input_ids, "pixel_values": pixel_values}
            else:
                return ["short"]

    processor = CapturingProcessor()
    collate_mod.default_collate_fn(
        [{"conversation": CONVERSATION}],
        processor,
        max_length=512,
    )

    assert captured_kwargs.get("truncation") is True
    assert captured_kwargs.get("max_length") == 512
    assert captured_kwargs.get("padding") == "max_length"


def test_default_collate_fn_no_truncation_with_drop_overlong(collate_mod, fake_qwen_utils, monkeypatch):
    """Verify that with max_length set and drop_overlong=True, truncation=False."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    captured_kwargs = {}

    class CapturingProcessor:
        tokenizer = DummyTokenizer()

        def apply_chat_template(self, conv_list, **kwargs):
            if kwargs.get("tokenize", False):
                captured_kwargs.update(kwargs)
                batch_size = len(conv_list)
                input_ids = torch.arange(1, 5).unsqueeze(0).repeat(batch_size, 1)
                pixel_values = torch.ones(batch_size, 3, 64, 64, dtype=torch.float32)
                return {"input_ids": input_ids, "pixel_values": pixel_values}
            else:
                # Called by _drop_overlong_samples for estimation
                return ["short"]

    processor = CapturingProcessor()
    collate_mod.default_collate_fn(
        [{"conversation": CONVERSATION}],
        processor,
        max_length=512,
        drop_overlong=True,
    )

    assert captured_kwargs.get("truncation") is False
    assert captured_kwargs.get("max_length") == 512
    assert captured_kwargs.get("padding") == "max_length"


def test_default_collate_fn_preserves_batch_size_after_drop_overlong(collate_mod, fake_qwen_utils, monkeypatch):
    """Dropped overlong samples can be backfilled for fixed-size PP batches."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)
    monkeypatch.setattr(
        collate_mod,
        "build_labels_from_template",
        lambda input_ids, conversations, processor: torch.zeros_like(input_ids),
    )

    short_conv = [{"role": "user", "content": [{"type": "text", "text": "x" * 10}]}]
    long_conv = [{"role": "user", "content": [{"type": "text", "text": "x" * 100}]}]
    captured = {}

    class PreserveProcessor(_DropTestProcessor):
        def apply_chat_template(self, convs, tokenize=False, **kwargs):
            if tokenize:
                captured["tokenized_batch_size"] = len(convs)
                captured["truncation"] = kwargs.get("truncation")
                return {"input_ids": torch.arange(1, 5).unsqueeze(0).repeat(len(convs), 1)}
            return super().apply_chat_template(convs, tokenize=tokenize, **kwargs)

    batch = collate_mod.default_collate_fn(
        [{"conversation": short_conv}, {"conversation": long_conv}],
        PreserveProcessor(),
        max_length=50,
        drop_overlong=True,
        preserve_batch_size=True,
    )

    assert captured["tokenized_batch_size"] == 2
    assert captured["truncation"] is False
    assert batch["input_ids"].shape[0] == 2


def test_estimate_media_tokens_with_pil_image(collate_mod):
    """Verify _estimate_media_tokens estimates tokens from PIL image dimensions."""
    img = PILImage.new("RGB", (560, 560))  # 560x560 image
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "What is this?"},
            ],
        },
    ]

    class ProcWithImageProcessor:
        class image_processor:
            patch_size = 14
            merge_size = 2
            size = {"shortest_edge": 56 * 56, "longest_edge": 14 * 14 * 4 * 1280}

    extra = collate_mod._estimate_media_tokens(conversation, ProcWithImageProcessor())
    # 560x560 → smart_resize → 560x560 (already aligned)
    # tokens = (560/14) * (560/14) / (2*2) = 40*40/4 = 400
    # extra = 400 - 1 = 399
    assert extra == 399


def test_estimate_media_tokens_no_image_processor(collate_mod):
    """Without image_processor, _estimate_media_tokens returns 0."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": PILImage.new("RGB", (100, 100))},
            ],
        },
    ]

    class ProcNoIP:
        pass

    assert collate_mod._estimate_media_tokens(conversation, ProcNoIP()) == 0


# ---------------------------------------------------------------------------
# _count_media_per_sample tests
# ---------------------------------------------------------------------------


class TestCountMediaPerSample:
    """Tests for the _count_media_per_sample helper."""

    def test_single_image_per_sample(self, collate_mod):
        conversations = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Hi"}]}],
            [{"role": "user", "content": [{"type": "text", "text": "No media"}]}],
        ]
        img_counts, vid_counts = collate_mod._count_media_per_sample(conversations)
        assert img_counts == [1, 0]
        assert vid_counts == [0, 0]

    def test_multi_image_per_sample(self, collate_mod):
        conversations = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "image"}]}],
            [{"role": "user", "content": [{"type": "image"}]}],
        ]
        img_counts, vid_counts = collate_mod._count_media_per_sample(conversations)
        assert img_counts == [3, 1]
        assert vid_counts == [0, 0]

    def test_videos_counted(self, collate_mod):
        conversations = [
            [{"role": "user", "content": [{"type": "video"}, {"type": "video"}]}],
            [{"role": "user", "content": [{"type": "image"}, {"type": "video"}]}],
        ]
        img_counts, vid_counts = collate_mod._count_media_per_sample(conversations)
        assert img_counts == [0, 1]
        assert vid_counts == [2, 1]

    def test_no_media(self, collate_mod):
        conversations = [
            [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        ]
        img_counts, vid_counts = collate_mod._count_media_per_sample(conversations)
        assert img_counts == [0]
        assert vid_counts == [0]

    def test_string_content_ignored(self, collate_mod):
        conversations = [
            [{"role": "user", "content": "just a string"}],
        ]
        img_counts, vid_counts = collate_mod._count_media_per_sample(conversations)
        assert img_counts == [0]
        assert vid_counts == [0]


# ---------------------------------------------------------------------------
# Per-sample count keys in collate outputs
# ---------------------------------------------------------------------------


def test_default_collate_fn_has_per_sample_image_counts(collate_mod, fake_qwen_utils, monkeypatch):
    """default_collate_fn should include n_images_per_sample when images are present."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    # Conversations with 2 images in sample 0 and 1 image in sample 1
    conv_with_images = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": PILImage.new("RGB", (10, 10))},
                        {"type": "image", "image": PILImage.new("RGB", (10, 10))},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            ]
        },
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": PILImage.new("RGB", (10, 10))},
                        {"type": "text", "text": "what"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "yes"}]},
            ]
        },
    ]

    processor = DummyDefaultProcessor()
    batch = collate_mod.default_collate_fn(conv_with_images, processor)

    assert "n_images_per_sample" in batch
    assert batch["n_images_per_sample"].tolist() == [2, 1]


def test_default_collate_fn_no_per_sample_counts_for_text_only(collate_mod, fake_qwen_utils, monkeypatch):
    """default_collate_fn should not include per-sample count keys for text-only batches.

    Note: With per-sample fake image injection, text-only samples arrive at the
    collate function already carrying fake images (with _injected_fake flag).
    When no flag is set (raw text-only samples without dataset wrapper), the
    collate function does not inject, so no per-sample counts are present.
    """
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()
    batch = collate_mod.default_collate_fn([{"conversation": CONVERSATION} for _ in range(2)], processor)

    # No _injected_fake flag and no real media → no per-sample count keys.
    # (If using a dataset wrapper, fake images would already be injected.)
    if "n_images_per_sample" in batch:
        assert batch["n_images_per_sample"][0].item() >= 0


def test_pad_collate_fn_has_per_sample_image_counts(collate_mod, fake_qwen_utils, monkeypatch):
    """pad_collate_fn should include n_images_per_sample when images are present."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()

    # Pre-tokenized samples with image media
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([10, 11, 12, 13]),
            "pixel_values": torch.randn(5, 3, 14, 14),  # 2 images worth of patches
            "image_grid_thw": torch.tensor([[1, 2, 2], [1, 1, 1]]),  # 2 images
        },
        {
            "input_ids": torch.tensor([5, 6, 7, 8]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([14, 15, 16, 17]),
            "pixel_values": torch.randn(4, 3, 14, 14),  # 1 image worth of patches
            "image_grid_thw": torch.tensor([[1, 2, 2]]),  # 1 image
        },
    ]

    batch = collate_mod.pad_collate_fn(examples, processor)

    assert "n_images_per_sample" in batch
    assert batch["n_images_per_sample"].tolist() == [2, 1]
    # image_grid_thw should be concatenated
    assert batch["image_grid_thw"].shape[0] == 3


def test_pad_collate_fn_has_per_sample_video_counts(collate_mod, fake_qwen_utils, monkeypatch):
    """pad_collate_fn should include n_videos_per_sample when videos are present."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()

    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([10, 11, 12, 13]),
            "pixel_values_videos": torch.randn(10, 3, 14, 14),
            "video_grid_thw": torch.tensor([[2, 2, 2], [1, 3, 3]]),  # 2 videos
        },
        {
            "input_ids": torch.tensor([5, 6, 7, 8]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([14, 15, 16, 17]),
            # No videos for this sample
        },
    ]

    batch = collate_mod.pad_collate_fn(examples, processor)

    assert "n_videos_per_sample" in batch
    assert batch["n_videos_per_sample"].tolist() == [2, 0]


def test_pad_collate_fn_no_per_sample_counts_without_media(collate_mod, fake_qwen_utils, monkeypatch):
    """pad_collate_fn should not include n_images/n_videos keys when only text is present."""
    monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

    processor = DummyDefaultProcessor()

    # Provide pixel_values but no grid tensors
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([10, 11, 12, 13]),
            "pixel_values": torch.randn(1, 3, 14, 14),
        },
        {
            "input_ids": torch.tensor([5, 6, 7, 8]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "labels": torch.tensor([14, 15, 16, 17]),
        },
    ]

    batch = collate_mod.pad_collate_fn(examples, processor)

    # No image_grid_thw or video_grid_thw → no per-sample count keys
    assert "n_images_per_sample" not in batch
    assert "n_videos_per_sample" not in batch


# ---------------------------------------------------------------------------
# make_robust_collate
# ---------------------------------------------------------------------------


class TestMakeRobustCollate:
    def test_success_on_first_try(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

        dataset = [{"x": 0}, {"x": 1}, {"x": 2}]
        collate_fn = lambda batch: {"result": sum(d["x"] for d in batch)}
        wrapped = make_robust_collate(dataset, collate_fn, max_retries=3)
        result = wrapped([{"x": 10}, {"x": 20}])
        assert result == {"result": 30}

    def test_retries_on_failure(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

        call_count = 0

        def flaky_collate(batch):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("flaky error")
            return {"ok": True}

        dataset = [{"x": i} for i in range(10)]
        wrapped = make_robust_collate(dataset, flaky_collate, max_retries=5)
        result = wrapped([{"x": 0}])
        assert result == {"ok": True}
        assert call_count == 3

    def test_raises_after_max_retries(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

        def always_fails(batch):
            raise ValueError("always fails")

        dataset = [{"x": i} for i in range(10)]
        wrapped = make_robust_collate(dataset, always_fails, max_retries=2)
        with pytest.raises(RuntimeError, match="Collate failed after 2 retries"):
            wrapped([{"x": 0}])


# ---------------------------------------------------------------------------
# neat_packed_vlm_collater — attn_implementation variants
# ---------------------------------------------------------------------------


class TestNeatPackedVlmCollaterAttnImpl:
    def _make_packed_sample(self, seq_len=16, n_images=1):
        """Create a minimal packed sample dict."""
        input_ids = torch.randint(100, 30000, (seq_len,))
        labels = torch.randint(100, 30000, (seq_len,))
        # Indexed attention mask: all tokens belong to sequence 1
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        position_ids = torch.arange(seq_len)
        sample = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if n_images > 0:
            sample["pixel_values"] = torch.randn(n_images, 3, 56, 56)
            sample["image_grid_thw"] = torch.tensor([[1, 2, 2]] * n_images)
        return sample

    def test_flash_attention_2_returns_2d_mask(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        batch = [self._make_packed_sample(16, 0), self._make_packed_sample(12, 0)]
        result = neat_packed_vlm_collater(batch, attn_implementation="flash_attention_2")
        # Flash attention keeps the 2D indexed mask
        assert result["attention_mask"].ndim == 2

    def test_sdpa_returns_4d_mask(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        batch = [self._make_packed_sample(16, 0), self._make_packed_sample(12, 0)]
        result = neat_packed_vlm_collater(batch, attn_implementation="sdpa")
        # SDPA produces 4D block-causal mask
        assert result["attention_mask"].ndim == 4

    def test_fixed_max_length_pads_to_max(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        batch = [self._make_packed_sample(10, 0)]
        result = neat_packed_vlm_collater(batch, max_length=32, attn_implementation="flash_attention_2")
        assert result["input_ids"].shape[1] == 32

    def test_concatenates_pixel_values(self):
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        s1 = self._make_packed_sample(16, n_images=2)
        s2 = self._make_packed_sample(12, n_images=1)
        result = neat_packed_vlm_collater([s1, s2], attn_implementation="flash_attention_2")
        assert result["pixel_values"].shape[0] == 3  # 2 + 1
        assert result["image_grid_thw"].shape[0] == 3


# ---------------------------------------------------------------------------
# Tests for public gemma4_inject_thinking_prefix hook
# ---------------------------------------------------------------------------


class _GemmaTokenizerStub:
    """Encodes the Gemma4 marker/prefix strings to fixed token sequences."""

    pad_token_id = 0

    _MARKER = "<|turn>model\n"
    _PREFIX = "<|channel>thought\n<channel|>"
    _MARKER_IDS = [10, 11]
    _PREFIX_IDS = [20, 21, 22]

    def encode(self, text, add_special_tokens=False):
        if text == self._MARKER:
            return list(self._MARKER_IDS)
        if text == self._PREFIX:
            return list(self._PREFIX_IDS)
        return []


class _NonGemmaTokenizerStub:
    """Encodes the Gemma4 marker/prefix to empty (no matching tokens)."""

    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return []


def test_gemma4_inject_thinking_prefix_inserts_after_marker(collate_mod):
    """Hook injects prefix ids after each <|turn>model\\n marker."""
    marker = _GemmaTokenizerStub._MARKER_IDS
    prefix = _GemmaTokenizerStub._PREFIX_IDS
    # [user...] [marker] [answer]
    seq = torch.tensor([[1, 2, 3, *marker, 4, 5]])
    batch = {"input_ids": seq.clone(), "attention_mask": torch.ones_like(seq)}

    out = collate_mod.gemma4_inject_thinking_prefix(batch, _GemmaTokenizerStub())
    expected = torch.tensor([[1, 2, 3, *marker, *prefix, 4, 5]])
    assert torch.equal(out["input_ids"], expected)
    assert out["attention_mask"].shape == expected.shape
    # injected positions are unmasked (visible)
    inject_start = 3 + len(marker)
    inject_end = inject_start + len(prefix)
    assert (out["attention_mask"][0, inject_start:inject_end] == 1).all()


def test_gemma4_inject_thinking_prefix_noop_for_non_gemma_tokenizer(collate_mod):
    """For tokenizers without the marker/prefix vocab, the batch is returned unchanged."""
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    batch = {"input_ids": seq.clone(), "attention_mask": torch.ones_like(seq)}

    out = collate_mod.gemma4_inject_thinking_prefix(batch, _NonGemmaTokenizerStub())
    assert torch.equal(out["input_ids"], seq)


def test_gemma4_inject_thinking_prefix_accepts_processor_or_tokenizer(collate_mod):
    """Hook unwraps processor.tokenizer and also accepts a raw tokenizer."""

    class _Processor:
        tokenizer = _GemmaTokenizerStub()

    marker = _GemmaTokenizerStub._MARKER_IDS
    seq = torch.tensor([[*marker, 9]])
    batch_a = {"input_ids": seq.clone(), "attention_mask": torch.ones_like(seq)}
    batch_b = {"input_ids": seq.clone(), "attention_mask": torch.ones_like(seq)}

    out_proc = collate_mod.gemma4_inject_thinking_prefix(batch_a, _Processor())
    out_tok = collate_mod.gemma4_inject_thinking_prefix(batch_b, _GemmaTokenizerStub())
    assert torch.equal(out_proc["input_ids"], out_tok["input_ids"])

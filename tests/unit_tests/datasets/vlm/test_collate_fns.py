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
        assert truncation is True
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


@pytest.mark.parametrize("fn_name", ["qwen2_5_collate_fn", "default_collate_fn", "qwen3_omni_collate_fn"])
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
    collate_mod.default_collate_fn(
        [{"conversation": CONVERSATION}], processor, max_length=512
    )

    assert captured_kwargs.get("max_length") == 512
    assert captured_kwargs.get("padding") == "max_length"


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

    # Stub build_labels to return deterministic labels
    # The collate fn does labels[:, 1:] so we need 5 elements to get 4 after shift
    labels_stub = torch.tensor([[10, 11, 12, 13, 14]], dtype=torch.long)

    def fake_build_labels(input_ids, conversations, processor_arg):
        assert processor_arg is processor
        return labels_stub

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

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
    monkeypatch.setattr(
        collate_mod, "build_labels", lambda *args, **kwargs: labels_stub, raising=True
    )

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
    monkeypatch.setattr(
        collate_mod, "build_labels", lambda *args, **kwargs: labels_stub, raising=True
    )

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
    monkeypatch.setattr(
        collate_mod, "build_labels", lambda *args, **kwargs: labels_stub, raising=True
    )

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

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

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
        self.forward_calls.append(
            {"text": text, "return_tensors": return_tensors, "medias": medias, **kwargs}
        )

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


def test_kimi_k25_vl_collate_fn_truncation(collate_mod, monkeypatch):
    """Test kimi_k25_vl_collate_fn truncates when max_length is smaller."""
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

    def fake_build_labels(input_ids, conversations, processor_arg):
        batch_size, seq_len = input_ids.shape
        return torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(collate_mod, "build_labels", fake_build_labels, raising=True)

    conversation = [
        {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "Hi"}]},
    ]
    examples = [{"conversation": conversation}]

    # Truncate to 20 tokens
    batch = collate_mod.kimi_k25_vl_collate_fn(examples, processor, max_length=20)

    # After :-1 shift, should be 19
    assert batch["input_ids"].shape[1] == 19


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

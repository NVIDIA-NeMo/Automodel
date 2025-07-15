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
from typing import List

import pytest
import torch

SKIP_TOKEN = 99  # token that must be masked with -100 in labels


class DummyTokenizer:
    """
    Mimics the tokenizer API used by create_loss_mask_with_start_of_response_token
    """

    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id

    def __call__(self, text, add_special_tokens=True):
        if text == "<start_of_turn>":
            return {"input_ids": [100]}  # single token for start of turn
        elif text == "<start_of_turn>model\n":
            return {"input_ids": [100, 101, 102]}  # multi-token response marker
        else:
            return {"input_ids": [10, 20, 30]}  # dummy token IDs


class DummyQwen25Processor:
    """
    Mimics the public API used by qwen2_5_collate_fn:
      • apply_chat_template(tokenize=bool)
      • __call__(text, images, padding, return_tensors="pt")
    """

    def __init__(self):
        self.call_counter = 0  # handy for assertions if you like
        self.tokenizer = DummyTokenizer(pad_token_id=0)  # Add tokenizer attribute

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

    def __init__(self):
        self.tokenizer = DummyTokenizer(pad_token_id=0)  # Add tokenizer attribute

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
    return (torch.zeros(3, 224, 224),)  # 1-tuple


def test_qwen25_collate_happy_path(collate_mod, patch_skipped, monkeypatch):
    # Patch the *imported symbol* inside collate_mod, not the module.
    monkeypatch.setattr(collate_mod, "process_vision_info", _fake_process_vision_info, raising=True)

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


class TestCreateLossMaskWithStartOfResponseToken:
    """Test cases for create_loss_mask_with_start_of_response_token function."""

    def test_no_start_of_response_token(self, collate_mod):
        """Test when start_of_response_token is None."""
        processor = DummyQwen25Processor()
        input_ids = torch.tensor([1, 2, 3, 4, 5])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token=None
        )

        # Should return all 1s (no masking) when no start token is provided
        expected = [1, 1, 1, 1, 1]
        assert result == expected

    def test_start_of_response_token_found_twice(self, collate_mod):
        """Test when start_of_response_token is found twice (normal case)."""
        processor = DummyQwen25Processor()
        # Create input_ids with start_of_turn token (100) appearing twice
        # [0, 100, 1, 2, 100, 101, 102, 3, 4]
        input_ids = torch.tensor([0, 100, 1, 2, 100, 101, 102, 3, 4])
        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        # First occurrence at index 1, second occurrence at index 4
        # Response starts at index 4 + 3 - 1 = 6 (after the response token sequence)
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert result == expected

    def test_start_of_response_token_found_only_once(self, collate_mod):
        """Test when start_of_response_token is found only once."""
        processor = DummyQwen25Processor()
        input_ids = torch.tensor([0, 100, 1, 2, 3])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        expected = [0, 1, 1, 1, 1]
        assert result == expected

    def test_start_of_response_token_not_found(self, collate_mod):
        """Test when start_of_response_token is not found in input_ids."""
        processor = DummyQwen25Processor()
        input_ids = torch.tensor([1, 2, 3, 4, 5])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        # Should return all 1s when token is not found
        expected = [1, 1, 1, 1, 1]
        assert result == expected

    def test_with_tensor_input(self, collate_mod):
        """Test that function works with tensor input."""
        processor = DummyQwen25Processor()
        input_ids = torch.tensor([0, 100, 1, 2, 100, 101, 102, 3, 4])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert result == expected

    def test_single_token_response_marker(self, collate_mod):
        """Test with single token response marker."""
        processor = DummyQwen25Processor()
        input_ids = torch.tensor([0, 100, 1, 2, 100, 3, 4])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>"
        )

        # Response starts at index 4 + 1 - 1 = 4
        expected = [0, 0, 0, 0, 1, 1, 1]
        assert result == expected

    def test_padding_tokens_masked(self, collate_mod):
        """Test that padding tokens are properly masked."""
        processor = DummyQwen25Processor()
        processor.tokenizer = DummyTokenizer(pad_token_id=0)
        # Input with padding tokens (0s) at the end
        input_ids = torch.tensor([1, 100, 2, 3, 100, 101, 102, 4, 0, 0])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        # Response starts at index 4 + 3 = 7, but padding tokens at end should be masked
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
        assert result == expected

    def test_padding_tokens_in_middle(self, collate_mod):
        """Test that padding tokens in the middle are also masked."""
        processor = DummyQwen25Processor()
        processor.tokenizer = DummyTokenizer(pad_token_id=0)
        # Input with padding tokens (0s) in the middle
        input_ids = torch.tensor([1, 100, 0, 3, 100, 101, 102, 4, 5])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        # Response starts at index 4 + 3 = 7, but padding token at index 2 should be masked
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 1]
        assert result == expected

    def test_custom_pad_token_id(self, collate_mod):
        """Test with custom pad token ID."""
        processor = DummyQwen25Processor()
        processor.tokenizer = DummyTokenizer(pad_token_id=999)
        # Input with custom padding tokens (999s) at the end
        input_ids = torch.tensor([1, 100, 2, 3, 100, 101, 102, 4, 999, 999])

        result = collate_mod.create_loss_mask_with_start_of_response_token(
            input_ids, processor, start_of_response_token="<start_of_turn>model\n"
        )

        # Response starts at index 4 + 3 = 7, but padding tokens at end should be masked
        expected = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
        assert result == expected


class TestCollateFunctionIntegration:
    """Test cases for the integration of loss mask creation in collate functions."""

    def test_qwen25_collate_fn_loss_mask_integration(self, collate_mod, patch_skipped, monkeypatch):
        """Test that qwen2_5_collate_fn properly creates loss masks."""
        # Patch the process_vision_info function
        monkeypatch.setattr(collate_mod, "process_vision_info", _fake_process_vision_info, raising=True)
        monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

        processor = DummyQwen25Processor()
        processor.tokenizer = DummyTokenizer(pad_token_id=0)
        examples = [{"conversation": "a"}, {"conversation": "b"}]

        batch = collate_mod.qwen2_5_collate_fn(examples, processor, start_of_response_token="<start_of_turn>model\n")

        # Verify loss_mask is present and properly shaped
        assert "loss_mask" in batch
        assert batch["loss_mask"].shape == batch["input_ids"].shape
        assert batch["loss_mask"].dtype == torch.float
        assert batch["loss_mask"].device == batch["input_ids"].device

    def test_default_collate_fn_loss_mask_integration(self, collate_mod, patch_skipped, monkeypatch):
        """Test that default_collate_fn properly creates loss masks."""
        monkeypatch.setattr(collate_mod, "HAVE_QWEN_VL_UTILS", True, raising=True)

        processor = DummyDefaultProcessor()
        examples = [{"conversation": "hello"}, {"conversation": "world"}]

        batch = collate_mod.default_collate_fn(examples, processor, start_of_response_token="<start_of_turn>model\n")

        # Verify loss_mask is present and properly shaped
        assert "loss_mask" in batch
        assert batch["loss_mask"].shape == batch["input_ids"].shape
        assert batch["loss_mask"].dtype == torch.float
        assert batch["loss_mask"].device == batch["input_ids"].device

    def test_inline_batch_processing_with_padding(self, collate_mod):
        """Test that the inline batch processing handles padding correctly."""
        processor = DummyQwen25Processor()
        processor.tokenizer = DummyTokenizer(pad_token_id=0)

        # Create a batch with input_ids that have padding tokens
        batch_input_ids = torch.tensor(
            [
                [1, 100, 2, 3, 100, 101, 102, 4, 0],  # With padding at end
                [5, 100, 6, 7, 100, 101, 102, 8, 9],  # No padding
            ]
        )

        # Test the inline list comprehension logic
        loss_masks = [
            collate_mod.create_loss_mask_with_start_of_response_token(input_ids, processor, "<start_of_turn>model\n")
            for input_ids in batch_input_ids
        ]
        result = torch.tensor(loss_masks, dtype=torch.float, device=batch_input_ids.device)

        expected = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 0],  # Padding token at end is masked
                [0, 0, 0, 0, 1, 1, 1, 1, 1],  # No padding tokens
            ],
            dtype=torch.float,
            device=batch_input_ids.device,
        )

        assert torch.equal(result, expected)

    def test_inline_batch_processing_mixed_sequences(self, collate_mod):
        """Test inline batch processing with mixed sequence types."""
        processor = DummyQwen25Processor()

        batch_input_ids = torch.tensor(
            [
                [0, 100, 1, 2, 100, 101, 102, 3, 4],  # Has valid response start
                [5, 6, 7, 8, 9, 10, 11, 12, 13],  # No start token
            ]
        )

        # Test the inline list comprehension logic
        loss_masks = [
            collate_mod.create_loss_mask_with_start_of_response_token(input_ids, processor, "<start_of_turn>model\n")
            for input_ids in batch_input_ids
        ]
        result = torch.tensor(loss_masks, dtype=torch.float, device=batch_input_ids.device)

        expected = torch.tensor(
            [
                [0, 0, 0, 0, 1, 1, 1, 1, 1],  # Valid response start at position 4
                [1, 1, 1, 1, 1, 1, 1, 1, 1],  # No masking (all 1s)
            ],
            dtype=torch.float,
            device=batch_input_ids.device,
        )

        assert torch.equal(result, expected)

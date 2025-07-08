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
# tests/test_utils.py
import types
import torch
import pytest

from nemo_automodel.datasets.vlm.utils import (
    PAD_TOKENS,
    extract_skipped_token_ids,
    json2token,
    process_text_batch,
)

class DummyTokenizer:
    def __init__(self, added_tokens_decoder):
        # maps id (int) -> token str (as in HF tokenizers)
        self.added_tokens_decoder = added_tokens_decoder


class DummyProcessor:
    """
    Mimics a HF-style processor.  Returns deterministic tensors so shape/dtype
    assertions are easy.
    """
    def __init__(self, tokenizer: DummyTokenizer | None = None):
        self.tokenizer = tokenizer or DummyTokenizer({})

    def __call__(
        self,
        *,
        text,
        images=None,
        padding=True,
        return_tensors="pt",
    ):
        seq_len = 4
        batch = {
            "input_ids": torch.arange(seq_len)
            .repeat(len(text), 1),
            "attention_mask": torch.ones(len(text), seq_len),
        }
        if images is not None:
            # emulate what a vision processor would return
            batch["pixel_values"] = torch.rand(
                len(images), 3, 224, 224, dtype=torch.float32
            )
        return batch


def _make_tokenizer_with_pads():
    """
    Build a tokenizer whose added_tokens_decoder contains
    both pad and non-pad tokens.
    """
    mapping = {
        1000: next(iter(PAD_TOKENS)),          # pad token – must be kept
        1001: "some_random_token",             # not a pad – should be ignored
        1002: list(PAD_TOKENS)[-1],            # another pad token
    }
    return DummyTokenizer(mapping)


def test_extract_skipped_token_ids_with_gemma_3n_tokens():
    """Test that extract_skipped_token_ids correctly identifies GEMMA_3N_TOKENS."""
    # Create a tokenizer with some GEMMA_3N_TOKENS
    mapping = {
        1000: "<image_soft_token>",   # from GEMMA_3N_TOKENS
        1001: "<audio_soft_token>",   # from GEMMA_3N_TOKENS
        1002: "<start_of_audio>",     # from GEMMA_3N_TOKENS
        1003: "some_random_token",    # not a pad token
        1004: "<end_of_image>",       # from GEMMA_3N_TOKENS
    }
    tokenizer = DummyTokenizer(mapping)
    
    ids = extract_skipped_token_ids(tokenizer)
    
    # Expected ids are the keys whose value is in PAD_TOKENS (which includes GEMMA_3N_TOKENS)
    expected = {k for k, v in mapping.items() if v in PAD_TOKENS}
    
    assert set(ids.tolist()) == expected
    # Should include all the GEMMA_3N_TOKENS we added
    assert 1000 in ids.tolist()  # <image_soft_token>
    assert 1001 in ids.tolist()  # <audio_soft_token>
    assert 1002 in ids.tolist()  # <start_of_audio>
    assert 1004 in ids.tolist()  # <end_of_image>
    # Should NOT include the random token
    assert 1003 not in ids.tolist()


@pytest.mark.parametrize("wrap_in_processor", [True, False])
def test_extract_skipped_token_ids(wrap_in_processor):
    tokenizer = _make_tokenizer_with_pads()
    processor = (
        DummyProcessor(tokenizer) if wrap_in_processor else tokenizer
    )

    ids = extract_skipped_token_ids(processor)

    # Expected ids are the keys whose value is in PAD_TOKENS
    expected = {k for k, v in tokenizer.added_tokens_decoder.items() if v in PAD_TOKENS}

    assert set(ids.tolist()) == expected
    # dtype should be IntTensor (torch.int32 by default on CPU)
    assert isinstance(ids, torch.Tensor)
    assert ids.dtype == torch.int32


def test_json2token_text_sequence_shortcut():
    assert json2token({"text_sequence": "hello"}) == "hello"


def test_json2token_list():
    assert json2token(["a", "b"]) == "a<sep/>b"


def test_json2token_simple_dict_sorted():
    d = {"b": "world", "a": "hello"}  # keys will sort as "b", "a"
    expected = "<s_b>world</s_b><s_a>hello</s_a>"
    assert json2token(d) == expected


def test_json2token_nested():
    nested = {"k1": ["v1", "v2"], "k2": {"inner": 7}}
    got = json2token(nested)
    # Exact ordering may depend on sort_json_key; simply assert fragments appear
    assert "<s_k1>" in got and "</s_k1>" in got
    assert "<s_k2>" in got and "</s_k2>" in got
    # list separator
    assert "<sep/>" in got
    # nested field
    assert "<s_inner>7</s_inner>" in got


def test_process_text_batch_text_only():
    processor = DummyProcessor()
    texts = ["foo", "bar"]
    batch = process_text_batch(processor, texts)

    assert set(batch) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape[0] == len(texts)


def test_process_text_batch_with_images():
    tokenizer = DummyTokenizer({})
    processor = DummyProcessor(tokenizer)
    texts = ["foo"]
    images = ["dummy_img"]  # content unused by DummyProcessor
    batch = process_text_batch(processor, texts, images)

    assert "pixel_values" in batch
    assert batch["pixel_values"].dtype == torch.bfloat16
    # unchanged fields still present
    assert "input_ids" in batch

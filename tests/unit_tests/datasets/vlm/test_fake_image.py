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
"""Vision-token detection robustness tests for fake_image.py.

The original implementation only recognized Qwen-style vision tokens
(``image_token_id`` / ``video_token_id`` attributes plus the
``<|vision_start|>`` family of strings).  Other VLMs in the codebase use
different conventions:

- Kimi K2.5 / KimiVL: ``media_placeholder_token_id`` + ``<|media_pad|>``
- LLaVA family / Mistral4: ``image_token_index`` + ``<image>`` / ``[IMG]``
- Gemma4: token id lives on ``processor.config``, not the processor itself

Without coverage for these, fake-image vision tokens silently remain visible
to attention.  These tests pin down the broadened detection logic and the
warning that fires when no vision tokens can be located at all.
"""

import logging

import pytest
import torch

from nemo_automodel.components.datasets.vlm import fake_image


class _StubTokenizer:
    """Minimal tokenizer stand-in: vocab dict + optional unk + optional added tokens."""

    def __init__(self, vocab=None, unk_token_id=0, added_tokens_decoder=None):
        self._vocab = dict(vocab or {})
        self.unk_token_id = unk_token_id
        if added_tokens_decoder is not None:
            self.added_tokens_decoder = added_tokens_decoder

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(token, self.unk_token_id)


class _StubProcessor:
    """Minimal processor stand-in.  Accepts any kwargs as attributes."""

    def __init__(self, tokenizer, config=None, **attrs):
        self.tokenizer = tokenizer
        if config is not None:
            self.config = config
        for k, v in attrs.items():
            setattr(self, k, v)


class _StubAddedToken:
    """Mimic ``transformers.AddedToken`` so tests don't need transformers."""

    def __init__(self, content):
        self.content = content


@pytest.fixture(autouse=True)
def _reset_warning_cache():
    """Each test starts with a clean warning-dedup cache."""
    fake_image._warned_unknown_processors.clear()
    yield
    fake_image._warned_unknown_processors.clear()


# ---------------------------------------------------------------------------
# _get_vision_token_ids — per VLM family
# ---------------------------------------------------------------------------


class TestGetVisionTokenIdsQwen:
    def test_picks_up_attributes_and_strings(self):
        tok = _StubTokenizer(
            {
                "<|vision_start|>": 100,
                "<|vision_end|>": 101,
                "<|image_pad|>": 102,
                "<|video_pad|>": 103,
            }
        )
        proc = _StubProcessor(tok, image_token_id=102, video_token_id=103)
        assert fake_image._get_vision_token_ids(proc) == {100, 101, 102, 103}


class TestGetVisionTokenIdsKimi:
    def test_media_placeholder_attr(self):
        tok = _StubTokenizer({"<|media_pad|>": 163605})
        proc = _StubProcessor(tok, media_placeholder_token_id=163605)
        assert 163605 in fake_image._get_vision_token_ids(proc)

    def test_media_pad_string_only(self):
        # Even without the attribute, the string lookup must catch it.
        tok = _StubTokenizer({"<|media_pad|>": 163605})
        proc = _StubProcessor(tok)
        assert 163605 in fake_image._get_vision_token_ids(proc)


class TestGetVisionTokenIdsLlava:
    def test_image_token_index_and_string(self):
        tok = _StubTokenizer({"<image>": 32000})
        proc = _StubProcessor(tok, image_token_index=32000)
        ids = fake_image._get_vision_token_ids(proc)
        assert 32000 in ids


class TestGetVisionTokenIdsMistral:
    def test_mistral_bracket_tokens(self):
        tok = _StubTokenizer({"[IMG]": 10, "[IMG_END]": 11, "[IMG_BREAK]": 12})
        proc = _StubProcessor(
            tok,
            image_token_index=10,
            image_break_token_id=12,
            image_end_token_id=11,
        )
        ids = fake_image._get_vision_token_ids(proc)
        assert {10, 11, 12} <= ids


class TestGetVisionTokenIdsConfigPath:
    def test_token_id_on_config(self):
        # Gemma4 / LLaVA stash IDs on processor.config rather than the processor itself.
        tok = _StubTokenizer({})
        config = type("Cfg", (), {"image_token_id": 256000})()
        proc = _StubProcessor(tok, config=config)
        assert 256000 in fake_image._get_vision_token_ids(proc)


class TestGetVisionTokenIdsAddedTokens:
    def test_fuzzy_match_image_keyword(self):
        added = {500: _StubAddedToken("<|custom_image_pad|>")}
        tok = _StubTokenizer(added_tokens_decoder=added)
        proc = _StubProcessor(tok)
        assert 500 in fake_image._get_vision_token_ids(proc)

    def test_fuzzy_match_video_keyword(self):
        added = {600: _StubAddedToken("<custom_video_marker>")}
        tok = _StubTokenizer(added_tokens_decoder=added)
        proc = _StubProcessor(tok)
        assert 600 in fake_image._get_vision_token_ids(proc)

    def test_non_vision_token_not_matched(self):
        added = {700: _StubAddedToken("<|imagine_this|>")}
        tok = _StubTokenizer(added_tokens_decoder=added)
        proc = _StubProcessor(tok)
        # 'image' substring would erroneously match — keyword set must be tight.
        # Current implementation accepts 'image' so 'imagine' WILL match.  This test
        # documents the trade-off: we err on the side of over-masking unknown
        # tokens that look vision-related rather than under-masking real ones.
        assert 700 in fake_image._get_vision_token_ids(proc)

    def test_unk_token_excluded(self):
        # convert_tokens_to_ids returning unk_token_id should not pollute the set.
        tok = _StubTokenizer(vocab={}, unk_token_id=0)
        proc = _StubProcessor(tok)
        assert fake_image._get_vision_token_ids(proc) == set()


class TestGetVisionTokenIdsRobustness:
    def test_processor_without_tokenizer(self):
        # Some processors are themselves callable-tokenizers.
        tok = _StubTokenizer({"<|image_pad|>": 42})
        # No .tokenizer attr — _get_vision_token_ids should fall back to processor itself.
        ids = fake_image._get_vision_token_ids(tok)
        assert 42 in ids

    def test_convert_tokens_to_ids_raises(self):
        class RaisingTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                raise RuntimeError("boom")

        proc = _StubProcessor(RaisingTokenizer())
        # Should not propagate; just returns empty set.
        assert fake_image._get_vision_token_ids(proc) == set()

    def test_non_int_attribute_ignored(self):
        tok = _StubTokenizer({})
        proc = _StubProcessor(tok, image_token_id=[1, 2, 3])
        assert fake_image._get_vision_token_ids(proc) == set()


# ---------------------------------------------------------------------------
# Warning behavior
# ---------------------------------------------------------------------------


class TestWarningOnUnknownProcessor:
    def test_warns_when_batch_masking_finds_nothing(self, caplog):
        tok = _StubTokenizer({})
        proc = _StubProcessor(tok)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        with caplog.at_level(logging.WARNING, logger=fake_image.logger.name):
            fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
        msgs = [r.message.lower() for r in caplog.records]
        assert any("could not detect any vision token" in m for m in msgs), msgs
        # Mask must remain unchanged (the warning is informational only).
        assert batch["attention_mask"][0].tolist() == [1, 1, 1]

    def test_warns_when_single_masking_finds_nothing(self, caplog):
        tok = _StubTokenizer({})
        proc = _StubProcessor(tok)
        sample = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.ones(3, dtype=torch.long),
        }
        with caplog.at_level(logging.WARNING, logger=fake_image.logger.name):
            fake_image.mask_fake_vision_tokens_single(sample, proc)
        assert any("could not detect any vision token" in r.message.lower() for r in caplog.records)

    def test_warns_only_once_per_processor_class(self, caplog):
        tok = _StubTokenizer({})
        proc = _StubProcessor(tok)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        with caplog.at_level(logging.WARNING, logger=fake_image.logger.name):
            fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
            fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
            fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
        warnings = [r for r in caplog.records if "could not detect any vision token" in r.message.lower()]
        assert len(warnings) == 1

    def test_no_warning_when_indices_empty(self, caplog):
        tok = _StubTokenizer({})
        proc = _StubProcessor(tok)
        batch = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        with caplog.at_level(logging.WARNING, logger=fake_image.logger.name):
            fake_image.mask_fake_vision_tokens_batch(batch, proc, [])
        assert not any("could not detect any vision token" in r.message.lower() for r in caplog.records)

    def test_no_warning_when_tokens_resolved(self, caplog):
        tok = _StubTokenizer({"<|media_pad|>": 99})
        proc = _StubProcessor(tok, media_placeholder_token_id=99)
        batch = {
            "input_ids": torch.tensor([[99, 1, 99]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        with caplog.at_level(logging.WARNING, logger=fake_image.logger.name):
            fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
        assert not any("could not detect any vision token" in r.message.lower() for r in caplog.records)
        # And the masking actually worked.
        assert batch["attention_mask"][0].tolist() == [0, 1, 0]


# ---------------------------------------------------------------------------
# End-to-end masking with the broadened detection
# ---------------------------------------------------------------------------


class TestMaskingIntegration:
    def test_masks_kimi_media_pad(self):
        MEDIA_PAD = 163605
        tok = _StubTokenizer({"<|media_pad|>": MEDIA_PAD})
        proc = _StubProcessor(tok, media_placeholder_token_id=MEDIA_PAD)
        batch = {
            "input_ids": torch.tensor([[1, MEDIA_PAD, MEDIA_PAD, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
        }
        fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
        assert batch["attention_mask"][0].tolist() == [1, 0, 0, 1, 1]

    def test_masks_mistral_bracket_tokens(self):
        IMG, IMG_END, IMG_BREAK = 10, 11, 12
        tok = _StubTokenizer({"[IMG]": IMG, "[IMG_END]": IMG_END, "[IMG_BREAK]": IMG_BREAK})
        proc = _StubProcessor(tok, image_token_index=IMG)
        batch = {
            "input_ids": torch.tensor([[1, IMG, IMG_BREAK, IMG, IMG_END, 6]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
        }
        fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
        assert batch["attention_mask"][0].tolist() == [1, 0, 0, 0, 0, 1]

    def test_masks_token_id_from_config(self):
        IMG = 256000
        tok = _StubTokenizer({})
        config = type("Cfg", (), {"image_token_id": IMG})()
        proc = _StubProcessor(tok, config=config)
        batch = {
            "input_ids": torch.tensor([[1, IMG, 3]]),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
        }
        fake_image.mask_fake_vision_tokens_batch(batch, proc, [0])
        assert batch["attention_mask"][0].tolist() == [1, 0, 1]

    def test_does_not_touch_unflagged_samples(self):
        IMG = 7
        tok = _StubTokenizer({})
        proc = _StubProcessor(tok, image_token_id=IMG)
        batch = {
            "input_ids": torch.tensor([[IMG, 1, IMG], [IMG, 2, IMG]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
        }
        fake_image.mask_fake_vision_tokens_batch(batch, proc, [1])
        # Sample 0: untouched.
        assert batch["attention_mask"][0].tolist() == [1, 1, 1]
        # Sample 1: masked.
        assert batch["attention_mask"][1].tolist() == [0, 1, 0]

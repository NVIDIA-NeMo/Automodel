# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from types import SimpleNamespace

from nemo_automodel.recipes.multimodal.finetune import _maybe_resize_bagel_vocab


class _FakeLanguageModel:
    def __init__(self, vocab_size: int):
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.resized_to = None

    def resize_token_embeddings(self, vocab_size: int) -> None:
        self.resized_to = vocab_size
        self.config.vocab_size = vocab_size


class _FakeBagelModel:
    def __init__(self, vocab_size: int):
        language_model = _FakeLanguageModel(vocab_size)
        self.model = SimpleNamespace(
            language_model=language_model,
            config=SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab_size)),
        )


def test_bagel_vocab_resize_does_not_shrink_padded_checkpoint_vocab():
    model = _FakeBagelModel(vocab_size=152064)

    _maybe_resize_bagel_vocab(model, tokenizer_vocab_size=151665, num_new_tokens=3)

    assert model.model.language_model.resized_to is None
    assert model.model.language_model.config.vocab_size == 152064
    assert model.model.config.text_config.vocab_size == 152064


def test_bagel_vocab_resize_still_grows_when_tokenizer_is_larger():
    model = _FakeBagelModel(vocab_size=152064)

    _maybe_resize_bagel_vocab(model, tokenizer_vocab_size=152100, num_new_tokens=4)

    assert model.model.language_model.resized_to == 152100
    assert model.model.language_model.config.vocab_size == 152100
    assert model.model.config.text_config.vocab_size == 152100

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.recipes.retrieval import train_bi_encoder as recipe


class _FakeTokenizer:
    pad_token = None
    eos_token = "</s>"


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()


class _FakeCollateConfig:
    def __init__(self):
        self.kwargs = None

    def instantiate(self, **kwargs):
        self.kwargs = kwargs
        return "collator"


def test_get_text_tokenizer_unwraps_processors(monkeypatch):
    monkeypatch.setattr(recipe, "ProcessorMixin", _FakeProcessor)
    processor = _FakeProcessor()

    assert recipe._get_text_tokenizer(processor) is processor.tokenizer
    assert recipe._get_text_tokenizer(processor.tokenizer) is processor.tokenizer


def test_instantiate_collate_fn_passes_processor_keyword_for_processors(monkeypatch):
    monkeypatch.setattr(recipe, "ProcessorMixin", _FakeProcessor)
    processor = _FakeProcessor()
    cfg = _FakeCollateConfig()

    assert recipe._instantiate_collate_fn(cfg, processor) == "collator"
    assert cfg.kwargs == {"processor": processor}


def test_instantiate_collate_fn_passes_tokenizer_keyword_for_tokenizers(monkeypatch):
    monkeypatch.setattr(recipe, "ProcessorMixin", _FakeProcessor)
    tokenizer = _FakeTokenizer()
    cfg = _FakeCollateConfig()

    assert recipe._instantiate_collate_fn(cfg, tokenizer) == "collator"
    assert cfg.kwargs == {"tokenizer": tokenizer}

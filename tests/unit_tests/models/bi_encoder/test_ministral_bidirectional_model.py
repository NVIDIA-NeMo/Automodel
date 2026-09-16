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

"""Unit tests for Ministral3 bidirectional encoder (retrieval / bi-encoder path)."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import (
    AutoModel,
    AutoProcessor,
    Mistral3Config,
    MistralConfig,
    PixtralImageProcessor,
    PixtralProcessor,
)
from transformers.tokenization_utils_tokenizers import TokenizersBackend

pytest.importorskip("transformers.models.ministral3", reason="Ministral3 not available in this transformers version")

from transformers.models.ministral3.modeling_ministral3 import Ministral3Model as HFMinistral3Model
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel._transformers.retrieval import (
    BiEncoderModel,
    CrossEncoderModel,
    _init_encoder_common,
    build_encoder_backbone,
    configure_encoder_metadata,
)
from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon, _maybe_save_custom_model_code
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Ministral3BidirectionalConfig,
    Ministral3BidirectionalModel,
    Mistral3BidirectionalConfig,
    Mistral3BidirectionalModel,
    Mistral3VLBidirectionalForSequenceClassification,
)
from nemo_automodel.components.models.ministral_bidirectional.processor import (
    Mistral3BiEncoderProcessor,
    PassageModality,
    load_image,
)
from nemo_automodel.recipes.retrieval.train_bi_encoder import _configure_sentence_transformer_export

# Over the default 5s budget on purpose: this module launches a fresh interpreter, which re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


def tiny_bidirectional_config() -> Ministral3BidirectionalConfig:
    cfg = Ministral3BidirectionalConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        pooling="avg",
        temperature=1.0,
    )
    cfg._attn_implementation = "eager"
    return cfg


class FakePixtralTokenizer(TokenizersBackend):
    def __init__(self):
        vocab = {
            "<unk>": 0,
            "<pad>": 1,
            "query": 2,
            "passage": 3,
            "literal": 4,
            "and": 5,
            "Image": 6,
            "doc": 7,
            "Text": 8,
            "Document": 9,
            "A": 10,
            "B": 11,
            "What": 12,
            "is": 13,
            "shown": 14,
            ":": 15,
            "?": 16,
            "[": 17,
            "]": 18,
            "IMG": 19,
            "INST": 20,
            "END": 21,
            "BREAK": 22,
            "_": 23,
        }
        backend = Tokenizer(WordPiece(vocab, unk_token="<unk>"))
        backend.pre_tokenizer = BertPreTokenizer()
        super().__init__(
            tokenizer_object=backend,
            unk_token="<unk>",
            pad_token="<pad>",
            additional_special_tokens=["[INST]", "[IMG]", "[IMG_BREAK]", "[IMG_END]"],
            model_max_length=99,
            padding_side="right",
        )
        self.calls = []

    def __call__(self, texts=None, **kwargs):
        recorded_texts = [texts] if isinstance(texts, str) else list(texts)
        self.calls.append({"texts": recorded_texts, "kwargs": kwargs.copy()})
        return super().__call__(texts, **kwargs)


class FakePixtralImageProcessor:
    model_input_names = ["pixel_values"]

    def __init__(self, size=None):
        self.calls = []
        self.size = size if size is not None else {"longest_edge": 1540}

    def fetch_images(self, images):
        """Return already-materialized PIL images for the processor test double."""
        return images

    def __call__(self, images, **kwargs):
        if not isinstance(images, list):
            images = [images]
        self.calls.append({"images": images, "kwargs": kwargs})
        image_sizes = [[image.height, image.width] for image in images]
        return {
            "pixel_values": torch.ones(len(images), 3, 4, 4),
            "image_sizes": image_sizes,
        }


@pytest.fixture
def pixtral_processor(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    return Mistral3BiEncoderProcessor(
        image_processor=FakePixtralImageProcessor(),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        q_max_length=8,
        p_max_length=64,
        pad_to_multiple_of=4,
        query_prefix="query:",
        passage_prefix="passage:",
    )


def test_ministral3_bidirectional_config_fields():
    cfg = Ministral3BidirectionalConfig(pooling="cls", temperature=0.5, vocab_size=100)
    assert cfg.pooling == "cls"
    assert isinstance(cfg.temperature, float)
    assert cfg.model_type == "ministral3_bidirec"


def test_mistral3_vlm_config_pooling_round_trips_and_exports():
    config = _tiny_mistral3_bidirectional_vlm_config()
    config.pooling = "cls"

    serialized = config.to_dict()
    reloaded = Mistral3BidirectionalConfig.from_dict(serialized)
    exported = Mistral3BidirectionalModel(config).get_hf_export_config()

    assert serialized["pooling"] == "cls"
    assert "_pooling" not in serialized
    assert serialized["text_config"]["pooling"] == "cls"
    assert reloaded.pooling == "cls"
    assert reloaded.text_config.pooling == "cls"
    assert exported.pooling == "cls"
    assert exported.text_config.pooling == "cls"


def test_mistral3_vlm_initializes_parent_once(monkeypatch):
    config = _tiny_mistral3_bidirectional_vlm_config()
    parent_init = Mistral3Model.__init__
    calls = []

    def tracking_init(model, model_config):
        calls.append(model_config)
        parent_init(model, model_config)

    monkeypatch.setattr(Mistral3Model, "__init__", tracking_init)

    model = Mistral3BidirectionalModel(config)

    assert calls == [config]
    assert isinstance(model.language_model, Ministral3BidirectionalModel)


def test_ministral3_biencoder_processor_loads_tokenizers_backend_directly(monkeypatch):
    expected = object()
    calls = []

    def fake_load_tokenizers_backend(model_id, *args, **kwargs):
        calls.append((model_id, args, kwargs))
        return expected

    monkeypatch.setattr(TokenizersBackend, "from_pretrained", fake_load_tokenizers_backend)

    actual = Mistral3BiEncoderProcessor._load_tokenizer_from_pretrained(
        "tokenizer",
        "mistralai/Ministral-3-3B-Instruct-2512",
        subfolder="tokenizer-assets",
        cache_dir="cache",
    )

    assert actual is expected
    assert calls == [
        (
            "mistralai/Ministral-3-3B-Instruct-2512",
            (),
            {
                "subfolder": "tokenizer-assets",
                "cache_dir": "cache",
                "fix_mistral_regex": True,
                "split_special_tokens": False,
            },
        )
    ]


def test_ministral3_biencoder_processor_validates_backend_without_importing_mistral_common(monkeypatch):
    backend = Tokenizer(WordLevel({"<unk>": 0, "[IMG]": 1}, unk_token="<unk>"))
    tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="<unk>", additional_special_tokens=["[IMG]"])
    original_resolver = Mistral3BiEncoderProcessor.get_possibly_dynamic_module

    def guarded_resolver(class_name):
        if class_name == "MistralCommonBackend":
            raise AssertionError("MistralCommonBackend must not be resolved")
        return original_resolver(class_name)

    monkeypatch.setattr(Mistral3BiEncoderProcessor, "get_possibly_dynamic_module", staticmethod(guarded_resolver))

    processor = Mistral3BiEncoderProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 56}),
        tokenizer=tokenizer,
        patch_size=14,
    )

    assert processor.tokenizer is tokenizer


def test_ministral3_biencoder_processor_rejects_injected_non_tokenizers_backend():
    processor = object.__new__(Mistral3BiEncoderProcessor)

    with pytest.raises(TypeError, match=r"from_pretrained\(\.\.\.\).+automatically"):
        processor.check_argument_for_proper_class("tokenizer", object())


@pytest.mark.parametrize(("keyword", "value"), [("backend", "mistral-common"), ("split_special_tokens", True)])
def test_ministral3_biencoder_processor_rejects_conflicting_tokenizer_policy(keyword, value):
    with pytest.raises(ValueError, match=keyword):
        Mistral3BiEncoderProcessor._load_tokenizer_from_pretrained(
            "tokenizer",
            "unused",
            **{keyword: value},
        )


def test_ministral3_biencoder_processor_processes_queries(pixtral_processor):
    output = pixtral_processor.process_queries(["What is shown?"])

    assert set(output) == {"input_ids", "attention_mask"}
    assert output["input_ids"].shape[0] == 1
    assert pixtral_processor.tokenizer.padding_side == "right"
    call = pixtral_processor.tokenizer.calls[-1]
    assert call["texts"] == ["query: What is shown?"]
    # ProcessorMixin owns final tensor conversion; tokenizer calls can request
    # lists first depending on the supported Transformers version.
    assert call["kwargs"]["return_tensors"] in (None, "pt")
    assert {key: value for key, value in call["kwargs"].items() if key != "return_tensors"} == {
        "padding": True,
        "pad_to_multiple_of": 4,
        "truncation": True,
        "max_length": 8,
        "padding_side": "right",
    }
    assert all(isinstance(value, torch.Tensor) for value in output.values())


def test_ministral3_biencoder_processor_processes_text_only_documents(pixtral_processor):
    output = pixtral_processor.process_documents({"images": ["", None], "texts": ["Document A", "Document B"]})

    assert set(output) == {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
    assert output["pixel_values"] is None
    assert output["image_sizes"] is None
    call = pixtral_processor.tokenizer.calls[-1]
    assert call["texts"] == ["passage: Document A", "passage: Document B"]
    assert call["kwargs"]["max_length"] == 64


def test_ministral3_biencoder_processor_processes_mixed_image_text_documents(pixtral_processor):
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    output = pixtral_processor.process_documents({"images": [image, ""], "texts": ["Image doc", "Text doc"]})

    assert set(output) == {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
    assert output["pixel_values"].shape == (1, 3, 4, 4)
    assert output["image_sizes"].tolist() == [[4, 4]]
    call = pixtral_processor.tokenizer.calls[-1]
    assert call["texts"] == ["passage: [IMG][IMG_END] Image doc", "passage: Text doc"]
    assert pixtral_processor.tokenizer.split_special_tokens is False


def test_ministral3_biencoder_processor_preserves_literal_control_text_without_activating_it(pixtral_processor):
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    tokenizer = pixtral_processor.tokenizer

    query = pixtral_processor.process_queries(["literal [INST] and [IMG]"])
    document = pixtral_processor.process_documents({"images": [image], "texts": ["literal [IMG_END]"]})

    query_ids = query["input_ids"][0].tolist()
    document_ids = document["input_ids"][0].tolist()
    assert tokenizer.convert_tokens_to_ids("[INST]") not in query_ids
    assert tokenizer.convert_tokens_to_ids("[IMG]") not in query_ids
    assert document_ids.count(tokenizer.convert_tokens_to_ids("[IMG]")) == 1
    assert document_ids.count(tokenizer.convert_tokens_to_ids("[IMG_END]")) == 1
    recorded_text = " ".join(text for call in tokenizer.calls for text in call["texts"])
    assert "literal [\u200cINST] and [\u200cIMG]" in recorded_text
    assert "literal [\u200cIMG_END]" in recorded_text


def test_ministral3_biencoder_processor_ownership_marker_is_token_neutral_and_preserves_unrelated_zwnj(
    pixtral_processor,
):
    tokenizer = pixtral_processor.tokenizer
    source = "query: literal [INST] and [IMG]"
    expected = tokenizer(
        source,
        add_special_tokens=False,
        split_special_tokens=True,
        return_tensors=None,
    )["input_ids"]

    marked = pixtral_processor._mark_user_text_ownership(source)
    actual = tokenizer(
        marked,
        add_special_tokens=False,
        split_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    assert actual == expected
    assert tokenizer.backend_tokenizer.normalizer.normalize_str("می\u200cروم") == "می\u200cروم"


def test_ministral3_biencoder_processor_saves_as_stock_pixtral_without_remote_code(tmp_path, monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 56}),
        tokenizer=FakePixtralTokenizer(),
        patch_size=14,
        chat_template=(
            "{%- macro render_content(content) -%}{{- content -}}{%- endmacro -%}"
            "{{- render_content(messages[0]['content']) -}}"
        ),
    )

    ownership_template = processor.chat_template
    assert ownership_template is not None
    assert "Derived from the Mistral" in ownership_template
    normalizer = str(processor.tokenizer.backend_tokenizer.normalizer)

    processor.save_pretrained(tmp_path)

    assert (tmp_path / "chat_template.jinja").read_text() == ownership_template
    processor_config = json.loads((tmp_path / "processor_config.json").read_text())
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    reloaded = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    rendered = reloaded.apply_chat_template(
        [
            {"role": "system", "content": "query:"},
            {"role": "user", "content": "literal [IMG]"},
        ],
        tokenize=False,
    )

    assert type(reloaded) is PixtralProcessor
    assert type(reloaded.tokenizer) is TokenizersBackend
    assert reloaded.tokenizer.split_special_tokens is False
    assert rendered == "query: literal [\u200cIMG]"
    assert "[INST]" not in rendered
    assert "auto_map" not in processor_config
    assert "auto_map" not in tokenizer_config
    assert not (tmp_path / "processor.py").exists()

    training_reload = Mistral3BiEncoderProcessor.from_pretrained(tmp_path)
    assert training_reload.chat_template == ownership_template
    assert str(training_reload.tokenizer.backend_tokenizer.normalizer) == normalizer
    source_with_zwnj = "literal [\u200cIMG]"
    marked_source = training_reload._mark_user_text_ownership(source_with_zwnj)
    assert training_reload.tokenizer.backend_tokenizer.normalizer.normalize_str(marked_source) == source_with_zwnj


@pytest.mark.parametrize("image_size", [(16, 16), (8, 16), (16, 8)])
def test_ministral3_exported_stock_processor_matches_training_image_preprocessing(
    tmp_path,
    monkeypatch,
    image_size,
):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 16}),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        p_max_length=128,
        padding=False,
        passage_prefix="passage:",
    )
    image = Image.new("RGB", image_size, (255, 0, 0))
    training = processor.process_documents(
        {"images": [image], "texts": ["literal [IMG]"]},
        return_tensors="pt",
    )

    processor.save_pretrained(tmp_path)
    exported = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    rendered = exported.apply_chat_template(
        [
            {"role": "system", "content": "passage:"},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "literal [IMG]"},
                ],
            },
        ],
        tokenize=False,
    )
    inference = exported(
        images=[image],
        text=[rendered],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=128,
    )

    for key in ("input_ids", "attention_mask", "pixel_values", "image_sizes"):
        torch.testing.assert_close(inference[key], training[key])


def test_ministral3_biencoder_processor_chat_template_matches_training_helpers(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=FakePixtralImageProcessor(),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        q_max_length=64,
        p_max_length=64,
        padding=False,
        query_prefix="query: [INST]",
        passage_prefix="passage: [INST]",
        chat_template='{{ bos_token }}[INST]{{ messages[0]["content"] }}[/INST]',
    )
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    query_rendered = processor.apply_chat_template(
        [
            {"role": "system", "content": "query: [INST]"},
            {"role": "user", "content": "literal [IMG]"},
        ],
        tokenize=False,
    )
    document_rendered = processor.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": "passage: [INST]"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "literal [IMG_END]"},
                    {"type": "image"},
                ],
            },
        ],
        tokenize=False,
    )

    assert query_rendered == "query: [INST] literal [\u200cIMG]"
    assert document_rendered == "passage: [INST] [IMG] literal [\u200cIMG_END]"
    assert (query_rendered + document_rendered).count("[INST]") == 2
    assert "[/INST]" not in query_rendered + document_rendered

    query_inference = processor(text=[query_rendered], padding=False, return_tensors="pt")
    document_inference = processor(images=[image], text=[document_rendered], padding=False, return_tensors="pt")
    query_training = processor.process_queries(["literal [IMG]"], padding=False)
    document_training = processor.process_documents(
        {"images": [image], "texts": ["literal [IMG_END]"]},
        padding=False,
    )

    torch.testing.assert_close(query_inference["input_ids"], query_training["input_ids"])
    torch.testing.assert_close(document_inference["input_ids"], document_training["input_ids"])


def test_ministral3_biencoder_processor_structured_image_ids_match_flat_reference(pixtral_processor):
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    tokenizer = pixtral_processor.tokenizer
    expected = tokenizer(
        ["passage: [IMG][IMG_END] Image doc"],
        split_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=64,
        pad_to_multiple_of=4,
        return_tensors="pt",
    )

    actual = pixtral_processor.process_documents({"images": [image], "texts": ["Image doc"]})

    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
    torch.testing.assert_close(actual["attention_mask"], expected["attention_mask"])


@pytest.mark.with_downloads
def test_ministral3_biencoder_processor_matches_real_checkpoint_pixtral_ids(tmp_path):
    model_id = "mistralai/Ministral-3-3B-Instruct-2512"
    image = Image.new("RGB", (56, 56), (255, 0, 0))
    processor = Mistral3BiEncoderProcessor.from_pretrained(
        model_id,
        p_max_length=128,
        padding=False,
    )
    reference_tokenizer = TokenizersBackend.from_pretrained(
        model_id,
        fix_mistral_regex=True,
        split_special_tokens=False,
    )
    reference = PixtralProcessor(
        image_processor=processor.image_processor,
        tokenizer=reference_tokenizer,
        patch_size=processor.patch_size,
        spatial_merge_size=processor.spatial_merge_size,
    )
    export_prompts = {}

    class _ExportModel:
        sentence_transformer_export_config = SimpleNamespace(input_mode="structured_multimodal")

        def configure_sentence_transformer_prompts(self, **kwargs):
            export_prompts.update(kwargs)

    _configure_sentence_transformer_export(
        _ExportModel(),
        SimpleNamespace(query_prefix="query:", passage_prefix="passage:", use_dataset_instruction=False),
    )
    assert export_prompts == {"query_prompt": "query:", "document_prompt": "passage:"}

    actual = processor.process_documents({"images": [image], "texts": ["ordinary document"]}, padding=False)
    expected = reference(
        images=[image],
        text=["passage: [IMG] ordinary document"],
        padding=False,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    torch.testing.assert_close(actual["input_ids"], expected["input_ids"])
    torch.testing.assert_close(actual["attention_mask"], expected["attention_mask"])

    literal = processor.process_documents(
        {"images": [image], "texts": ["literal [IMG] [IMG_END] می‌روم"]},
        padding=False,
    )
    literal_ids = literal["input_ids"][0].tolist()
    assert literal_ids.count(processor.tokenizer.convert_tokens_to_ids("[IMG]")) == 4
    assert literal_ids.count(processor.tokenizer.convert_tokens_to_ids("[IMG_END]")) == 1

    processor.save_pretrained(tmp_path)
    reloaded = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    rendered = reloaded.apply_chat_template(
        [
            {"role": "system", "content": export_prompts["document_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "literal [IMG] [IMG_END] می‌روم"},
                ],
            },
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    inference = reloaded(
        images=[image],
        text=[rendered],
        padding=False,
        truncation=True,
        max_length=4096,
        return_tensors="pt",
    )
    inference_ids = inference["input_ids"][0].tolist()

    assert type(reloaded) is PixtralProcessor
    assert type(reloaded.tokenizer) is TokenizersBackend
    assert inference_ids.count(reloaded.image_token_id) == 4
    assert inference_ids.count(reloaded.image_end_token_id) == 1
    torch.testing.assert_close(inference["input_ids"], literal["input_ids"])
    assert not (tmp_path / "processor.py").exists()


def test_ministral3_biencoder_processor_rejects_partial_image_structure(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=FakePixtralImageProcessor(),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        p_max_length=1,
        passage_prefix="passage:",
    )
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    with pytest.raises(ValueError, match="complete image token structure.+p_max_length"):
        processor.process_documents({"images": [image], "texts": ["Image doc"]})


def test_ministral3_biencoder_processor_rejects_remote_image_urls():
    with pytest.raises(ValueError, match="Remote image URLs are not supported"):
        load_image({"url": "https://example.com/image.png"})


def test_ministral3_biencoder_processor_canonicalizes_image_longest_edge(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    image_processor = FakePixtralImageProcessor()
    processor = Mistral3BiEncoderProcessor(
        image_processor=image_processor,
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        image_longest_edge=448,
        passage_prefix="passage:",
    )
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    processor.process_documents({"images": [image], "texts": ["Image doc"]})

    assert processor.image_longest_edge == 448
    assert image_processor.size == {"longest_edge": 448}
    assert "size" not in image_processor.calls[-1]["kwargs"]


def test_ministral3_biencoder_processor_preserves_nested_longest_edge(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    image_processor = FakePixtralImageProcessor(size={"longest_edge": 1540})

    processor = Mistral3BiEncoderProcessor(
        image_processor=image_processor,
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
    )

    assert processor.image_longest_edge == 1540
    assert image_processor.size == {"longest_edge": 1540}


def test_ministral3_biencoder_processor_attribute_and_call_overrides(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    image_processor = FakePixtralImageProcessor()
    processor = Mistral3BiEncoderProcessor(
        image_processor=image_processor,
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
    )
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    processor.image_longest_edge = 336
    processor.process_documents(
        {"images": [image], "texts": ["Image doc"]},
        size={"longest_edge": 224},
    )

    assert image_processor.size == {"longest_edge": 336}
    assert processor.image_longest_edge == 336
    assert image_processor.calls[-1]["kwargs"]["size"] == {"longest_edge": 224}


@pytest.mark.parametrize(
    ("processor_config", "expected_longest_edge"),
    [
        ({}, 1540),
        ({"image_longest_edge": 448}, 448),
    ],
)
def test_ministral3_biencoder_processor_serializes_only_nested_longest_edge(
    monkeypatch,
    processor_config,
    expected_longest_edge,
):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    image_processor = PixtralImageProcessor(size={"longest_edge": 1540})
    processor = Mistral3BiEncoderProcessor.from_args_and_dict(
        [image_processor, FakePixtralTokenizer()],
        processor_config,
    )

    serialized = json.loads(processor.to_json_string())

    assert "image_longest_edge" not in serialized
    assert serialized["image_processor"]["size"] == {"longest_edge": expected_longest_edge}


def test_ministral3_biencoder_processor_merges_biencoder_batch(pixtral_processor):
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    features = [
        {"question": "Question 0", "doc_text": ["Doc 0", "Doc 1"], "doc_image": [image, ""]},
        {"question": "Question 1", "doc_text": ["Doc 2", "Doc 3"], "doc_image": ["", image]},
    ]

    output = pixtral_processor.process_queries_documents_biencoder(features)

    assert set(output) == {
        "q_input_ids",
        "q_attention_mask",
        "d_input_ids",
        "d_attention_mask",
        "d_pixel_values",
        "d_image_sizes",
        "passage_modality",
        "labels",
    }
    assert output["q_input_ids"].shape[0] == 2
    assert output["d_input_ids"].shape[0] == 4
    assert output["d_pixel_values"].shape[0] == 2
    assert output["d_image_sizes"].tolist() == [[4, 4], [4, 4]]
    assert output["passage_modality"].tolist() == [
        PassageModality.IMAGE_TEXT,
        PassageModality.TEXT_ONLY,
        PassageModality.TEXT_ONLY,
        PassageModality.IMAGE_TEXT,
    ]
    assert torch.equal(output["labels"], torch.zeros(2, dtype=torch.long))


def test_ministral3_crossencoder_processor_collates_flattened_mixed_pairs(pixtral_processor):
    red_image = Image.new("RGB", (4, 4), (255, 0, 0))
    blue_image = Image.new("RGB", (4, 4), (0, 0, 255))
    pixtral_processor.rerank_max_length = 64
    pixtral_processor.use_prompt_template = True
    features = [
        {"question": "Question 0", "doc_text": "Image doc", "doc_image": red_image, "num_labels": 2},
        {"question": "Question 0", "doc_text": "Text doc", "doc_image": "", "num_labels": 2},
        {"question": "Question 1", "doc_text": "Another text doc", "doc_image": None, "num_labels": 2},
        {"question": "Question 1", "doc_text": "Another image doc", "doc_image": blue_image, "num_labels": 2},
    ]

    output = pixtral_processor.process_queries_documents_crossencoder(features)

    assert output["input_ids"].shape[0] == 4
    assert output["pixel_values"].shape == (2, 3, 4, 4)
    assert output["image_sizes"].tolist() == [[4, 4], [4, 4]]
    assert torch.equal(output["labels"], torch.zeros(2, dtype=torch.long))

    image_call = pixtral_processor.image_processor.calls[-1]
    assert image_call["images"] == [red_image, blue_image]
    texts = pixtral_processor.tokenizer.calls[-1]["texts"]
    assert len(texts) == 4
    assert texts[0].endswith("query: Question 0\n\npassage: Image doc")
    assert texts[1] == "query: Question 0\n\npassage: Text doc"
    assert texts[2] == "query: Question 1\n\npassage: Another text doc"
    assert texts[3].endswith("query: Question 1\n\npassage: Another image doc")
    assert [text.startswith("[IMG]") for text in texts] == [True, False, False, True]


def test_ministral3_crossencoder_processor_marks_user_owned_reserved_tokens(pixtral_processor):
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    pixtral_processor.rerank_max_length = 64
    pixtral_processor.use_prompt_template = True
    features = [
        {
            "question": "Question [IMG]",
            "doc_text": "Passage [IMG] [IMG_BREAK] [IMG_END]",
            "doc_image": image,
        }
    ]

    pixtral_processor.process_queries_documents_crossencoder(features)

    text = pixtral_processor.tokenizer.calls[-1]["texts"][0]
    assert text.startswith("[IMG]")
    assert text.count("[IMG]") == 1
    assert "query: Question [\u200cIMG]" in text
    assert "passage: Passage [\u200cIMG] [\u200cIMG_BREAK] [\u200cIMG_END]" in text


def test_ministral3_biencoder_processor_returns_numpy_batch(pixtral_processor):
    features = [
        {"question": "Question 0", "doc_text": ["Doc 0", "Doc 1"], "doc_image": ["", ""]},
        {"question": "Question 1", "doc_text": ["Doc 2", "Doc 3"], "doc_image": ["", ""]},
    ]

    output = pixtral_processor.process_queries_documents_biencoder(features, return_tensors="np")

    assert isinstance(output["q_input_ids"], np.ndarray)
    assert isinstance(output["d_input_ids"], np.ndarray)
    assert isinstance(output["labels"], np.ndarray)
    np.testing.assert_array_equal(output["labels"], np.zeros(2, dtype=np.int64))


def test_legacy_ministral_checkpoint_loads_in_fresh_process(tmp_path):
    model_dir = tmp_path / "legacy_ministral"
    Ministral3BidirectionalModel(tiny_bidirectional_config()).save_pretrained(model_dir)

    code = (
        "from nemo_automodel._transformers.retrieval import build_encoder_backbone; "
        f"model = build_encoder_backbone(r'{model_dir}', 'embedding', pooling='avg'); "
        "assert type(model).__name__ == 'Ministral3BidirectionalModel'; "
        "assert model.config.model_type == 'ministral3_bidirec'"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("is_causal", [False, True])
def test_ministral3_bidirectional_model_init_and_mask(is_causal):
    cfg = tiny_bidirectional_config()
    cfg.is_causal = is_causal
    model = Ministral3BidirectionalModel(cfg)
    model.eval()

    assert all(getattr(layer.self_attn, "is_causal", None) is is_causal for layer in model.layers)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 3))
    mask = torch.tensor([[1, 1, 0]])
    out = model(input_ids=input_ids, attention_mask=mask)
    assert out.last_hidden_state is not None and out.last_hidden_state.shape == (1, 3, cfg.hidden_size)

    out_no_mask = model(input_ids=input_ids)
    assert out_no_mask.last_hidden_state is not None
    assert out_no_mask.last_hidden_state.shape == (1, 3, cfg.hidden_size)


def test_ministral3_bidirectional_attention_symmetric():
    """Changing a later token should affect earlier positions (non-causal)."""
    cfg = tiny_bidirectional_config()
    model = Ministral3BidirectionalModel(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    attn = torch.ones(1, 4, dtype=torch.long)

    with torch.no_grad():
        out_base = model(input_ids=input_ids, attention_mask=attn).last_hidden_state.clone()
        modified = input_ids.clone()
        modified[0, -1] = (input_ids[0, -1] + 1) % cfg.vocab_size
        out_modified = model(input_ids=modified, attention_mask=attn).last_hidden_state

    assert not torch.allclose(out_base[0, 0], out_modified[0, 0], atol=1e-6), (
        "Bidirectional Ministral3: changing last token should affect first token hidden state"
    )


@pytest.mark.parametrize("is_causal", [False, True])
def test_ministral3_dual_mode_matches_hf_parent(is_causal):
    torch.manual_seed(42)
    config = tiny_bidirectional_config()
    config.is_causal = is_causal
    actual_model = Ministral3BidirectionalModel(config).eval()
    reference_model = HFMinistral3Model(config).eval()
    reference_model.load_state_dict(actual_model.state_dict())
    input_ids = torch.randint(0, config.vocab_size, (1, 4))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        actual = actual_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        reference = reference_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_ministral3_causal_attention_blocks_future_token_influence():
    cfg = tiny_bidirectional_config()
    cfg.is_causal = True
    model = Ministral3BidirectionalModel(cfg).eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    modified = input_ids.clone()
    modified[0, -1] = (modified[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        original = model(input_ids=input_ids).last_hidden_state
        changed = model(input_ids=modified).last_hidden_state
    torch.testing.assert_close(original[0, 0], changed[0, 0])


def test_ministral3_bidirectional_forward_paths():
    cfg = tiny_bidirectional_config()
    model = Ministral3BidirectionalModel(cfg)
    bsz, seqlen = 2, 3
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen))
    attn = torch.ones(bsz, seqlen, dtype=torch.long)

    with pytest.raises(ValueError):
        model(input_ids=None, inputs_embeds=None)

    with pytest.raises((ValueError, TypeError, AttributeError)):
        model(input_ids=input_ids, attention_mask=attn, past_key_values=123)

    model.eval()
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    assert hasattr(out, "last_hidden_state")
    assert out.past_key_values is not None


# --- BiEncoderModel.build + registry (mirrors Llama bidirectional build tests) ---


class FakeLM(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()

        class Cfg:
            def __init__(self):
                self.hidden_size = hidden

        self.config = Cfg()
        self.linear = nn.Linear(hidden, hidden)
        self.saved = []

    def save_pretrained(self, out_dir):
        self.saved.append(out_dir)


def test_encoder_build_legacy_ministral_registry_path(tmp_path, monkeypatch):
    class FakeBidirectionalModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(hidden=16)

    monkeypatch.setattr(
        ModelRegistry,
        "model_arch_name_to_cls",
        {"Ministral3BidirectionalModel": FakeBidirectionalModel},
    )

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "ministral3_bidirec"}))

    model = BiEncoderModel.build(
        model_name_or_path=str(model_dir),
        pooling="avg",
        l2_normalize=True,
    )
    assert isinstance(model, BiEncoderModel)
    outdir = tmp_path / "save1"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outdir))
    assert any("save1" in p for p in model.model.saved)


def test_ministral_dispatch_preserves_standard_and_legacy_mappings():
    """Verify standard and legacy Ministral dispatch mappings remain stable."""
    from nemo_automodel._transformers.retrieval import SUPPORTED_BACKBONES

    assert "ministral3" not in SUPPORTED_BACKBONES
    assert SUPPORTED_BACKBONES["ministral3_bidirec"]["embedding"] == "Ministral3BidirectionalModel"
    assert SUPPORTED_BACKBONES["mistral3"]["embedding"] == "Mistral3BidirectionalModel"
    assert SUPPORTED_BACKBONES["mistral3_bidirec"]["embedding"] == "Mistral3BidirectionalModel"


def test_configure_encoder_metadata_sets_auto_map_for_ministral_retrieval():
    FakeRetrievalModel = type("Ministral3BidirectionalModel", (), {})
    fake = FakeRetrievalModel()
    FakeCfg = type("Ministral3BidirectionalConfig", (), {})
    fake.config = FakeCfg()

    configure_encoder_metadata(fake, fake.config)

    assert fake.config.architectures == ["Ministral3BidirectionalModel"]
    assert "auto_map" in vars(fake.config)
    assert "AutoModel" in fake.config.auto_map


def test_init_encoder_common_name_or_path_ministral_retrieval(tmp_path):
    """Registered retrieval architectures advertise their source package for checkpoint export.

    Must use the real ``Ministral3BidirectionalModel`` class: a class defined in this test
    file would resolve to this test file, not ``.../ministral_bidirectional/model.py``.
    """
    cfg = tiny_bidirectional_config()
    backbone = Ministral3BidirectionalModel(cfg)

    encoder = nn.Module()
    _init_encoder_common(encoder, backbone)

    assert encoder.name_or_path.endswith("ministral_bidirectional")

    _maybe_save_custom_model_code(encoder.name_or_path, str(tmp_path))

    assert (tmp_path / "model.py").is_file()
    assert (tmp_path / "processor.py").is_file()


def _tiny_mistral3_bidirectional_vlm_config() -> Mistral3BidirectionalConfig:
    """Build a tiny VL retrieval config for stock checkpoint export tests."""
    config = Mistral3BidirectionalConfig(
        text_config={
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "sliding_window": None,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 16,
            "patch_size": 4,
            "num_channels": 3,
        },
        spatial_merge_size=1,
        pooling="avg",
    )
    config.text_config.auto_map = {"AutoModel": "legacy_model.LegacyTextModel"}
    config.vision_config.auto_map = {"AutoModel": "legacy_model.LegacyVisionModel"}
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    return config


def test_mistral3_vlm_causal_export_scopes_attention_policy_to_text_tower():
    """Portable VLM export must not apply text causality to the vision tower."""
    config = _tiny_mistral3_bidirectional_vlm_config()
    assert hasattr(config.text_config, "is_causal")
    config.text_config.is_causal = True

    export_config = Mistral3BidirectionalModel(config).get_hf_export_config()
    serialized_config = export_config.to_dict()

    assert "is_causal" not in serialized_config
    assert serialized_config["text_config"]["is_causal"] is True
    assert serialized_config["vision_config"].get("is_causal", False) is False


@pytest.mark.parametrize("is_causal", [False, True])
def test_mistral3_vlm_saves_as_stock_bidirectional_model_without_remote_code(tmp_path, is_causal):
    config = _tiny_mistral3_bidirectional_vlm_config()
    config.text_config.is_causal = is_causal
    config.task_instructions = {"query": "stale query: "}
    training_config = config.to_dict()
    assert training_config["model_type"] == "mistral3_bidirec"
    assert training_config["text_config"]["model_type"] == "ministral3_bidirec"

    backbone = Mistral3BidirectionalModel(config).eval()
    encoder = BiEncoderModel(backbone, pooling="avg", l2_normalize=True)
    assert encoder.config.architectures == ["Mistral3BidirectionalModel"]
    encoder.disable_sentence_transformer_export()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)

    image_input_ids = torch.tensor([[config.image_token_id] * 16 + [1, 2]])
    image_attention_mask = torch.ones_like(image_input_ids)
    pixel_values = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) / 255
    image_sizes = torch.tensor([[16, 16]])

    with torch.no_grad():
        expected = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        expected_multimodal = backbone(
            input_ids=image_input_ids,
            attention_mask=image_attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        ).last_hidden_state

    legacy_source = tmp_path / "legacy_source"
    legacy_source.mkdir()
    (legacy_source / "model.py").write_text("class LegacyModel: pass\n")
    (legacy_source / "processor.py").write_text("class LegacyProcessor: pass\n")
    encoder.config.name_or_path = str(legacy_source)
    model_state = SimpleNamespace(model=[encoder])
    checkpointer = SimpleNamespace(config=SimpleNamespace())
    original_model_path = Checkpointer._get_original_model_path(checkpointer, model_state)
    assert original_model_path == str(legacy_source)

    custom_code_dir = tmp_path / "custom_code"
    custom_code_dir.mkdir()
    ConsolidatedHFAddon().pre_save(
        model_state=model_state,
        hf_metadata_dir=str(custom_code_dir),
        fqn_to_file_index_mapping={},
        original_model_path=original_model_path,
    )
    assert not list(custom_code_dir.rglob("*.py"))
    assert (custom_code_dir / "config.json").is_file()

    encoder.save_pretrained(str(tmp_path))

    saved_config = json.loads((tmp_path / "config.json").read_text())
    reloaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=False).eval()
    training_reload = Mistral3BidirectionalModel.from_pretrained(tmp_path).eval()

    assert type(reloaded) is Mistral3Model
    assert saved_config["model_type"] == "mistral3"
    assert saved_config["architectures"] == ["Mistral3Model"]
    assert "is_causal" not in saved_config
    assert saved_config["pooling"] == "avg"
    assert saved_config["text_config"]["is_causal"] is is_causal
    assert "auto_map" not in saved_config["text_config"]
    assert "auto_map" not in saved_config["vision_config"]
    assert "auto_map" not in saved_config
    assert "task_instructions" not in saved_config
    assert not list(tmp_path.glob("*.py"))
    assert encoder.name_or_path is None
    assert set(reloaded.state_dict()) == set(backbone.state_dict())
    for key, tensor in backbone.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], tensor)
        torch.testing.assert_close(training_reload.state_dict()[key], tensor)

    with torch.no_grad():
        actual = reloaded(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        actual_multimodal = reloaded(
            input_ids=image_input_ids,
            attention_mask=image_attention_mask,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
        ).last_hidden_state
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_multimodal, expected_multimodal)

    changed_ids = input_ids.clone()
    changed_ids[0, -1] = 5
    with torch.no_grad():
        changed = reloaded(input_ids=changed_ids, attention_mask=attention_mask).last_hidden_state
    if is_causal:
        torch.testing.assert_close(actual[0, 0], changed[0, 0])
    else:
        assert not torch.allclose(actual[0, 0], changed[0, 0], atol=1e-6)


def test_mistral3_vlm_portable_export_requires_processor(tmp_path):
    encoder = BiEncoderModel(
        Mistral3BidirectionalModel(_tiny_mistral3_bidirectional_vlm_config()).eval(),
        pooling="avg",
        l2_normalize=True,
    )

    with pytest.raises(ValueError, match="requires a processor"):
        encoder.save_pretrained(tmp_path)
    assert not any(tmp_path.iterdir())
    with pytest.raises(ValueError, match="requires a processor"):
        encoder._get_consolidated_hf_metadata_exporter(tokenizer=None, original_model_path=None)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("tokenizer", None, "processor with a tokenizer"),
        ("image_processor", None, "processor with an image processor"),
        ("chat_template", "", "processor with a chat template"),
    ],
)
def test_mistral3_vlm_portable_export_rejects_incomplete_processor(
    tmp_path,
    attribute,
    value,
    message,
):
    encoder = BiEncoderModel(
        Mistral3BidirectionalModel(_tiny_mistral3_bidirectional_vlm_config()).eval(),
        pooling="avg",
        l2_normalize=True,
    )
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(model_max_length=64),
        image_processor=object(),
        chat_template="{{ messages }}",
        model_max_length=64,
    )
    setattr(processor, attribute, value)

    with pytest.raises(ValueError, match=message):
        encoder.save_pretrained(tmp_path, tokenizer=processor)
    with pytest.raises(ValueError, match=message):
        encoder._get_consolidated_hf_metadata_exporter(tokenizer=processor, original_model_path=None)


def test_mistral3_vlm_portable_export_rejects_non_processor_before_writing(tmp_path):
    encoder = BiEncoderModel(
        Mistral3BidirectionalModel(_tiny_mistral3_bidirectional_vlm_config()).eval(),
        pooling="avg",
        l2_normalize=True,
    )
    processor_like = SimpleNamespace(
        tokenizer=SimpleNamespace(model_max_length=64),
        image_processor=object(),
        chat_template="{{ messages }}",
        model_max_length=64,
        save_pretrained=lambda *_args, **_kwargs: None,
        apply_chat_template=lambda *_args, **_kwargs: "",
    )

    with pytest.raises(TypeError, match="ProcessorMixin"):
        encoder.save_pretrained(tmp_path, tokenizer=processor_like)

    assert not any(tmp_path.iterdir())


def test_mistral3_vlm_portable_export_rejects_processor_subclass_without_explicit_opt_in(tmp_path, monkeypatch):
    class _NonPortableProcessor(Mistral3BiEncoderProcessor):
        def process_documents(self, *args, **kwargs):
            raise AssertionError("custom preprocessing is not representable by stock PixtralProcessor")

    monkeypatch.setattr(_NonPortableProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = _NonPortableProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 16}),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
    )
    encoder = BiEncoderModel(
        Mistral3BidirectionalModel(_tiny_mistral3_bidirectional_vlm_config()).eval(),
        pooling="avg",
        l2_normalize=True,
    )

    with pytest.raises(TypeError, match="explicitly opt in to stock processor export"):
        encoder.save_pretrained(tmp_path, tokenizer=processor)

    assert not any(tmp_path.iterdir())


def test_mistral3_vlm_portable_export_rejects_model_subclass_without_explicit_opt_in(tmp_path):
    class _NonPortableModel(Mistral3BidirectionalModel):
        pass

    encoder = BiEncoderModel(
        _NonPortableModel(_tiny_mistral3_bidirectional_vlm_config()).eval(),
        pooling="avg",
        l2_normalize=True,
    )

    with pytest.raises(TypeError, match="explicitly opt in to stock model export"):
        encoder.save_pretrained(tmp_path, tokenizer=MagicMock())

    assert not any(tmp_path.iterdir())


def test_mistral3_vlm_portable_export_allows_subclasses_to_explicitly_reassert_compatibility(monkeypatch):
    class _PortableProcessor(Mistral3BiEncoderProcessor):
        _export_as_stock_processor = True

    class _PortableModel(Mistral3BidirectionalModel):
        _export_as_stock_model = True

    monkeypatch.setattr(_PortableProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = _PortableProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 16}),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
    )
    model = _PortableModel(_tiny_mistral3_bidirectional_vlm_config())

    assert type(processor.get_hf_export_processor()) is PixtralProcessor
    assert model.get_hf_export_config().model_type == "mistral3"


def test_mistral3_vlm_exports_sentence_transformers_checkpoint(tmp_path, monkeypatch):
    from sentence_transformers import SentenceTransformer

    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 16}),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        q_max_length=64,
        p_max_length=64,
        padding=True,
        query_prefix="query:",
        passage_prefix="passage:",
    )
    config = _tiny_mistral3_bidirectional_vlm_config()
    assert hasattr(config, "image_token_id")
    config.image_token_id = processor.tokenizer.convert_tokens_to_ids("[IMG]")
    encoder = BiEncoderModel(
        Mistral3BidirectionalModel(config).eval(),
        pooling="avg",
        l2_normalize=True,
    ).eval()
    encoder.configure_sentence_transformer_prompts(
        query_prompt="query:",
        document_prompt="passage:",
    )
    assert encoder.sentence_transformer_export_config is not None
    assert encoder.sentence_transformer_export_config.input_mode == "structured_multimodal"
    image = Image.new("RGB", (16, 16), (255, 0, 0))
    training_batch = processor.process_documents(
        {"images": [image], "texts": ["Image doc"]},
        return_tensors="pt",
    )
    with torch.no_grad():
        expected = encoder(training_batch)

    encoder.save_pretrained(tmp_path, tokenizer=processor)

    assert json.loads((tmp_path / "modules.json").read_text()) == [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.transformer.Transformer"},
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.sentence_transformer.modules.pooling.Pooling",
        },
        {
            "idx": 2,
            "name": "2",
            "path": "2_Normalize",
            "type": "sentence_transformers.sentence_transformer.modules.normalize.Normalize",
        },
    ]
    sentence_transformer_config = json.loads((tmp_path / "config_sentence_transformers.json").read_text())
    assert sentence_transformer_config["prompts"] == {
        "query": "query:",
        "document": "passage:",
    }
    transformer_config = json.loads((tmp_path / "sentence_bert_config.json").read_text())
    assert transformer_config["max_seq_length"] == 64
    assert transformer_config["do_lower_case"] is False
    assert transformer_config["module_output_name"] == "token_embeddings"
    assert set(transformer_config["modality_config"]) == {"text", "image", "message"}
    assert transformer_config["modality_config"]["message"]["format"] == "structured"
    pooling_config = json.loads((tmp_path / "1_Pooling" / "config.json").read_text())
    assert pooling_config["pooling_mode"] == "mean"
    assert pooling_config["include_prompt"] is True
    for asset in (
        "processor_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        assert (tmp_path / asset).is_file(), asset
    assert not list(tmp_path.rglob("*.py"))

    reloaded_processor = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    assert type(reloaded_processor) is PixtralProcessor
    sentence_transformer = SentenceTransformer(str(tmp_path), device="cpu", trust_remote_code=False)
    assert sentence_transformer.prompts == {"query": "query:", "document": "passage:"}
    assert [type(module).__name__ for module in sentence_transformer] == ["Transformer", "Pooling", "Normalize"]
    assert type(sentence_transformer[0].processor) is PixtralProcessor
    assert sentence_transformer[1].pooling_mode == "mean"
    assert type(sentence_transformer[0].model) is Mistral3Model

    query_batch = processor.process_queries(["Text query"], return_tensors="pt")
    with torch.no_grad():
        expected_query = encoder(query_batch)
    actual_query = sentence_transformer.encode_query(["Text query"], convert_to_tensor=True)
    torch.testing.assert_close(actual_query, expected_query)

    text_document_batch = processor.process_documents(
        {"images": [None], "texts": ["Text doc"]},
        return_tensors="pt",
    )
    with torch.no_grad():
        expected_text_document = encoder(text_document_batch)
    actual_text_document = sentence_transformer.encode_document(["Text doc"], convert_to_tensor=True)
    torch.testing.assert_close(actual_text_document, expected_text_document)

    image_only_batch = processor.process_documents(
        {"images": [image], "texts": [None]},
        return_tensors="pt",
    )
    with torch.no_grad():
        expected_image_only = encoder(image_only_batch)
    actual_image_only = sentence_transformer.encode_document([image], convert_to_tensor=True)
    torch.testing.assert_close(actual_image_only, expected_image_only)

    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Image doc"},
            ],
        }
    ]
    actual = sentence_transformer.encode_document([message], convert_to_tensor=True)

    assert actual.shape == expected.shape == (1, 16)
    torch.testing.assert_close(torch.linalg.vector_norm(actual, dim=-1), torch.ones(1))
    torch.testing.assert_close(actual, expected)

    consolidated_dir = tmp_path / "consolidated"
    encoder.model.save_pretrained(consolidated_dir)
    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[encoder]),
        hf_metadata_dir=str(consolidated_dir),
        tokenizer=processor,
        fqn_to_file_index_mapping={},
        original_model_path=None,
    )
    consolidated = SentenceTransformer(
        str(consolidated_dir),
        device="cpu",
        trust_remote_code=False,
    )
    consolidated_actual = consolidated.encode_document([message], convert_to_tensor=True)
    assert type(consolidated[0].model) is Mistral3Model
    assert type(consolidated[0].processor) is PixtralProcessor
    assert not list(consolidated_dir.rglob("*.py"))
    torch.testing.assert_close(consolidated_actual, expected)


def test_mistral3_vlm_liger_patch_targets_text_tower():
    """Only the compatible text kernels are enabled for the bidirectional VLM."""
    language_model = object()
    model = SimpleNamespace(language_model=language_model)
    liger_kernel_transformers = SimpleNamespace(apply_liger_kernel_to_mistral=MagicMock())

    Mistral3BidirectionalModel._nemo_apply_liger_kernel(model, liger_kernel_transformers)

    liger_kernel_transformers.apply_liger_kernel_to_mistral.assert_called_once_with(
        model=language_model,
        rope=False,
        cross_entropy=False,
        fused_linear_cross_entropy=False,
    )


def test_mistral3_vlm_runs_dummy_vision_inside_model_forward(monkeypatch):
    events = []
    embedding = nn.Embedding(16, 4)

    class FakeVisionTower(nn.Module):
        patch_size = 2

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(
            self,
            pixel_values: torch.Tensor,
            image_sizes: torch.Tensor,
            output_hidden_states: bool,
            return_dict: bool,
        ) -> SimpleNamespace:
            """Return a differentiable fake vision hidden state.

            Args:
                pixel_values: Tensor of shape [batch, channels, height, width].
                image_sizes: Tensor of shape [batch, 2].
                output_hidden_states: Whether hidden states are requested.
                return_dict: Whether to return a structured output.

            Returns:
                Namespace containing a hidden-state tensor of shape [batch, tokens, hidden].
            """
            events.append("vision")
            hidden_state = pixel_values.sum().reshape(1, 1, 1) * self.weight
            return SimpleNamespace(hidden_states=(hidden_state,))

    class FakeProjector(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, image_features: torch.Tensor, image_sizes: torch.Tensor) -> torch.Tensor:
            """Project fake image features while retaining a gradient path.

            Args:
                image_features: Tensor of shape [tokens, vision_hidden].
                image_sizes: Tensor of shape [batch, 2].

            Returns:
                Tensor of shape [tokens, vision_hidden].
            """
            events.append("projector")
            return image_features * self.weight

    model = Mistral3BidirectionalModel.__new__(Mistral3BidirectionalModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(spatial_merge_size=1, vision_feature_layer=-1)
    model.vision_tower = FakeVisionTower()
    model.multi_modal_projector = FakeProjector()

    monkeypatch.setattr(Mistral3BidirectionalModel, "get_input_embeddings", lambda self: embedding)

    def parent_forward(self, **kwargs):
        events.append("parent")
        return kwargs

    monkeypatch.setattr(Mistral3Model, "forward", parent_forward)

    outputs = model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        run_dummy_vision=True,
    )

    assert events == ["vision", "projector", "parent"]
    assert outputs["input_ids"] is None
    assert outputs["inputs_embeds"].shape == (1, 2, 4)
    outputs["inputs_embeds"].sum().backward()
    assert model.vision_tower.weight.grad is not None
    assert model.multi_modal_projector.weight.grad is not None

    events.clear()
    input_ids = torch.tensor([[3, 4]])
    outputs = model(input_ids=input_ids, run_dummy_vision=False)

    assert events == ["parent"]
    assert torch.equal(outputs["input_ids"], input_ids)
    assert outputs["inputs_embeds"] is None


def test_mistral3_vlm_crossencoder_scores_vision_inputs_and_backpropagates():
    config = Mistral3BidirectionalConfig(
        text_config={
            "model_type": "ministral3",
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "attention_dropout": 0.0,
            "use_cache": False,
        },
        vision_config={
            "model_type": "pixtral",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "head_dim": 8,
            "image_size": 8,
            "patch_size": 4,
            "num_channels": 3,
        },
        image_token_index=10,
        spatial_merge_size=1,
        num_labels=1,
        pooling="avg",
        temperature=0.5,
    )
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "eager"
    model = Mistral3VLBidirectionalForSequenceClassification(config)

    assert model.effective_score_temperature == 0.5
    model.eval()
    text_only_input_ids = torch.tensor([[1, 2, 3, 4]])
    modified_input_ids = text_only_input_ids.clone()
    modified_input_ids[0, -1] = 5
    text_only_attention_mask = torch.ones_like(text_only_input_ids)
    with torch.no_grad():
        original_hidden_states = model.model(
            input_ids=text_only_input_ids,
            attention_mask=text_only_attention_mask,
            run_dummy_vision=False,
        ).last_hidden_state
        modified_hidden_states = model.model(
            input_ids=modified_input_ids,
            attention_mask=text_only_attention_mask,
            run_dummy_vision=False,
        ).last_hidden_state
    assert not torch.allclose(original_hidden_states[0, 0], modified_hidden_states[0, 0], atol=1e-6)

    input_ids = torch.tensor([[10, 10, 10, 10, 1, 2]])
    attention_mask = torch.ones_like(input_ids)
    pixel_values = torch.randn(1, 3, 8, 8)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_sizes=torch.tensor([[8, 8]]),
        return_dict=True,
    )

    assert outputs.logits.shape == (1, 1)
    assert torch.isfinite(outputs.logits).all()

    outputs.logits.square().mean().backward()
    for module in (model.model.vision_tower, model.model.multi_modal_projector, model.model.language_model):
        grad_magnitude = sum(
            float(parameter.grad.abs().sum()) for parameter in module.parameters() if parameter.grad is not None
        )
        assert grad_magnitude > 0.0, f"No nonzero gradient reached {type(module).__name__}"
    assert model.score.weight.grad is not None
    assert model.score.weight.grad.abs().sum() > 0.0


def test_mistral3_native_mistral_tower_keeps_stock_dispatch(tmp_path: Path) -> None:
    """An outer mistral3 config must not coerce a native Mistral text tower to Ministral3."""
    text_config = MistralConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
    )
    config = Mistral3Config(
        text_config=text_config,
        vision_config=_tiny_mistral3_bidirectional_vlm_config().vision_config,
        image_token_index=10,
        spatial_merge_size=1,
    )
    config._attn_implementation = "eager"
    reference = Mistral3Model(config).eval()
    reference.save_pretrained(tmp_path)
    assert not Mistral3BidirectionalModel.supports_config(config)
    assert not Mistral3VLBidirectionalForSequenceClassification.supports_config(config)
    actual = build_encoder_backbone(str(tmp_path), "embedding", is_causal=True, attn_implementation="eager").eval()
    assert type(actual) is Mistral3Model
    assert actual.config.text_config.model_type == "mistral"
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    with torch.no_grad():
        torch.testing.assert_close(
            actual(**inputs).last_hidden_state, reference(**inputs).last_hidden_state, rtol=0, atol=0
        )
    assert actual.state_dict().keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("reranker", [False, True])
def test_mistral3_model_owns_layer_groups(reranker: bool) -> None:
    """Real model-owned groups expose both towers without generic family paths."""
    config = _tiny_mistral3_bidirectional_vlm_config()
    model_class = Mistral3VLBidirectionalForSequenceClassification if reranker else Mistral3BidirectionalModel
    assert model_class.supports_config(config)
    model = model_class(config)
    backbone = model.model if reranker else model
    assert model.get_model_layer_groups() == {
        "language": list(backbone.language_model.layers),
        "vision": list(backbone.vision_tower.transformer.layers),
    }


def test_mistral3_unsharded_eval_dummy_vision_is_opt_in() -> None:
    """Ordinary text inference avoids vision; opting in changes no hidden states."""
    model = Mistral3BidirectionalModel(_tiny_mistral3_bidirectional_vlm_config()).eval()
    calls = []
    hook = model.vision_tower.register_forward_hook(lambda *args: calls.append(True))
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    with torch.no_grad():
        expected = model(**inputs).last_hidden_state
        assert not calls
        actual = model(**inputs, run_dummy_vision=True).last_hidden_state
    hook.remove()
    assert calls == [True]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mistral3_reranker_preserves_internal_bf16_temperature_order() -> None:
    """Scaling occurs in score dtype before the recipe casts [batch, 1] logits to FP32."""
    config = _tiny_mistral3_bidirectional_vlm_config()
    config.temperature = 0.02
    config.num_labels = 1
    model = Mistral3VLBidirectionalForSequenceClassification(config).to(torch.bfloat16).eval()
    raw_scores = []
    hook = model.score.register_forward_hook(lambda module, inputs, output: raw_scores.append(output.detach().clone()))
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[1, 2, 3, 4]]), attention_mask=torch.ones(1, 4, dtype=torch.long)).logits
    hook.remove()
    assert logits.dtype == torch.bfloat16
    torch.testing.assert_close(logits, raw_scores[0] / config.temperature, rtol=0, atol=0)


@pytest.mark.parametrize("use_text_in_document", [False, True])
def test_mistral3_corpus_image_caption_policy_reaches_processor(
    pixtral_processor: Mistral3BiEncoderProcessor, use_text_in_document: bool
) -> None:
    """The corpus-ID loader and both collators honor the examples' explicit caption policy."""
    from nemo_automodel.components.datasets.llm.retrieval_dataset import _transform_func
    from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import flatten_bi_encoder_to_cross_encoder

    pixtral_processor.rerank_max_length = 64
    image = Image.new("RGB", (4, 4), (255, 0, 0))
    corpus = SimpleNamespace(get_document_by_id=lambda doc_id: {"text": "literal", "image": image})
    feature = _transform_func(
        {"question": "What is shown?", "corpus_id": "fixture", "pos_doc": ["doc"], "neg_doc": []},
        num_neg_docs=0,
        corpus_dict={"fixture": corpus},
        use_text_in_document=use_text_in_document,
    )
    assert feature["doc_text"] == (["literal"] if use_text_in_document else [""])
    embedding_batch = pixtral_processor.process_queries_documents_biencoder([feature])
    assert (
        pixtral_processor.tokenizer.convert_tokens_to_ids("literal") in embedding_batch["d_input_ids"]
    ) is use_text_in_document
    pairs = flatten_bi_encoder_to_cross_encoder({key: [value] for key, value in feature.items()})
    pair_features = [dict(zip(pairs, values)) for values in zip(*pairs.values())]
    reranker_batch = pixtral_processor.process_queries_documents_crossencoder(pair_features)
    assert (
        pixtral_processor.tokenizer.convert_tokens_to_ids("literal") in reranker_batch["input_ids"]
    ) is use_text_in_document


def test_mistral3_reranker_direct_export_reloads_without_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A direct export includes its model/processor code and preserves text/image scores exactly."""
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=PixtralImageProcessor(size={"longest_edge": 16}),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        padding=True,
        rerank_max_length=128,
        export_as_stock_processor=False,
    )
    config = _tiny_mistral3_bidirectional_vlm_config()
    config.image_token_id = processor.image_token_id
    config.num_labels = 1
    config.temperature = 0.02
    encoder = CrossEncoderModel(Mistral3VLBidirectionalForSequenceClassification(config)).eval()
    features = [
        {"question": "What is shown?", "doc_text": "literal", "doc_image": ""},
        {"question": "What is shown?", "doc_text": "Image doc", "doc_image": Image.new("RGB", (16, 16), (255, 0, 0))},
    ]
    batch = processor.process_queries_documents_crossencoder(features)
    model_inputs = {key: value for key, value in batch.items() if key != "labels"}
    with torch.no_grad():
        expected = encoder.model(**model_inputs).logits
    export_dir = tmp_path / "export"
    encoder.save_pretrained(str(export_dir), tokenizer=processor)
    assert (export_dir / "model.py").is_file()
    assert (export_dir / "processor.py").is_file()
    torch.save(
        {"inputs": model_inputs, "logits": expected, "state": encoder.model.state_dict()}, tmp_path / "expected.pt"
    )
    code = """
import sys
import torch
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoProcessor
directory, expected_path = sys.argv[1:]
model = AutoModelForSequenceClassification.from_pretrained(directory, trust_remote_code=True, attn_implementation="eager").eval()
processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True)
batch = processor.process_queries_documents_crossencoder([
    {"question": "What is shown?", "doc_text": "literal", "doc_image": ""},
    {"question": "What is shown?", "doc_text": "Image doc", "doc_image": Image.new("RGB", (16, 16), (255, 0, 0))},
])
inputs = {key: value for key, value in batch.items() if key != "labels"}
expected = torch.load(expected_path, weights_only=True)
assert set(inputs) == set(expected["inputs"])
for key, value in inputs.items():
    torch.testing.assert_close(value, expected["inputs"][key], rtol=0, atol=0)
assert model.state_dict().keys() == expected["state"].keys()
for key, value in model.state_dict().items():
    torch.testing.assert_close(value, expected["state"][key], rtol=0, atol=0)
with torch.no_grad():
    torch.testing.assert_close(model(**inputs).logits, expected["logits"], rtol=0, atol=0)
assert not any(name == "nemo_automodel" or name.startswith("nemo_automodel.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(export_dir), str(tmp_path / "expected.pt")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

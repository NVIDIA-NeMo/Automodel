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
from transformers import AutoProcessor, PixtralImageProcessor, PixtralProcessor
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.tokenization_utils_tokenizers import TokenizersBackend

pytest.importorskip("transformers.models.ministral3", reason="Ministral3 not available in this transformers version")

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel._transformers.retrieval import BiEncoderModel, _init_encoder_common, configure_encoder_metadata
from nemo_automodel.components.checkpoint.addons import _maybe_save_custom_model_code
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Ministral3BidirectionalConfig,
    Ministral3BidirectionalModel,
    Mistral3BidirectionalModel,
)
from nemo_automodel.components.models.ministral_bidirectional.processor import Mistral3BiEncoderProcessor, load_image


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
    assert call["kwargs"] == {
        "padding": True,
        "pad_to_multiple_of": 4,
        "truncation": True,
        "max_length": 8,
        "return_tensors": None,
        "padding_side": "right",
    }


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


def test_ministral3_biencoder_processor_chat_template_matches_training_helpers(monkeypatch):
    monkeypatch.setattr(Mistral3BiEncoderProcessor, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    processor = Mistral3BiEncoderProcessor(
        image_processor=FakePixtralImageProcessor(),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        q_max_length=64,
        p_max_length=64,
        padding=False,
        chat_template='{{ bos_token }}[INST]{{ messages[0]["content"] }}[/INST]',
    )
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    query_rendered = processor.apply_chat_template(
        [
            {"role": "system", "content": "query:"},
            {"role": "user", "content": "literal [IMG]"},
        ],
        tokenize=False,
    )
    document_rendered = processor.apply_chat_template(
        [
            {"role": "system", "content": [{"type": "text", "text": "passage:"}]},
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

    assert query_rendered == "query: literal [\u200cIMG]"
    assert document_rendered == "passage: [IMG] literal [\u200cIMG_END]"
    assert "[INST]" not in query_rendered + document_rendered

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
            {"role": "system", "content": "passage:"},
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
        "labels",
    }
    assert output["q_input_ids"].shape[0] == 2
    assert output["d_input_ids"].shape[0] == 4
    assert output["d_pixel_values"].shape[0] == 2
    assert output["d_image_sizes"].tolist() == [[4, 4], [4, 4]]
    assert torch.equal(output["labels"], torch.zeros(2, dtype=torch.long))


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


def test_ministral3_bidirectional_model_init_and_mask():
    cfg = tiny_bidirectional_config()
    model = Ministral3BidirectionalModel(cfg)
    model.eval()

    assert all(getattr(layer.self_attn, "is_causal", True) is False for layer in model.layers)

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


def test_ministral_dispatch_uses_custom_class_only_for_legacy_checkpoints():
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
    """Retrieval architectures advertise only the model source file for checkpoint export.

    Must use the real ``Ministral3BidirectionalModel`` class: a class defined in this test
    file would resolve to this test file, not ``.../ministral_bidirectional/model.py``.
    """
    cfg = tiny_bidirectional_config()
    backbone = Ministral3BidirectionalModel(cfg)

    encoder = nn.Module()
    _init_encoder_common(encoder, backbone)

    assert encoder.name_or_path.endswith("ministral_bidirectional/model.py")

    _maybe_save_custom_model_code(encoder.name_or_path, str(tmp_path))

    assert (tmp_path / "model.py").is_file()
    assert not (tmp_path / "processor.py").exists()


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

    model = Mistral3BidirectionalModel.__new__(Mistral3BidirectionalModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(spatial_merge_size=1)
    model.vision_tower = FakeVisionTower()

    monkeypatch.setattr(Mistral3BidirectionalModel, "get_input_embeddings", lambda self: embedding)

    def get_image_features(self, pixel_values, image_sizes, return_dict):
        events.append("vision")
        return SimpleNamespace(pooler_output=(pixel_values.sum().reshape(1) * self.vision_tower.weight,))

    monkeypatch.setattr(Mistral3BidirectionalModel, "get_image_features", get_image_features)

    def parent_forward(self, **kwargs):
        events.append("parent")
        return kwargs

    monkeypatch.setattr(Mistral3Model, "forward", parent_forward)

    outputs = model(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        run_dummy_vision=True,
    )

    assert events == ["vision", "parent"]
    assert outputs["input_ids"] is None
    assert outputs["inputs_embeds"].shape == (1, 2, 4)
    outputs["inputs_embeds"].sum().backward()
    assert model.vision_tower.weight.grad is not None

    events.clear()
    input_ids = torch.tensor([[3, 4]])
    outputs = model(input_ids=input_ids, run_dummy_vision=False)

    assert events == ["parent"]
    assert torch.equal(outputs["input_ids"], input_ids)
    assert outputs["inputs_embeds"] is None

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

import pytest
import torch
import torch.nn as nn
from PIL import Image
from transformers.processing_utils import ProcessorMixin

pytest.importorskip("transformers.models.ministral3", reason="Ministral3 not available in this transformers version")

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel._transformers.retrieval import BiEncoderModel, _init_encoder_common, configure_encoder_metadata
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Ministral3BidirectionalConfig,
    Ministral3BidirectionalModel,
)
from nemo_automodel.components.models.ministral_bidirectional.processor import Ministral3BiEncoderProcessor


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


class FakePixtralTokenizer:
    def __init__(self):
        self.model_input_names = ["input_ids", "attention_mask"]
        self.model_max_length = 99
        self.padding_side = "right"
        self.init_kwargs = {}
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return {"[IMG]": 10, "[IMG_BREAK]": 11, "[IMG_END]": 12}.get(token, 1)

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        self.calls.append({"texts": list(texts), "kwargs": kwargs})

        input_ids = [self._tokenize_text(text) for text in texts]
        if kwargs.get("padding"):
            max_length = max(len(ids) for ids in input_ids)
            for ids in input_ids:
                ids.extend([0] * (max_length - len(ids)))
        attention_mask = [[1 if token_id != 0 else 0 for token_id in ids] for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _tokenize_text(self, text):
        token_ids = []
        idx = 0
        special_tokens = [("[IMG_BREAK]", 11), ("[IMG_END]", 12), ("[IMG]", 10)]
        while idx < len(text):
            matched_special = False
            for token, token_id in special_tokens:
                if text.startswith(token, idx):
                    token_ids.append(token_id)
                    idx += len(token)
                    matched_special = True
                    break
            if matched_special:
                continue
            if text[idx].isspace():
                idx += 1
                continue
            while idx < len(text) and not text[idx].isspace():
                if any(text.startswith(token, idx) for token, _ in special_tokens):
                    break
                idx += 1
            token_ids.append(1)
        return token_ids or [1]


class FakePixtralImageProcessor:
    model_input_names = ["pixel_values"]

    def __init__(self):
        self.calls = []

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
    monkeypatch.setattr(ProcessorMixin, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    return Ministral3BiEncoderProcessor(
        image_processor=FakePixtralImageProcessor(),
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        q_max_length=7,
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
        "max_length": 7,
        "return_tensors": None,
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
    assert call["texts"][0].startswith("passage: [IMG]")
    assert call["texts"][0].endswith("[IMG_END] Image doc")
    assert call["texts"][1] == "passage: Text doc"


def test_ministral3_biencoder_processor_forwards_image_longest_edge(monkeypatch):
    monkeypatch.setattr(ProcessorMixin, "check_argument_for_proper_class", lambda *args, **kwargs: None)
    image_processor = FakePixtralImageProcessor()
    processor = Ministral3BiEncoderProcessor(
        image_processor=image_processor,
        tokenizer=FakePixtralTokenizer(),
        patch_size=4,
        image_longest_edge=448,
        passage_prefix="passage:",
    )
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    processor.process_documents({"images": [image], "texts": ["Image doc"]})

    assert image_processor.calls[-1]["kwargs"]["size"] == {"longest_edge": 448}


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


def test_encoder_build_ministral3_registry_path(tmp_path, monkeypatch):
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
    (model_dir / "config.json").write_text(json.dumps({"model_type": "ministral3"}))

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


@pytest.mark.parametrize("top_level_model_type", ["ministral3", "ministral3_bidirec"])
def test_encoder_build_ministral_supported_model_types(tmp_path, monkeypatch, top_level_model_type):
    """Hub / local text configs use ministral3; saved bidirectional checkpoints use ministral3_bidirec."""

    class FakeBidirectionalModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(hidden=16)

    monkeypatch.setattr(
        ModelRegistry,
        "model_arch_name_to_cls",
        {"Ministral3BidirectionalModel": FakeBidirectionalModel},
    )

    model_dir = tmp_path / "hub" / "checkpoint"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": top_level_model_type}))

    model = BiEncoderModel.build(model_name_or_path=str(model_dir), pooling="avg", l2_normalize=True)
    assert isinstance(model, BiEncoderModel)


def test_configure_encoder_metadata_sets_auto_map_for_ministral_retrieval():
    FakeRetrievalModel = type("Ministral3BidirectionalModel", (), {})
    fake = FakeRetrievalModel()
    FakeCfg = type("Ministral3BidirectionalConfig", (), {})
    fake.config = FakeCfg()

    configure_encoder_metadata(fake, fake.config)

    assert fake.config.architectures == ["Ministral3BidirectionalModel"]
    assert "auto_map" in vars(fake.config)
    assert "AutoModel" in fake.config.auto_map


def test_init_encoder_common_name_or_path_ministral_retrieval():
    """Retrieval architectures set name_or_path from dirname(inspect.getfile(model class)).

    Must use the real ``Ministral3BidirectionalModel`` class: a class defined in this test
    file would resolve to ``.../bi_encoder/``, not ``.../ministral_bidirectional/``.
    """
    cfg = tiny_bidirectional_config()
    backbone = Ministral3BidirectionalModel(cfg)

    encoder = nn.Module()
    _init_encoder_common(encoder, backbone)

    assert "ministral_bidirectional" in encoder.name_or_path

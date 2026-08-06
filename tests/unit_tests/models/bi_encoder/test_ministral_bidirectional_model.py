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
from tokenizers.models import WordLevel
from transformers import PixtralImageProcessor
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model
from transformers.tokenization_utils_tokenizers import TokenizersBackend

pytest.importorskip("transformers.models.ministral3", reason="Ministral3 not available in this transformers version")

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel._transformers.retrieval import BiEncoderModel, _init_encoder_common, configure_encoder_metadata
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


class FakePixtralTokenizer:
    def __init__(self):
        self.model_input_names = ["input_ids", "attention_mask"]
        self.model_max_length = 99
        self.padding_side = "right"
        self.init_kwargs = {}
        self.calls = []
        self.all_special_tokens = ["[INST]", "[IMG]", "[IMG_BREAK]", "[IMG_END]"]

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
        processor.check_argument_for_proper_class("tokenizer", FakePixtralTokenizer())


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


def test_ministral3_biencoder_processor_escapes_user_control_tokens_but_preserves_image_markers(pixtral_processor):
    image = Image.new("RGB", (4, 4), (255, 0, 0))

    pixtral_processor.process_queries(["literal [INST] and [IMG]"])
    query_text = pixtral_processor.tokenizer.calls[-1]["texts"][0]
    pixtral_processor.process_documents({"images": [image], "texts": ["literal [IMG_END]"]})
    document_text = pixtral_processor.tokenizer.calls[-1]["texts"][0]

    assert "[INST]" not in query_text
    assert "[IMG]" not in query_text
    assert "[IMG]" in document_text
    assert document_text.count("[IMG_END]") == 1


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

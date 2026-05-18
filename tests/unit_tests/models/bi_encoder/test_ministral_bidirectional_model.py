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

pytest.importorskip("transformers.models.ministral3", reason="Ministral3 not available in this transformers version")

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel._transformers.retrieval import BiEncoderModel, _init_encoder_common, configure_encoder_metadata
from nemo_automodel.components.models.ministral_bidirectional.model import (
    Ministral3BidirectionalConfig,
    Ministral3BidirectionalModel,
)


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


def test_ministral3_bidirectional_config_fields():
    cfg = Ministral3BidirectionalConfig(pooling="cls", temperature=0.5, vocab_size=100)
    assert cfg.pooling == "cls"
    assert isinstance(cfg.temperature, float)
    assert cfg.model_type == "ministral3_bidirec"


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

    ModelRegistry.model_arch_name_to_cls["Ministral3BidirectionalModel"] = FakeBidirectionalModel
    monkeypatch.setattr(ModelRegistry, "model_arch_name_to_cls", ModelRegistry.model_arch_name_to_cls)

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

    ModelRegistry.model_arch_name_to_cls["Ministral3BidirectionalModel"] = FakeBidirectionalModel
    monkeypatch.setattr(ModelRegistry, "model_arch_name_to_cls", ModelRegistry.model_arch_name_to_cls)

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

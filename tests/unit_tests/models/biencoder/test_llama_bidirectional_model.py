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
import json
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

# Import from the new canonical locations
from nemo_automodel._transformers.biencoder import (
    BiencoderModel,
    pool,
)
from nemo_automodel.recipes.biencoder.train_biencoder import contrastive_scores_and_labels
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalConfig,
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)
from transformers.modeling_outputs import BaseModelOutputWithPast


def test_contrastive_scores_and_labels_shapes_and_labels():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.2, 0.8]])
    scores, labels = contrastive_scores_and_labels(q, k, current_train_n_passages=2)
    assert scores.shape == (2, 2)
    assert torch.all(labels == 0) and labels.shape == (2,)


@pytest.mark.parametrize("pool_type", ["avg", "weighted_avg", "cls", "cls_last", "colbert"])
def test_pool_basic_modes(pool_type):
    last_hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]],
            [[2.0, 1.0], [4.0, 3.0], [1.5, 0.5]],
        ]
    )
    attn = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = pool(last_hidden, attn, pool_type)
    if pool_type == "avg":
        # First seq avg over first 2 tokens
        assert torch.allclose(out[0], torch.tensor([(1.0 + 3.0) / 2, (2.0 + 4.0) / 2]))
    elif pool_type == "weighted_avg":
        # Sum (mask applied) for first two tokens of first seq
        assert torch.allclose(out[0], torch.tensor([1.0 + 3.0, 2.0 + 4.0]))
    elif pool_type in ("cls", "cls_last"):
        assert torch.allclose(out[:, :], last_hidden[:, 0])
    elif pool_type == "colbert":
        assert out.shape == last_hidden.shape


def test_pool_last_with_left_padding_and_right_padding():
    last_hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    # Case 1: left_padding -> attn[:, -1] sum equals batch_size
    attn_left = torch.tensor([[0, 0, 1], [0, 0, 1]])
    out_left = pool(last_hidden, attn_left, "last")
    assert torch.allclose(out_left, last_hidden[:, -1])
    # Case 2: right padding -> pick last non-padded token per sample
    attn_right = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out_right = pool(last_hidden, attn_right, "last")
    # For first sample, last index 1; for second, 2
    assert torch.allclose(out_right[0], last_hidden[0, 1])
    assert torch.allclose(out_right[1], last_hidden[1, 2])


def test_pool_unsupported_raises():
    with pytest.raises(ValueError):
        pool(torch.zeros(1, 1, 1), torch.ones(1, 1), "unsupported")


def test_llama_bidirectional_config_fields():
    cfg = LlamaBidirectionalConfig(pooling="cls", temperature=0.5, vocab_size=100)
    assert cfg.pooling == "cls"
    # Some downstream configs may overwrite; just ensure attribute exists and is float-like
    assert isinstance(cfg.temperature, float)



# --- Fakes for classification and biencoder tests ---
class FakeOutputs:
    def __init__(self, last_hidden_state=None, hidden_states=None):
        self.last_hidden_state = last_hidden_state
        self.hidden_states = hidden_states
        self.past_key_values = None
        self.attentions = None

    def __getitem__(self, idx):
        seq = (self.last_hidden_state, self.past_key_values, self.hidden_states, self.attentions)
        return seq[idx]


class FakeLM(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()

        class Cfg:
            def __init__(self):
                self.hidden_size = hidden

        self.config = Cfg()
        self.linear = nn.Linear(hidden, hidden)
        self._ckpt = False
        self.saved = []

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, output_hidden_states=True, **kwargs):
        bsz = input_ids.shape[0]
        seq = input_ids.shape[1]
        h = self.config.hidden_size
        # deterministic tiny hidden states
        last = torch.ones(bsz, seq, h)
        hstates = [last * (i + 1) for i in range(3)]
        return FakeOutputs(last_hidden_state=last, hidden_states=hstates)

    def gradient_checkpointing_enable(self):
        self._ckpt = True

    def save_pretrained(self, out_dir):
        self.saved.append(out_dir)


def test_sequence_classification_forward_variants(monkeypatch):
    # Build instance without running HF parent __init__
    hidden = 8
    inst = object.__new__(LlamaBidirectionalForSequenceClassification)
    # Initialize nn.Module base so we can attach submodules safely
    nn.Module.__init__(inst)

    class DummyCfg:
        def __init__(self):
            self.pooling = "avg"
            self.temperature = 2.0
            self.problem_type = None
            self.use_return_dict = True

    inst.config = DummyCfg()
    inst.model = FakeLM(hidden=hidden)
    inst.num_labels = 1
    inst.score = nn.Linear(hidden, 1)
    bsz, seqlen = 2, 3
    input_ids = torch.ones(bsz, seqlen, dtype=torch.long)
    attn = torch.ones(bsz, seqlen, dtype=torch.long)
    # Regression
    out_reg = inst(input_ids=input_ids, attention_mask=attn, labels=torch.zeros(bsz, 1))
    assert isinstance(out_reg, SequenceClassifierOutputWithPast)
    assert out_reg.loss is not None
    # Single label classification
    inst.num_labels = 3
    inst.score = nn.Linear(hidden, 3)
    inst.config.problem_type = None
    out_s = inst(input_ids=input_ids, attention_mask=attn, labels=torch.zeros(bsz, dtype=torch.long))
    assert out_s.loss is not None
    # Multi label classification
    inst.config.problem_type = None
    out_m = inst(input_ids=input_ids, attention_mask=attn, labels=torch.zeros(bsz, 3))
    assert out_m.loss is not None
    # return_dict=False path
    ret = inst(input_ids=input_ids, attention_mask=attn, return_dict=False)
    assert isinstance(ret, tuple) and torch.is_tensor(ret[0])


def test_biencoder_encode_and_compute_scores_and_forward(monkeypatch):
    # Fake encoder that lacks token_type_ids argument, to exercise removal in _encode
    class NoTTIDLm(FakeLM):
        def forward(self, input_ids=None, attention_mask=None, return_dict=True, output_hidden_states=True, **kwargs):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_hidden_states=output_hidden_states,
            )

    lm_q = NoTTIDLm(hidden=8)
    lm_p = NoTTIDLm(hidden=8)
    model = BiencoderModel(
        lm_q=lm_q, lm_p=lm_p, pooling="avg", l2_normalize=True
    )
    # encode removes token_type_ids and normalizes
    q = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
    }
    v = model.encode(q, encoder="query")
    assert v.shape == (2, 8)
    assert torch.allclose(torch.linalg.norm(v, dim=-1), torch.ones(2), atol=1e-5)
    # Compute scores explicitly to avoid coupling to internal repeat implementation
    p = {"input_ids": torch.ones(4, 3, dtype=torch.long), "attention_mask": torch.ones(4, 3, dtype=torch.long)}
    q_reps = model.encode(q, encoder="query")
    p_reps = model.encode(p, encoder="passage")
    assert q_reps.shape == (2, 8) and p_reps.shape == (4, 8)
    scores, labels = contrastive_scores_and_labels(q_reps, p_reps, current_train_n_passages=2)
    assert scores.shape == (2, 2) and torch.all(labels == 0)
    # Test explicit loss computation
    model.eval()
    p2 = {
        "input_ids": torch.ones(4, 3, dtype=torch.long),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
    }
    q_reps2 = model.encode(q, encoder="query")
    p_reps2 = model.encode(p2, encoder="passage")
    scores2, labels2 = contrastive_scores_and_labels(q_reps2, p_reps2, current_train_n_passages=2)
    loss2 = F.cross_entropy(scores2, labels2)
    assert loss2 is not None and torch.is_tensor(loss2)

    # encode path using hidden_states when last_hidden_state absent
    class OnlyHiddenOutputs:
        def __init__(self, hidden_states):
            self.hidden_states = hidden_states

    class NoLastLM(FakeLM):
        def forward(self, input_ids=None, attention_mask=None, return_dict=True, output_hidden_states=True, **kwargs):
            bsz, seqlen = input_ids.shape[:2]
            h = self.config.hidden_size
            hidden_states = [torch.ones(bsz, seqlen, h) * (i + 1) for i in range(2)]
            return OnlyHiddenOutputs(hidden_states)

    # Test with model using NoLastLM for query encoder
    model_no_last = BiencoderModel(
        lm_q=NoLastLM(hidden=8), lm_p=NoTTIDLm(hidden=8), pooling="avg", l2_normalize=True
    )
    v2 = model_no_last.encode(
        {"input_ids": torch.ones(2, 3, dtype=torch.long), "attention_mask": torch.ones(2, 3, dtype=torch.long)},
        encoder="query"
    )
    assert v2.shape == (2, 8)


def test_biencoder_build_and_save(tmp_path, monkeypatch):
    # Patch ModelClass.from_pretrained to return FakeLM
    class FakeBidirectionalModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(hidden=16)

    # Patch the registry to return our fake model
    ModelRegistry.model_arch_name_to_cls["LlamaBidirectionalModel"] = FakeBidirectionalModel
    monkeypatch.setattr(ModelRegistry, "model_arch_name_to_cls", ModelRegistry.model_arch_name_to_cls)

    # Directory path with config.json to hit config-reading branch
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama"}))

    model = BiencoderModel.build(
        model_name_or_path=str(model_dir),
        share_encoder=True,
        pooling="avg",
        l2_normalize=True,
    )
    assert isinstance(model, BiencoderModel)
    outdir = tmp_path / "save1"
    outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(outdir))
    assert any("save1" in p for p in model.lm_q.saved)

    # build with share_encoder=False
    model2 = BiencoderModel.build(
        model_name_or_path=str(model_dir),
        share_encoder=False,
    )
    outdir2 = tmp_path / "save2"
    model2.save_pretrained(str(outdir2))
    # separate subdirs created
    assert os.path.isdir(outdir2 / "query_model")
    assert os.path.isdir(outdir2 / "passage_model")


def test_llama_bidirectional_forward_paths(monkeypatch):
    cfg = LlamaBidirectionalConfig(
        vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=1, intermediate_size=32, pad_token_id=0
    )
    model = LlamaBidirectionalModel(cfg)
    bsz, seqlen = 2, 3
    input_ids = torch.randint(0, cfg.vocab_size, (bsz, seqlen))
    attn = torch.ones(bsz, seqlen, dtype=torch.long)
    # Error on invalid combination (neither provided)
    with pytest.raises(ValueError):
        model(input_ids=None, inputs_embeds=None)
    # Error on legacy past_key_values type
    with pytest.raises(AttributeError):
        model(input_ids=input_ids, attention_mask=attn, past_key_values=123)
    # Normal forward with outputs requested
    model.eval()
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        use_cache=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    assert isinstance(out, BaseModelOutputWithPast.__mro__[0]) or hasattr(out, "last_hidden_state")
    assert out.past_key_values is not None


def test_sequence_classification_regression_multi_output(monkeypatch):
    # Use manual instance with dummy config as before
    hidden = 8
    inst = object.__new__(LlamaBidirectionalForSequenceClassification)
    nn.Module.__init__(inst)

    class DummyCfg:
        def __init__(self):
            self.pooling = "avg"
            self.temperature = 1.0
            self.problem_type = "regression"
            self.use_return_dict = True

    inst.config = DummyCfg()
    inst.model = FakeLM(hidden=hidden)
    inst.num_labels = 2
    inst.score = nn.Linear(hidden, 2)
    bsz, seqlen = 2, 3
    input_ids = torch.ones(bsz, seqlen, dtype=torch.long)
    attn = torch.ones(bsz, seqlen, dtype=torch.long)
    out = inst(input_ids=input_ids, attention_mask=attn, labels=torch.zeros(bsz, 2))
    assert isinstance(out, SequenceClassifierOutputWithPast)
    assert out.loss is not None


def test_biencoder_build_llama_bidirec_model_type_generic_path(tmp_path, monkeypatch):
    """Regression test: model_type 'llama_bidirec' at a generic path without 'llama' in it.

    When the customizer API downloads the model, the path is something like
    /var/run/scratch/job/model which does not contain 'llama'. The build()
    method must still recognise the model via config.json's model_type field.
    """

    class FakeBidirectionalModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(hidden=16)

    # Patch the registry to return our fake model
    ModelRegistry.model_arch_name_to_cls["LlamaBidirectionalModel"] = FakeBidirectionalModel
    monkeypatch.setattr(ModelRegistry, "model_arch_name_to_cls", ModelRegistry.model_arch_name_to_cls)

    # Create a model directory whose path has no 'llama' substring
    model_dir = tmp_path / "scratch" / "job" / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "llama_bidirec"}))

    # Mock AutoConfig.from_pretrained to return a config with the llama_bidirec model_type
    import nemo_automodel._transformers.biencoder as biencoder_module

    class FakeConfig:
        model_type = "llama_bidirec"

    def fake_auto_config_from_pretrained(*args, **kwargs):
        return FakeConfig()

    monkeypatch.setattr(biencoder_module.AutoConfig, "from_pretrained", fake_auto_config_from_pretrained)

    model = BiencoderModel.build(
        model_name_or_path=str(model_dir),
        share_encoder=True,
        pooling="avg",
        l2_normalize=True,
        t=0.5,
    )
    assert isinstance(model, BiencoderModel)


def test_biencoder_build_hub_and_errors(tmp_path, monkeypatch):
    # Patch ModelClass.from_pretrained to return FakeLM for hub path
    class FakeBidirectionalModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(hidden=16)

    # Patch the registry to return our fake model
    ModelRegistry.model_arch_name_to_cls["LlamaBidirectionalModel"] = FakeBidirectionalModel
    monkeypatch.setattr(ModelRegistry, "model_arch_name_to_cls", ModelRegistry.model_arch_name_to_cls)

    # Unsupported model type from config
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "config.json").write_text(json.dumps({"model_type": "bert"}))
    with pytest.raises(ValueError):
        BiencoderModel.build(model_name_or_path=str(bad_dir))

    # For hub path tests, we need to mock AutoConfig.from_pretrained since the new code
    # calls it first to determine model type before using the registry
    import nemo_automodel._transformers.biencoder as biencoder_module

    class FakeConfig:
        model_type = "llama"

    original_auto_config = biencoder_module.AutoConfig

    def fake_auto_config_from_pretrained(*args, **kwargs):
        return FakeConfig()

    monkeypatch.setattr(biencoder_module.AutoConfig, "from_pretrained", fake_auto_config_from_pretrained)

    # Hub path with share_encoder True
    m1 = BiencoderModel.build(model_name_or_path="llama-tiny", share_encoder=True)
    assert isinstance(m1, BiencoderModel)
    # Hub path with share_encoder False
    m2 = BiencoderModel.build(model_name_or_path="llama-tiny", share_encoder=False)
    assert isinstance(m2, BiencoderModel)

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
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutputWithPast
from transformers.models.llama.modeling_llama import LlamaModel

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel._transformers.retrieval import (
    BiEncoderModel,
    CrossEncoderModel,
    _init_encoder_common,
    configure_encoder_metadata,
    pool,
)
from nemo_automodel.components.models.llama_bidirectional.model import (
    LlamaBidirectionalConfig,
    LlamaBidirectionalForSequenceClassification,
    LlamaBidirectionalModel,
)
from nemo_automodel.recipes.retrieval.train_bi_encoder import contrastive_scores_and_labels


def test_contrastive_scores_and_labels_shapes_and_labels():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    k = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.2, 0.8]])
    scores, labels = contrastive_scores_and_labels(q, k, current_train_n_passages=2)
    assert scores.shape == (2, 2)
    assert torch.all(labels == 0) and labels.shape == (2,)


@pytest.mark.parametrize("pool_type", ["avg", "weighted_avg", "cls", "colbert"])
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
    elif pool_type == "cls":
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


def test_llama_bidirectional_model_init_and_mask():
    cfg = LlamaBidirectionalConfig(
        vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=1, intermediate_size=64, pad_token_id=0
    )
    model = LlamaBidirectionalModel(cfg)
    model.eval()

    # All attention layers should be non-causal
    assert all(getattr(layer.self_attn, "is_causal", True) is False for layer in model.layers)

    # Forward with padding mask produces valid output
    input_ids = torch.randint(0, cfg.vocab_size, (1, 3))
    mask = torch.tensor([[1, 1, 0]])
    out = model(input_ids=input_ids, attention_mask=mask)
    assert out.last_hidden_state is not None and out.last_hidden_state.shape == (1, 3, 32)

    # Forward without attention mask also works
    out_no_mask = model(input_ids=input_ids)
    assert out_no_mask.last_hidden_state is not None and out_no_mask.last_hidden_state.shape == (1, 3, 32)


def test_bidirectional_attention_is_symmetric():
    """Verify that the bidirectional model produces symmetric attention behavior:
    changing a token at position i should affect the hidden state at position j
    and vice versa (unlike causal models where earlier tokens can't see later ones)."""
    cfg = LlamaBidirectionalConfig(
        vocab_size=128, hidden_size=32, num_hidden_layers=1, num_attention_heads=1, intermediate_size=64, pad_token_id=0
    )
    model = LlamaBidirectionalModel(cfg)
    model.eval()

    input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
    attn = torch.ones(1, 4, dtype=torch.long)

    with torch.no_grad():
        out_base = model(input_ids=input_ids, attention_mask=attn).last_hidden_state.clone()

        # Change last token — in a bidirectional model, this should affect ALL positions
        modified = input_ids.clone()
        modified[0, -1] = (input_ids[0, -1] + 1) % cfg.vocab_size
        out_modified = model(input_ids=modified, attention_mask=attn).last_hidden_state

    # Position 0 should be different because it can attend to the changed last token
    assert not torch.allclose(out_base[0, 0], out_modified[0, 0], atol=1e-6), (
        "Bidirectional model: changing last token should affect first token's hidden state"
    )



# --- Fakes for classification and encoder tests ---
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


def test_encoder_encode_and_compute_scores_and_forward(monkeypatch):
    # Fake encoder that lacks token_type_ids argument, to exercise removal in _encode
    class NoTTIDLm(FakeLM):
        def forward(self, input_ids=None, attention_mask=None, return_dict=True, output_hidden_states=True, **kwargs):
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=return_dict,
                output_hidden_states=output_hidden_states,
            )

    lm = NoTTIDLm(hidden=8)
    model = BiEncoderModel(
        model=lm, pooling="avg", l2_normalize=True
    )
    # encode removes token_type_ids and normalizes
    q = {
        "input_ids": torch.ones(2, 3, dtype=torch.long),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
    }
    v = model.encode(q)
    assert v.shape == (2, 8)
    assert torch.allclose(torch.linalg.norm(v, dim=-1), torch.ones(2), atol=1e-5)
    # Compute scores explicitly to avoid coupling to internal repeat implementation
    p = {"input_ids": torch.ones(4, 3, dtype=torch.long), "attention_mask": torch.ones(4, 3, dtype=torch.long)}
    q_reps = model.encode(q)
    p_reps = model.encode(p)
    assert q_reps.shape == (2, 8) and p_reps.shape == (4, 8)
    scores, labels = contrastive_scores_and_labels(q_reps, p_reps, current_train_n_passages=2)
    assert scores.shape == (2, 2) and torch.all(labels == 0)
    # Test explicit loss computation
    model.eval()
    p2 = {
        "input_ids": torch.ones(4, 3, dtype=torch.long),
        "attention_mask": torch.ones(4, 3, dtype=torch.long),
    }
    q_reps2 = model.encode(q)
    p_reps2 = model.encode(p2)
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
    model_no_last = BiEncoderModel(
        model=NoLastLM(hidden=8), pooling="avg", l2_normalize=True
    )
    v2 = model_no_last.encode(
        {"input_ids": torch.ones(2, 3, dtype=torch.long), "attention_mask": torch.ones(2, 3, dtype=torch.long)},
    )
    assert v2.shape == (2, 8)


def test_encoder_build_and_save(tmp_path, monkeypatch):
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


def test_encoder_build_llama_bidirec_model_type_generic_path(tmp_path, monkeypatch):
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
    import nemo_automodel._transformers.retrieval as encoder_module

    class FakeConfig:
        model_type = "llama_bidirec"

    def fake_auto_config_from_pretrained(*args, **kwargs):
        return FakeConfig()

    monkeypatch.setattr(encoder_module.AutoConfig, "from_pretrained", fake_auto_config_from_pretrained)

    model = BiEncoderModel.build(
        model_name_or_path=str(model_dir),
        pooling="avg",
        l2_normalize=True,
    )
    assert isinstance(model, BiEncoderModel)


def test_encoder_build_hub_and_errors(tmp_path, monkeypatch):
    # Patch ModelClass.from_pretrained to return FakeLM for hub path
    class FakeBidirectionalModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(hidden=16)

    # Patch the registry to return our fake model
    ModelRegistry.model_arch_name_to_cls["LlamaBidirectionalModel"] = FakeBidirectionalModel
    monkeypatch.setattr(ModelRegistry, "model_arch_name_to_cls", ModelRegistry.model_arch_name_to_cls)

    # Model type not in SUPPORTED_BACKBONES should fall back to AutoModel
    import nemo_automodel._transformers.retrieval as encoder_module

    class FakeAutoModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            inst = cls(hidden=16)
            inst.config.name_or_path = args[0] if args else ""
            return inst

    monkeypatch.setattr(encoder_module.AutoModel, "from_pretrained", FakeAutoModel.from_pretrained)

    bert_dir = tmp_path / "bert_model"
    bert_dir.mkdir()
    (bert_dir / "config.json").write_text(json.dumps({"model_type": "bert"}))
    m_bert = BiEncoderModel.build(model_name_or_path=str(bert_dir))
    assert isinstance(m_bert, BiEncoderModel)

    # For hub path tests, we need to mock AutoConfig.from_pretrained since the new code
    # calls it first to determine model type before using the registry
    import nemo_automodel._transformers.retrieval as encoder_module

    class FakeConfig:
        model_type = "llama"

    def fake_auto_config_from_pretrained(*args, **kwargs):
        return FakeConfig()

    monkeypatch.setattr(encoder_module.AutoConfig, "from_pretrained", fake_auto_config_from_pretrained)

    # Hub path
    m1 = BiEncoderModel.build(model_name_or_path="llama-tiny")
    assert isinstance(m1, BiEncoderModel)


def test_build_generic_hf_model_score_task(tmp_path, monkeypatch):
    """CrossEncoderModel should use AutoModelForSequenceClassification for unsupported model types."""
    import nemo_automodel._transformers.retrieval as encoder_module

    class FakeSeqClsModel(FakeLM):
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            inst = cls(hidden=16)
            inst.config.name_or_path = args[0] if args else ""
            return inst

        def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
            bsz = input_ids.shape[0]
            return SequenceClassifierOutputWithPast(logits=torch.zeros(bsz, 2))

    class FakeConfig:
        model_type = "qwen3"

    monkeypatch.setattr(encoder_module.AutoConfig, "from_pretrained", lambda *a, **kw: FakeConfig())
    monkeypatch.setattr(
        encoder_module.AutoModelForSequenceClassification, "from_pretrained", FakeSeqClsModel.from_pretrained
    )

    model = CrossEncoderModel.build(model_name_or_path=str(tmp_path), trust_remote_code=True)
    assert isinstance(model, CrossEncoderModel)


def test_configure_encoder_metadata_skips_auto_map_for_generic():
    """auto_map should only be set for retrieval architectures, not generic HF models."""
    Qwen3Model = type("Qwen3Model", (), {})
    fake = Qwen3Model()

    class FakeCfg:
        pass

    fake.config = FakeCfg()
    configure_encoder_metadata(fake, fake.config)

    assert fake.config.architectures == ["Qwen3Model"]
    assert not hasattr(fake.config, "auto_map")


def test_configure_encoder_metadata_sets_auto_map_for_retrieval():
    """auto_map should be set for registered retrieval architectures."""

    class FakeRetrievalModel:
        pass

    FakeRetrievalModel.__name__ = "LlamaBidirectionalModel"
    FakeRetrievalModel = type("LlamaBidirectionalModel", (), {})
    fake = FakeRetrievalModel()

    class FakeCfg:
        pass

    FakeCfg.__name__ = "LlamaBidirectionalConfig"
    FakeCfg = type("LlamaBidirectionalConfig", (), {})
    fake.config = FakeCfg()

    configure_encoder_metadata(fake, fake.config)

    assert fake.config.architectures == ["LlamaBidirectionalModel"]
    assert "auto_map" in vars(fake.config)
    assert "AutoModel" in fake.config.auto_map


def test_init_encoder_common_name_or_path_for_generic():
    """For generic HF models, name_or_path should come from config, not inspect.getfile."""

    class FakeCfg:
        name_or_path = "Qwen/Qwen3-1.7B"
        hidden_size = 16

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeCfg()
            self.linear = nn.Linear(16, 16)

    # Use a class name that is NOT a retrieval arch
    FakeModel.__name__ = "Qwen3Model"
    FakeModel = type("Qwen3Model", (nn.Module,), {
        "__init__": FakeModel.__init__,
        "config": property(lambda self: self._config),
    })
    fake = object.__new__(FakeModel)
    nn.Module.__init__(fake)
    fake._config = FakeCfg()

    encoder = nn.Module()
    _init_encoder_common(encoder, fake)

    assert encoder.name_or_path == "Qwen/Qwen3-1.7B"


# ---------------------------------------------------------------------------
# is_causal=False refactor: config, forward delegation, encode kwarg,
# extract_submodel, and generic-path is_causal flags
# ---------------------------------------------------------------------------


def _tiny_bidirec_config(**overrides):
    defaults = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=32,
        pad_token_id=0,
    )
    defaults.update(overrides)
    return LlamaBidirectionalConfig(**defaults)


def test_config_is_causal_set_to_false():
    """config.is_causal must be False after init — required for create_causal_mask
    to redirect to create_bidirectional_mask in the parent forward()."""
    cfg = _tiny_bidirec_config()
    model = LlamaBidirectionalModel(cfg)
    assert model.config.is_causal is False


def test_no_forward_override():
    """LlamaBidirectionalModel must NOT define its own forward().
    It relies on the parent LlamaModel.forward() which calls create_causal_mask
    and respects config.is_causal = False."""
    assert "forward" not in LlamaBidirectionalModel.__dict__
    assert LlamaBidirectionalModel.forward is LlamaModel.forward


def test_bidirectional_output_via_parent_forward():
    """Parent forward() should produce bidirectional attention when
    config.is_causal = False — changing a later token must affect
    an earlier token's hidden state."""
    cfg = _tiny_bidirec_config()
    model = LlamaBidirectionalModel(cfg)
    model.eval()

    ids_a = torch.tensor([[1, 2, 3, 4]])
    ids_b = torch.tensor([[1, 2, 3, 5]])  # only last token differs
    mask = torch.ones_like(ids_a)

    with torch.no_grad():
        out_a = model(input_ids=ids_a, attention_mask=mask)
        out_b = model(input_ids=ids_b, attention_mask=mask)

    # In bidirectional attention, changing token 4 affects ALL positions,
    # including position 0. In causal attention, position 0 would be identical.
    hidden_a = out_a.last_hidden_state[0, 0]
    hidden_b = out_b.last_hidden_state[0, 0]
    assert not torch.allclose(hidden_a, hidden_b, atol=1e-5), (
        "Position 0 hidden state should differ when a later token changes "
        "(bidirectional attention). If identical, attention is still causal."
    )


class _SpyModel(nn.Module):
    """A fake model that records the kwargs passed to forward()."""

    def __init__(self, hidden_size=16):
        super().__init__()
        self.config = MagicMock(hidden_size=hidden_size)
        self.captured_kwargs = {}

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        self.captured_kwargs = dict(kwargs)
        bsz, seq = input_ids.shape
        h = self.config.hidden_size
        last = torch.ones(bsz, seq, h)
        return BaseModelOutputWithPast(
            last_hidden_state=last,
            hidden_states=[last],
        )


def test_encode_passes_is_causal_false():
    """BiEncoderModel.encode() must pass is_causal=False to the model's
    forward() so FA2/SDPA kernels don't apply causal masking."""
    spy = _SpyModel(hidden_size=16)
    encoder = BiEncoderModel(model=spy, pooling="avg", l2_normalize=False)

    input_dict = {
        "input_ids": torch.ones(2, 4, dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
    }
    encoder.encode(input_dict)

    assert "is_causal" in spy.captured_kwargs, (
        "encode() must pass is_causal kwarg to model forward"
    )
    assert spy.captured_kwargs["is_causal"] is False


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "fake_text"})()
        self.layers = nn.ModuleList()
        self.linear = nn.Linear(8, 8)


class _FakeVLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.vision_model = nn.Linear(4, 4)


def test_extract_submodel(monkeypatch):
    """build_encoder_backbone with extract_submodel='language_model' should
    return the language_model submodule, not the full VLM."""
    from nemo_automodel._transformers import retrieval

    vlm = _FakeVLM()

    monkeypatch.setattr(
        retrieval, "AutoModel",
        type("AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **kw: vlm)}),
    )
    monkeypatch.setattr(
        retrieval, "AutoConfig",
        type("AutoConfig", (), {"from_pretrained": staticmethod(
            lambda *a, **kw: type("Cfg", (), {"model_type": "fake_vlm"})()
        )}),
    )

    result = retrieval.build_encoder_backbone(
        model_name_or_path="fake/vlm",
        task="embedding",
        extract_submodel="language_model",
    )
    assert result is vlm.language_model


class _FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_causal = True


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeDecoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Cfg", (), {"model_type": "fake_decoder"})()
        self.layers = nn.ModuleList([_FakeDecoderLayer(), _FakeDecoderLayer()])


def _mock_generic_automodel(monkeypatch):
    from nemo_automodel._transformers import retrieval

    fake_model = _FakeDecoderModel()
    monkeypatch.setattr(
        retrieval, "AutoModel",
        type("AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **kw: fake_model)}),
    )
    monkeypatch.setattr(
        retrieval, "AutoConfig",
        type("AutoConfig", (), {"from_pretrained": staticmethod(
            lambda *a, **kw: type("Cfg", (), {"model_type": "fake_decoder"})()
        )}),
    )
    return fake_model


def test_build_encoder_backbone_sets_config_is_causal(monkeypatch):
    """build_encoder_backbone must set config.is_causal = False on generic models."""
    _mock_generic_automodel(monkeypatch)
    from nemo_automodel._transformers import retrieval

    result = retrieval.build_encoder_backbone("fake/path", task="embedding")
    assert result.config.is_causal is False


def test_build_encoder_backbone_sets_attention_is_causal(monkeypatch):
    """build_encoder_backbone must set module.is_causal = False on all attention layers."""
    _mock_generic_automodel(monkeypatch)
    from nemo_automodel._transformers import retrieval

    result = retrieval.build_encoder_backbone("fake/path", task="embedding")
    for layer in result.layers:
        assert layer.self_attn.is_causal is False

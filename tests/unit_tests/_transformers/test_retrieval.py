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

"""Tests for nemo_automodel._transformers.retrieval (build_encoder_backbone, etc.)."""

import json

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

# ---------------------------------------------------------------------------
# extract_submodel: basic extraction
# ---------------------------------------------------------------------------


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


def _mock_auto_classes(monkeypatch, model, config):
    from nemo_automodel._transformers import retrieval

    monkeypatch.setattr(
        retrieval,
        "AutoModel",
        type("AutoModel", (), {"from_pretrained": staticmethod(lambda *a, **kw: model)}),
    )
    monkeypatch.setattr(
        retrieval,
        "AutoConfig",
        type("AutoConfig", (), {"from_pretrained": staticmethod(lambda *a, **kw: config)}),
    )


def test_extract_submodel(monkeypatch):
    """build_encoder_backbone with extract_submodel='language_model' should
    return the language_model submodule, not the full VLM."""
    from nemo_automodel._transformers import retrieval

    vlm = _FakeVLM()
    cfg = type("Cfg", (), {"model_type": "fake_vlm"})()
    _mock_auto_classes(monkeypatch, vlm, cfg)

    result = retrieval.build_encoder_backbone(
        model_name_or_path="fake/vlm",
        task="embedding",
        extract_submodel="language_model",
    )
    assert result is vlm.language_model


# ---------------------------------------------------------------------------
# extract_submodel: config must be the submodel's config, not the VLM's
# ---------------------------------------------------------------------------


def test_extract_submodel_config_is_text_config(monkeypatch):
    """After extraction, result.config must be the text backbone's config,
    not the parent VLM config. This is critical for save_pretrained to
    write the correct config.json."""
    from nemo_automodel._transformers import retrieval

    vlm = _FakeVLM()
    vlm.config = type("VlmCfg", (), {"model_type": "vlm_parent"})()
    # language_model already has its own config with model_type "fake_text"

    _mock_auto_classes(monkeypatch, vlm, vlm.config)

    result = retrieval.build_encoder_backbone(
        model_name_or_path="fake/vlm",
        task="embedding",
        extract_submodel="language_model",
    )
    assert result.config.model_type == "fake_text", (
        f"Expected text config (model_type='fake_text'), "
        f"got VLM config (model_type='{result.config.model_type}')"
    )


# ---------------------------------------------------------------------------
# extract_submodel: state_dict keys must be clean (no VLM prefix)
# ---------------------------------------------------------------------------


def test_extract_submodel_state_dict_keys_are_clean(monkeypatch):
    """After extraction, state_dict keys must not have the VLM parent prefix
    (e.g. 'linear.weight', not 'language_model.linear.weight')."""
    from nemo_automodel._transformers import retrieval

    vlm = _FakeVLM()
    _mock_auto_classes(monkeypatch, vlm, type("Cfg", (), {"model_type": "fake_vlm"})())

    result = retrieval.build_encoder_backbone(
        model_name_or_path="fake/vlm",
        task="embedding",
        extract_submodel="language_model",
    )
    keys = list(result.state_dict().keys())
    for key in keys:
        assert not key.startswith("language_model."), (
            f"State dict key '{key}' has VLM prefix — should be relative to extracted submodel"
        )


# ---------------------------------------------------------------------------
# extract_submodel: round-trip save/reload with a real tiny model
# ---------------------------------------------------------------------------


def test_extract_submodel_save_reload_round_trip(monkeypatch, tmp_path):
    """Extract a text backbone from a VLM, save it, reload it, and verify
    the config and weights survive the round-trip."""
    from nemo_automodel._transformers import retrieval

    # Build a tiny "VLM" with a real LlamaModel as the language_model
    text_cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        intermediate_size=32,
    )
    text_model = LlamaModel(text_cfg)

    vlm = nn.Module()
    vlm.language_model = text_model
    vlm.vision_tower = nn.Linear(4, 4)

    vlm_cfg = type("VlmCfg", (), {"model_type": "fake_vlm"})()
    _mock_auto_classes(monkeypatch, vlm, vlm_cfg)

    extracted = retrieval.build_encoder_backbone(
        model_name_or_path="fake/vlm",
        task="embedding",
        extract_submodel="language_model",
    )

    # Save the extracted model
    save_dir = tmp_path / "saved_model"
    extracted.save_pretrained(str(save_dir))

    # Verify saved config is the text config, not VLM config
    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["model_type"] == "llama"

    # Reload and verify weights match
    reloaded = AutoModel.from_pretrained(str(save_dir))
    assert reloaded.config.model_type == "llama"

    original_sd = extracted.state_dict()
    reloaded_sd = reloaded.state_dict()
    assert set(original_sd.keys()) == set(reloaded_sd.keys())
    for key in original_sd:
        assert torch.equal(original_sd[key], reloaded_sd[key]), f"Weight mismatch for {key}"


# ---------------------------------------------------------------------------
# extract_submodel: submodule without its own config (not a PreTrainedModel)
# ---------------------------------------------------------------------------


class _PlainSubmodule(nn.Module):
    """A submodule that is NOT a PreTrainedModel — has no .config."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)


class _VLMWithPlainSubmodule(nn.Module):
    """VLM where the language_model is a plain nn.Module without .config."""

    def __init__(self):
        super().__init__()
        self.language_model = _PlainSubmodule()
        self.vision_tower = nn.Linear(4, 4)


def test_extract_submodel_without_config_raises(monkeypatch):
    """If the extracted submodel has no .config, build_encoder_backbone should
    raise a clear error rather than silently returning a broken model."""
    from nemo_automodel._transformers import retrieval

    vlm = _VLMWithPlainSubmodule()
    _mock_auto_classes(monkeypatch, vlm, type("Cfg", (), {"model_type": "fake_vlm"})())

    try:
        result = retrieval.build_encoder_backbone(
            model_name_or_path="fake/vlm",
            task="embedding",
            extract_submodel="language_model",
        )
        # If it doesn't raise, the extracted model must at least have a config
        assert hasattr(result, "config"), (
            "Extracted submodel has no .config — save_pretrained will fail"
        )
    except (AttributeError, ValueError):
        pass  # An explicit error is acceptable


# ---------------------------------------------------------------------------
# extract_submodel: Mistral3 VLM end-to-end (tiny config, CPU, no downloads)
# ---------------------------------------------------------------------------


def _tiny_mistral3_vlm_config():
    """Build a tiny Mistral3 VLM config that can be instantiated on CPU."""
    from transformers import Mistral3Config

    return Mistral3Config(
        text_config=dict(
            model_type="mistral",
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        ),
        vision_config=dict(
            model_type="pixtral",
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
            num_channels=3,
        ),
    )


def test_extract_submodel_mistral3_vlm_forward_save_reload(monkeypatch, tmp_path):
    """End-to-end: build tiny Mistral3 VLM → extract language_model →
    forward pass → save → reload → verify config and weights."""
    from nemo_automodel._transformers import retrieval

    vlm_cfg = _tiny_mistral3_vlm_config()
    vlm = AutoModel.from_config(vlm_cfg)

    _mock_auto_classes(monkeypatch, vlm, vlm_cfg)

    # Extract language_model via build_encoder_backbone
    backbone = retrieval.build_encoder_backbone(
        model_name_or_path="fake/mistral3-vlm",
        task="embedding",
        extract_submodel="language_model",
    )

    # Verify config is the text config, not VLM config
    assert backbone.config.model_type == "mistral", (
        f"Expected text config model_type='mistral', got '{backbone.config.model_type}'"
    )

    # Forward pass
    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    backbone.eval()
    with torch.no_grad():
        out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert out.last_hidden_state.shape == (2, 8, 16)

    # Save
    save_dir = tmp_path / "mistral3_text_backbone"
    backbone.save_pretrained(str(save_dir))

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["model_type"] == "mistral"

    # Reload and verify weights match
    reloaded = AutoModel.from_pretrained(str(save_dir))
    assert reloaded.config.model_type == "mistral"

    original_sd = backbone.state_dict()
    reloaded_sd = reloaded.state_dict()
    assert set(original_sd.keys()) == set(reloaded_sd.keys())
    for key in original_sd:
        assert torch.equal(original_sd[key], reloaded_sd[key]), f"Weight mismatch for {key}"


def test_extract_submodel_mistral3_vlm_train_step_and_reload(monkeypatch, tmp_path):
    """Extract language_model from VLM → train one step → save → reload →
    verify the trained weights (not the initial ones) survived the round-trip."""
    from nemo_automodel._transformers import retrieval

    vlm_cfg = _tiny_mistral3_vlm_config()
    vlm = AutoModel.from_config(vlm_cfg)

    _mock_auto_classes(monkeypatch, vlm, vlm_cfg)

    backbone = retrieval.build_encoder_backbone(
        model_name_or_path="fake/mistral3-vlm",
        task="embedding",
        extract_submodel="language_model",
    )

    # Snapshot weights before training
    pre_train_sd = {k: v.clone() for k, v in backbone.state_dict().items()}

    # One training step
    backbone.train()
    input_ids = torch.randint(0, 32, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    loss = out.last_hidden_state.mean()
    loss.backward()

    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()

    # Verify weights actually changed
    post_train_sd = backbone.state_dict()
    changed = sum(
        1 for k in pre_train_sd if not torch.equal(pre_train_sd[k], post_train_sd[k])
    )
    assert changed > 0, "No weights changed after training step"

    # Save trained model
    save_dir = tmp_path / "trained_backbone"
    backbone.save_pretrained(str(save_dir))

    # Reload and verify trained weights round-tripped
    reloaded = AutoModel.from_pretrained(str(save_dir))
    reloaded_sd = reloaded.state_dict()

    assert set(post_train_sd.keys()) == set(reloaded_sd.keys())
    for key in post_train_sd:
        assert torch.equal(post_train_sd[key], reloaded_sd[key]), (
            f"Trained weight mismatch for {key}"
        )


# ---------------------------------------------------------------------------
# extract_submodel: real Ministral-3-3B VLM (GPU, downloads weights)
# ---------------------------------------------------------------------------

_MINISTRAL3_VLM = "mistralai/Ministral-3-3B-Instruct-2512"


@pytest.mark.with_downloads
@pytest.mark.run_only_on("GPU")
def test_extract_submodel_real_ministral3_train_save_reload(tmp_path):
    """End-to-end with real Ministral-3-3B-Instruct-2512 VLM weights:
    extract language_model → forward → train one step → save → reload → verify."""
    from nemo_automodel._transformers import retrieval

    device = torch.device("cuda")

    # Load real VLM and extract language_model
    backbone = retrieval.build_encoder_backbone(
        model_name_or_path=_MINISTRAL3_VLM,
        task="embedding",
        extract_submodel="language_model",
        torch_dtype=torch.bfloat16,
    )
    backbone = backbone.to(device)

    # Config must be the text config, not the VLM config (mistral3)
    assert backbone.config.model_type != "mistral3", (
        "Got VLM config model_type='mistral3' — expected the text backbone config"
    )
    assert backbone.config.model_type == "ministral3", (
        f"Expected text config model_type='ministral3', got '{backbone.config.model_type}'"
    )

    # State dict keys must be clean
    for key in backbone.state_dict():
        assert not key.startswith("language_model."), f"VLM prefix in key: {key}"

    # Forward pass
    input_ids = torch.randint(0, backbone.config.vocab_size, (2, 16), device=device)
    attention_mask = torch.ones_like(input_ids)

    backbone.eval()
    with torch.no_grad():
        out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    assert out.last_hidden_state.shape == (2, 16, backbone.config.hidden_size)

    # One training step
    backbone.train()
    out = backbone(input_ids=input_ids, attention_mask=attention_mask)
    loss = out.last_hidden_state.float().mean()
    loss.backward()

    optimizer = torch.optim.SGD(backbone.parameters(), lr=1e-4)
    optimizer.step()
    optimizer.zero_grad()

    # Save trained model
    save_dir = tmp_path / "ministral3_trained"
    backbone.save_pretrained(str(save_dir))

    saved_config = json.loads((save_dir / "config.json").read_text())
    assert saved_config["model_type"] == "ministral3"

    # Reload on CPU and verify weights round-tripped
    reloaded = AutoModel.from_pretrained(str(save_dir), torch_dtype=torch.bfloat16)

    original_sd = {k: v.cpu() for k, v in backbone.state_dict().items()}
    reloaded_sd = reloaded.state_dict()

    assert set(original_sd.keys()) == set(reloaded_sd.keys()), (
        f"Key mismatch: extra={set(original_sd.keys()) - set(reloaded_sd.keys())}, "
        f"missing={set(reloaded_sd.keys()) - set(original_sd.keys())}"
    )
    for key in original_sd:
        assert torch.equal(original_sd[key], reloaded_sd[key]), (
            f"Weight mismatch for {key}: "
            f"max diff = {(original_sd[key].float() - reloaded_sd[key].float()).abs().max().item():.6e}"
        )

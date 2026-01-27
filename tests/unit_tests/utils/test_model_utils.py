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

from __future__ import annotations

from typing import Dict

import pytest
import torch
import torch.nn as nn

import nemo_automodel.components.utils.model_utils as model_utils


@pytest.fixture()
def dummy_model() -> nn.Module:
    """
    Create a minimal but representative model containing:
    - An nn.Embedding layer
    - A `vision_tower` sub-module
    - Another module whose name contains “visual” (to test name-based pattern)
    - A `language_model` backbone
    - An additional unfrozen Linear layer (“other”) for sanity checks
    """

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embed = nn.Embedding(10, 3)  # embeddings
            self.vision_tower = nn.Sequential(nn.Linear(4, 4))  # vision tower attr
            self.visual_extra = nn.Sequential(nn.Linear(5, 5))  # triggers "visual" name pattern
            self.language_model = nn.Sequential(nn.Linear(6, 6))
            self.other = nn.Linear(7, 7)

        def forward(self, x):  # pragma: no cover
            pass

    return DummyModel()


def _all_requires_grad(module: nn.Module) -> bool:
    """Return True if every parameter in `module` requires gradients."""
    return all(p.requires_grad for p in module.parameters())


def _any_requires_grad(module: nn.Module) -> bool:
    """Return True if at least one parameter in `module` requires gradients."""
    return any(p.requires_grad for p in module.parameters())


def test_print_trainable_parameters_counts(dummy_model, caplog, monkeypatch):
    """
    Ensure the helper returns correct (trainable, total) counts
    and prints to stdout only when rank == 0.
    """
    import logging
    caplog.set_level(logging.DEBUG)
    dummy_model.other.weight.requires_grad = False
    trainable, total = model_utils.print_trainable_parameters(dummy_model)

    assert trainable == sum(p.numel() for p in dummy_model.parameters() if p.requires_grad)
    assert total == sum(p.numel() for p in dummy_model.parameters())

    # Check logging output
    assert "Trainable parameters" in caplog.text
    assert "Total parameters" in caplog.text

def test_print_trainable_parameters_non_zero_rank(dummy_model, capsys, monkeypatch):
    """
    Helper must stay silent for non-zero ranks.
    """
    _ = model_utils.print_trainable_parameters(dummy_model)
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize(
    "freeze_cfg, expect",
    [
        (
            {"freeze_embeddings": True, "freeze_vision_tower": False, "freeze_language_model": False},
            {"emb": False, "vision": True, "lang": True, "other": True},
        ),
        (
            {"freeze_embeddings": False, "freeze_vision_tower": True, "freeze_language_model": False},
            {"emb": True, "vision": False, "lang": True, "other": True},
        ),
        (
            {"freeze_embeddings": False, "freeze_vision_tower": False, "freeze_language_model": True},
            {"emb": True, "vision": True, "lang": False, "other": True},
        ),
        (
            {},  # rely on in-code defaults: embeddings=True, vision=True, language=False
            {"emb": False, "vision": False, "lang": True, "other": True},
        ),
    ],
)
def test_apply_parameter_freezing(dummy_model, freeze_cfg: Dict, expect: Dict):
    """
    Parametrized test to verify that each freeze flag affects the right sub-modules.

    `expect` dict uses:
        emb   -> require_grad status for Embedding
        vision-> require_grad status for *all* vision-related parameters
        lang  -> require_grad status for language_model
        other -> require_grad status for the unrelated `other` layer
    A value of True means gradients SHOULD be enabled; False means frozen.
    """
    # Reset all grads before every run (pytest reuses the same fixture instance)
    for p in dummy_model.parameters():
        p.requires_grad = True

    model_utils.apply_parameter_freezing(dummy_model, freeze_cfg)

    # embeddings
    assert dummy_model.token_embed.weight.requires_grad is expect["emb"]

    # vision tower(s)
    assert _all_requires_grad(dummy_model.vision_tower) is expect["vision"]
    assert _all_requires_grad(dummy_model.visual_extra) is expect["vision"]

    # language model
    assert _all_requires_grad(dummy_model.language_model) is expect["lang"]

    # unrelated layer
    assert dummy_model.other.weight.requires_grad is expect["other"]
    assert dummy_model.other.bias.requires_grad is expect["other"]


def test_init_empty_weights_moves_params_to_meta_and_preserves_requires_grad():
    """
    Creating parameters inside the context should place them on meta device and
    preserve their requires_grad flags.
    """

    class CustomModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty(2, 3))  # requires_grad=True by default
            self.b = nn.Parameter(torch.empty(3), requires_grad=False)

    with model_utils.init_empty_weights():
        m = CustomModule()

    # Device moved to meta
    assert m.w.device.type == "meta"
    assert m.b.device.type == "meta"
    # requires_grad preserved
    assert m.w.requires_grad is True
    assert m.b.requires_grad is False
    # shapes and types preserved
    assert m.w.shape == (2, 3)
    assert isinstance(m.w, nn.Parameter)
    assert isinstance(m.b, nn.Parameter)


def test_init_empty_weights_restores_register_parameter_on_exception():
    """
    Even if an exception occurs inside the context, the original register_parameter
    must be restored.
    """
    original = nn.Module.register_parameter
    with pytest.raises(RuntimeError):
        with model_utils.init_empty_weights():
            raise RuntimeError("boom")
    assert nn.Module.register_parameter is original


def test_init_empty_weights_torchao_branch_with_fake_weight(monkeypatch):
    """
    Simulate the torchao branch by monkeypatching torch_ao and providing a
    fake WeightWithDynamicFloat8CastTensor that subclasses nn.Parameter.
    Verify:
      - parameters are moved to meta
      - requires_grad is preserved based on the wrapped tensor
      - mapped attributes (_linear_mm_config, _dtype, _precomputed_scale) are copied through
    """

    class FakeWeight(nn.Parameter):
        def __new__(
            cls,
            tensor: torch.Tensor,
            linear_mm_config=None,
            dtype=None,
            precomputed_scale=None,
        ):
            # Mirror torchao behavior: requires_grad comes from the wrapped tensor
            obj = nn.Parameter.__new__(cls, tensor, requires_grad=tensor.requires_grad)
            # store with underscore names to match model_utils' mapping lookup
            obj._linear_mm_config = linear_mm_config
            obj._dtype = dtype
            obj._precomputed_scale = precomputed_scale
            return obj

    class _DummyFSDPUtils:
        WeightWithDynamicFloat8CastTensor = FakeWeight

    class _DummyFloat8:
        fsdp_utils = _DummyFSDPUtils

    class _DummyTorchAO:
        float8 = _DummyFloat8

    monkeypatch.setattr(model_utils, "HAVE_TORCHAO", True, raising=True)
    monkeypatch.setattr(model_utils, "torch_ao", _DummyTorchAO, raising=True)

    class Mod(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # create a FakeWeight so isinstance(..., WeightWithDynamicFloat8CastTensor) is True
            base = torch.empty(3, requires_grad=False)
            # requires_grad False to ensure preservation through the branch
            self.p = FakeWeight(
                base,
                "cfg",
                torch.float32,
                torch.tensor(1.0),
            )

    with model_utils.init_empty_weights():
        m = Mod()

    # type preserved as FakeWeight
    assert isinstance(m.p, FakeWeight)
    # moved to meta
    assert m.p.device.type == "meta"
    # requires_grad preserved from wrapped tensor
    assert m.p.requires_grad is False
    # mapped attributes preserved via mapping
    assert getattr(m.p, "_linear_mm_config") == "cfg"
    assert getattr(m.p, "_dtype") == torch.float32
    assert isinstance(getattr(m.p, "_precomputed_scale"), torch.Tensor)


class TestGetTextModule:
    """Tests for get_text_module function."""

    def test_returns_language_model_when_present(self):
        """Test that language_model attribute is returned when present."""
        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Linear(10, 10)
                self.visual = nn.Linear(5, 5)

        model = VLMModel()
        result = model_utils.get_text_module(model)
        assert result is model.language_model

    def test_returns_text_model_when_present(self):
        """Test that text_model attribute is returned when present."""
        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_model = nn.Linear(10, 10)
                self.vision_encoder = nn.Linear(5, 5)

        model = VLMModel()
        result = model_utils.get_text_module(model)
        assert result is model.text_model

    def test_returns_text_decoder_when_present(self):
        """Test that text_decoder attribute is returned when present."""
        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.text_decoder = nn.Linear(10, 10)

        model = VLMModel()
        result = model_utils.get_text_module(model)
        assert result is model.text_decoder

    def test_returns_model_when_no_text_attr(self):
        """Test that model itself is returned when no text module attribute exists."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Linear(10, 10)

        model = SimpleModel()
        result = model_utils.get_text_module(model)
        assert result is model

    def test_returns_none_when_model_is_none(self):
        """Test that None is returned when model is None."""
        result = model_utils.get_text_module(None)
        assert result is None

    def test_priority_order_language_model_first(self):
        """Test that language_model has priority over text_model."""
        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = nn.Linear(10, 10)
                self.text_model = nn.Linear(5, 5)

        model = VLMModel()
        result = model_utils.get_text_module(model)
        assert result is model.language_model

    def test_skips_none_attribute(self):
        """Test that None attributes are skipped."""
        class VLMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = None
                self.text_model = nn.Linear(10, 10)

        model = VLMModel()
        result = model_utils.get_text_module(model)
        assert result is model.text_model


class TestConstants:
    """Tests for TEXT_MODULE_ATTRS and MULTIMODAL_SUFFIXES constants."""

    def test_text_module_attrs_contains_expected_values(self):
        """Test TEXT_MODULE_ATTRS contains the expected attribute names."""
        assert "language_model" in model_utils.TEXT_MODULE_ATTRS
        assert "text_model" in model_utils.TEXT_MODULE_ATTRS
        assert "text_decoder" in model_utils.TEXT_MODULE_ATTRS

    def test_multimodal_suffixes_contains_vision_attrs(self):
        """Test MULTIMODAL_SUFFIXES contains vision-related suffixes."""
        assert "vision_tower" in model_utils.MULTIMODAL_SUFFIXES
        assert "visual" in model_utils.MULTIMODAL_SUFFIXES
        assert "image_encoder" in model_utils.MULTIMODAL_SUFFIXES
        assert "vision_encoder" in model_utils.MULTIMODAL_SUFFIXES

    def test_multimodal_suffixes_contains_audio_attrs(self):
        """Test MULTIMODAL_SUFFIXES contains audio-related suffixes."""
        assert "audio_tower" in model_utils.MULTIMODAL_SUFFIXES
        assert "audio_encoder" in model_utils.MULTIMODAL_SUFFIXES
        assert "audio_model" in model_utils.MULTIMODAL_SUFFIXES

    def test_multimodal_suffixes_contains_projector_attrs(self):
        """Test MULTIMODAL_SUFFIXES contains projector-related suffixes."""
        assert "mm_projector" in model_utils.MULTIMODAL_SUFFIXES
        assert "multi_modal_projector" in model_utils.MULTIMODAL_SUFFIXES
        assert "multimodal_projector" in model_utils.MULTIMODAL_SUFFIXES

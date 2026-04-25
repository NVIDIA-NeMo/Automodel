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
            # freeze_vision_tower=False means vision stays trainable
            {"freeze_vision_tower": False, "freeze_language_model": False},
            {"vision": True, "lang": True, "other": True},
        ),
        (
            # freeze_vision_tower=True means vision is frozen
            {"freeze_vision_tower": True, "freeze_language_model": False},
            {"vision": False, "lang": True, "other": True},
        ),
        (
            # freeze_language_model=True means language_model is frozen
            {"freeze_vision_tower": False, "freeze_language_model": True},
            {"vision": True, "lang": False, "other": True},
        ),
        (
            # defaults: freeze_vision_tower=True, freeze_language_model=False
            {},
            {"vision": False, "lang": True, "other": True},
        ),
    ],
)
def test_apply_parameter_freezing(dummy_model, freeze_cfg: Dict, expect: Dict):
    """
    Parametrized test to verify that each freeze flag affects the right sub-modules.

    `expect` dict uses:
        vision-> require_grad status for *all* vision-related parameters
        lang  -> require_grad status for language_model
        other -> require_grad status for the unrelated `other` layer
    A value of True means gradients SHOULD be enabled; False means frozen.

    Note: freeze_embeddings was removed from apply_parameter_freezing.
    Embeddings are no longer frozen by this function.
    """
    # Reset all grads before every run (pytest reuses the same fixture instance)
    for p in dummy_model.parameters():
        p.requires_grad = True

    model_utils.apply_parameter_freezing(dummy_model, freeze_cfg)

    # vision tower(s)
    assert _all_requires_grad(dummy_model.vision_tower) is expect["vision"]
    assert _all_requires_grad(dummy_model.visual_extra) is expect["vision"]

    # language model
    assert _all_requires_grad(dummy_model.language_model) is expect["lang"]

    # unrelated layer (including embeddings - not frozen by this function)
    assert dummy_model.other.weight.requires_grad is expect["other"]
    assert dummy_model.other.bias.requires_grad is expect["other"]
    # embeddings are always trainable now (freeze_embeddings was removed)
    assert dummy_model.token_embed.weight.requires_grad is True


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


def test_init_empty_weights_preserves_is_hf_initialized_attribute():
    """
    Test that _is_hf_initialized attribute is preserved when moving parameters
    to meta device. This attribute is used by HuggingFace transformers to track
    which parameters have been initialized.
    """

    class ModuleWithHFInitializedParam(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty(2, 3))
            # Simulate HF setting _is_hf_initialized on the parameter
            self.w._is_hf_initialized = True

    with model_utils.init_empty_weights():
        m = ModuleWithHFInitializedParam()

    # Device moved to meta
    assert m.w.device.type == "meta"
    # _is_hf_initialized should be preserved
    assert hasattr(m.w, "_is_hf_initialized")
    assert m.w._is_hf_initialized is True


def test_init_empty_weights_handles_missing_is_hf_initialized():
    """
    Test that parameters without _is_hf_initialized attribute work correctly
    and don't have the attribute added spuriously.
    """

    class ModuleWithoutHFInitialized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty(2, 3))
            # No _is_hf_initialized attribute set

    with model_utils.init_empty_weights():
        m = ModuleWithoutHFInitialized()

    # Device moved to meta
    assert m.w.device.type == "meta"
    # _is_hf_initialized should NOT be present since it wasn't set originally
    assert not hasattr(m.w, "_is_hf_initialized")


# =============================================================================
# Tests for freeze_audio_tower with "speech" pattern
# =============================================================================


@pytest.fixture()
def audio_model() -> nn.Module:
    """Model with audio and speech submodules for freeze tests."""

    class AudioModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_encoder = nn.Linear(4, 4)
            self.speech_adapter = nn.Linear(4, 4)
            self.language_head = nn.Linear(4, 4)

        def forward(self, x):
            pass

    return AudioModel()


def test_freeze_audio_tower_freezes_speech_pattern(audio_model):
    """freeze_audio_tower=True should freeze modules matching 'audio' and 'speech'."""
    model_utils.apply_parameter_freezing(audio_model, {"freeze_audio_tower": True, "freeze_vision_tower": False})

    assert not _any_requires_grad(audio_model.audio_encoder)
    assert not _any_requires_grad(audio_model.speech_adapter)
    assert _all_requires_grad(audio_model.language_head)


def test_freeze_audio_tower_false_keeps_speech_trainable(audio_model):
    """freeze_audio_tower=False should keep speech modules trainable."""
    model_utils.apply_parameter_freezing(audio_model, {"freeze_audio_tower": False, "freeze_vision_tower": False})

    assert _all_requires_grad(audio_model.audio_encoder)
    assert _all_requires_grad(audio_model.speech_adapter)


# =============================================================================
# Tests for cast_mixed_dtype_params_to_bf16
# =============================================================================


def test_cast_mixed_dtype_params_to_bf16():
    """cast_mixed_dtype_params_to_bf16 converts fp32 params and buffers to bf16."""
    m = nn.Linear(4, 4)  # fp32 by default
    m.register_buffer("my_buf", torch.ones(3, dtype=torch.float32))
    assert m.weight.dtype == torch.float32
    assert m.my_buf.dtype == torch.float32

    model_utils.cast_mixed_dtype_params_to_bf16(m)

    assert m.weight.dtype == torch.bfloat16
    assert m.bias.dtype == torch.bfloat16
    assert m.my_buf.dtype == torch.bfloat16


def test_cast_mixed_dtype_preserves_bf16():
    """Already-bf16 params should not be changed."""
    m = nn.Linear(4, 4).to(torch.bfloat16)
    model_utils.cast_mixed_dtype_params_to_bf16(m)
    assert m.weight.dtype == torch.bfloat16


# =============================================================================
# Tests for phi4mm-specific logic in apply_parameter_freezing
# =============================================================================


def test_phi4mm_use_cache_disabled():
    """For phi4mm models, apply_parameter_freezing sets use_cache=False."""
    import types

    m = nn.Linear(4, 4)
    m.config = types.SimpleNamespace(model_type="phi4mm", use_cache=True)

    model_utils.apply_parameter_freezing(m, {"freeze_vision_tower": False})
    assert m.config.use_cache is False


def test_non_phi4mm_use_cache_unchanged():
    """For non-phi4mm models, use_cache should not be changed."""
    import types

    m = nn.Linear(4, 4)
    m.config = types.SimpleNamespace(model_type="gemma3", use_cache=True)

    model_utils.apply_parameter_freezing(m, {"freeze_vision_tower": False})
    assert m.config.use_cache is True


class TestFilterForwardKwargs:
    def test_supports_padding_mask_when_explicit(self):
        class Model(nn.Module):
            def forward(self, input_ids, padding_mask=None):
                return input_ids

        assert model_utils._supports_padding_mask(Model()) is True

    def test_supports_padding_mask_when_var_keyword(self):
        class Model(nn.Module):
            def forward(self, input_ids, **kwargs):
                return input_ids

        assert model_utils._supports_padding_mask(Model()) is True

    def test_does_not_support_padding_mask_when_missing(self):
        class Model(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                return input_ids

        assert model_utils._supports_padding_mask(Model()) is False

    def test_drops_unsupported_kwargs(self):
        class Model(nn.Module):
            def forward(self, input_ids, attention_mask=None):
                return input_ids

        model = Model()
        batch = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        }

        filtered = model_utils.filter_forward_kwargs(model, batch)

        assert set(filtered.keys()) == {"input_ids", "attention_mask"}
        assert "padding_mask" in batch  # original dict is unchanged

    def test_keeps_kwargs_when_forward_accepts_var_keyword(self):
        class Model(nn.Module):
            def forward(self, input_ids, **kwargs):
                return input_ids, kwargs

        model = Model()
        batch = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "padding_mask": torch.zeros(2, 3, dtype=torch.bool),
        }

        filtered = model_utils.filter_forward_kwargs(model, batch)

        assert filtered == batch

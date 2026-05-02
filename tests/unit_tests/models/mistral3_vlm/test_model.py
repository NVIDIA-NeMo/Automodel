# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for Mistral3FP8VLMForConditionalGeneration class behavior.

These tests avoid instantiating the full 128B model. They validate:
  * supports_config() pattern matrix
  * _skip_init_weights_on_load class attribute (load-bearing in checkpointing)
  * The dequantize=True flip on quantization_config (skips HF's FP8Linear swap)
  * State-dict adapter attached on construction
  * _rotary_reinit_self_hook is idempotent and recovers inv_freq via the
    class's own __init__
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.mistral3_vlm.model import (
    Mistral3FP8VLMForConditionalGeneration,
    _rotary_reinit_self_hook,
)
from nemo_automodel.components.models.mistral3_vlm.state_dict_adapter import (
    Mistral3FP8StateDictAdapter,
)


# --------------------------------------------------------------------------- #
# supports_config                                                             #
# --------------------------------------------------------------------------- #
class TestSupportsConfig:
    """Claim only FP8-native Mistral3 VLM configs."""

    def test_fp8_mistral3_vlm_config_supported_via_dict(self):
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(model_type="ministral3"),
            quantization_config={"quant_method": "fp8"},
        )
        assert Mistral3FP8VLMForConditionalGeneration.supports_config(cfg) is True

    def test_fp8_mistral3_vlm_config_supported_via_object(self):
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(model_type="ministral3"),
            quantization_config=SimpleNamespace(quant_method="fp8"),
        )
        assert Mistral3FP8VLMForConditionalGeneration.supports_config(cfg) is True

    def test_no_text_config_rejected(self):
        cfg = SimpleNamespace(quantization_config={"quant_method": "fp8"})
        # text_config is None → reject (text_config attr missing → getattr returns None)
        assert Mistral3FP8VLMForConditionalGeneration.supports_config(cfg) is False

    def test_text_config_other_model_type_rejected(self):
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(model_type="llama"),
            quantization_config={"quant_method": "fp8"},
        )
        assert Mistral3FP8VLMForConditionalGeneration.supports_config(cfg) is False

    def test_no_quantization_config_rejected(self):
        cfg = SimpleNamespace(text_config=SimpleNamespace(model_type="ministral3"))
        assert Mistral3FP8VLMForConditionalGeneration.supports_config(cfg) is False

    def test_non_fp8_quantization_method_rejected(self):
        cfg = SimpleNamespace(
            text_config=SimpleNamespace(model_type="ministral3"),
            quantization_config={"quant_method": "awq"},
        )
        assert Mistral3FP8VLMForConditionalGeneration.supports_config(cfg) is False


# --------------------------------------------------------------------------- #
# Class attribute used as a checkpointing gate                                #
# --------------------------------------------------------------------------- #
class TestSkipInitWeightsOnLoad:
    def test_class_attr_set_to_true(self):
        # Used by Checkpointer.initialize_model_weights to opt out of HF's
        # initialize_weights() — without this, PP setups deadlock on
        # stage-divergent DTensor collectives during HF init.
        assert Mistral3FP8VLMForConditionalGeneration._skip_init_weights_on_load is True


# --------------------------------------------------------------------------- #
# __init__ side effects                                                       #
# --------------------------------------------------------------------------- #
def _make_dummy_init_super():
    """Returns a no-op replacement for HF's heavy __init__ that records the call."""
    captured = {}

    def _init(self, config):
        captured["config"] = config
        # Minimal nn.Module bookkeeping so subsequent attribute access doesn't fail.
        nn.Module.__init__(self)

    return _init, captured


class _DictQC(dict):
    """Marker dict subclass; used to keep test wiring explicit."""


@pytest.fixture
def patched_super_init():
    """Patch the HF parent's __init__ so we can construct our class without
    standing up the full 128B model."""
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    fake_init, captured = _make_dummy_init_super()
    with patch.object(Mistral3ForConditionalGeneration, "__init__", fake_init):
        yield captured


class TestInitFlipsDequantize:
    """The class flips quantization_config.dequantize=True before super().__init__,
    so HF's replace_with_fp8_linear early-returns and our adapter owns FP8 dequant."""

    def test_dict_quantization_config_dequantize_set_true(self, patched_super_init):
        qc = {"quant_method": "fp8"}
        cfg = SimpleNamespace(quantization_config=qc)
        Mistral3FP8VLMForConditionalGeneration(cfg)
        assert qc["dequantize"] is True

    def test_object_quantization_config_dequantize_set_true(self, patched_super_init):
        qc = SimpleNamespace(quant_method="fp8")
        cfg = SimpleNamespace(quantization_config=qc)
        Mistral3FP8VLMForConditionalGeneration(cfg)
        assert qc.dequantize is True

    def test_no_quantization_config_does_not_crash(self, patched_super_init):
        cfg = SimpleNamespace(quantization_config=None)
        Mistral3FP8VLMForConditionalGeneration(cfg)  # should not raise


class TestInitAttachesAdapter:
    def test_state_dict_adapter_for_vlm_full(self, patched_super_init):
        qc = {"quant_method": "fp8"}
        cfg = SimpleNamespace(quantization_config=qc)
        m = Mistral3FP8VLMForConditionalGeneration(cfg)
        assert isinstance(m.state_dict_adapter, Mistral3FP8StateDictAdapter)
        assert m.state_dict_adapter._layout_name == "vlm_full"


class TestInitRegistersRotaryHooks:
    def test_hook_registered_on_modules_with_inv_freq(self, patched_super_init):
        # Simulate that super().__init__ already created the rotary submodules
        # (they registered `inv_freq` buffers). We add them after super so the
        # post-super loop in our __init__ picks them up.
        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

        # Replace HF's __init__ with one that DOES set up two rotary submodules
        # with inv_freq buffers — that's what our hook iterates over.
        def _init_with_rotaries(self, config):
            nn.Module.__init__(self)
            self.text_rotary = nn.Module()
            self.text_rotary.register_buffer("inv_freq", torch.zeros(4), persistent=False)
            self.vision_rotary = nn.Module()
            self.vision_rotary.register_buffer("inv_freq", torch.zeros(4), persistent=False)

        with patch.object(Mistral3ForConditionalGeneration, "__init__", _init_with_rotaries):
            m = Mistral3FP8VLMForConditionalGeneration(SimpleNamespace(quantization_config=None))

        # Both rotary modules got the one-shot reinit pre-hook registered.
        assert len(m.text_rotary._forward_pre_hooks) == 1
        assert len(m.vision_rotary._forward_pre_hooks) == 1
        # Sentinel to gate the one-shot reinit
        assert m.text_rotary._mistral3_fp8_rotary_reinit_done is False
        assert m.vision_rotary._mistral3_fp8_rotary_reinit_done is False


# --------------------------------------------------------------------------- #
# _rotary_reinit_self_hook                                                    #
# --------------------------------------------------------------------------- #
class _DummyRotary(nn.Module):
    """Minimal rotary stand-in: registers inv_freq in __init__ from `config.x`.

    Our hook re-runs ``type(module).__init__(module, module.config, device=device)``,
    so calling it twice should overwrite inv_freq idempotently.
    """

    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        x = float(getattr(config, "x", 0.0))
        self.register_buffer("inv_freq", torch.tensor([x], device=device), persistent=False)


class TestRotaryReinitSelfHook:
    def test_hook_is_idempotent_and_marks_done(self):
        cfg = SimpleNamespace(x=42.0)
        rot = _DummyRotary(cfg)
        # Pretend init produced an "uninitialized" buffer — overwrite to a
        # sentinel value, then check that the hook restores via __init__.
        rot.inv_freq.fill_(0.0)
        assert rot._buffers["inv_freq"].item() == 0.0
        rot._mistral3_fp8_rotary_reinit_done = False

        _rotary_reinit_self_hook(rot, args=(), kwargs={})
        # First fire: re-ran __init__, recomputed inv_freq from config.x.
        assert rot.inv_freq.item() == 42.0
        assert rot._mistral3_fp8_rotary_reinit_done is True

        # Second fire is a no-op (idempotent gate).
        rot.inv_freq.fill_(0.0)
        _rotary_reinit_self_hook(rot, args=(), kwargs={})
        assert rot.inv_freq.item() == 0.0  # untouched

    def test_hook_swallows_init_errors(self):
        # If type(module).__init__ raises, the hook must mark done and return
        # rather than propagating — a rotary class we don't recognize should
        # not crash the forward.
        class _BadRotary(nn.Module):
            def __init__(self, config, device=None):
                # First call (in test setup) succeeds; second call (from hook)
                # raises a non-TypeError to exercise the bare except branch.
                super().__init__()
                self.config = config
                self.register_buffer("inv_freq", torch.zeros(2), persistent=False)
                if getattr(config, "_called_once", False):
                    raise RuntimeError("boom")
                config._called_once = True

        cfg = SimpleNamespace()
        rot = _BadRotary(cfg)
        rot._mistral3_fp8_rotary_reinit_done = False
        # Should not raise.
        _rotary_reinit_self_hook(rot, args=(), kwargs={})
        assert rot._mistral3_fp8_rotary_reinit_done is True

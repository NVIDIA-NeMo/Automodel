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

"""Unit tests for the NemotronOmni state-dict adapter.

The adapter sits between the HF ``NemotronH_Nano_Omni_Reasoning_V3`` checkpoint
layout and the custom Automodel layout. The LLM portion is delegated to
``NemotronV3StateDictAdapter`` (already covered by its own tests), so we mock
that delegate here and focus on the vision/audio key remapping that this PR
introduces.
"""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.nemotron_omni.state_dict_adapter import (
    _VISION_PROJ_CUSTOM_TO_HF,
    _VISION_PROJ_HF_TO_CUSTOM,
    NemotronOmniStateDictAdapter,
)


@pytest.fixture
def adapter():
    """Adapter with the LLM delegate mocked out (LLM remap tested elsewhere)."""
    a = NemotronOmniStateDictAdapter.__new__(NemotronOmniStateDictAdapter)
    a.config = MagicMock()
    a.llm_config = MagicMock()
    a.moe_config = MagicMock()
    a.backend = MagicMock()
    a.dtype = torch.bfloat16
    # Echo input keys through unchanged — lets us verify *only* the
    # NemotronOmni-side key handling without depending on NemotronV3 internals.
    a._llm_adapter = MagicMock()
    a._llm_adapter.from_hf.side_effect = lambda sd, **kwargs: dict(sd)
    a._llm_adapter.to_hf.side_effect = lambda sd, **kwargs: dict(sd)
    a._llm_adapter.convert_single_tensor_to_hf.side_effect = lambda fqn, tensor, **kwargs: [(fqn, tensor)]
    return a


# ---------------------------------------------------------------------------
# from_hf — HF layout -> Automodel layout
# ---------------------------------------------------------------------------


def test_from_hf_remaps_vision_projector(adapter):
    """``mlp1.{0,1,3}.weight`` should become ``vision_projector.{norm,linear1,linear2}.weight``."""
    hf_sd = {
        "mlp1.0.weight": torch.zeros(8),
        "mlp1.1.weight": torch.zeros(16, 8),
        "mlp1.3.weight": torch.zeros(4, 16),
    }
    out = adapter.from_hf(dict(hf_sd))

    assert "vision_projector.norm.weight" in out
    assert "vision_projector.linear1.weight" in out
    assert "vision_projector.linear2.weight" in out
    # No HF-style mlp1.* keys leak through.
    assert not any(k.startswith("mlp1.") for k in out)


def test_from_hf_passes_vision_model_through(adapter):
    """``vision_model.*`` keys (the RADIO encoder) are emitted unchanged."""
    hf_sd = {
        "vision_model.radio_model.model.blocks.0.norm1.weight": torch.zeros(2),
        "vision_model.radio_model.input_conditioner.norm_mean": torch.zeros(3),
    }
    out = adapter.from_hf(dict(hf_sd))
    assert set(out) == set(hf_sd)


def test_from_hf_strips_sound_encoder_inner_prefix(adapter):
    """``sound_encoder.encoder.*`` -> ``sound_encoder.*`` (Parakeet wrapper unwrap)."""
    hf_sd = {
        "sound_encoder.encoder.layers.0.weight": torch.zeros(4),
    }
    out = adapter.from_hf(dict(hf_sd))
    assert "sound_encoder.layers.0.weight" in out
    assert "sound_encoder.encoder.layers.0.weight" not in out


def test_from_hf_passes_sound_projection_through(adapter):
    """``sound_projection.*`` keys are identical between HF and custom layouts."""
    hf_sd = {
        "sound_projection.norm.weight": torch.zeros(8),
        "sound_projection.linear1.weight": torch.zeros(16, 8),
        "sound_projection.linear2.weight": torch.zeros(4, 16),
    }
    out = adapter.from_hf(dict(hf_sd))
    assert set(out) == set(hf_sd)


def test_from_hf_routes_llm_through_v3_adapter_with_prefix(adapter):
    """``language_model.*`` keys are stripped, delegated, then re-prefixed."""
    hf_sd = {
        "language_model.backbone.embeddings.weight": torch.zeros(2),
        "language_model.lm_head.weight": torch.zeros(2),
    }
    out = adapter.from_hf(dict(hf_sd))

    # The mock delegate echoes input — keys come back with the prefix re-added.
    assert "language_model.backbone.embeddings.weight" in out
    assert "language_model.lm_head.weight" in out
    # Delegate was called with stripped keys.
    args, _ = adapter._llm_adapter.from_hf.call_args
    delegated = args[0]
    assert "backbone.embeddings.weight" in delegated
    assert "lm_head.weight" in delegated
    assert not any(k.startswith("language_model.") for k in delegated)


# ---------------------------------------------------------------------------
# to_hf — Automodel layout -> HF layout (round-trip)
# ---------------------------------------------------------------------------


def test_to_hf_reverses_vision_projector_remap(adapter):
    custom_sd = {
        "vision_projector.norm.weight": torch.zeros(8),
        "vision_projector.linear1.weight": torch.zeros(16, 8),
        "vision_projector.linear2.weight": torch.zeros(4, 16),
    }
    out = adapter.to_hf(dict(custom_sd))
    assert "mlp1.0.weight" in out
    assert "mlp1.1.weight" in out
    assert "mlp1.3.weight" in out
    assert not any(k.startswith("vision_projector.") for k in out)


def test_to_hf_re_adds_sound_encoder_inner_prefix(adapter):
    custom_sd = {"sound_encoder.layers.0.weight": torch.zeros(4)}
    out = adapter.to_hf(dict(custom_sd))
    assert "sound_encoder.encoder.layers.0.weight" in out


def test_round_trip_vision_projector(adapter):
    """from_hf -> to_hf yields the original HF keys for the vision projector."""
    hf_sd = {
        "mlp1.0.weight": torch.tensor([1.0, 2.0]),
        "mlp1.1.weight": torch.tensor([[3.0]]),
        "mlp1.3.weight": torch.tensor([[4.0]]),
    }
    custom = adapter.from_hf(dict(hf_sd))
    back = adapter.to_hf(custom)
    assert set(back) == set(hf_sd)
    for k in hf_sd:
        assert torch.equal(back[k], hf_sd[k])


def test_round_trip_sound_encoder(adapter):
    """from_hf -> to_hf preserves ``sound_encoder.encoder.*`` keys."""
    hf_sd = {"sound_encoder.encoder.layers.3.bias": torch.tensor([0.5])}
    custom = adapter.from_hf(dict(hf_sd))
    back = adapter.to_hf(custom)
    assert set(back) == set(hf_sd)


def test_to_hf_exclude_key_regex(adapter):
    """``exclude_key_regex`` should filter out matching keys before remap."""
    custom_sd = {
        "vision_projector.norm.weight": torch.zeros(2),
        "sound_projection.norm.weight": torch.zeros(2),
    }
    out = adapter.to_hf(dict(custom_sd), exclude_key_regex=r"sound_projection\..*")
    assert "mlp1.0.weight" in out
    assert "sound_projection.norm.weight" not in out


# ---------------------------------------------------------------------------
# convert_single_tensor_to_hf — single-key path used by streaming save
# ---------------------------------------------------------------------------


def test_convert_single_tensor_vision_projector(adapter):
    pairs = adapter.convert_single_tensor_to_hf("vision_projector.linear1.weight", torch.zeros(2, 2))
    assert pairs == [("mlp1.1.weight", pairs[0][1])]


def test_convert_single_tensor_sound_encoder(adapter):
    pairs = adapter.convert_single_tensor_to_hf("sound_encoder.layers.0.weight", torch.zeros(2))
    new_fqn, _ = pairs[0]
    assert new_fqn == "sound_encoder.encoder.layers.0.weight"


def test_convert_single_tensor_pass_through(adapter):
    """``vision_model.*`` and ``sound_projection.*`` keep their original FQNs."""
    pairs_v = adapter.convert_single_tensor_to_hf("vision_model.radio_model.x", torch.zeros(1))
    assert pairs_v[0][0] == "vision_model.radio_model.x"

    pairs_s = adapter.convert_single_tensor_to_hf("sound_projection.norm.weight", torch.zeros(1))
    assert pairs_s[0][0] == "sound_projection.norm.weight"


def test_convert_single_tensor_llm_delegates(adapter):
    """LLM keys are stripped, delegated, then re-prefixed."""
    pairs = adapter.convert_single_tensor_to_hf("language_model.lm_head.weight", torch.zeros(1))
    assert pairs[0][0] == "language_model.lm_head.weight"
    args, _ = adapter._llm_adapter.convert_single_tensor_to_hf.call_args
    assert args[0] == "lm_head.weight"


# ---------------------------------------------------------------------------
# Internal mapping table sanity
# ---------------------------------------------------------------------------


def test_mapping_tables_are_inverses():
    """``_VISION_PROJ_CUSTOM_TO_HF`` is the inverse of ``_VISION_PROJ_HF_TO_CUSTOM``."""
    for hf_key, custom_key in _VISION_PROJ_HF_TO_CUSTOM.items():
        assert _VISION_PROJ_CUSTOM_TO_HF[custom_key] == hf_key
    assert len(_VISION_PROJ_HF_TO_CUSTOM) == len(_VISION_PROJ_CUSTOM_TO_HF)

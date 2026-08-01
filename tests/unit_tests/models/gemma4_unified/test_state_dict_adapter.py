# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel._transformers.model_init import _maybe_attach_hf_native_state_dict_adapter
from nemo_automodel.components.models.gemma4_unified.state_dict_adapter import Gemma4UnifiedStateDictAdapter

GEMMA4_UNIFIED_KEY_PAIRS = [
    ("embed_vision.patch_ln1.weight", "vision_embedder.patch_ln1.weight"),
    ("embed_vision.patch_dense.bias", "vision_embedder.patch_dense.bias"),
    ("embed_vision.patch_ln2.weight", "vision_embedder.patch_ln2.weight"),
    ("embed_vision.pos_embedding", "vision_embedder.pos_embedding"),
    ("embed_vision.pos_norm.bias", "vision_embedder.pos_norm.bias"),
    (
        "embed_vision.multimodal_embedder.embedding_projection.weight",
        "embed_vision.embedding_projection.weight",
    ),
    (
        "model.embed_vision.multimodal_embedder.embedding_projection.weight",
        "model.embed_vision.embedding_projection.weight",
    ),
]


class TestGemma4UnifiedStateDictAdapter:
    @pytest.mark.parametrize(("model_key", "hf_key"), GEMMA4_UNIFIED_KEY_PAIRS)
    def test_to_hf_restores_hf_checkpoint_key(self, model_key, hf_key):
        tensor = torch.ones(1)

        assert Gemma4UnifiedStateDictAdapter().to_hf({model_key: tensor}) == {hf_key: tensor}

    @pytest.mark.parametrize(("model_key", "hf_key"), GEMMA4_UNIFIED_KEY_PAIRS)
    def test_round_trips_back_to_model_keys(self, model_key, hf_key):
        """Save writes HF FQNs; both base-model init and DCP resume must land back on model FQNs."""
        adapter = Gemma4UnifiedStateDictAdapter()
        tensor = torch.ones(1)

        exported = adapter.to_hf({model_key: tensor})

        assert exported == {hf_key: tensor}
        assert adapter.from_hf(exported) == {model_key: tensor}

    def test_leaves_unrelated_keys_untouched(self):
        adapter = Gemma4UnifiedStateDictAdapter()
        state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.ones(1), "lm_head.weight": torch.zeros(1)}

        assert adapter.to_hf(state_dict) == state_dict
        assert adapter.from_hf(state_dict) == state_dict

    def test_to_hf_drops_excluded_keys(self):
        tensor = torch.ones(1)
        state_dict = {
            "embed_vision.patch_ln1.weight": tensor,
            "model.layers.0.self_attn._extra_state": torch.zeros(1),
        }

        exported = Gemma4UnifiedStateDictAdapter().to_hf(state_dict, exclude_key_regex=r".*_extra_state.*")

        assert exported == {"vision_embedder.patch_ln1.weight": tensor}

    def test_rejects_export_key_collision(self):
        state_dict = {
            "embed_vision.embedding_projection.weight": torch.ones(1),
            "embed_vision.multimodal_embedder.embedding_projection.weight": torch.zeros(1),
        }

        with pytest.raises(ValueError, match="key collision"):
            Gemma4UnifiedStateDictAdapter().to_hf(state_dict)


class TestAttachHFNativeStateDictAdapter:
    def test_attaches_for_gemma4_unified(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))

        _maybe_attach_hf_native_state_dict_adapter(model)

        assert isinstance(model.state_dict_adapter, Gemma4UnifiedStateDictAdapter)

    def test_leaves_other_model_types_alone(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4"))

        _maybe_attach_hf_native_state_dict_adapter(model)

        assert not hasattr(model, "state_dict_adapter")

    def test_keeps_an_existing_adapter(self):
        existing = object()
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"), state_dict_adapter=existing)

        _maybe_attach_hf_native_state_dict_adapter(model)

        assert model.state_dict_adapter is existing

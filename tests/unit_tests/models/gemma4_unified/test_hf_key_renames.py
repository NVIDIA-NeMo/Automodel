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

from nemo_automodel.components.models.gemma4_unified.hf_key_renames import maybe_rename_gemma4_unified_keys

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


class TestGemma4UnifiedHFExportKeys:
    @pytest.mark.parametrize(("model_key", "hf_key"), GEMMA4_UNIFIED_KEY_PAIRS)
    def test_restores_hf_checkpoint_key(self, model_key, hf_key):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))
        tensor = torch.ones(1)

        result = maybe_rename_gemma4_unified_keys(model, {model_key: tensor}, to_hf=True)

        assert result == {hf_key: tensor}

    @pytest.mark.parametrize(("model_key", "hf_key"), GEMMA4_UNIFIED_KEY_PAIRS)
    def test_round_trips_back_to_model_keys(self, model_key, hf_key):
        """A DCP resume renames to HF FQNs on load and must land back on the model FQNs."""
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))
        tensor = torch.ones(1)

        exported = maybe_rename_gemma4_unified_keys(model, {model_key: tensor}, to_hf=True)
        restored = maybe_rename_gemma4_unified_keys(model, exported, to_hf=False)

        assert exported == {hf_key: tensor}
        assert restored == {model_key: tensor}

    def test_leaves_unrelated_keys_untouched(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))
        state_dict = {"model.layers.0.self_attn.q_proj.weight": torch.ones(1), "lm_head.weight": torch.zeros(1)}

        assert maybe_rename_gemma4_unified_keys(model, state_dict, to_hf=True) == state_dict
        assert maybe_rename_gemma4_unified_keys(model, state_dict, to_hf=False) == state_dict

    @pytest.mark.parametrize("to_hf", [True, False])
    def test_preserves_internal_keys_for_other_model_types(self, to_hf):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4"))
        state_dict = {"embed_vision.multimodal_embedder.embedding_projection.weight": torch.ones(1)}

        assert maybe_rename_gemma4_unified_keys(model, state_dict, to_hf=to_hf) is state_dict

    def test_rejects_export_key_collision(self):
        model = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))
        state_dict = {
            "embed_vision.embedding_projection.weight": torch.ones(1),
            "embed_vision.multimodal_embedder.embedding_projection.weight": torch.zeros(1),
        }

        with pytest.raises(ValueError, match="HF export key collision"):
            maybe_rename_gemma4_unified_keys(model, state_dict, to_hf=True)

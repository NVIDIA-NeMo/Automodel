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

"""Tests for ``nemo_automodel.components.checkpoint.conversion_mapping``."""

from nemo_automodel.components.checkpoint._backports.hf_storage import _get_key_renaming_mapping
from nemo_automodel.components.checkpoint.conversion_mapping import get_combined_key_mapping


def test_gemma3_strips_legacy_vision_model_prefix():
    """Covers the v5.8 gemma3 vision_tower flattening rule.

    transformers 5.8 dropped the ``vision_model.`` wrapper inside Gemma3's
    vision_tower, so HF gemma3 checkpoints saved before this flip (keys like
    ``vision_tower.vision_model.X``) must be renamed to the new flat in-memory
    FQNs (``model.vision_tower.X``). The new rule must win over the generic
    ``vision_tower.`` rule under ``_get_key_renaming_mapping``'s first-match
    semantics.
    """
    mapping = get_combined_key_mapping("gemma3")
    assert mapping is not None

    # v4-format key with legacy vision_model. wrapper -> flat v5 key.
    legacy = "vision_tower.vision_model.embeddings.patch_embedding.weight"
    assert _get_key_renaming_mapping(legacy, mapping) == "model.vision_tower.embeddings.patch_embedding.weight"

    # A v5-format key without the wrapper still gets the outer model. prefix.
    flat = "vision_tower.embeddings.patch_embedding.weight"
    assert _get_key_renaming_mapping(flat, mapping) == "model.vision_tower.embeddings.patch_embedding.weight"

    # Sibling rules still work (regression guard against ordering mistakes).
    assert (
        _get_key_renaming_mapping("language_model.model.embed_tokens.weight", mapping)
        == "model.language_model.embed_tokens.weight"
    )
    assert _get_key_renaming_mapping("language_model.lm_head.weight", mapping) == "lm_head.weight"
    assert (
        _get_key_renaming_mapping("multi_modal_projector.mm_input_projection_weight", mapping)
        == "model.multi_modal_projector.mm_input_projection_weight"
    )

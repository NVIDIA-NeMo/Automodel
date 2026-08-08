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

import pytest

from nemo_automodel.components.checkpoint._backports.hf_storage import _get_key_renaming_mapping
from nemo_automodel.components.checkpoint.conversion_mapping import (
    build_hf_export_key_renames,
    get_combined_key_mapping,
    get_hf_load_key_mapping,
)


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


class _StubConfig:
    def __init__(self, model_type: str):
        self.model_type = model_type


class _StubModel:
    """Stands in for a `PreTrainedModel`; only the attributes the lookup reads are needed."""

    def __init__(self, model_type: str, checkpoint_conversion_mapping: dict[str, str] | None = None):
        self.config = _StubConfig(model_type)
        if checkpoint_conversion_mapping is not None:
            self._checkpoint_conversion_mapping = checkpoint_conversion_mapping


def test_load_key_mapping_prefers_the_model_attribute():
    """A model that declares its own mapping must not be overridden by the model-type tables."""
    own = {r"^foo\.": "bar."}
    assert get_hf_load_key_mapping(_StubModel("gemma4_unified", own)) is own


def test_load_key_mapping_falls_back_to_the_model_type_tables():
    """gemma4_unified's renames live only in the Transformers tables, not on the model class."""
    mapping = get_hf_load_key_mapping(_StubModel("gemma4_unified"))
    assert mapping is not None
    assert _get_key_renaming_mapping("vision_embedder.patch_ln1.weight", mapping) == "embed_vision.patch_ln1.weight"


def test_load_key_mapping_is_none_without_a_model_type():
    assert get_hf_load_key_mapping(object()) is None


def test_export_renames_invert_a_load_mapping():
    """The published names must be recoverable from the FQNs the model is actually held under."""
    mapping = {r"^outer\.inner\.": "flat.", r"^kept\.": "renamed."}
    published = ["outer.inner.weight", "kept.weight", "model.layers.0.self_attn.q_proj.weight"]

    renames = build_hf_export_key_renames(published, mapping)

    assert renames == {
        "flat.weight": "outer.inner.weight",
        "renamed.weight": "kept.weight",
    }
    # Keys the load mapping leaves alone are exported under their own name.
    assert "model.layers.0.self_attn.q_proj.weight" not in renames


def test_export_renames_round_trip_the_real_gemma4_unified_tables():
    """Canary over the installed Transformers tables, the case that motivated this path."""
    mapping = get_hf_load_key_mapping(_StubModel("gemma4_unified"))
    published = ["vision_embedder.patch_ln1.weight", "embed_vision.embedding_projection.weight"]

    renames = build_hf_export_key_renames(published, mapping)

    assert renames == {
        "embed_vision.patch_ln1.weight": "vision_embedder.patch_ln1.weight",
        "embed_vision.multimodal_embedder.embedding_projection.weight": "embed_vision.embedding_projection.weight",
    }


def test_export_renames_are_empty_without_a_load_mapping():
    assert build_hf_export_key_renames(["lm_head.weight"], None) == {}
    assert build_hf_export_key_renames(["lm_head.weight"], {}) == {}


def test_export_renames_reject_an_ambiguous_collapse():
    """Two published keys loading into one model key would make the export name arbitrary."""
    mapping = {r"^a\.": "shared.", r"^b\.": "shared."}
    with pytest.raises(ValueError, match="ambiguous"):
        build_hf_export_key_renames(["a.weight", "b.weight"], mapping)


def test_export_renames_tolerate_a_repeated_published_key():
    mapping = {r"^a\.": "shared."}
    assert build_hf_export_key_renames(["a.weight", "a.weight"], mapping) == {"shared.weight": "a.weight"}

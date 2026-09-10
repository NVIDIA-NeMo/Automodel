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

"""Task resolution against SUPPORTED_BACKBONES.

Kept separate from ``test_retrieval.py`` so these stay importable without the vision
extras that the VLM processors in that module pull in.
"""

import pytest

from nemo_automodel._transformers.retrieval import (
    SUPPORTED_BACKBONES,
    _get_supported_backbone_class,
)


def test_qwen3_registers_score_only():
    """qwen3 is registered for the causal reranker task only."""
    assert SUPPORTED_BACKBONES["qwen3"] == {"score": "Qwen3RerankerForCausalReranking"}


def test_qwen3_score_resolves_to_causal_reranker():
    from nemo_automodel.components.models.qwen3_reranker.model import Qwen3RerankerForCausalReranking

    assert _get_supported_backbone_class("qwen3", "score") is Qwen3RerankerForCausalReranking


def test_qwen3_embedding_falls_back_instead_of_raising():
    """Registering qwen3 for "score" must not break qwen3 embedding backbones.

    ``None`` means "no custom class registered", which routes the caller to the generic
    ``AutoModel`` + ``is_causal=False`` path that qwen3 embedding used before qwen3 was
    added to SUPPORTED_BACKBONES.
    """
    assert _get_supported_backbone_class("qwen3", "embedding") is None


def test_registered_embedding_backbone_still_wins():
    """The fallback must not shadow a model type that does register an embedding class."""
    assert SUPPORTED_BACKBONES["llama_nemotron_vl"]["embedding"] == "LlamaNemotronVLModel"
    resolved = _get_supported_backbone_class("llama_nemotron_vl", "embedding")
    assert resolved is not None
    assert resolved.__name__ == "LlamaNemotronVLModel"


def test_unknown_task_still_raises_for_known_model_type():
    """Only "embedding" falls through; a genuinely unknown task must still fail loudly."""
    with pytest.raises(ValueError, match="Unsupported task 'captioning'"):
        _get_supported_backbone_class("qwen3", "captioning")


def test_unknown_model_type_returns_none():
    assert _get_supported_backbone_class("not_a_real_backbone", "embedding") is None

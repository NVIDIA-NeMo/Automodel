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

"""Unit tests for MineHardNegativesRecipe — attn_implementation support."""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.retrieval.mine_hard_negatives import MINING_DEFAULTS, MineHardNegativesRecipe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal required mining params that pass _validate_mining_params.
_BASE_MINING = {
    "model_name_or_path": "/fake/model",
    "train_qa_file_path": "/fake/input.json",
    "train_file_output_path": "/fake/output.json",
}


def _make_recipe(mining_overrides=None):
    """Create a MineHardNegativesRecipe with a real ConfigNode config.

    The recipe's mining_cfg is set directly (bypassing setup()) so that
    _extract_mining_params can be tested in isolation.
    """
    mining_dict = dict(_BASE_MINING, **(mining_overrides or {}))
    cfg = ConfigNode({"mining": mining_dict})
    recipe = MineHardNegativesRecipe(cfg)
    # Simulate what setup() does before calling _extract_mining_params:
    recipe.mining_cfg = cfg.get("mining")
    return recipe


def _run_setup_and_capture_from_pretrained(mining_overrides=None):
    """Run recipe.setup() with only the truly heavy pieces stubbed out.

    build_distributed, NeMoAutoModelBiEncoder, _configure_tokenizer,
    _load_data, _build_document_mappings, and _prepare_data are mocked
    because they require GPU / filesystem / model weights.

    _extract_mining_params and _validate_mining_params run for real so
    we test the actual wiring end-to-end.

    Returns the mock for NeMoAutoModelBiEncoder so callers can inspect
    from_pretrained call args.
    """
    mining_dict = dict(_BASE_MINING, **(mining_overrides or {}))
    cfg = ConfigNode({"mining": mining_dict})
    recipe = MineHardNegativesRecipe(cfg)

    mock_model = MagicMock()
    mock_model.to.return_value = mock_model

    with (
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.build_distributed") as mock_dist,
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.NeMoAutoModelBiEncoder") as mock_auto,
        patch.object(recipe, "_configure_tokenizer"),
        patch.object(recipe, "_load_data"),
        patch.object(recipe, "_build_document_mappings"),
        patch.object(recipe, "_prepare_data"),
    ):
        mock_dist.return_value = MagicMock(device="cpu")
        mock_auto.from_pretrained.return_value = mock_model

        recipe.setup()

        return mock_auto


# ---------------------------------------------------------------------------
# MINING_DEFAULTS
# ---------------------------------------------------------------------------


def test_mining_defaults_contains_attn_implementation():
    """attn_implementation should be present in MINING_DEFAULTS and default to None."""
    assert "attn_implementation" in MINING_DEFAULTS
    assert MINING_DEFAULTS["attn_implementation"] is None


# ---------------------------------------------------------------------------
# _extract_mining_params — attn_implementation plumbing
# ---------------------------------------------------------------------------


def test_extract_mining_params_attn_implementation_default():
    """When attn_implementation is absent from config, it should default to None."""
    recipe = _make_recipe()
    recipe._extract_mining_params()
    assert recipe.attn_implementation is None


def test_extract_mining_params_attn_implementation_explicit_none():
    """When attn_implementation is explicitly set to None, attribute should be None."""
    recipe = _make_recipe({"attn_implementation": None})
    recipe._extract_mining_params()
    assert recipe.attn_implementation is None


@pytest.mark.parametrize("value", ["sdpa", "flash_attention_2", "eager"])
def test_extract_mining_params_attn_implementation_explicit(value):
    """When attn_implementation is set in config, it should be extracted."""
    recipe = _make_recipe({"attn_implementation": value})
    recipe._extract_mining_params()
    assert recipe.attn_implementation == value


@pytest.mark.parametrize("unknown_field", ["pooling", "l2_normalize", "query_prefx", "hard_negative_to_mine"])
def test_extract_mining_params_rejects_unknown_fields(unknown_field):
    """Unsupported mining keys should fail loudly instead of being ignored."""
    recipe = _make_recipe({unknown_field: "unused"})

    with pytest.raises(ValueError, match=f"Unknown mining config field\\(s\\): {unknown_field}"):
        recipe._extract_mining_params()


# ---------------------------------------------------------------------------
# setup() — model loading with/without attn_implementation
# ---------------------------------------------------------------------------


def test_setup_without_attn_implementation():
    """When attn_implementation is absent, from_pretrained should NOT receive it."""
    mock_auto = _run_setup_and_capture_from_pretrained()
    mock_auto.from_pretrained.assert_called_once()
    args, kwargs = mock_auto.from_pretrained.call_args
    assert args == ("/fake/model",)
    assert "attn_implementation" not in kwargs
    assert kwargs["use_liger_kernel"] is False
    assert kwargs["use_sdpa_patching"] is True


def test_setup_with_attn_implementation_explicit_none():
    """When attn_implementation is explicitly None, from_pretrained should NOT receive it."""
    mock_auto = _run_setup_and_capture_from_pretrained({"attn_implementation": None})
    mock_auto.from_pretrained.assert_called_once()
    args, kwargs = mock_auto.from_pretrained.call_args
    assert args == ("/fake/model",)
    assert "attn_implementation" not in kwargs
    assert kwargs["use_liger_kernel"] is False
    assert kwargs["use_sdpa_patching"] is True


@pytest.mark.parametrize("attn_impl", ["sdpa", "flash_attention_2", "eager"])
def test_setup_with_attn_implementation(attn_impl):
    """When attn_implementation is set, from_pretrained should receive it."""
    mock_auto = _run_setup_and_capture_from_pretrained({"attn_implementation": attn_impl})
    mock_auto.from_pretrained.assert_called_once()
    args, kwargs = mock_auto.from_pretrained.call_args
    assert args == ("/fake/model",)
    assert kwargs["attn_implementation"] == attn_impl
    assert kwargs["use_liger_kernel"] is False
    assert kwargs["use_sdpa_patching"] is True


def test_write_output_preserves_original_question_id(tmp_path):
    """Mined outputs should keep query lineage added by unroll_pos_docs.py."""
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text(json.dumps({"corpus": {"path": "/fake/corpus"}, "data": []}))

    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(output_file),
        }
    )
    recipe.train_qa_file_path = str(input_file)
    recipe.train_file_output_path = str(output_file)
    recipe.questions_dataset = [
        {
            "question_id": "q0_0",
            "original_question_id": "q0",
            "question": "Which doc is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": "p1"}],
            "neg_doc": [{"id": "old"}],
            "pos_score": 0.7,
            "neg_scores": [0.1],
        }
    ]
    recipe._build_negative_docs_by_question_id = lambda: {"q0_0": [{"id": "n1", "score": 0.2}]}
    recipe._build_positive_scores_by_question_id = lambda: {"q0_0": [0.9]}
    recipe._get_mining_args_dict = lambda: {}

    recipe._write_output()

    output = json.loads(output_file.read_text())
    assert output["corpus"] == {"path": "/fake/corpus"}
    assert output["mining"] == {"args": {}}
    assert output["data"] == [
        {
            "question_id": "q0_0",
            "original_question_id": "q0",
            "question": "Which doc is positive?",
            "corpus_id": "demo",
            "pos_doc": [{"id": "p1", "score": 0.9}],
            "neg_doc": [{"id": "n1", "score": 0.2}],
        }
    ]


def test_encode_texts_empty_input_returns_empty_array():
    recipe = _make_recipe()

    embeddings = recipe._encode_texts(texts=[], batch_size=2, max_length=16)

    assert embeddings.shape == (0, 0)
    assert embeddings.dtype == np.float32


def test_load_cached_chunk_ignored_when_cache_loading_disabled(tmp_path):
    recipe = _make_recipe({"load_embeddings_from_cache": False})
    recipe.load_embeddings_from_cache = False
    cache_path = tmp_path / "chunk_0000.npz"
    np.savez(cache_path, np.ones((1, 2), dtype=np.float32))

    assert recipe._load_cached_chunk(cache_path) is None


def test_mine_hard_negatives_drops_margin_filtered_candidates():
    recipe = _make_recipe()
    recipe.dist_env = MagicMock(device=torch.device("cpu"))
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    document_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.96, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )

    neg_indices, neg_scores, pos_scores = recipe._mine_hard_negatives(
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        pos_doc_indices=[[0]],
        batch_size=1,
        num_negs=2,
        hard_neg_margin=0.95,
        hard_neg_margin_type="perc",
    )

    assert neg_indices == [[2]]
    assert neg_scores[0][0] == pytest.approx(0.2)
    assert pos_scores[0][0] == pytest.approx(1.0)

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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode, load_yaml_config
from nemo_automodel._transformers.mining import CheckpointMiningEncoderConfig
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


class _RecordingDocuments:
    def __init__(self):
        self.fetched_ids = []

    def get_document_by_id(self, doc_id):
        self.fetched_ids.append(doc_id)
        return {"text": f"text-{doc_id}", "image": f"image-{doc_id}"}


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


def test_setup_forwards_explicit_remote_code_flag():
    mock_auto = _run_setup_and_capture_from_pretrained({"trust_remote_code": True})
    assert mock_auto.from_pretrained.call_args.kwargs["trust_remote_code"] is True


def test_configure_tokenizer_forwards_explicit_force_default_and_remote_code():
    recipe = _make_recipe({"tokenizer_force_default": True, "trust_remote_code": True})
    recipe._extract_mining_params()
    tokenizer = MagicMock(pad_token="<pad>")
    with patch(
        "nemo_automodel.recipes.retrieval.mine_hard_negatives.NeMoAutoTokenizer.from_pretrained",
        return_value=tokenizer,
    ) as from_pretrained:
        recipe._configure_tokenizer()
    assert from_pretrained.call_args.kwargs["force_default"] is True
    assert from_pretrained.call_args.kwargs["trust_remote_code"] is True


def test_multimodal_reusable_cache_is_rejected_before_query_or_corpus_reads(tmp_path):
    np.savez(tmp_path / "query_embeddings.npz", np.ones((1, 2), dtype=np.float32))
    np.savez(tmp_path / "passage_embeddings.npz", np.ones((2, 2), dtype=np.float32))
    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.multimodal_encoder = MagicMock()
    recipe.load_embeddings_from_cache = True
    recipe.cache_embeddings_dir = tmp_path
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True)

    with pytest.raises(ValueError, match="processor, model, query, corpus, and source-image identity"):
        recipe._generate_embeddings()


def test_multimodal_scratch_does_not_overwrite_or_read_text_chunk_cache(tmp_path):
    text_chunk_dir = tmp_path / "corpus_chunks"
    text_chunk_dir.mkdir()
    text_embedding = np.array([[99.0, 99.0]], dtype=np.float32)
    np.savez(text_chunk_dir / "chunk_0000.npz", text_embedding)

    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.cache_embeddings_dir = tmp_path
    recipe.multimodal_encoder = MagicMock()
    recipe.multimodal_encoder.encode_documents.return_value = np.array([[0.8, 0.2]], dtype=np.float32)
    recipe.document_embedding_batch_size = 1
    recipe.corpus_chunk_size = 1
    recipe.idx_to_doc = {0: "doc-0"}
    recipe.documents_dataset = MagicMock()
    recipe.documents_dataset.get_document_by_id.return_value = {"text": "same", "image": "image"}
    recipe.dist_env = SimpleNamespace(world_size=1, is_main=True)

    actual = recipe._encode_all_documents()

    np.testing.assert_array_equal(actual, np.array([[0.8, 0.2]], dtype=np.float32))
    np.testing.assert_array_equal(np.load(text_chunk_dir / "chunk_0000.npz")["arr_0"], text_embedding)
    assert (tmp_path / "multimodal_scratch" / "corpus_chunks" / "chunk_0000.npz").exists()


def test_multimodal_documents_are_fetched_in_bounded_batches_and_stable_order():
    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.document_embedding_batch_size = 2
    recipe.idx_to_doc = {idx: f"doc-{idx}" for idx in range(5)}
    recipe.documents_dataset = _RecordingDocuments()
    recipe.multimodal_encoder = MagicMock()
    fetched_counts_at_encode = []

    def encode_documents(documents, *, batch_size):
        fetched_counts_at_encode.append(len(recipe.documents_dataset.fetched_ids))
        values = [float(document["_mining_document_id"].split("-")[-1]) for document in documents]
        return np.asarray([[value, -value] for value in values], dtype=np.float32)

    recipe.multimodal_encoder.encode_documents.side_effect = encode_documents

    embeddings = recipe._encode_document_indices(list(range(5)))

    assert recipe.documents_dataset.fetched_ids == [f"doc-{idx}" for idx in range(5)]
    assert fetched_counts_at_encode == [2, 4, 5]
    np.testing.assert_array_equal(embeddings[:, 0], np.arange(5, dtype=np.float32))


def test_distributed_multimodal_chunk_fetches_only_rank_owned_documents(tmp_path):
    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.document_embedding_batch_size = 2
    recipe.idx_to_doc = {idx: f"doc-{idx}" for idx in range(5)}
    recipe.documents_dataset = _RecordingDocuments()
    recipe.multimodal_encoder = MagicMock()
    recipe.multimodal_encoder.encode_documents.return_value = np.ones((2, 2), dtype=np.float32)
    recipe.dist_env = SimpleNamespace(world_size=2, rank=1, is_main=False)
    recipe._synchronize_ranks = MagicMock()

    result = recipe._encode_chunk_distributed(list(range(5)), tmp_path / "chunk_0000.npz")

    assert recipe.documents_dataset.fetched_ids == ["doc-3", "doc-4"]
    assert result.shape == (0, 0)


def test_mining_metadata_records_json_serializable_multimodal_config_and_disabled_cache_reuse(monkeypatch):
    monkeypatch.setenv("MINING_TEST_PREFIX", "look up:")
    target = "nemo_automodel._transformers.mining.CheckpointMiningEncoderConfig"
    cfg = ConfigNode(
        {
            "mining": {
                **_BASE_MINING,
                "multimodal_encoder": {
                    "_target_": target,
                    "query_prefix": "${oc.env:MINING_TEST_PREFIX}",
                    "p_max_length": 4096,
                    "image_longest_edge": 1284,
                    "use_text_in_document": True,
                    "use_images": True,
                },
            }
        }
    )
    recipe = MineHardNegativesRecipe(cfg)
    model = MagicMock(pooling="avg", l2_normalize=True)
    model.to.return_value = model
    encoder = MagicMock()

    with (
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.build_distributed") as build_dist,
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.NeMoAutoModelBiEncoder") as auto_model,
        patch(
            "nemo_automodel._transformers.mining.CheckpointMiningEncoderConfig.build",
            return_value=encoder,
        ),
        patch.object(recipe, "_load_data"),
        patch.object(recipe, "_build_document_mappings"),
        patch.object(recipe, "_prepare_data"),
    ):
        build_dist.return_value = SimpleNamespace(device=torch.device("cpu"))
        auto_model.from_pretrained.return_value = model
        recipe.setup()

    metadata = recipe._get_mining_args_dict()

    assert metadata["multimodal_encoder"]["_target_"] == target
    assert metadata["multimodal_encoder"]["query_prefix"] == "look up:"
    assert metadata["cache_reuse"] is False
    json.dumps(metadata)


def test_multimodal_mining_example_resolves_typed_encoder_without_downloads():
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_yaml_config(repo_root / "examples/retrieval/data_utils/mining_multimodal_config.yaml")

    encoder_config = cfg.mining.multimodal_encoder.instantiate()

    assert isinstance(encoder_config, CheckpointMiningEncoderConfig)
    assert encoder_config.p_max_length == 4096
    assert cfg.mining.load_embeddings_from_cache is False


def test_empty_distributed_rank_writes_metadata_shards_without_encoder_calls(tmp_path):
    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.cache_embeddings_dir = tmp_path
    recipe.multimodal_encoder = MagicMock()
    recipe.questions = ["only-query"]
    recipe.query_embedding_batch_size = 1
    recipe.document_embedding_batch_size = 1
    recipe.dist_env = SimpleNamespace(world_size=2, rank=1, is_main=False)
    recipe._synchronize_ranks = MagicMock()

    query_result = recipe._encode_queries_sharded()
    document_result = recipe._encode_chunk_distributed([0], tmp_path / "document_chunk.npz")

    assert query_result.shape == (0, 0)
    assert document_result.shape == (0, 0)
    recipe.multimodal_encoder.encode_queries.assert_not_called()
    recipe.multimodal_encoder.encode_documents.assert_not_called()


def test_main_rank_assembles_nonempty_and_empty_shards_in_exact_order(tmp_path):
    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.cache_embeddings_dir = tmp_path
    recipe.multimodal_encoder = MagicMock()
    recipe.multimodal_encoder.encode_queries.return_value = np.array([[1.0, 2.0]], dtype=np.float32)
    recipe.multimodal_encoder.encode_documents.return_value = np.array([[3.0, 4.0]], dtype=np.float32)
    recipe.questions = ["only-query"]
    recipe.query_embedding_batch_size = 1
    recipe.document_embedding_batch_size = 1
    recipe.idx_to_doc = {0: "doc-0"}
    recipe.documents_dataset = _RecordingDocuments()
    recipe.dist_env = SimpleNamespace(world_size=2, rank=0, is_main=True)
    recipe._synchronize_ranks = MagicMock()

    query_shard_dir = tmp_path / "multimodal_scratch" / "query_shards"
    query_shard_dir.mkdir(parents=True)
    np.savez(query_shard_dir / "queries_rank0001.npz", np.empty((0, 0), dtype=np.float32))
    document_cache_path = tmp_path / "document_chunk.npz"
    np.savez(tmp_path / "document_chunk_rank0001.npz", np.empty((0, 0), dtype=np.float32))

    query_embeddings = recipe._encode_queries_sharded()
    document_embeddings = recipe._encode_chunk_distributed([0], document_cache_path)

    np.testing.assert_array_equal(query_embeddings, [[1.0, 2.0]])
    np.testing.assert_array_equal(document_embeddings, [[3.0, 4.0]])
    assert np.isfinite(query_embeddings).all()
    assert np.isfinite(document_embeddings).all()

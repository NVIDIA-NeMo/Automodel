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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from nemo_automodel.recipes.retrieval.mining_encoder import CheckpointMiningEncoderConfig
from nemo_automodel.components.config.loader import ConfigNode, load_yaml_config

from nemo_automodel.recipes.retrieval.mine_hard_negatives import (
    DOCUMENT_EMBEDDINGS_FNAME,
    MINING_DEFAULTS,
    QUERY_EMBEDDINGS_FNAME,
    MineHardNegativesRecipe,
)

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


class _FakeDocumentsDataset:
    path = "/fake/corpus"

    def __init__(self, text="NVLink is a high-bandwidth GPU interconnect."):
        self.text = text

    def get_document_by_id(self, doc_id):
        return {"text": self.text, "image": "", "nr_ocr": ""}


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


def _make_cache_ready_recipe(tmp_path):
    """Create a recipe with enough state for embedding cache validation."""
    recipe = _make_recipe({"cache_embeddings_dir": str(tmp_path), "load_embeddings_from_cache": True})
    recipe.cache_embeddings_dir = str(tmp_path)
    recipe.load_embeddings_from_cache = True
    recipe.model_name_or_path = "/fake/model"
    recipe.tokenizer_name_or_path = "/fake/tokenizer"
    recipe.train_qa_file_path = "/fake/input.json"
    recipe.corpus_id = "demo"
    recipe.corpus_path = "/fake/corpus"
    recipe.query_prefix = "query: "
    recipe.passage_prefix = "passage: "
    recipe.query_max_length = 16
    recipe.passage_max_length = 32
    recipe.add_bos_token = None
    recipe.add_eos_token = False
    recipe.attn_implementation = None
    recipe.questions = ["what is nvlink?"]
    recipe.question_ids = ["q0"]
    recipe.idx_to_doc = {0: "d0"}
    recipe.documents_dataset = _FakeDocumentsDataset()
    recipe.dist_env = SimpleNamespace(world_size=1, rank=0, is_main=True, device=torch.device("cpu"))
    recipe.model = SimpleNamespace(pooling="avg", l2_normalize=True)
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
    with np.load(text_chunk_dir / "chunk_0000.npz", allow_pickle=False) as cached:
        np.testing.assert_array_equal(cached["arr_0"], text_embedding)
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


def test_mining_metadata_records_json_serializable_multimodal_config_and_disabled_cache_reuse():
    target = "nemo_automodel.recipes.retrieval.mining_encoder.CheckpointMiningEncoderConfig"
    cfg = ConfigNode(
        {
            "mining": {
                **_BASE_MINING,
                "multimodal_encoder": {
                    "_target_": target,
                },
            }
        }
    )
    recipe = MineHardNegativesRecipe(cfg)
    model = MagicMock(pooling="avg", l2_normalize=True)
    model.to.return_value = model
    encoder = MagicMock()
    encoder.model = model
    encoder.pooling = "avg"
    encoder.l2_normalize = True

    with (
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.build_distributed") as build_dist,
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.NeMoAutoModelBiEncoder") as auto_model,
        patch(
            "nemo_automodel.recipes.retrieval.mining_encoder.CheckpointMiningEncoderConfig.build",
            return_value=encoder,
        ),
        patch.object(recipe, "_load_data"),
        patch.object(recipe, "_build_document_mappings"),
        patch.object(recipe, "_prepare_data"),
    ):
        build_dist.return_value = SimpleNamespace(device=torch.device("cpu"))
        auto_model.from_pretrained.return_value = model
        recipe.setup()

    assert recipe.model is None
    metadata = recipe._get_mining_args_dict()

    assert metadata["multimodal_encoder"] == {"_target_": target}
    for legacy_setting in ("query_prefix", "passage_prefix", "query_max_length", "passage_max_length"):
        assert legacy_setting not in metadata
    assert metadata["cache_reuse"] is False
    auto_model.from_pretrained.assert_not_called()
    json.dumps(metadata)


def test_multimodal_mining_example_resolves_typed_encoder_without_downloads():
    repo_root = Path(__file__).resolve().parents[3]
    cfg = load_yaml_config(repo_root / "examples/retrieval/data_utils/mining_multimodal_config.yaml")

    encoder_config = cfg.mining.multimodal_encoder.instantiate()

    assert isinstance(encoder_config, CheckpointMiningEncoderConfig)
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


def test_setup_rejects_colbert_pooling():
    """The public setup path should reject token-level ColBERT embeddings before mining."""
    cfg = ConfigNode({"mining": dict(_BASE_MINING)})
    recipe = MineHardNegativesRecipe(cfg)
    mock_model = MagicMock(pooling="colbert")
    mock_model.to.return_value = mock_model

    with (
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.build_distributed") as mock_dist,
        patch("nemo_automodel.recipes.retrieval.mine_hard_negatives.NeMoAutoModelBiEncoder") as mock_auto,
        pytest.raises(ValueError, match="ColBERT pooling"),
    ):
        mock_dist.return_value = MagicMock(device="cpu")
        mock_auto.from_pretrained.return_value = mock_model

        recipe.setup()


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


@pytest.mark.parametrize("positive", [{"id": "p1"}, "p1"])
def test_write_output_preserves_custom_row_metadata(tmp_path, positive):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "question_id": "q0",
                        "question": "Which doc is positive?",
                        "corpus_id": "demo",
                        "pos_doc": [positive],
                        "neg_doc": [{"id": "old"}],
                        "source_split": "curated-dev",
                    }
                ]
            }
        )
    )
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
            "question_id": "q0",
            "question": "Which doc is positive?",
            "corpus_id": "demo",
            "pos_doc": [positive],
            "neg_doc": [],
        }
    ]
    recipe._build_negative_docs_by_question_id = lambda: {"q0": [{"id": "n1", "score": 0.2}]}
    recipe._build_positive_scores_by_question_id = lambda: {"q0": [0.9]}
    recipe._get_mining_args_dict = lambda: {}

    recipe._write_output()

    output = json.loads(output_file.read_text())
    assert output["data"][0]["source_split"] == "curated-dev"
    assert output["data"][0]["neg_doc"] == [{"id": "n1", "score": 0.2}]
    assert output["data"][0]["pos_doc"] == [{"id": "p1", "score": 0.9}]


def test_write_output_rewrites_relative_corpus_path_for_new_output_dir(tmp_path):
    input_dir = tmp_path / "source"
    output_dir = tmp_path / "mined"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "FEVER_corpus").mkdir()
    input_file = input_dir / "train.json"
    output_file = output_dir / "train.json"
    input_file.write_text(
        json.dumps(
            {
                "corpus": [{"path": "./FEVER_corpus"}],
                "data": [
                    {
                        "question_id": "q0",
                        "question": "Which doc is positive?",
                        "corpus_id": "demo",
                        "pos_doc": [{"id": "p1"}],
                        "neg_doc": [],
                    }
                ],
            }
        )
    )
    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(output_file),
        }
    )
    recipe.train_qa_file_path = str(input_file)
    recipe.train_file_output_path = str(output_file)
    recipe.questions_dataset = json.loads(input_file.read_text())["data"]
    recipe._build_negative_docs_by_question_id = lambda: {"q0": []}
    recipe._build_positive_scores_by_question_id = lambda: {"q0": [0.9]}
    recipe._get_mining_args_dict = lambda: {}

    recipe._write_output()

    output = json.loads(output_file.read_text())
    corpus_path = output["corpus"][0]["path"]
    assert os.path.normpath(corpus_path) == os.path.normpath("../source/FEVER_corpus")
    assert (output_dir / corpus_path).resolve() == (input_dir / "FEVER_corpus").resolve()


def test_write_output_uses_atomic_replace_on_dump_failure(tmp_path, monkeypatch):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "question_id": "q0",
                        "question": "Which doc is positive?",
                        "corpus_id": "demo",
                        "pos_doc": [{"id": "p1"}],
                        "neg_doc": [],
                    }
                ]
            }
        )
    )
    output_file.write_text("existing mined data")
    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(output_file),
            "overwrite_output": True,
        }
    )
    recipe.train_qa_file_path = str(input_file)
    recipe.train_file_output_path = str(output_file)
    recipe.overwrite_output = True
    recipe.questions_dataset = json.loads(input_file.read_text())["data"]
    recipe._build_negative_docs_by_question_id = lambda: {"q0": []}
    recipe._build_positive_scores_by_question_id = lambda: {"q0": [0.9]}
    recipe._get_mining_args_dict = lambda: {}

    original_dump = json.dump

    def failing_dump(obj, fp, *args, **kwargs):
        if isinstance(obj, dict) and "mining" in obj:
            fp.write("partial")
            raise OSError("disk full")
        return original_dump(obj, fp, *args, **kwargs)

    monkeypatch.setattr(json, "dump", failing_dump)

    with pytest.raises(OSError, match="disk full"):
        recipe._write_output()

    assert output_file.read_text() == "existing mined data"
    assert not list(tmp_path.glob(".output.json.*.tmp"))


def test_write_output_removes_stale_positive_score_for_non_finite_current_score(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "question_id": "q0",
                        "question": "Which doc is positive?",
                        "corpus_id": "demo",
                        "pos_doc": [{"id": "p1", "score": 0.9}],
                        "neg_doc": [],
                    }
                ]
            }
        )
    )
    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(output_file),
        }
    )
    recipe.train_qa_file_path = str(input_file)
    recipe.train_file_output_path = str(output_file)
    recipe.questions_dataset = json.loads(input_file.read_text())["data"]
    recipe._build_negative_docs_by_question_id = lambda: {"q0": []}
    recipe._build_positive_scores_by_question_id = lambda: {"q0": [None]}
    recipe._get_mining_args_dict = lambda: {}

    recipe._write_output()

    output = json.loads(output_file.read_text())
    assert output["data"][0]["pos_doc"] == [{"id": "p1"}]


def test_validate_mining_params_rejects_existing_output_without_overwrite(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text("{}")
    output_file.write_text("existing")
    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(output_file),
        }
    )
    recipe._extract_mining_params()

    with pytest.raises(ValueError, match="Output file already exists"):
        recipe._validate_mining_params()


def test_validate_mining_params_allows_existing_output_with_overwrite(tmp_path):
    input_file = tmp_path / "input.json"
    output_file = tmp_path / "output.json"
    input_file.write_text("{}")
    output_file.write_text("existing")
    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(output_file),
            "overwrite_output": True,
        }
    )
    recipe._extract_mining_params()

    recipe._validate_mining_params()


def test_validate_mining_params_rejects_input_output_same_path(tmp_path):
    input_file = tmp_path / "input.json"
    input_file.write_text("{}")
    recipe = _make_recipe(
        {
            "train_qa_file_path": str(input_file),
            "train_file_output_path": str(input_file),
            "overwrite_output": True,
        }
    )
    recipe._extract_mining_params()

    with pytest.raises(ValueError, match="must be different"):
        recipe._validate_mining_params()


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


def test_load_cached_chunk_rejects_shape_mismatch(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._reuse_partial_embedding_cache = True
    cache_path = tmp_path / "chunk_0000.npz"
    np.savez(cache_path, np.ones((2, 2), dtype=np.float32))

    assert recipe._load_cached_chunk(cache_path, expected_size=1) is None


def test_cache_fingerprint_includes_corpus_chunk_size(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe.corpus_chunk_size = 2
    first_fingerprint = recipe._build_cache_fingerprint()

    recipe.corpus_chunk_size = 3

    assert first_fingerprint["corpus_chunk_size"] == 2
    assert recipe._build_cache_fingerprint()["corpus_chunk_size"] == 3


def test_load_partial_cache_shard_rejects_shape_mismatch(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._reuse_partial_embedding_cache = True
    cache_path = tmp_path / "query_shards" / "queries_rank0000.npz"
    cache_path.parent.mkdir()
    np.savez(cache_path, np.ones((2, 2), dtype=np.float32))

    assert recipe._load_partial_cache_shard(cache_path, expected_size=1, label="query shard") is None


def test_load_partial_cache_shard_rejects_unreadable_npz(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._reuse_partial_embedding_cache = True
    cache_path = tmp_path / "chunk_0000_rank0000.npz"
    cache_path.write_text("not an npz")

    assert recipe._load_partial_cache_shard(cache_path, expected_size=1, label="document rank shard") is None


def test_load_partial_cache_shard_rejects_wrong_ndim(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._reuse_partial_embedding_cache = True
    cache_path = tmp_path / "queries_rank0000.npz"
    np.savez(cache_path, np.ones((1, 2, 3), dtype=np.float32))

    assert recipe._load_partial_cache_shard(cache_path, expected_size=1, label="query shard") is None


def test_load_partial_cache_shard_rejects_width_mismatch_from_metadata(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._reuse_partial_embedding_cache = True
    cache_path = tmp_path / "queries_rank0000.npz"
    np.savez(cache_path, np.ones((1, 2), dtype=np.float32))
    recipe._write_cache_metadata(
        query_embeddings=np.ones((1, 3), dtype=np.float32),
        document_embeddings=np.ones((1, 3), dtype=np.float32),
    )

    assert (
        recipe._load_partial_cache_shard(
            cache_path,
            expected_size=1,
            label="query shard",
            expected_width=recipe._get_cached_embedding_width("query_shape"),
        )
        is None
    )


def test_full_embeddings_cache_requires_matching_metadata(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    query_embeddings = np.ones((1, 2), dtype=np.float32)
    document_embeddings = np.ones((1, 2), dtype=np.float32)

    recipe._save_embeddings_to_cache(query_embeddings, document_embeddings)

    assert recipe._has_full_embeddings_cache()
    cached_query_embeddings, cached_document_embeddings = recipe._load_embeddings_from_cache()
    np.testing.assert_array_equal(cached_query_embeddings, query_embeddings)
    np.testing.assert_array_equal(cached_document_embeddings, document_embeddings)

    recipe.query_prefix = "different query: "
    assert not recipe._has_full_embeddings_cache()
    assert recipe._load_embeddings_from_cache() == (None, None)


def test_full_embeddings_cache_rejects_changed_document_content(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._save_embeddings_to_cache(np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32))

    recipe.documents_dataset.text = "The corpus text changed under the same document ID."

    assert not recipe._has_full_embeddings_cache()
    assert recipe._load_embeddings_from_cache() == (None, None)


def test_full_embeddings_cache_rejects_shape_mismatch(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    query_embeddings = np.ones((1, 2), dtype=np.float32)
    document_embeddings = np.ones((1, 2), dtype=np.float32)
    recipe._save_embeddings_to_cache(query_embeddings, document_embeddings)
    np.savez(tmp_path / QUERY_EMBEDDINGS_FNAME, np.ones((2, 2), dtype=np.float32))

    assert not recipe._has_full_embeddings_cache()
    assert recipe._load_embeddings_from_cache() == (None, None)


def test_generate_embeddings_recomputes_when_full_cache_payload_is_invalid(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe._save_embeddings_to_cache(np.ones((1, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32))
    np.savez(tmp_path / QUERY_EMBEDDINGS_FNAME, np.ones((2, 2), dtype=np.float32))
    recomputed_query = np.full((1, 2), 2.0, dtype=np.float32)
    recomputed_documents = np.full((1, 2), 3.0, dtype=np.float32)
    recipe._encode_queries = lambda: recomputed_query
    recipe._encode_all_documents = lambda: recomputed_documents

    query_embeddings, document_embeddings = recipe._generate_embeddings()

    np.testing.assert_array_equal(query_embeddings, recomputed_query)
    np.testing.assert_array_equal(document_embeddings, recomputed_documents)


def test_generate_embeddings_reuses_partial_cache_when_metadata_matches(tmp_path):
    recipe = _make_cache_ready_recipe(tmp_path)
    expected_query = np.ones((1, 2), dtype=np.float32)
    expected_documents = np.ones((1, 2), dtype=np.float32)
    recipe._write_cache_metadata(expected_query, expected_documents)

    def encode_queries():
        assert recipe._reuse_partial_embedding_cache
        return expected_query

    def encode_documents():
        assert recipe._reuse_partial_embedding_cache
        return expected_documents

    recipe._encode_queries = encode_queries
    recipe._encode_all_documents = encode_documents

    query_embeddings, document_embeddings = recipe._generate_embeddings()

    np.testing.assert_array_equal(query_embeddings, expected_query)
    np.testing.assert_array_equal(document_embeddings, expected_documents)


def test_prepare_data_rejects_duplicate_question_ids():
    recipe = _make_recipe()
    recipe.questions_dataset = [
        {"question_id": "q0", "question": "first", "corpus_id": "demo", "pos_doc": [{"id": "d0"}]},
        {"question_id": "q0", "question": "second", "corpus_id": "demo", "pos_doc": [{"id": "d0"}]},
    ]
    recipe.doc_to_idx = {"d0": 0}
    recipe.corpus_id = "demo"
    recipe.use_negatives_from_file = False

    with pytest.raises(ValueError, match="unique question_id"):
        recipe._prepare_data()


def test_prepare_data_rejects_mismatched_corpus_id():
    recipe = _make_recipe()
    recipe.corpus_id = "expected"
    recipe.questions_dataset = [
        {"question_id": "q0", "question": "first", "corpus_id": "other", "pos_doc": [{"id": "d0"}]},
    ]
    recipe.doc_to_idx = {"d0": 0}
    recipe.use_negatives_from_file = False

    with pytest.raises(ValueError, match="Run one corpus per mining job"):
        recipe._prepare_data()


def test_encode_queries_sharded_handles_empty_rank_shard(tmp_path):
    recipe = _make_recipe({"cache_embeddings_dir": str(tmp_path), "load_embeddings_from_cache": False})
    recipe.cache_embeddings_dir = str(tmp_path)
    recipe.load_embeddings_from_cache = False
    recipe.query_embedding_batch_size = 2
    recipe.query_max_length = 16
    recipe.query_prefix = "query: "
    recipe.questions = ["what is nvlink?"]
    recipe.dist_env = SimpleNamespace(world_size=2, rank=0, is_main=True, device=torch.device("cpu"))
    recipe.idx_to_doc = {0: "d0"}
    recipe.documents_dataset = _FakeDocumentsDataset()
    recipe._encode_texts = lambda **_: np.ones((1, 2), dtype=np.float32)

    def write_empty_peer_shard():
        shard_dir = tmp_path / "query_shards"
        np.savez(shard_dir / "queries_rank0001.npz", np.empty((0, 0), dtype=np.float32))

    recipe._synchronize_ranks = write_empty_peer_shard

    query_embeddings = recipe._encode_queries_sharded()

    assert query_embeddings.shape == (1, 2)


def test_encode_queries_sharded_recomputes_stale_local_query_shard(tmp_path):
    recipe = _make_recipe({"cache_embeddings_dir": str(tmp_path), "load_embeddings_from_cache": True})
    recipe.cache_embeddings_dir = str(tmp_path)
    recipe.load_embeddings_from_cache = True
    recipe._reuse_partial_embedding_cache = True
    recipe.query_embedding_batch_size = 2
    recipe.query_max_length = 16
    recipe.query_prefix = "query: "
    recipe.questions = ["what is nvlink?"]
    recipe.dist_env = SimpleNamespace(world_size=2, rank=0, is_main=True, device=torch.device("cpu"))
    shard_dir = tmp_path / "query_shards"
    shard_dir.mkdir()
    np.savez(shard_dir / "queries_rank0000.npz", np.ones((2, 2), dtype=np.float32))
    recomputed = np.full((1, 2), 4.0, dtype=np.float32)
    recipe.idx_to_doc = {0: "d0"}
    recipe.documents_dataset = _FakeDocumentsDataset()
    recipe._encode_texts = lambda **_: recomputed

    def write_empty_peer_shard():
        np.savez(shard_dir / "queries_rank0001.npz", np.empty((0, 0), dtype=np.float32))

    recipe._synchronize_ranks = write_empty_peer_shard

    query_embeddings = recipe._encode_queries_sharded()

    np.testing.assert_array_equal(query_embeddings, recomputed)


def test_encode_queries_sharded_recomputes_corrupt_local_query_shard(tmp_path):
    recipe = _make_recipe({"cache_embeddings_dir": str(tmp_path), "load_embeddings_from_cache": True})
    recipe.cache_embeddings_dir = str(tmp_path)
    recipe.load_embeddings_from_cache = True
    recipe._reuse_partial_embedding_cache = True
    recipe.query_embedding_batch_size = 2
    recipe.query_max_length = 16
    recipe.query_prefix = "query: "
    recipe.questions = ["what is nvlink?"]
    recipe.dist_env = SimpleNamespace(world_size=2, rank=0, is_main=True, device=torch.device("cpu"))
    shard_dir = tmp_path / "query_shards"
    shard_dir.mkdir()
    (shard_dir / "queries_rank0000.npz").write_text("corrupt")
    recomputed = np.full((1, 2), 5.0, dtype=np.float32)
    recipe.idx_to_doc = {0: "d0"}
    recipe.documents_dataset = _FakeDocumentsDataset()
    recipe._encode_texts = lambda **_: recomputed

    def write_empty_peer_shard():
        np.savez(shard_dir / "queries_rank0001.npz", np.empty((0, 0), dtype=np.float32))

    recipe._synchronize_ranks = write_empty_peer_shard

    query_embeddings = recipe._encode_queries_sharded()

    np.testing.assert_array_equal(query_embeddings, recomputed)


def test_encode_chunk_distributed_handles_empty_rank_shard(tmp_path):
    recipe = _make_recipe({"cache_embeddings_dir": str(tmp_path), "load_embeddings_from_cache": False})
    recipe.cache_embeddings_dir = str(tmp_path)
    recipe.load_embeddings_from_cache = False
    recipe.document_embedding_batch_size = 2
    recipe.passage_max_length = 16
    recipe.passage_prefix = "passage: "
    recipe.dist_env = SimpleNamespace(world_size=2, rank=0, is_main=True, device=torch.device("cpu"))
    recipe.idx_to_doc = {0: "d0"}
    recipe.documents_dataset = _FakeDocumentsDataset()
    recipe._encode_texts = lambda **_: np.ones((1, 2), dtype=np.float32)
    cache_path = tmp_path / "chunk_0000.npz"

    def write_empty_peer_shard():
        np.savez(tmp_path / "chunk_0000_rank0001.npz", np.empty((0, 0), dtype=np.float32))

    recipe._synchronize_ranks = write_empty_peer_shard

    document_embeddings = recipe._encode_chunk_distributed([0], cache_path)

    assert document_embeddings.shape == (1, 2)
    assert (tmp_path / DOCUMENT_EMBEDDINGS_FNAME).exists() is False


def test_encode_chunk_distributed_recomputes_corrupt_local_rank_shard(tmp_path):
    recipe = _make_recipe({"cache_embeddings_dir": str(tmp_path), "load_embeddings_from_cache": True})
    recipe.cache_embeddings_dir = str(tmp_path)
    recipe.load_embeddings_from_cache = True
    recipe._reuse_partial_embedding_cache = True
    recipe.document_embedding_batch_size = 2
    recipe.passage_max_length = 16
    recipe.passage_prefix = "passage: "
    recipe.dist_env = SimpleNamespace(world_size=2, rank=0, is_main=True, device=torch.device("cpu"))
    cache_path = tmp_path / "chunk_0000.npz"
    (tmp_path / "chunk_0000_rank0000.npz").write_text("corrupt")
    recomputed = np.full((1, 2), 6.0, dtype=np.float32)
    recipe.idx_to_doc = {0: "d0"}
    recipe.documents_dataset = _FakeDocumentsDataset()
    recipe._encode_texts = lambda **_: recomputed

    def write_empty_peer_shard():
        np.savez(tmp_path / "chunk_0000_rank0001.npz", np.empty((0, 0), dtype=np.float32))

    recipe._synchronize_ranks = write_empty_peer_shard

    document_embeddings = recipe._encode_chunk_distributed([0], cache_path)

    np.testing.assert_array_equal(document_embeddings, recomputed)


def test_encode_documents_chunk_uses_broadcast_cache_hit_on_non_main_rank(tmp_path, monkeypatch):
    recipe = _make_cache_ready_recipe(tmp_path)
    recipe.dist_env = SimpleNamespace(world_size=2, rank=1, is_main=False, device=torch.device("cpu"))
    cache_path = tmp_path / "corpus_chunks" / "chunk_0000.npz"
    cache_path.parent.mkdir()
    np.savez(cache_path, np.ones((1, 2), dtype=np.float32))
    recipe._write_cache_metadata(
        query_embeddings=np.ones((1, 2), dtype=np.float32),
        document_embeddings=np.ones((1, 2), dtype=np.float32),
    )
    recipe._encode_chunk_distributed = MagicMock(side_effect=AssertionError("non-main rank should follow cache hit"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broadcast_cache_hit(flag, src):
        flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast_cache_hit)

    embeddings = recipe._encode_documents_chunk([0], cache_path)

    assert embeddings.shape == (0, 0)
    recipe._encode_chunk_distributed.assert_not_called()


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


def test_mine_hard_negatives_drops_raw_non_finite_scores():
    recipe = _make_recipe()
    recipe.dist_env = MagicMock(device=torch.device("cpu"))
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    document_embeddings = np.array(
        [
            [1.0, 0.0],
            [np.nan, 0.0],
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
    )

    assert neg_indices == [[2]]
    assert neg_scores[0][0] == pytest.approx(0.2)
    assert pos_scores[0][0] == pytest.approx(1.0)


def test_mine_hard_negatives_skips_margin_when_positive_score_is_non_finite():
    recipe = _make_recipe()
    recipe.dist_env = MagicMock(device=torch.device("cpu"))
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    document_embeddings = np.array(
        [
            [np.nan, 0.0],
            [0.2, 0.0],
        ],
        dtype=np.float32,
    )

    neg_indices, neg_scores, pos_scores = recipe._mine_hard_negatives(
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        pos_doc_indices=[[0]],
        batch_size=1,
        num_negs=1,
        hard_neg_margin=0.95,
        hard_neg_margin_type="perc",
    )

    assert neg_indices == [[1]]
    assert neg_scores[0][0] == pytest.approx(0.2)
    assert pos_scores[0][0] is None


def test_mine_hard_negatives_rejects_token_level_embeddings():
    recipe = _make_recipe()
    recipe.dist_env = MagicMock(device=torch.device("cpu"))

    with pytest.raises(ValueError, match="2D single-vector embeddings"):
        recipe._mine_hard_negatives(
            query_embeddings=np.ones((1, 2, 3), dtype=np.float32),
            document_embeddings=np.ones((1, 2, 3), dtype=np.float32),
            pos_doc_indices=[[0]],
            batch_size=1,
            num_negs=1,
        )


def _cache_failure_worker(rank, init_file, result_dir):
    from datetime import timedelta

    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=20)
    )
    try:
        recipe = _make_recipe({"load_embeddings_from_cache": True, "cache_embeddings_dir": result_dir})
        recipe.load_embeddings_from_cache = True
        recipe.cache_embeddings_dir = result_dir
        recipe.dist_env = SimpleNamespace(world_size=2, rank=rank, is_main=rank == 0, device=torch.device("cpu"))
        with patch.object(recipe, "_build_cache_fingerprint", side_effect=OSError("unreadable corpus")):
            with pytest.raises(RuntimeError, match="Rank0 could not validate"):
                recipe._generate_embeddings()
        Path(result_dir, f"rank-{rank}").write_text("validation failure propagated")
    finally:
        torch.distributed.destroy_process_group()


def test_cache_validation_failure_reaches_both_real_gloo_ranks(tmp_path):
    """A rank0 fingerprint failure must reach peers instead of stranding them in a collective."""
    torch.multiprocessing.start_processes(
        _cache_failure_worker,
        args=(str(tmp_path / "gloo-init"), str(tmp_path)),
        nprocs=2,
        join=True,
        start_method="fork",
    )
    assert (tmp_path / "rank-0").read_text() == (tmp_path / "rank-1").read_text()


@pytest.mark.parametrize("reuse", [True, False])
def test_interrupted_refresh_cannot_authorize_mixed_cache_generations(tmp_path, reuse):
    recipe = _make_cache_ready_recipe(tmp_path)
    original_queries = np.array([[1.0, 2.0]], dtype=np.float32)
    original_docs = np.array([[3.0, 4.0]], dtype=np.float32)
    recipe._save_embeddings_to_cache(original_queries, original_docs)
    recipe.questions = ["changed query"]
    recipe.load_embeddings_from_cache = reuse
    recipe._encode_queries = lambda: np.array([[7.0, 8.0]], dtype=np.float32)

    def interrupt_document_encoding():
        # Simulate an interruption after the query cache has been replaced.
        np.savez(tmp_path / QUERY_EMBEDDINGS_FNAME, np.array([[7.0, 8.0]], dtype=np.float32))
        raise OSError("interrupted refresh")

    recipe._encode_all_documents = interrupt_document_encoding
    with pytest.raises(OSError, match="interrupted refresh"):
        recipe._generate_embeddings()
    recipe.questions = ["what is nvlink?"]
    assert recipe._load_embeddings_from_cache() == (None, None)

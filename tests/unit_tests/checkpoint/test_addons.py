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

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from nemo_automodel.components.checkpoint.addons import (
    ConsolidatedHFAddon,
    _extract_target_modules,
    _group_barrier,
    _is_group_rank_0,
    _maybe_save_custom_model_code,
    _maybe_strip_quantization_config,
    _resolve_sentence_transformer_max_seq_length,
    _save_generated_sentence_transformer_assets,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState


def test_group_barrier_uses_model_process_group():
    group = object()
    with (
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.barrier") as barrier,
    ):
        _group_barrier(group)
    barrier.assert_called_once_with(group=group)


def test_group_rank_zero_is_relative_to_model_process_group():
    group = object()
    with (
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.get_rank", return_value=0) as get_rank,
    ):
        assert _is_group_rank_0(group)
    get_rank.assert_called_once_with(group=group)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_maybe_save_custom_model_code_copies_py_files_and_structure(tmp_path):
    # Arrange: create a nested source tree with .py and non-.py files
    src_root = tmp_path / "src_model_code"
    dst_root = tmp_path / "hf_meta"
    src_root.mkdir(parents=True)
    dst_root.mkdir(parents=True)

    files = {
        "main.py": "print('main')\n",
        "pkg/__init__.py": "# pkg init\n",
        "pkg/subpkg/module.py": "def foo():\n    return 1\n",
        "pkg/readme.txt": "do not copy\n",
    }
    for rel, content in files.items():
        _write(os.path.join(src_root, rel), content)

    # Act
    _maybe_save_custom_model_code(str(src_root), str(dst_root))

    # Assert: .py files copied with preserved structure; non-.py and __init__.py ignored
    assert (dst_root / "main.py").exists()
    assert not (dst_root / "pkg" / "__init__.py").exists()
    assert (dst_root / "pkg" / "subpkg" / "module.py").exists()
    assert not (dst_root / "pkg" / "readme.txt").exists()

    # Verify contents match
    with open(dst_root / "pkg" / "subpkg" / "module.py", "r") as f:
        assert "def foo()" in f.read()


def test_maybe_save_custom_model_code_noop_for_none_or_non_dir(tmp_path):
    dst_root = tmp_path / "hf_meta"
    dst_root.mkdir(parents=True)

    # None input should be a no-op
    _maybe_save_custom_model_code(None, str(dst_root))
    assert list(dst_root.rglob("*.py")) == []

    # Non-directory input should be a no-op
    some_file = tmp_path / "not_a_dir.txt"
    some_file.write_text("hello")
    _maybe_save_custom_model_code(str(some_file), str(dst_root))
    assert list(dst_root.rglob("*.py")) == []


def test_consolidated_hf_addon_generates_sentence_transformer_metadata_from_effective_model(tmp_path):
    source_dir = tmp_path / "source"
    metadata_dir = tmp_path / "model" / ".hf_metadata"
    consolidated_dir = tmp_path / "model" / "consolidated"
    source_files = {
        ".gitattributes": "git attributes",
        "1_Pooling/config.json": '{"pooling_mode_mean_tokens": true}',
        "2_Dense/config.json": '{"in_features": 2048, "out_features": 1024}',
        "2_Dense/model.safetensors": "stale module weights",
        "LICENSE": "license",
        "LICENSE.md": "markdown license",
        "NOTICE": "notice",
        "NOTICE.md": "markdown notice",
        "README.md": "model card",
        "config.json": '{"architectures": ["SourceModel"], "pooling": "avg"}',
        "config_sentence_transformers.json": (
            '{"prompts": {"query": "old query: ", "document": "old passage: "}, "similarity_fn_name": "cosine"}'
        ),
        "custom_module.py": "class CustomModule: pass",
        "model.safetensors": "old weights",
        "modules.json": '[{"idx": 0, "path": "2_Dense", "type": "stale.Dense"}]',
        "sentence_bert_config.json": '{"max_seq_length": 8192, "do_lower_case": false}',
    }
    for relative_path, content in source_files.items():
        _write(str(source_dir / relative_path), content)
    metadata_dir.mkdir(parents=True)
    consolidated_dir.mkdir(parents=True)

    class DeployConfig:
        hidden_size = 2048
        max_position_embeddings = 32768

        def to_json_string(self, use_diff=False):
            del use_diff
            return '{"architectures": ["DeployableModel"], "pooling": "cls"}'

    model = nn.Module()
    model.config = DeployConfig()
    model.pooling = "cls"
    model.l2_normalize = False
    model.sentence_transformer_export_config = SimpleNamespace(
        query_prompt="search: ",
        document_prompt="index: ",
        max_seq_length=4096,
        similarity_fn_name="dot",
        do_lower_case=False,
        include_prompt=True,
    )
    model.get_hf_export_config = lambda: model.config
    tokenizer = MagicMock()
    ddp_model = nn.Module()
    ddp_model.module = model
    compiled_model = nn.Module()
    compiled_model._orig_mod = ddp_model

    addon = ConsolidatedHFAddon()
    addon.pre_save(
        model_state=SimpleNamespace(model=[compiled_model]),
        hf_metadata_dir=str(metadata_dir),
        tokenizer=tokenizer,
        fqn_to_file_index_mapping={"w": 1},
        fqn_to_dtype_mapping={"w": "BF16"},
        original_model_path=str(source_dir),
        v4_compatible=True,
    )

    (consolidated_dir / "model.safetensors").write_text("new weights")
    (consolidated_dir / "model.safetensors.index.json").write_text('{"weight_map": {"w": "model.safetensors"}}')
    addon.post_save(
        consolidated_path=str(consolidated_dir),
        hf_metadata_path=str(metadata_dir),
    )

    actual_manifest = {
        path.relative_to(consolidated_dir).as_posix() for path in consolidated_dir.rglob("*") if path.is_file()
    }
    assert actual_manifest == {
        ".gitattributes",
        "1_Pooling/config.json",
        "LICENSE",
        "LICENSE.md",
        "NOTICE",
        "NOTICE.md",
        "config.json",
        "config_sentence_transformers.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "modules.json",
        "sentence_bert_config.json",
    }
    assert json.loads((consolidated_dir / "config.json").read_text()) == {
        "architectures": ["DeployableModel"],
        "pooling": "cls",
    }
    assert not (consolidated_dir / "config.v5.json").exists()
    assert json.loads((consolidated_dir / "modules.json").read_text()) == [
        {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
        {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
    ]
    pooling_config = json.loads((consolidated_dir / "1_Pooling" / "config.json").read_text())
    assert pooling_config["pooling_mode_cls_token"] is True
    assert pooling_config["pooling_mode_mean_tokens"] is False
    assert pooling_config["include_prompt"] is True
    assert pooling_config["word_embedding_dimension"] == 2048
    assert json.loads((consolidated_dir / "config_sentence_transformers.json").read_text()) == {
        "prompts": {"query": "search: ", "document": "index: "},
        "default_prompt_name": None,
        "similarity_fn_name": "dot",
    }
    assert json.loads((consolidated_dir / "sentence_bert_config.json").read_text()) == {
        "max_seq_length": 4096,
        "do_lower_case": False,
    }
    assert (consolidated_dir / "model.safetensors").read_text() == "new weights"
    assert not (consolidated_dir / "2_Dense").exists()
    assert not (consolidated_dir / "custom_module.py").exists()
    tokenizer.save_pretrained.assert_called_once_with(str(metadata_dir))


def test_generated_sentence_transformer_assets_regenerate_semantics_and_preserve_source_limit(tmp_path):
    source_dir = tmp_path / "source"
    metadata_dir = tmp_path / "metadata"
    source_dir.mkdir()
    metadata_dir.mkdir()
    (source_dir / "config_sentence_transformers.json").write_text(
        json.dumps(
            {
                "prompts": {"query": "source query: ", "document": "source document: "},
                "similarity_fn_name": "cosine",
            }
        )
    )
    (source_dir / "sentence_bert_config.json").write_text(json.dumps({"max_seq_length": 8192, "do_lower_case": False}))
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=32768),
    )
    export_config = SimpleNamespace(
        query_prompt=None,
        document_prompt=None,
        max_seq_length=None,
        similarity_fn_name=None,
        do_lower_case=None,
        include_prompt=True,
    )
    tokenizer = SimpleNamespace(model_max_length=512)

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        str(source_dir),
        str(metadata_dir),
        tokenizer,
    )

    sentence_config = json.loads((metadata_dir / "config_sentence_transformers.json").read_text())
    assert sentence_config["prompts"] == {"query": "", "document": ""}
    assert sentence_config["similarity_fn_name"] == "cosine"
    assert json.loads((metadata_dir / "sentence_bert_config.json").read_text()) == {
        "max_seq_length": 8192,
        "do_lower_case": False,
    }
    assert json.loads((metadata_dir / "modules.json").read_text())[-1] == {
        "idx": 2,
        "name": "2",
        "path": "2_Normalize",
        "type": "sentence_transformers.models.Normalize",
    }
    pooling_config = json.loads((metadata_dir / "1_Pooling" / "config.json").read_text())
    assert pooling_config["pooling_mode_mean_tokens"] is True


def test_inferred_sentence_transformer_max_seq_length_uses_smallest_finite_capability():
    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=512))
    tokenizer = SimpleNamespace(model_max_length=4096)

    assert (
        _resolve_sentence_transformer_max_seq_length(
            model,
            SimpleNamespace(max_seq_length=None),
            tokenizer,
        )
        == 512
    )
    assert (
        _resolve_sentence_transformer_max_seq_length(
            model,
            SimpleNamespace(max_seq_length=256),
            tokenizer,
        )
        == 256
    )

    with pytest.raises(ValueError, match="exceeds the model's max_position_embeddings"):
        _resolve_sentence_transformer_max_seq_length(
            model,
            SimpleNamespace(max_seq_length=1024),
            tokenizer,
        )


def test_generated_sentence_transformer_assets_remove_training_tokenizer_state(tmp_path):
    source_dir = tmp_path / "source"
    metadata_dir = tmp_path / "metadata"
    source_dir.mkdir()
    metadata_dir.mkdir()
    (source_dir / "tokenizer.json").write_text(json.dumps({"truncation": None, "padding": None, "model": {}}))
    (source_dir / "tokenizer_config.json").write_text(json.dumps({"local_files_only": False, "model_max_length": 512}))
    (metadata_dir / "tokenizer.json").write_text(
        json.dumps({"truncation": {"max_length": 32}, "padding": {"length": 32}, "model": {}})
    )
    (metadata_dir / "tokenizer_config.json").write_text(
        json.dumps({"local_files_only": True, "model_max_length": 512, "processor_class": "PixtralProcessor"})
    )
    (metadata_dir / "processor_config.json").write_text('{"processor_class": "PixtralProcessor"}')
    (metadata_dir / "preprocessor_config.json").write_text('{"image_processor_type": "PixtralImageProcessor"}')
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="query: ",
        document_prompt="passage: ",
        max_seq_length=None,
        similarity_fn_name=None,
        do_lower_case=None,
        include_prompt=True,
    )

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        str(source_dir),
        str(metadata_dir),
        SimpleNamespace(model_max_length=512),
    )

    tokenizer_json = json.loads((metadata_dir / "tokenizer.json").read_text())
    tokenizer_config = json.loads((metadata_dir / "tokenizer_config.json").read_text())
    assert tokenizer_json["truncation"] is None
    assert tokenizer_json["padding"] is None
    assert tokenizer_config["local_files_only"] is False
    assert "processor_class" not in tokenizer_config
    assert not (metadata_dir / "processor_config.json").exists()
    assert not (metadata_dir / "preprocessor_config.json").exists()


def test_generated_sentence_transformer_assets_preserve_repository_and_model_root_legal_assets(tmp_path):
    repository_root = tmp_path / "repository"
    model_root = repository_root / "encoder"
    metadata_dir = tmp_path / "metadata"
    (repository_root / "legal").mkdir(parents=True)
    model_root.mkdir()
    metadata_dir.mkdir()
    (repository_root / "LICENSE.md").write_text("repository license")
    (repository_root / "NOTICE.md").write_text("repository notice")
    (repository_root / "COPYING").write_text("copying")
    (repository_root / "legal" / "LICENSE.txt").write_text("nested license")
    (model_root / "LICENSE.md").write_text("model license")

    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        source_repository_path=str(repository_root),
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
        max_seq_length=512,
        similarity_fn_name="cosine",
        do_lower_case=False,
        include_prompt=True,
    )

    _save_generated_sentence_transformer_assets(
        model,
        export_config,
        str(model_root),
        str(metadata_dir),
        SimpleNamespace(model_max_length=512),
    )

    assert (metadata_dir / "LICENSE.md").read_text() == "model license"
    assert (metadata_dir / "NOTICE.md").read_text() == "repository notice"
    assert not (metadata_dir / "COPYING").exists()
    assert not (metadata_dir / "LICENSE.txt").exists()


def test_generated_sentence_transformer_assets_reject_unrepresentable_pooling(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    model = SimpleNamespace(
        pooling="weighted_avg",
        l2_normalize=False,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
        max_seq_length=512,
        similarity_fn_name="dot",
        do_lower_case=False,
        include_prompt=True,
    )

    with pytest.raises(ValueError, match="cannot be represented"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(metadata_dir),
            tokenizer=None,
        )


@pytest.mark.parametrize(
    ("l2_normalize", "similarity_fn_name"),
    [(False, "cosine"), (True, "dot")],
)
def test_generated_sentence_transformer_assets_reject_similarity_that_contradicts_normalization(
    tmp_path,
    l2_normalize,
    similarity_fn_name,
):
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=l2_normalize,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
        max_seq_length=512,
        similarity_fn_name=similarity_fn_name,
        do_lower_case=False,
        include_prompt=True,
    )

    with pytest.raises(ValueError, match="does not match"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(tmp_path),
            tokenizer=SimpleNamespace(model_max_length=512),
        )


def test_generated_sentence_transformer_assets_reject_prompt_exclusion_not_used_by_training(tmp_path):
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="query: ",
        document_prompt="passage: ",
        max_seq_length=512,
        similarity_fn_name="cosine",
        do_lower_case=False,
        include_prompt=False,
    )

    with pytest.raises(ValueError, match="include_prompt=False"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(tmp_path),
            tokenizer=SimpleNamespace(model_max_length=512),
        )


def test_generated_sentence_transformer_assets_reject_lowercasing_not_used_by_training(tmp_path):
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
        max_seq_length=512,
        similarity_fn_name="cosine",
        do_lower_case=True,
        include_prompt=True,
    )

    with pytest.raises(ValueError, match="do_lower_case=True"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(tmp_path),
            tokenizer=SimpleNamespace(model_max_length=512),
        )


def test_generated_sentence_transformer_assets_require_tokenizer(tmp_path):
    model = SimpleNamespace(
        pooling="avg",
        l2_normalize=True,
        config=SimpleNamespace(hidden_size=8, max_position_embeddings=512),
    )
    export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
        max_seq_length=512,
        similarity_fn_name="cosine",
        do_lower_case=False,
        include_prompt=True,
    )

    with pytest.raises(ValueError, match="tokenizer is required"):
        _save_generated_sentence_transformer_assets(
            model,
            export_config,
            original_model_path=None,
            hf_metadata_dir=str(tmp_path),
            tokenizer=None,
        )


def test_consolidated_hf_addon_validates_sentence_transformer_export_on_nonzero_rank(tmp_path):
    model = nn.Module()
    model.pooling = "weighted_avg"
    model.l2_normalize = False
    model.config = SimpleNamespace(hidden_size=8, max_position_embeddings=512)
    model.sentence_transformer_export_config = SimpleNamespace(
        query_prompt="",
        document_prompt="",
        max_seq_length=512,
        similarity_fn_name="dot",
        do_lower_case=False,
        include_prompt=True,
    )

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.barrier") as mock_barrier,
        pytest.raises(ValueError, match="cannot be represented"),
    ):
        ConsolidatedHFAddon().pre_save(
            model_state=SimpleNamespace(model=[model]),
            hf_metadata_dir=str(tmp_path),
            tokenizer=None,
            fqn_to_file_index_mapping={"w": 1},
            fqn_to_dtype_mapping=None,
            original_model_path=None,
            v4_compatible=False,
        )

    mock_barrier.assert_not_called()


def test_consolidated_hf_addon_keeps_generated_metadata_path_for_other_models(tmp_path):
    source_dir = tmp_path / "source"
    generated_metadata_dir = tmp_path / "generated"
    metadata_dir = tmp_path / "model" / ".hf_metadata"
    _write(str(source_dir / "config.json"), '{"architectures": ["SourceModel"]}')
    _write(str(source_dir / "LICENSE"), "source license")
    _write(str(generated_metadata_dir / "modeling_custom.py"), "class CustomModel: pass")
    metadata_dir.mkdir(parents=True)

    class GeneratedConfig:
        def to_json_string(self, use_diff=False):
            del use_diff
            return '{"architectures": ["GeneratedModel"]}'

    model = nn.Module()
    model.config = GeneratedConfig()
    tokenizer = MagicMock()

    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(metadata_dir),
        tokenizer=tokenizer,
        fqn_to_file_index_mapping={"w": 1},
        fqn_to_dtype_mapping=None,
        original_model_path=str(source_dir),
        generated_metadata_path=str(generated_metadata_dir),
        v4_compatible=False,
    )

    assert (metadata_dir / "config.json").read_text() == '{"architectures": ["GeneratedModel"]}'
    assert (metadata_dir / "modeling_custom.py").exists()
    assert not (metadata_dir / "LICENSE").exists()
    assert not (metadata_dir / "modules.json").exists()
    tokenizer.save_pretrained.assert_called_once_with(str(metadata_dir))


def test_model_state_keeps_lm_head_when_storage_not_shared():
    """Config-tied model with a separate lm_head keeps lm_head.weight on save.

    The resolver reports the top-level config intent (so ``uses_tied_lm_head`` is
    True here), but ModelState gates lm_head dropping on the storage-based
    ``has_local_tied_lm_head``. With a separate lm_head and no shared embedding,
    the save path must keep lm_head.weight. This is the safety that previously came
    from a force-untied exclusion list, now provided by the storage check.
    """

    class _DummyConfig:
        tie_word_embeddings = True

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    _DummyModel.__name__ = "Qwen3OmniMoeThinkerForConditionalGeneration"

    model = _DummyModel()
    state = ModelState([model])

    assert state.uses_tied_lm_head is True  # follows the top-level config flag
    assert state.has_local_tied_lm_head is False  # but the tensors do not actually share storage

    state_dict = state.state_dict()
    assert "lm_head.weight" in state_dict  # so the head is kept (storage-gated safety)


def test_model_state_drops_lm_head_when_storage_shared():
    """Config-tied model whose lm_head shares storage with the embedding drops lm_head.weight on save."""

    class _DummyConfig:
        tie_word_embeddings = True

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # genuine tie

    model = _DummyModel()
    state = ModelState([model])

    assert state.uses_tied_lm_head is True
    assert state.has_local_tied_lm_head is True

    state_dict = state.state_dict()
    assert "lm_head.weight" not in state_dict  # dropped because storage is shared
    assert "model.embed_tokens.weight" in state_dict


def test_peft_model_state_can_skip_default_group_broadcast():
    """Subset-mesh ranks already load PEFT state and must not broadcast globally."""

    class _DummyConfig:
        tie_word_embeddings = False

    model = nn.Linear(2, 2)
    model.config = _DummyConfig()
    state = ModelState(model, is_peft=True)

    with patch("nemo_automodel.components.checkpoint.stateful_wrappers.set_model_state_dict") as set_state:
        state.load_state_dict({}, strict=False, broadcast_from_rank0=False)

    options = set_state.call_args.kwargs["options"]
    assert options.full_state_dict is True
    assert options.broadcast_from_rank0 is False


# _extract_target_modules tests
def _make_model_with_named_modules(module_names):
    """Build a dummy model whose ``named_modules`` yields the given names.

    We simulate LoRA sub-modules by adding ``nn.Identity`` leaves under
    the requested paths.  ``_extract_target_modules`` looks for any
    module whose name contains "lora", so we add leaves like
    ``<target>.lora_A``.
    """
    root = nn.Module()
    for name in module_names:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
    return root


class TestExtractTargetModules:
    """Tests for _extract_target_modules with combined-projection expansion."""

    def test_simple_non_combined_modules(self):
        """Non-combined module names pass through unchanged."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_B",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.o_proj" in result
        assert "model.layers.0.mlp.down_proj" in result

    def test_qkv_proj_expanded(self):
        """qkv_proj is expanded to q_proj, k_proj, v_proj."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_B",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        # Combined name should NOT appear
        assert all("qkv_proj" not in m for m in result)

    def test_gate_up_proj_expanded(self):
        """gate_up_proj is expanded to gate_proj, up_proj."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_B",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.mlp.gate_proj" in result
        assert "model.layers.0.mlp.up_proj" in result
        assert all("gate_up_proj" not in m for m in result)

    def test_mixed_combined_and_regular(self):
        """Mixed combined and regular module names."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        expected = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        }
        assert set(result) == expected

    def test_torch_compile_prefix_stripped(self):
        """_orig_mod. prefix from torch.compile is stripped before expansion."""
        model = _make_model_with_named_modules(
            [
                "_orig_mod.model.layers.0.self_attn.qkv_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        assert all("_orig_mod" not in m for m in result)

    def test_result_is_sorted(self):
        """Return value is sorted."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert result == sorted(result)

    def test_encoder_target_modules_remapped(self):
        """Encoder model.* target modules have model. prefix stripped."""
        from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.q_proj.lora_A",
                "model.layers.0.self_attn.k_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        model.state_dict_adapter = EncoderStateDictAdapter()
        result = _extract_target_modules(model)
        assert "layers.0.self_attn.q_proj" in result
        assert "layers.0.self_attn.k_proj" in result
        assert "layers.0.mlp.down_proj" in result
        assert all(not m.startswith("model.") for m in result)


class TestMaybeStripQuantizationConfig:
    """Tests for _maybe_strip_quantization_config."""

    @staticmethod
    def _make_config_with_quant():
        cfg = type("Config", (), {})()
        cfg.quantization_config = {"quant_method": "mxfp4"}
        return cfg

    def test_strips_quantization_config_when_all_params_bf16(self):
        """quantization_config is removed when all params are standard floating-point."""
        model = nn.Linear(4, 4, dtype=torch.bfloat16)
        model.config = self._make_config_with_quant()

        _maybe_strip_quantization_config(model)
        assert not hasattr(model.config, "quantization_config")

    def test_keeps_quantization_config_when_uint8_params_exist(self):
        """quantization_config is preserved when quantized (uint8) parameters exist."""
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(torch.ones(4, 4, dtype=torch.uint8), requires_grad=False))
        model.config = self._make_config_with_quant()

        _maybe_strip_quantization_config(model)
        assert hasattr(model.config, "quantization_config")

    def test_noop_when_no_quantization_config(self):
        """No error when config has no quantization_config attribute."""
        model = nn.Linear(4, 4)
        model.config = type("Config", (), {})()

        _maybe_strip_quantization_config(model)
        assert not hasattr(model.config, "quantization_config")

    def test_noop_when_no_config(self):
        """No error when model has no config attribute."""
        model = nn.Linear(4, 4)
        _maybe_strip_quantization_config(model)

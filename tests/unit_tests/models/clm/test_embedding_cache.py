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

"""Feature-cache identity covers the full frozen-encoder contract."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nemo_automodel.components.models.clm.heads import CLMEmbeddingCacheConfig, EmbeddingTable


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_cache_save_never_publishes_partial_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    path = tmp_path / "embeddings.pt"
    table = EmbeddingTable(["state"], torch.ones(1, 8))
    if existing:
        table.save(path)
    previous = path.read_bytes() if existing else None

    def interrupted_save(value: object, destination: Path) -> None:
        destination.write_bytes(b"incomplete tensor archive")
        raise OSError("interrupted cache publication")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", interrupted_save)
        with pytest.raises(OSError, match="interrupted cache publication"):
            table.save(path)
    assert path.exists() == existing
    if existing:
        assert path.read_bytes() == previous
        torch.testing.assert_close(EmbeddingTable.load(path).lookup(["state"]), torch.ones(1, 8))
    assert list(tmp_path.iterdir()) == ([path] if existing else [])
    table.save(path)
    torch.testing.assert_close(EmbeddingTable.load(path).lookup(["state"]), torch.ones(1, 8))


def config():
    return CLMEmbeddingCacheConfig(
        encoder_name="org/encoder",
        encoder_revision="a" * 40,
        tokenizer_name="org/encoder",
        tokenizer_revision="a" * 40,
        transformers_version="5.17.0",
    )


def test_identical_inputs_reuse_path_without_creating_outputs(tmp_path):
    cache_dir = tmp_path / "cache"
    first = config().build(cache_dir=cache_dir, texts=["state", "option"])
    assert first == config().build(cache_dir=cache_dir, texts=["state", "option"])
    assert first.parent == cache_dir
    assert not cache_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder_name", "org/other"),
        ("encoder_revision", "b" * 40),
        ("tokenizer_name", "org/tokenizer"),
        ("tokenizer_revision", "b" * 40),
        ("max_length", 512),
        ("pooling", "mean"),
        ("l2_normalize", False),
        ("is_causal", False),
        ("dtype", "float32"),
        ("attention_implementation", "eager"),
        ("transformers_version", "5.18.0"),
    ],
)
def test_changed_embedding_contract_cannot_reuse_features(tmp_path, field, value):
    original = config()
    changed = replace(original, **{field: value})
    assert original.build(cache_dir=tmp_path, texts=["state"]) != changed.build(cache_dir=tmp_path, texts=["state"])


def test_exact_text_content_and_order_invalidate_cache(tmp_path):
    cfg = config()
    original = cfg.build(cache_dir=tmp_path, texts=["state", "option"])
    assert original != cfg.build(cache_dir=tmp_path, texts=["state!", "option"])
    assert original != cfg.build(cache_dir=tmp_path, texts=["option", "state"])


@pytest.mark.parametrize("field,value", [("encoder_revision", "main"), ("tokenizer_revision", ""), ("max_length", 0)])
def test_unpinned_or_invalid_embedding_contract_is_rejected(field, value):
    with pytest.raises(ValueError):
        replace(config(), **{field: value})


def test_recipe_cache_respects_encoder_dtype_alias(tmp_path):
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.decision.train_clm_heads import TrainCLMHeadsRecipe

    def cache_path(**precision):
        cfg = ConfigNode(
            {
                "encoder": {
                    "pretrained_model_name_or_path": "org/encoder",
                    "revision": "a" * 40,
                    **precision,
                },
                "encode": {"cache_dir": str(tmp_path / "cache")},
            }
        )
        recipe = TrainCLMHeadsRecipe(cfg)
        recipe.output_dir = tmp_path / "run"
        return recipe._cache_path(["state", "option"])

    original = cache_path(torch_dtype="bfloat16")
    assert cache_path(dtype="bfloat16") == original
    assert cache_path(torch_dtype="bfloat16", dtype="float32") != original
    assert cache_path(torch_dtype="bfloat16", dtype="float32") == cache_path(torch_dtype="float32")
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("pad_token", [None, "existing-pad"])
def test_model_owned_tokenizer_preserves_clm_serving_contract(pad_token):
    from types import SimpleNamespace
    from unittest.mock import patch

    from nemo_automodel.components.models.clm.prompting import prepare_tokenizer

    tokenizer = SimpleNamespace(pad_token=pad_token, eos_token="eos")
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
        actual = prepare_tokenizer(name_or_path="org/tokenizer", revision="pinned-revision")
    load.assert_called_once_with("org/tokenizer", revision="pinned-revision")
    assert actual is tokenizer
    assert actual.padding_side == "right"
    assert actual.truncation_side == "left"
    assert actual.pad_token == ("eos" if pad_token is None else pad_token)

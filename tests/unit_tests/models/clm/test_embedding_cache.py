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

import pytest

from nemo_automodel.components.models.clm.heads import CLMEmbeddingCacheConfig


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

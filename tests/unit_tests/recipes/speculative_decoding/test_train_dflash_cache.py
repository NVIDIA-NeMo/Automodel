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


"""Offline routing and the target tensors that the cached recipe still needs."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import PreTrainedConfig

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.speculative.dflash.cached_target import DFlashTargetTablesConfig
from nemo_automodel.recipes.speculative_decoding.train_dflash import TrainDFlashRecipe


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"loss_type": "mmflash"}, "train_cache_path"),
        ({"val_cache_path": "val"}, "train_cache_path"),
        ({"train_cache_path": "train", "val_data_path": "val.jsonl"}, "raw data"),
        ({"train_cache_path": "train", "packed_sequence_size": 64}, "packing"),
    ],
)
def test_cache_options_reject_incompatible_routes(args: dict[str, str | int], message: str) -> None:
    recipe = TrainDFlashRecipe(ConfigNode({"recipe_args": args}))
    with pytest.raises(ValueError, match=message):
        recipe._validate_cached_inputs()


def test_precomputed_target_batch_routes_the_same_tensors_to_training_and_validation() -> None:
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.target_wrapper = None
    recipe.compute_dtype = torch.float32
    batch = {
        "input_ids": torch.tensor([[4, 8, 6, 0]]),
        "attention_mask": torch.tensor([[True, True, True, False]]),
        "loss_mask": torch.tensor([[0, 1, 1, 0]]),
        "hidden_states": torch.ones(1, 4, 12, dtype=torch.bfloat16),
        "visual_score": torch.tensor([[0, 1000, 2000, -1]]),
        "position_ids": torch.tensor([[0, 1, 2, 0]]),
        "seq_lens": torch.tensor([[3]]),
        "doc_remaining": torch.tensor([[3, 2, 1, 0]]),
    }
    result = recipe._prepare_target_batch(batch)
    assert result.hidden_states.dtype == torch.float32
    for name in (
        "input_ids",
        "attention_mask",
        "loss_mask",
        "visual_score",
        "position_ids",
        "seq_lens",
        "doc_remaining",
    ):
        assert getattr(result, name) is batch[name]


def test_cached_tied_target_uses_embedding_weight_without_a_model_forward(tmp_path: Path) -> None:
    embedding = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    save_file({"model.embed_tokens.weight": embedding}, tmp_path / "model.safetensors")
    config = PreTrainedConfig(vocab_size=8, hidden_size=3, tie_word_embeddings=True)
    tables = DFlashTargetTablesConfig().build(
        str(tmp_path), config=config, device=torch.device("cpu"), dtype=torch.float32
    )
    assert not tables.embed_tokens.weight.requires_grad
    assert tables.lm_head.weight is tables.embed_tokens.weight
    torch.testing.assert_close(
        tables.lm_head(torch.tensor([[1.0, 2.0, 3.0]])),
        torch.nn.functional.linear(torch.tensor([[1.0, 2.0, 3.0]]), embedding),
    )


def test_cached_untied_target_loads_only_the_two_indexed_tables(tmp_path: Path) -> None:
    embed_key, head_key = "model.language_model.embed_tokens.weight", "lm_head.weight"
    embedding, head = torch.ones(8, 3), torch.zeros(8, 3)
    save_file({embed_key: embedding}, tmp_path / "embed.safetensors")
    save_file({head_key: head}, tmp_path / "head.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    embed_key: "embed.safetensors",
                    head_key: "head.safetensors",
                    "model.layers.0.weight": "unused.safetensors",
                }
            }
        )
    )
    config = PreTrainedConfig(vocab_size=8, hidden_size=3, tie_word_embeddings=False)
    tables = DFlashTargetTablesConfig(embedding_key=embed_key).build(
        str(tmp_path), config=config, device=torch.device("cpu"), dtype=torch.float32
    )
    torch.testing.assert_close(tables.embed_tokens.weight, embedding)
    torch.testing.assert_close(tables.lm_head.weight, head)
    assert not tables.lm_head.weight.requires_grad


def test_absent_lm_head_requires_owning_config_to_declare_tying(tmp_path: Path) -> None:
    save_file({"model.embed_tokens.weight": torch.ones(8, 3)}, tmp_path / "model.safetensors")
    config = PreTrainedConfig(tie_word_embeddings=False)
    config.text_config = PreTrainedConfig(vocab_size=8, hidden_size=3, tie_word_embeddings=True)
    with pytest.raises(ValueError, match="Untied target checkpoint"):
        DFlashTargetTablesConfig().build(str(tmp_path), config=config, device=torch.device("cpu"), dtype=torch.float32)


def test_checkpoint_reads_cached_layer_order_from_draft_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recipe = TrainDFlashRecipe.__new__(TrainDFlashRecipe)
    recipe.target_wrapper = None
    recipe.block_size = 16
    recipe.mask_token_id = 248070
    recipe.draft_model = SimpleNamespace(config=SimpleNamespace(dflash_config={"target_layer_ids": [1, 8, 15, 22, 29]}))
    captured = {}
    monkeypatch.setattr(recipe, "_save_meta", lambda *args, **fields: captured.update(fields))
    recipe._save_extra_state(str(tmp_path), epoch=1)
    assert captured["target_layer_ids"] == [1, 8, 15, 22, 29]

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for loading SpecForge DFlash captures without a target forward."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.cached_data import (
    CachedDFlashDataConfig,
    CachedDFlashDataset,
    CachedDFlashRow,
    collate_cached_dflash,
)
from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

pytestmark = pytest.mark.cpu_only

_TARGET = "example/target"
_LAYERS = [0, 2]
_WIDTH = 3
_VOCAB_SIZE = 20
_BLOCK_SIZE = 3
_MASK_TOKEN_ID = 19
_MAX_SEQ_LEN = 10


def _write_manifest(path: Path, **metadata: int | str | list[int] | None) -> None:
    fields = {
        "format_version": 1,
        "target_model_name_or_path": _TARGET,
        "target_layer_ids": _LAYERS,
        "target_hidden_size": _WIDTH,
        "target_vocab_size": _VOCAB_SIZE,
        "block_size": _BLOCK_SIZE,
        "mask_token_id": _MASK_TOKEN_ID,
        "hidden_states_dtype": "bfloat16",
        "visual_score_scale": 10_000,
        "num_samples": 1,
    }
    (path / "manifest.json").write_text(json.dumps(fields | metadata), encoding="utf-8")


def _dataset(path: Path, *, require_visual_score: bool = False) -> CachedDFlashDataset:
    return CachedDFlashDataset(
        path,
        target_model_name_or_path=_TARGET,
        target_layer_ids=_LAYERS,
        target_hidden_size=_WIDTH,
        target_vocab_size=_VOCAB_SIZE,
        block_size=_BLOCK_SIZE,
        mask_token_id=_MASK_TOKEN_ID,
        max_seq_len=_MAX_SEQ_LEN,
        require_visual_score=require_visual_score,
    )


def _loader(path: Path, *, pad_token_id: int, shuffle: bool = False) -> DataLoader[CachedDFlashRow]:
    return CachedDFlashDataConfig(cache_dir=path, max_seq_len=_MAX_SEQ_LEN, batch_size=2, shuffle=shuffle).build(
        target_model_name_or_path=_TARGET,
        target_layer_ids=_LAYERS,
        target_hidden_size=_WIDTH,
        target_vocab_size=_VOCAB_SIZE,
        block_size=_BLOCK_SIZE,
        mask_token_id=_MASK_TOKEN_ID,
        pad_token_id=pad_token_id,
    )


def test_single_rank_cached_shuffle_restores_an_epoch_order(tmp_path: Path) -> None:
    _write_manifest(tmp_path, num_samples=6)
    for index in range(6):
        torch.save(_sample(), tmp_path / f"{index:08}.ckpt")
    initial = _loader(tmp_path, pad_token_id=0, shuffle=True).sampler
    epoch_zero = list(initial)
    initial.set_epoch(2)
    resumed = _loader(tmp_path, pad_token_id=0, shuffle=True).sampler
    resumed.set_epoch(2)
    assert list(initial) == list(resumed)
    assert list(resumed) != epoch_zero


def _sample(length: int = 5, *, text_only: bool = False) -> dict[str, torch.Tensor]:
    """Return unshifted input ids, assistant mask and score [sequence], and features [sequence, layers * hidden]."""
    ids = torch.arange(length, dtype=torch.long) + 2
    loss_mask = (torch.arange(length) > 0).long()
    hidden = torch.arange(length * len(_LAYERS) * _WIDTH, dtype=torch.bfloat16).reshape(length, -1)
    scores = torch.arange(length, dtype=torch.long) * 500
    if text_only:
        scores.fill_(-1)
    return {"input_ids": ids, "loss_mask": loss_mask, "hidden_states": hidden, "visual_score": scores}


def test_reads_the_same_unshifted_rows_for_dflash_and_mmflash(tmp_path: Path) -> None:
    _write_manifest(tmp_path, num_samples=2)
    image = _sample()
    text = _sample(text_only=True)
    torch.save(text, tmp_path / "00000005.ckpt")
    torch.save(image, tmp_path / "00000001.ckpt")

    dflash = _dataset(tmp_path)
    mmflash = _dataset(tmp_path, require_visual_score=True)

    assert len(dflash) == len(mmflash) == 2
    for dataset in (dflash, mmflash):
        for i, expected in enumerate((image, text)):
            for key in expected:
                torch.testing.assert_close(dataset[i][key], expected[key])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("format_version", 2),
        ("target_model_name_or_path", "example/other-target"),
        ("target_layer_ids", [2, 0]),
        ("target_hidden_size", 4),
        ("target_vocab_size", 22),
        ("block_size", 4),
        ("mask_token_id", 18),
        ("hidden_states_dtype", "int8"),
        ("visual_score_scale", 100),
        ("num_samples", 2),
    ],
)
def test_incompatible_cache_manifest_fails_at_setup(tmp_path: Path, field: str, value: int | str | list[int]) -> None:
    _write_manifest(tmp_path, **{field: value})
    torch.save(_sample(), tmp_path / "00000001.ckpt")

    with pytest.raises(ValueError, match=field):
        _dataset(tmp_path)


def test_missing_manifest_fails_at_setup(tmp_path: Path) -> None:
    torch.save(_sample(), tmp_path / "00000001.ckpt")

    with pytest.raises(FileNotFoundError, match=r"manifest\.json"):
        _dataset(tmp_path)


def test_scoreless_cache_is_usable_by_dflash(tmp_path: Path) -> None:
    _write_manifest(tmp_path, visual_score_scale=None)
    sample = _sample()
    del sample["visual_score"]
    torch.save(sample, tmp_path / "00000001.ckpt")

    dataset = _dataset(tmp_path)
    batch = collate_cached_dflash([dataset[0]], pad_token_id=0)

    assert "visual_score" not in batch
    with pytest.raises(ValueError, match="visual_score"):
        _dataset(tmp_path, require_visual_score=True)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("input_ids", torch.tensor([2, 3, 4, 5, 20]), "input_ids"),
        ("input_ids", torch.tensor([2, 3, 4, 5, -100]), "input_ids"),
        ("input_ids", torch.tensor([2, 3, 4, 5, 6], dtype=torch.int32), "input_ids"),
        ("loss_mask", torch.tensor([0, 1, 1, 2, 1]), "loss_mask"),
        ("loss_mask", torch.tensor([0, 1, 1, 1]), "loss_mask"),
        ("loss_mask", torch.tensor([0, 0, 0, 1, 1]), "anchor"),
        ("hidden_states", torch.zeros(5, 7, dtype=torch.bfloat16), "hidden_states"),
        ("hidden_states", torch.zeros(5, 6, dtype=torch.float32), "hidden_states"),
        ("visual_score", torch.zeros(5), "visual_score"),
        ("visual_score", torch.tensor([0, 1, 2, 3, -2]), "visual_score"),
        ("visual_score", torch.tensor([0, 1, 2, 3, 10_001]), "visual_score"),
        ("visual_score", torch.tensor([0, 1, 2, 3]), "visual_score"),
        ("visual_score", torch.tensor([0, 1, 2, 3, -1]), "visual_score"),
        ("visual_score", torch.tensor([1, 1, 2, 3, 4]), "visual_score"),
        ("visual_score", None, "visual_score"),
    ],
)
def test_malformed_rows_fail_when_read(tmp_path: Path, field: str, value: torch.Tensor | None, error: str) -> None:
    """Assert a malformed replacement is rejected when the sample is loaded.

    Args:
        tmp_path: Cache directory.
        field: Cache field to replace.
        value: Replacement tensor with arbitrary shape and dtype, or None to
            remove the field. Valid shapes are [sequence] for token ids,
            mask and score, and [sequence, layers * hidden] for hidden states.
        error: Expected diagnostic text.
    """
    _write_manifest(tmp_path)
    sample = _sample()
    if value is None:
        del sample[field]
    else:
        sample[field] = value
    torch.save(sample, tmp_path / "00000009.ckpt")
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match=rf"00000009\.ckpt.*{error}"):
        dataset[0]


def test_reader_rejects_overlength_rows(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    torch.save(_sample(_MAX_SEQ_LEN + 1), tmp_path / "00000001.ckpt")
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="sequence length"):
        dataset[0]


def test_collator_keeps_shorter_rows_training_blocks_inside_real_tokens(tmp_path: Path) -> None:
    _write_manifest(tmp_path, num_samples=2)
    shorter = _sample(length=5)
    longer = _sample(length=7, text_only=True)
    torch.save(shorter, tmp_path / "00000001.ckpt")
    torch.save(longer, tmp_path / "00000002.ckpt")
    batch = next(iter(_loader(tmp_path, pad_token_id=2)))

    torch.testing.assert_close(batch["input_ids"], torch.tensor([[2, 3, 4, 5, 6, 2, 2], [2, 3, 4, 5, 6, 7, 8]]))
    torch.testing.assert_close(batch["loss_mask"], torch.tensor([[0, 1, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1, 1]]))
    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[True] * 5 + [False] * 2, [True] * 7]))
    torch.testing.assert_close(batch["seq_lens"], torch.tensor([[5], [7]]))
    torch.testing.assert_close(batch["position_ids"], torch.arange(7).expand(2, -1))
    torch.testing.assert_close(batch["doc_remaining"], torch.tensor([[4, 3, 2, 1, 0, 0, 0], [6, 5, 4, 3, 2, 1, 0]]))
    torch.testing.assert_close(batch["hidden_states"][0, :5], shorter["hidden_states"])
    torch.testing.assert_close(batch["hidden_states"][0, 5:], torch.zeros(2, 6, dtype=torch.bfloat16))
    torch.testing.assert_close(batch["hidden_states"][1], longer["hidden_states"])
    torch.testing.assert_close(batch["visual_score"], torch.tensor([[0, 500, 1000, 1500, 2000, -1, -1], [-1] * 7]))


def test_loader_requires_valid_padding_token_even_with_masked_loss(tmp_path: Path) -> None:
    _write_manifest(tmp_path)
    torch.save(_sample(), tmp_path / "00000001.ckpt")

    with pytest.raises(ValueError, match="pad_token_id"):
        _loader(tmp_path, pad_token_id=_VOCAB_SIZE)


def test_padded_cache_training_matches_unpadded_rows(tmp_path: Path) -> None:
    _write_manifest(tmp_path, num_samples=2)
    for i, length in enumerate((5, 7)):
        torch.save(_sample(length), tmp_path / f"{i:08d}.ckpt")
    loader = _loader(tmp_path, pad_token_id=0)
    batch = next(iter(loader))

    draft = Qwen3DFlashDraftModel(
        Qwen3Config(
            vocab_size=_VOCAB_SIZE,
            hidden_size=_WIDTH,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=2,
            max_position_embeddings=32,
            num_target_layers=3,
            block_size=_BLOCK_SIZE,
            attn_implementation="sdpa",
            dflash_config={"mask_token_id": _MASK_TOKEN_ID, "target_layer_ids": _LAYERS},
        )
    )
    trainer = DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(_WIDTH, _VOCAB_SIZE, bias=False),
        target_embed_tokens=torch.nn.Embedding(_VOCAB_SIZE, _WIDTH),
        mask_token_id=_MASK_TOKEN_ID,
        block_size=_BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=_MAX_SEQ_LEN,
    )

    with torch.no_grad():
        packed = trainer(
            input_ids=batch["input_ids"],
            hidden_states=batch["hidden_states"].float(),
            loss_mask=batch["loss_mask"],
            position_ids=batch["position_ids"],
            seq_lens=batch["seq_lens"],
            doc_remaining=batch["doc_remaining"],
        )
        individual = [
            trainer(
                input_ids=row["input_ids"].unsqueeze(0),
                hidden_states=row["hidden_states"].unsqueeze(0).float(),
                loss_mask=row["loss_mask"].unsqueeze(0),
            )
            for row in loader.dataset
        ]

    assert packed.valid_blocks == sum(result.valid_blocks for result in individual)
    assert packed.valid_tokens == sum(result.valid_tokens for result in individual)
    torch.testing.assert_close(
        packed.loss * packed.loss_weight,
        sum(result.loss * result.loss_weight for result in individual),
    )

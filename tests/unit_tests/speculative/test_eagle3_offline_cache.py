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

"""Unit tests for the EAGLE-3 offline target-output cache (SpecForge offline path).

The headline guarantee is *equivalence*: training from the precomputed cache
must produce exactly the same loss as the online path, because the cache simply
stores the output of ``_compute_target_distribution`` that the online trainer
would otherwise compute on the fly.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.datasets.llm.eagle3_cache import (
    CACHE_KEYS,
    CachedEagle3Dataset,
    build_cached_eagle3_dataloader,
    read_manifest,
    read_target_embeddings,
    write_manifest,
    write_shard,
    write_target_embeddings,
)
from nemo_automodel.components.speculative.eagle.core import Eagle3TrainerModule, _compute_target_distribution
from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.precompute_eagle3 import _build_parser, _compute_batch_cache, _validate_args

_VOCAB = 64
_DRAFT_VOCAB = 16
_HIDDEN = 32


def _build_module(ttt_steps=2):
    config = LlamaConfig(
        hidden_size=_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=_VOCAB,
        max_position_embeddings=64,
    )
    config.torch_dtype = torch.float32
    config.draft_vocab_size = _DRAFT_VOCAB
    config.target_hidden_size = _HIDDEN
    draft = LlamaEagle3DraftModel(config).to(torch.float32)
    selected_token_ids = torch.arange(_DRAFT_VOCAB, dtype=torch.long)
    selected_token_mask = torch.zeros(_VOCAB, dtype=torch.bool)
    selected_token_mask[selected_token_ids] = True
    module = Eagle3TrainerModule(
        draft,
        selected_token_ids=selected_token_ids,
        selected_token_mask=selected_token_mask,
        ttt_steps=ttt_steps,
    )
    return module, selected_token_ids, selected_token_mask


def _fake_target_batch(batch_size=2, seq_len=6):
    torch.manual_seed(0)
    return SimpleNamespace(
        input_ids=torch.randint(0, _VOCAB, (batch_size, seq_len)),
        attention_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        loss_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        aux_hidden_states=torch.randn(batch_size, seq_len, _HIDDEN * 3),
        logits=torch.randn(batch_size, seq_len, _VOCAB),
    )


# ---------------------------------------------------------------------------
# Equivalence: cached (target_probs/position_mask) == live (target_logits)
# ---------------------------------------------------------------------------


def test_forward_cached_matches_live():
    module, ids, mask = _build_module()
    tb = _fake_target_batch()

    live = module(
        input_ids=tb.input_ids,
        attention_mask=tb.attention_mask,
        loss_mask=tb.loss_mask,
        aux_hidden_states=tb.aux_hidden_states,
        target_logits=tb.logits,
    )
    target_probs, position_mask = _compute_target_distribution(
        target_logits=tb.logits, selected_token_ids=ids, selected_token_mask=mask, loss_mask=tb.loss_mask
    )
    cached = module(
        input_ids=tb.input_ids,
        attention_mask=tb.attention_mask,
        loss_mask=tb.loss_mask,
        aux_hidden_states=tb.aux_hidden_states,
        target_probs=target_probs,
        position_mask=position_mask,
    )
    torch.testing.assert_close(live.loss, cached.loss)
    torch.testing.assert_close(live.accuracy, cached.accuracy)
    torch.testing.assert_close(live.valid_tokens, cached.valid_tokens)


def test_forward_requires_a_supervision_source():
    module, _, _ = _build_module()
    tb = _fake_target_batch()
    with pytest.raises(ValueError, match="either target_logits"):
        module(
            input_ids=tb.input_ids,
            attention_mask=tb.attention_mask,
            loss_mask=tb.loss_mask,
            aux_hidden_states=tb.aux_hidden_states,
        )


def test_forward_rejects_both_supervision_sources():
    module, ids, mask = _build_module()
    tb = _fake_target_batch()
    target_probs, position_mask = _compute_target_distribution(
        target_logits=tb.logits, selected_token_ids=ids, selected_token_mask=mask, loss_mask=tb.loss_mask
    )
    with pytest.raises(ValueError, match="exactly one supervision source"):
        module(
            input_ids=tb.input_ids,
            attention_mask=tb.attention_mask,
            loss_mask=tb.loss_mask,
            aux_hidden_states=tb.aux_hidden_states,
            target_logits=tb.logits,
            target_probs=target_probs,
            position_mask=position_mask,
        )


# ---------------------------------------------------------------------------
# _compute_batch_cache (the producer's per-batch step)
# ---------------------------------------------------------------------------


def test_compute_batch_cache_matches_compute_target_distribution():
    _, ids, mask = _build_module()
    tb = _fake_target_batch()
    cache = _compute_batch_cache(tb, ids, mask, cache_dtype=torch.float32)

    assert set(cache) == set(CACHE_KEYS)
    target_probs, position_mask = _compute_target_distribution(
        target_logits=tb.logits, selected_token_ids=ids, selected_token_mask=mask, loss_mask=tb.loss_mask
    )
    torch.testing.assert_close(cache["target_probs"], target_probs)
    torch.testing.assert_close(cache["position_mask"], position_mask)
    torch.testing.assert_close(cache["aux_hidden_states"], tb.aux_hidden_states)
    assert cache["input_ids"].dtype == torch.long
    assert cache["position_mask"].dtype == torch.bool


def test_compute_batch_cache_downcasts_floats():
    _, ids, mask = _build_module()
    tb = _fake_target_batch()
    cache = _compute_batch_cache(tb, ids, mask, cache_dtype=torch.bfloat16)
    assert cache["aux_hidden_states"].dtype == torch.bfloat16
    assert cache["target_probs"].dtype == torch.bfloat16
    assert cache["input_ids"].dtype == torch.long  # ids/masks stay exact


# ---------------------------------------------------------------------------
# On-disk round-trip via the cache dataset
# ---------------------------------------------------------------------------


def _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4):
    samples = {
        "input_ids": torch.randint(0, _VOCAB, (num_samples, seq_len), dtype=torch.long),
        "attention_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        "loss_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        "aux_hidden_states": torch.randn(num_samples, seq_len, _HIDDEN * 3),
        "target_probs": torch.rand(num_samples, seq_len, _DRAFT_VOCAB),
        "position_mask": torch.ones(num_samples, seq_len, 1, dtype=torch.bool),
    }
    shard_index = 0
    for start in range(0, num_samples, shard_size):
        chunk = {k: v[start : start + shard_size] for k, v in samples.items()}
        write_shard(cache_dir, shard_index, chunk)
        shard_index += 1
    write_manifest(
        cache_dir,
        {
            "target_model": "tiny",
            "target_vocab_size": _VOCAB,
            "draft_vocab_size": _DRAFT_VOCAB,
            "seq_length": seq_len,
            "dtype": "fp32",
            "num_samples": num_samples,
            "shard_size": shard_size,
            "aux_hidden_dim": _HIDDEN * 3,
            "aux_layer_ids": [1, 2, 3],
            "selected_token_ids": list(range(_DRAFT_VOCAB)),
        },
    )
    return samples


def test_cache_dataset_round_trip(tmp_path):
    cache_dir = str(tmp_path / "cache")
    samples = _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4)

    dataset = CachedEagle3Dataset(cache_dir)
    assert len(dataset) == 3
    item = dataset[2]  # last sample lives in the trailing partial shard
    assert set(item) == set(CACHE_KEYS)
    torch.testing.assert_close(item["aux_hidden_states"], samples["aux_hidden_states"][2])
    torch.testing.assert_close(item["target_probs"], samples["target_probs"][2])
    assert int(item["input_ids"][0]) == int(samples["input_ids"][2][0])


def test_cache_dataloader_batches(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _write_tiny_cache(cache_dir, num_samples=4, shard_size=2, seq_len=4)
    loader = build_cached_eagle3_dataloader(cache_dir=cache_dir, batch_size=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["aux_hidden_states"].shape == (2, 4, _HIDDEN * 3)
    assert batches[0]["target_probs"].shape == (2, 4, _DRAFT_VOCAB)


def test_cache_dataset_rejects_missing_shards(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _write_tiny_cache(cache_dir, num_samples=4, shard_size=2, seq_len=4)
    # Declare more samples than shards cover.
    manifest = read_manifest(cache_dir)
    manifest.pop("format_version", None)
    manifest["num_samples"] = 99
    write_manifest(cache_dir, manifest)
    with pytest.raises(ValueError, match="shard files"):
        CachedEagle3Dataset(cache_dir)


def test_target_embeddings_round_trip(tmp_path):
    cache_dir = str(tmp_path / "cache")
    weight = torch.randn(_VOCAB, _HIDDEN)
    write_target_embeddings(cache_dir, weight)
    loaded = read_target_embeddings(cache_dir)
    torch.testing.assert_close(loaded, weight)


def test_read_target_embeddings_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="target embeddings"):
        read_target_embeddings(str(tmp_path))


# ---------------------------------------------------------------------------
# Producer arg validation
# ---------------------------------------------------------------------------


def _args(**overrides):
    base = dict(batch_size=4, shard_size=256, draft_vocab_size=8192, dtype="bf16")
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        ({"batch_size": 0}, "batch-size"),
        ({"shard_size": 0}, "shard-size"),
        ({"shard_size": 100, "batch_size": 8}, "multiple of"),  # 100 % 8 != 0
        ({"draft_vocab_size": 0}, "draft-vocab-size"),
        ({"dtype": "int4"}, "dtype"),
    ],
)
def test_validate_args_rejects_invalid(overrides, pattern):
    with pytest.raises(ValueError, match=pattern):
        _validate_args(_args(**overrides))


def test_validate_args_accepts_valid():
    _validate_args(_args())
    _validate_args(_args(draft_vocab_size=None))  # None = full vocab is allowed


def test_parser_requires_core_args():
    parser = _build_parser()
    args = parser.parse_args(
        ["--target-model", "m", "--input-data", "d", "--output-dir", "o", "--shard-size", "8", "--batch-size", "4"]
    )
    assert args.shard_size == 8 and args.batch_size == 4

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

import json
import os
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from nemo_automodel.components.datasets.llm import eagle3_cache as ec
from nemo_automodel.components.datasets.llm.eagle3_cache import (
    CACHE_KEYS,
    CachedEagle3Dataset,
    build_cached_eagle3_dataloader,
    compress_target_probs,
    existing_shard_indices,
    manifest_path,
    read_manifest,
    read_target_embeddings,
    reconstruct_target_probs,
    write_manifest,
    write_shard,
    write_target_embeddings,
)
from nemo_automodel.components.speculative import precompute_eagle3 as pe
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


def test_compute_batch_cache_lossless_matches_compute_target_distribution():
    _, ids, mask = _build_module()
    tb = _fake_target_batch()
    # topk=0 (the default) stores the full distribution -- exact, no factorization.
    cache = _compute_batch_cache(tb, ids, mask, cache_dtype=torch.float32, target_probs_topk=0)

    assert set(cache) == set(CACHE_KEYS)
    target_probs, position_mask = _compute_target_distribution(
        target_logits=tb.logits, selected_token_ids=ids, selected_token_mask=mask, loss_mask=tb.loss_mask
    )
    torch.testing.assert_close(cache["target_probs"], target_probs)
    torch.testing.assert_close(cache["position_mask"], position_mask)
    torch.testing.assert_close(cache["aux_hidden_states"], tb.aux_hidden_states)
    assert cache["input_ids"].dtype == torch.long
    assert cache["position_mask"].dtype == torch.bool


def test_compute_batch_cache_compressed_stores_topk():
    _, ids, mask = _build_module()
    tb = _fake_target_batch()
    # A positive k below the draft vocab switches to the top-k factorization.
    cache = _compute_batch_cache(tb, ids, mask, cache_dtype=torch.float32, target_probs_topk=4)

    assert "target_probs" not in cache
    assert cache["target_probs_values"].shape[-1] == 4
    assert cache["target_probs_indices"].dtype == torch.int32
    target_probs, _ = _compute_target_distribution(
        target_logits=tb.logits, selected_token_ids=ids, selected_token_mask=mask, loss_mask=tb.loss_mask
    )
    reconstructed = reconstruct_target_probs(cache["target_probs_values"], cache["target_probs_indices"], _DRAFT_VOCAB)
    # The kept top tokens carry the same argmax as the live distribution.
    assert (reconstructed.argmax(-1) == target_probs.argmax(-1)).all()


def test_compute_batch_cache_downcasts_floats():
    _, ids, mask = _build_module()
    tb = _fake_target_batch()
    cache = _compute_batch_cache(tb, ids, mask, cache_dtype=torch.bfloat16, target_probs_topk=4)
    assert cache["aux_hidden_states"].dtype == torch.bfloat16
    assert cache["target_probs_values"].dtype == torch.bfloat16
    assert cache["target_probs_indices"].dtype == torch.int32  # indices stay exact
    assert cache["input_ids"].dtype == torch.long  # ids/masks stay exact


# ---------------------------------------------------------------------------
# target_probs top-k compression / reconstruction
# ---------------------------------------------------------------------------


def test_compress_reconstruct_preserves_peaked_mass():
    torch.manual_seed(0)
    target_probs = torch.softmax(torch.randn(3, 5, _DRAFT_VOCAB) * 6.0, dim=-1)
    values, indices = compress_target_probs(target_probs, topk=4)

    assert values.shape[-1] == 4 and indices.dtype == torch.int32
    # The kept mass is renormalized to a proper distribution.
    torch.testing.assert_close(values.sum(-1), torch.ones(3, 5))
    reconstructed = reconstruct_target_probs(values, indices, _DRAFT_VOCAB)
    # The argmax (the always-supervised top token) is never dropped.
    assert (reconstructed.argmax(-1) == target_probs.argmax(-1)).all()


def test_compress_lossless_when_topk_at_least_vocab():
    target_probs = torch.softmax(torch.randn(2, 4, _DRAFT_VOCAB), dim=-1)
    values, indices = compress_target_probs(target_probs, topk=_DRAFT_VOCAB + 10)

    assert values.shape[-1] == _DRAFT_VOCAB  # clamped to the vocab width
    reconstructed = reconstruct_target_probs(values, indices, _DRAFT_VOCAB)
    torch.testing.assert_close(reconstructed, target_probs)


# ---------------------------------------------------------------------------
# On-disk round-trip via the cache dataset
# ---------------------------------------------------------------------------


def _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4, topk=0):
    """Write a tiny cache (lossless by default, top-k when ``topk`` compresses).

    Returns ``(disk_samples, expected_target_probs)`` where the expectation is what
    the reader yields for ``target_probs`` (reconstructed under compression).
    """
    target_probs = torch.softmax(torch.randn(num_samples, seq_len, _DRAFT_VOCAB), dim=-1)
    samples = {
        "input_ids": torch.randint(0, _VOCAB, (num_samples, seq_len), dtype=torch.long),
        "attention_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        "loss_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        "aux_hidden_states": torch.randn(num_samples, seq_len, _HIDDEN * 3),
        "position_mask": torch.ones(num_samples, seq_len, 1, dtype=torch.bool),
    }
    if 0 < topk < _DRAFT_VOCAB:
        values, indices = compress_target_probs(target_probs, topk)
        samples["target_probs_values"] = values
        samples["target_probs_indices"] = indices
        expected = reconstruct_target_probs(values, indices, _DRAFT_VOCAB)
    else:
        samples["target_probs"] = target_probs
        expected = target_probs

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
            "target_probs_topk": topk,
            "aux_hidden_dim": _HIDDEN * 3,
            "aux_layer_ids": [1, 2, 3],
            "selected_token_ids": list(range(_DRAFT_VOCAB)),
        },
    )
    return samples, expected


def test_cache_dataset_round_trip_lossless(tmp_path):
    cache_dir = str(tmp_path / "cache")
    samples, expected_target_probs = _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4)

    dataset = CachedEagle3Dataset(cache_dir)
    assert len(dataset) == 3
    item = dataset[2]  # last sample lives in the trailing partial shard
    assert set(item) == set(CACHE_KEYS)
    torch.testing.assert_close(item["aux_hidden_states"], samples["aux_hidden_states"][2])
    torch.testing.assert_close(item["target_probs"], expected_target_probs[2])
    assert int(item["input_ids"][0]) == int(samples["input_ids"][2][0])


def test_cache_dataset_round_trip_compressed(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _, expected_target_probs = _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4, topk=4)

    dataset = CachedEagle3Dataset(cache_dir)
    item = dataset[2]
    # The reader yields full-width target_probs (reconstructed), not the disk keys.
    assert set(item) == set(CACHE_KEYS)
    assert item["target_probs"].shape == (4, _DRAFT_VOCAB)
    torch.testing.assert_close(item["target_probs"], expected_target_probs[2])


def test_write_shard_rejects_ambiguous_target_probs(tmp_path):
    cache_dir = str(tmp_path / "cache")
    base = {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
        "loss_mask": torch.ones(1, 4, dtype=torch.long),
        "aux_hidden_states": torch.zeros(1, 4, _HIDDEN * 3),
        "position_mask": torch.ones(1, 4, 1, dtype=torch.bool),
    }
    # Neither representation present.
    with pytest.raises(ValueError, match="exactly one target_probs"):
        write_shard(cache_dir, 0, dict(base))
    # Both representations present.
    both = dict(base, target_probs=torch.rand(1, 4, _DRAFT_VOCAB))
    both["target_probs_values"] = torch.rand(1, 4, 2)
    both["target_probs_indices"] = torch.zeros(1, 4, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="exactly one target_probs"):
        write_shard(cache_dir, 0, both)


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
    base = dict(batch_size=4, shard_size=256, draft_vocab_size=8192, target_probs_topk=64, dtype="bf16")
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        ({"batch_size": 0}, "batch-size"),
        ({"shard_size": 0}, "shard-size"),
        ({"shard_size": 100, "batch_size": 8}, "multiple of"),  # 100 % 8 != 0
        ({"draft_vocab_size": 0}, "draft-vocab-size"),
        ({"target_probs_topk": -1}, "target-probs-topk"),
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


# ---------------------------------------------------------------------------
# Producer end-to-end (`precompute_eagle3.main`) with the target/data faked,
# so the orchestration loop -- token-map scan, shard flushing, trailing
# partial shard, manifest, and the resume skip path -- runs on CPU.
# ---------------------------------------------------------------------------


class _FakeTargetModel:
    def __init__(self):
        self.config = SimpleNamespace(vocab_size=_VOCAB, hidden_size=_HIDDEN)
        self._embed = SimpleNamespace(weight=torch.randn(_VOCAB, _HIDDEN))

    def to(self, device):
        return self

    def get_input_embeddings(self):
        return self._embed


class _FakeWrapper:
    def __init__(self, model, aux_layer_ids=None):
        self.aux_layer_ids = list(aux_layer_ids) if aux_layer_ids else [0, 1, 2]

    def generate_batch(self, *, input_ids, attention_mask, loss_mask):
        b, s = input_ids.shape
        return SimpleNamespace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            aux_hidden_states=torch.randn(b, s, _HIDDEN * len(self.aux_layer_ids)),
            logits=torch.randn(b, s, _VOCAB),
        )


class _FakeDS:
    def __init__(self, n):
        self._n = n

    def __len__(self):
        return self._n


class _FakeLoader:
    def __init__(self, batches, num_samples):
        self._batches = batches
        self.dataset = _FakeDS(num_samples)

    def __iter__(self):
        return iter(self._batches)


def _patch_producer(monkeypatch, *, num_samples, batch_size, seq_len=6):
    """Patch the producer's model/tokenizer/dataloader so ``main`` runs on CPU."""
    ids = torch.arange(_DRAFT_VOCAB, dtype=torch.long)
    mask = torch.zeros(_VOCAB, dtype=torch.bool)
    mask[ids] = True
    batches, remaining = [], num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        batches.append(
            {
                "input_ids": torch.randint(0, _VOCAB, (b, seq_len)),
                "attention_mask": torch.ones(b, seq_len, dtype=torch.long),
                "loss_mask": torch.ones(b, seq_len, dtype=torch.long),
            }
        )
        remaining -= b
    loader = _FakeLoader(batches, num_samples)
    monkeypatch.setattr(pe, "NeMoAutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()))
    monkeypatch.setattr(
        pe, "NeMoAutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *a, **k: _FakeTargetModel())
    )
    monkeypatch.setattr(pe, "HFEagle3TargetModel", _FakeWrapper)
    monkeypatch.setattr(pe, "build_eagle3_dataloader", lambda **k: loader)
    monkeypatch.setattr(pe, "build_eagle3_token_mapping", lambda *a, **k: (ids, mask))


def _producer_argv(out_dir, *, extra=None):
    argv = [
        "--target-model",
        "fake-target",
        "--input-data",
        "fake-data",
        "--output-dir",
        str(out_dir),
        "--device",
        "cpu",
        "--batch-size",
        "2",
        "--shard-size",
        "2",
        "--draft-vocab-size",
        str(_DRAFT_VOCAB),
        "--seq-length",
        "6",
    ]
    return argv + (extra or [])


def test_producer_main_writes_full_sharded_cache(monkeypatch, tmp_path):
    # 5 samples, shard_size 2 -> shards 0,1 full + shard 2 trailing partial.
    _patch_producer(monkeypatch, num_samples=5, batch_size=2)
    assert pe.main(_producer_argv(tmp_path)) == 0

    manifest = read_manifest(str(tmp_path))
    assert manifest["num_samples"] == 5
    assert manifest["draft_vocab_size"] == _DRAFT_VOCAB
    assert manifest["aux_hidden_dim"] == _HIDDEN * 3
    assert sorted(existing_shard_indices(str(tmp_path))) == [0, 1, 2]

    ds = CachedEagle3Dataset(str(tmp_path))
    assert len(ds) == 5
    assert set(ds[0]) == set(CACHE_KEYS)
    # negative index wraps; out-of-range raises (CachedEagle3Dataset.__getitem__).
    torch.testing.assert_close(ds[-1]["input_ids"], ds[4]["input_ids"])
    with pytest.raises(IndexError):
        ds[99]


def test_producer_main_resume_skips_existing_shards(monkeypatch, tmp_path):
    _patch_producer(monkeypatch, num_samples=5, batch_size=2)
    assert pe.main(_producer_argv(tmp_path)) == 0
    shard0 = os.path.join(str(tmp_path), "shard-000000.safetensors")
    mtime_before = os.path.getmtime(shard0)

    # Re-running without --resume must refuse to clobber an existing cache.
    with pytest.raises(ValueError, match="already has shards"):
        pe.main(_producer_argv(tmp_path))

    # With --resume every shard is already present, so the loop takes the skip
    # branch and rewrites nothing.
    _patch_producer(monkeypatch, num_samples=5, batch_size=2)
    assert pe.main(_producer_argv(tmp_path, extra=["--resume"])) == 0
    assert sorted(existing_shard_indices(str(tmp_path))) == [0, 1, 2]
    assert os.path.getmtime(shard0) == mtime_before


def test_producer_main_resume_rejects_changed_config(monkeypatch, tmp_path):
    """--resume with a config that no longer matches the recorded manifest must
    refuse: old shards encode the old selected_token_ids / seq_length and would
    be silently mixed with shards from the new configuration."""
    _patch_producer(monkeypatch, num_samples=5, batch_size=2)
    assert pe.main(_producer_argv(tmp_path)) == 0
    manifest_before = read_manifest(str(tmp_path))

    # Same data, different seq_length: shard contents would no longer line up.
    _patch_producer(monkeypatch, num_samples=5, batch_size=2, seq_len=8)
    argv = [a if a != "6" else "8" for a in _producer_argv(tmp_path, extra=["--resume"])]
    with pytest.raises(ValueError, match="mismatched fields.*seq_length"):
        pe.main(argv)

    # The recorded manifest must be untouched by the refused run.
    assert read_manifest(str(tmp_path)) == manifest_before


def test_producer_main_resume_rejects_shards_without_manifest(monkeypatch, tmp_path):
    """--resume against shards whose manifest is gone cannot verify compatibility."""
    _patch_producer(monkeypatch, num_samples=5, batch_size=2)
    assert pe.main(_producer_argv(tmp_path)) == 0
    os.remove(os.path.join(str(tmp_path), "manifest.json"))

    _patch_producer(monkeypatch, num_samples=5, batch_size=2)
    with pytest.raises(ValueError, match="manifest is missing"):
        pe.main(_producer_argv(tmp_path, extra=["--resume"]))


def test_producer_writes_manifest_before_shards(monkeypatch, tmp_path):
    """The manifest must exist before the first shard is written, so a crashed
    run can be resumed with the compatibility check above."""
    _patch_producer(monkeypatch, num_samples=5, batch_size=2)

    real_write_shard = pe.write_shard
    seen = []

    def _spy_write_shard(cache_dir, shard_index, samples):
        seen.append(os.path.exists(os.path.join(cache_dir, "manifest.json")))
        return real_write_shard(cache_dir, shard_index, samples)

    monkeypatch.setattr(pe, "write_shard", _spy_write_shard)
    assert pe.main(_producer_argv(tmp_path)) == 0
    assert seen and all(seen)


# ---------------------------------------------------------------------------
# Error / edge branches in eagle3_cache and the trainer module.
# ---------------------------------------------------------------------------


def test_producer_main_exact_multiple_no_trailing_shard(monkeypatch, tmp_path):
    # 4 samples, shard_size 2 -> two full shards and a no-op trailing flush
    # (buffered == 0), so no extra partial shard is written.
    _patch_producer(monkeypatch, num_samples=4, batch_size=2)
    assert pe.main(_producer_argv(tmp_path)) == 0
    assert sorted(existing_shard_indices(str(tmp_path))) == [0, 1]
    assert read_manifest(str(tmp_path))["num_samples"] == 4


def test_existing_shard_indices_returns_empty_for_missing_dir(tmp_path):
    assert existing_shard_indices(str(tmp_path / "does-not-exist")) == set()


def test_read_manifest_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        read_manifest(str(tmp_path))


def test_read_manifest_wrong_format_version_raises(tmp_path):
    with open(manifest_path(str(tmp_path)), "w") as f:
        json.dump({"format_version": 999, "num_samples": 1, "shard_size": 1}, f)
    with pytest.raises(ValueError, match="format_version"):
        read_manifest(str(tmp_path))


def test_write_shard_missing_fields_raises(tmp_path):
    # A valid target_probs representation but missing the common fields.
    with pytest.raises(ValueError, match="missing required cache fields"):
        write_shard(
            str(tmp_path),
            0,
            {
                "input_ids": torch.zeros(1, 1, dtype=torch.long),
                "target_probs": torch.rand(1, 1, _DRAFT_VOCAB),
            },
        )


def test_load_safetensors_missing_raises(monkeypatch):
    monkeypatch.setattr(ec, "safe_import_from", lambda *a, **k: (False, None))
    with pytest.raises(ImportError, match="safetensors"):
        ec._load_safetensors()


def test_trainer_module_rejects_nonpositive_ttt_steps():
    with pytest.raises(ValueError, match="ttt_steps"):
        _build_module(ttt_steps=0)

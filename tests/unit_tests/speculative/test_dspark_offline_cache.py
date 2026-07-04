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

"""Unit tests for the DSpark offline target-supervision cache."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config

from nemo_automodel.components.datasets.llm.dspark_cache import (
    CACHE_KEYS,
    CachedDSparkDataset,
    build_cached_dspark_dataloader,
    existing_shard_indices,
    read_manifest,
    read_target_weight_modules,
    write_manifest,
    write_shard,
    write_target_weights,
)
from nemo_automodel.components.speculative.precompute_dspark import (
    _build_parser,
    _compute_batch_cache,
    _ensure_resume_compatible,
    _run,
    _validate_args,
)

_VOCAB = 64
_HIDDEN = 16
_LAYERS = [1, 3, 5]


def _fake_target_batch(batch_size: int = 2, seq_len: int = 5):
    torch.manual_seed(0)
    return SimpleNamespace(
        input_ids=torch.randint(0, _VOCAB, (batch_size, seq_len)),
        loss_mask=torch.ones(batch_size, seq_len, dtype=torch.long),
        target_hidden_states=torch.randn(batch_size, seq_len, _HIDDEN * len(_LAYERS)),
        target_last_hidden_states=torch.randn(batch_size, seq_len, _HIDDEN),
    )


def _write_tiny_cache(cache_dir: str, num_samples: int = 3, shard_size: int = 2, seq_len: int = 4):
    samples = {
        "input_ids": torch.randint(0, _VOCAB, (num_samples, seq_len), dtype=torch.long),
        "attention_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        "loss_mask": torch.ones(num_samples, seq_len, dtype=torch.long),
        "target_hidden_states": torch.randn(num_samples, seq_len, _HIDDEN * len(_LAYERS)),
        "target_last_hidden_states": torch.randn(num_samples, seq_len, _HIDDEN),
    }
    for shard_index, start in enumerate(range(0, num_samples, shard_size)):
        write_shard(cache_dir, shard_index, {k: v[start : start + shard_size] for k, v in samples.items()})
    write_manifest(
        cache_dir,
        {
            "target_model": "tiny",
            "target_model_type": "qwen3",
            "target_vocab_size": _VOCAB,
            "hidden_size": _HIDDEN,
            "num_hidden_layers": 6,
            "seq_length": seq_len,
            "dtype": "fp32",
            "num_samples": num_samples,
            "shard_size": shard_size,
            "target_hidden_dim": _HIDDEN * len(_LAYERS),
            "target_last_hidden_dim": _HIDDEN,
            "target_layer_ids": list(_LAYERS),
        },
    )
    return samples


def test_compute_batch_cache_downcasts_only_float_tensors():
    tb = _fake_target_batch()
    attention_mask = torch.ones_like(tb.input_ids)
    cache = _compute_batch_cache(tb, attention_mask, cache_dtype=torch.bfloat16)

    assert set(cache) == set(CACHE_KEYS)
    assert cache["input_ids"].dtype == torch.long
    assert cache["attention_mask"].dtype == torch.long
    assert cache["loss_mask"].dtype == torch.long
    assert cache["target_hidden_states"].dtype == torch.bfloat16
    assert cache["target_last_hidden_states"].dtype == torch.bfloat16
    torch.testing.assert_close(cache["input_ids"], tb.input_ids)


def test_cache_dataset_round_trip(tmp_path):
    cache_dir = str(tmp_path / "cache")
    samples = _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4)

    dataset = CachedDSparkDataset(cache_dir)
    assert len(dataset) == 3
    item = dataset[2]
    assert set(item) == set(CACHE_KEYS)
    torch.testing.assert_close(item["target_hidden_states"], samples["target_hidden_states"][2])
    torch.testing.assert_close(item["target_last_hidden_states"], samples["target_last_hidden_states"][2])
    assert int(item["input_ids"][0]) == int(samples["input_ids"][2][0])


def test_cache_dataloader_batches(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _write_tiny_cache(cache_dir, num_samples=4, shard_size=2, seq_len=4)
    loader = build_cached_dspark_dataloader(cache_dir=cache_dir, batch_size=2, shuffle=False)

    batches = list(loader)
    assert len(batches) == 2
    assert batches[0]["target_hidden_states"].shape == (2, 4, _HIDDEN * len(_LAYERS))
    assert batches[0]["target_last_hidden_states"].shape == (2, 4, _HIDDEN)


def test_cache_dataset_rejects_missing_shards(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _write_tiny_cache(cache_dir, num_samples=4, shard_size=2, seq_len=4)
    manifest = read_manifest(cache_dir)
    manifest.pop("format_version", None)
    manifest["num_samples"] = 99
    write_manifest(cache_dir, manifest)

    with pytest.raises(ValueError, match="shard indices"):
        CachedDSparkDataset(cache_dir)


def test_cache_dataset_rejects_non_contiguous_shard_indices(tmp_path):
    cache_dir = str(tmp_path / "cache")
    samples = _write_tiny_cache(cache_dir, num_samples=2, shard_size=1, seq_len=4)
    write_shard(cache_dir, 99, {k: v[1:2] for k, v in samples.items()})
    (tmp_path / "cache" / "shard-000001.safetensors").unlink()

    with pytest.raises(ValueError, match="shard indices"):
        CachedDSparkDataset(cache_dir)


def test_write_shard_rejects_missing_cache_field(tmp_path):
    cache_dir = str(tmp_path / "cache")
    samples = _write_tiny_cache(cache_dir, num_samples=1, shard_size=1, seq_len=4)
    samples.pop("target_last_hidden_states")

    with pytest.raises(ValueError, match="missing required"):
        write_shard(cache_dir, 1, samples)


def test_target_weights_round_trip(tmp_path):
    cache_dir = str(tmp_path / "cache")
    embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
    head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
    write_target_weights(cache_dir, embed, head)

    loaded_embed, loaded_head = read_target_weight_modules(cache_dir)
    torch.testing.assert_close(loaded_embed.weight, embed.weight.detach().float())
    torch.testing.assert_close(loaded_head.weight, head.weight.detach().float())


def test_existing_shard_indices(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _write_tiny_cache(cache_dir, num_samples=3, shard_size=2, seq_len=4)
    assert existing_shard_indices(cache_dir) == {0, 1}


def _args(**overrides):
    base = dict(batch_size=4, shard_size=256, draft_num_hidden_layers=5, dtype="bf16")
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    "overrides,pattern",
    [
        ({"batch_size": 0}, "batch-size"),
        ({"shard_size": 0}, "shard-size"),
        ({"shard_size": 7}, "multiple"),
        ({"draft_num_hidden_layers": 0}, "draft-num-hidden-layers"),
        ({"dtype": "bad"}, "dtype"),
    ],
)
def test_validate_args_rejects_invalid_values(overrides, pattern):
    with pytest.raises(ValueError, match=pattern):
        _validate_args(_args(**overrides))


def test_parser_accepts_required_args():
    args = _build_parser().parse_args(["--target-model", "tiny", "--input-data", "data.jsonl", "--output-dir", "out"])
    assert args.target_model == "tiny"
    assert args.dtype == "bf16"


def test_resume_compatibility_rejects_manifest_mismatch(tmp_path):
    cache_dir = str(tmp_path / "cache")
    _write_tiny_cache(cache_dir, num_samples=2, shard_size=2, seq_len=4)
    manifest = read_manifest(cache_dir)
    manifest.pop("format_version", None)
    manifest["seq_length"] = 8

    with pytest.raises(ValueError, match="does not match"):
        _ensure_resume_compatible(cache_dir, manifest, existing_shards={0})


def test_precompute_resume_mismatch_does_not_overwrite_target_weights(monkeypatch, tmp_path):
    import nemo_automodel.components.speculative.precompute_dspark as precompute_dspark

    cache_dir = str(tmp_path / "cache")
    old_embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
    old_head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
    write_target_weights(cache_dir, old_embed, old_head)
    samples = _write_tiny_cache(cache_dir, num_samples=1, shard_size=1, seq_len=4)
    write_shard(cache_dir, 0, {k: v[:1] for k, v in samples.items()})

    old_embed_weight = old_embed.weight.detach().float().clone()
    old_head_weight = old_head.weight.detach().float().clone()

    class _FakeTargetModel:
        config = Qwen3Config(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            intermediate_size=2 * _HIDDEN,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )

        def __init__(self):
            self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
            self.head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)
            with torch.no_grad():
                self.embed.weight.fill_(123.0)
                self.head.weight.fill_(456.0)

        def to(self, _device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, _requires_grad):
            return self

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    class _FakeWrapper:
        def __init__(self, _model, target_layer_ids):
            self.target_layer_ids = list(target_layer_ids)

        def generate_batch(self, *_args, **_kwargs):
            raise AssertionError("resume mismatch should fail before target forwards")

    class _FakeLoader:
        dataset = [object()]

        def __iter__(self):
            raise AssertionError("resume mismatch should fail before iterating the dataset")

    monkeypatch.setattr(precompute_dspark, "_read_target_model_type", lambda *_args, **_kwargs: "qwen3")
    monkeypatch.setattr(
        precompute_dspark.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: _FakeTargetModel.config
    )
    monkeypatch.setattr(
        precompute_dspark.NeMoAutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            chat_template="{{ messages }}", apply_chat_template=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(
        precompute_dspark.NeMoAutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTargetModel(),
    )
    monkeypatch.setattr(precompute_dspark, "HFDSparkTargetModel", _FakeWrapper)
    monkeypatch.setattr(precompute_dspark, "build_eagle3_dataloader", lambda **_kwargs: _FakeLoader())

    args = SimpleNamespace(
        target_model="tiny-qwen3",
        input_data="data.jsonl",
        output_dir=cache_dir,
        seq_length=8,
        batch_size=1,
        shard_size=1,
        dtype="fp32",
        device="cpu",
        num_workers=0,
        split=None,
        shuffle_seed=42,
        draft_num_hidden_layers=5,
        target_layer_ids=list(_LAYERS),
        chat_template=None,
        mask_reasoning_content=False,
        target_attn_implementation=None,
        target_force_hf=False,
        trust_remote_code=False,
        resume=True,
    )

    with pytest.raises(ValueError, match="does not match"):
        _run(args)

    loaded_embed, loaded_head = read_target_weight_modules(cache_dir)
    torch.testing.assert_close(loaded_embed.weight, old_embed_weight)
    torch.testing.assert_close(loaded_head.weight, old_head_weight)


def test_precompute_run_writes_cache(monkeypatch, tmp_path):
    import nemo_automodel.components.speculative.precompute_dspark as precompute_dspark

    class _FakeTargetModel:
        config = Qwen3Config(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            intermediate_size=2 * _HIDDEN,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )

        def __init__(self):
            self.embed = torch.nn.Embedding(_VOCAB, _HIDDEN)
            self.head = torch.nn.Linear(_HIDDEN, _VOCAB, bias=False)

        def to(self, _device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, _requires_grad):
            return self

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.head

    class _FakeWrapper:
        target_layer_ids = list(_LAYERS)

        def __init__(self, _model, target_layer_ids):
            self.target_layer_ids = list(target_layer_ids)

        def generate_batch(self, input_ids, attention_mask, loss_mask):
            batch_size, seq_len = input_ids.shape
            return SimpleNamespace(
                input_ids=input_ids,
                loss_mask=loss_mask,
                target_hidden_states=torch.ones(batch_size, seq_len, _HIDDEN * len(self.target_layer_ids)),
                target_last_hidden_states=torch.ones(batch_size, seq_len, _HIDDEN),
            )

    class _FakeLoader:
        dataset = [object(), object(), object()]

        def __iter__(self):
            yield {
                "input_ids": torch.arange(8, dtype=torch.long).view(2, 4),
                "attention_mask": torch.ones(2, 4, dtype=torch.long),
                "loss_mask": torch.ones(2, 4, dtype=torch.long),
            }
            yield {
                "input_ids": torch.arange(4, dtype=torch.long).view(1, 4),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "loss_mask": torch.ones(1, 4, dtype=torch.long),
            }

    fake_model = _FakeTargetModel()
    monkeypatch.setattr(precompute_dspark, "_read_target_model_type", lambda *_args, **_kwargs: "qwen3")
    monkeypatch.setattr(precompute_dspark.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: fake_model.config)
    monkeypatch.setattr(
        precompute_dspark.NeMoAutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            chat_template="{{ messages }}", apply_chat_template=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(
        precompute_dspark.NeMoAutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: fake_model,
    )
    monkeypatch.setattr(precompute_dspark, "HFDSparkTargetModel", _FakeWrapper)
    monkeypatch.setattr(precompute_dspark, "build_eagle3_dataloader", lambda **_kwargs: _FakeLoader())

    cache_dir = str(tmp_path / "cache")
    args = SimpleNamespace(
        target_model="tiny-qwen3",
        input_data="data.jsonl",
        output_dir=cache_dir,
        seq_length=4,
        batch_size=2,
        shard_size=2,
        dtype="fp32",
        device="cpu",
        num_workers=0,
        split=None,
        shuffle_seed=42,
        draft_num_hidden_layers=5,
        target_layer_ids=list(_LAYERS),
        chat_template=None,
        mask_reasoning_content=False,
        target_attn_implementation=None,
        target_force_hf=False,
        trust_remote_code=False,
        resume=False,
    )

    assert _run(args) == 0

    manifest = read_manifest(cache_dir)
    assert manifest["num_samples"] == 3
    assert manifest["target_layer_ids"] == _LAYERS
    dataset = CachedDSparkDataset(cache_dir)
    assert len(dataset) == 3
    assert dataset[0]["target_hidden_states"].shape == (4, _HIDDEN * len(_LAYERS))
    loaded_embed, loaded_head = read_target_weight_modules(cache_dir)
    torch.testing.assert_close(loaded_embed.weight, fake_model.embed.weight.detach().float())
    torch.testing.assert_close(loaded_head.weight, fake_model.head.weight.detach().float())

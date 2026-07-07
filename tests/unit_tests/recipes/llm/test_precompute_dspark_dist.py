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

"""CPU unit tests for the distributed DSpark precompute entry point.

Covers the config-driven glue in ``precompute_dspark_dist``: cache-setting
validation, the manifest schema, existing-directory compatibility guarding, target
dispatch (large sharded targets vs replicated text targets vs the multimodal
rejection), the all-reduce reducer gating, and a world_size=1 end-to-end ``run`` that
writes a cache the DSpark reader loads back -- all without CUDA or a process group.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import Qwen3Config

import nemo_automodel.recipes.llm.precompute_dspark_dist as pdd
from nemo_automodel.components.datasets.llm.dspark_cache import (
    CachedDSparkDataset,
    read_manifest,
    read_target_weight_modules,
)

_VOCAB = 16
_HIDDEN = 8
_SEQ = 4
_LAYERS = [1, 3]


class _Cfg(dict):
    """A minimal ConfigNode stand-in: dict with attribute access and ``get``."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _recipe_cfg(**overrides):
    base = {
        "target_model_name_or_path": "tiny-qwen3",
        "train_data_path": "data.jsonl",
        "seq_length": _SEQ,
        "micro_batch_size": 2,
        "cache_output_dir": None,
        "cache_dtype": "fp32",
        "cache_shard_size": 2,
        "target_layer_ids": list(_LAYERS),
        "draft_num_hidden_layers": 5,
        "num_workers": 0,
        "trust_remote_code": False,
    }
    base.update(overrides)
    return _Cfg(base)


# ---------------------------------------------------------------------------
# _resolve_cache_settings
# ---------------------------------------------------------------------------


def test_resolve_cache_settings_happy_path():
    output_dir, dtype, shard_size = pdd._resolve_cache_settings(_recipe_cfg(cache_output_dir="/tmp/c"))
    assert (output_dir, dtype, shard_size) == ("/tmp/c", "fp32", 2)


def test_resolve_cache_settings_requires_output_dir():
    with pytest.raises(ValueError, match="cache_output_dir is required"):
        pdd._resolve_cache_settings(_recipe_cfg())


def test_resolve_cache_settings_rejects_bad_dtype():
    with pytest.raises(ValueError, match="cache_dtype"):
        pdd._resolve_cache_settings(_recipe_cfg(cache_output_dir="/tmp/c", cache_dtype="int8"))


def test_resolve_cache_settings_rejects_unaligned_shard_size():
    with pytest.raises(ValueError, match="multiple of micro_batch_size"):
        pdd._resolve_cache_settings(_recipe_cfg(cache_output_dir="/tmp/c", cache_shard_size=3, micro_batch_size=2))


def test_resolve_cache_settings_rejects_nonpositive_shard_size():
    with pytest.raises(ValueError, match="cache_shard_size must be >= 1"):
        pdd._resolve_cache_settings(_recipe_cfg(cache_output_dir="/tmp/c", cache_shard_size=0, micro_batch_size=1))


# ---------------------------------------------------------------------------
# build_cache_manifest / _ensure_output_dir_compatible
# ---------------------------------------------------------------------------


def _config_stub():
    return SimpleNamespace(hidden_size=_HIDDEN, vocab_size=_VOCAB, num_hidden_layers=6)


def test_build_manifest_schema():
    manifest = pdd.build_cache_manifest(
        target_model="tiny",
        target_model_type="qwen3",
        target_text_config=_config_stub(),
        seq_length=_SEQ,
        dtype="fp32",
        num_samples=10,
        shard_size=2,
        target_layer_ids=_LAYERS,
    )
    assert manifest["target_hidden_dim"] == _HIDDEN * len(_LAYERS)
    assert manifest["target_last_hidden_dim"] == _HIDDEN
    assert manifest["num_hidden_layers"] == 6
    assert manifest["target_layer_ids"] == _LAYERS
    assert manifest["num_samples"] == 10


def test_ensure_output_dir_compatible_absent_manifest_is_noop(tmp_path):
    pdd._ensure_output_dir_compatible(str(tmp_path), {"num_samples": 1})  # no manifest -> ok


def test_ensure_output_dir_compatible_matching_manifest_ok(tmp_path):
    from nemo_automodel.components.datasets.llm.dspark_cache import write_manifest

    manifest = pdd.build_cache_manifest(
        target_model="tiny",
        target_model_type="qwen3",
        target_text_config=_config_stub(),
        seq_length=_SEQ,
        dtype="fp32",
        num_samples=4,
        shard_size=2,
        target_layer_ids=_LAYERS,
    )
    write_manifest(str(tmp_path), manifest)
    pdd._ensure_output_dir_compatible(str(tmp_path), manifest)  # identical -> ok


def test_ensure_output_dir_compatible_mismatch_raises(tmp_path):
    from nemo_automodel.components.datasets.llm.dspark_cache import write_manifest

    base_kwargs = dict(
        target_model="tiny",
        target_model_type="qwen3",
        target_text_config=_config_stub(),
        seq_length=_SEQ,
        dtype="fp32",
        shard_size=2,
        target_layer_ids=_LAYERS,
    )
    write_manifest(str(tmp_path), pdd.build_cache_manifest(num_samples=4, **base_kwargs))
    with pytest.raises(ValueError, match="different configuration"):
        pdd._ensure_output_dir_compatible(str(tmp_path), pdd.build_cache_manifest(num_samples=8, **base_kwargs))


# ---------------------------------------------------------------------------
# _build_target dispatch
# ---------------------------------------------------------------------------


def test_build_target_rejects_minimax():
    with pytest.raises(ValueError, match="multimodal"):
        pdd._build_target(
            cfg=_Cfg(),
            recipe_cfg=_recipe_cfg(),
            world_size=8,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            model_type=list(pdd._MINIMAX_M3_MODEL_TYPES)[0],
            target_path="mm",
            trust_remote_code=False,
        )


@pytest.mark.parametrize(
    "model_type,builder_attr",
    [(pdd._DEEPSEEK_V4_MODEL_TYPE, "build_deepseek_v4_target"), (pdd._GLM_5_2_MODEL_TYPE, "build_glm_5_2_target")],
)
def test_build_target_dispatches_sharded_builders(monkeypatch, model_type, builder_attr):
    captured = {}

    def _fake_builder(**kwargs):
        captured.update(kwargs)
        return "target-config", "target-model", "distributed-setup"

    monkeypatch.setattr(pdd, builder_attr, _fake_builder)
    config, model = pdd._build_target(
        cfg=_Cfg({"x": 1}),
        recipe_cfg=_recipe_cfg(),
        world_size=16,
        device=torch.device("cuda"),
        compute_dtype=torch.bfloat16,
        model_type=model_type,
        target_path="big",
        trust_remote_code=True,
    )
    assert (config, model) == ("target-config", "target-model")
    assert captured["world_size"] == 16
    assert captured["target_path"] == "big"
    assert captured["trust_remote_code"] is True


def test_build_target_generic_replicated_path(monkeypatch):
    fake_model = SimpleNamespace(config=_config_stub(), to=lambda _d: fake_model)
    captured = {}

    def _fake_from_pretrained(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return fake_model

    monkeypatch.setattr(pdd.NeMoAutoModelForCausalLM, "from_pretrained", _fake_from_pretrained)
    config, model = pdd._build_target(
        cfg=_Cfg(),
        recipe_cfg=_recipe_cfg(target_attn_implementation="eager"),
        world_size=4,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        model_type="qwen3",
        target_path="tiny-qwen3",
        trust_remote_code=False,
    )
    assert model is fake_model
    assert config is fake_model.config
    assert captured["path"] == "tiny-qwen3"
    assert captured["torch_dtype"] == torch.float32
    assert captured["attn_implementation"] == "eager"


# ---------------------------------------------------------------------------
# _make_sync_max_steps
# ---------------------------------------------------------------------------


def test_make_sync_max_steps_identity_without_process_group():
    assert pdd._make_sync_max_steps(1, torch.device("cpu")) is None
    # world_size > 1 but no initialized process group also falls back to identity.
    assert pdd._make_sync_max_steps(8, torch.device("cpu")) is None


def test_make_sync_max_steps_all_reduce_over_single_process_group():
    """With a live (single-process gloo) group, the reducer all-reduces MAX."""
    import torch.distributed as dist

    store = dist.HashStore()
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        sync = pdd._make_sync_max_steps(2, torch.device("cpu"))
        assert sync is not None
        assert sync(5) == 5  # MAX over the single rank is the rank's own value
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# run() end-to-end (world_size == 1, no process group)
# ---------------------------------------------------------------------------


class _TokenDataset(Dataset):
    def __init__(self, num_samples):
        self._n = num_samples

    def __len__(self):
        return self._n

    def __getitem__(self, index):
        return {
            "input_ids": torch.full((_SEQ,), index, dtype=torch.long),
            "attention_mask": torch.ones(_SEQ, dtype=torch.long),
            "loss_mask": torch.ones(_SEQ, dtype=torch.long),
        }


class _FakeTargetModel:
    def __init__(self):
        self.config = Qwen3Config(
            vocab_size=_VOCAB,
            hidden_size=_HIDDEN,
            intermediate_size=2 * _HIDDEN,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
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
    def __init__(self, _model, target_layer_ids):
        self.target_layer_ids = list(target_layer_ids)

    def generate_batch(self, input_ids, attention_mask, loss_mask):
        bsz, seq_len = input_ids.shape
        return SimpleNamespace(
            input_ids=input_ids,
            loss_mask=loss_mask,
            target_hidden_states=torch.ones(bsz, seq_len, _HIDDEN * len(self.target_layer_ids)),
            target_last_hidden_states=torch.ones(bsz, seq_len, _HIDDEN),
        )


def test_run_writes_cache_single_process(monkeypatch, tmp_path):
    fake_model = _FakeTargetModel()
    monkeypatch.setattr(
        pdd,
        "initialize_distributed",
        lambda **_kw: SimpleNamespace(device=torch.device("cpu"), world_size=1, rank=0, is_main=True),
    )
    monkeypatch.setattr(pdd, "_read_target_model_type", lambda *_a, **_k: "qwen3")
    monkeypatch.setattr(pdd.NeMoAutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: fake_model)
    monkeypatch.setattr(
        pdd.NeMoAutoTokenizer,
        "from_pretrained",
        lambda *_a, **_k: SimpleNamespace(chat_template="{{ messages }}", apply_chat_template=lambda *a, **k: None),
    )
    monkeypatch.setattr(pdd, "HFDSparkTargetModel", _FakeWrapper)
    monkeypatch.setattr(
        pdd,
        "build_eagle3_dataloader",
        lambda **_kw: DataLoader(
            _TokenDataset(3),
            batch_size=2,
            shuffle=False,
            collate_fn=lambda fs: {k: torch.stack([f[k] for f in fs], 0) for k in fs[0]},
            drop_last=False,
        ),
    )

    cache_dir = str(tmp_path / "cache")
    cfg = _Cfg(
        {
            "dist_env": {"backend": "gloo"},
            "recipe_args": _recipe_cfg(cache_output_dir=cache_dir),
        }
    )
    assert pdd.run(cfg) == 0

    manifest = read_manifest(cache_dir)
    assert manifest["num_samples"] == 3
    assert manifest["target_layer_ids"] == _LAYERS
    dataset = CachedDSparkDataset(cache_dir)
    assert len(dataset) == 3
    assert dataset[0]["target_hidden_states"].shape == (_SEQ, _HIDDEN * len(_LAYERS))
    for i in range(3):
        assert torch.equal(dataset[i]["input_ids"], torch.full((_SEQ,), i, dtype=torch.long))
    loaded_embed, loaded_head = read_target_weight_modules(cache_dir)
    torch.testing.assert_close(loaded_embed.weight, fake_model.embed.weight.detach().float())
    torch.testing.assert_close(loaded_head.weight, fake_model.head.weight.detach().float())

    # Re-running the same config is idempotent: it overwrites the present shards and
    # still yields a valid cache (exercises the existing-shards overwrite path).
    assert pdd.run(cfg) == 0
    assert len(CachedDSparkDataset(cache_dir)) == 3


def test_main_parses_config_and_runs(monkeypatch):
    sentinel_cfg = object()
    monkeypatch.setattr(pdd, "parse_args_and_load_config", lambda config_path: sentinel_cfg)
    seen = {}

    def _fake_run(cfg):
        seen["cfg"] = cfg
        return 0

    monkeypatch.setattr(pdd, "run", _fake_run)
    assert pdd.main("some.yaml") == 0
    assert seen["cfg"] is sentinel_cfg

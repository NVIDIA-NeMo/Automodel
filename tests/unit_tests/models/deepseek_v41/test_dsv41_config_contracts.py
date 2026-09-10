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

"""Pinned tokenizer construction and independent integer-buffer restoration checks."""

import copy
import json

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41VisionConfig
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41EngramHasher, EngramLayout
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from tests.unit_tests.models.deepseek_v41.conftest import tiny_backend, tiny_config
from tests.unit_tests.models.deepseek_v41.test_dsv41_config import _RELEASED_STYLE_CONFIG


def _tokenizer():
    vocab = {"[UNK]": 0, "A": 1, "a": 2, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")), unk_token="[UNK]"
    )


def _hash_config(**kwargs):
    values = dict(
        vocab_size=8,
        num_hidden_layers=2,
        compress_ratios=[0, 0],
        kv_source_layer_ids=[],
        index_source_layer_ids=[],
        candidate_source_layer_id=-1,
        candidate_topk_blocks=0,
        candidate_block_size=0,
        engram_num_embeddings=[64],
        engram_vocab_size=5,
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_compressed_vocab_size=7,
    )
    values.update(kwargs)
    return tiny_config(**values)


def test_flat_text_and_typed_vision_roundtrip_retains_unknown_metadata(tmp_path):
    payload = copy.deepcopy(_RELEASED_STYLE_CONFIG)
    payload["text_config"]["checkpoint_note"] = "retained"
    payload["text_config"]["num_hidden_layers"] = 4
    config = DeepseekV41Config.from_dict(payload)
    config.validate_layer_layout()
    config.save_pretrained(tmp_path)
    restored = DeepseekV41Config.from_pretrained(tmp_path)
    assert restored.hidden_size == 5120 and restored.num_hidden_layers == 4
    assert restored.checkpoint_note == "retained"
    assert restored.kv_source_layer_ids == [2, 8, 14, 20]
    assert restored.engram_layer_ids == [1, 14]
    assert restored.compress_ratios == payload["text_config"]["compress_ratios"]
    assert isinstance(restored.vision_config, DeepseekV41VisionConfig)
    assert restored.vision_config.num_hidden_layers == 32
    assert "quantization_config" not in restored.to_dict()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 0),
        ("num_attention_heads", 3),
        ("qk_rope_head_dim", 3),
        ("num_key_value_heads", 2),
        ("n_routed_experts", 1),
        ("n_shared_experts", 2),
        ("rms_norm_eps", float("nan")),
        ("attention_dropout", 1),
        ("engram_max_ngram_size", 1),
        ("engram_pad_token_id", -1),
        ("rope_scaling", {"factor": 0}),
        ("initializer_range", -1),
    ],
)
def test_invalid_dimensions_are_rejected(field, value):
    with pytest.raises((ValueError, TypeError)):
        tiny_config(**{field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        dict(compress_ratios=[0, 0, -1]),
        dict(kv_source_layer_ids=[2, 2, 4]),
        dict(index_source_layer_ids=[2, 4, 99]),
        dict(engram_layer_ids=[-1]),
        dict(engram_num_embeddings=[0]),
        dict(candidate_topk_blocks=0),
    ],
)
def test_invalid_schedule_is_rejected_before_allocation(overrides):
    config = tiny_config(**overrides)
    with pytest.raises(ValueError):
        config.validate_layer_layout()


def test_implicit_swa_and_identity_defaults_remain_available():
    config = DeepseekV41Config()
    config.validate_layer_layout()
    assert all(config.compress_ratio(i) == 0 for i in range(config.num_hidden_layers))
    assert config.vision_config.num_hidden_layers == 0
    hasher = DeepseekV41EngramHasher(
        _hash_config(engram_compressed_vocab_size=0), EngramLayout.from_config(_hash_config())
    )
    assert hasher.token_map.tolist() == list(range(8))


def test_pinned_tokenizer_is_built_before_forward_and_not_serialized(monkeypatch):
    tokenizer = _tokenizer()
    calls = []

    def load(source, **kwargs):
        calls.append((source, kwargs))
        return tokenizer

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    config = _hash_config(name_or_path="local/pinned", _commit_hash="revision-sha")
    model = DeepseekV41ForCausalLM(config, backend=tiny_backend(experts="torch"))
    assert calls == [("local/pinned", dict(revision="revision-sha", trust_remote_code=False, use_fast=True))]
    assert model.model.engram_hasher.token_map.tolist() == [0, 1, 1, 2, 3, 4, 5, 6]
    json.dumps(config.to_dict())
    assert "tokenizer" not in config.__dict__

    def forbidden(*args, **kwargs):
        raise AssertionError("tokenizer I/O entered numerical forward")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    assert torch.isfinite(model(torch.tensor([[1, 3, 4]])).logits).all()


def test_explicit_tokenizer_and_identity_map_need_no_source(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected tokenizer I/O")

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", forbidden)
    config = _hash_config()
    with pytest.raises(ValueError, match="checkpoint source"):
        config.build_tokenizer()
    model = DeepseekV41ForCausalLM(config, backend=tiny_backend(experts="torch"))
    with pytest.raises(RuntimeError, match="before forward"):
        model(torch.tensor([[1, 3]]))
    model.set_engram_tokenizer(_tokenizer())
    assert model.model.engram_hasher.token_map.tolist() == [0, 1, 1, 2, 3, 4, 5, 6]
    injected = DeepseekV41ForCausalLM(config, backend=tiny_backend(experts="torch"), tokenizer=_tokenizer())
    assert injected.model.engram_hasher.has_token_map
    identity = DeepseekV41ForCausalLM(
        _hash_config(engram_compressed_vocab_size=8), backend=tiny_backend(experts="torch")
    )
    assert identity.model.engram_hasher.token_map.tolist() == list(range(8))


def test_slow_tokenizer_fails_at_explicit_and_automatic_boundaries(monkeypatch):
    config = _hash_config(name_or_path="local/pinned")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    with pytest.raises(TypeError, match="fast tokenizer"):
        config.build_tokenizer()
    hasher = DeepseekV41EngramHasher(config, EngramLayout.from_config(config))
    with pytest.raises(TypeError, match="fast tokenizer"):
        hasher.set_tokenizer(object())


@pytest.mark.parametrize("identity", [False, True])
def test_meta_hasher_restores_independent_primes_multipliers_map_and_hashes(identity):
    config = _hash_config(engram_compressed_vocab_size=8 if identity else 7)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(
            config, backend=tiny_backend(experts="torch"), tokenizer=None if identity else _tokenizer()
        )
    model.to_empty(device="cpu")
    hasher = model.model.engram_hasher
    for buffer in hasher.buffers():
        buffer.fill_(-99)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    mapping = list(range(8)) if identity else [0, 1, 1, 2, 3, 4, 5, 6]
    compressed = 8 if identity else 7
    generator = np.random.default_rng(10007)
    expected_multipliers = (
        generator.integers(0, (np.iinfo(np.int64).max // compressed) // 2, size=3, dtype=np.int64) * 2 + 1
    ).tolist()
    assert hasher.primes.tolist() == [[[5, 7], [11, 13]]]
    assert hasher.offsets.tolist() == [[0, 5, 12, 23]]
    assert hasher.multipliers.tolist() == [expected_multipliers]
    assert hasher.token_map.tolist() == mapping
    ids = [3, 4, 5, 6]
    positions = [0, 1, 0, 1]
    mask = [True, True, True, False]
    expected = []
    for i in range(4):
        blocked = False
        product = []
        for shift in range(3):
            blocked = blocked or positions[i] < shift or not mask[i - shift]
            token = mapping[2] if blocked else mapping[ids[i - shift]]
            product.append(token * expected_multipliers[shift])
        rolling = product[0] ^ product[1]
        expected.append(
            [rolling % 5, rolling % 7 + 5, (rolling ^ product[2]) % 11 + 12, (rolling ^ product[2]) % 13 + 23]
        )
    actual = hasher(torch.tensor([ids]), torch.tensor([positions]), torch.tensor([mask]))
    assert actual.tolist() == [[[row] for row in expected]]
    assert all(
        name not in model.state_dict() for name in ("model.engram_hasher.token_map", "model.engram_hasher.primes")
    )

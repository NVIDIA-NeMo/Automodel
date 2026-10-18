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

"""CPU tests for the ``serve_dspark_target`` entry point and its shared target helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

import nemo_automodel.recipes.llm._dspark_target_build as tdb
import nemo_automodel.recipes.llm.serve_dspark_target as serve_module
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.speculative.dspark.registry import build_target_layer_ids
from nemo_automodel.components.speculative.dspark.remote import DSparkTargetModelServer

VOCAB = 64
HIDDEN = 32
NUM_LAYERS = 6


def _tiny_model() -> torch.nn.Module:
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


def _cfg(recipe_args: dict | None = None, **top_level) -> ConfigNode:
    return ConfigNode(
        {
            "recipe_args": {"target_model_name_or_path": "tiny-qwen3", **(recipe_args or {})},
            **top_level,
        }
    )


def _patch_single_process(monkeypatch, *, world_size: int = 1, model=None) -> dict:
    captured = {}
    monkeypatch.setattr(
        serve_module,
        "initialize_distributed",
        lambda **_kw: SimpleNamespace(device=torch.device("cpu"), world_size=world_size),
    )
    monkeypatch.setattr(serve_module, "read_target_model_type", lambda *_a, **_k: "qwen3")

    def _fake_build(**kwargs):
        captured.update(kwargs)
        return model.config, model

    monkeypatch.setattr(serve_module, "build_dspark_target", _fake_build)
    return captured


# --- build_server -----------------------------------------------------------


def test_build_server_serves_the_configured_capture_layers(monkeypatch):
    model = _tiny_model()
    captured = _patch_single_process(monkeypatch, model=model)

    server = serve_module.build_server(_cfg({"target_layer_ids": [1, 4]}), host="127.0.0.1", nccl_port=9101)

    assert isinstance(server, DSparkTargetModelServer)
    assert json.loads(server.handle_model_info(b"")) == {
        "num_hidden_layers": NUM_LAYERS,
        "hidden_size": HIDDEN,
        "vocab_size": VOCAB,
        "target_layer_ids": [1, 4],
    }
    assert captured["world_size"] == 1
    assert captured["compute_dtype"] == torch.float32
    assert captured["model_type"] == "qwen3"
    assert not any(p.requires_grad for p in model.parameters())


def test_build_server_defaults_capture_layers_from_draft_depth(monkeypatch):
    _patch_single_process(monkeypatch, model=_tiny_model())

    server = serve_module.build_server(_cfg({"draft_num_hidden_layers": 2}), host="127.0.0.1", nccl_port=9101)

    served = json.loads(server.handle_model_info(b""))["target_layer_ids"]
    assert served == build_target_layer_ids(NUM_LAYERS, 2)


def test_build_server_rejects_multi_rank_launch(monkeypatch):
    _patch_single_process(monkeypatch, world_size=2, model=_tiny_model())

    with pytest.raises(NotImplementedError, match="world_size=2"):
        serve_module.build_server(_cfg(), host="127.0.0.1", nccl_port=9101)


# --- main -------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_server,expected",
    [
        (None, ("0.0.0.0", 8001, 8101)),
        ({"host": "10.0.0.5", "port": 9000}, ("10.0.0.5", 9000, 9100)),
    ],
)
def test_main_reads_server_address_from_target_server_block(monkeypatch, target_server, expected):
    cfg = _cfg() if target_server is None else _cfg(target_server=target_server)
    calls = {}
    monkeypatch.setattr(serve_module, "parse_args_and_load_config", lambda _path: cfg)
    monkeypatch.setattr(
        serve_module, "build_server", lambda _cfg, *, host, nccl_port: calls.update(nccl_port=nccl_port) or "logic"
    )
    monkeypatch.setattr(
        serve_module, "serve", lambda logic, host, port: calls.update(logic=logic, address=(host, port))
    )

    serve_module.main("config.yaml")

    assert calls["logic"] == "logic"
    assert (*calls["address"], calls["nccl_port"]) == expected


# --- shared target helpers --------------------------------------------------


def test_resolve_target_layer_ids_prefers_explicit_ids():
    assert tdb.resolve_dspark_target_layer_ids(ConfigNode({"target_layer_ids": [0, 3]}), NUM_LAYERS) == [0, 3]


def test_resolve_target_layer_ids_rejects_out_of_range_ids():
    with pytest.raises(AssertionError, match="out of range"):
        tdb.resolve_dspark_target_layer_ids(ConfigNode({"target_layer_ids": [0, NUM_LAYERS]}), NUM_LAYERS)


def test_build_dspark_target_returns_gemma4_text_config(monkeypatch):
    text_config = SimpleNamespace(num_hidden_layers=NUM_LAYERS)
    model = SimpleNamespace(config=SimpleNamespace(text_config=text_config))
    model.to = lambda _device: model
    monkeypatch.setattr(tdb.NeMoAutoModelForCausalLM, "from_pretrained", lambda *_a, **_k: model)

    config, built = tdb.build_dspark_target(
        cfg=ConfigNode({}),
        recipe_cfg=ConfigNode({}),
        world_size=1,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        model_type="gemma4",
        target_path="tiny-gemma4",
        trust_remote_code=False,
    )

    assert config is text_config
    assert built is model

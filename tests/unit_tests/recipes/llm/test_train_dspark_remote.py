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

"""CPU tests for ``TrainDSparkRecipe`` with ``target_model_backend: remote``.

The recipe talks to a real in-process ``DSparkTargetModelServer`` over HTTP (wire
data plane). The tests pin that setup never loads the target locally, seeds the
draft from the served weights, rejects a server whose target differs from the
config, and feeds the trainer exactly the features a colocated target captures.
"""

from __future__ import annotations

import os
import socket
import threading
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

os.environ["NEMO_EAGLE_ENABLE_NCCL"] = "0"

from http.server import ThreadingHTTPServer  # noqa: E402

import nemo_automodel.recipes.llm.train_dspark as train_dspark_module  # noqa: E402
from nemo_automodel.components.speculative.dspark.remote import (  # noqa: E402
    DSparkTargetModelServer,
    RemoteDSparkTargetModel,
)
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel  # noqa: E402
from nemo_automodel.components.speculative.eagle.remote.server import _make_request_handler  # noqa: E402
from nemo_automodel.recipes.llm.train_dspark import (  # noqa: E402
    TrainDSparkRecipe,
    _validate_remote_target_options,
)

VOCAB = 64
HIDDEN = 32
NUM_LAYERS = 4
LAYER_IDS = [1, 3]


class _CfgNode(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def to_dict(self):
        return dict(self)


def _target_model() -> torch.nn.Module:
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32).eval()
    model.config.architectures = ["Qwen3ForCausalLM"]
    return model


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served_model():
    """A tiny Qwen3 target served by an in-process DSpark server: ``(url, model)``."""
    model = _target_model()
    wrapper = HFDSparkTargetModel(model, target_layer_ids=LAYER_IDS)
    port = _free_port()
    logic = DSparkTargetModelServer(wrapper, model.config, nccl_port=port + 100, host="127.0.0.1")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _make_request_handler(logic))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{port}", model
    finally:
        httpd.shutdown()
        httpd.server_close()


def _batches() -> list[dict[str, torch.Tensor]]:
    torch.manual_seed(2)
    return [
        {
            "input_ids": torch.randint(0, VOCAB, (1, 8)),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
            "loss_mask": torch.ones(1, 8, dtype=torch.long),
        }
    ]


def _remote_cfg(tmp_path, **recipe_overrides) -> _CfgNode:
    recipe_args = _CfgNode(
        target_model_name_or_path="tiny-qwen3",
        target_model_backend="remote",
        remote_urls=["http://127.0.0.1:9"],
        remote_timeout=10,
        remote_max_retries=0,
        train_data_path="unused.jsonl",
        seq_length=8,
        micro_batch_size=1,
        mask_token_id=7,
        num_epochs=1,
        output_dir=str(tmp_path / "out"),
        target_layer_ids=LAYER_IDS,
        draft_num_hidden_layers=1,
        num_anchors=4,
        block_size=2,
        markov_rank=8,
        attention_backend="flex_attention",
        trust_remote_code=False,
    )
    recipe_args.update(recipe_overrides)
    return _CfgNode(
        recipe_args=recipe_args,
        optimizer=_CfgNode(lr=1e-4, warmup_ratio=0.0, min_lr_ratio=0.1),
        checkpoint=_CfgNode(enabled=False),
        raw_config={},
    )


def _patch_recipe_setup(monkeypatch, target_config) -> None:
    """Stub the distributed/IO edges of ``setup`` and forbid any local target load."""
    monkeypatch.setattr(
        train_dspark_module,
        "initialize_distributed",
        lambda **_kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=1, is_main=True),
    )
    monkeypatch.setattr(train_dspark_module, "setup_logging", lambda: None)
    monkeypatch.setattr(train_dspark_module, "_read_target_model_type", lambda *_args, **_kwargs: "qwen3")
    monkeypatch.setattr(train_dspark_module.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: target_config)
    monkeypatch.setattr(
        train_dspark_module.NeMoAutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            chat_template="{{ messages }}", apply_chat_template=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(
        train_dspark_module.NeMoAutoModelForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("the remote backend must not load the target model"),
    )
    monkeypatch.setattr(train_dspark_module, "build_eagle3_dataloader", lambda **_kwargs: _batches())
    monkeypatch.setattr(TrainDSparkRecipe, "_build_checkpointer", lambda self, _target_path: None)
    monkeypatch.setattr(TrainDSparkRecipe, "load_checkpoint", lambda self, restore_from=None: None)


# --- option gates -----------------------------------------------------------


_VALID_OPTIONS = dict(cached_target_path=None, is_multimodal=False, packed_sequence_size=0, cp_size=1)


@pytest.mark.parametrize(
    "override,error,pattern",
    [
        (dict(cached_target_path="/cache"), ValueError, "mutually exclusive"),
        (dict(is_multimodal=True), NotImplementedError, "multimodal"),
        (dict(packed_sequence_size=4096), NotImplementedError, "sequence packing"),
        (dict(cp_size=2), NotImplementedError, "context parallelism"),
    ],
)
def test_remote_options_reject_unsupported_combinations(override, error, pattern):
    with pytest.raises(error, match=pattern):
        _validate_remote_target_options(**{**_VALID_OPTIONS, **override})


def test_remote_options_accept_the_supported_combination():
    _validate_remote_target_options(**_VALID_OPTIONS)


# --- setup + forward through a real server ----------------------------------


def test_remote_setup_seeds_draft_from_server_and_feeds_colocated_features(monkeypatch, tmp_path, served_model):
    url, model = served_model
    _patch_recipe_setup(monkeypatch, model.config)

    recipe = TrainDSparkRecipe(_remote_cfg(tmp_path, remote_urls=[url]))
    recipe.setup()

    assert recipe.target_model is None
    assert isinstance(recipe.target_wrapper, RemoteDSparkTargetModel)
    torch.testing.assert_close(recipe.draft_model.embed_tokens.weight.detach(), model.get_input_embeddings().weight)
    torch.testing.assert_close(recipe.draft_model.lm_head.weight.detach(), model.get_output_embeddings().weight)

    recorded = {}
    recipe.trainer_module = lambda **kwargs: recorded.update(kwargs)
    batch = _batches()[0]
    recipe._forward_batch(batch)

    reference = HFDSparkTargetModel(model, target_layer_ids=LAYER_IDS).generate_batch(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], loss_mask=batch["loss_mask"]
    )
    torch.testing.assert_close(recorded["target_hidden_states"], reference.target_hidden_states, rtol=0, atol=0)
    torch.testing.assert_close(
        recorded["target_last_hidden_states"], reference.target_last_hidden_states, rtol=0, atol=0
    )
    torch.testing.assert_close(recorded["input_ids"], batch["input_ids"], rtol=0, atol=0)
    assert recorded["seq_lens"] is None


def test_remote_setup_rejects_a_server_with_different_capture_layers(monkeypatch, tmp_path, served_model):
    url, model = served_model
    _patch_recipe_setup(monkeypatch, model.config)

    recipe = TrainDSparkRecipe(_remote_cfg(tmp_path, remote_urls=[url], target_layer_ids=[0, 2]))
    with pytest.raises(ValueError, match=r"mismatched fields: \['target_layer_ids'\]"):
        recipe.setup()


def test_remote_setup_requires_remote_urls(monkeypatch, tmp_path):
    _patch_recipe_setup(monkeypatch, _target_model().config)

    recipe = TrainDSparkRecipe(_remote_cfg(tmp_path, remote_urls=[]))
    with pytest.raises(ValueError, match="at least one server URL"):
        recipe.setup()


class _StopSetup(Exception):
    pass


@pytest.mark.parametrize("world_size,expected_nccl", [(1, True), (2, False)])
def test_ranks_sharing_servers_stay_on_the_wire_plane(monkeypatch, tmp_path, world_size, expected_nccl):
    _patch_recipe_setup(monkeypatch, _target_model().config)
    monkeypatch.setattr(
        train_dspark_module,
        "initialize_distributed",
        lambda **_kwargs: SimpleNamespace(device=torch.device("cpu"), world_size=world_size, is_main=True),
    )
    captured = {}

    def _capture(urls, **kwargs):
        captured.update(kwargs, urls=urls)
        raise _StopSetup

    monkeypatch.setattr(train_dspark_module, "RemoteDSparkTargetModel", _capture)

    with pytest.raises(_StopSetup):
        TrainDSparkRecipe(_remote_cfg(tmp_path, remote_urls=["http://a:1", "http://b:1"])).setup()

    assert captured["enable_nccl"] is expected_nccl
    assert captured["urls"] == ["http://a:1", "http://b:1"]
    assert (captured["timeout"], captured["max_retries"]) == (10, 0)


def test_setup_rejects_unknown_target_backend(monkeypatch, tmp_path):
    _patch_recipe_setup(monkeypatch, _target_model().config)

    recipe = TrainDSparkRecipe(_remote_cfg(tmp_path, target_model_backend="vllm"))
    with pytest.raises(ValueError, match="Unknown recipe_args.target_model_backend='vllm'"):
        recipe.setup()


# --- shutdown ---------------------------------------------------------------


def _finished_recipe(target_wrapper, *, is_main: bool) -> TrainDSparkRecipe:
    """A recipe whose run loop returns at once because every epoch already ran."""
    recipe = TrainDSparkRecipe.__new__(TrainDSparkRecipe)
    recipe.trainer_module = torch.nn.Linear(1, 1)
    recipe._resume_epoch = 1
    recipe.num_epochs = 1
    recipe.dist_env = SimpleNamespace(is_main=is_main)
    recipe.target_wrapper = target_wrapper
    return recipe


@pytest.mark.parametrize("is_main,expected_closes", [(True, 1), (False, 0)])
def test_only_the_main_rank_disconnects_from_the_remote_target(monkeypatch, is_main, expected_closes):
    remote = RemoteDSparkTargetModel(["http://127.0.0.1:9"], device=torch.device("cpu"), max_retries=0)
    closed = []
    monkeypatch.setattr(remote, "close", lambda: closed.append(True))

    _finished_recipe(remote, is_main=is_main).run_train_validation_loop()

    assert len(closed) == expected_closes


def test_finished_run_leaves_a_colocated_target_alone():
    colocated = SimpleNamespace(close=lambda: pytest.fail("a colocated target has nothing to disconnect"))
    _finished_recipe(colocated, is_main=True).run_train_validation_loop()

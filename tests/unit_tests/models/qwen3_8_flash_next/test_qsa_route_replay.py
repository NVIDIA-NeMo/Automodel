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

"""QSA route replay under activation checkpointing: the frozen indexer runs once per block."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import noop_context_fn

from nemo_automodel.components.distributed.activation_checkpointing import make_selective_checkpoint_context_fn
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    QSARouteReplayRecorder,
    QSARouteSelection,
    current_qsa_route_replay,
    qsa_route_replay_scope,
)
from nemo_automodel.components.moe.parallelizer import _with_model_checkpoint_context
from tests.unit_tests.models.qwen3_8_flash_next.test_qwen3_8_flash_next_model import (
    _tiny_config,
    _tiny_moe_config,
)


def _build_model(reuse_routes: bool) -> Qwen3_8_FlashNextForConditionalGeneration:
    config = _tiny_config()
    config.text_config.qsa_reuse_routes_on_recompute = reuse_routes
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    torch.manual_seed(0)
    model = Qwen3_8_FlashNextForConditionalGeneration.from_config(
        config,
        moe_config=_tiny_moe_config(config.text_config),
        backend=backend,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.train()
    return model


def _run_step(model: nn.Module, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """One forward/backward; returns the loss and the gradient of every trainable parameter."""
    model.zero_grad(set_to_none=True)
    loss = model(input_ids=input_ids).logits.float().square().mean()
    loss.backward()
    grads = {
        # checkpoint_wrapper prefixes wrapped parameters; strip it so runs stay comparable.
        name.replace("_checkpoint_wrapped_module.", ""): p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    return loss.detach(), grads


def _count_indexer_calls(model: nn.Module) -> list[int]:
    calls = [0]
    # checkpoint_wrapper forwards attribute access to the wrapped block.
    indexer = model.model.language_model.layers["0"].self_attn.indexer
    indexer.register_forward_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    return calls


def _wrap_layer_with_checkpoint(model: nn.Module, *, selective: bool) -> None:
    layers = model.model.language_model.layers
    block = layers["0"]
    base_context_fn = make_selective_checkpoint_context_fn() if selective else None
    context_fn = _with_model_checkpoint_context(block, base_context_fn) or noop_context_fn
    layers["0"] = checkpoint_wrapper(block, preserve_rng_state=True, context_fn=context_fn)


def test_recorder_replays_in_order_then_reports_misses() -> None:
    recorder = QSARouteReplayRecorder()
    first = QSARouteSelection(selected_token_ids=torch.tensor([[[0]]], dtype=torch.int32), flex_mask=None)
    second = QSARouteSelection(selected_token_ids=torch.tensor([[[1]]], dtype=torch.int32), flex_mask=None)
    recorder.record(first)
    recorder.record(second)
    assert len(recorder) == 2
    assert recorder.take() is first
    assert recorder.take() is second
    assert recorder.take() is None
    assert recorder.replay_misses == 1
    recorder.rewind()
    assert recorder.take() is first


def test_replay_scope_binds_and_restores_thread_state() -> None:
    assert current_qsa_route_replay() is None
    recorder = QSARouteReplayRecorder()
    with qsa_route_replay_scope(recorder, "record"):
        assert current_qsa_route_replay() == (recorder, "record")
        with qsa_route_replay_scope(recorder, "replay"):
            assert current_qsa_route_replay() == (recorder, "replay")
        assert current_qsa_route_replay() == (recorder, "record")
    assert current_qsa_route_replay() is None
    with qsa_route_replay_scope(None, "record"):
        assert current_qsa_route_replay() is None


def test_parallelizer_hook_only_wraps_blocks_that_expose_it() -> None:
    plain = nn.Linear(2, 2)
    sentinel = object()
    assert _with_model_checkpoint_context(plain, sentinel) is sentinel

    model = _build_model(reuse_routes=True)
    block = model.model.language_model.layers["0"]
    wrapped = _with_model_checkpoint_context(block, None)
    assert callable(wrapped)
    forward_ctx, recompute_ctx = wrapped()
    with forward_ctx:
        binding = current_qsa_route_replay()
        assert binding is not None and binding[1] == "record"
    with recompute_ctx:
        binding = current_qsa_route_replay()
        assert binding is not None and binding[1] == "replay"

    disabled = _build_model(reuse_routes=False)
    assert _with_model_checkpoint_context(disabled.model.language_model.layers["0"], None) is None


@pytest.mark.runtime_budget(
    60,
    hard_timeout=180,
    reason="five CPU forward/backward passes of the tiny HC decoder, two under selective checkpointing",
)
def test_checkpoint_replay_runs_indexer_once_and_is_bitwise_identical() -> None:
    """Checkpointed block with replay: indexer once, loss and every gradient identical to no-AC and to AC without replay."""
    input_ids = torch.randint(2, 64, (2, 6), generator=torch.Generator().manual_seed(1))

    reference = _build_model(reuse_routes=True)
    reference_calls = _count_indexer_calls(reference)
    ref_loss, ref_grads = _run_step(reference, input_ids)
    assert reference_calls[0] == 1

    for selective in (False, True):
        no_replay = copy.deepcopy(reference)
        no_replay.model.language_model.layers["0"].self_attn.reuse_routes_on_recompute = False
        _wrap_layer_with_checkpoint(no_replay, selective=selective)
        no_replay_calls = _count_indexer_calls(no_replay)
        nr_loss, nr_grads = _run_step(no_replay, input_ids)
        assert no_replay_calls[0] == 2, "without replay the recompute must rerun the indexer"

        replay = copy.deepcopy(reference)
        _wrap_layer_with_checkpoint(replay, selective=selective)
        replay_calls = _count_indexer_calls(replay)
        rp_loss, rp_grads = _run_step(replay, input_ids)
        assert replay_calls[0] == 1, "with replay the recompute must reuse the recorded routes"

        assert torch.equal(rp_loss, ref_loss) and torch.equal(nr_loss, ref_loss)
        assert rp_grads.keys() == ref_grads.keys() == nr_grads.keys()
        for name in ref_grads:
            assert torch.equal(rp_grads[name], ref_grads[name]), f"replay gradient differs for {name}"
            assert torch.equal(nr_grads[name], ref_grads[name]), f"checkpoint gradient differs for {name}"

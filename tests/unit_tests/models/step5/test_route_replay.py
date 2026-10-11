# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CSA routes use the native checkpoint recorder across live microbatches."""

import copy
import gc
import weakref

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.utils.checkpoint import noop_context_fn

from nemo_automodel.components.distributed.activation_checkpointing import make_selective_checkpoint_context_fn
from nemo_automodel.components.models.step5.layers import region_replay
from nemo_automodel.components.models.step5.model import Step5ForCausalLM
from nemo_automodel.components.moe.parallelizer import _with_model_checkpoint_context
from tests.unit_tests.models.step5.test_model import make_model


def _step(model: Step5ForCausalLM, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run tiny native model loss/backward on integer [batch, sequence] inputs."""
    model.zero_grad(set_to_none=True)
    loss = model(inputs, labels=inputs).loss
    loss.backward()
    gradients = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), gradients


@pytest.mark.runtime_budget(20, reason="Three tiny-model backward passes with cold CPU compilation and native AC")
@pytest.mark.parametrize("selective", [False, True])
def test_native_checkpoint_replays_selector_once_with_exact_loss_and_gradients(selective: bool) -> None:
    reference = make_model()
    candidate = copy.deepcopy(reference)
    no_replay = copy.deepcopy(reference)
    inputs = torch.arange(16)[None]
    expected_loss, expected_gradients = _step(reference, inputs)
    for model, replay_enabled in [(no_replay, False), (candidate, True)]:
        calls = []
        module = model.model.layers["3"]
        module.self_attn.indexer.register_forward_hook(lambda *args: calls.append(1))
        base = make_selective_checkpoint_context_fn() if selective else None
        context_fn = _with_model_checkpoint_context(module, base) if replay_enabled else base
        model.model.layers["3"] = checkpoint_wrapper(
            module, preserve_rng_state=True, context_fn=context_fn or noop_context_fn
        )
        loss, gradients = _step(model, inputs)
        assert len(calls) == (1 if replay_enabled else 2)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        assert gradients.keys() == expected_gradients.keys()
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=0, atol=0, msg=name)
        assert region_replay.current() is None


def test_live_microbatches_have_distinct_routes_and_release_records_after_backward() -> None:
    reference = make_model()
    candidate = copy.deepcopy(reference)
    module = candidate.model.layers["3"]
    routes = []
    module.self_attn.indexer.register_forward_hook(lambda module, args, result: routes.append(weakref.ref(result)))
    candidate.model.layers["3"] = checkpoint_wrapper(module, context_fn=_with_model_checkpoint_context(module, None))
    examples = [
        (torch.arange(16)[None], torch.arange(16)[None]),
        ((torch.arange(16) + 17)[None], torch.arange(8).repeat(2)[None]),
    ]
    expected = [reference(ids, labels=ids, position_ids=positions).loss for ids, positions in examples]
    actual = [candidate(ids, labels=ids, position_ids=positions).loss for ids, positions in examples]
    assert len(routes) == 2 and routes[0]() is not routes[1]()
    for result in reversed(actual):
        result.backward()
    for result in reversed(expected):
        result.backward()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    for name, parameter in candidate.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, ref_grads[name.replace("_checkpoint_wrapped_module.", "")], rtol=0, atol=0
            )
    assert len(routes) == 2
    del actual, result, got
    gc.collect()
    assert all(route() is None for route in routes)
    assert region_replay.current() is None

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""RL roles are assembled by the CALLER, not the Engine.

The Engine holds no role policy (like Megatron-core): a critic is just a model
that emits ``values``; a reference is an Engine with no optimizer + a frozen
model. These are expressed via generic mechanisms — duck-typed output
extraction, the ``hooks`` model-surgery seam, and "inject no optimizer" — not
``output_head`` / ``trainable`` config.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.training.model_output import ModelOutput
from nemo_automodel.components.training.engine import Engine


class ToyLM(nn.Module):
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, position_ids=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


class ToyValueModel(nn.Module):
    """A critic: emits a per-token scalar value instead of vocab logits."""

    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.value_head = nn.Linear(dim, 1)

    def forward(self, input_ids, position_ids=None):
        return SimpleNamespace(values=self.value_head(self.embed(input_ids)))  # [B, T, 1]


class ToyLoRAModel(ToyLM):
    """LM with a togglable additive adapter and a PEFT-style disable_adapter ctx."""

    def __init__(self, vocab=16, dim=8):
        super().__init__(vocab, dim)
        self.adapter = nn.Linear(dim, vocab)
        self._adapter_on = True

    def forward(self, input_ids, position_ids=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        if self._adapter_on:
            logits = logits + self.adapter(h)
        return SimpleNamespace(logits=logits)

    @contextmanager
    def disable_adapter(self):
        prev = self._adapter_on
        self._adapter_on = False
        try:
            yield
        finally:
            self._adapter_on = prev


def _datums(vocab=16):
    return [
        Datum(
            input_ids=torch.randint(0, vocab, (4,)),
            loss_inputs={"target_tokens": torch.randint(0, vocab, (4,)), "weights": torch.ones(4)},
        ),
        Datum(
            input_ids=torch.randint(0, vocab, (2,)),
            loss_inputs={"target_tokens": torch.randint(0, vocab, (2,)), "weights": torch.ones(2)},
        ),
    ]


def _trainable(model, lr=0.1):
    return Engine(model_parts=[model], optimizers=[torch.optim.SGD(model.parameters(), lr=lr)])


def _reference(model):
    """Caller-side reference role: no optimizer + frozen, eval. No Engine flag."""
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return Engine(model_parts=[model], optimizers=[])


# ── Duck-typed extraction: the Engine surfaces what the model emits ─────────


def test_lm_model_yields_logprobs():
    eng = _trainable(ToyLM())
    out = eng.forward(_datums())
    assert out.logprobs is not None and out.values is None
    assert [lp.shape[0] for lp in out.logprobs] == [4, 2]


def test_value_model_yields_values():
    eng = _trainable(ToyValueModel())
    out = eng.forward(_datums())
    assert out.values is not None and out.logprobs is None  # no output_head config needed
    assert [v.shape[0] for v in out.values] == [4, 2]


def test_critic_trains_with_custom_value_loss():
    critic = _trainable(ToyValueModel())

    def value_loss(model_output, datums):
        return [v.pow(2) for v in model_output.values]  # token-level MSE-to-zero

    out = critic.forward_backward(_datums(), loss_fn=value_loss)
    assert isinstance(out, ModelOutput) and torch.isfinite(out.loss)
    ok, _ = critic.optimizer_step()
    assert ok


# ── Reference = no optimizer + frozen (caller-assembled, no Engine flag) ────


def test_reference_has_no_optimizer_and_runs_forward():
    ref = _reference(ToyLM())
    assert ref.optimizers == []
    assert all(not p.requires_grad for p in ref.parts[0].parameters())
    assert not ref.parts[0].training
    assert ref.forward(_datums()).logprobs is not None


def test_optimizer_step_without_optimizer_raises():
    ref = _reference(ToyLM())
    with pytest.raises(RuntimeError, match="no optimizers"):
        ref.optimizer_step()


# ── hooks: the generic model-surgery seam (how a value head/freeze is done) ──


def test_hook_freezes_module_in_place():
    model = ToyLM()

    def freeze_embed(m):
        m.embed.weight.requires_grad_(False)
        return None

    eng = Engine(
        Engine.Config(hooks=[freeze_embed]),
        model_parts=[model],
        optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)],
    )
    assert not eng.parts[0].embed.weight.requires_grad


def test_hook_can_replace_module():
    replacement = ToyLM()
    eng = Engine(Engine.Config(hooks=[lambda m: replacement]), model_parts=[ToyLM()])
    assert eng.parts[0] is replacement


# ── disable_adapters: generic PEFT mechanism (used by SFT eval + RL ref) ────


def test_disable_adapters_changes_output():
    torch.manual_seed(0)
    model = ToyLoRAModel()
    eng = _trainable(model)
    datums = _datums()
    with_adapter = eng.forward(datums).logprobs
    without_adapter = eng.forward(datums, disable_adapters=True).logprobs
    assert not torch.allclose(with_adapter[0], without_adapter[0])
    assert model._adapter_on is True  # restored after the context


def test_disable_adapter_noop_without_adapters():
    eng = _trainable(ToyLM())
    out = eng.forward(_datums(), disable_adapters=True)  # no-op, still runs
    assert out.logprobs is not None


# ── Deliverable: actor / critic / ref are three CALLER-assembled Engines ────


def test_actor_critic_ref_are_caller_assembled():
    torch.manual_seed(0)
    actor = _trainable(ToyLM())  # policy: logits -> logprobs, trainable
    critic = _trainable(ToyValueModel())  # value model -> values, trainable
    ref = _reference(ToyLM())  # frozen, no optimizer

    datums = _datums()
    assert actor.forward(datums).logprobs is not None
    assert critic.forward(datums).values is not None
    assert ref.forward(datums).logprobs is not None
    assert actor.optimizers and critic.optimizers and ref.optimizers == []
    # The Engine class is identical for all three — only the model + optimizer differ.
    assert type(actor) is type(critic) is type(ref) is Engine

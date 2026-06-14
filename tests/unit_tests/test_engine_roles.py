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
"""Phase 4 — RL role semantics: trainable, output_head, hooks, disable_adapters."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.training.model_output import ModelOutput
from nemo_automodel.engine import Engine


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


def _engine(model, *, trainable=True, output_head="lm", lr=0.1):
    cfg = Engine.Config(trainable=trainable, output_head=output_head)
    opt = [torch.optim.SGD(model.parameters(), lr=lr)] if trainable else None
    return Engine(cfg, model_parts=[model], optimizers=opt)


# ── trainable=False (reference / reward model) ──────────────────────────────


def test_non_trainable_freezes_and_drops_optimizer():
    torch.manual_seed(0)
    ref = _engine(ToyLM(), trainable=False, lr=0.1)
    assert ref.optimizers == []
    assert all(not p.requires_grad for p in ref.parts[0].parameters())
    assert not ref.parts[0].training  # eval()
    # forward still works (ref logprobs).
    out = ref.forward(_datums())
    assert [lp.shape[0] for lp in out.logprobs] == [4, 2]


def test_non_trainable_optimizer_step_raises():
    ref = _engine(ToyLM(), trainable=False)
    with pytest.raises(RuntimeError, match="no optimizers"):
        ref.optimizer_step()


# ── output_head="value" (critic) ────────────────────────────────────────────


def test_value_head_forward_emits_values():
    critic = _engine(ToyValueModel(), output_head="value")
    out = critic.forward(_datums())
    assert out.values is not None and out.logprobs is None
    assert [v.shape[0] for v in out.values] == [4, 2]  # per-token, per-datum


def test_value_head_train_with_custom_value_loss():
    critic = _engine(ToyValueModel(), output_head="value")
    datums = _datums()

    def value_loss(model_output, datums):
        # simple MSE-to-zero critic loss; token-level (per-token tensor per datum)
        return [v.pow(2) for v in model_output.values]

    out = critic.forward_backward(datums, loss_fn=value_loss)
    assert isinstance(out, ModelOutput)
    assert torch.isfinite(out.loss)
    ok, _ = critic.optimizer_step()
    assert ok


def test_value_head_missing_values_raises():
    # An LM model under output_head="value" has no `.values`.
    eng = _engine(ToyLM(), output_head="value")
    with pytest.raises(ValueError, match="no `.values`"):
        eng.forward(_datums())


# ── hooks (post-construction) ───────────────────────────────────────────────


def test_hooks_applied_to_model_parts():
    def freeze_embed(m):
        m.embed.weight.requires_grad_(False)
        return None  # in-place

    model = ToyLM()
    cfg = Engine.Config(hooks=[freeze_embed])
    eng = Engine(cfg, model_parts=[model], optimizers=[torch.optim.SGD(model.parameters(), lr=0.1)])
    assert not eng.parts[0].embed.weight.requires_grad


def test_hook_can_replace_module():
    replacement = ToyLM()
    eng = Engine(Engine.Config(hooks=[lambda m: replacement]), model_parts=[ToyLM()])
    assert eng.parts[0] is replacement


# ── disable_adapters (LoRA ref without a second engine) ─────────────────────


def test_disable_adapters_changes_output():
    torch.manual_seed(0)
    model = ToyLoRAModel()
    eng = _engine(model)
    datums = _datums()
    with_adapter = eng.forward(datums).logprobs
    without_adapter = eng.forward(datums, disable_adapters=True).logprobs
    # adapter on vs off must produce different logprobs
    assert not torch.allclose(with_adapter[0], without_adapter[0])
    # adapter restored after the context
    assert model._adapter_on is True


def test_disable_adapter_noop_without_adapters():
    eng = _engine(ToyLM())
    # no disable_adapter method on the model -> no-op, still runs
    out = eng.forward(_datums(), disable_adapters=True)
    assert out.logprobs is not None


# ── Deliverable: actor / critic / ref as three Engines ──────────────────────


def test_actor_critic_ref_roles():
    torch.manual_seed(0)
    actor = _engine(ToyLM(), trainable=True)
    critic = _engine(ToyValueModel(), trainable=True, output_head="value")
    ref = _engine(ToyLM(), trainable=False)

    datums = _datums()
    assert actor.forward(datums).logprobs is not None
    assert critic.forward(datums).values is not None
    assert ref.forward(datums).logprobs is not None
    # only actor + critic are trainable
    assert actor.optimizers and critic.optimizers and ref.optimizers == []

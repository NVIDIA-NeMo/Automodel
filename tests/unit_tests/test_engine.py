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
"""CPU unit tests for the Engine spine via the injection construction path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.training.model_output import ModelOutput
from nemo_automodel.engine import Engine


class ToyLM(nn.Module):
    """Tiny next-token model: embedding -> linear -> [B, T, V] logits."""

    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, position_ids=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def toy_loss(logits=None, labels=None, num_label_tokens=None):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def _engine(vocab=16, dim=8, lr=0.1):
    torch.manual_seed(0)
    model = ToyLM(vocab, dim)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return Engine(model_parts=[model], optimizers=[opt]), model


def _datums(vocab=16):
    return [
        Datum(
            input_ids=torch.randint(0, vocab, (5,)),
            loss_inputs={"target_tokens": torch.randint(0, vocab, (5,)), "weights": torch.ones(5)},
        ),
        Datum(
            input_ids=torch.randint(0, vocab, (3,)),
            loss_inputs={"target_tokens": torch.randint(0, vocab, (3,)), "weights": torch.ones(3)},
        ),
    ]


# ── construction / introspection ────────────────────────────────────────────


def test_injection_construction():
    engine, model = _engine()
    assert engine.parts == [model]
    assert engine.model is model
    assert engine.pp_enabled is False
    assert engine.dp_size == 1
    assert engine.dp_rank == 0


# ── forward_backward ─────────────────────────────────────────────────────────


def test_forward_backward_dict_returns_dict_and_grads():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    out = engine.forward_backward(batch, loss_fn=toy_loss)
    assert isinstance(out, dict) and "loss" in out and "metrics" in out
    assert torch.isfinite(out["loss"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_forward_backward_datums_returns_model_output():
    engine, _ = _engine()
    out = engine.forward_backward(_datums(), loss_fn=toy_loss)
    assert isinstance(out, ModelOutput)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.metrics["loss"] == pytest.approx(float(out.loss))


def test_forward_only_leaves_no_grads():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss, forward_only=True)
    assert all(p.grad is None for p in model.parameters())


def test_list_of_microbatches_accumulates():
    engine, model = _engine()
    mbs = [
        {"input_ids": torch.randint(0, 16, (2, 4)), "labels": torch.randint(0, 16, (2, 4))},
        {"input_ids": torch.randint(0, 16, (2, 4)), "labels": torch.randint(0, 16, (2, 4))},
    ]
    out = engine.forward_backward(mbs, loss_fn=toy_loss)
    assert torch.isfinite(out["loss"])


def test_pp_path_raises():
    engine, _ = _engine()
    engine.pp = object()  # simulate AutoPipeline present
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        engine.forward_backward({"input_ids": torch.zeros(1, 1, dtype=torch.long)}, loss_fn=toy_loss)


# ── forward (per-datum logprobs / entropy) ───────────────────────────────────


def test_forward_returns_per_datum_logprobs_and_entropy():
    engine, _ = _engine()
    datums = _datums()
    out = engine.forward(datums)
    assert isinstance(out, ModelOutput)
    assert [lp.shape[0] for lp in out.logprobs] == [d.seq_len for d in datums]
    assert [e.shape[0] for e in out.entropy] == [d.seq_len for d in datums]
    # logprobs are log-probabilities -> non-positive.
    assert all((lp <= 1e-4).all() for lp in out.logprobs)


def test_forward_requires_datums():
    engine, _ = _engine()
    with pytest.raises(TypeError, match="Datum"):
        engine.forward([{"input_ids": torch.zeros(3, dtype=torch.long)}])


# ── optimizer ─────────────────────────────────────────────────────────────────


def test_optimizer_step_updates_params():
    engine, model = _engine(lr=0.5)
    before = model.head.weight.detach().clone()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss)
    ok, grad_norm = engine.optimizer_step()
    assert ok is True
    assert grad_norm >= 0.0
    assert not torch.equal(before, model.head.weight)


def test_optim_step_sets_lr():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss)
    metrics = engine.optim_step(lr=0.123)
    assert metrics["lr"] == 0.123
    assert metrics["update_succeeded"] is True
    assert all(g["lr"] == 0.123 for opt in engine.optimizers for g in opt.param_groups)


def test_zero_grad():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss)
    engine.zero_grad()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())


# ── modes / device / export ──────────────────────────────────────────────────


def test_train_eval_mode_restore():
    engine, model = _engine()
    model.eval()
    with engine.train_mode():
        assert model.training is True
    assert model.training is False  # restored


def test_to_cpu_is_safe():
    engine, model = _engine()
    engine.to("cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_export_weights_yields_named_tensors():
    engine, _ = _engine()
    weights = dict(engine.export_weights())
    assert "head.weight" in weights and "embed.weight" in weights
    assert isinstance(weights["head.weight"], torch.Tensor)

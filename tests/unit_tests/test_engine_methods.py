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

"""End-to-end tests for the Engine's method bodies.

Bypasses the HF model-build pipeline by constructing a tiny nn.Module
manually and wiring it into an Engine. This isolates the Engine's own
logic (microbatch loop, gradient prep hooks, grad clip, LR scheduler
construction, weight export, offload/onload) from the recipe builder
plumbing.

A separate functional test exercises Engine.build() end-to-end with a real
HF model under distributed init.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn


# ── Shared fixtures ──────────────────────────────────────────────────


def _single_rank_dist():
    """Initialize torch.distributed with a single CPU rank using gloo."""
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    dist.init_process_group(backend="gloo", world_size=1, rank=0)


@pytest.fixture(scope="module")
def dist_env():
    _single_rank_dist()
    yield
    # Leave the process group in place — tearing it down breaks other tests
    # in the same process. Single-rank gloo is cheap to keep around.


@pytest.fixture
def tiny_model_engine(dist_env):
    """Build an Engine wrapping a tiny manually-constructed nn.Linear model."""
    from nemo_automodel.engine import Engine

    engine = Engine(Engine.Config(
        model=None,
        distributed=SimpleNamespace(),         # not used — we skip build()
        optimizer=SimpleNamespace(),
        lr_scheduler=None,
        max_grad_norm=1.0,
    ))

    # Manually populate state — same as what build() would leave behind,
    # minus the real model/distributed plumbing.
    model = nn.Linear(8, 8)
    engine.model = model
    engine.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    engine.mesh = None   # no mesh → Engine falls back to safe defaults
    return engine


# ── _build_lr_scheduler — direct construction (bug fix #3) ───────────


def test_lr_scheduler_direct_construction(dist_env):
    """The fix bypasses recipes.build_lr_scheduler; verify the direct path works."""
    from nemo_automodel.engine import Engine, LRSchedulerConfig

    engine = Engine(Engine.Config(
        model=None,
        distributed=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        lr_scheduler=LRSchedulerConfig(
            total_steps=100,
            lr_warmup_steps=10,
            lr_decay_style="cosine",
            init_lr_ratio=0.1,
            min_lr_ratio=0.01,
        ),
    ))

    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    scheduler = engine._build_lr_scheduler(optimizer)

    assert scheduler is not None
    # OptimizerParamScheduler bumps step internally; cycle a step.
    scheduler.step(increment=1)
    lr_after = optimizer.param_groups[0]["lr"]
    assert lr_after > 0, "LR scheduler step produced a non-positive LR"


def test_lr_scheduler_none_when_cfg_missing(dist_env):
    from nemo_automodel.engine import Engine

    engine = Engine(Engine.Config(
        model=None,
        distributed=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        lr_scheduler=None,
    ))
    assert engine._build_lr_scheduler(torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)) is None


def test_lr_scheduler_warmup_ratio(dist_env):
    """Warmup steps can be specified as a ratio of total_steps."""
    from nemo_automodel.engine import Engine, LRSchedulerConfig

    engine = Engine(Engine.Config(
        model=None, distributed=SimpleNamespace(),
        optimizer=SimpleNamespace(),
        lr_scheduler=LRSchedulerConfig(total_steps=200, lr_warmup_steps_ratio=0.05),
    ))
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    scheduler = engine._build_lr_scheduler(optimizer)
    assert scheduler is not None
    # warmup_steps should be 0.05 * 200 = 10
    # We don't have a public accessor for it; assert via _check_and_set_lr_warmup_steps if exposed,
    # otherwise just verify scheduler builds and steps cleanly.
    scheduler.step(increment=1)
    assert optimizer.param_groups[0]["lr"] > 0


def test_lr_scheduler_missing_total_steps_raises():
    """The recipe-layer resolver rejects cfg without ``total_steps``."""
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.llm.train_ft import _resolve_lr_scheduler_config

    with pytest.raises(ValueError, match="total_steps"):
        _resolve_lr_scheduler_config(ConfigNode({"lr_warmup_steps": 10}))


# ── forward_backward — manual model path ─────────────────────────────


class _TinyLossModel(nn.Module):
    """nn.Module whose forward returns a SimpleNamespace with logits + loss."""

    def __init__(self, in_dim: int = 8, vocab: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab, in_dim)
        self.head = nn.Linear(in_dim, vocab)

    def forward(self, input_ids, labels=None, **kw):
        h = self.embed(input_ids)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(logits=logits, loss=loss, hidden_states=None)


@pytest.fixture
def forward_engine(dist_env):
    """Engine wired to the TinyLossModel — exercises forward_backward end-to-end."""
    from nemo_automodel.engine import Engine

    engine = Engine(Engine.Config(
        model=None, distributed=SimpleNamespace(),
        optimizer=SimpleNamespace(), lr_scheduler=None,
        max_grad_norm=1.0,
    ))
    model = _TinyLossModel()
    engine.model = model
    engine.optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    engine.mesh = None
    return engine


def test_forward_backward_loss_fn_none(forward_engine):
    """When loss_fn is None, Engine reads model_output.loss (standard SFT path)."""
    batch = {
        "input_ids": torch.randint(0, 16, (4, 8)),
        "labels": torch.randint(0, 16, (4, 8)),
    }
    out = forward_engine.forward_backward(batch, num_microbatches=1)
    assert "loss" in out
    assert torch.is_tensor(out["loss"])
    assert torch.isfinite(out["loss"])


def test_forward_backward_microbatching(forward_engine):
    """Batch is split into microbatches; result is the mean loss."""
    batch = {
        "input_ids": torch.randint(0, 16, (8, 8)),
        "labels": torch.randint(0, 16, (8, 8)),
    }
    out = forward_engine.forward_backward(batch, num_microbatches=4)
    assert torch.is_tensor(out["loss"])
    # Gradients should have accumulated.
    grads_present = any(p.grad is not None for p in forward_engine.model.parameters())
    assert grads_present, "No gradients after forward_backward"


def test_forward_backward_forward_only(forward_engine):
    """forward_only=True skips backward; no gradients populated."""
    forward_engine.zero_grad()
    batch = {
        "input_ids": torch.randint(0, 16, (4, 8)),
        "labels": torch.randint(0, 16, (4, 8)),
    }
    out = forward_engine.forward_backward(batch, num_microbatches=1, forward_only=True)
    assert torch.is_tensor(out["loss"])
    # No grads should have been computed.
    for p in forward_engine.model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_optimizer_step_clips_and_steps(forward_engine):
    """optimizer_step returns (ok, grad_norm) and actually updates params."""
    batch = {
        "input_ids": torch.randint(0, 16, (4, 8)),
        "labels": torch.randint(0, 16, (4, 8)),
    }
    forward_engine.zero_grad()
    forward_engine.forward_backward(batch, num_microbatches=1)
    before = {n: p.detach().clone() for n, p in forward_engine.model.named_parameters()}
    ok, grad_norm = forward_engine.optimizer_step()
    assert ok is True
    assert grad_norm >= 0
    moved = any(
        not torch.allclose(before[n], p.detach())
        for n, p in forward_engine.model.named_parameters()
    )
    assert moved, "optimizer_step did not update any parameters"


# ── List-of-microbatches path ─────────────────────────────────────────


def test_forward_backward_accepts_list_of_batches(forward_engine):
    """Engine accepts a pre-split list[dict] instead of a single batch + n."""
    batches = [
        {"input_ids": torch.randint(0, 16, (2, 8)), "labels": torch.randint(0, 16, (2, 8))}
        for _ in range(3)
    ]
    out = forward_engine.forward_backward(batches, num_microbatches=1)  # n ignored when list
    assert torch.is_tensor(out["loss"])
    assert len(out["losses"]) == 3


# ── MoE aux-loss scale (replaces TestRunTrainOptimStepSetsMoEScale) ───


def test_moe_aux_loss_scale_set_when_moe_cfg_present(dist_env):
    """Engine.forward_backward sets MoEAuxLossAutoScaler.main_loss_backward_scale
    when mesh.moe_config is present (matches the recipe's pre-migration MoE path).
    """
    from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
    from nemo_automodel.engine import Engine

    MoEAuxLossAutoScaler.main_loss_backward_scale = None

    engine = Engine(Engine.Config(
        model=None, distributed=SimpleNamespace(),
        optimizer=SimpleNamespace(), lr_scheduler=None,
    ))
    model = _TinyLossModel()
    engine.model = model
    engine.optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    # mesh.moe_config presence is what gates the aux-loss scale logic.
    engine.mesh = SimpleNamespace(
        device_mesh=None, moe_mesh=None, moe_config=object(),
        strategy_config=None, pipeline_config=None,
    )

    batch = {"input_ids": torch.randint(0, 16, (4, 8)), "labels": torch.randint(0, 16, (4, 8))}
    engine.forward_backward([batch], loss_fn=None, num_label_tokens=10)

    assert MoEAuxLossAutoScaler.main_loss_backward_scale is not None
    # Non-PP path uses dp_size; with single-rank dist init that's 1.
    expected_dp = float(engine.dp_size)
    assert MoEAuxLossAutoScaler.main_loss_backward_scale.item() == pytest.approx(expected_dp)


# ── export_weights ───────────────────────────────────────────────────


def test_export_weights_iterates_all_params(forward_engine):
    """export_weights yields every state-dict entry."""
    expected = set(forward_engine.model.state_dict().keys())
    seen: set[str] = set()
    for name, tensor in forward_engine.export_weights(to_hf=False):
        assert torch.is_tensor(tensor)
        seen.add(name)
    assert seen == expected


# ── train_mode / eval_mode context managers ─────────────────────────


def test_train_mode_restores_state(forward_engine):
    forward_engine.model.eval()
    assert forward_engine.model.training is False
    with forward_engine.train_mode():
        assert forward_engine.model.training is True
    assert forward_engine.model.training is False


def test_eval_mode_restores_state(forward_engine):
    forward_engine.model.train()
    assert forward_engine.model.training is True
    with forward_engine.eval_mode():
        assert forward_engine.model.training is False
    assert forward_engine.model.training is True


# ── to() ─────────────────────────────────────────────────────────────


def test_to_cpu_no_optimizer_state(forward_engine):
    """to('cpu') should not crash even when optimizer state is empty."""
    # Optimizer hasn't stepped yet → no state. Should still be a clean call.
    forward_engine.to("cpu", model=True, optimizer=True, grad=True)
    # Model params land on CPU.
    for p in forward_engine.model.parameters():
        assert p.device.type == "cpu"


# ── Lifecycle compose: full SFT loop ─────────────────────────────────


def test_full_train_loop_runs(forward_engine):
    """One end-to-end SFT loop iteration using the Engine's public methods."""
    batch = {
        "input_ids": torch.randint(0, 16, (4, 8)),
        "labels": torch.randint(0, 16, (4, 8)),
    }
    with forward_engine.train_mode():
        forward_engine.zero_grad()
        out = forward_engine.forward_backward(batch, num_microbatches=2)
        ok, grad_norm = forward_engine.optimizer_step()
    assert torch.is_tensor(out["loss"])
    assert ok
    assert grad_norm >= 0

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for EMAManager — pure math correctness, FSDP-agnostic.

After N updates with constant decay, the closed form is

    ema_N = decay^N · ema_0 + (1 - decay) · sum_{i=0..N-1} decay^(N-1-i) · w_i

where ``w_i`` is the train-param value used in the i-th update. Tests check
the recurrence end-to-end, plus boundary cases (decay=0, decay=1) and the
state-dict round trip.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.training.ema import EMAManager


def _tiny_model() -> nn.Module:
    """A 2-param model is enough to validate the math."""
    return nn.Linear(2, 2, bias=False)


def test_ema_initial_state_matches_model():
    torch.manual_seed(0)
    m = _tiny_model()
    init = m.weight.detach().clone()
    ema = EMAManager(m, decay=0.5)
    assert torch.equal(ema.state_dict()["weight"], init)


def test_ema_closed_form_5_updates():
    """Match the closed-form value after 5 updates with decay=0.9 and known weights."""
    torch.manual_seed(0)
    m = _tiny_model()
    init = m.weight.detach().clone()
    decay = 0.9

    ema = EMAManager(m, decay=decay)

    # Generate 5 known weight states + apply each as a fake "after optimizer.step()".
    weight_history = []
    for _ in range(5):
        m.weight.data = torch.randn_like(m.weight)
        weight_history.append(m.weight.detach().clone())
        ema.update(m)

    # Closed form: ema_N = decay^N * init + (1-decay) * sum_i decay^(N-1-i) * w_i
    N = 5
    expected = (decay**N) * init
    for i, w in enumerate(weight_history):
        expected = expected + (1.0 - decay) * (decay ** (N - 1 - i)) * w

    assert torch.allclose(ema.state_dict()["weight"], expected, atol=1e-6, rtol=1e-6)


def test_ema_decay_zero_tracks_train_exactly():
    """decay=0 collapses to ema = train each step."""
    torch.manual_seed(0)
    m = _tiny_model()
    ema = EMAManager(m, decay=0.0)

    m.weight.data.fill_(7.0)
    ema.update(m)
    assert torch.allclose(ema.state_dict()["weight"], m.weight.data)

    m.weight.data.fill_(-3.0)
    ema.update(m)
    assert torch.allclose(ema.state_dict()["weight"], m.weight.data)


def test_ema_decay_one_never_changes():
    """decay=1 freezes the shadow at the initial value."""
    torch.manual_seed(0)
    m = _tiny_model()
    init = m.weight.detach().clone()
    ema = EMAManager(m, decay=1.0)

    m.weight.data = torch.randn_like(m.weight) * 10
    ema.update(m)
    m.weight.data = torch.randn_like(m.weight) * 10
    ema.update(m)

    assert torch.equal(ema.state_dict()["weight"], init)


def test_ema_skips_frozen_params():
    """Params with requires_grad=False are not tracked by the EMA."""
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    # Freeze the second linear.
    for p in m[1].parameters():
        p.requires_grad = False

    ema = EMAManager(m, decay=0.5)
    names = set(ema.state_dict().keys())
    assert any(n.startswith("0.") for n in names), "trainable layer not tracked"
    assert not any(n.startswith("1.") for n in names), "frozen layer should not be tracked"


def test_ema_state_dict_roundtrip():
    torch.manual_seed(0)
    m1 = _tiny_model()
    ema1 = EMAManager(m1, decay=0.5)
    m1.weight.data = torch.randn_like(m1.weight)
    ema1.update(m1)
    state = ema1.state_dict()

    # Fresh EMA on a different (same-shape) model — load and check.
    m2 = _tiny_model()
    ema2 = EMAManager(m2, decay=0.5)
    ema2.load_state_dict(state)

    for name, tensor in state.items():
        assert torch.equal(ema2.state_dict()[name], tensor)


def test_ema_distributed_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    m1 = _tiny_model()
    ema1 = EMAManager(m1, decay=0.5)
    m1.weight.data = torch.randn_like(m1.weight)
    ema1.update(m1)
    expected = ema1.state_dict()

    checkpointer = Checkpointer(
        config=CheckpointingConfig(
            enabled=True,
            checkpoint_dir=tmp_path,
            model_save_format="torch_save",
            model_cache_dir=tmp_path,
            model_repo_id="bagel-ema-test",
            save_consolidated=False,
            is_peft=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )
    checkpointer.save_distributed_state(ema1, "ema", str(tmp_path))

    m2 = _tiny_model()
    ema2 = EMAManager(m2, decay=0.5)
    for shadow in ema2._shadow.values():
        shadow.zero_()

    checkpointer.load_distributed_state(ema2, "ema", str(tmp_path))

    for name, tensor in expected.items():
        assert torch.equal(ema2.state_dict()[name], tensor)


def test_ema_load_state_dict_strict_mismatch():
    """Strict mode catches name mismatches."""
    m = _tiny_model()
    ema = EMAManager(m, decay=0.5)
    bad_state = {"nonexistent": torch.zeros_like(m.weight)}
    with pytest.raises(KeyError):
        ema.load_state_dict(bad_state, strict=True)


def test_ema_decay_validation():
    m = _tiny_model()
    with pytest.raises(ValueError):
        EMAManager(m, decay=-0.1)
    with pytest.raises(ValueError):
        EMAManager(m, decay=1.1)

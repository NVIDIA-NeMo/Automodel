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

"""Smoke test for the Engine surface.

Verifies the class loads, can be constructed with bare configs, and exposes
the public methods / properties documented in the design doc. Distributed
behavior (build, forward_backward, etc.) is covered by separate integration
tests that require a GPU + torch.distributed init.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch


def _stub_cfg(**kw):
    """A minimal config stand-in (any duck-typed object with attributes works)."""
    return SimpleNamespace(**kw)


def test_engine_import_path():
    """Engine is importable both as a module and via the top-level alias."""
    import nemo_automodel
    from nemo_automodel.engine import Engine as A

    B = nemo_automodel.Engine
    assert A is B


def test_engine_skips_construction_when_model_is_none():
    """Engine.__init__ with model=None skips the build chain entirely."""
    from nemo_automodel.engine import Engine

    engine = Engine(
        Engine.Config(
            model=None,
            distributed=_stub_cfg(),
            optimizer=_stub_cfg(_target_="torch.optim.AdamW", lr=1e-4),
            lr_scheduler=None,
            max_grad_norm=1.0,
        )
    )
    assert engine.model is None
    assert engine.optimizer is None
    assert engine.lr_scheduler is None
    assert engine.mesh is None


def test_engine_introspection_defaults_when_unbuilt():
    """Introspection properties return safe defaults when not yet built."""
    from nemo_automodel.engine import Engine

    engine = Engine(
        Engine.Config(
            model=None,
            distributed=_stub_cfg(),
            optimizer=_stub_cfg(),
        )
    )
    assert engine.parts == []
    assert engine.pp_enabled is False
    # dp_rank / dp_size fall back to torch.distributed.get_world_size() when
    # initialized, else 0 / 1. In a unit test environment, dist is not init'd.
    if not torch.distributed.is_initialized():
        assert engine.dp_rank == 0
        assert engine.dp_size == 1
        assert engine.dp_group is None


def test_engine_methods_match_design():
    """Engine exposes the methods/properties documented in the design."""
    from nemo_automodel.engine import Engine

    expected_methods = {
        "forward_backward",
        "zero_grad",
        "optimizer_step",
        "lr_scheduler_step",
        "train_mode",
        "eval_mode",
        "save_checkpoint",
        "load_checkpoint",
        "export_weights",
        "to",
    }
    expected_properties = {"parts", "pp_enabled", "device", "dp_rank", "dp_size", "dp_group"}

    for name in expected_methods:
        attr = getattr(Engine, name, None)
        assert callable(attr), f"Engine.{name} should be a method"

    for name in expected_properties:
        attr = inspect.getattr_static(Engine, name, None)
        assert isinstance(attr, property), f"Engine.{name} should be a property"


def test_forward_backward_signature():
    """Signature of forward_backward matches the design."""
    from nemo_automodel.engine import Engine

    sig = inspect.signature(Engine.forward_backward)
    params = sig.parameters
    assert "batch" in params
    assert "loss_fn" in params
    assert params["num_microbatches"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["num_microbatches"].default == 1
    assert params["forward_only"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["forward_only"].default is False


def test_split_into_microbatches_basic():
    """The batch_split primitive splits tensors along dim 0."""
    from nemo_automodel.components.training.batch_split import split_into_microbatches

    batch = {
        "input_ids": torch.arange(8).view(8),
        "labels": torch.arange(8).view(8),
        "extra_dict": {"foo": torch.zeros(2)},  # opaque dict, should broadcast
        "name": "constant",  # non-tensor, should broadcast
    }
    chunks = split_into_microbatches(batch, 4)
    assert len(chunks) == 4
    for c in chunks:
        assert c["input_ids"].numel() == 2
        assert c["labels"].numel() == 2
        # Non-tensors broadcast unchanged.
        assert c["extra_dict"] is batch["extra_dict"]
        assert c["name"] == "constant"


def test_split_into_microbatches_one_is_passthrough():
    from nemo_automodel.components.training.batch_split import split_into_microbatches

    batch = {"input_ids": torch.arange(4)}
    chunks = split_into_microbatches(batch, 1)
    assert chunks == [batch]


def test_split_into_microbatches_rejects_too_many():
    from nemo_automodel.components.training.batch_split import split_into_microbatches

    with pytest.raises(ValueError):
        split_into_microbatches({"x": torch.arange(3)}, 4)

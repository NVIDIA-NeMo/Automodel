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

"""Unit tests for generic tensor-parallel plan resolution."""

from types import SimpleNamespace

import pytest
from torch import nn

from nemo_automodel.components.distributed.tp_plan import get_tp_plan


class _Model(nn.Module):
    pass


def test_caller_dict_takes_priority() -> None:
    model = _Model()
    plan = {"custom": object()}

    assert get_tp_plan(model, tp_shard_plan=plan) is plan


def test_model_local_factory_takes_priority_over_hf_plan() -> None:
    model = _Model()
    expected = {"model.layers.*": object()}
    model._nemo_tp_plan_factory = lambda target, *, sequence_parallel: expected

    assert get_tp_plan(model, hf_plan_resolver=lambda target: {"hf": object()}) is expected


def test_factory_failure_falls_back_to_hf_plan() -> None:
    model = _Model()

    def failing_factory(target, *, sequence_parallel):
        raise RuntimeError("sidecar unavailable")

    expected = {"hf": object()}
    model._nemo_tp_plan_factory = failing_factory

    assert get_tp_plan(model, hf_plan_resolver=lambda target: expected) is expected


def test_hf_plan_is_used_when_no_sidecar_is_attached() -> None:
    expected = {"model.layers.*": object()}

    assert get_tp_plan(_Model(), hf_plan_resolver=lambda target: expected) is expected


def test_missing_plan_is_a_clear_error_only_when_tp_is_enabled() -> None:
    with pytest.raises(ValueError, match="model-local `parallelizer.py`"):
        get_tp_plan(_Model(), tp_size=2, hf_plan_resolver=lambda target: (_ for _ in ()).throw(RuntimeError("none")))

    assert (
        get_tp_plan(_Model(), tp_size=1, hf_plan_resolver=lambda target: (_ for _ in ()).throw(RuntimeError("none")))
        == {}
    )


def test_named_plan_is_delegated_to_the_model_local_factory() -> None:
    model = _Model()
    expected = {"model.layers.*": object()}
    model._nemo_tp_plan_factory = lambda target, *, sequence_parallel: expected

    assert get_tp_plan(model, tp_shard_plan="model_default") is expected


def test_named_plan_without_a_model_factory_is_rejected() -> None:
    with pytest.raises(ValueError, match="model-local `_nemo_tp_plan_factory`"):
        get_tp_plan(_Model(), tp_shard_plan="model_default")


def test_import_path_plan_can_be_a_constant_or_factory(monkeypatch) -> None:
    from nemo_automodel.components.distributed import tp_plan

    expected = {"custom": object()}
    module = SimpleNamespace(PLAN=expected, build_plan=lambda: expected)
    monkeypatch.setattr(tp_plan.importlib, "import_module", lambda module_name: module)

    assert get_tp_plan(_Model(), tp_shard_plan="custom.module.PLAN") is expected
    assert get_tp_plan(_Model(), tp_shard_plan="custom.module.build_plan") is expected


def test_import_path_plan_must_resolve_to_a_dict(monkeypatch) -> None:
    from nemo_automodel.components.distributed import tp_plan

    monkeypatch.setattr(tp_plan.importlib, "import_module", lambda module_name: SimpleNamespace(PLAN=[]))

    with pytest.raises(ValueError, match="must be a dictionary"):
        get_tp_plan(_Model(), tp_shard_plan="custom.module.PLAN")

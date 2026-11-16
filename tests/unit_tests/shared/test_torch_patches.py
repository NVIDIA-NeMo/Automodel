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

import enum
import sys
import types

import pytest
import torch

from nemo_automodel.shared.torch_patches import disable_unsupported_cutedsl_overrides

_CUTEDSL_OPS = ["_fused_rms_norm", "_fused_rms_norm_backward"]


class _Arch(enum.Enum):
    """Targets known to nvidia-cutlass-dsl 4.6.2 around SM 10.x; it has none for SM 10.7 (VR200)."""

    sm_90a = (9, 0, "a")
    sm_100a = (10, 0, "a")
    sm_103a = (10, 3, "a")


@pytest.fixture
def deregistered(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install fake ``torch._native`` and CuTe DSL modules; return the record of deregistered DSLs."""
    calls: list[str] = []
    native = types.ModuleType("torch._native")
    native.registry = types.SimpleNamespace(get_dsl_operations=lambda dsl: _CUTEDSL_OPS if dsl == "cutedsl" else [])
    native.cutedsl_utils = types.SimpleNamespace(deregister_op_overrides=lambda: calls.append("cutedsl"))
    arch_module = types.ModuleType("cutlass.base_dsl.arch")
    arch_module.Arch = _Arch
    monkeypatch.setitem(sys.modules, "torch._native", native)
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.arch", arch_module)
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    return calls


def _use_capability(monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int]) -> None:
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)


def test_gpu_unknown_to_cute_dsl_falls_back_to_aten(deregistered, monkeypatch):
    _use_capability(monkeypatch, (10, 7))

    assert disable_unsupported_cutedsl_overrides(0) == _CUTEDSL_OPS
    assert deregistered == ["cutedsl"]


def test_gpu_known_to_cute_dsl_keeps_overrides(deregistered, monkeypatch):
    _use_capability(monkeypatch, (10, 0))

    assert disable_unsupported_cutedsl_overrides(0) == []
    assert deregistered == []


def test_cute_dsl_arch_override_selects_the_target(deregistered, monkeypatch):
    _use_capability(monkeypatch, (10, 7))
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")

    assert disable_unsupported_cutedsl_overrides(0) == []
    assert deregistered == []

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

"""Strict fp32 parameters pass through every family's state-dict adapter unchanged.

The adapters do no dtype work for ``_keep_in_fp32_modules_strict`` tensors: checkpoint
loads ``copy_`` into the model's fp32 parameters and the HF export pins them to F32 in
the checkpointer. This file checks, once for every family in
``test_fp32_strict_contract.FAMILIES``, that a bf16 copy of those tensors round-trips
native -> HF -> native with the same keys and the very same tensor objects. The adapters
receive copies because some (NemotronH) consume the mapping they are given.
"""

from __future__ import annotations

import pytest
import torch

from tests.unit_tests.models import test_fp32_strict_contract as strict_contract
from tests.unit_tests.models.test_fp32_strict_contract import Builder

# Families whose tiny builder cannot carry an adapter, or whose adapter legitimately
# rewrites a strict tensor; their own adapter tests cover them:
# - kimi_k3 reshapes ``A_log`` between the HF and native layouts,
# - qwen3_8_flash_next's adapter requires a PLE layer the tiny config does not have,
# - deepseek_v4 / deepseek_v41 builders construct the model without an adapter.
_COVERED_ELSEWHERE = {"kimi_k3", "qwen3_8_flash_next", "deepseek_v4", "deepseek_v41"}
FAMILIES = [family for family in strict_contract.FAMILIES if family.id not in _COVERED_ELSEWHERE]


@pytest.mark.parametrize("build", FAMILIES)
def test_strict_fp32_tensors_round_trip_through_the_adapter_unchanged(
    build: Builder, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = strict_contract._backend
    monkeypatch.setattr(
        strict_contract, "_backend", lambda **overrides: backend(enable_hf_state_dict_adapter=True, **overrides)
    )
    model, expected = build(torch.float32)
    adapter = model.state_dict_adapter
    native = model.state_dict()
    sample = {name: native[name].to(torch.bfloat16) for name in sorted(expected)}
    assert sample

    hf_state = adapter.to_hf(dict(sample))
    restored = adapter.from_hf(dict(hf_state))

    assert {id(tensor) for tensor in hf_state.values()} == {id(tensor) for tensor in sample.values()}
    assert restored.keys() == sample.keys()
    for name, tensor in sample.items():
        assert restored[name] is tensor, name

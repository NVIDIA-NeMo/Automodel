# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for Qwen3-Next fp32-aware GatedDeltaNet gate routing."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common.gated_delta_net_fp32 import HOLDER_NAME, isolate_fp32_params
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextFp32GatedDeltaNet


def _bare_gdn(num_v: int = 4, dtype: torch.dtype = torch.float32) -> Qwen3NextFp32GatedDeltaNet:
    """Build a Qwen3NextFp32GatedDeltaNet shell with only A_log/dt_bias (no full init)."""
    gdn = Qwen3NextFp32GatedDeltaNet.__new__(Qwen3NextFp32GatedDeltaNet)
    nn.Module.__init__(gdn)
    gdn.A_log = nn.Parameter(torch.ones(num_v, dtype=dtype))
    gdn.dt_bias = nn.Parameter(torch.zeros(num_v, dtype=dtype))
    return gdn


def _expected_gate(a: torch.Tensor) -> torch.Tensor:
    return -torch.ones(a.shape[-1]).exp() * F.softplus(torch.zeros(a.shape[-1]))


def test_compute_gate_fallback_without_holder():
    gdn = _bare_gdn()
    a = torch.zeros(4)
    g = gdn._compute_gate(a)
    assert g.dtype == torch.float32
    assert torch.allclose(g, _expected_gate(a), atol=1e-5)


def test_compute_gate_routes_through_holder():
    gdn = _bare_gdn()
    isolate_fp32_params(gdn)
    assert HOLDER_NAME in gdn._modules
    # A_log/dt_bias now resolve from the holder via __getattr__.
    assert gdn.A_log is gdn._fp32_params._parameters["A_log"]

    a = torch.zeros(4)
    g = gdn._compute_gate(a)
    assert g.dtype == torch.float32
    assert torch.allclose(g, _expected_gate(a), atol=1e-5)


def test_compute_gate_fp32_with_bf16_input():
    gdn = _bare_gdn()
    isolate_fp32_params(gdn)
    g = gdn._compute_gate(torch.zeros(4, dtype=torch.bfloat16))
    assert g.dtype == torch.float32

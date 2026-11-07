# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engram compilation scratch operations must not enter selective AC replay."""

from unittest.mock import Mock

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.models.deepseek_v41 import engram

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("masked", [False, True])
def test_compiled_gate_recompute_scratch_preserves_gradients(monkeypatch, masked: bool) -> None:
    """Model cold-autotune scratch work that occurs only on the second call."""
    torch.manual_seed(19)
    calls = 0

    def cold_compiled_gate(
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        eps: float,
        token_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Emulate compiler scratch around the unchanged mathematical gate.

        Args:
            hidden_states: Tensor of shape [batch, sequence, streams, hidden].
            key: Tensor of shape [batch, sequence, streams * hidden].
            value: Tensor of shape [batch, sequence, hidden].
            q_weight: Tensor of shape [streams, hidden].
            k_weight: Tensor of shape [streams, hidden].
            eps: RMS normalization epsilon.
            token_mask: Optional bool tensor of shape [batch, sequence].

        Returns:
            Tensor with the shape and dtype of hidden_states.
        """
        nonlocal calls
        calls += 1
        if calls == 2:
            # Inductor benchmarks a newly specialized reduction during AC
            # replay. The zero is a scratch buffer, not a model activation.
            torch.empty(32).zero_()
        return engram._engram_gate_core(hidden_states, key, value, q_weight, k_weight, eps, token_mask)

    monkeypatch.setattr(engram, "_engram_gate", engram._engram_gate_core)
    monkeypatch.setattr(engram, "_ENGRAM_GATE_COMPILED", False)
    monkeypatch.setattr(torch, "compile", Mock(return_value=cold_compiled_gate))
    engram.compile_engram_gate()
    parameters = (
        torch.randn(1, 7, 4, 16, requires_grad=True),
        torch.randn(1, 7, 64, requires_grad=True),
        torch.randn(1, 7, 16, requires_grad=True),
        torch.randn(4, 16, requires_grad=True),
        torch.randn(4, 16, requires_grad=True),
    )
    mask = torch.tensor([[True, False, True, True, False, True, True]]) if masked else None
    upstream = torch.randn_like(parameters[0])
    output = checkpoint(
        engram._engram_gate,
        *parameters,
        1e-20,
        mask,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(
            lambda *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE
        ),
    )
    gradients = torch.autograd.grad(output, parameters, upstream)
    assert calls == 2
    expected = engram._engram_gate_core(*parameters, 1e-20, mask)
    expected_gradients = torch.autograd.grad(expected, parameters, upstream)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    for actual, reference in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

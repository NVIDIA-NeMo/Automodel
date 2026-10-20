# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cold compiled Muse norm islands under selective activation checkpointing."""

import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.models.muse_spark.layers import _compiled_norm_core, _norm_residual_core


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
@pytest.mark.parametrize("input_scale", [False, True])
def test_cold_compiled_norm_with_selective_checkpoint(input_scale: bool) -> None:
    """Full-shape forward and recompute preserve all input and parameter gradients."""
    torch.manual_seed(81)
    stream = torch.randn(1, 4096, 8192, device="cuda", requires_grad=True)
    branch = torch.randn_like(stream, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(8192, device="cuda", requires_grad=True)
    gate = torch.randn(8192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn_like(stream)
    parameters = (stream, branch, weight, gate)

    compiled = _compiled_norm_core(_norm_residual_core)
    calls = []

    def forward(*args):
        calls.append("forward")
        return compiled(*args, input_scale=input_scale, norm_eps=1e-5, temperature=0.3, residual_eps=0.001)

    output = checkpoint(
        forward,
        *parameters,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(
            lambda *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE
        ),
    )
    gradients = torch.autograd.grad(output, parameters, upstream)
    assert len(calls) >= 2, "The test must execute checkpoint recomputation."
    # Evaluate the uncheckpointed oracle after the cold checkpointed execution:
    # warming the kernel before the test would hide first-autotune failures.
    reference = compiled(*parameters, input_scale=input_scale, norm_eps=1e-5, temperature=0.3, residual_eps=0.001)
    expected_gradients = torch.autograd.grad(reference, parameters, upstream)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA and Triton")
def test_compiled_expert_activation_with_nested_checkpoint() -> None:
    """Nested expert AC preserves SwiGLU input and routing gradients."""
    from nemo_automodel.components.models.muse_spark.layers import _checkpoint_recomputed_math
    from nemo_automodel.components.moe.megatron.moe_utils import weighted_bias_swiglu_impl

    torch.manual_seed(82)
    hidden = torch.randn(513, 8192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    probs = torch.rand(513, 1, device="cuda", requires_grad=True)
    activation = _checkpoint_recomputed_math(weighted_bias_swiglu_impl)
    calls = []

    def chunk(x, p):
        calls.append("chunk")
        return activation(x, p)

    def forward(x, p):
        return checkpoint(chunk, x, p, use_reentrant=False)

    output = checkpoint(
        forward,
        hidden,
        probs,
        use_reentrant=False,
        context_fn=lambda: create_selective_checkpoint_contexts(
            lambda *args, **kwargs: CheckpointPolicy.PREFER_RECOMPUTE
        ),
    )
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, (hidden, probs), upstream)
    assert len(calls) >= 2
    reference = activation(hidden, probs)
    expected = torch.autograd.grad(reference, (hidden, probs), upstream)
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Numerics tests for Muse Spark Open text primitives."""

import pytest
import torch

from nemo_automodel.components.models.muse_spark.layers import (
    MuseSparkInputScaleRMSNorm,
    MuseSparkResidual,
    MuseSparkRMSNorm,
    MuseSparkRotaryEmbedding,
    apply_rotary_embedding,
)


def test_text_rope_rotates_adjacent_pairs() -> None:
    """The released model uses GPT-J/interleaved, not NeoX/split-half, RoPE."""
    rotary = MuseSparkRotaryEmbedding(dim=4, theta=10_000.0)
    hidden = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
    positions = torch.tensor([[1]])

    output = apply_rotary_embedding(hidden, rotary(hidden, positions))

    angles = torch.tensor([1.0, 0.01])
    expected = torch.tensor(
        [
            1.0 * angles[0].cos() - 2.0 * angles[0].sin(),
            2.0 * angles[0].cos() + 1.0 * angles[0].sin(),
            3.0 * angles[1].cos() - 4.0 * angles[1].sin(),
            4.0 * angles[1].cos() + 3.0 * angles[1].sin(),
        ]
    ).reshape(1, 1, 1, 4)
    torch.testing.assert_close(output, expected)


@pytest.mark.parametrize("input_scale", [False, True])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.runtime_budget(
    30, hard_timeout=60, reason="Cold CPU Inductor compilation is part of this numerical regression"
)
def test_compiled_norm_matches_functional_reference(input_scale: bool, has_weight: bool) -> None:
    """FP32 normalization preserves gain placement, optional affine and gradients."""
    torch.manual_seed(83)
    cls = MuseSparkInputScaleRMSNorm if input_scale else MuseSparkRMSNorm
    module = cls(
        16,
        1e-5,
        dtype=torch.float32,
        has_weight=has_weight,
        zero_centered_gamma=True,
        gain_center=0.3,
        compile_norm=True,
    )
    if module.weight is not None:
        with torch.no_grad():
            module.weight.normal_(0.5, 0.2)
    x = torch.randn(2, 7, 16, requires_grad=True)
    upstream = torch.randn_like(x)
    effective = None if module.weight is None else module.weight + 0.3
    reference_input = x * effective if input_scale and effective is not None else x
    reference = torch.nn.functional.rms_norm(reference_input, (16,), eps=1e-5)
    if not input_scale and effective is not None:
        reference = reference * effective
    output = module(x)
    parameters = (x,) if module.weight is None else (x, module.weight)
    expected_gradients = torch.autograd.grad(reference, parameters, upstream, retain_graph=True)
    gradients = torch.autograd.grad(output, parameters, upstream)
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("input_scale", [False, True])
@pytest.mark.runtime_budget(
    30, hard_timeout=60, reason="Cold CPU Inductor compilation is part of this numerical regression"
)
def test_compiled_norm_residual_preserves_branch_rounding(input_scale: bool) -> None:
    """Fusing across modules preserves BF16 branch rounding and FP32 residual output."""
    from nemo_automodel.components.models.muse_spark.layers import _compiled_norm_core, _norm_residual_core

    torch.manual_seed(94)
    cls = MuseSparkInputScaleRMSNorm if input_scale else MuseSparkRMSNorm
    norm = cls(16, 1e-5, dtype=torch.float32, zero_centered_gamma=True, gain_center=0.25)
    residual = MuseSparkResidual(16, dtype=torch.float32)
    with torch.no_grad():
        norm.weight.normal_(0.8, 0.2)
        residual.gate.copy_(torch.linspace(-5, 5, 16))
    stream = torch.randn(2, 7, 16, requires_grad=True)
    branch = torch.randn(2, 7, 16, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn_like(stream)
    reference = residual(stream, norm(branch))
    fused = _compiled_norm_core(_norm_residual_core)
    output = fused(
        stream,
        branch,
        norm.effective_weight(),
        residual.gate,
        input_scale=input_scale,
        norm_eps=norm.eps,
        temperature=residual.temperature,
        residual_eps=residual.eps,
    )
    parameters = (stream, branch, norm.weight, residual.gate)
    expected_gradients = torch.autograd.grad(reference, parameters, upstream, retain_graph=True)
    gradients = torch.autograd.grad(output, parameters, upstream)
    assert output.dtype == torch.float32
    torch.testing.assert_close(output, reference, atol=2e-6, rtol=2e-5)
    for actual, expected in zip(gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, atol=0.035, rtol=0.02)


def test_checkpoint_recompute_preserves_compile_budget_across_threads() -> None:
    """Recomputation must inherit the caller budget and restore worker state."""
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import contextmanager

    from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM
    from tests.unit_tests.models.muse_spark.conftest import tiny_backend, tiny_muse_spark_config

    backend = tiny_backend()
    backend.compile_norm = True
    block = MuseSpark1xOmniForCausalLM(tiny_muse_spark_config(), backend=backend).model.layers[0]
    events = []

    @contextmanager
    def original_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    with torch._dynamo.config.patch(cache_size_limit=13, accumulated_cache_size_limit=51):
        factory = block.nemo_checkpoint_context_fn(lambda: (original_context(), original_context()))
        forward, recompute = factory()

    def run_recompute():
        with torch._dynamo.config.patch(cache_size_limit=1, accumulated_cache_size_limit=2):
            with forward:
                assert torch._dynamo.config.cache_size_limit == 13
            assert torch._dynamo.config.cache_size_limit == 1
            with pytest.raises(RuntimeError, match="test unwind"):
                with recompute:
                    assert torch._dynamo.config.cache_size_limit == 13
                    assert torch._dynamo.config.accumulated_cache_size_limit == 51
                    raise RuntimeError("test unwind")
            assert torch._dynamo.config.cache_size_limit == 1
            assert torch._dynamo.config.accumulated_cache_size_limit == 2

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(run_recompute).result()
    assert events == ["enter", "exit", "enter", "exit"]


@pytest.mark.parametrize("norm_class", [MuseSparkRMSNorm, MuseSparkInputScaleRMSNorm])
def test_compiled_empty_expert_norm_keeps_zero_gradients(norm_class) -> None:
    """An empty expert result must preserve its gain's zero gradient."""
    norm = norm_class(8, 1e-5, dtype=torch.float32, compile_norm=True)
    norm.reset_parameters()
    hidden = torch.empty(0, 8, requires_grad=True)

    output = norm(hidden)
    output.sum().backward()

    assert output.shape == hidden.shape
    assert output.dtype == hidden.dtype
    assert hidden.grad is not None and hidden.grad.shape == hidden.shape
    torch.testing.assert_close(norm.weight.grad, torch.zeros_like(norm.weight))


@pytest.mark.parametrize("packed", [False, True])
def test_hf_rope_preserves_bf16_products_and_gradients(packed: bool) -> None:
    """HF split-half arithmetic and converted adjacent-pair arithmetic agree."""
    torch.manual_seed(447)
    shape = (7, 3, 64) if packed else (1, 7, 3, 64)
    hidden = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    rotary = MuseSparkRotaryEmbedding(dim=64, theta=500_000.0)
    cos, sin = rotary(hidden, torch.arange(7).unsqueeze(0))
    output = apply_rotary_embedding(hidden, (cos, sin), upcast=False)
    # Independent split-half layout used by the released HF implementation.
    split = torch.cat((hidden[..., ::2], hidden[..., 1::2]), dim=-1)
    c = torch.cat((cos, cos), dim=-1)
    s = torch.cat((sin, sin), dim=-1)
    if packed:
        c, s = c.squeeze(0).unsqueeze(1), s.squeeze(0).unsqueeze(1)
    else:
        c, s = c.unsqueeze(2), s.unsqueeze(2)
    rotated_half = torch.cat((-split[..., 32:], split[..., :32]), dim=-1)
    expected_split = split * c + rotated_half * s
    expected = torch.stack((expected_split[..., :32], expected_split[..., 32:]), dim=-1).flatten(-2)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    upstream = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, hidden, upstream, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, hidden, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    # This case exercises the rounding boundary, not only a layout permutation.
    assert not torch.equal(output, apply_rotary_embedding(hidden, (cos, sin)))


def test_eager_attention_does_not_dispatch_to_sdpa(monkeypatch) -> None:
    """Eager is an explicit attention oracle rather than an alias for SDPA."""
    from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model

    torch.manual_seed(91)
    model = tiny_muse_spark_model().eval()
    ids = torch.arange(1, 9).unsqueeze(0)
    expected = model(input_ids=ids).logits
    model.backend.attn = "eager"

    def reject_sdpa(*args, **kwargs):
        raise AssertionError("eager attention dispatched to SDPA")

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", reject_sdpa)
    actual = model(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    actual.sum().backward()
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()


@pytest.mark.parametrize("document_length", [7, 8, 9])
@pytest.mark.parametrize("layer_index", [0, 3])
def test_packed_cp_attention_preserves_window_across_document_lengths(document_length, layer_index) -> None:
    """Each PP-local layer selects transport from the actual attention window."""
    from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model

    class RecordingAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cp_group = object()
            self.cp_global_ranks = [0, 1]
            self.cp_stream = None
            self.cp_comm_type = "p2p"
            self.window = None

        def set_context_parallel_group(self, group, ranks, stream, cp_comm_type):
            self.cp_comm_type = cp_comm_type

        def forward(self, query, key, value, **kwargs):
            self.window = kwargs["window_size"]
            return query

    model = tiny_muse_spark_model()
    attn = model.model.layers[layer_index].self_attn
    attn.backend.attn = "te"
    attn.attn_module = RecordingAttention()
    hidden = torch.randn(document_length, model.config.hidden_size)
    rotary = MuseSparkRotaryEmbedding(model.config.head_dim, theta=10_000.0)
    positions = rotary(hidden, torch.arange(document_length).unsqueeze(0))
    output = attn(
        hidden,
        positions,
        qkv_format="thd",
        cu_seqlens=torch.tensor([0, document_length], dtype=torch.int32),
        _muse_spark_thd_max_real_seqlen=document_length,
    )
    uses_window = layer_index == 0 and document_length > 8
    assert attn.attn_module.cp_comm_type == ("all_gather" if uses_window else "p2p")
    assert attn.attn_module.window == ((7, 0) if uses_window else (-1, 0))
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()

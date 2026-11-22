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

import json
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5 import layers as layers_module
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.layers import Step5Attention, Step5RMSNorm
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compile_norm", [False, True])
def test_norm_cpu_independent_oracle_and_gradients(dtype: torch.dtype, compile_norm: bool) -> None:
    torch.manual_seed(981)
    module = Step5RMSNorm(192, compile_norm=compile_norm)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192) + 0.00123)
    hidden = torch.randn(2, 17, 192, dtype=dtype, requires_grad=True)
    reference = F.rms_norm(hidden.float(), (192,), module.weight + 1, module.variance_epsilon).to(dtype)
    actual = module(hidden)
    upstream = torch.randn_like(hidden)
    expected_grads = torch.autograd.grad(reference, (hidden, module.weight), upstream)
    actual_grads = torch.autograd.grad(actual, (hidden, module.weight), upstream)
    for observed, expected in [(actual, reference), *zip(actual_grads, expected_grads)]:
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=2e-6)
    assert actual.dtype == dtype
    assert module.weight.dtype == torch.float32
    assert list(module.state_dict()) == ["weight"]


def test_norm_preserves_fp32_weight_and_residual_boundaries() -> None:
    torch.manual_seed(982)
    module = Step5RMSNorm(192)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192) + 0.00123)
    residual = torch.randn(2, 17, 192, dtype=torch.float32)
    correct = module(residual).bfloat16()
    early_cast = module(residual.bfloat16())
    rounded_weight = F.rms_norm(
        residual, (192,), module.weight.bfloat16().float() + 1, module.variance_epsilon
    ).bfloat16()
    assert not torch.equal(correct, early_cast)
    assert not torch.equal(correct, rounded_weight)
    assert torch.equal(module(residual), Step3p5RMSNorm.forward(module, residual))


def test_residual_norm_cast_keeps_input_and_parameter_gradients() -> None:
    torch.manual_seed(986)
    module = Step5RMSNorm(192, compile_norm=True)
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192) + 0.00123)
    residual = torch.randn(2, 17, 192, requires_grad=True)
    expected = Step3p5RMSNorm.forward(module, residual).bfloat16()
    actual = module(residual, output_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (residual, module.weight), upstream)
    actual_grads = torch.autograd.grad(actual, (residual, module.weight), upstream)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    assert actual.dtype == torch.bfloat16
    assert residual.dtype == module.weight.dtype == torch.float32


@pytest.mark.parametrize("span", [0, 32, 64, 192])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rope_complex_oracle_and_packed_resets(span: int, dtype: torch.dtype) -> None:
    torch.manual_seed(984)
    config = Step5Config()
    config.rope_theta = 10000.0
    positions = torch.tensor([[0, 1, 8191, 131071, 0, 11], [7, 0, 31, 0, 65535, 128001]])
    hidden = torch.randn(2, 6, 3, 192, dtype=dtype, requires_grad=True)
    actual = Step5RotaryEmbedding(config, 0, span=span)(hidden, positions)
    if span == 0:
        assert actual is hidden
        return
    # Complex multiplication is independent of the production half-split formula.
    # The checkpoint rounds the positive power before reciprocal. Negative
    # powers are mathematically equal but change FP32 phases at long positions.
    inverse = 1.0 / (10000.0 ** (torch.arange(0, span, 2, dtype=torch.float32) / span))
    phases = positions.float().unsqueeze(-1) * inverse
    table_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float32
    table = torch.complex(phases.cos().to(table_dtype).double(), phases.sin().to(table_dtype).double())
    half = span // 2
    values = torch.complex(hidden[..., :half].double(), hidden[..., half:span].double())
    rotated = values * table.unsqueeze(-2)
    expected = torch.cat((rotated.real, rotated.imag, hidden[..., span:].double()), -1).to(dtype)
    tolerance = 0.03125 if dtype == torch.bfloat16 else 5e-7
    torch.testing.assert_close(actual, expected, rtol=0.003 if dtype == torch.bfloat16 else 5e-7, atol=tolerance)
    assert torch.equal(actual[..., span:], hidden[..., span:])
    if span:
        assert torch.equal(actual[0, 0], hidden[0, 0])
        assert torch.equal(actual[0, 4], hidden[0, 4])
    upstream = torch.randn_like(hidden)
    observed_grad = torch.autograd.grad(actual, hidden, upstream)[0]
    expected_grad = torch.autograd.grad(expected, hidden, upstream)[0]
    torch.testing.assert_close(observed_grad, expected_grad, rtol=0.005, atol=tolerance)


def test_mixed_rotary_shapes_share_bounded_compile_entries(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, str], None]
) -> None:
    """Count Dynamo shape guards on metadata tensors without requiring CUDA.

    No reset occurs between cases. This catches compiler-cache exhaustion but
    does not replace the matching functional test using actual CUDA kernels.
    """
    from nemo_automodel.components.models.step5 import rope as implementation

    counts = {"full": 0, "partial": 0, "single_head": 0}

    def full_backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor | torch.SymInt]) -> Callable:
        counts["full"] += 1
        return graph.forward

    def partial_backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor | torch.SymInt]) -> Callable:
        counts["partial"] += 1
        return graph.forward

    def single_head_backend(graph: torch.fx.GraphModule, inputs: list[torch.Tensor | torch.SymInt]) -> Callable:
        counts["single_head"] += 1
        return graph.forward

    for name, backend in [("full", full_backend), ("partial", partial_backend), ("single_head", single_head_backend)]:
        attribute = f"_apply_{name}_rotary_compiled"
        original = getattr(implementation, attribute)._torchdynamo_orig_callable
        monkeypatch.setattr(
            implementation, attribute, torch.compile(original, backend=backend, fullgraph=True, dynamic=True)
        )
    for sequence in [4096, 8192, 0, 1]:
        for span, heads, dim in [
            (64, 64, 192),
            (64, 4, 192),
            (192, 64, 192),
            (192, 4, 192),
            (32, 16, 256),
            (32, 1, 256),
        ]:
            for grad in [True, False]:
                hidden = torch.empty(1, sequence, heads, dim, device="meta", dtype=torch.bfloat16, requires_grad=grad)
                cos = torch.empty(1, sequence, 1, span // 2, device="meta", dtype=torch.float32)
                with torch.set_grad_enabled(grad):
                    output = implementation._apply_rotary_compiled(hidden, cos, cos, span)
                assert output.shape == hidden.shape
                if sequence == 0:
                    assert output is hidden
    assert 0 < counts["full"] <= 8
    assert 0 < counts["partial"] <= 8
    assert 0 < counts["single_head"] <= 8
    record_property("compile_entries", json.dumps(counts))


@pytest.mark.parametrize("span", [0, 64, 192])
@pytest.mark.parametrize("sequence", [0, 1, 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_paired_rope_reuses_tables_and_preserves_strided_qkv_gradients(
    span: int, sequence: int, dtype: torch.dtype
) -> None:
    torch.manual_seed(987)
    config = Step5Config()
    rope = Step5RotaryEmbedding(config, 0, span=span)
    source = torch.randn(2, sequence, 4, 18, 192, dtype=dtype)
    expected_input = source.clone().requires_grad_()
    actual_input = source.clone().requires_grad_()
    positions = torch.arange(sequence).expand(2, -1) + 131071
    positions[:, sequence // 2 :] = torch.arange(sequence - sequence // 2)

    def qk(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value[..., :16, :].reshape(2, sequence, 64, 192), value[..., 16, :]

    expected_q, expected_k = qk(expected_input)
    actual_q, actual_k = qk(actual_input)
    with patch.object(rope, "_tables", wraps=rope._tables) as tables:
        expected = rope(expected_q, positions), rope(expected_k, positions)
        assert tables.call_count == (2 if sequence and span else 0)
        tables.reset_mock()
        actual = rope.forward_pair(actual_q, actual_k, positions)
        assert tables.call_count == (1 if sequence and span else 0)
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    upstream = tuple(torch.randn_like(value) for value in expected)
    expected_grad = torch.autograd.grad(expected, expected_input, upstream)[0]
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    if sequence == 0 or span == 0:
        assert actual[0] is actual_q and actual[1] is actual_k


def test_paired_rope_rejects_different_rounding_dtypes() -> None:
    rope = Step5RotaryEmbedding(Step5Config(), 0, span=64)
    query = torch.randn(1, 1, 64, 192, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 4, 192, dtype=torch.float32)
    with pytest.raises(ValueError, match="matching Q/K"):
        rope.forward_pair(query, key, torch.ones(1, 1, dtype=torch.long))


class _StopAfterQueryNorm(Exception):
    """End the component observation before attention or rotary computation."""


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compile_norm", [False, True])
@pytest.mark.parametrize("batch,sequence", [(1, 5), (2, 7)])
def test_attention_query_norm_preserves_projection_view_and_math(
    dtype: torch.dtype, compile_norm: bool, batch: int, sequence: int
) -> None:
    torch.manual_seed(1137)
    groups, heads, dim, width = 2, 4, 192, 32
    config = Step5Config(
        hidden_size=width,
        num_hidden_layers=4,
        num_attention_heads=groups * heads,
        num_attention_groups=groups,
        head_dim=dim,
        num_nextn_predict_layers=0,
        moe_layer_list=[3],
        sparse_config={"enabled": False},
    )
    attention = Step5Attention(
        config, 0, BackendConfig(attn="eager", linear="torch", compile_norm=compile_norm), dtype=dtype
    )
    with torch.no_grad():
        attention.q_norm.weight.copy_(torch.linspace(-0.7, 0.7, dim) + 0.00123)
    hidden = torch.randn(batch, sequence, width, dtype=dtype, requires_grad=True)
    observed: dict[str, torch.Tensor] = {}

    def projection_hook(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """Record the actual projection storage.

        Args:
            module: Native projection module.
            args: Hidden input [batch, sequence, hidden].
            output: Interleaved QKV [batch, sequence, groups * (heads + 2) * dim].
        """
        observed["projection"] = output

    def norm_hook(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """Record the norm boundary without running attention.

        Args:
            module: Native query norm module.
            args: Query input [batch, sequence, groups, heads, dim].
            output: Normalized query with the same shape.
        """
        observed["input"] = args[0]
        observed["output"] = output
        raise _StopAfterQueryNorm

    positions = torch.arange(sequence).expand(batch, -1)
    padding = torch.zeros(batch, sequence, dtype=torch.bool)
    with attention.qkv_proj.register_forward_hook(projection_hook), attention.q_norm.register_forward_hook(norm_hook):
        with pytest.raises(_StopAfterQueryNorm):
            attention(hidden, positions, padding, positions, padding)

    # This storage assertion fails on the old pre-norm reshape, even though
    # that implementation has the same mathematical forward result.
    assert observed["input"].untyped_storage().data_ptr() == observed["projection"].untyped_storage().data_ptr()
    assert observed["input"].shape == (batch, sequence, groups, heads, dim)
    assert not observed["input"].is_contiguous()
    assert observed["output"].is_contiguous()
    actual = observed["output"].flatten(2, 3)
    assert actual.untyped_storage().data_ptr() == observed["output"].untyped_storage().data_ptr()
    assert actual.dtype == dtype
    assert attention.q_norm.weight.dtype == torch.float32

    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_projection_weight = attention.qkv_proj.weight.detach().clone().requires_grad_()
    reference_scale = attention.q_norm.weight.detach().clone().requires_grad_()
    reference_projection = F.linear(reference_hidden, reference_projection_weight)
    reference_qkv = reference_projection.view(batch, sequence, groups, heads + 2, dim)
    # Independent public RMSNorm computes the normalized feature axis in FP32.
    expected = (
        F.rms_norm(
            reference_qkv[..., :heads, :].float(), (dim,), reference_scale + 1, attention.q_norm.variance_epsilon
        )
        .to(dtype)
        .flatten(2, 3)
    )
    upstream = torch.randn_like(expected)
    actual_grads = torch.autograd.grad(actual, (hidden, attention.qkv_proj.weight, attention.q_norm.weight), upstream)
    reference_grads = torch.autograd.grad(
        expected, (reference_hidden, reference_projection_weight, reference_scale), upstream
    )
    for actual_value, reference_value in zip((actual, *actual_grads), (expected, *reference_grads)):
        torch.testing.assert_close(actual_value, reference_value, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_compiled_query_entry_keeps_group_stride_without_materialization(
    dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    # This checks Python dispatch and layout only. Actual compiled CUDA execution
    # and absence of CUDA clone operations have separate functional coverage.
    observed: list[torch.Tensor] = []

    def compiled_entry(
        hidden: torch.Tensor, weight: torch.Tensor, eps: float, output_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Observe the compile boundary without launching CUDA.

        Args:
            hidden: Fake CUDA grouped queries [token_groups, heads, channels].
            weight: Fake FP32 additive scale [channels].
            eps: Variance epsilon.
            output_dtype: Output dtype.

        Returns:
            Contiguous fake query output with hidden's shape and an inverse RMS
            tensor with shape [token_groups, heads, 1].
        """
        observed.append(hidden)
        return torch.empty_like(hidden, memory_format=torch.contiguous_format), hidden.new_empty(
            (*hidden.shape[:-1], 1)
        )

    monkeypatch.setattr(layers_module, "_grouped_rms_norm", compiled_entry)
    with FakeTensorMode(), torch.device("cuda"):
        module = Step5RMSNorm(192, compile_norm=True)
        projection = torch.empty(2, 7, 4, 18, 192, dtype=dtype)
        query = projection[..., :16, :]
        output = module(query)
    assert len(observed) == 1
    assert observed[0].shape == (2 * 7 * 4, 16, 192)
    assert observed[0].stride() == (18 * 192, 192, 1)
    assert observed[0]._base is projection
    assert output.shape == query.shape
    assert output.is_contiguous()


@pytest.mark.parametrize("sequence", [0, 1, 7])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_query_norm_empty_singleton_and_replay(sequence: int, dtype: torch.dtype) -> None:
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(1138)
    module = Step5RMSNorm(192, compile_norm=True)
    projection = torch.randn(2, sequence, 4, 18, 192, dtype=dtype, requires_grad=True)
    query = projection[..., :16, :]
    actual = module(query)
    upstream = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (projection, module.weight), upstream)
    with torch.no_grad():
        inference = module(query)
    replay = checkpoint(module, query, use_reentrant=False)
    replay_gradients = torch.autograd.grad(replay, (projection, module.weight), upstream)
    for observed, expected in zip((inference, replay, *replay_gradients), (actual, actual, *gradients)):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert actual.is_contiguous()
    # The K/V rows are excluded from query normalization, including backward.
    assert torch.count_nonzero(gradients[0][..., 16:, :]) == 0

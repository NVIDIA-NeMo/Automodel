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

"""QKV-slice layout and independent RMSNorm forward/backward controls."""

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5 import norm as norm_module
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.layers import Step5Attention
from nemo_automodel.components.models.step5.norm import Step5RMSNorm


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

    monkeypatch.setattr(norm_module, "_grouped_rms_norm", compiled_entry)
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

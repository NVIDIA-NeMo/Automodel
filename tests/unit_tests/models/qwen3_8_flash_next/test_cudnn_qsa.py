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

"""CPU contracts for the optional cuDNN QSA adapter."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next import cudnn_qsa
from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextQSAAttention
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    gathered_qsa_gqa_attention,
    qsa_gqa_attention,
    select_qsa_cuda_kernel,
)

pytestmark = pytest.mark.cpu_only


def test_cpu_dispatch_preserves_oracle_and_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cudnn_qsa, "_load_cudnn", lambda: pytest.fail("CPU must not import cuDNN"))
    torch.manual_seed(17)
    inputs = [torch.randn(2, 7, h, 4, requires_grad=True) for h in (4, 2, 2)]
    routes = torch.randint(-1, 7, (2, 7, 9))
    actual = qsa_gqa_attention(*inputs, routes, cuda_kernel=select_qsa_cuda_kernel("cudnn"))
    expected = gathered_qsa_gqa_attention(*inputs, routes)
    dy = torch.randn_like(actual)
    for a, b in zip(
        (actual, *torch.autograd.grad(actual, inputs, dy)), (expected, *torch.autograd.grad(expected, inputs, dy))
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_pack_indices_preserves_batch_boundaries_and_padding() -> None:
    routes = torch.full((2, 7, 7), -1, dtype=torch.int64)
    routes[0, 6] = torch.arange(7)
    routes[1, 3, :4] = torch.arange(4)
    routes[1, 4, :4] = torch.tensor([-(1 << 40), 1 << 40, 7, -2])
    original = routes.clone()
    packed = cudnn_qsa._pack_indices(routes, 7)
    expected = torch.full((14, 8), -1, dtype=torch.int32)
    expected[6, :7] = torch.arange(7)
    expected[10, :4] = torch.arange(7, 11)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(routes, original, rtol=0, atol=0)
    assert packed.is_contiguous() and packed.data_ptr() != routes.data_ptr()


def test_optional_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    cudnn_qsa._load_cudnn.cache_clear()
    monkeypatch.setattr(cudnn_qsa, "safe_import_from", lambda *args: (False, None))
    with pytest.raises(ImportError, match="nvidia-cudnn-frontend>=1.32.1"):
        cudnn_qsa._load_cudnn()
    cudnn_qsa._load_cudnn.cache_clear()


@pytest.mark.parametrize(
    "case", ["rank", "cp", "kv", "heads", "six_heads", "dim", "empty", "routes", "width", "dtype", "ids", "cpu"]
)
def test_reject_unsupported_metadata(case: str) -> None:
    q = torch.empty(1, 7, 24, 256, dtype=torch.bfloat16)
    k = torch.empty(1, 7, 2, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    routes = torch.zeros(1, 7, 7, dtype=torch.int32)
    error, match = ValueError, "same CUDA device"
    if case == "rank":
        q, match = q[0], "layout"
    elif case == "cp":
        q, match = q[:, :3], "CP local-Q"
    elif case == "kv":
        v, match = v[:, :3], "matching"
    elif case == "heads":
        q, match = q[:, :, :8], "head counts"
    elif case == "six_heads":
        q, k, v, match = q[:, :, :6], k[:, :, :1], v[:, :, :1], "head counts"
    elif case == "dim":
        q, match = q[..., :128], "head_dim"
    elif case == "empty":
        q, k, v, match = q[:, :0], k[:, :0], v[:, :0], "nonempty"
    elif case == "routes":
        routes, match = routes[:, :3], "routes must have shape"
    elif case == "width":
        routes, match = routes.expand(1, 7, 7).repeat(1, 1, 300), "route width"
    elif case == "dtype":
        q, error, match = q.float(), TypeError, "BF16"
    elif case == "ids":
        routes, error, match = routes.float(), TypeError, "signed"
    with pytest.raises(error, match=match):
        cudnn_qsa.cudnn_sparse_gqa_attention(q, k, v, routes)


def test_autograd_saves_native_fp32_output(monkeypatch: pytest.MonkeyPatch) -> None:
    q = torch.randn(3, 6, 256, requires_grad=True)
    k = torch.randn(3, 1, 256, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    indices = torch.zeros(3, 4, dtype=torch.int32)
    saved_fp32 = torch.full_like(q, 1.0001)
    saved_lse = torch.randn(6, 3)

    def forward(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, *, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return distinguishable BF16-rounded and native FP32 saved outputs.

        Args:
            q: Query [tokens, query_heads, 256].
            k: Key [tokens, 1, 256].
            v: Value with k's layout.
            indices: Integer selection [tokens, selected_tokens].
            scale: Score multiplier.

        Returns:
            Rounded output and native output with q's layout, and LSE
            [query_heads, tokens].
        """
        return saved_fp32.to(torch.bfloat16).float(), saved_fp32, saved_lse

    def backward(
        dy: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out32: torch.Tensor,
        lse: torch.Tensor,
        ids: torch.Tensor,
        *,
        scale: float,
        selection_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Verify saved-state ownership and return known gradients.

        Args:
            dy: Output gradient [tokens, query_heads, 256].
            q: Query with dy's layout.
            k: Key [tokens, 1, 256].
            v: Value with k's layout.
            out32: Native FP32 output with q's layout.
            lse: Natural-log normalizer [query_heads, tokens].
            ids: Integer selection [tokens, selected_tokens].
            scale: Score multiplier.
            selection_mode: Requested public backward mode.

        Returns:
            Gradients matching q, k and v respectively.
        """
        assert out32 is saved_fp32 and lse is saved_lse and ids is indices
        assert not torch.equal(out32, out32.to(torch.bfloat16).float())
        assert scale == 0.125 and selection_mode == "general" and dy.is_contiguous()
        return dy, torch.ones_like(k), torch.full_like(v, 2)

    monkeypatch.setattr(cudnn_qsa, "_load_cudnn", lambda: (forward, backward))
    dy = torch.randn_like(q)
    out = cudnn_qsa._CudnnQSA.apply(q, k, v, indices, 0.125)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dy)
    for a, b in ((dq, dy), (dk, torch.ones_like(k)), (dv, torch.full_like(v, 2))):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("ratio,budget", [(2, 8), (4, 2052), (4, 8)])
def test_model_configuration_and_cp_guard(ratio: int, budget: int) -> None:
    config = Qwen3_8_FlashNextTextConfig(
        hidden_size=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        ple_layer_ids=[],
    )
    backend = Qwen3_8_FlashNextBackendConfig(
        attn="cudnn", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch", rope_fusion=False
    )
    if ratio != 4 or budget > 2048:
        with pytest.raises(ValueError, match="indexer_compress_ratio=4"):
            Qwen3_8_FlashNextQSAAttention(config, 0, backend)
    else:
        layer = Qwen3_8_FlashNextQSAAttention(config, 0, backend)
        assert layer.backend.attn == "cudnn" and not any(p.requires_grad for p in layer.indexer.parameters())
        with pytest.raises(ValueError, match="does not support context parallelism"):
            layer.setup_cp_attention(SimpleNamespace(size=lambda: 2))

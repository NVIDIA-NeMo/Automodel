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
"""`backend.attn="cudnn"` routes DeepSeek-V4.1's sparse attention to the shared cuDNN kernels."""

from __future__ import annotations

import pytest
import torch

import nemo_automodel.components.models.deepseek_v4.optimized_kernels as kernels
import nemo_automodel.components.models.deepseek_v41.attention as attention_mod
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention, DeepseekV41AttentionState
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig


def _config(**overrides) -> DeepseekV41TextConfig:
    values = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        sliding_window=2,
        compress_ratios=[0, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4],
        candidate_source_layer_id=3,
        candidate_block_size=2,
        candidate_topk_blocks=2,
        engram_layer_ids=[],
        dtype="float32",
        rope_scaling={},
    )
    values.update(overrides)
    return DeepseekV41TextConfig(**values)


def _backend(attn: str) -> BackendConfig:
    return BackendConfig(attn=attn, linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")


def test_cudnn_branch_flattens_batches_into_global_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    """The batched DSV4 call becomes one token axis with per-batch key offsets; -1 stays masked."""
    seen: dict = {}

    def fake_helper(q, kv_latent, topk_indices, softmax_scale, *, attn_sink, **kwargs):
        seen.update(q=q, kv=kv_latent, idx=topk_indices, scale=softmax_scale, sink=attn_sink)
        return torch.zeros(q.shape[0], q.shape[1], 512, dtype=q.dtype)

    import nemo_automodel.components.models.common.cudnn_sparse_attention as shared_cudnn

    monkeypatch.setattr(shared_cudnn, "cudnn_sparse_attention", fake_helper)
    monkeypatch.setattr(shared_cudnn, "is_cudnn_sparse_attention_available", lambda: True)
    batch, sequence, heads, dim, kv_sequence = 2, 3, 4, 512, 5
    q = torch.randn(batch, sequence, heads, dim).to(torch.bfloat16)
    kv = torch.randn(batch, kv_sequence, dim).to(torch.bfloat16)
    sinks = torch.arange(heads, dtype=torch.float32)
    idx = torch.tensor([[[0, 1, -1], [1, 2, -1], [4, -1, -1]], [[0, -1, -1], [2, 3, -1], [1, 4, 3]]], dtype=torch.int32)
    out = kernels.dsv4_sparse_attention(q, kv, sinks, idx, 0.25, backend="cudnn", reference_rounding=True)
    assert out.shape == (batch, sequence, heads, 512)
    assert seen["q"].shape == (batch * sequence, heads, dim) and seen["kv"].shape == (batch * kv_sequence, 1, dim)
    assert seen["scale"] == 0.25
    torch.testing.assert_close(seen["sink"], sinks)
    expected = idx.clone().to(torch.int64)
    expected[1] = torch.where(expected[1] >= 0, expected[1] + kv_sequence, expected[1])
    torch.testing.assert_close(seen["idx"], expected.reshape(batch * sequence, 1, -1).to(torch.int32))


def test_cudnn_backend_requires_the_optional_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    import nemo_automodel.components.models.common.cudnn_sparse_attention as shared_cudnn

    monkeypatch.setattr(shared_cudnn, "is_cudnn_sparse_attention_available", lambda: False)
    q = torch.zeros(1, 2, 4, 512, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="backend.attn='cudnn' requires"):
        kernels.dsv4_sparse_attention(
            q,
            torch.zeros(1, 2, 512, dtype=torch.bfloat16),
            torch.zeros(4),
            torch.zeros(1, 2, 2, dtype=torch.int32),
            0.1,
            backend="cudnn",
        )


def test_attention_accepts_cudnn_rejects_dropout_and_matches_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the kernel stubbed by the torch reference, the cuDNN route reproduces the eager (dense-mask) layer."""
    with pytest.raises(ValueError, match="attention_dropout=0"):
        DeepseekV41Attention(_config(attention_dropout=0.1), 0, _backend("cudnn"))
    torch.manual_seed(3)
    layer = DeepseekV41Attention(_config(), 0, _backend("cudnn"))
    with torch.no_grad():
        layer.attn_sink.copy_(torch.tensor([-3.0, -0.5, 1.0, 4.0]))
    eager = DeepseekV41Attention(_config(), 0, _backend("eager"))
    eager.load_state_dict(layer.state_dict())
    calls: list[str] = []

    def reference_kernel(q, kv, sinks, topk_idxs, sm_scale, *, backend, reference_rounding=False):
        calls.append(backend)
        return kernels.sparse_attention_torch(q.float(), kv.float(), sinks, topk_idxs.long(), sm_scale).to(q.dtype)

    monkeypatch.setattr(attention_mod, "dsv4_sparse_attention", reference_kernel)
    hidden = torch.randn(2, 6, 16)
    positions = torch.arange(6)[None]
    actual = layer(hidden, position_ids=positions, state=DeepseekV41AttentionState()).hidden_states
    expected = eager(hidden, position_ids=positions, state=DeepseekV41AttentionState()).hidden_states
    assert calls == ["cudnn"]
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

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

"""The DSA indexer must rotate the leading rope dims of Q/K with half-split RoPE.

The indexer selection cases below exercise sequences longer than top-k. Direct helper
cases also check the BF16 precision path against an independent complex rotation.

A wrong rope layout only changes which keys are selected once the sequence is longer than
``index_topk`` (below that the indexer keeps every causal key), so every selection case uses
``seq_len > index_topk`` and compares the selected key sets of the late positions.
"""

import pytest
import torch
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config as HFDeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32Indexer as HFDeepseekV32Indexer

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.rope_utils import apply_rotary_emb_half_split
from nemo_automodel.components.models.deepseek_v32 import layers as dsv32_layers
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import DeepseekV32Indexer

HIDDEN, Q_LORA_RANK, N_HEADS, HEAD_DIM, ROPE_DIM, TOPK = 32, 16, 2, 8, 4, 3
BATCH, SEQ = 2, 11


@pytest.fixture(autouse=True)
def identity_rotate_activation(monkeypatch):
    # The Hadamard rotation is orthonormal (dot products unchanged) and runs in bf16 only; dropping it
    # keeps the comparison exact in fp32. HF's indexer skips it for the same reason.
    monkeypatch.setattr(dsv32_layers, "_rotate_activation", lambda x: x)


@pytest.fixture
def indexer():
    torch.manual_seed(0)
    config = DeepseekV32Config(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA_RANK,
        index_n_heads=N_HEADS,
        index_head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        index_topk=TOPK,
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch")
    module = DeepseekV32Indexer(config, backend).float()
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p))
    return module


def _angles() -> torch.Tensor:
    inv_freq = torch.tensor([0.9, 0.35])  # ROPE_DIM // 2 frequencies, deliberately non-monotone in pair index
    return torch.arange(SEQ).float()[:, None] * inv_freq  # [S, ROPE_DIM // 2]


def _inputs():
    torch.manual_seed(1)
    x = torch.randn(BATCH, SEQ, HIDDEN)
    q_resid = torch.randn(BATCH, SEQ, Q_LORA_RANK)
    angles = _angles().expand(BATCH, -1, -1)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # [B, S, ROPE_DIM // 2] complex
    causal = torch.full((SEQ, SEQ), float("-inf")).triu(1)
    return x, q_resid, angles, freqs_cis, causal


def _late_sets(indices: torch.Tensor) -> torch.Tensor:
    # Positions >= TOPK are the only ones whose selection depends on the scores; compare them as sets.
    return indices[:, TOPK:].sort(dim=-1).values


def test_selected_keys_match_independent_half_split_rotation(indexer):
    x, q_resid, angles, freqs_cis, causal = _inputs()
    # Independent reference: explicit rotation matrices on the leading rope dims, pairing j with j + d/2.
    rotation = torch.eye(HEAD_DIM).repeat(BATCH, SEQ, 1, 1)
    half = ROPE_DIM // 2
    for j in range(half):
        cos, sin = angles[..., j].cos(), angles[..., j].sin()
        rotation[..., j, j] = cos
        rotation[..., j + half, j + half] = cos
        rotation[..., j, j + half] = -sin
        rotation[..., j + half, j] = sin
    q = indexer.wq_b(q_resid).reshape(BATCH, SEQ, N_HEADS, HEAD_DIM)
    k = indexer.k_norm(indexer.wk(x))
    q = torch.einsum("bsij,bshj->bshi", rotation, q)
    k = torch.einsum("bsij,bsj->bsi", rotation, k)
    scores = torch.einsum("bshd,btd->bsht", q, k).relu()
    weights = indexer.weights_proj(x) * N_HEADS**-0.5 * HEAD_DIM**-0.5
    scores = torch.einsum("bsht,bsh->bst", scores, weights) + causal
    expected = scores.topk(TOPK, dim=-1).indices

    actual = indexer(x, q_resid, freqs_cis, attention_mask=causal[None, None])
    torch.testing.assert_close(_late_sets(actual), _late_sets(expected))


def test_selected_keys_match_transformers_indexer(indexer):
    x, q_resid, angles, freqs_cis, causal = _inputs()
    hf_config = HFDeepseekV32Config(
        hidden_size=HIDDEN,
        q_lora_rank=Q_LORA_RANK,
        index_n_heads=N_HEADS,
        index_head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE_DIM,
        index_topk=TOPK,
    )
    hf_indexer = HFDeepseekV32Indexer(hf_config, layer_idx=0).float()
    hf_indexer.load_state_dict(indexer.state_dict())
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)  # HF rotate_half convention: [B, S, ROPE_DIM]
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    position_ids = torch.arange(SEQ).expand(BATCH, -1)
    expected = hf_indexer(x, q_resid, (cos, sin), attention_mask=None, position_ids=position_ids)

    actual = indexer(x, q_resid, freqs_cis, attention_mask=causal[None, None])
    torch.testing.assert_close(_late_sets(actual), _late_sets(expected.long()))


def test_thd_layout_selects_the_same_keys_as_bshd(indexer):
    x, q_resid, angles, freqs_cis, causal = _inputs()
    bshd = indexer(x, q_resid, freqs_cis, attention_mask=causal[None, None])
    for b in range(BATCH):
        thd = indexer(x[b], q_resid[b], freqs_cis[b], attention_mask=causal)
        torch.testing.assert_close(_late_sets(thd[None]), _late_sets(bshd[b : b + 1]))


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("qkv_format,with_heads", [("bshd", True), ("bshd", False), ("thd", True), ("thd", False)])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_half_split_rotation_preserves_fp32_arithmetic(
    device: str, dtype: torch.dtype, qkv_format: str, with_heads: bool, noncontiguous: bool
) -> None:
    """Check Q/K forward and input gradients against an independent complex rotation."""
    torch.manual_seed(19)
    shape = (2, 11) if qkv_format == "bshd" else (11,)
    angles = torch.arange(11, device=device, dtype=torch.float32)[:, None] * torch.linspace(
        0.13, 1.37, 32, device=device
    )
    freqs = torch.polar(torch.ones_like(angles), angles)
    if qkv_format == "bshd":
        freqs = freqs.expand(2, -1, -1)
    if with_heads:
        shape += (3,)
    storage = torch.randn(*shape, 128 if noncontiguous else 64, device=device, dtype=dtype)
    x = (storage[..., ::2] if noncontiguous else storage).requires_grad_()
    reference_input = x.detach().clone().requires_grad_()
    # DeepSeek official non-interleaved layout: the two halves are the real and
    # imaginary parts of each complex pair. Preserve the FP32 frequency table
    # and round only after multiplication, as in vLLM's CUDA RoPE arithmetic.
    reference_float = reference_input.float()
    complex_input = torch.complex(reference_float[..., :32], reference_float[..., 32:])
    rotated = complex_input * (freqs.unsqueeze(-2) if with_heads else freqs)
    expected = torch.cat((rotated.real, rotated.imag), dim=-1).to(dtype)

    actual = apply_rotary_emb_half_split(x, freqs, qkv_format)
    assert actual.shape == x.shape
    assert actual.dtype == dtype
    # CUDA complex multiply may fuse FP32 arithmetic; allow its near-zero
    # cancellation error, but no BF16-relative tolerance. The old BF16
    # arithmetic differs by 0.015625 and fails this bound.
    relative_tolerance = 0.0 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=relative_tolerance)

    upstream_grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, x, upstream_grad)[0]
    expected_grad = torch.autograd.grad(expected, reference_input, upstream_grad)[0]
    torch.testing.assert_close(actual_grad, expected_grad, atol=1e-6, rtol=relative_tolerance)

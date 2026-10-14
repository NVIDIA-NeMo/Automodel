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

import pytest
import torch
from torch._dynamo.utils import counters
from transformers.models.qwen3_moe.modeling_qwen3_moe import apply_rotary_pos_emb

from nemo_automodel.components.models.gpt_oss.rope_utils import (
    RotaryEmbedding,
    apply_rotary_emb_qk,
    position_ids_to_freqs_cis,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled RoPE requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("rotary_dim", [64, 128])
def test_compiled_rope_matches_hf_forward_and_backward(dtype, packed, rotary_dim):
    # Exercise every specialization rather than Dynamo's eager cache-limit fallback.
    torch.compiler.reset()
    graphs_before = counters["stats"]["unique_graphs"]
    torch.manual_seed(1973)
    shape = (73, 8, 128) if packed else (2, 73, 8, 128)
    x = torch.randn(*shape[:-1], 256, dtype=dtype, device="cuda")[..., ::2].requires_grad_()
    k = torch.randn(*shape[:-2], 2, 256, dtype=dtype, device="cuda")[..., ::2].requires_grad_()
    original = x.detach().clone()
    original_k = k.detach().clone()
    angles = torch.randn(*shape[:-2], rotary_dim // 2, device="cuda")
    cos, sin = angles.cos(), angles.sin()
    grads = (torch.randn_like(x), torch.randn_like(k))

    # HuggingFace rotates with duplicated, dtype-rounded tables. Keep its
    # independent formulation as the reference, including the unrotated tail.
    reference_q, reference_k = apply_rotary_pos_emb(
        x[..., :rotary_dim],
        k[..., :rotary_dim],
        torch.cat((cos, cos), dim=-1).to(dtype),
        torch.cat((sin, sin), dim=-1).to(dtype),
        unsqueeze_dim=1 if packed else 2,
    )
    reference = (
        torch.cat((reference_q, x[..., rotary_dim:]), dim=-1),
        torch.cat((reference_k, k[..., rotary_dim:]), dim=-1),
    )
    freqs_cis = torch.cat((cos, sin), dim=-1)
    actual = apply_rotary_emb_qk(x, k, freqs_cis, format="thd" if packed else "bshd", rope_fusion=False)
    expected_grads = torch.autograd.grad(reference, (x, k), grads)
    actual_grads = torch.autograd.grad(actual, (x, k), grads)
    repeated_grads = torch.autograd.grad(
        apply_rotary_emb_qk(x, k, freqs_cis, format="thd" if packed else "bshd", rope_fusion=False),
        (x, k),
        grads,
    )

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)
    torch.testing.assert_close(repeated_grads, actual_grads, rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    torch.testing.assert_close(k, original_k, rtol=0, atol=0)
    assert counters["stats"]["unique_graphs"] > graphs_before


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_rope_reuses_compiled_graph_across_token_counts(dtype: torch.dtype) -> None:
    torch.compiler.reset()
    graphs_before = counters["stats"]["unique_graphs"]
    torch.manual_seed(41)
    for tokens in range(100, 112):
        q = torch.randn(tokens, 8, 64, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(tokens, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
        angles = torch.randn(tokens, 32, device="cuda")
        cos, sin = angles.cos(), angles.sin()
        expected = apply_rotary_pos_emb(
            q, k, torch.cat((cos, cos), -1).to(dtype), torch.cat((sin, sin), -1).to(dtype), unsqueeze_dim=1
        )
        actual = apply_rotary_emb_qk(q, k, torch.cat((cos, sin), -1), format="thd", rope_fusion=False)
        upstream = (torch.randn_like(q), torch.randn_like(k))
        actual_grads = torch.autograd.grad(actual, (q, k), upstream)
        expected_grads = torch.autograd.grad(expected, (q, k), upstream)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)
    # A disabled compiled branch produces zero; static specialization reaches
    # the default eight-graph limit and then silently switches to eager.
    assert 1 <= counters["stats"]["unique_graphs"] - graphs_before <= 2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rotary_dim", [64, 128])
def test_packed_document_resets_and_padding_match_hf(dtype: torch.dtype, rotary_dim: int) -> None:
    """Compare padded THD documents independently, including partial YaRN rotation and gradients."""
    torch.compiler.reset()
    graphs_before = counters["stats"]["unique_graphs"]
    torch.manual_seed(4346)
    lengths, extents = (1, 7, 3, 18), (4, 8, 4, 20)
    # The third document has a nonzero starting position; padding uses position 0.
    offsets = (0, 0, 65536, 0)
    positions = torch.cat(
        [
            torch.cat((torch.arange(length) + offset, torch.zeros(extent - length)))
            for length, extent, offset in zip(lengths, extents, offsets)
        ]
    ).to(device="cuda", dtype=torch.int64)
    rotary = RotaryEmbedding(
        head_dim=128,
        base=10000,
        dtype=torch.float32,
        scaling_factor=4.0,
        partial_rotary_factor=rotary_dim / 128,
        device=torch.device("cuda"),
    )
    frequencies = position_ids_to_freqs_cis(rotary, positions, qkv_format="thd", for_fused_rope=False)
    concentration, inv_freq = rotary._compute_concentration_and_inv_freq()
    q = torch.randn(sum(extents), 8, 256, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    k = torch.randn(sum(extents), 2, 256, device="cuda", dtype=dtype)[..., ::2].requires_grad_()
    actual = apply_rotary_emb_qk(q, k, frequencies, format="thd", rope_fusion=False)
    expected_q, expected_k = [], []
    for q_doc, k_doc, length, extent, offset in zip(q.split(extents), k.split(extents), lengths, extents, offsets):
        doc_positions = torch.cat(
            (torch.arange(length, device="cuda") + offset, torch.zeros(extent - length, device="cuda"))
        )
        angles = doc_positions.float().unsqueeze(-1) * inv_freq
        cos, sin = (angles.cos() * concentration).to(dtype), (angles.sin() * concentration).to(dtype)
        q_rot, k_rot = apply_rotary_pos_emb(
            q_doc[..., :rotary_dim],
            k_doc[..., :rotary_dim],
            torch.cat((cos, cos), -1),
            torch.cat((sin, sin), -1),
            unsqueeze_dim=1,
        )
        expected_q.append(torch.cat((q_rot, q_doc[..., rotary_dim:]), -1))
        expected_k.append(torch.cat((k_rot, k_doc[..., rotary_dim:]), -1))
    expected = (torch.cat(expected_q), torch.cat(expected_k))
    upstream = (torch.randn_like(q), torch.randn_like(k))
    actual_grads = torch.autograd.grad(actual, (q, k), upstream)
    expected_grads = torch.autograd.grad(expected, (q, k), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grads, expected_grads, rtol=0, atol=0)
    assert counters["stats"]["unique_graphs"] > graphs_before

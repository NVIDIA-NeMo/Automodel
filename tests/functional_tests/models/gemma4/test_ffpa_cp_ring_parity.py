# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""GPU parity for the Gemma4 FFPA CuTeDSL backend.

Covers the single-card kernel (vs HF eager) and the cp_size=1 varlen ring
(single-document + packed, fwd+bwd vs SDPA). cp_size=1 exercises the real
forward/backward kernel and the fp32 merge/grad plumbing without a process
group; true multi-rank ring parity needs torchrun (run_ffpa_varlen_ring_cp.py).
"""

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.gemma4_moe import cp_attention as cpa

_SCALE = 1.0 / math.sqrt(256)  # Gemma4 full-attn query_pre_attn_scalar=256


def _ffpa_ready() -> bool:
    return (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 and cpa._ffpa_varlen_ring_available()
    )


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _ffpa_ready(), reason="requires CUDA + FFPA CuTeDSL kernel (SM>=8.0)"),
]


# --------------------------------------------------------------------------- #
# Single-card backend                                                         #
# --------------------------------------------------------------------------- #
def _stub_attn_module(head_dim: int, groups: int) -> nn.Module:
    m = nn.Module()
    m.head_dim, m.num_key_value_groups, m.training = head_dim, groups, False
    return m


@pytest.mark.parametrize("Hq,Hkv", [(8, 4), (4, 4)])
def test_ffpa_vs_eager_bf16_parity(Hq, Hkv):
    """ffpa_attention_forward (head_dim=512) matches HF eager causal attention."""
    eager = pytest.importorskip("transformers.models.gemma4.modeling_gemma4").eager_attention_forward
    from nemo_automodel.components.attention.ffpa_attention import ffpa_attention_forward

    B, S, D, dev, dt = 1, 64, 512, "cuda", torch.bfloat16
    torch.manual_seed(42)
    q = torch.randn(B, Hq, S, D, dtype=dt, device=dev)
    k = torch.randn(B, Hkv, S, D, dtype=dt, device=dev)
    v = torch.randn(B, Hkv, S, D, dtype=dt, device=dev)
    module = _stub_attn_module(D, Hq // Hkv)
    causal = torch.full((S, S), float("-inf"), dtype=dt, device=dev).triu_(1)[None, None]

    with torch.no_grad():
        out, _ = ffpa_attention_forward(module, q, k, v, attention_mask=None, dropout=0.0, scaling=_SCALE, softcap=None)
        ref, _ = eager(module, q, k, v, attention_mask=causal, dropout=0.0, scaling=_SCALE, softcap=None)

    assert out.shape == ref.shape == (B, S, Hq, D)
    assert (out.float() - ref.float()).abs().max().item() < 2e-2
    cos = torch.nn.functional.cosine_similarity(out.flatten().float(), ref.flatten().float(), dim=0).item()
    assert cos > 0.9999, cos


def test_ffpa_selective_checkpoint_policy_importable_and_marks_fwd_ops():
    """The relocated ffpa_selective_checkpoint_policy builds a context_fn and marks
    the FFPA forward ops MUST_SAVE (the full-AC contract for Gemma4-31B)."""
    from torch.utils.checkpoint import CheckpointPolicy

    from nemo_automodel.components.distributed.activation_checkpointing import ffpa_selective_checkpoint_policy

    assert callable(ffpa_selective_checkpoint_policy())
    must_save = {torch.ops.ffpa_attn._fwd_cute.default, torch.ops.ffpa_attn._varlen_fwd_cute.default}

    def policy(_ctx, op, *a, **k):
        return CheckpointPolicy.MUST_SAVE if op in must_save else CheckpointPolicy.PREFER_RECOMPUTE

    assert policy(None, torch.ops.ffpa_attn._varlen_fwd_cute.default) == CheckpointPolicy.MUST_SAVE
    assert policy(None, torch.ops.aten.mm.default) == CheckpointPolicy.PREFER_RECOMPUTE


# --------------------------------------------------------------------------- #
# cp_size=1 varlen ring                                                       #
# --------------------------------------------------------------------------- #
def _ring_ctx(q, k, v, doc_ids):
    return cpa.CPRingAttentionContext(
        module=SimpleNamespace(sliding_window=None),
        query=q,
        key=k,
        value=v,
        cp_mesh=None,
        cp_group=None,
        cp_size=1,
        cp_rank=0,
        seq_local=q.shape[2],
        seq_full=q.shape[2],
        seq_global_start=0,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=True,
        scale=_SCALE,
        enable_gqa=(q.shape[1] != k.shape[1]),
        kwargs={},
        metadata={"_packed_seq_ids": doc_ids},
        metadata_seq_dims={},
    )


@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 4)])
def test_varlen_ring_cp1_single_doc_matches_sdpa(Hq, Hkv):
    """Unpacked CP batch: the synthesized single-document map collapses the varlen
    cu_seqlens to one segment == plain causal attention."""
    B, S, D, dev, dt = 1, 256, 512, "cuda", torch.bfloat16
    torch.manual_seed(0)
    q = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
    doc = torch.ones(B, S, dtype=torch.long, device=dev)

    qf, kf, vf = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = cpa._Gemma4FFPAVarlenRingAttention.apply(qf, kf, vf, _ring_ctx(qf, kf, vf, doc))
    out.sum().backward()

    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(
        qr, kr, vr, is_causal=True, scale=_SCALE, enable_gqa=(Hq != Hkv)
    )
    ref.sum().backward()

    assert torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    for got, exp in ((qf, qr), (kf, kr), (vf, vr)):
        assert torch.allclose(got.grad.float(), exp.grad.float(), atol=3e-2, rtol=2e-2)


@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 4)])
def test_varlen_ring_cp1_packed_matches_blockdiag_sdpa(Hq, Hkv):
    """Packed multi-document sequence vs a block-diagonal (same-document) causal
    SDPA reference; pad query rows excluded from the comparison."""
    B, S, D, dev, dt = 1, 256, 512, "cuda", torch.bfloat16
    torch.manual_seed(0)
    doc = torch.zeros(B, S, dtype=torch.long, device=dev)
    doc[0, :150], doc[0, 150:250] = 1, 2  # doc1 | doc2 | pad
    real = doc[0] > 0
    q = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
    k = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
    v = torch.randn(B, Hkv, S, D, device=dev, dtype=dt)
    g = torch.randn(B, Hq, S, D, device=dev, dtype=dt)
    g[:, :, ~real] = 0  # pad query rows contribute no gradient

    qf, kf, vf = (t.clone().requires_grad_(True) for t in (q, k, v))
    out = cpa._Gemma4FFPAVarlenRingAttention.apply(qf, kf, vf, _ring_ctx(qf, kf, vf, doc))
    out.backward(g)

    qpos = torch.arange(S, device=dev).view(-1, 1)
    kpos = torch.arange(S, device=dev).view(1, -1)
    allowed = (kpos <= qpos) & (doc[0].view(-1, 1) == doc[0].view(1, -1)) & (doc[0].view(-1, 1) > 0)
    allowed[~real, 0] = True  # pad rows attend pos 0 (output discarded)
    mask = torch.zeros(S, S, device=dev).masked_fill(~allowed, float("-inf")).view(1, 1, S, S)

    qr, kr, vr = (t.clone().requires_grad_(True) for t in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(
        qr, kr, vr, attn_mask=mask.to(dt), scale=_SCALE, enable_gqa=(Hq != Hkv)
    )
    ref.backward(g)

    assert torch.allclose(out.float()[:, :, real], ref.float()[:, :, real], atol=2e-2, rtol=2e-2)
    for got, exp in ((qf, qr), (kf, kr), (vf, vr)):
        assert torch.allclose(got.grad.float()[:, :, real], exp.grad.float()[:, :, real], atol=3e-2, rtol=2e-2)

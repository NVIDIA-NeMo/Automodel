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

import importlib.util

import pytest
import torch

from nemo_automodel.components.attention import fa4
from nemo_automodel.components.attention.fa4 import _varlen_segments, document_causal_fa4_attention


def _fa4_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        return False
    try:
        return importlib.util.find_spec("flash_attn.cute.interface") is not None
    except ModuleNotFoundError:
        return False


def _document_causal_mask(q_doc_ids: torch.Tensor, kv_doc_ids: torch.Tensor, q_global_start: int) -> torch.Tensor:
    """FP64 additive mask [batch, 1, q, k]: causal within a document; padding queries read key 0 only."""
    q_pos = torch.arange(q_doc_ids.shape[1], device=q_doc_ids.device) + q_global_start
    k_pos = torch.arange(kv_doc_ids.shape[1], device=kv_doc_ids.device)
    allowed = (k_pos[None, None, :] <= q_pos[None, :, None]) & (q_doc_ids[:, :, None] == kv_doc_ids[:, None, :])
    allowed &= kv_doc_ids[:, None, :] > 0
    allowed = torch.where((q_doc_ids <= 0)[:, :, None], (k_pos == 0)[None, None, :], allowed)
    mask = torch.zeros(allowed.shape, dtype=torch.float64, device=q_doc_ids.device)
    return mask.masked_fill(~allowed, float("-inf")).unsqueeze(1)


def test_import_fa4_modules_reports_missing(monkeypatch):
    monkeypatch.setattr(fa4, "safe_import", lambda name: (name != "cutlass.cute", None))
    with pytest.raises(ImportError, match="could not import cutlass.cute"):
        fa4.import_fa4_modules("flash_attn.cute.interface", "cutlass.cute")


def test_varlen_segments_split_documents_at_the_local_shard():
    # Global row: doc 1 [0, 3), doc 2 [3, 7), padding [7, 8); local queries [4, 6).
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 0]])
    q_lengths, k_lengths = _varlen_segments(doc_ids, q_length=2, q_global_start=4)

    # doc 1 has no local queries; doc 2 contributes queries [4, 6) over keys [3, 6), its tail [6, 7) and the
    # padding run are key-only pieces, so the key segments tile the row.
    assert q_lengths == [0, 2, 0, 0]
    assert k_lengths == [3, 3, 1, 1]


def test_varlen_segments_tile_every_row():
    doc_ids = torch.tensor([[1, 1, 2, 2], [1, 1, 1, 0]])
    q_lengths, k_lengths = _varlen_segments(doc_ids, q_length=4, q_global_start=0)
    assert q_lengths == k_lengths == [2, 2, 3, 1]


@pytest.mark.skipif(not _fa4_available(), reason="requires FlashAttention 4 on an SM90 or SM100 GPU")
@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="the first case in a process JIT-compiles FA4's CuTe forward and backward kernels (~6 s on H100 CI)",
)
@pytest.mark.parametrize(
    ("layout", "q_global_start", "local_len"),
    [
        ("causal", 0, 512),
        ("packed", 0, 512),
        ("left_pad", 0, 512),
        ("causal", 0, 256),
        ("causal", 256, 256),
        ("packed", 256, 256),
        ("empty_rank", 256, 256),
        ("two_rows", 128, 256),
    ],
)
def test_document_causal_fa4_attention_matches_reference(layout, q_global_start, local_len):
    torch.manual_seed(0)
    global_len, heads = 512, 4
    batch = 2 if layout == "two_rows" else 1
    doc_ids = torch.ones(batch, global_len, dtype=torch.int32, device="cuda")
    if layout in ("packed", "two_rows"):
        doc_ids[0, 170:] = 2
        doc_ids[0, 384:] = 3
        doc_ids[0, -7:] = 0
    if layout == "two_rows":
        doc_ids[1, 300:] = 2
        doc_ids[1, 301:] = 3
    elif layout == "left_pad":
        doc_ids[:, :37] = 0
    elif layout == "empty_rank":
        doc_ids[:, local_len:] = 0
    q_doc_ids = doc_ids[:, q_global_start : q_global_start + local_len]

    shape = (batch, heads)
    query = torch.randn(*shape, local_len, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(*shape, global_len, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn(*shape, global_len, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad_output = torch.randn(*shape, local_len, 128, device="cuda", dtype=torch.bfloat16)
    grad_output = grad_output.masked_fill((q_doc_ids <= 0)[:, None, :, None], 0)
    scale = 192**-0.5

    output = document_causal_fa4_attention(
        query, key, value, q_doc_ids=q_doc_ids, kv_doc_ids=doc_ids, q_global_start=q_global_start, scale=scale
    )
    grads = torch.autograd.grad(output, (query, key, value), grad_output)

    ref_inputs = [t.detach().double().requires_grad_() for t in (query, key, value)]
    mask = _document_causal_mask(q_doc_ids, doc_ids, q_global_start)
    scores = ref_inputs[0] @ ref_inputs[1].transpose(-1, -2) * scale + mask
    ref_output = (scores.softmax(-1) @ ref_inputs[2]).masked_fill((q_doc_ids <= 0)[:, None, :, None], 0)
    ref_grads = torch.autograd.grad(ref_output, ref_inputs, grad_output.double())

    for actual, expected in [(output, ref_output), *zip(grads, ref_grads)]:
        assert torch.isfinite(actual).all()
        rel_l2 = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-12)
        assert rel_l2 < 1e-2

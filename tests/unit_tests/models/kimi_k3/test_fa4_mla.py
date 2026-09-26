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

import nemo_automodel.components.models.kimi_k3.model as kimi_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext, build_document_causal_mask
from nemo_automodel.components.models.kimi_k3.fa4_mla import _varlen_segments


def _fa4_available() -> bool:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (9, 10):
        return False
    try:
        return importlib.util.find_spec("flash_attn.cute.interface") is not None
    except ModuleNotFoundError:
        return False


def _small_config(**kwargs) -> KimiK3TextConfig:
    return KimiK3TextConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=32,
        kv_lora_rank=16,
        num_hidden_layers=4,
        torch_dtype=torch.float32,
        **kwargs,
    )


def test_mla_attn_backend_defaults_and_rejects_unknown():
    assert _small_config().mla_attn_backend == "default"
    assert _small_config(mla_attn_backend="fa4").mla_attn_backend == "fa4"
    with pytest.raises(ValueError, match="mla_attn_backend"):
        _small_config(mla_attn_backend="flex")


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


class _FakeCPMesh:
    def get_group(self):
        return None


def _fake_attention(calls, name):
    def attention(query, key, value, *, scale, q_doc_ids=None, kv_doc_ids=None, q_global_start=None):
        calls.append((name, None if q_doc_ids is None else tuple(q_doc_ids.shape), q_global_start))
        return torch.zeros(*query.shape[:-1], value.shape[-1], dtype=value.dtype)

    return attention


@pytest.mark.parametrize("backend", ["default", "fa4"])
def test_cp_forward_dispatches_on_mla_attn_backend(monkeypatch, backend):
    calls = []
    monkeypatch.setattr(kimi_model, "all_gather_sequence", lambda tensor, group, dim: torch.cat([tensor] * 2, dim))
    monkeypatch.setattr(kimi_model, "document_causal_flex_attention", _fake_attention(calls, "flex"))
    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", _fake_attention(calls, "fa4"))

    module = kimi_model.KimiMLAAttention(
        _small_config(mla_attn_backend=backend), 3, BackendConfig(attn="eager", linear="torch")
    )
    module.setup_cp_attention(_FakeCPMesh())
    hidden_states = torch.randn(1, 4, 64)
    doc_ids = torch.ones(1, 8, dtype=torch.int32)
    output = module(hidden_states, packed_context=KimiPackedContext(doc_ids, seq_start=4, cp_size=2))

    assert output.shape == hidden_states.shape
    assert calls == [("flex" if backend == "default" else "fa4", (1, 4), 4)]


@pytest.mark.parametrize(("packed", "expected"), [(False, ("causal", None, None)), (True, ("fa4", (1, 8), 0))])
def test_non_cp_forward_uses_fa4(monkeypatch, packed, expected):
    calls = []
    monkeypatch.setattr(kimi_model, "causal_fa4_attention", _fake_attention(calls, "causal"))
    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", _fake_attention(calls, "fa4"))

    module = kimi_model.KimiMLAAttention(
        _small_config(mla_attn_backend="fa4"), 3, BackendConfig(attn="eager", linear="torch")
    )
    hidden_states = torch.randn(1, 8, 64)
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0]], dtype=torch.int32)
    packed_context = KimiPackedContext(doc_ids) if packed else None
    output = module(hidden_states, packed_context=packed_context)

    assert output.shape == hidden_states.shape
    assert calls == [expected]


@pytest.mark.skipif(not _fa4_available(), reason="requires FlashAttention 4 on an SM90 or SM100 GPU")
@pytest.mark.parametrize(
    ("layout", "q_global_start", "local_len"),
    [
        ("causal", 0, 512),
        ("packed", 0, 512),
        ("causal", 0, 256),
        ("causal", 256, 256),
        ("packed", 256, 256),
        ("empty_rank", 256, 256),
        ("two_rows", 128, 256),
    ],
)
def test_document_causal_fa4_attention_matches_reference(layout, q_global_start, local_len):
    from nemo_automodel.components.models.kimi_k3.fa4_mla import document_causal_fa4_attention

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
    mask = build_document_causal_mask(q_doc_ids, doc_ids, q_global_start=q_global_start, dtype=torch.float64)
    scores = ref_inputs[0] @ ref_inputs[1].transpose(-1, -2) * scale + mask
    ref_output = (scores.softmax(-1) @ ref_inputs[2]).masked_fill((q_doc_ids <= 0)[:, None, :, None], 0)
    ref_grads = torch.autograd.grad(ref_output, ref_inputs, grad_output.double())

    for actual, expected in [(output, ref_output), *zip(grads, ref_grads)]:
        assert torch.isfinite(actual).all()
        rel_l2 = (actual.double() - expected).norm() / expected.norm().clamp_min(1e-12)
        assert rel_l2 < 1e-2

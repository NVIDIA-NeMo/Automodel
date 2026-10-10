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

import nemo_automodel.components.models.kimi_k3.model as kimi_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext


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


@pytest.mark.parametrize(("attn", "use_fa4"), [("eager", False), ("fa4", True)])
def test_backend_attn_selects_fa4(attn, use_fa4):
    module = kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn=attn, linear="torch"))
    assert module.use_fa4 is use_fa4
    # FA4 is called directly; no TE/SDPA attention module is built for it.
    assert module.attn_module is None


def test_fa4_rejects_attention_dropout():
    with pytest.raises(ValueError, match="does not support attention dropout"):
        kimi_model.KimiMLAAttention(_small_config(attention_dropout=0.1), 3, BackendConfig(attn="fa4", linear="torch"))


def test_backend_attn_rejects_unsupported():
    with pytest.raises(ValueError, match="does not support backend.attn"):
        kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn="flex", linear="torch"))


class _FakeCPMesh:
    def get_group(self):
        return None


def _fake_attention(calls, name):
    def attention(query, key, value, *, scale, q_doc_ids=None, kv_doc_ids=None, q_global_start=None):
        calls.append((name, None if q_doc_ids is None else tuple(q_doc_ids.shape), q_global_start))
        return torch.zeros(*query.shape[:-1], value.shape[-1], dtype=value.dtype)

    return attention


@pytest.mark.parametrize("attn", ["eager", "fa4"])
def test_cp_forward_dispatches_on_backend_attn(monkeypatch, attn):
    calls = []
    monkeypatch.setattr(kimi_model, "all_gather_sequence", lambda tensor, group, dim: torch.cat([tensor] * 2, dim))
    monkeypatch.setattr(kimi_model, "document_causal_flex_attention", _fake_attention(calls, "flex"))
    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", _fake_attention(calls, "fa4"))

    module = kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn=attn, linear="torch"))
    module.setup_cp_attention(_FakeCPMesh())
    hidden_states = torch.randn(1, 4, 64)
    doc_ids = torch.ones(1, 8, dtype=torch.int32)
    output = module(hidden_states, packed_context=KimiPackedContext(doc_ids, seq_start=4, cp_size=2))

    assert output.shape == hidden_states.shape
    assert calls == [("flex" if attn == "eager" else "fa4", (1, 4), 4)]


@pytest.mark.parametrize(
    ("layout", "expected"),
    [
        (None, ("causal", None, None)),
        ([1, 1, 1, 2, 2, 2, 0, 0], ("fa4", (1, 8), 0)),
        # A single left-padded document must not fall back to plain causal: valid queries would see padding keys.
        ([0, 0, 1, 1, 1, 1, 1, 1], ("fa4", (1, 8), 0)),
        ([1, 1, 1, 1, 1, 1, 0, 0], ("fa4", (1, 8), 0)),
    ],
)
def test_non_cp_forward_uses_fa4(monkeypatch, layout, expected):
    calls = []
    monkeypatch.setattr(kimi_model, "causal_fa4_attention", _fake_attention(calls, "causal"))
    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", _fake_attention(calls, "fa4"))

    module = kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn="fa4", linear="torch"))
    hidden_states = torch.randn(1, 8, 64)
    packed_context = None if layout is None else KimiPackedContext(torch.tensor([layout], dtype=torch.int32))
    output = module(hidden_states, packed_context=packed_context)

    assert output.shape == hidden_states.shape
    assert calls == [expected]


def test_non_cp_forward_routes_standalone_padding_mask_to_document_path(monkeypatch):
    calls = []
    monkeypatch.setattr(kimi_model, "causal_fa4_attention", _fake_attention(calls, "causal"))

    def document_attention(query, key, value, *, scale, q_doc_ids, kv_doc_ids, q_global_start):
        calls.append(("fa4", q_doc_ids.tolist(), q_global_start))
        return torch.zeros(*query.shape[:-1], value.shape[-1], dtype=value.dtype)

    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", document_attention)
    module = kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn="fa4", linear="torch"))
    padding_mask = torch.tensor([[True, True, False, False, False, False]])
    module(torch.randn(1, 6, 64), padding_mask=padding_mask)

    assert calls == [("fa4", [[0, 0, 1, 1, 1, 1]], 0)]


def _reference_document_attention(query, key, value, *, scale, q_doc_ids, kv_doc_ids, q_global_start):
    # [batch, heads, sequence, dim] eager stand-in for the FA4 document-causal kernel.
    mask = kimi_model.build_document_causal_mask(
        q_doc_ids, kv_doc_ids, q_global_start=q_global_start, dtype=query.dtype
    )
    weights = torch.softmax(query @ key.transpose(-2, -1) * scale + mask, dim=-1)
    return weights @ value


def _reference_causal_attention(query, key, value, *, scale):
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scale)


def test_standalone_padding_mask_makes_valid_outputs_independent_of_padding(monkeypatch):
    monkeypatch.setattr(kimi_model, "causal_fa4_attention", _reference_causal_attention)
    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", _reference_document_attention)
    torch.manual_seed(0)
    module = kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn="fa4", linear="torch")).eval()
    padding_mask = torch.tensor([[True, True, True, False, False, False, False, False]])
    hidden_states = torch.randn(1, 8, 64)
    perturbed = hidden_states.clone()
    perturbed[padding_mask] = torch.randn(int(padding_mask.sum()), 64) * 10

    with torch.no_grad():
        output = module(hidden_states, padding_mask=padding_mask)
        perturbed_output = module(perturbed, padding_mask=padding_mask)

    valid = ~padding_mask
    torch.testing.assert_close(perturbed_output[valid], output[valid], rtol=0, atol=0)

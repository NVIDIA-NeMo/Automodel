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


@pytest.mark.parametrize(("packed", "expected"), [(False, ("causal", None, None)), (True, ("fa4", (1, 8), 0))])
def test_non_cp_forward_uses_fa4(monkeypatch, packed, expected):
    calls = []
    monkeypatch.setattr(kimi_model, "causal_fa4_attention", _fake_attention(calls, "causal"))
    monkeypatch.setattr(kimi_model, "document_causal_fa4_attention", _fake_attention(calls, "fa4"))

    module = kimi_model.KimiMLAAttention(_small_config(), 3, BackendConfig(attn="fa4", linear="torch"))
    hidden_states = torch.randn(1, 8, 64)
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 2, 0, 0]], dtype=torch.int32)
    packed_context = KimiPackedContext(doc_ids) if packed else None
    output = module(hidden_states, packed_context=packed_context)

    assert output.shape == hidden_states.shape
    assert calls == [expected]

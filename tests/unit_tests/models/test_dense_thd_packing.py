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

"""CPU parity coverage for dense Llama/Qwen2 packed THD execution."""

from __future__ import annotations

import importlib
import warnings

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, Qwen2Config, Qwen3Config

from nemo_automodel.components.models.common import BackendConfig

MODEL_KINDS = ("llama", "qwen2", "qwen3")


class _ReferencePackedAttention(nn.Module):
    """CPU reference for TE's causal variable-length THD attention."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> torch.Tensor:
        """Attend independently within each packed document.

        Args:
            query: Packed queries ``[T, Hq, D]``.
            key: Packed keys ``[T, Hkv, D]``.
            value: Packed values ``[T, Hkv, D]``.
            **kwargs: Must contain ``cu_seqlens_q`` shaped ``[N + 1]`` for
                ``N`` packed documents.

        Returns:
            Packed output ``[T, Hq, D]`` in the same document order.
        """
        cu_seqlens = kwargs["cu_seqlens_q"].tolist()
        outputs = []
        for start, end in zip(cu_seqlens, cu_seqlens[1:]):
            q_doc = query[start:end].transpose(0, 1).unsqueeze(0)
            k_doc = key[start:end].transpose(0, 1).unsqueeze(0)
            v_doc = value[start:end].transpose(0, 1).unsqueeze(0)
            output = F.scaled_dot_product_attention(
                q_doc,
                k_doc,
                v_doc,
                is_causal=True,
                enable_gqa=True,
                scale=self.scale,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)


def _reference_te_factory(**kwargs):
    attention = _ReferencePackedAttention(kwargs["softmax_scale"])
    return attention, attention.__call__


def _build_model(model_kind: str, monkeypatch: pytest.MonkeyPatch) -> nn.Module:
    if model_kind == "llama":
        module = importlib.import_module("nemo_automodel.components.models.llama.model")
        model_cls = module.LlamaForCausalLM
        config = LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
    elif model_kind == "qwen2":
        module = importlib.import_module("nemo_automodel.components.models.qwen2.model")
        model_cls = module.Qwen2ForCausalLM
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )
    else:
        module = importlib.import_module("nemo_automodel.components.models.qwen3.model")
        model_cls = module.Qwen3ForCausalLM
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            max_position_embeddings=32,
            attention_dropout=0.0,
            tie_word_embeddings=False,
        )

    monkeypatch.setattr(module, "initialize_attn_module_and_func", _reference_te_factory)
    config._attn_implementation = "sdpa"
    config.use_cache = False
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", rope_fusion=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return model_cls(config, backend=backend).to(torch.float32)


@pytest.mark.parametrize("model_kind", MODEL_KINDS)
def test_packed_thd_matches_per_document_logits_and_gradients(model_kind, monkeypatch):
    """Packed THD must preserve document boundaries in forward and backward."""
    torch.manual_seed(1234)
    reference_model = _build_model(model_kind, monkeypatch).train()
    packed_model = _build_model(model_kind, monkeypatch).train()
    packed_model.load_state_dict(reference_model.state_dict())

    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)

    reference_logits = torch.cat(
        [
            reference_model(
                input_ids[:3].unsqueeze(0),
                position_ids=position_ids[:3].unsqueeze(0),
                use_cache=False,
            ).logits,
            reference_model(
                input_ids[3:].unsqueeze(0),
                position_ids=position_ids[3:].unsqueeze(0),
                use_cache=False,
            ).logits,
        ],
        dim=1,
    )
    reference_logits.square().sum().backward()

    packed_logits = packed_model(
        input_ids,
        position_ids=position_ids,
        qkv_format="thd",
        cu_seqlens=cu_seqlens,
        max_seqlen=5,
    ).logits
    packed_logits.square().sum().backward()

    torch.testing.assert_close(packed_logits, reference_logits, atol=1e-6, rtol=1e-6)
    reference_grads = dict(reference_model.named_parameters())
    for name, packed_param in packed_model.named_parameters():
        reference_grad = reference_grads[name].grad
        assert reference_grad is not None, name
        assert packed_param.grad is not None, name
        torch.testing.assert_close(packed_param.grad, reference_grad, atol=2e-6, rtol=2e-5)

    assert packed_model.ModelCapabilities().supports_cp is True


@pytest.mark.parametrize("model_kind", MODEL_KINDS)
def test_single_token_thd_preserves_sequence_axis(model_kind, monkeypatch):
    """A one-token packed sequence must retain the THD token dimension."""
    model = _build_model(model_kind, monkeypatch).eval()

    logits = model(
        torch.tensor([1]),
        position_ids=torch.tensor([0]),
        qkv_format="thd",
        cu_seqlens=torch.tensor([0, 1], dtype=torch.int32),
        max_seqlen=torch.tensor(1),
    ).logits

    assert logits.shape == (1, 1, model.config.vocab_size)


@pytest.mark.parametrize("model_kind", MODEL_KINDS)
def test_packed_thd_rejects_non_te_attention(model_kind, monkeypatch):
    """THD must fail before silently treating packed documents as one sequence."""
    model = _build_model(model_kind, monkeypatch).eval()
    for layer in model.model.layers:
        del layer.self_attn.attn_module

    with pytest.raises(ValueError, match="requires backend.attn='te'"):
        model(
            torch.tensor([1, 2, 3, 4]),
            position_ids=torch.tensor([0, 1, 0, 1]),
            qkv_format="thd",
            cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32),
            max_seqlen=2,
        )

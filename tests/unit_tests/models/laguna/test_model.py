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

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.distributed.blockdiag_cp import exchange as blockdiag_exchange
from nemo_automodel.components.distributed.blockdiag_cp import state as blockdiag_state
from nemo_automodel.components.distributed.blockdiag_cp.state import BlockdiagCpModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.laguna import model as laguna_model_module
from nemo_automodel.components.models.laguna.config import LagunaConfig
from nemo_automodel.components.models.laguna.model import LagunaForCausalLM
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.megatron import moe_utils


def _backend() -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


def _te_backend() -> BackendConfig:
    return BackendConfig(
        attn="te",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


def _sdpa_backend() -> BackendConfig:
    backend = _backend()
    backend.attn = "sdpa"
    return backend


class _IdentityGather:
    @staticmethod
    def apply(tensor, group, seq_dim):
        del group, seq_dim
        return tensor


class _ReferencePackedAttention(nn.Module):
    """Independent CPU reference for TE's causal variable-length THD attention."""

    def __init__(self, scale: float, attention_dropout: float) -> None:
        super().__init__()
        self.scale = scale
        self.attention_dropout = attention_dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs) -> torch.Tensor:
        cu_seqlens = kwargs["cu_seqlens_q"].tolist()
        left_window, right_window = kwargs.get("window_size", (-1, 0))
        outputs = []
        for start, end in zip(cu_seqlens, cu_seqlens[1:]):
            query_doc = query[start:end].transpose(0, 1).unsqueeze(0)
            key_doc = key[start:end].transpose(0, 1).unsqueeze(0)
            value_doc = value[start:end].transpose(0, 1).unsqueeze(0)
            attention_mask = None
            is_causal = True
            if left_window >= 0:
                positions = torch.arange(end - start)
                query_positions = positions[:, None]
                key_positions = positions[None, :]
                attention_mask = (key_positions >= query_positions - left_window) & (
                    key_positions <= query_positions + right_window
                )
                is_causal = False
            output = F.scaled_dot_product_attention(
                query_doc,
                key_doc,
                value_doc,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                enable_gqa=True,
                scale=self.scale,
            )
            outputs.append(output.squeeze(0).transpose(0, 1))
        return torch.cat(outputs)


def _reference_te_factory(**kwargs):
    attention = _ReferencePackedAttention(kwargs["softmax_scale"], kwargs["attention_dropout"])
    return attention, attention.__call__


def _tiny_config() -> LagunaConfig:
    cfg = LagunaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_attention_heads_per_layer=[2, 4],
        num_key_value_heads=1,
        head_dim=4,
        gating="per-head",
        gating_types=["per_head", "per_head"],
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=4,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        moe_routed_scaling_factor=2.5,
        mlp_layer_types=["dense", "sparse"],
        router_aux_loss_coef=0.0,
        torch_dtype="float32",
    )
    cfg._attn_implementation = "eager"
    return cfg


def _dense_tiny_config() -> LagunaConfig:
    cfg = _tiny_config()
    cfg.mlp_layer_types = ["dense", "dense"]
    cfg.mlp_only_layers = [0, 1]
    return cfg


def _patch_weighted_swiglu(monkeypatch) -> None:
    def weighted_swiglu_eager(y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        y1, y2 = torch.chunk(y, 2, -1)
        return (torch.nn.functional.silu(y1) * y2 * weights).to(y.dtype)

    # Keep Laguna CPU smokes independent of torch.compile availability; shared MoE tests cover the compiled helper.
    monkeypatch.setattr(moe_utils, "weighted_swiglu", weighted_swiglu_eager)


def test_laguna_forward_tiny_config():
    model = LagunaForCausalLM(_dense_tiny_config(), backend=_backend())
    model.eval()

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 4, 32)


def test_laguna_forward_tiny_config_with_moe_layer(monkeypatch):
    _patch_weighted_swiglu(monkeypatch)
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())
    model.eval()

    assert isinstance(model.model.layers["1"].mlp, MoE)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits.shape == (1, 4, 32)


def test_laguna_initialize_weights_cpu_path(monkeypatch):
    _patch_weighted_swiglu(monkeypatch)
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())

    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    sparse_layer = model.model.layers["1"].mlp

    assert sparse_layer.gate.e_score_correction_bias.dtype == torch.float32
    assert all(torch.isfinite(param).all().item() for param in model.parameters())

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert torch.isfinite(output.logits).all()


def test_laguna_moe_defaults_match_checkpoint_routing():
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())
    sparse_layer = model.model.layers["1"].mlp

    assert isinstance(sparse_layer, MoE)
    assert model.model.moe_config.score_func == "sigmoid"
    assert model.model.moe_config.softmax_before_topk is False
    assert model.model.moe_config.norm_topk_prob is True
    assert model.model.moe_config.route_scale == 2.5
    assert model.model.moe_config.n_shared_experts == 1
    assert sparse_layer.gate.e_score_correction_bias is not None
    assert model.backend.gate_precision is torch.float32


def test_laguna_rejects_unsupported_swa_attention_sink():
    cfg = _tiny_config()
    cfg.swa_attention_sink_enabled = True

    with pytest.raises(NotImplementedError, match="swa_attention_sink_enabled=True"):
        LagunaForCausalLM(cfg, backend=_backend())


def test_laguna_attention_uses_per_layer_head_counts_and_per_head_gate():
    model = LagunaForCausalLM(_tiny_config(), backend=_backend())

    layer0_attn = model.model.layers["0"].self_attn
    layer1_attn = model.model.layers["1"].self_attn

    assert layer0_attn.q_proj.weight.shape == (8, 16)
    assert layer0_attn.g_proj.weight.shape == (2, 16)
    assert layer1_attn.q_proj.weight.shape == (16, 16)
    assert layer1_attn.g_proj.weight.shape == (4, 16)


def test_laguna_packed_thd_matches_per_document_logits_and_gradients(monkeypatch):
    """THD must preserve Laguna document boundaries, RoPE, gating, and backward."""
    monkeypatch.setattr(laguna_model_module, "initialize_attn_module_and_func", _reference_te_factory)
    torch.manual_seed(1234)
    reference_model = LagunaForCausalLM(_dense_tiny_config(), backend=_te_backend()).to(torch.float32).train()
    packed_model = LagunaForCausalLM(_dense_tiny_config(), backend=_te_backend()).to(torch.float32).train()
    packed_model.load_state_dict(reference_model.state_dict())

    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
    reference_logits = torch.cat(
        [
            reference_model(
                input_ids[:3].unsqueeze(0),
                position_ids=position_ids[:3].unsqueeze(0),
            ).logits,
            reference_model(
                input_ids[3:].unsqueeze(0),
                position_ids=position_ids[3:].unsqueeze(0),
            ).logits,
        ],
        dim=1,
    )
    reference_logits.square().sum().backward()

    packed_logits = packed_model(
        input_ids.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
        qkv_format="thd",
        cu_seqlens=cu_seqlens.unsqueeze(0),
        max_seqlen=torch.tensor([5], dtype=torch.int32),
    ).logits
    packed_logits.square().sum().backward()

    torch.testing.assert_close(packed_logits, reference_logits, atol=1e-5, rtol=1e-5)
    reference_params = dict(reference_model.named_parameters())
    for name, packed_param in packed_model.named_parameters():
        reference_grad = reference_params[name].grad
        assert reference_grad is not None, name
        assert packed_param.grad is not None, name
        torch.testing.assert_close(packed_param.grad, reference_grad, atol=2e-5, rtol=2e-4)

    capabilities = packed_model.ModelCapabilities()
    assert capabilities.supports_cp is True
    assert capabilities.supports_thd is True


def test_laguna_blockdiag_thd_matches_per_document_sliding_attention(monkeypatch):
    """The production THD+CP dispatch must preserve Laguna's sliding window."""
    monkeypatch.setattr(blockdiag_exchange, "_AllGatherSeqDiff", _IdentityGather)
    torch.manual_seed(4321)
    reference_model = LagunaForCausalLM(_dense_tiny_config(), backend=_sdpa_backend()).to(torch.float32).train()
    packed_model = LagunaForCausalLM(_dense_tiny_config(), backend=_sdpa_backend()).to(torch.float32).train()
    packed_model.load_state_dict(reference_model.state_dict())

    input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    position_ids = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4])
    reference_logits = torch.cat(
        [
            reference_model(input_ids[:3].unsqueeze(0), position_ids=position_ids[:3].unsqueeze(0)).logits,
            reference_model(input_ids[3:].unsqueeze(0), position_ids=position_ids[3:].unsqueeze(0)).logits,
        ],
        dim=1,
    )
    reference_logits.square().sum().backward()

    step_state = {
        "group": None,
        "doc_ids": torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2]]),
        "row_offset": 0,
        "seq_dim": 2,
        "attn_backend": "dense",
        "kv_exchange": "allgather",
        "model_state": BlockdiagCpModelState(
            group=None,
            packed_cu_seqlens=torch.tensor([0, 3, 8]),
            packed_cu_seqlens_cpu=torch.tensor([0, 3, 8]),
        ),
    }
    token = blockdiag_state._CP_BLOCKDIAG_STATE.set(step_state)
    try:
        packed_logits = packed_model(
            input_ids.unsqueeze(0),
            position_ids=position_ids.unsqueeze(0),
            qkv_format="thd",
        ).logits
        packed_logits.square().sum().backward()
    finally:
        blockdiag_state._CP_BLOCKDIAG_STATE.reset(token)

    torch.testing.assert_close(packed_logits, reference_logits, atol=1e-5, rtol=1e-5)
    reference_params = dict(reference_model.named_parameters())
    for name, packed_param in packed_model.named_parameters():
        torch.testing.assert_close(packed_param.grad, reference_params[name].grad, atol=2e-5, rtol=2e-4)


def test_laguna_reports_sdpa_cp_and_packing_support():
    from nemo_automodel._transformers.capabilities import ModelSupports

    model = LagunaForCausalLM(_dense_tiny_config(), backend=_sdpa_backend())
    supports = ModelSupports(model, None)

    assert supports.supports_cp is True
    assert supports.supports_sequence_packing is True


def test_laguna_packed_thd_rejects_non_te_attention():
    model = LagunaForCausalLM(_dense_tiny_config(), backend=_backend()).eval()

    with pytest.raises(ValueError, match="requires backend.attn='te'"):
        model(
            torch.tensor([[1, 2, 3, 4]]),
            position_ids=torch.tensor([[0, 1, 0, 1]]),
            qkv_format="thd",
            cu_seqlens=torch.tensor([[0, 2, 4]], dtype=torch.int32),
            max_seqlen=torch.tensor([2], dtype=torch.int32),
        )

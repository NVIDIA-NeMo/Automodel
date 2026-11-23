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

"""Real cuDNN/FlashMLA execution for Kimi K3.1, including 96 heads and backward."""

import copy

import pytest
import torch

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components.models.common import cudnn_sparse_attention as shared_cudnn
from nemo_automodel.components.models.common.cudnn_sparse_attention import is_cudnn_sparse_attention_available
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext, build_document_causal_mask
from nemo_automodel.components.models.kimi_k3.model import KimiMLAAttention
from tests.unit_tests.models.kimi_k3.test_dsa import _backend, _tiny_dsa_config, _tiny_text_config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] != 10
    or not is_cudnn_sparse_attention_available(),
    reason="requires SM100 and cuDNN/FlashMLA sparse attention",
)


def _config() -> KimiK3TextConfig:
    config = _tiny_text_config(_tiny_dsa_config(index_topk=64, index_num_heads=32, index_head_dim=128))
    config.hidden_size = 64
    config.num_attention_heads = 96
    config.num_key_value_heads = 96
    config.q_lora_rank = 32
    config.kv_lora_rank = 512
    config.qk_nope_head_dim = 128
    config.qk_rope_head_dim = 64
    config.v_head_dim = 128
    config.torch_dtype = torch.bfloat16
    return config


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12)).item()


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_cudnn_attention_forward_backward(padding: bool, lora: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    head_counts = []
    original_pad = shared_cudnn._pad_attention_heads

    def observe_heads(query: torch.Tensor, sink: torch.Tensor, kernel_heads: int) -> tuple[torch.Tensor, torch.Tensor]:
        head_counts.append((query.shape[1], kernel_heads))
        return original_pad(query, sink, kernel_heads)

    monkeypatch.setattr(shared_cudnn, "_pad_attention_heads", observe_heads)
    torch.manual_seed(4645)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = _config()
    attention = KimiMLAAttention(config, 1, _backend("cudnn")).cuda()
    attention.init_weights(torch.device("cuda"), 0.03)
    if lora:
        attention.kv_b_proj = LinearLoRA(attention.kv_b_proj, dim=4, alpha=8, lora_dtype=torch.bfloat16)
        with torch.no_grad():
            attention.kv_b_proj.lora_B.weight.normal_(0, 0.03)
    reference = KimiMLAAttention(config, 1, _backend("eager")).cuda()
    if lora:
        reference.kv_b_proj = LinearLoRA(reference.kv_b_proj, dim=4, alpha=8, lora_dtype=torch.bfloat16)
    reference.load_state_dict(attention.state_dict())
    x = torch.randn(1, 256, config.hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    doc_ids = torch.ones(1, 256, device="cuda", dtype=torch.int32)
    if padding:
        doc_ids[:, 192:] = 0
    context = KimiPackedContext(doc_ids)
    metadata = context.cudnn_dsa_metadata(0)
    with torch.no_grad():
        query_input, _ = attention._project_queries(x)
        keys = attention.indexer.project_keys(x)
        indices = attention.indexer.select_indices(x[0], query_input[0], keys[0], metadata)
        queries = attention.indexer.wq_b(query_input).view(1, 256, 32, 128).float()
        weights = attention.indexer.weights_proj(x).float() * attention.indexer.score_scale
        scores = torch.einsum("bqhd,bkd->bqhk", queries, keys.float()).relu()
        scores = torch.einsum("bqhk,bqh->bqk", scores, weights)[0]
        positions = torch.arange(256, device="cuda")
        allowed = (positions[:, None] >= positions[None, :]) & (doc_ids[0, :, None] > 0) & (doc_ids[0, None, :] > 0)
        scores.masked_fill_(~allowed, -torch.inf)
        oracle = scores.topk(64, dim=-1)
        selected = torch.zeros_like(allowed, dtype=torch.int32)
        selected.scatter_add_(1, indices[:, 0].long().clamp_min(0), (indices[:, 0] >= 0).to(torch.int32))
        selected = selected > 0
        expected = torch.zeros_like(allowed).scatter_(1, oracle.indices, oracle.values.isfinite())
        agreement = (selected & expected).sum(-1).float() / expected.sum(-1).clamp_min(1)
        sparse_rows = allowed.sum(-1) > 64
        print("selection_agreement", agreement[sparse_rows].mean().item())
        assert agreement[sparse_rows].mean() > 0.995
        assert not (selected & ~allowed).any()
        assert (selected.sum(-1) == allowed.sum(-1).clamp_max(64)).all()

    # Hold the discrete selection fixed to isolate latent absorption and kernel gradients.
    reference.indexer.select = lambda *args, **kwargs: selected[None]
    oracle_module = copy.deepcopy(reference).float()
    oracle_x = x.detach().float().requires_grad_()
    mask = build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=x.dtype)
    oracle_mask = build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=torch.float32)
    oracle_output = oracle_module(oracle_x, attention_mask=oracle_mask)
    actual = attention(x, packed_context=context)
    expected = reference(ref_x, attention_mask=mask)
    valid = doc_ids > 0
    forward_error = _relative_error(actual[valid], expected[valid])
    print("forward_relative_error", forward_error)
    assert forward_error < 0.015
    upstream = torch.randn_like(actual) * valid.unsqueeze(-1)
    actual.backward(upstream)
    assert head_counts == [(96, 128), (96, 96)]
    expected.backward(upstream)
    oracle_output.backward(upstream.float())
    input_error = _relative_error(x.grad, ref_x.grad)
    print("input_gradient_relative_error", input_error)
    assert input_error < 0.03
    ref_parameters = dict(reference.named_parameters())
    oracle_parameters = dict(oracle_module.named_parameters())
    comparisons = [
        ("output", actual[valid], expected[valid], oracle_output[valid]),
        ("input gradient", x.grad, ref_x.grad, oracle_x.grad),
    ]
    for name, parameter in attention.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            continue
        assert torch.isfinite(parameter.grad).all(), name
        error = _relative_error(parameter.grad, ref_parameters[name].grad)
        print(name, error)
        assert error < 0.04, (name, error)
        comparisons.append((name, parameter.grad, ref_parameters[name].grad, oracle_parameters[name].grad))
    # Judge BF16 absorption against the independent expanded FP32 computation and
    # the existing BF16 path's error, rather than treating every BF16 gap as noise.
    for name, actual_value, eager_value, oracle_value in comparisons:
        error = _relative_error(actual_value, oracle_value)
        eager_error = _relative_error(eager_value, oracle_value)
        print("fp32_oracle", name, error, eager_error)
        assert error <= 2 * eager_error + 2e-3, (name, error, eager_error)
    if lora:
        assert attention.kv_b_proj.lora_A.weight.grad.abs().sum() > 0
        assert attention.kv_b_proj.lora_B.weight.grad.abs().sum() > 0
    if padding:
        assert torch.count_nonzero(actual[~valid]) == 0
        assert torch.count_nonzero(x.grad[~valid]) == 0


def test_cudnn_empty_shard_with_transposed_output_gradient() -> None:
    """An empty CP shard must accept the head-major gradient from V expansion."""
    torch.manual_seed(4645)
    query = torch.randn(64, 96, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    latent = torch.randn(64, 1, 576, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    indices = torch.full((64, 1, 64), -1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(64, device="cuda", dtype=torch.int32)
    valid_rows = torch.empty(0, device="cuda", dtype=torch.int64)
    output = shared_cudnn.cudnn_sparse_attention(
        query, latent, indices, 192**-0.5, topk_length=lengths, valid_row_indices=valid_rows
    )
    upstream = torch.randn(96, 64, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    assert not upstream.is_contiguous()
    output.backward(upstream)
    assert torch.count_nonzero(output) == 0
    assert torch.count_nonzero(query.grad) == 0
    assert torch.count_nonzero(latent.grad) == 0

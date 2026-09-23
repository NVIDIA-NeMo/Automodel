# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.layers import Glm5NextKPoolIndexer, _indexer_fp8_qdq


def _reference_qdq(x: torch.Tensor) -> torch.Tensor:
    """Independent dense-matrix reference for the Indexer QDQ.

    Args:
        x: Tensor of shape [..., 128], with arbitrary leading dimensions.

    Returns:
        FP32 tensor of shape [..., 128] in the normalized Hadamard basis.
    """
    h = torch.ones(1, 1, device=x.device)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    rotated = (x.bfloat16().float() @ h / 128**0.5).bfloat16().float()
    scale = 2.0 ** torch.ceil(torch.log2(rotated.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
    return (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale


@pytest.mark.parametrize("shape", [(0, 128), (7, 128), (1, 5, 2, 128)])
def test_qdq_matches_dense_hadamard_reference(shape):
    torch.manual_seed(3)
    # Binary fractions give exact FP32 sums in either reduction order.
    x = torch.randint(-64, 64, shape).float() / 16
    original = x.clone()
    actual = _indexer_fp8_qdq(x)
    torch.testing.assert_close(actual, _reference_qdq(x), rtol=0, atol=0)
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert actual.dtype == torch.float32


def test_qdq_zero_and_small_vectors_are_finite():
    x = torch.zeros(3, 128)
    x[1, 0] = 1e-10
    x[2, 0] = 1e3
    result = _indexer_fp8_qdq(x)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[0]) == 0
    torch.testing.assert_close(result, _reference_qdq(x), rtol=0, atol=0)


@pytest.mark.parametrize("kwargs", [{"index_head_dim": 64}, {"qk_rope_head_dim": 64}])
def test_qdq_rejects_unsupported_layout(kwargs):
    with pytest.raises(ValueError, match="indexer_fp8_fake_quant requires"):
        Glm5NextTextConfig(indexer_fp8_fake_quant=True, **kwargs)


@pytest.mark.parametrize("enabled", [False, True])
def test_indexer_selection_matches_reference_and_preserves_weights(enabled):
    torch.manual_seed(13)
    config = Glm5NextTextConfig(
        hidden_size=16,
        q_lora_rank=8,
        index_head_dim=128,
        index_n_heads=2,
        index_kpool=2,
        index_topk=4,
        indexer_fp8_fake_quant=enabled,
    )
    assert Glm5NextTextConfig.from_dict(config.to_dict()).indexer_fp8_fake_quant is enabled
    model = Glm5NextKPoolIndexer(config, 3, torch.bfloat16)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    hidden = torch.randn(1, 13, 16).bfloat16().requires_grad_()
    resid = torch.randn(1, 13, 8).bfloat16().requires_grad_()
    positions = torch.arange(13)
    pools, pool_indices = model.prepare_pools(hidden)
    actual = model.select(hidden, resid, positions, pools, pool_indices, 13)
    with torch.no_grad():
        keys = model.k_norm(model.wk(hidden))[0, :12].reshape(6, 2, 128)
        gates = F.linear(hidden[0], model.index_kpool_compress_gate)[:12].reshape(6, 2, 128)
        probs = (gates.float() + model.index_kpool_compress_ape.float()).softmax(1)
        reference_pools = (probs * keys.float()).sum(1) if enabled else (probs.bfloat16() * keys).sum(1)
        q = model.wq_b(resid).reshape(13, 2, 128)
        if enabled:
            q, reference_pools = _reference_qdq(q), _reference_qdq(reference_pools)
        torch.testing.assert_close(pools, reference_pools, rtol=0, atol=0)
        weights = (
            F.linear(hidden.float(), model.weights_proj.weight.float())
            if enabled
            else model.weights_proj(hidden).float()
        )
        scores = ((q.float() @ reference_pools.float().T) * 128**-0.5).relu()
        scores = (scores * (weights[0] * 2**-0.5).unsqueeze(-1)).sum(1)
        for t in range(13):
            visible = (t + 1) // 2
            expected = []
            if visible:
                chosen = scores[t, :visible].topk(min(2, visible)).indices
                expected = pool_indices[chosen].flatten().tolist()
            if (t + 1) % 2:
                expected.append(t)
            assert sorted(actual[0, t][actual[0, t] >= 0].tolist()) == sorted(expected)
    assert not pools.requires_grad and not actual.requires_grad
    assert before.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    empty_pools, empty_indices = model.prepare_pools(hidden[:, :1])
    tail = model.select(hidden[:, :1], resid[:, :1], positions[:1], empty_pools, empty_indices, 1)
    assert tail[tail >= 0].tolist() == [0]


def test_qdq_enabled_hybrid_model_backward_is_finite():
    from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
    from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config

    torch.manual_seed(7)
    config = tiny_glm5_next_config()
    config.text_config.index_head_dim = 128
    config.text_config.indexer_fp8_fake_quant = True
    model = Glm5NextForConditionalGeneration(config, backend=tiny_backend()).train()
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    documents = torch.ones_like(tokens, dtype=torch.int32)
    logits = model(input_ids=tokens, attention_mask=documents).logits
    logits.square().mean().backward()
    assert torch.isfinite(logits).all()
    layer = model.model.language_model.layers["3"].self_attn
    assert layer.indexer.indexer_fp8_fake_quant
    assert layer.q_a_proj.weight.grad is not None
    assert torch.isfinite(layer.q_a_proj.weight.grad).all()
    assert layer.indexer.wq_b.weight.grad is None


def test_qdq_pooling_respects_per_slot_gate_before_quantization():
    torch.manual_seed(5)
    config = Glm5NextTextConfig(
        hidden_size=16,
        q_lora_rank=8,
        index_head_dim=128,
        index_n_heads=2,
        index_kpool=2,
        index_topk=4,
        indexer_fp8_fake_quant=True,
    )
    model = Glm5NextKPoolIndexer(config, 3, torch.bfloat16)
    with torch.no_grad():
        # Saturate the gate so every completed pool selects its first raw key.
        model.index_kpool_compress_ape[0].fill_(1000)
        model.index_kpool_compress_ape[1].fill_(-1000)
        hidden = torch.randn(1, 7, 16).bfloat16()
        expected = _reference_qdq(model.k_norm(model.wk(hidden))[0, :6:2])
    pools, indices = model.prepare_pools(hidden)
    torch.testing.assert_close(pools, expected, rtol=0, atol=0)
    assert indices.tolist() == [[0, 1], [2, 3], [4, 5]]

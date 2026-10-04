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

from unittest.mock import patch

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer as HFGlmMoeDsaIndexer

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaIndexer


def _config(interleaved, rope_dim):
    overrides = {} if interleaved is None else {"indexer_rope_interleave": interleaved}
    return GlmMoeDsaConfig(
        hidden_size=16,
        num_hidden_layers=2,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=16,
        qk_rope_head_dim=rope_dim,
        index_topk=2,
        # The indexer's layout is independent of the main attention's setting.
        rope_interleave=not interleaved,
        torch_dtype="float32",
        **overrides,
    )


@pytest.fixture
def backend():
    return BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", experts="torch", dispatcher="torch")


def _reference_scores(indexer, x, q_resid, angles, *, interleaved):
    """Compute index scores using explicit two-dimensional rotation matrices.

    Args:
        indexer: Indexer supplying the projection weights and normalization.
        x: Tensor of shape [batch, sequence, hidden].
        q_resid: Tensor of shape [batch, sequence, q_lora_rank].
        angles: Tensor of shape [batch, sequence, rotary_pairs].
        interleaved: Whether a rotation pairs adjacent channels or opposite halves.

    Returns:
        Causally masked score tensor of shape [batch, sequence, sequence].
    """
    batch, sequence, pairs = angles.shape
    rotation = torch.eye(indexer.head_dim).expand(batch, sequence, -1, -1).clone()
    for pair in range(pairs):
        first, second = (2 * pair, 2 * pair + 1) if interleaved else (pair, pair + pairs)
        rotation[..., first, first] = angles[..., pair].cos()
        rotation[..., first, second] = -angles[..., pair].sin()
        rotation[..., second, first] = angles[..., pair].sin()
        rotation[..., second, second] = angles[..., pair].cos()

    q = indexer.wq_b(q_resid).reshape(batch, sequence, indexer.num_heads, indexer.head_dim)
    k = indexer.k_norm(indexer.wk(x))
    q = torch.einsum("bsij,bshj->bshi", rotation, q)
    k = torch.einsum("bsij,bsj->bsi", rotation, k)
    per_head = torch.einsum("bshd,btd->bsht", q, k).mul(indexer.softmax_scale).relu()
    weights = indexer.weights_proj(x) * indexer.num_heads**-0.5
    scores = torch.einsum("bsh,bsht->bst", weights, per_head)
    future = torch.ones(sequence, sequence, dtype=torch.bool).triu(1)
    return scores.masked_fill(future, float("-inf"))


@pytest.mark.parametrize("interleaved", [True, False, None], ids=["interleaved", "half-split", "default"])
@pytest.mark.parametrize("qkv_format", ["bshd", "thd"])
@pytest.mark.parametrize("rope_dim", [4, 8])
def test_indexer_rope_scores_selection_and_gradients(backend, interleaved, qkv_format, rope_dim):
    torch.manual_seed(17)
    config = _config(interleaved, rope_dim)
    indexer = GlmMoeDsaIndexer(config, backend)
    batch, sequence = (2 if qkv_format == "bshd" else 1), 8
    x = torch.randn(batch, sequence, config.hidden_size, requires_grad=True)
    q_resid = torch.randn(batch, sequence, config.q_lora_rank, requires_grad=True)
    positions = torch.arange(1, sequence + 1, dtype=torch.float32).expand(batch, -1)
    frequencies = torch.linspace(0.15, 1.0, rope_dim // 2)
    angles = positions.unsqueeze(-1) * frequencies
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    # Observe the real dense scores without replacing their computation or top-k.
    captured = []
    tensor_topk = torch.Tensor.topk

    def capture_scores(scores, *args, **kwargs):
        """Record the score input and delegate to the real top-k operation.

        Args:
            scores: Tensor of shape [batch, sequence, sequence] or [tokens, tokens].
            *args: Positional arguments for Tensor.topk.
            **kwargs: Keyword arguments for Tensor.topk.

        Returns:
            Top-k values and indices with the input's leading dimensions.
        """
        captured.append(scores)
        return tensor_topk(scores, *args, **kwargs)

    with patch.object(torch.Tensor, "topk", capture_scores):
        if qkv_format == "thd":
            selected = indexer(x.squeeze(0), q_resid.squeeze(0), freqs_cis.squeeze(0)).unsqueeze(0)
        else:
            selected = indexer(x, q_resid, freqs_cis)
    assert len(captured) == 1
    actual = captured[0].reshape(batch, sequence, sequence)
    expected = _reference_scores(indexer, x, q_resid, angles, interleaved=interleaved is not False)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    # Past the top-k boundary, matching all keys would hide a wrong rotary layout.
    assert sequence > config.index_topk
    expected_selected = expected.topk(config.index_topk, dim=-1).indices
    torch.testing.assert_close(selected[:, config.index_topk :], expected_selected[:, config.index_topk :])

    upstream = torch.randn_like(expected)
    future = torch.ones(sequence, sequence, dtype=torch.bool).triu(1)
    parameters = (x, q_resid, *indexer.parameters())
    actual_gradients = torch.autograd.grad((actual.masked_fill(future, 0) * upstream).sum(), parameters)
    expected_gradients = torch.autograd.grad((expected.masked_fill(future, 0) * upstream).sum(), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-6, rtol=1e-5)


def test_interleaved_indexer_matches_transformers(backend):
    torch.manual_seed(19)
    config = _config(True, 8)
    indexer = GlmMoeDsaIndexer(config, backend)
    reference = HFGlmMoeDsaIndexer(config, layer_idx=0)
    reference.load_state_dict(indexer.state_dict(), strict=True)
    batch, sequence = 2, 9
    x = torch.randn(batch, sequence, config.hidden_size)
    q_resid = torch.randn(batch, sequence, config.q_lora_rank)
    positions = torch.arange(sequence).expand(batch, -1)
    angles = positions.unsqueeze(-1) * torch.tensor([1.0, 0.7, 0.3, 0.1])
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    # HF's table repeats the per-pair angles across two halves. Its output channel
    # order differs, but the same permutation on Q and K preserves their scores.
    doubled_angles = torch.cat((angles, angles), dim=-1)
    embeddings = (doubled_angles.cos(), doubled_angles.sin())

    actual = indexer(x, q_resid, freqs_cis)
    expected = reference(x, q_resid, embeddings, attention_mask=None, position_ids=positions)

    torch.testing.assert_close(actual[:, config.index_topk :], expected[:, config.index_topk :].long())

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

"""Additional CSA2 lifecycle and training contracts not covered by the reference sweeps."""

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionState,
    _Indexer,
    _select_candidate_blocks,
)
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
from tests.unit_tests.models.deepseek_v41.test_attention import _backend, _config


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_attention_dropout_is_training_only_and_keeps_gradients_finite(backend):
    torch.manual_seed(107)
    config = _config()
    config.attention_dropout = 0.5
    layer = DeepseekV41Attention(config, 0, _backend(backend))
    inputs = torch.randn(2, 8, config.hidden_size, requires_grad=True)
    kwargs = dict(position_ids=torch.arange(8)[None], state=DeepseekV41AttentionState())
    layer.eval()
    expected = layer(inputs, **kwargs).hidden_states
    torch.testing.assert_close(layer(inputs, **kwargs).hidden_states, expected, atol=0, rtol=0)
    layer.train()
    torch.manual_seed(108)
    actual = layer(inputs, **kwargs).hidden_states
    assert not torch.equal(actual, expected)
    torch.manual_seed(108)
    torch.testing.assert_close(layer(inputs, **kwargs).hidden_states, actual, atol=0, rtol=0)
    actual.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert layer.sinks_param.weight.grad is not None
    assert torch.isfinite(layer.sinks_param.weight.grad).all()
    layer.eval()
    torch.testing.assert_close(layer(inputs, **kwargs).hidden_states, expected, atol=0, rtol=0)


def test_tilelang_requires_zero_dropout_and_attention_requires_supported_projections():
    config = _config()
    config.attention_dropout = 0.1
    with pytest.raises(ValueError, match="attention_dropout=0"):
        DeepseekV41Attention(config, 0, _backend("tilelang"))
    config.attention_dropout = 0
    with pytest.raises(ValueError, match="torch linear"):
        DeepseekV41Attention(config, 0, replace(_backend(), linear="te"))
    with pytest.raises(ValueError, match="torch_fp32"):
        DeepseekV41Attention(config, 0, replace(_backend(), rms_norm="torch"))


@pytest.mark.parametrize("batch,sequence", [(2, 6), (1, 5)])
def test_reuse_rejects_state_from_another_batch_or_sequence(batch, sequence):
    config = _config()
    full = DeepseekV41Attention(config, 1, _backend())
    reuse = DeepseekV41Attention(config, 2, _backend())
    source = full(
        torch.randn(1, 6, config.hidden_size),
        position_ids=torch.arange(6)[None],
        state=DeepseekV41AttentionState(),
    )
    with pytest.raises(ValueError, match="different batch or sequence"):
        reuse(
            torch.randn(batch, sequence, config.hidden_size),
            position_ids=torch.arange(sequence)[None],
            state=source.state,
        )
    with pytest.raises(ValueError, match="same compression ratio"):
        reuse(
            torch.randn(1, 6, config.hidden_size),
            position_ids=torch.arange(6)[None],
            state=replace(source.state, compression_ratio=1),
        )
    with pytest.raises(FrozenInstanceError):
        source.state.compression_ratio = 1


def test_indexer_rejects_missing_latent_keys_and_candidates():
    config = _config()
    x = torch.randn(1, 4, config.hidden_size)
    kwargs = dict(
        query_latent=torch.randn(1, 4, config.q_lora_rank),
        latent=None,
        angles=torch.zeros(1, 4, config.qk_rope_head_dim // 2),
        compressed_angles=torch.zeros(1, 3, config.qk_rope_head_dim // 2),
    )
    full = _Indexer(config, layer_idx=1, dtype=torch.float32)
    reindex = _Indexer(config, layer_idx=4, dtype=torch.float32)
    with pytest.raises(ValueError, match="unrotated compressed latent"):
        full(x, **kwargs, state=DeepseekV41AttentionState())
    with pytest.raises(ValueError, match="index keys"):
        reindex(x, **kwargs, state=DeepseekV41AttentionState())
    state = DeepseekV41AttentionState(compression_ratio=1, index_keys=torch.randn(1, 3, config.index_head_dim))
    with pytest.raises(ValueError, match="requires candidates"):
        reindex(x, **kwargs, state=state)
    with pytest.raises(ValueError, match="requires candidates"):
        reindex(x, **kwargs, state=replace(state, candidates=torch.ones(1, 4, 2, dtype=torch.bool)))


def test_candidate_blocks_with_no_visible_keys():
    scores = torch.full((2, 3, 7), -torch.inf)
    lengths = torch.zeros(2, 3, 1, dtype=torch.long)
    assert not _select_candidate_blocks(scores, lengths, topk_blocks=2, block_size=4).any()


@pytest.mark.parametrize("format,block_size", [("int8", 32), ("fp8", 0), ("nvfp4", -1)])
def test_cache_quantization_rejects_unsupported_format_or_group_size(format, block_size):
    with pytest.raises(ValueError, match="positive block_size"):
        quantize_cache(torch.zeros(2, 20), format=format, block_size=block_size)

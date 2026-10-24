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

"""Behavioral regressions for DeepSeek V3.2 causal and padding masks."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v32 import layers
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.layers import DeepseekV32MLA


def _cpu_hadamard(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Use a dense orthogonal transform in CPU mask tests.

    Args:
        x: Tensor of shape [..., channels], with a power-of-two final dimension.
        scale: Scalar multiplier applied after the transform.

    Returns:
        Tensor of shape [..., channels], on the input device and in its dtype.
    """
    matrix = torch.ones(1, 1, device=x.device)
    while matrix.shape[0] < x.shape[-1]:
        matrix = torch.cat((torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)), dim=0)
    return (x.float() @ matrix * scale).to(x.dtype)


@pytest.fixture
def mla(monkeypatch: pytest.MonkeyPatch) -> DeepseekV32MLA:
    # Only the optional CUDA Hadamard kernel is replaced; projections, indexer,
    # mask construction, SDPA and autograd execute their actual implementations.
    monkeypatch.setattr(layers, "hadamard_transform", _cpu_hadamard)
    torch.manual_seed(17)
    config = DeepseekV32Config(
        hidden_size=16,
        num_attention_heads=2,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_head_dim=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        torch_dtype="float32",
        max_position_embeddings=32,
    )
    module = DeepseekV32MLA(config, BackendConfig(attn="sdpa", linear="torch", rms_norm="torch"))
    module.float()
    module.init_weights(torch.device("cpu"))
    return module


@pytest.mark.parametrize("mask_kind", ["none", "bool", "int", "float", "bool_4d"])
def test_indexer_respects_causality_and_padding(mla: DeepseekV32MLA, mask_kind: str) -> None:
    """Selected keys match an independently constructed causal mask after top-k activates."""
    batch, length = 2, 12
    x = torch.randn(batch, length, 16)
    q_resid = torch.randn(batch, length, 8) * 100
    freqs = torch.polar(torch.ones(batch, length, 2), torch.randn(batch, length, 2))
    keep = torch.ones(batch, length, dtype=torch.bool)
    if mask_kind != "none":
        keep[0, :2] = False
        keep[1, -2:] = False
    allowed = torch.tensor(
        [
            [[key <= query and bool(keep[b, key]) for key in range(length)] for query in range(length)]
            for b in range(batch)
        ]
    )
    oracle_mask = torch.zeros(batch, 1, length, length).masked_fill(~allowed[:, None], -torch.inf)
    if mask_kind == "none":
        mask = None
    elif mask_kind == "bool_4d":
        mask = allowed[:, None]
    else:
        dtype = {"bool": torch.bool, "int": torch.int64, "float": torch.float32}[mask_kind]
        mask = keep.to(dtype)
    actual = mla.indexer(x, q_resid, freqs, attention_mask=mask)
    expected = mla.indexer(x, q_resid, freqs, attention_mask=oracle_mask)
    # Before there are K valid keys, topk returns masked filler indices too.
    # The final attention mask must remove those; do not mistake them for selection errors.
    sparse_queries = allowed.sum(-1) > mla.index_topk
    assert sparse_queries.sum() > 0
    torch.testing.assert_close(actual.sort(-1).values[sparse_queries], expected.sort(-1).values[sparse_queries])
    assert allowed.gather(-1, actual)[sparse_queries].all()


@pytest.mark.parametrize("as_bool", [True, False])
def test_short_sequence_topk_fillers_remain_masked(mla: DeepseekV32MLA, as_bool: bool) -> None:
    length = 3
    indices = torch.arange(length).expand(2, length, length)
    mask = mla._build_sparse_mask(indices, length, "bshd", bsz=2, dtype=torch.float32, as_bool=as_bool)
    allowed = mask if as_bool else torch.isfinite(mask)
    expected = torch.tensor([[True, False, False], [True, True, False], [True, True, True]])
    torch.testing.assert_close(allowed, expected.expand(2, 1, length, length))


@pytest.mark.parametrize("dtype", [torch.bool, torch.int64, torch.float32])
@pytest.mark.parametrize("as_bool", [True, False])
def test_padding_masks_include_all_zero_rows(mla: DeepseekV32MLA, dtype: torch.dtype, as_bool: bool) -> None:
    length = 6
    keep = torch.tensor([[0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=dtype)
    indices = torch.arange(length).expand(3, length, length)
    mask = mla._build_sparse_mask(
        indices, length, "bshd", bsz=3, dtype=torch.float32, attention_mask=keep, as_bool=as_bool
    )
    allowed = mask if as_bool else torch.isfinite(mask)
    expected = torch.tensor(
        [
            [[[key <= query and bool(keep[b, key]) for key in range(length)] for query in range(length)]]
            for b in range(3)
        ]
    )
    torch.testing.assert_close(allowed, expected)


@pytest.mark.parametrize("masked_value", [-torch.inf, torch.finfo(torch.float32).min])
def test_additive_mask_is_preserved(mla: DeepseekV32MLA, masked_value: float) -> None:
    length = 6
    indices = torch.arange(length).expand(2, length, length)
    mask = torch.zeros(2, 1, length, length)
    mask[:, :, :, 1] = masked_value
    actual = mla._build_sparse_mask(indices, length, "bshd", bsz=2, attention_mask=mask, as_bool=True)
    expected = torch.tensor([[key <= query and key != 1 for key in range(length)] for query in range(length)])
    torch.testing.assert_close(actual, expected.expand(2, 1, length, length))


@pytest.mark.parametrize("topk", [4, 16])
def test_mla_future_tokens_cannot_change_prefix_or_receive_its_gradient(mla: DeepseekV32MLA, topk: int) -> None:
    mla.indexer.index_topk = topk
    x = torch.randn(2, 12, 16, requires_grad=True)
    freqs = torch.polar(torch.ones(2, 12, 2), torch.randn(2, 12, 2))
    output = mla(x, freqs)
    changed = x.detach().clone()
    changed[:, 8:] = torch.randn_like(changed[:, 8:]) * 10
    torch.testing.assert_close(output[:, :8], mla(changed, freqs)[:, :8], rtol=0, atol=0)
    upstream = torch.randn_like(output[:, :8])
    (output[:, :8] * upstream).sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[:, 8:]) == 0
    assert torch.count_nonzero(x.grad[:, :8]) > 0


def test_mla_padding_values_cannot_change_valid_outputs(mla: DeepseekV32MLA) -> None:
    x = torch.randn(2, 12, 16)
    freqs = torch.polar(torch.ones(2, 12, 2), torch.randn(2, 12, 2))
    keep = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.float32
    )
    output = mla(x, freqs, attention_mask=keep)
    changed = x.clone()
    changed[~keep.bool()] = torch.randn_like(changed[~keep.bool()]) * 100
    other = mla(changed, freqs, attention_mask=keep)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[keep.bool()], other[keep.bool()], rtol=0, atol=0)


@pytest.mark.parametrize("length", [1, 12])
def test_thd_additive_mask_keeps_single_token_and_causal_edges(mla: DeepseekV32MLA, length: int) -> None:
    indices = torch.arange(length).expand(length, length)
    mask = mla._build_sparse_mask(indices, length, "thd", attention_mask=torch.zeros(length, length), as_bool=True)
    expected = torch.tensor([[key <= query for key in range(length)] for query in range(length)])
    torch.testing.assert_close(mask, expected[None, None])

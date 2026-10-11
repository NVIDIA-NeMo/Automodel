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

"""Independent operation-count and packed-boundary checks for Step-5 MFU."""

import pytest
import torch

from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.flops import step5_batch_flops, step5_flops
from nemo_automodel.components.utils.flops_utils import get_batch_flops_counter, get_flops_formula_for_hf_config


def _config(*, sparse_enabled: bool = True) -> Step5Config:
    return Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        sliding_window=3,
        num_nextn_predict_layers=0,
        sparse_config=dict(
            enabled=sparse_enabled,
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=2,
        ),
    )


def _enumerated_flops(config: Step5Config, length: int, supervised: int) -> int:
    """Enumerate matrix products and actual selected keys, without production count helpers."""
    total = 6 * supervised * config.hidden_size * config.vocab_size
    sparse = config.sparse_config
    for layer in range(config.num_hidden_layers):
        shapes = [
            (config.hidden_size, config.num_attention_heads * config.head_dim),
            (config.hidden_size, config.num_attention_groups * config.head_dim),
            (config.hidden_size, config.num_attention_groups * config.head_dim),
            (config.num_attention_heads * config.head_dim, config.hidden_size),
            (config.hidden_size, config.num_attention_heads),
        ]
        widths = (
            [config.moe_intermediate_size] * config.moe_top_k + [config.share_expert_dim]
            if layer in config.moe_layer_list
            else [config.intermediate_size]
        )
        if layer in config.moe_layer_list:
            shapes.append((config.hidden_size, config.moe_num_experts))
        for width in widths:
            shapes.extend([(config.hidden_size, width), (config.hidden_size, width), (width, config.hidden_size)])
        total += sum(6 * length * rows * columns for rows, columns in shapes)
        sparse_layer = sparse.enabled and config.layer_types[layer] == "full_attention"
        if sparse_layer:
            index_shapes = [
                (config.hidden_size, sparse.sparse_indexer_num_heads * sparse.proxy_dim),
                (config.hidden_size, sparse.proxy_dim),
                (config.hidden_size, sparse.proxy_dim),
                (config.hidden_size, sparse.sparse_indexer_num_heads),
            ]
            total += sum(2 * length * rows * columns for rows, columns in index_shapes)
        for query in range(length):
            causal = list(range(query + 1))
            if config.layer_types[layer] == "sliding_attention":
                keys = [key for key in causal if query - key < config.sliding_window]
            elif sparse_layer:
                historical_regions = sorted(
                    {
                        key // sparse.region_block_size
                        for key in causal
                        if key // sparse.region_block_size < query // sparse.region_block_size
                    }
                )
                selected = historical_regions[-sparse.topk :] + [query // sparse.region_block_size]
                keys = [key for key in causal if key // sparse.region_block_size in selected]
                total += 2 * len(historical_regions) * sparse.sparse_indexer_num_heads * sparse.proxy_dim
            else:
                keys = causal
            total += 2 * 6 * len(keys) * config.num_attention_heads * config.head_dim
    return total


@pytest.mark.parametrize("sparse_enabled", [False, True])
@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 6, 7, 11, 17])
def test_shape_formula_matches_independently_enumerated_operations(length: int, sparse_enabled: bool) -> None:
    cfg = _config(sparse_enabled=sparse_enabled)
    assert step5_flops(cfg, gbs=3, seq_len=length) == 3 * _enumerated_flops(cfg, length, length)


def test_registry_uses_the_model_owned_formula_and_batch_counter() -> None:
    cfg = _config()
    assert get_flops_formula_for_hf_config(cfg) is step5_flops
    counter = get_batch_flops_counter(cfg)
    assert counter is not None
    batch = {"input_ids": torch.zeros(1, 7, dtype=torch.long), "labels": torch.zeros(1, 7, dtype=torch.long)}
    assert counter(batch, ignore_index=-100).flops == _enumerated_flops(cfg, 7, 7)


def test_packed_documents_and_supervised_head_ignore_physical_padding() -> None:
    cfg = _config()
    positions = torch.tensor([[0, 1, 2, 0, 0, 1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 2, 3, 4, 0, 0, 0, 0]])
    valid = torch.tensor([[1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2], [1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0]])
    labels = torch.tensor([[1, -100, 2, -100, 1, -7, 2, 1, 2, 1, 1], [1, -7, 2, 1, -100, 2, 1, -100, -100, -100, -100]])
    batch = {
        "input_ids": torch.zeros_like(positions),
        "labels": labels,
        "position_ids": positions,
        "attention_mask": valid,
    }
    originals = {name: tensor.clone() for name, tensor in batch.items()}
    result = step5_batch_flops(cfg, batch, ignore_index=-7)
    assert result.num_tokens == 17
    assert result.num_supervised_tokens == 13
    expected = sum(_enumerated_flops(cfg, length, length) for length in [3, 7, 2, 5])
    assert result.all_token_head_flops == expected
    assert result.flops == expected - 6 * 4 * cfg.hidden_size * cfg.vocab_size
    assert result.all_token_head_flops < step5_flops(cfg, seq_len=17)
    for name, tensor in batch.items():
        torch.testing.assert_close(tensor, originals[name])


def test_counter_rejects_post_cp_fragments_and_supervised_padding() -> None:
    cfg = _config()
    batch = {
        "input_ids": torch.zeros(1, 3, dtype=torch.long),
        "labels": torch.zeros(1, 3, dtype=torch.long),
        "position_ids": torch.tensor([[8, 9, 10]]),
    }
    with pytest.raises(ValueError, match="before context-parallel"):
        step5_batch_flops(cfg, batch, ignore_index=-100)
    batch["position_ids"] = torch.tensor([[0, 1, 2]])
    batch["attention_mask"] = torch.tensor([[1, 1, 0]])
    with pytest.raises(ValueError, match="padding"):
        step5_batch_flops(cfg, batch, ignore_index=-100)


def test_full_checkpoint_geometry_long_single_document_count() -> None:
    cfg = Step5Config(num_nextn_predict_layers=0)
    assert cfg.hidden_size == 4096 and cfg.num_attention_heads == 64
    assert step5_flops(cfg, gbs=8, seq_len=131072) == 186839187784204288.0

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

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.llava_onevision.rice_vit import (
    RiceAttention,
    RiceSdpaAttention,
    RiceTransformer,
    _block_diagonal_attention_mask,
)


def _tiny_config() -> SimpleNamespace:
    return SimpleNamespace(
        spatial_merge_size=1,
        patch_size=1,
        temporal_patch_size=1,
        in_channels=3,
        hidden_size=8,
        text_hidden_size=8,
        num_heads=2,
        intermediate_size=16,
        hidden_act="gelu",
        depth=0,
        layer_norm_eps=1e-5,
    )


def test_block_diagonal_attention_mask():
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, False, False],
            [True, True, True, False, False],
            [False, False, False, True, True],
            [False, False, False, True, True],
        ]
    ).unsqueeze(0)

    boolean_mask = _block_diagonal_attention_mask(cu_seqlens, sequence_length=5)
    additive_mask = _block_diagonal_attention_mask(cu_seqlens, sequence_length=5, dtype=torch.float32)

    assert torch.equal(boolean_mask, expected)
    assert torch.equal(additive_mask == 0, expected)
    assert torch.equal(additive_mask == torch.finfo(torch.float32).min, ~expected)


def test_cls_insertion_and_removal_preserves_patch_order():
    model = RiceTransformer(_tiny_config())
    pixel_values = torch.randn(3, 3, 1, 1, requires_grad=True)
    grid_thw = torch.tensor([[1, 1, 2], [1, 1, 1]], dtype=torch.long)

    output = model(pixel_values, grid_thw)
    patch_embeddings = model.patch_embed(pixel_values)
    expected = model.merger(model.pre_layernorm(patch_embeddings))

    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert pixel_values.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cross-device coverage")
@pytest.mark.parametrize("attention_type", [RiceAttention, RiceSdpaAttention])
def test_attention_accepts_cpu_offsets_with_cuda_inputs(attention_type):
    attention = attention_type(dim=8, num_heads=2).cuda()
    hidden_states = torch.randn(5, 8, device="cuda")
    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    position_embeddings = (
        torch.ones(5, 4, device="cuda"),
        torch.zeros(5, 4, device="cuda"),
    )

    output = attention(hidden_states, cu_seqlens, position_embeddings)

    assert output.shape == hidden_states.shape

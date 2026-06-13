# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import torch

from nemo_automodel.components.models.llama_nemotron_vl.model import (
    _filter_vision_embeddings_by_image_flags,
    _replace_image_token_embeddings,
)

IMG_TOKEN_ID = 128258


def _old_boolean_index_replace(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    vit_embeds: torch.Tensor,
) -> torch.Tensor:
    batch_size, seq_len, hidden_size = input_embeds.shape
    flat_embeds = input_embeds.reshape(batch_size * seq_len, hidden_size)
    flat_input_ids = input_ids.reshape(batch_size * seq_len)
    selected = flat_input_ids == IMG_TOKEN_ID
    flat_embeds = flat_embeds.clone()
    flat_embeds[selected] = flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, hidden_size)
    return flat_embeds.reshape(batch_size, seq_len, hidden_size)


def test_replace_image_token_embeddings_matches_boolean_index_outputs_and_grads():
    input_ids = torch.tensor(
        [
            [1, IMG_TOKEN_ID, IMG_TOKEN_ID, 2, 3],
            [4, 5, IMG_TOKEN_ID, IMG_TOKEN_ID, 6],
        ],
        dtype=torch.long,
    )
    input_embeds = torch.randn(2, 5, 4, requires_grad=True)
    vit_embeds = torch.randn(4, 4, requires_grad=True)

    old_input = input_embeds.detach().clone().requires_grad_(True)
    old_vit = vit_embeds.detach().clone().requires_grad_(True)
    old_out = _old_boolean_index_replace(old_input, input_ids, old_vit)
    old_out.square().sum().backward()

    new_input = input_embeds.detach().clone().requires_grad_(True)
    new_vit = vit_embeds.detach().clone().requires_grad_(True)
    new_out = _replace_image_token_embeddings(new_input, input_ids, new_vit, IMG_TOKEN_ID)
    new_out.square().sum().backward()

    torch.testing.assert_close(new_out, old_out)
    torch.testing.assert_close(new_input.grad, old_input.grad)
    torch.testing.assert_close(new_vit.grad, old_vit.grad)


def test_filter_vision_embeddings_skips_filter_when_image_flags_absent():
    vit_embeds = torch.randn(3, 2, 4)

    filtered = _filter_vision_embeddings_by_image_flags(vit_embeds, None)

    assert filtered is vit_embeds


def test_filter_vision_embeddings_keeps_flagged_rows():
    vit_embeds = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(4, 2, 3)
    image_flags = torch.tensor([[1], [0], [1], [0]], dtype=torch.long)

    filtered = _filter_vision_embeddings_by_image_flags(vit_embeds, image_flags)

    torch.testing.assert_close(filtered, vit_embeds[[0, 2]])

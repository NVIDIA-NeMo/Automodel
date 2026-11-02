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

"""Packed Gemma4 image attention must respect the sliding window."""

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    create_masks_for_vision_model,
    get_block_sequence_ids_for_mask,
)

from nemo_automodel.components.models.gemma4_moe.model import _build_packed_gemma4_causal_mask_mapping

pytestmark = pytest.mark.cpu_only


def _expected_masks(
    documents: list[int],
    image_spans: list[tuple[int, int]],
    window: int | None,
) -> dict[str, torch.Tensor]:
    """Enumerate eligible keys independently of production mask composition.

    Args:
        documents: Document id per token; zero denotes padding.
        image_spans: Half-open image intervals in the sequence.
        window: Maximum causal lookback including the query, or no limit.

    Returns:
        Full and sliding boolean masks of shape [1, 1, sequence, sequence].
    """
    size = len(documents)
    full = torch.zeros(size, size, dtype=torch.bool)
    sliding = torch.zeros_like(full)
    for query, document in enumerate(documents):
        if document == 0:
            continue
        causal_keys = {key for key in range(query + 1) if documents[key] == document}
        full[query, sorted(causal_keys)] = True
        image_keys = set()
        for start, end in image_spans:
            if start <= query < end:
                image_keys.update(key for key in range(start, end) if documents[key] == document)
        keys = sorted(causal_keys | image_keys)
        if window is not None:
            keys = [key for key in keys if key > query - window]
        sliding[query, keys] = True
    return {"full_attention": full[None, None], "sliding_attention": sliding[None, None]}


@pytest.mark.parametrize("mask_format", ["bool", "additive", "flex"])
@pytest.mark.parametrize(
    "documents,image_spans,window",
    [
        pytest.param([1] * 12, [(2, 10)], 4, id="image-longer-than-window"),
        pytest.param([1] * 1124, [(2, 1122)], 1024, id="1120-image-tokens"),
        pytest.param([1] * 6 + [2] * 6 + [0] * 2, [(1, 11)], 4, id="image-at-document-boundary"),
        pytest.param([1] * 12, [(2, 4), (6, 8)], 4, id="separate-short-images"),
        pytest.param([1] * 6 + [2] * 4 + [0] * 2, [], 4, id="text-and-padding"),
        pytest.param([1] * 12, [(2, 10)], None, id="unbounded-window"),
    ],
)
def test_packed_image_masks(
    documents: list[int],
    image_spans: list[tuple[int, int]],
    window: int | None,
    mask_format: str,
) -> None:
    """Check both layer masks against enumerated document and image key sets."""
    packed = torch.tensor([documents])
    mm_types = torch.zeros_like(packed)
    for start, end in image_spans:
        mm_types[:, start:end] = 1
    actual = _build_packed_gemma4_causal_mask_mapping(
        packed,
        mm_types,
        dtype=torch.float32,
        sliding_window=window,
        as_additive=mask_format == "additive",
        as_block_mask=mask_format == "flex",
    )
    expected = _expected_masks(documents, image_spans, window)
    for layer_type, mask in actual.items():
        if mask_format == "flex":
            # Evaluate the real BlockMask callback, including its padding-row guard.
            mask = create_mask(mask.mask_mod, 1, 1, len(documents), len(documents), device="cpu")
            expected[layer_type][0, 0, packed[0] == 0, 0] = True
        elif mask_format == "additive":
            assert torch.all((mask == 0) | (mask == torch.finfo(torch.float32).min))
            mask = mask == 0
        torch.testing.assert_close(mask, expected[layer_type], rtol=0, atol=0)


@pytest.mark.parametrize("seq_len,image_end,window", [(12, 10, 4), (1124, 1122, 1024)])
def test_packed_single_document_matches_hf_vision_mask(seq_len: int, image_end: int, window: int) -> None:
    """Check the packed masks and the actual HF helper against the same oracle."""
    config = Gemma4TextConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=window,
        use_bidirectional_attention="vision",
    )
    config._attn_implementation = "eager"
    packed = torch.ones(1, seq_len, dtype=torch.long)
    mm_types = torch.zeros_like(packed)
    mm_types[:, 2:image_end] = 1
    reference = create_masks_for_vision_model(
        config=config,
        inputs_embeds=torch.zeros(1, seq_len, 8),
        attention_mask=None,
        past_key_values=None,
        position_ids=torch.arange(seq_len)[None],
        block_sequence_ids=get_block_sequence_ids_for_mask(mm_types, device=torch.device("cpu")),
    )
    actual = _build_packed_gemma4_causal_mask_mapping(packed, mm_types, dtype=torch.float32, sliding_window=window)
    expected = _expected_masks([1] * seq_len, [(2, image_end)], window)
    for layer_type in actual:
        torch.testing.assert_close(reference[layer_type] == 0, expected[layer_type], rtol=0, atol=0)
        torch.testing.assert_close(actual[layer_type], expected[layer_type], rtol=0, atol=0)


@pytest.mark.parametrize("as_additive", [False, True])
def test_packed_long_image_attention_forward_and_backward(as_additive: bool) -> None:
    """Verify dense-mask SDPA outputs and gradients for a long image and two documents."""
    documents = [1] * 12 + [2] * 4
    packed = torch.tensor([documents])
    mm_types = torch.zeros_like(packed)
    mm_types[:, 2:10] = 1
    mask = _build_packed_gemma4_causal_mask_mapping(
        packed, mm_types, dtype=torch.float32, sliding_window=4, as_additive=as_additive
    )["sliding_attention"]
    expected = _expected_masks(documents, [(2, 10)], 4)["sliding_attention"]
    generator = torch.Generator().manual_seed(42)
    query, key, value = (torch.randn(1, 2, 16, 8, generator=generator, requires_grad=True) for _ in range(3))
    output_grad = torch.randn(1, 2, 16, 8, generator=generator)
    actual_output = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    expected_output = F.scaled_dot_product_attention(query, key, value, attn_mask=expected)
    actual_grads = torch.autograd.grad(actual_output, (query, key, value), output_grad)
    expected_grads = torch.autograd.grad(expected_output, (query, key, value), output_grad)
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

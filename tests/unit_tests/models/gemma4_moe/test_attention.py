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

"""Native document masks preserve Gemma4 sliding and image attention."""

import pytest
import torch
from transformers import Gemma4TextConfig

from nemo_automodel.components.models.gemma4_moe.attention import prepare_gemma4_document_mask


def _text_config():
    return Gemma4TextConfig(
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        use_bidirectional_attention="vision",
    )


def _causal(documents):
    length = documents.shape[-1]
    return (
        (documents[:, None, :, None] == documents[:, None, None, :])
        & documents[:, None, :, None].ne(0)
        & torch.ones(length, length, dtype=torch.bool).tril()
    )


def test_image_future_edges_preserve_document_padding_and_window_policy():
    config = _text_config()
    config._attn_implementation = "sdpa"
    config.sliding_window = 2
    documents = torch.tensor([[1, 1, 0, 1, 2, 2, 0, 0]])
    modalities = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
    allowed = _causal(documents)
    original = allowed.clone()
    masks = prepare_gemma4_document_mask(config, allowed, modalities, documents, dtype=torch.float32)
    sliding = masks["sliding_attention"]
    assert sliding.dtype == masks["full_attention"].dtype == torch.bool
    assert sliding[0, 0, 1, 3]  # Future tokens of the same image.
    assert not sliding[0, 0, 3, 1]  # Native sliding window still applies.
    assert not sliding[0, 0, 3, 4]  # Adjacent image tokens in different documents.
    assert not sliding[0, 0, [2, 6, 7]].any()
    assert not sliding[0, 0, :, [2, 6, 7]].any()
    torch.testing.assert_close(masks["full_attention"], original)
    torch.testing.assert_close(allowed, original)
    assert torch.isfinite(masks["sliding_attention"]).all()


def test_cross_document_causal_policy_keeps_original_image_groups():
    config = _text_config()
    config._attn_implementation = "eager"
    documents = torch.tensor([[1, 1, 2, 2, 0]])
    allowed = _causal(documents.ne(0).long())
    masks = prepare_gemma4_document_mask(
        config, allowed, torch.tensor([[1, 1, 1, 1, 0]]), documents, dtype=torch.float32
    )
    sliding = masks["sliding_attention"] == 0
    assert sliding[0, 0, 0, 1]  # Same image may attend to the future.
    assert not sliding[0, 0, 1, 2]  # Different images remain causally separated.
    assert sliding[0, 0, 2, 1]  # Caller allows causal cross-document attention.
    torch.testing.assert_close(masks["full_attention"] == 0, allowed)


def test_flash_policy_is_not_silently_approximated():
    config = _text_config()
    config._attn_implementation = "flash_attention_2"
    documents = torch.ones((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="SDPA or eager"):
        prepare_gemma4_document_mask(config, _causal(documents), None, documents, dtype=torch.float32)


def test_native_cp_leaves_mask_construction_to_ring_consumer():
    from types import SimpleNamespace

    from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

    # CP shards document IDs before forward while modality IDs remain full-length.
    # The ordinary document-mask compositor must not combine these layouts.
    consumer = SimpleNamespace(_cp_enabled=True)
    local_ids = torch.tensor([[1, 1, 2, 2]])
    full_modalities = torch.ones((1, 8), dtype=torch.long)
    assert (
        Gemma4ForConditionalGeneration.prepare_document_attention_mask(consumer, local_ids, full_modalities, local_ids)
        is None
    )

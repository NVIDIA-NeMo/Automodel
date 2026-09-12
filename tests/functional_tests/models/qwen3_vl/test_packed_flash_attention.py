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

"""Qwen3-VL packed FlashAttention coverage with real vision-tower inputs."""

import importlib.util

import pytest
import torch
from transformers import Qwen3VLConfig

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.qwen3_vl.model import Qwen3VLForConditionalGeneration

_HAS_FA = torch.cuda.is_available() and importlib.util.find_spec("flash_attn") is not None


def _build_tiny_qwen3_vl() -> Qwen3VLForConditionalGeneration:
    """Build a tiny Qwen3-VL whose text and vision towers use FlashAttention."""
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "use_cache": False,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "out_hidden_size": 16,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 64,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=30,
        video_token_id=31,
        attn_implementation="flash_attention_2",
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).to(device="cuda", dtype=torch.bfloat16).eval()


@pytest.mark.skipif(not _HAS_FA, reason="requires CUDA and flash-attn")
def test_packed_flash_attention_with_image_matches_isolated_documents() -> None:
    """Packed text metadata must reach the language tower without leaking into vision attention."""
    model = _build_tiny_qwen3_vl()
    config = model.config
    doc_with_image = [5, config.image_token_id, 9, 13]
    text_doc = [7, 3, 11]
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    patch_dim = (
        config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
    )
    torch.manual_seed(1)
    pixel_values = torch.randn(4, patch_dim)
    packed_sample = {
        "input_ids": torch.tensor(doc_with_image + text_doc),
        "labels": torch.tensor(doc_with_image + text_doc),
        "attention_mask": torch.tensor([1] * len(doc_with_image) + [2] * len(text_doc)),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2]] * 3),
        "mm_token_type_ids": torch.tensor([0, 1, 0, 0, 0, 0, 0]),
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
        "n_images": 1,
    }
    batch = neat_packed_vlm_collater([packed_sample], attn_implementation="flash_attention_2")

    with torch.no_grad():
        packed_logits = model(
            input_ids=batch["input_ids"].cuda(),
            position_ids=batch["position_ids"].cuda(),
            pixel_values=batch["pixel_values"].cuda(),
            image_grid_thw=batch["image_grid_thw"].cuda(),
            mm_token_type_ids=batch["mm_token_type_ids"].cuda(),
            cu_seq_lens_q=batch["cu_seq_lens_q"].cuda(),
            cu_seq_lens_k=batch["cu_seq_lens_k"].cuda(),
            max_length_q=batch["max_length_q"],
            max_length_k=batch["max_length_k"],
        ).logits[0]
        image_logits = model(
            input_ids=torch.tensor([doc_with_image], device="cuda"),
            position_ids=torch.tensor([[0, 1, 2, 3]] * 3, device="cuda").unsqueeze(1),
            pixel_values=batch["pixel_values"].cuda(),
            image_grid_thw=batch["image_grid_thw"].cuda(),
            mm_token_type_ids=torch.tensor([[0, 1, 0, 0]], device="cuda"),
        ).logits[0]
        text_logits = model(
            input_ids=torch.tensor([text_doc], device="cuda"),
            position_ids=torch.tensor([[0, 1, 2]] * 3, device="cuda").unsqueeze(1),
        ).logits[0]

    torch.testing.assert_close(packed_logits[: len(doc_with_image)].float(), image_logits.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(packed_logits[len(doc_with_image) :].float(), text_logits.float(), rtol=2e-2, atol=2e-2)

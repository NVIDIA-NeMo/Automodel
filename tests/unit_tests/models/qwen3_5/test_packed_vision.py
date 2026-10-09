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

"""Packed text boundaries must not override HF image/video frame boundaries."""

import copy

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.vision_utils import get_vision_attention_seqlens

from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing_for_models
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration


def _model() -> Qwen3_5ForConditionalGeneration:
    torch.manual_seed(42)
    text = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        layer_types=["full_attention"],
        pad_token_id=0,
        torch_dtype="float32",
        attn_implementation="sdpa",
    )
    vision = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        attn_implementation="sdpa",
    )
    config = Qwen3_5Config(
        text_config=text.to_dict(),
        vision_config=vision.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    return Qwen3_5ForConditionalGeneration(config, backend=backend).float().eval()


@pytest.mark.parametrize("media", ["image", "video", "both", "text"])
@pytest.mark.parametrize("precomputed", [False, True])
def test_packed_media_matches_independent_documents(media: str, precomputed: bool) -> None:
    model = _model()
    reference = copy.deepcopy(model)
    contract = configure_packing_for_models([model])
    documents = []
    for document_index in range(2):
        tokens = [3 + document_index]
        document = {}
        for kind, token, frames, pixels_key in (
            ("image", 60, 1, "pixel_values"),
            ("video", 61, 2, "pixel_values_videos"),
        ):
            if media in (kind, "both"):
                tokens.extend([62] + [token] * (frames * 4) + [63])
                # The collator stores media in BF16; use identical inputs in the reference.
                document[pixels_key] = torch.randn(frames * 4, 12, dtype=torch.bfloat16)
                document[f"{kind}_grid_thw"] = torch.tensor([[frames, 2, 2]])
        tokens.extend([5] * (document_index + 1))
        document.update(
            input_ids=torch.tensor(tokens),
            labels=torch.tensor(tokens),
            attention_mask=torch.full((len(tokens),), document_index + 1),
            position_ids=torch.arange(len(tokens)).expand(3, -1),
        )
        documents.append(document)

    packed = {}
    for key in documents[0]:
        packed[key] = torch.cat([document[key] for document in documents], dim=-1 if key == "position_ids" else 0)
    length = packed["input_ids"].numel()
    # Exercise real collator metadata: 2D text offsets, uneven documents, and padding.
    batch = neat_packed_vlm_collater([packed], packing=contract, max_length=length + 2)
    original_boundaries = batch["cu_seqlens"].clone()
    assert original_boundaries.ndim == 2
    if precomputed:
        for kind in ("image", "video"):
            if media in (kind, "both"):
                cu_seqlens, _ = get_vision_attention_seqlens(batch[f"{kind}_grid_thw"], model.config.vision_config)
                batch[f"{kind}_cu_seqlens"] = cu_seqlens
                batch[f"{kind}_max_seqlen"] = 4

    actual = model(**batch).logits[:, :length]
    expected = torch.cat(
        [
            reference(
                **{
                    key: value.unsqueeze(1 if key == "position_ids" else 0)
                    if key in ("input_ids", "position_ids")
                    else value
                    for key, value in document.items()
                    if key not in ("labels", "attention_mask")
                }
            ).logits
            for document in documents
        ],
        dim=1,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(batch["cu_seqlens"], original_boundaries)

    upstream = torch.randn_like(actual) / actual.numel()
    actual.backward(upstream)
    expected.backward(upstream)
    for name, param in model.named_parameters():
        reference_grad = reference.get_parameter(name).grad
        if reference_grad is None:
            assert param.grad is None, name
        else:
            assert param.grad is not None, name
            torch.testing.assert_close(
                param.grad, reference_grad, rtol=1e-4, atol=2e-6, msg=lambda message: f"{name}: {message}"
            )

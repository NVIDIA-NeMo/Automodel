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

"""Model-owned worker positions agree with the native dense Qwen3-VL model."""

import pickle

import pytest
import torch
from PIL import Image

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.datasets.vlm.datasets import encode_vlm_document
from nemo_automodel.components.models.qwen3_vl.model import Qwen3VLForConditionalGeneration
from nemo_automodel.components.models.qwen3_vl.preprocessing import Qwen3VLPositionState
from tests.utils.qwen3_vlm_fixtures import processor as processor
from tests.utils.qwen3_vlm_fixtures import qwen2_processor as qwen2_processor
from tests.utils.qwen3_vlm_fixtures import tiny_qwen3_vl_config


@pytest.mark.parametrize("sizes", [(), ((8, 12), (16, 8))])
def test_serializable_position_state_matches_native_model(processor, sizes):
    document = {
        "conversation": [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": Image.new("RGB", size)} for size in sizes),
                    {"type": "text", "text": "Describe image"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
        ]
    }
    config = tiny_qwen3_vl_config()
    config._attn_implementation = "sdpa"
    model = Qwen3VLForConditionalGeneration(config)
    state = pickle.loads(pickle.dumps(resolve_image_position_state(model)))  # noqa: S301 - local config
    assert isinstance(state, Qwen3VLPositionState) and not isinstance(state, torch.nn.Module)
    encoded = encode_vlm_document(document, processor, inject_fake_images=False)
    kwargs = {
        "mm_token_type_ids": encoded["mm_token_type_ids"].unsqueeze(0),
        "image_grid_thw": encoded.get("image_grid_thw"),
        "attention_mask": encoded["attention_mask"],
    }
    actual = state.positions(encoded["input_ids"], **kwargs)
    expected, _ = model.model.get_rope_index(
        encoded["input_ids"].unsqueeze(0), **{**kwargs, "attention_mask": kwargs["attention_mask"].unsqueeze(0)}
    )
    torch.testing.assert_close(actual, expected[:, 0], atol=0, rtol=0)
    assert actual.shape == (3, encoded["input_ids"].numel())
    if sizes:
        assert not actual[1].equal(actual[2])
        assert encoded["input_ids"].eq(9).sum() == sum(width * height // 16 for width, height in sizes)

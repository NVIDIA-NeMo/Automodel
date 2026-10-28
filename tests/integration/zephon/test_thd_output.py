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

"""Real Zephon streams hand off the native THD schema and replay unchanged."""

from contextlib import closing
from dataclasses import replace

import torch

from nemo_automodel.components.distributed.thd_utils import process_input_for_thd
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


def test_real_thd_pipeline_preserves_packing_shifting_xdoc_and_checkpoint(config, tokenizer):
    for xdoc in (False, True):
        selected = replace(
            config,
            zephon=replace(
                config.zephon, training=replace(config.zephon.training, allow_cross_document_attention=xdoc)
            ),
        )
        with (
            closing(selected.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as batched,
            closing(
                replace(selected, output_format="thd").build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
            ) as thd,
        ):
            expected, actual = next(batched), next(thd)
            assert actual["qkv_format"] == "thd"
            assert "document_ids" not in actual
            for key in ("input_ids", "labels", "position_ids", "padding_mask", "_packed_seq_ids"):
                torch.testing.assert_close(actual[key], expected[key])
            prepared = process_input_for_thd(actual)
            assert prepared["input_ids"].shape == (14,)
            assert prepared["cu_seqlens"].tolist() == ([0, 7, 14] if xdoc else [0, 4, 7, 11, 14])
            assert not prepared["padding_mask"].any()  # PAD=EOS is content here.
            checkpoint = thd.state_dict()
            following = next(thd)
        with closing(
            replace(selected, output_format="thd").build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
        ) as restored:
            restored.load_state_dict(checkpoint)
            replay = next(restored)
            for key, value in following.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(replay[key], value)
                else:
                    assert replay[key] == value

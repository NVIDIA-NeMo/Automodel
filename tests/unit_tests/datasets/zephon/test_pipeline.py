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

"""Operator controls reach Zephon's public construction API without execution."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from transformers import PreTrainedTokenizerFast
from zephon import Pipeline

from nemo_automodel.components.datasets.zephon.config import (
    ZephonDataloaderConfig,
    ZephonPretokenizedTextSourceConfig,
    ZephonTextSourceConfig,
)
from nemo_automodel.components.datasets.zephon.pipeline import create_text_pipeline
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


@pytest.mark.parametrize("source_kind", ["text", "pretokenized"])
def test_text_operator_parallelism_reaches_public_pipeline(config, tokenizer, source_kind):
    if source_kind == "pretokenized":
        config = replace(config, dataset_config=ZephonPretokenizedTextSourceConfig())
    config = replace(
        config,
        zephon=replace(
            config.zephon,
            workers=replace(
                config.zephon.workers,
                fetch_parallelism=2,
                prefetch_parallelism=3,
                map_parallelism=4,
                tokenize_parallelism=5,
                pack_parallelism=6,
                shuffle_parallelism=7,
            ),
            runtime=replace(config.zephon.runtime, prefetch_buffer_size=8),
            packing=replace(config.zephon.packing, shuffle_after_pack=True, shuffle_buffer_size=8),
        ),
    )
    with (
        patch.object(Pipeline, "fetch_parallelism", autospec=True, side_effect=Pipeline.fetch_parallelism) as fetch,
        patch.object(Pipeline, "prefetch", autospec=True, side_effect=Pipeline.prefetch) as prefetch,
        patch.object(Pipeline, "map_transform", autospec=True, side_effect=Pipeline.map_transform) as normalize,
        patch.object(Pipeline, "tokenize", autospec=True, side_effect=Pipeline.tokenize) as text,
        patch.object(Pipeline, "tokenize_chat", autospec=True, side_effect=Pipeline.tokenize_chat) as chat,
        patch.object(Pipeline, "pack_flat", autospec=True, side_effect=Pipeline.pack_flat) as pack,
        patch.object(Pipeline, "shuffle", autospec=True, side_effect=Pipeline.shuffle) as shuffle,
        patch.object(Pipeline, "map_batch", autospec=True, side_effect=Pipeline.map_batch) as batch_map,
    ):
        create_text_pipeline(config, tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
        batch_map.assert_not_called()
        assert fetch.call_args.args[1] == 2
        assert prefetch.call_args.kwargs["parallelism"] == 3
        assert pack.call_args.kwargs["parallelism"] == 6
        assert shuffle.call_args.kwargs["parallelism"] == 7
        if source_kind == "text":
            normalize.assert_not_called()
            chat.assert_not_called()
            assert text.call_args.kwargs["parallelism"] == 5
        else:
            assert normalize.call_args.kwargs["parallelism"] == 4
            text.assert_not_called()
            chat.assert_not_called()


def test_numeric_overrides_reject_unknown_vocabulary_ids_before_source_start(
    config: ZephonDataloaderConfig, tokenizer: PreTrainedTokenizerFast
) -> None:
    source_config = ZephonTextSourceConfig(tokenize_special_tokens="eos", eos_token_id=9999)
    config = replace(config, dataset_config=source_config)
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_source_pipeline") as create_source:
        with pytest.raises(ValueError, match="eos_token_id=9999 is not an existing tokenizer vocabulary ID"):
            create_text_pipeline(config, tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
        create_source.assert_not_called()

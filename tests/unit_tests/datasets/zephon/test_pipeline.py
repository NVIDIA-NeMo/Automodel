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
from zephon.work import StaticMixtureWorkSource, TokenEstimation

from nemo_automodel.components.datasets.loader import DataloaderConfig
from nemo_automodel.components.datasets.zephon.config import (
    ZephonInstructionDatasetConfig,
    ZephonPretokenizedTextDatasetConfig,
    ZephonTextDatasetConfig,
)
from nemo_automodel.components.datasets.zephon.pipeline import create_source_pipeline, create_text_pipeline
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


@pytest.mark.parametrize("source_kind", ["text", "instruction", "pretokenized"])
def test_text_operator_parallelism_reaches_public_pipeline(config, tokenizer, source_kind):
    config = config.dataset_config
    if source_kind == "instruction":
        config = ZephonInstructionDatasetConfig(zephon=config.zephon, column_mapping={"question": "q", "answer": "a"})
    elif source_kind == "pretokenized":
        config = ZephonPretokenizedTextDatasetConfig(zephon=config.zephon)
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
            source=replace(config.zephon.source, token_estimation={"calibration_samples": 4}),
            packing=replace(config.zephon.packing, shuffle_after_pack=True, shuffle_buffer_size=8),
        ),
    )
    with (
        patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source,
        patch("zephon.work.TokenEstimation", wraps=TokenEstimation) as estimation,
        patch.object(
            Pipeline, "preflight_tokenizers", autospec=True, side_effect=Pipeline.preflight_tokenizers
        ) as preflight,
        patch.object(Pipeline, "fetch_parallelism", autospec=True, side_effect=Pipeline.fetch_parallelism) as fetch,
        patch.object(Pipeline, "prefetch", autospec=True, side_effect=Pipeline.prefetch) as prefetch,
        patch.object(Pipeline, "map_transform", autospec=True, side_effect=Pipeline.map_transform) as normalize,
        patch.object(Pipeline, "tokenize", autospec=True, side_effect=Pipeline.tokenize) as text,
        patch.object(Pipeline, "tokenize_chat", autospec=True, side_effect=Pipeline.tokenize_chat) as chat,
        patch.object(Pipeline, "pack_flat", autospec=True, side_effect=Pipeline.pack_flat) as pack,
        patch.object(Pipeline, "shuffle", autospec=True, side_effect=Pipeline.shuffle) as shuffle,
        patch.object(Pipeline, "map_batch", autospec=True, side_effect=Pipeline.map_batch) as batch_map,
    ):
        create_text_pipeline(config, tokenizer=tokenizer, seed=91, batch_size=2, dp_rank=0, dp_world_size=1)
        batch_map.assert_not_called()
        assert source.call_args.kwargs["seed"] == 91
        assert shuffle.call_args.kwargs["seed"] == 91
        assert estimation.call_args.kwargs["calibration_samples"] == 4
        assert (estimation.call_args.kwargs["measure"] is not None) == (source_kind == "pretokenized")
        assert preflight.call_args.kwargs["strict"] is True
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
            if source_kind == "instruction":
                assert chat.call_args.kwargs["parallelism"] == 5
                assert chat.call_args.kwargs["special_tokens"] == "tokenizer_default"
                assert "add_special_tokens" not in chat.call_args.kwargs
                assert "append_eos" not in chat.call_args.kwargs
            else:
                chat.assert_not_called()


@pytest.mark.parametrize("instruction", [False, True])
def test_numeric_overrides_reject_unknown_vocabulary_ids_before_source_start(
    config: DataloaderConfig, tokenizer: PreTrainedTokenizerFast, instruction: bool
) -> None:
    source_config = (
        ZephonInstructionDatasetConfig(
            zephon=config.dataset_config.zephon,
            column_mapping={"question": "q", "answer": "a"},
            special_tokens="eos",
            eos_token_id=9999,
        )
        if instruction
        else ZephonTextDatasetConfig(
            zephon=config.dataset_config.zephon, tokenize_special_tokens="eos", eos_token_id=9999
        )
    )
    config = source_config
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_source_pipeline") as create_source:
        with pytest.raises(ValueError, match="eos_token_id=9999 is not an existing tokenizer vocabulary ID"):
            create_text_pipeline(config, tokenizer=tokenizer, seed=42, batch_size=2, dp_rank=0, dp_world_size=1)
        create_source.assert_not_called()


@pytest.mark.parametrize("answer_only", [False, True])
def test_chat_header_supervision_only_applies_to_assistant_loss(
    config: DataloaderConfig, tokenizer: PreTrainedTokenizerFast, answer_only: bool
) -> None:
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    config = ZephonInstructionDatasetConfig(
        zephon=config.dataset_config.zephon,
        column_mapping={"question": "q", "answer": "a"},
        use_hf_chat_template=True,
        answer_only_loss_mask=answer_only,
    )
    with patch.object(Pipeline, "tokenize_chat", autospec=True, side_effect=Pipeline.tokenize_chat) as chat:
        create_text_pipeline(config, tokenizer=tokenizer, seed=42, batch_size=2, dp_rank=0, dp_world_size=1)
    assert chat.call_args.kwargs["supervise_assistant_header"] is answer_only
    assert chat.call_args.kwargs["loss_scope"] == ("all_assistant" if answer_only else "all_tokens")


@pytest.mark.parametrize(
    "setting,source_count,enabled",
    [("auto", 1, False), ("auto", 2, True), (False, 2, False), (None, 2, False), (True, 1, True), ({}, 1, True)],
)
def test_token_estimation_policy(config, setting, source_count, enabled):
    options = config.dataset_config.zephon
    options = replace(
        options,
        sources=tuple(replace(options.sources[0], name=f"source_{i}") for i in range(source_count)),
        source=replace(options.source, token_estimation=setting),
    )
    with (
        patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source,
        patch("zephon.work.TokenEstimation", wraps=TokenEstimation) as estimation,
    ):
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    assert estimation.called is enabled
    assert (source.call_args.kwargs["token_estimation"] is not None) is enabled
    # Resolving an automatic choice must not mutate the reusable config.
    assert options.source.token_estimation == setting


@pytest.mark.parametrize("policy,passes", [(None, 1), ("repeat", None)])
def test_unset_pass_limit_uses_zephon_exhaustion_policy(config, policy, passes):
    options = config.dataset_config.zephon
    options = replace(options, source=replace(options.source, exhausted_policy=policy))
    with patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source:
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    kwargs = source.call_args.kwargs
    assert "stop_after_passes" not in kwargs
    actual = StaticMixtureWorkSource(**kwargs)
    assert actual.state_dict()["stop_after_passes"] == passes


def test_explicit_pass_limit_reaches_zephon(config):
    options = config.dataset_config.zephon
    options = replace(options, source=replace(options.source, stop_after_passes=2))
    with patch("zephon.work.StaticMixtureWorkSource", wraps=StaticMixtureWorkSource) as source:
        create_source_pipeline(options, seed=42, dp_rank=0, dp_world_size=1)
    assert source.call_args.kwargs["stop_after_passes"] == 2


@pytest.mark.parametrize("apply_template,expected", [(False, "tokenizer_default"), (True, "none")])
def test_instruction_default_special_tokens_follow_template_ownership(config, tokenizer, apply_template, expected):
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    dataset = ZephonInstructionDatasetConfig(
        zephon=config.dataset_config.zephon,
        column_mapping={"question": "q", "answer": "a"},
        use_hf_chat_template=apply_template,
    )
    with patch.object(Pipeline, "tokenize_chat", autospec=True, side_effect=Pipeline.tokenize_chat) as chat:
        create_text_pipeline(dataset, tokenizer=tokenizer, seed=42, batch_size=2, dp_rank=0, dp_world_size=1)
    assert chat.call_args.kwargs["special_tokens"] == expected
    assert dataset.special_tokens is None


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {
            "prefetch_batches": 24,
            "default_stage_prefetch": 8,
            "op_queue_capacity": 12,
            "mtp_buffer": 6,
            "mtp_prefetch": 2,
        },
        {"prefetch_batches": 0, "default_stage_prefetch": 0, "mtp_prefetch": 0},
    ],
)
def test_runtime_buffers_reach_zephon_without_replacing_none_or_zero(
    config: DataloaderConfig, tokenizer: PreTrainedTokenizerFast, overrides: dict[str, int]
) -> None:
    dataset = config.dataset_config
    dataset = replace(dataset, zephon=replace(dataset.zephon, runtime=replace(dataset.zephon.runtime, **overrides)))
    with patch.object(Pipeline, "options", autospec=True, side_effect=Pipeline.options) as options:
        create_text_pipeline(dataset, tokenizer=tokenizer, seed=42, batch_size=2, dp_rank=0, dp_world_size=1)
    forwarded = next(call.kwargs for call in options.call_args_list if "mtp_mode" in call.kwargs)
    for name in ("prefetch_batches", "default_stage_prefetch", "op_queue_capacity", "mtp_buffer", "mtp_prefetch"):
        assert forwarded[name] == overrides.get(name)

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

from dataclasses import dataclass

import pytest
import torch
from transformers import ProcessorMixin

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig
from nemo_automodel.components.datasets.vlm.collate_fns import (
    neat_packed_vlm_collater,
    packed_sequence_thd_vlm_collater,
)
from nemo_automodel.components.datasets.vlm.datasets import CordV2DatasetConfig, PreTokenizedDatasetWrapperConfig
from nemo_automodel.components.datasets.vlm.loader import (
    _COMPACT_MASK_BACKENDS,
    VlmCollatorConfig,
    VlmDataloaderConfig,
    VlmProcessorConfig,
    VlmVideoProcessorConfig,
)
from nemo_automodel.components.datasets.vlm.mock import MockVlmDatasetConfig
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig
from nemo_automodel.recipes._typed_config import RecipeConfig


class DummyProcessor:
    def __init__(self):
        self.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()


@dataclass
class StaticDatasetConfig:
    events: list[str]

    def build(self):
        self.events.append("dataset")
        return ["one", "two"]


class BuildContext:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter")

    def __exit__(self, *_):
        self.events.append("exit")


@pytest.fixture
def build_packed_dataloader(monkeypatch):
    """Build a VLM dataloader with dataset construction stubbed out, leaving collater selection real.

    Dataset, pretokenization and packing are stubbed because only the collater ``build`` selects is
    under test. Returns a callable taking ``packing=`` (defaults to a NEAT ``NeatPackConfig``) plus
    any ``VlmDataloaderConfig.build`` keyword; the kwargs packing was built with are recorded on
    ``.packing_kwargs``.
    """
    packing_kwargs = {}

    def _stub_packing_build(self, dataset, **kwargs):
        packing_kwargs.update(kwargs)
        return dataset

    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", _stub_packing_build)

    def _build(*, packing=None, **build_kwargs):
        config = VlmDataloaderConfig(
            dataset_config=StaticDatasetConfig([]),
            processor_config=VlmProcessorConfig(factory=DummyProcessor),
            pretokenization=PreTokenizedDatasetWrapperConfig(),
            packing=NeatPackConfig() if packing is None else packing,
            shuffle=False,
        )
        return config.build(
            pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2, **build_kwargs
        )

    _build.packing_kwargs = packing_kwargs
    return _build


def test_recipe_config_separates_vlm_dataset_wrapper_and_packing_fields():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset",
                    "num_samples": 4,
                    "max_length": 128,
                    "pretokenize": True,
                    "inject_fake_images": False,
                },
                "packed_sequence": {
                    "pretokenize": True,
                    "max_length": 128,
                    "pack_size": 128,
                    "packing_ratio": 0.9,
                    "collate_max_length": 128,
                    "packing_format": "thd",
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader

    assert config is not None
    assert isinstance(config.dataset_config, MockVlmDatasetConfig)
    assert config.dataset_config.max_length == 128
    assert config.pretokenization.max_length == 128
    assert config.pretokenization.inject_fake_images is False
    assert config.packing.pack_size == 128
    assert config.packing.packing_ratio == 0.9
    assert config.packing.collate_max_length == 128
    assert config.packing.packing_format == "thd"


def test_recipe_config_rejects_unknown_vlm_packing_format():
    raw = ConfigNode(
        {
            "dataset": {"_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset"},
            "packed_sequence": {"pack_size": 128, "packing_format": "unknown"},
        }
    )

    with pytest.raises(ValueError, match="Unsupported VLM packing_format"):
        _ = RecipeConfig(raw).vlm_dataloader


def test_recipe_config_accepts_cord_v2_sample_limit():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset",
                    "limit_dataset_samples": 100,
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader

    assert isinstance(config.dataset_config, CordV2DatasetConfig)
    assert config.dataset_config.limit_dataset_samples == 100


def test_vlm_dataloader_builds_processor_and_dataset_inside_context_then_iterates():
    events = []
    processor = DummyProcessor()

    def build_processor():
        events.append("processor")
        return processor

    def collate(examples, *, processor, prefix):
        return [prefix + example for example in examples], processor

    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig(events),
        processor_config=VlmProcessorConfig(factory=build_processor),
        collator=VlmCollatorConfig(factory=collate, kwargs={"prefix": "item:"}),
        shuffle=False,
        num_workers=0,
    )

    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        dataset_build_context=BuildContext(events),
    )
    batch, batch_processor = next(iter(result.dataloader))

    assert events == ["enter", "processor", "dataset", "exit"]
    assert result.processor is processor
    assert batch == ["item:one", "item:two"]
    assert batch_processor is processor


def test_vlm_processor_builds_independently_configured_video_processor():
    video_processor = object()
    calls = []

    def build_video_processor(*, pretrained_model_name_or_path, size, fps, max_frames):
        calls.append(("video", pretrained_model_name_or_path, size, fps, max_frames))
        return video_processor

    def build_processor(*, model_id, video_processor):
        calls.append(("processor", model_id, video_processor))
        return DummyProcessor()

    config = VlmProcessorConfig(
        factory=build_processor,
        kwargs={"model_id": "outer-model"},
        video_processor=VlmVideoProcessorConfig(
            factory=build_video_processor,
            kwargs={
                "size": {"shortest_edge": 1024, "longest_edge": 524288},
                "fps": 2,
                "max_frames": 8,
            },
        ),
    )

    result = config.build(pretrained_model_name_or_path="runtime-model")

    assert isinstance(result, DummyProcessor)
    assert calls == [
        ("video", "runtime-model", {"shortest_edge": 1024, "longest_edge": 524288}, 2, 8),
        ("processor", "outer-model", video_processor),
    ]


def test_recipe_config_resolves_nested_vlm_video_processor():
    def build_video_processor(**kwargs):
        return kwargs

    def build_processor(**kwargs):
        return kwargs

    config = RecipeConfig(
        ConfigNode(
            {
                "processor": {
                    "_target_": build_processor,
                    "pretrained_model_name_or_path": "outer-model",
                    "video_processor": {
                        "_target_": build_video_processor,
                        "size": {"shortest_edge": 1024, "longest_edge": 524288},
                        "fps": 2,
                        "max_frames": 8,
                    },
                },
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset",
                    "num_samples": 1,
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader.processor_config

    assert config.factory is build_processor
    assert config.kwargs == {"pretrained_model_name_or_path": "outer-model"}
    assert config.video_processor is not None
    assert config.video_processor.factory is build_video_processor
    assert config.video_processor.kwargs == {
        "size": {"shortest_edge": 1024, "longest_edge": 524288},
        "fps": 2,
        "max_frames": 8,
    }


def test_vlm_dataloader_selects_thd_collater(build_packed_dataloader):
    # THD carries document bounds in ``cu_seqlens``; the consumer declaration only concerns NEAT packing.
    result = build_packed_dataloader(
        packing=NeatPackConfig(packing_format="thd"), cp_size=4, consumes_packed_seq_ids=True
    )

    assert result.dataloader.collate_fn.func is packed_sequence_thd_vlm_collater
    assert result.dataloader.collate_fn.keywords == {"padding_idx": 0, "max_length": None}
    assert build_packed_dataloader.packing_kwargs["cp_size"] == 4


# The value space of ``packing_attn_implementation``: every ``BackendConfig.attn`` name, every
# Transformers dispatch key packing resolves, ``None``, and a string from neither vocabulary. Not
# every row is reachable from a shipped VLM recipe today -- the point is that the rule is total over
# the value space. ``None`` is reachable: the validation dataloader is built without the argument
# (``recipes/vlm/finetune.py``). The ``cp_size=8`` rows guard behaviour that predates this change --
# ``flash_attention_2`` there is what ``minimax_m3_vl_sft_tulu3_text_cp8_16k.yaml`` resolves to
# through ``packed_sequence.attn_implementation`` -- while the decision this change makes is at
# ``cp_size=1``.
_DENSE_MASK_CASES = (
    ("te", 1, True),
    ("flash_attention_2", 1, False),
    ("flash_attention_3", 1, False),
    ("flash_attention_4", 1, False),
    ("sdpa", 1, True),
    ("eager", 1, True),
    ("cudnn", 1, True),
    ("flex", 1, True),
    ("magi", 1, True),
    ("tilelang", 1, True),
    ("torch", 1, True),
    (None, 1, True),
    ("not-a-backend", 1, True),
    ("te", 8, False),
    ("sdpa", 8, False),
    ("flash_attention_2", 8, False),
)


@pytest.mark.parametrize("consumes_packed_seq_ids", [False, True], ids=["undeclared", "declared"])
@pytest.mark.parametrize(("attn_implementation", "cp_size", "dense"), _DENSE_MASK_CASES)
def test_vlm_dataloader_builds_the_dense_mask_only_for_backends_that_read_it(
    build_packed_dataloader, attn_implementation, cp_size, dense, consumes_packed_seq_ids
):
    """Only the flash-attention family, or a model that declares itself a consumer, skips the quadratic mask.

    Flash attention rebuilds ``cu_seqlens`` from the indexed ``[batch, sequence]`` document map, so it
    never reads the dense ``[batch, 1, sequence, sequence]`` tensor that building it costs. Every other
    name -- ``te`` included, since Transformer Engine reads any non-None mask as a padding mask -- keeps
    the dense mask unless ``consumes_packed_seq_ids`` says the model rebuilds document isolation from
    ``_packed_seq_ids`` itself, in which case the name is irrelevant (such a model may not even read it).
    The one-document pack below also pins the second half of the contract: whenever the dense mask
    is skipped, the compact map has to reach the model as ``_packed_seq_ids``, because nothing else
    carries document bounds. What the mask *contains* is the collater's own contract and is pinned
    by ``test_collate_fns.py``; this test only decides which representation the collater is asked for.
    """
    result = build_packed_dataloader(
        packing_attn_implementation=attn_implementation,
        cp_size=cp_size,
        consumes_packed_seq_ids=consumes_packed_seq_ids,
    )
    expect_dense = dense and not consumes_packed_seq_ids
    collate_fn = result.dataloader.collate_fn
    assert collate_fn.func is neat_packed_vlm_collater
    assert collate_fn.keywords["attn_implementation"] == attn_implementation

    # One pack holding one document: the dense mask is [1, 1, 8, 8] and the compact map is [1, 8].
    pack = {
        "input_ids": torch.zeros(8, dtype=torch.long),
        "labels": torch.zeros(8, dtype=torch.long),
        "attention_mask": torch.ones(8, dtype=torch.long),
        "position_ids": torch.arange(8),
    }
    batch = collate_fn([pack])

    assert batch["attention_mask"].shape == ((1, 1, 8, 8) if expect_dense else (1, 8))
    # One document is the case that can see this change: a multi-document pack emits
    # ``_packed_seq_ids`` on either path, so only here does the key track the representation.
    assert ("_packed_seq_ids" in batch) is not expect_dense


def test_compact_mask_backends_covers_every_flash_attention_name():
    """Every flash-attention name packing knows about has to be in the compact-mask set.

    ``datasets/`` imports nothing from ``models/``, so the set repeats those names as literals.
    A name added to ``_FLASH_ATTN_IMPLEMENTATIONS`` alone would silently go back to receiving the
    dense mask that flash attention cannot use.
    """
    from nemo_automodel.components.models.common.packing import _FLASH_ATTN_IMPLEMENTATIONS

    assert set(_FLASH_ATTN_IMPLEMENTATIONS) <= _COMPACT_MASK_BACKENDS


class _NoEosProcessor(ProcessorMixin):
    """Processor stub whose tokenizer lives at ``.tokenizer`` and that does not proxy
    tokenizer attributes (no ``eos_token_id``), matching a real LlavaProcessor."""

    attributes = []

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


@dataclass
class _RecordingTokenizerDatasetConfig:
    """TokenizerDatasetConfig stub (a text dataset) that records the arg it is built with."""

    accepts_tokenizer: bool = True
    received_tokenizer: object = None

    def build(self, *, tokenizer):
        self.received_tokenizer = tokenizer
        return ["sample"]


def test_build_source_forwards_processor_tokenizer_to_text_dataset():
    # A genuine processor does not expose tokenizer attributes (e.g. eos_token_id);
    # text datasets must receive processor.tokenizer, not the processor itself.
    inner_tokenizer = object()
    processor = _NoEosProcessor(inner_tokenizer)
    ds_cfg = _RecordingTokenizerDatasetConfig()
    config = VlmDataloaderConfig(
        dataset_config=ds_cfg,
        processor_config=VlmProcessorConfig(factory=lambda: processor),
    )

    dataset, returned_processor = config._build_source(
        pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, dataset_build_context=None
    )

    assert ds_cfg.received_tokenizer is inner_tokenizer  # tokenizer, not the processor
    assert returned_processor is processor  # processor still returned for pretokenization/collation
    assert dataset == ["sample"]


def test_vlm_dataloader_drops_dataset_tokenizer_block():
    # A `dataset.tokenizer` block is valid on the LLM path (the tokenizer is a runtime
    # build arg, popped before building the dataset config). The VLM path must drop it
    # too; otherwise it reaches ChatDatasetConfig, which rejects unknown fields.
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset",
                    "path_or_dataset_id": "unused.jsonl",
                    "seq_length": 512,
                    "tokenizer": {
                        "_target_": "transformers.AutoTokenizer.from_pretrained",
                        "pretrained_model_name_or_path": "unused-model",
                    },
                },
                "dataloader": {
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "num_workers": 0,
                },
            }
        )
    ).vlm_dataloader

    assert isinstance(config.dataset_config, ChatDatasetConfig)
    assert config.dataset_config.path_or_dataset_id == "unused.jsonl"
    assert config.dataset_config.seq_length == 512


def test_build_source_forwards_bare_tokenizer_unchanged():
    # When the processor slot holds a bare tokenizer (not a ProcessorMixin), it is
    # forwarded unchanged so tokenizer-only recipes keep working.
    bare_tokenizer = object()
    ds_cfg = _RecordingTokenizerDatasetConfig()
    config = VlmDataloaderConfig(
        dataset_config=ds_cfg,
        processor_config=VlmProcessorConfig(factory=lambda: bare_tokenizer),
    )

    config._build_source(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, dataset_build_context=None)

    assert ds_cfg.received_tokenizer is bare_tokenizer

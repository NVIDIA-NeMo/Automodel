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

import inspect
from dataclasses import dataclass, fields
from pathlib import Path

import pytest
import torch
from torch.utils.data import IterableDataset

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm import agent_chat, chat_dataset, squad
from nemo_automodel.components.datasets.llm.megatron.sampler import MegatronSamplerConfig
from nemo_automodel.components.datasets.llm.retrieval_dataset import RetrievalDatasetConfig
from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import InlineRetrievalDatasetConfig
from nemo_automodel.components.datasets.loader import (
    _DATASET_CONFIGS,
    CollatorConfig,
    DataloaderConfig,
    NeatPackingConfig,
    ParallelAwareDataloader,
    StreamDatasetConfig,
    TextStreamBuildContext,
    ThdPackingConfig,
    _resolve_target,
    make_collate_fn,
    make_packing_config,
)
from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT, BatchOutputFormat
from nemo_automodel.recipes._typed_config import RecipeConfig


class TokenizerAwareCollator:
    init_count = 0

    def __init__(self, tokenizer: object, prefix: str) -> None:
        type(self).init_count += 1
        self.tokenizer = tokenizer
        self.prefix = prefix

    def __call__(self, batch: list[str]) -> list[str]:
        return [self.prefix + item for item in batch]


@dataclass
class StaticDatasetConfig:
    values: object

    def build(self) -> object:
        return self.values


@dataclass
class StreamDataloaderConfig(DataloaderConfig):
    dataset_config: StreamDatasetConfig[TextStreamBuildContext, object]

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_config, StreamDatasetConfig):
            raise TypeError("This loader requires a stream dataset")

    @property
    def is_packed(self) -> bool:
        return self.dataset_config.is_packed

    @property
    def packed_sequence_length(self) -> int:
        return self.dataset_config.packed_sequence_length

    @property
    def output_format(self) -> BatchOutputFormat:
        return self.dataset_config.output_format


@pytest.mark.parametrize(("legacy_target", "config_target"), _DATASET_CONFIGS.items())
def test_legacy_dataset_registration_preserves_public_parameter_parity(legacy_target, config_target):
    legacy_factory = _resolve_target(legacy_target, {})
    config_type = _resolve_target(config_target, {})
    legacy_parameters = {
        name
        for name, parameter in inspect.signature(legacy_factory).parameters.items()
        if name != "self" and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    config_fields = {field.name for field in fields(config_type)}
    runtime_parameters = set(inspect.signature(config_type.build).parameters) - {"self"}

    # Megatron's legacy builder accepted schedule values separately. The typed config deliberately receives
    # their DatasetBuildSchedule owner as one runtime build argument instead of duplicating them in YAML.
    if legacy_target == "nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining":
        runtime_parameters.update(
            {"micro_batch_size", "global_batch_size", "trainer_max_steps", "trainer_val_check_interval"}
        )

    missing = legacy_parameters - config_fields - runtime_parameters
    assert not missing, f"{config_target} does not preserve legacy parameter(s): {', '.join(sorted(missing))}"


def test_typed_dataset_configs_forward_reviewed_legacy_parameters(monkeypatch):
    forwarded = {}

    def capture_chat_dataset(**kwargs):
        forwarded["chat"] = kwargs
        return "chat"

    def capture_squad_dataset(**kwargs):
        forwarded["squad"] = kwargs
        return "squad"

    def capture_agent_chat_dataset(tokenizer, **kwargs):
        forwarded["agent"] = {"tokenizer": tokenizer, **kwargs}
        return "agent"

    monkeypatch.setattr(chat_dataset, "ChatDataset", capture_chat_dataset)
    monkeypatch.setattr(squad, "make_squad_dataset", capture_squad_dataset)
    monkeypatch.setattr(agent_chat, "make_agent_chat_dataset", capture_agent_chat_dataset)
    tokenizer = object()

    assert chat_dataset.ChatDatasetConfig("data.jsonl", mask_history=True).build(tokenizer=tokenizer) == "chat"
    assert squad.SquadConfig(chat_template="template.jinja").build(tokenizer=tokenizer) == "squad"
    assert agent_chat.AgentChatConfig(path="data.jsonl", truncate_history=True).build(tokenizer=tokenizer) == "agent"
    assert forwarded["chat"]["mask_history"] is True
    assert forwarded["squad"]["chat_template"] == "template.jinja"
    assert forwarded["agent"]["truncate_history"] is True


def test_collator_config_builds_class_once_with_runtime_tokenizer():
    TokenizerAwareCollator.init_count = 0
    tokenizer = object()
    config = make_collate_fn(TokenizerAwareCollator, {"prefix": "query: "})

    assert isinstance(config, CollatorConfig)
    collator = config.build(tokenizer=tokenizer)

    assert TokenizerAwareCollator.init_count == 1
    assert collator.tokenizer is tokenizer
    assert collator(["one"]) == ["query: one"]
    assert collator(["two"]) == ["query: two"]
    assert TokenizerAwareCollator.init_count == 1


def test_dataloader_build_instantiates_collator_once():
    TokenizerAwareCollator.init_count = 0
    config = DataloaderConfig(
        dataset_config=StaticDatasetConfig(["one", "two"]),
        batch_size=1,
        collate_fn=CollatorConfig(TokenizerAwareCollator, {"prefix": "query: "}),
        num_workers=0,
    )

    loader = config.build(tokenizer=object(), dp_rank=0, dp_world_size=1)

    assert list(loader) == [["query: one"], ["query: two"]]
    assert TokenizerAwareCollator.init_count == 1


def test_recipe_config_resolves_top_level_retrieval_dataset_and_collator():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": ("nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig"),
                    "data_dir_list": ["train.jsonl"],
                    "n_passages": 3,
                },
                "dataloader": {
                    "collate_fn": {
                        "_target_": "tests.unit_tests.datasets.test_dataset_loader.TokenizerAwareCollator",
                        "prefix": "passage: ",
                    },
                    "batch_size": 2,
                    "num_workers": 0,
                },
            }
        )
    ).dataloader

    assert config is not None
    assert isinstance(config.dataset_config, RetrievalDatasetConfig)
    assert config.dataset_config.n_passages == 3
    assert isinstance(config.collate_fn, CollatorConfig)
    assert config.batch_size == 2


def test_recipe_config_supports_legacy_nested_dataset():
    raw = ConfigNode(
        {
            "dataloader": {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                    "data_dir_list": ["train.jsonl"],
                    "n_passages": 3,
                },
                "batch_size": 2,
            }
        }
    )

    with pytest.warns(FutureWarning, match="dataloader.dataset.*top-level `dataset`"):
        config = RecipeConfig(raw).dataloader

    assert config is not None
    assert isinstance(config.dataset_config, RetrievalDatasetConfig)
    assert config.dataset_config.n_passages == 3
    assert config.batch_size == 2


def test_recipe_config_prefers_top_level_dataset_over_legacy_nested_dataset():
    raw = ConfigNode(
        {
            "dataset": {
                "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                "data_dir_list": ["top-level.jsonl"],
                "n_passages": 5,
            },
            "dataloader": {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                    "data_dir_list": ["nested.jsonl"],
                    "n_passages": 2,
                }
            },
        }
    )

    with pytest.warns(FutureWarning, match="dataloader.dataset.*top-level `dataset`"):
        config = RecipeConfig(raw).dataloader

    assert config is not None
    assert config.dataset_config.data_dir_list == ["top-level.jsonl"]
    assert config.dataset_config.n_passages == 5


def test_recipe_config_supports_legacy_nested_validation_dataset():
    raw = ConfigNode(
        {
            "validation_dataloader": {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                    "data_dir_list": ["validation.jsonl"],
                    "n_passages": 4,
                },
                "batch_size": 2,
            }
        }
    )

    with pytest.warns(FutureWarning, match="validation_dataloader.dataset.*top-level `validation_dataset`"):
        configs = RecipeConfig(raw).validation_dataloaders

    assert set(configs) == {"default"}
    assert configs["default"].dataset_config.n_passages == 4
    assert configs["default"].batch_size == 2


def test_legacy_inline_retrieval_target_uses_exact_typed_config():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": (
                        "nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset"
                    ),
                    "data_dir_list": ["train.jsonl"],
                }
            }
        )
    ).dataloader

    assert config is not None
    assert isinstance(config.dataset_config, InlineRetrievalDatasetConfig)


def test_recipe_config_rejects_unknown_dataset_field():
    raw = ConfigNode(
        {
            "dataset": {
                "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset.RetrievalDatasetConfig",
                "data_dir_list": ["train.jsonl"],
                "n_passsages": 3,
            }
        }
    )

    with pytest.raises(TypeError, match="n_passsages"):
        _ = RecipeConfig(raw).dataloader


def test_packing_config_rejects_unknown_field():
    with pytest.raises(TypeError, match="packed_sequnce_size"):
        make_packing_config("thd", {"packed_sequence_size": 8, "packed_sequnce_size": 8})


def test_packing_config_warns_for_ignored_legacy_field():
    with pytest.warns(FutureWarning, match="split_across_pack"):
        config = make_packing_config("thd", {"packed_sequence_size": 8, "split_across_pack": False})

    assert isinstance(config, ThdPackingConfig)


def test_neat_packing_uses_default_contract(monkeypatch):
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.neat_packing.neat_pack_dataset",
        lambda dataset, **_: dataset,
    )
    config = NeatPackingConfig(packed_sequence_size=8)

    dataset, collate_fn = config.build([])

    assert dataset == []
    assert collate_fn.keywords["packing"] is DEFAULT_PACKED_SEQUENCE_CONTRACT


def test_packing_config_build_preserves_deprecated_attention_keyword(monkeypatch):
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.neat_packing.neat_pack_dataset",
        lambda dataset, **_: dataset,
    )
    config = NeatPackingConfig(packed_sequence_size=8)

    with pytest.warns(FutureWarning, match="attn_implementation is deprecated"):
        _, collate_fn = config.build([], attn_implementation="flash_attention_2")

    assert collate_fn.keywords["packing"].packed_mask_type == "flash_varlen"


def test_dataloader_build_preserves_deprecated_attention_keyword(monkeypatch):
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.neat_packing.neat_pack_dataset",
        lambda dataset, **_: dataset,
    )
    config = DataloaderConfig(
        dataset_config=StaticDatasetConfig([{"input_ids": [1]}]),
        packing=NeatPackingConfig(packed_sequence_size=8),
        shuffle=False,
    )

    with pytest.warns(FutureWarning, match="attn_implementation is deprecated"):
        loader = config.build(
            dp_rank=0,
            dp_world_size=1,
            attn_implementation="flash_attention_2",
        )

    assert loader.collate_fn.keywords["packing"].packed_mask_type == "flash_varlen"


def test_megatron_loader_preserves_schedule_and_sampler_config():
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining",
                    "paths": ["train"],
                    "splits_to_build": "train",
                },
                "dataloader": {"dataloader_type": "single"},
                "step_scheduler": {
                    "local_batch_size": 2,
                    "global_batch_size": 16,
                    "max_steps": 50,
                    "val_every_steps": 5,
                },
            }
        )
    ).dataloader

    assert config is not None
    assert config.dataset_builds_on_all_ranks is True
    assert config.dataset_build_schedule.local_batch_size == 2
    assert config.dataset_build_schedule.global_batch_size == 16
    assert config.dataset_build_schedule.max_steps == 50
    assert config.dataset_build_schedule.val_check_interval == 5
    assert isinstance(config.batch_sampler_config, MegatronSamplerConfig)
    assert next(iter(config.batch_sampler_config.build(dataset_len=32, rank=1, world_size=2))) == [2, 3]


def test_iterable_shuffle_failure_is_not_silently_ignored():
    class BrokenShuffleDataset(IterableDataset):
        def __iter__(self):
            yield "sample"

        def shard(self, *_):
            return self

        def shuffle(self, **_):
            raise ValueError("invalid shuffle configuration")

    with pytest.raises(ValueError, match="invalid shuffle configuration"):
        ParallelAwareDataloader(
            BrokenShuffleDataset(),
            dp_rank=0,
            dp_world_size=1,
            shuffle=True,
            num_workers=0,
        )


def test_prebatched_iterable_disables_automatic_batching():
    class PrebatchedDataset(IterableDataset):
        def __iter__(self):
            yield {"input_ids": torch.ones((2, 4), dtype=torch.long)}

        def shard(self, *_):
            return self

    config = DataloaderConfig(dataset_config=StaticDatasetConfig(PrebatchedDataset()), batch_size=None)

    batch = next(iter(config.build(dp_rank=0, dp_world_size=1)))

    assert batch["input_ids"].shape == (2, 4)


def test_rank_ordering_context_wraps_only_dataset_construction():
    events = []

    class BuildContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_):
            events.append("exit")

    class RecordingDatasetConfig:
        def build(self):
            events.append("dataset")
            return ["sample"]

    class RecordingPackingConfig:
        prepacked = False

        def build(self, dataset, **_):
            events.append("packing")
            return dataset, None

    config = DataloaderConfig(
        dataset_config=RecordingDatasetConfig(),
        packing=RecordingPackingConfig(),
        batch_size=1,
    )

    config.build(dp_rank=0, dp_world_size=1, dataset_build_context=BuildContext())

    assert events == ["enter", "dataset", "exit", "packing"]


def test_native_thd_packing_build_preserves_dataset_and_collator(monkeypatch):
    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

    documents = [{"input_ids": [2, 3]}]
    observed = []

    def pack(dataset, **kwargs):
        observed.append(kwargs)
        return dataset

    monkeypatch.setattr("nemo_automodel.components.datasets.llm.packed_sequence.pack_dataset", pack)
    dataset, collator = ThdPackingConfig(packed_sequence_size=8).build(documents, cp_size=2)
    assert dataset is documents
    assert collator is packed_sequence_thd_collater
    assert observed[0]["cp_size"] == 2


def test_native_packing_uses_model_owned_cp_alignment(monkeypatch):
    observed = []

    def pack(dataset, **kwargs):
        observed.append(kwargs["cp_size"])
        return dataset

    monkeypatch.setattr("nemo_automodel.components.datasets.llm.packed_sequence.pack_dataset", pack)
    monkeypatch.setattr(
        "nemo_automodel.components.datasets.loader.ParallelAwareDataloader", lambda *args, **kwargs: kwargs
    )
    config = DataloaderConfig(dataset_config=StaticDatasetConfig([]), packing=ThdPackingConfig(packed_sequence_size=8))
    config.build(dp_rank=0, dp_world_size=1, cp_size=2, packing_cp_size=1)
    assert observed == [1]


def test_thd_rejects_unsupported_model_instead_of_silently_disabling_packing():
    with pytest.raises(ValueError, match="THD packing requires a model that consumes seq_lens"):
        ThdPackingConfig(packed_sequence_size=8).build([], supports_seq_lens=False)


def test_prepacked_thd_is_reported_as_packed_without_online_packing():
    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater

    config = DataloaderConfig(dataset_config=StaticDatasetConfig([]), collate_fn=packed_sequence_thd_collater)
    assert config.is_packed and config.output_format == "thd"


def test_public_close_hook_is_optional_and_propagates_errors():
    from nemo_automodel.components.datasets.loader import close_dataloader

    close_dataloader(object())
    calls = []

    class Loader:
        def close(self):
            calls.append("closed")
            raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="cleanup failed"):
        close_dataloader(Loader())
    assert calls == ["closed"]


@pytest.mark.parametrize("fail_construction", [False, True])
def test_build_owns_ranked_rng_and_restores_ambient_state(monkeypatch, fail_construction):
    import random
    from types import SimpleNamespace

    import numpy as np

    from nemo_automodel.components.datasets import loader as loader_module
    from nemo_automodel.components.datasets.loader import PackingConfig
    from nemo_automodel.shared.rng import ScopedRNG

    def draw():
        return random.random(), float(np.random.random()), float(torch.rand(()))

    seed, rank = 31, 3
    with ScopedRNG(seed=seed + rank):
        expected = [draw() for _ in range(4)]
    seen = []

    class Source:
        def build(self):
            seen.append(draw())
            return [1]

    class Pack(PackingConfig):
        def build(self, dataset, **kwargs):
            seen.append(draw())
            return dataset, None

    class Sampler:
        def build(self, **kwargs):
            seen.append(draw())
            return [[0]]

    def loader(*args, **kwargs):
        seen.append(draw())
        assert kwargs["seed"] == seed  # The explicit sampler seed is not rank-adjusted.
        if fail_construction:
            raise RuntimeError("construction failed")
        return SimpleNamespace()

    monkeypatch.setattr(loader_module, "ParallelAwareDataloader", loader)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    config = DataloaderConfig(
        dataset_config=Source(),
        packing=Pack(packed_sequence_size=8),
        batch_sampler_config=Sampler(),
        seed=seed,
    )
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    if fail_construction:
        with pytest.raises(RuntimeError, match="construction failed"):
            config.build(dp_rank=0, dp_world_size=1)
    else:
        config.build(dp_rank=0, dp_world_size=1)
    assert seen == expected
    assert random.getstate() == py_state
    after_np = np.random.get_state()
    assert after_np[0] == np_state[0] and after_np[2:] == np_state[2:]
    np.testing.assert_array_equal(after_np[1], np_state[1])
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_emits_thd_compatibility_alias_warns():
    config = DataloaderConfig(dataset_config=StaticDatasetConfig([]))
    with pytest.warns(FutureWarning, match="output_format"):
        assert config.emits_thd is False


@pytest.mark.parametrize("output_format", ["batched", "thd"])
def test_stream_builder_returns_its_loader_and_receives_runtime_inputs(output_format):
    from nemo_automodel.components.datasets.loader import (
        DatasetBuildSchedule,
        StreamDatasetConfig,
        TextStreamBuildContext,
    )

    contexts = []
    stream_loader = iter(["batch"])

    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        is_packed = True

        @property
        def packed_sequence_length(self) -> int:
            return 17

        def build(self, *, context: TextStreamBuildContext) -> object:
            contexts.append(context)
            return stream_loader

    class NativeOnlyContext:
        def __enter__(self):
            raise AssertionError("A complete stream must bypass native dataset construction")

        def __exit__(self, *_):
            return False

    source = StreamConfig()
    source.output_format = output_format
    config = StreamDataloaderConfig(dataset_config=source, seed=91, batch_size=2)
    tokenizer = object()
    schedule = DatasetBuildSchedule(local_batch_size=2, global_batch_size=8, max_steps=10, val_check_interval=None)
    result = config.build(
        dp_rank=1,
        dp_world_size=2,
        tokenizer=tokenizer,
        cp_size=3,
        packing_cp_size=1,
        training_schedule=schedule,
        is_training=False,
        ignore_index=-17,
        dataset_build_context=NativeOnlyContext(),
    )

    assert result is stream_loader
    assert len(contexts) == 1
    context = contexts[0]
    assert context.tokenizer is tokenizer
    assert context.training_schedule is schedule
    assert (context.seed, context.batch_size, context.dp_rank, context.dp_world_size) == (91, 2, 1, 2)
    assert context.cp_size == 3
    assert context.is_training is False
    assert context.ignore_index == -17
    assert config.is_packed
    assert config.output_format == output_format
    assert config.packed_sequence_length == 17
    assert config.dataset_builds_on_all_ranks


@pytest.mark.parametrize("fail", [False, True])
def test_stream_builder_scopes_ranked_rng_and_preserves_shared_seed(monkeypatch, fail):
    from nemo_automodel.components.datasets.loader import StreamDatasetConfig, TextStreamBuildContext

    observations = []
    result = iter([])

    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        def build(self, *, context: TextStreamBuildContext) -> object:
            observations.append((context.seed, torch.rand(()).item()))
            if fail:
                raise ValueError("stream construction failed")
            return result

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 4)
    config = StreamDataloaderConfig(dataset_config=StreamConfig(), seed=77)
    before = torch.get_rng_state().clone()
    expected = torch.rand((), generator=torch.Generator().manual_seed(81)).item()

    if fail:
        with pytest.raises(ValueError, match="stream construction failed"):
            config.build(dp_rank=1, dp_world_size=2)
    else:
        assert config.build(dp_rank=1, dp_world_size=2) is result

    assert observations == [(77, expected)]
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize(
    "native_settings",
    [
        {"packing": ThdPackingConfig(packed_sequence_size=8)},
        {"collate_fn": lambda batch: batch},
        {"num_workers": 2},
        {"shuffle": False},
    ],
)
def test_stream_builder_rejects_native_settings_before_construction(native_settings):
    from nemo_automodel.components.datasets.loader import StreamDatasetConfig, TextStreamBuildContext

    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        def build(self, *, context: TextStreamBuildContext) -> object:
            raise AssertionError("Invalid native settings must fail before the stream is built")

    config = StreamDataloaderConfig(dataset_config=StreamConfig(), **native_settings)
    with pytest.raises(ValueError, match="stream dataset owns packing"):
        config.build(dp_rank=0, dp_world_size=1)


@pytest.mark.parametrize("packed_sequence", [{}, {"packed_sequence_size": 0}])
def test_stream_config_rejects_explicit_native_packing_at_the_recipe_boundary(packed_sequence):
    from nemo_automodel.components.datasets.loader import StreamDatasetConfig, TextStreamBuildContext

    @dataclass
    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        def build(self, *, context: TextStreamBuildContext) -> object:
            raise AssertionError("Config resolution must not build a stream")

    raw = ConfigNode({"dataset": {"_target_": StreamConfig}, "packed_sequence": packed_sequence})
    with pytest.raises(ValueError, match="Stream datasets own packing"):
        RecipeConfig(raw).dataloader


@pytest.mark.parametrize(
    "loader_target",
    [StreamDataloaderConfig, f"{__name__}.StreamDataloaderConfig"],
)
def test_recipe_selects_explicit_stream_loader_subclass(loader_target):
    @dataclass
    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        output_format = "thd"
        is_packed = True

        @property
        def packed_sequence_length(self) -> int:
            return 17

        def build(self, *, context: TextStreamBuildContext) -> object:
            raise AssertionError("Config resolution must not construct the stream")

    raw = ConfigNode(
        {
            "dataset": {"_target_": StreamConfig},
            "dataloader": {"_target_": loader_target},
            "seed": 71,
            "step_scheduler": {"local_batch_size": 3},
        }
    )
    config = RecipeConfig(raw).dataloader
    assert isinstance(config, StreamDataloaderConfig)
    assert config.seed == 71 and config.batch_size == 3
    assert config.output_format == "thd" and config.is_packed
    assert config.packed_sequence_length == 17


def test_native_loader_rejects_a_stream_without_a_compatible_config():
    @dataclass
    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        def build(self, *, context: TextStreamBuildContext) -> object:
            raise AssertionError("Invalid pairing must fail before construction")

    with pytest.raises(TypeError, match="compatible DataloaderConfig subclass"):
        DataloaderConfig(dataset_config=StreamConfig())
    with pytest.raises(TypeError, match="compatible DataloaderConfig subclass"):
        RecipeConfig(ConfigNode({"dataset": {"_target_": StreamConfig}})).dataloader


@pytest.mark.parametrize("explicit_native", [False, True])
def test_recipe_uses_dataset_loader_default_only_when_target_is_omitted(explicit_native):
    @dataclass
    class StreamConfig(StreamDatasetConfig[TextStreamBuildContext, object]):
        @property
        def dataloader_config_type(self) -> type[StreamDataloaderConfig]:
            return StreamDataloaderConfig

        def build(self, *, context: TextStreamBuildContext) -> object:
            raise AssertionError("Config resolution must not construct the stream")

    raw = ConfigNode(
        {
            "dataset": {"_target_": StreamConfig},
            "dataloader": {"_target_": "torch.utils.data.DataLoader"} if explicit_native else {},
            "seed": 71,
            "step_scheduler": {"local_batch_size": 3},
        }
    )
    if explicit_native:
        with pytest.raises(TypeError, match="compatible DataloaderConfig subclass"):
            RecipeConfig(raw).dataloader
    else:
        config = RecipeConfig(raw).dataloader
        assert isinstance(config, StreamDataloaderConfig)
        assert config.seed == 71 and config.batch_size == 3


def test_recipe_rejects_a_loader_target_that_is_not_a_config_subclass():
    raw = ConfigNode(
        {
            "dataset": {"_target_": StaticDatasetConfig, "values": []},
            "dataloader": {"_target_": list},
        }
    )
    with pytest.raises(ValueError, match="DataloaderConfig subclass"):
        RecipeConfig(raw).dataloader


_SYNTHETIC_PRETRAIN_CONFIGS = sorted(
    path
    for path in (Path(__file__).resolve().parents[3] / "examples/llm_pretrain").rglob("*.yaml")
    if "# Synthetic pretraining counterpart of" in path.read_text()
)


@pytest.mark.parametrize("config_path", _SYNTHETIC_PRETRAIN_CONFIGS, ids=lambda path: path.stem)
def test_synthetic_pretraining_uses_recipe_batch_size(config_path: Path) -> None:
    """Real migrated loaders honor local batch overrides without an extra axis."""
    import yaml

    raw = yaml.safe_load(config_path.read_text())
    # Keep model vocabulary/exclusions, but make the CPU smoke workload tiny.
    raw["dataset"]["seq_len"] = 16
    raw["step_scheduler"]["local_batch_size"] = 3
    raw["dataloader"].update(num_workers=0, pin_memory=False)
    data_config = {key: raw[key] for key in ("dataset", "dataloader", "step_scheduler")}
    config = RecipeConfig(ConfigNode(data_config)).dataloader
    batch = next(iter(config.build(tokenizer=None, dp_rank=0, dp_world_size=1)))
    assert batch["input_ids"].shape == (3, 16)
    assert batch["labels"].shape == (3, 16)
    torch.testing.assert_close(batch["labels"][:, :-1], batch["input_ids"][:, 1:])
    assert batch["labels"][:, -1].eq(-100).all()
    assert not batch["padding_mask"].any()
    assert "position_ids" not in batch
    assert batch["input_ids"].min() >= 0
    assert batch["input_ids"].max() < raw["dataset"]["vocab_size"]
    for token_id in raw["dataset"]["exclude_token_ids"]:
        assert batch["input_ids"].ne(token_id).all()

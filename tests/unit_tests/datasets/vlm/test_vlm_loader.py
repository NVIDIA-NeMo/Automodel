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

import logging
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from transformers import ProcessorMixin

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDatasetConfig
from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT
from nemo_automodel.components.datasets.vlm import loader as vlm_loader
from nemo_automodel.components.datasets.vlm.collate_fns import (
    neat_packed_vlm_collater,
    packed_sequence_thd_vlm_collater,
)
from nemo_automodel.components.datasets.vlm.datasets import CordV2DatasetConfig, PreTokenizedDatasetWrapperConfig
from nemo_automodel.components.datasets.vlm.loader import (
    VlmCollatorConfig,
    VlmDataloaderConfig,
    VlmProcessorConfig,
    VlmVideoProcessorConfig,
)
from nemo_automodel.components.datasets.vlm.mock import MockVlmDatasetConfig
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import NeatPackConfig
from nemo_automodel.components.distributed.thd_utils import process_input_for_thd
from nemo_automodel.recipes._typed_config import RecipeConfig

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def one_visible_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)


class DummyProcessor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(pad_token_id=0)


def _collate_prefixed_examples(examples, *, processor, prefix):
    return [prefix + example for example in examples], processor.tokenizer.pad_token_id


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


@pytest.mark.parametrize("multiprocessing_context", [None, "spawn"])
def test_vlm_dataloader_builds_processor_and_dataset_inside_context_then_iterates(monkeypatch, multiprocessing_context):
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    events = []
    processor = DummyProcessor()

    def build_processor():
        events.append("processor")
        return processor

    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig(events),
        processor_config=VlmProcessorConfig(factory=build_processor),
        collator=VlmCollatorConfig(factory=_collate_prefixed_examples, kwargs={"prefix": "item:"}),
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
    result.dataloader.multiprocessing_context = multiprocessing_context
    batches = list(result.dataloader)
    batch, padding_idx = batches[0]

    assert events == ["enter", "processor", "dataset", "exit"]
    assert result.processor is processor
    assert batch == ["item:one", "item:two"]
    assert padding_idx == processor.tokenizer.pad_token_id


@pytest.mark.parametrize(
    ("affinity", "cpu_count", "num_gpus", "expected"),
    [
        ({4, 5}, 128, 1, 2),
        (set(range(24)), 128, 2, 8),
        (set(range(64)), 128, 8, 32),
        (set(range(6)), 128, 2, 6),
        (OSError("unavailable"), 64, 2, 8),
        (AttributeError("unavailable"), 4, 2, 4),
        (NotImplementedError(), None, 1, 1),
        (set(), 2, 1, 2),
        (set(range(24)), 128, 0, 24),
        (OSError("unavailable"), 3, 0, 3),
        (NotImplementedError(), None, 0, 1),
    ],
)
def test_zero_vlm_workers_caps_by_gpus_and_available_cpus_without_mutating_config(
    monkeypatch, caplog, affinity, cpu_count, num_gpus, expected
):
    def get_affinity(_):
        if isinstance(affinity, Exception):
            raise affinity
        return affinity

    monkeypatch.setattr(vlm_loader, "os", SimpleNamespace(sched_getaffinity=get_affinity, cpu_count=lambda: cpu_count))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: num_gpus)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=DummyProcessor),
        num_workers=0,
        persistent_workers=True,
        prefetch_factor=3,
    )

    with caplog.at_level(logging.WARNING, logger=vlm_loader.__name__):
        for _ in range(2):
            result = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=1)
            assert result.dataloader.num_workers == expected
            assert result.dataloader.persistent_workers is (expected > 0)
            assert result.dataloader.prefetch_factor == (3 if expected > 0 else None)

    assert config.num_workers == 0
    messages = [record.getMessage() for record in caplog.records if record.name == vlm_loader.__name__]
    worker_messages = [message for message in messages if "num_workers=0" in message]
    assert len(worker_messages) == 2
    assert all(f"using num_workers={expected}" in message for message in worker_messages)
    assert all("prefetching" in message and "per dataloader" in message for message in worker_messages)


def test_positive_vlm_workers_is_preserved_without_auto_selection_warning(monkeypatch, caplog):
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: set(range(8)))
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=DummyProcessor),
        num_workers=3,
    )

    result = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=1)

    assert result.dataloader.num_workers == 3
    assert config.num_workers == 3
    assert "num_workers=0" not in caplog.text


def test_negative_vlm_workers_is_rejected():
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=DummyProcessor),
        num_workers=-1,
    )

    with pytest.raises(
        ValueError,
        match=r"^VLM num_workers must be zero \(automatic workers\) or a positive worker count\.$",
    ):
        config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=1)


def _collate_worker_samples(examples, *, processor):
    return examples, torch.utils.data.get_worker_info().id


class _IndexDataset(torch.utils.data.Dataset):
    def __init__(self, offset=0):
        self.offset = offset

    def __len__(self):
        return 11

    def __getitem__(self, index):
        return self.offset + index


class _ResumeDataset(_IndexDataset):
    def state_dict(self):
        """Return a mapping containing an int64 scalar offset tensor of shape []."""
        return {"offset": torch.tensor(self.offset)}

    def load_state_dict(self, state):
        """Restore the sample-index offset.

        Args:
            state: Mapping with ``offset``, an int64 scalar tensor of shape [].
        """
        self.offset = int(state["offset"])


class _WorkerLocalDataset(_ResumeDataset):
    def __getitem__(self, index):
        self.offset += 1
        return super().__getitem__(index)


class _WorkerOnlyRestoreDataset(_ResumeDataset):
    def load_state_dict(self, state):
        """Restore the sample-index offset inside a worker.

        Args:
            state: Mapping with ``offset``, an int64 scalar tensor of shape [].
        """
        if torch.utils.data.get_worker_info() is None:
            raise RuntimeError("This dataset must restore inside a worker")
        super().load_state_dict(state)


@dataclass
class _ResumeDatasetConfig:
    dataset_type: type[torch.utils.data.Dataset] = _ResumeDataset
    offset: int = 0

    def build(self):
        return self.dataset_type(self.offset)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("multiprocessing_context", [None, "spawn"])
@pytest.mark.parametrize(
    ("consumed", "drop_last", "rank", "configured_workers"),
    [(0, False, 0, 0), (1, False, 1, 0), (2, False, 0, 0), (1, True, 1, 0), (3, False, 1, 0), (1, False, 1, 3)],
    ids=["initial", "mid-epoch", "last-batch", "drop-last", "finished", "explicit-workers"],
)
def test_vlm_workers_resume_legacy_synchronous_checkpoint(
    monkeypatch, tmp_path, multiprocessing_context, consumed, drop_last, rank, configured_workers
):
    from torchdata.stateful_dataloader import StatefulDataLoader

    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    config = VlmDataloaderConfig(
        dataset_config=_ResumeDatasetConfig(),
        processor_config=VlmProcessorConfig(factory=lambda: None),
        collator=VlmCollatorConfig(factory=_collate_worker_samples),
        num_workers=configured_workers,
        drop_last=drop_last,
    )
    # The old recipe used the same distributed sampler, but processed samples in the main process.
    dataset = _ResumeDataset(offset=100)
    sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True)
    sampler.set_epoch(7)
    old = StatefulDataLoader(dataset, sampler=sampler, batch_size=4, num_workers=0, drop_last=drop_last)
    iterator = iter(old)
    for _ in range(min(consumed, len(old))):
        next(iterator)
    if consumed > len(old):
        with pytest.raises(StopIteration):
            next(iterator)
    state = old.state_dict()
    state_dir = tmp_path / "dataloader"
    state_dir.mkdir()
    torch.save(state, state_dir / f"dataloader_dp_rank_{rank}.pt")
    expected = list(old if state["_iterator_finished"] else iterator)

    resumed = config.build(
        pretrained_model_name_or_path="unused", dp_rank=rank, dp_world_size=2, batch_size=4
    ).dataloader
    resumed.multiprocessing_context = multiprocessing_context
    resumed.sampler.set_epoch(7)
    reader = Checkpointer.__new__(Checkpointer)
    reader.dp_rank = rank
    reader.config = SimpleNamespace(allow_legacy_pickle_restore=False)
    rng_before = torch.get_rng_state().clone()
    reader.load_on_dp_ranks(resumed, "dataloader", str(tmp_path))
    assert torch.equal(torch.get_rng_state(), rng_before)
    batches = list(resumed)

    assert resumed.num_workers == (configured_workers or 2)
    assert config.num_workers == configured_workers
    assert [samples for samples, _ in batches] == [batch.tolist() for batch in expected]
    assert all(worker_id in range(resumed.num_workers) for _, worker_id in batches)
    assert resumed.dataset.offset == 100
    assert state["dataset_state"]["offset"].item() == 100
    assert "_snapshot" not in state


@pytest.mark.timeout(60)
@pytest.mark.parametrize("num_gpus", [0, 1])
@pytest.mark.parametrize("consumed", [0, 1])
@pytest.mark.parametrize(("available_cpus", "configured_workers"), [(2, 0), (1, 0), (3, 0), (3, 3)])
def test_vlm_workers_roundtrip_native_checkpoint(
    monkeypatch, tmp_path, num_gpus, consumed, available_cpus, configured_workers
):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: num_gpus)
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    config = VlmDataloaderConfig(
        dataset_config=_ResumeDatasetConfig(_IndexDataset, offset=200),
        processor_config=VlmProcessorConfig(factory=lambda: None),
        collator=VlmCollatorConfig(factory=_collate_worker_samples),
        num_workers=0,
    )
    original = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=3).dataloader
    original.sampler.set_epoch(7)
    original.snapshot_every_n_steps = 3
    original.load_state_dict({})
    iterator = iter(original)
    for _ in range(consumed):
        next(iterator)
    checkpointer = Checkpointer.__new__(Checkpointer)
    checkpointer.dp_rank = checkpointer.tp_rank = checkpointer.pp_rank = 0
    checkpointer.process_group = None
    checkpointer.config = SimpleNamespace(allow_legacy_pickle_restore=False)
    checkpointer.save_on_dp_ranks(original, "dataloader", str(tmp_path))
    path = tmp_path / "dataloader" / "dataloader_dp_rank_0.pt"
    expected = list(iterator)

    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: set(range(available_cpus)))
    resumed_config = replace(config, num_workers=configured_workers)
    resumed = resumed_config.build(
        pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=3
    ).dataloader
    assert resumed.num_workers == (configured_workers or available_cpus)
    resumed.sampler.set_epoch(7)
    state = torch.load(path, weights_only=True)
    assert state["_steps_since_snapshot"] == consumed
    rng_before = torch.get_rng_state().clone()
    checkpointer.load_on_dp_ranks(resumed, "dataloader", str(tmp_path))
    assert torch.equal(torch.get_rng_state(), rng_before)

    actual = list(resumed)
    assert [samples for samples, _ in actual] == [samples for samples, _ in expected]
    assert all(worker_id in range(resumed.num_workers) for _, worker_id in actual)
    assert resumed.num_workers == (configured_workers or available_cpus)
    assert resumed_config.num_workers == configured_workers
    assert state["_snapshot"]["_main_snapshot"]["_num_workers"] == 2
    assert config.num_workers == 0


@pytest.mark.timeout(60)
@pytest.mark.parametrize("configured_workers", [0, 3])
@pytest.mark.parametrize("dataset_type", [_ResumeDataset, _WorkerLocalDataset])
def test_vlm_worker_count_change_requires_stateless_datasets(monkeypatch, configured_workers, dataset_type):
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    config = VlmDataloaderConfig(
        dataset_config=_ResumeDatasetConfig(dataset_type),
        processor_config=VlmProcessorConfig(factory=lambda: None),
        collator=VlmCollatorConfig(factory=_collate_worker_samples),
    )
    original = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2).dataloader
    iterator = iter(original)
    next(iterator)
    state = deepcopy(original.state_dict())
    list(iterator)
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0})
    resumed = (
        replace(config, num_workers=configured_workers)
        .build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2)
        .dataloader
    )
    expected_workers = configured_workers or 1

    with pytest.raises(ValueError, match=f"from 2 to {expected_workers}.*stateless map-style datasets"):
        resumed.load_state_dict(state)
    assert resumed.num_workers == expected_workers
    assert config.num_workers == 0


@pytest.mark.timeout(60)
@pytest.mark.parametrize("multiprocessing_context", [None, "spawn"])
def test_native_stateful_vlm_checkpoint_restores_only_in_workers(monkeypatch, multiprocessing_context):
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    config = VlmDataloaderConfig(
        dataset_config=_ResumeDatasetConfig(_WorkerOnlyRestoreDataset, offset=100),
        processor_config=VlmProcessorConfig(factory=lambda: None),
        collator=VlmCollatorConfig(factory=_collate_worker_samples),
        shuffle=False,
    )
    original = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2).dataloader
    original.multiprocessing_context = multiprocessing_context
    iterator = iter(original)
    next(iterator)
    state = deepcopy(original.state_dict())
    expected = list(iterator)
    resumed = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2).dataloader
    resumed.multiprocessing_context = multiprocessing_context
    try:
        resumed.load_state_dict(state)
        assert list(resumed) == expected
    finally:
        if resumed._iterator is not None:
            resumed._iterator._shutdown_workers()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("persistent_workers", [False, True])
@pytest.mark.parametrize("checkpoint_at_end", [False, True])
def test_migrated_vlm_checkpoint_can_be_saved_and_resumed_again(
    monkeypatch, tmp_path, persistent_workers, checkpoint_at_end
):
    from torchdata.stateful_dataloader import StatefulDataLoader

    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    config = VlmDataloaderConfig(
        dataset_config=_ResumeDatasetConfig(),
        processor_config=VlmProcessorConfig(factory=lambda: None),
        collator=VlmCollatorConfig(factory=_collate_worker_samples),
        shuffle=False,
        persistent_workers=persistent_workers,
    )
    dataset = _ResumeDataset(offset=100)
    old = StatefulDataLoader(
        dataset,
        sampler=torch.utils.data.DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=False),
        batch_size=2,
        num_workers=0,
    )
    next(iter(old))
    directory = tmp_path / "dataloader"
    directory.mkdir()
    path = directory / "dataloader_dp_rank_0.pt"
    torch.save(old.state_dict(), path)
    reader = Checkpointer.__new__(Checkpointer)
    reader.dp_rank = 0
    reader.config = SimpleNamespace(allow_legacy_pickle_restore=False)
    first = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2).dataloader
    second = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2).dataloader
    try:
        reader.load_on_dp_ranks(first, "dataloader", str(tmp_path))
        first.load_state_dict({})
        iterator = iter(first)
        assert next(iterator)[0] == [102, 103]
        if checkpoint_at_end:
            list(iterator)
        torch.save(first.state_dict(), path)
        expected = list(first if checkpoint_at_end else iterator)
        reader.load_on_dp_ranks(second, "dataloader", str(tmp_path))
        actual = list(second)
        assert [samples for samples, _ in actual] == [samples for samples, _ in expected]
        assert [sample for samples, _ in actual for sample in samples] == list(
            range(100 if checkpoint_at_end else 104, 111)
        )
    finally:
        for loader in (first, second):
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()


@pytest.mark.parametrize("validation", [False, True])
def test_vlm_recipe_auto_workers_prefetch_and_iterate_in_workers(monkeypatch, validation):
    monkeypatch.setattr(vlm_loader.os, "sched_getaffinity", lambda _: {0, 1})
    recipe = RecipeConfig(
        ConfigNode(
            {
                "validation_dataset" if validation else "dataset": {
                    "_target_": "nemo_automodel.components.datasets.vlm.mock.build_mock_vlm_dataset",
                },
                "validation_dataloader" if validation else "dataloader": {
                    "num_workers": 0,
                    "shuffle": False,
                },
            }
        )
    )
    config = recipe.vlm_validation_dataloader if validation else recipe.vlm_dataloader
    config.dataset_config = StaticDatasetConfig([])
    config.processor_config = VlmProcessorConfig(factory=lambda: None)
    config.collator = VlmCollatorConfig(factory=_collate_worker_samples)
    result = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=1)

    batches = list(result.dataloader)

    assert result.dataloader.num_workers == 2
    assert result.dataloader.prefetch_factor == 2
    assert batches == [(["one"], 0), (["two"], 1)]
    assert config.num_workers == 0


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


def test_vlm_dataloader_selects_thd_collater(monkeypatch):
    processor = DummyProcessor()
    packing_kwargs = {}

    def _build_packing(self, dataset, **kwargs):
        packing_kwargs.update(kwargs)
        return dataset

    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", _build_packing)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=lambda: processor),
        pretokenization=PreTokenizedDatasetWrapperConfig(),
        packing=NeatPackConfig(packing_format="thd"),
        shuffle=False,
    )

    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        cp_size=4,
    )

    assert result.dataloader.collate_fn.func is packed_sequence_thd_vlm_collater
    assert result.dataloader.collate_fn.keywords == {"padding_idx": 0, "max_length": None}
    assert packing_kwargs["cp_size"] == 4


def test_qwen3_vl_moe_recipe_preserves_te_document_boundaries(monkeypatch):
    recipe_path = Path(__file__).resolve().parents[4] / "examples/vlm_finetune/qwen3/qwen3_vl_moe_30b_neat_packing.yaml"
    raw = yaml.safe_load(recipe_path.read_text())
    config = RecipeConfig(
        ConfigNode({key: raw[key] for key in ("dataset", "dataloader", "packed_sequence")})
    ).vlm_dataloader
    sample = {
        "input_ids": torch.tensor([1, 2, 3, 4, 5]),
        "labels": torch.tensor([2, -100, 4, 5, -100]),
        "position_ids": torch.tensor([[0, 1, 0, 1, 2]] * 3),
        "attention_mask": torch.tensor([1, 1, 2, 2, 2]),
    }
    # Replace media loading/tokenization/knapsack work with an already-packed sample.
    # The recipe's typed configuration still selects and runs the real collator.
    monkeypatch.setattr(type(config.dataset_config), "build", lambda self: [sample])
    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", lambda self, dataset, **kwargs: dataset)
    config.processor_config = VlmProcessorConfig(factory=DummyProcessor)

    result = config.build(pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=1)
    batch = result.dataloader.collate_fn([sample])

    assert "attention_mask" not in batch
    assert batch["qkv_format"] == "thd"
    thd = process_input_for_thd(batch)
    assert thd["cu_seqlens"].tolist() == [0, 2, 5]
    assert tuple(thd["position_ids"].shape) == (3, 1, 5)


def test_vlm_dataloader_neat_packing_uses_default_contract(monkeypatch):
    processor = DummyProcessor()
    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", lambda self, dataset, **kwargs: dataset)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=lambda: processor),
        pretokenization=PreTokenizedDatasetWrapperConfig(),
        packing=NeatPackConfig(),
        shuffle=False,
    )

    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
    )

    assert result.dataloader.collate_fn.keywords["packing"] is DEFAULT_PACKED_SEQUENCE_CONTRACT


def test_vlm_dataloader_preserves_deprecated_python_packing_backend(monkeypatch):
    processor = DummyProcessor()
    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", lambda self, dataset, **kwargs: dataset)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=lambda: processor),
        pretokenization=PreTokenizedDatasetWrapperConfig(),
        packing=NeatPackConfig(attn_implementation="flash_attention_2"),
        shuffle=False,
    )

    with pytest.warns(FutureWarning, match="attn_implementation is deprecated"):
        result = config.build(
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=2,
        )

    assert result.dataloader.collate_fn.keywords["packing"].packed_mask_type == "flash_varlen"


def test_vlm_dataloader_skips_dense_neat_packing_mask_under_cp(monkeypatch):
    processor = DummyProcessor()
    monkeypatch.setattr(PreTokenizedDatasetWrapperConfig, "build", lambda self, dataset, processor: dataset)
    monkeypatch.setattr(NeatPackConfig, "build", lambda self, dataset, **kwargs: dataset)
    config = VlmDataloaderConfig(
        dataset_config=StaticDatasetConfig([]),
        processor_config=VlmProcessorConfig(factory=lambda: processor),
        pretokenization=PreTokenizedDatasetWrapperConfig(),
        packing=NeatPackConfig(),
        shuffle=False,
    )

    packing = SimpleNamespace(
        packed_mask_type="block_causal",
        requires_packed_sequence_metadata=False,
    )
    result = config.build(
        pretrained_model_name_or_path="unused",
        dp_rank=0,
        dp_world_size=1,
        batch_size=2,
        packing_contract=packing,
        cp_size=32,
    )

    assert result.dataloader.collate_fn.func is neat_packed_vlm_collater
    assert result.dataloader.collate_fn.keywords["packing"] is packing
    assert result.dataloader.collate_fn.keywords["materialize_4d_mask"] is False


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

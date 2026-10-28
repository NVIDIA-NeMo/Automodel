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

"""Source policies, worker tokenization and calibrated mixtures through real pipelines."""

import json
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.loader import DataloaderConfig
from nemo_automodel.components.datasets.zephon import ZephonDataloaderConfig
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonTextSourceConfig,
)
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.utils.zephon_text_fixtures import config as config
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


def test_process_worker_tokenization(config, tokenizer):
    config = replace(
        config, zephon=replace(config.zephon, runtime=replace(config.zephon.runtime, runner="process", mtp_mode=True))
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        assert next(loader)["labels"].tolist()[0] == [3, 4, 1, 3, 4, 5, 1]


def _mixture_config(tmp_path, *, estimation, policy="repeat"):
    sources = []
    for name, text in (("short", "w0"), ("long", " ".join(["w1"] * 7))):
        path = tmp_path / name
        path.mkdir(exist_ok=True)
        (path / "data.jsonl").write_text((json.dumps({"text": text}) + "\n") * 100)
        sources.append(ZephonSource(name=name, path=str(path), fmt="jsonl"))
    return ZephonDataloaderConfig(
        dataset_config=ZephonTextSourceConfig(tokenize_special_tokens="eos"),
        zephon=ZephonOptions(
            sources=tuple(sources),
            packing=ZephonPackingOptions(seq_len=31),
            source=ZephonSourceOptions(
                chunk_size=16,
                shuffle_shards=False,
                shuffle_within_shard=False,
                shuffle_block_size=None,
                token_estimation=estimation,
                exhausted_policy=policy,
            ),
            runtime=ZephonRuntimeOptions(runner="threads", mtp_mode=False),
        ),
    )


def _short_token_share(loader, batches=80):
    total = short = 0
    for _ in range(batches):
        batch = next(loader)
        tokens = torch.cat((batch["input_ids"], batch["labels"][:, -1:]), dim=1)
        short += int(tokens.eq(2).sum())
        short += int((tokens[:, 1:].eq(1) & tokens[:, :-1].eq(2)).sum())
        total += tokens.numel()
    return short / total


def test_token_aware_mixture_uses_tokens_not_document_counts(tmp_path, tokenizer):
    sample_config = _mixture_config(tmp_path, estimation=False)
    token_config = replace(
        sample_config,
        zephon=replace(
            sample_config.zephon,
            source=replace(
                sample_config.zephon.source,
                token_estimation={"calibration_samples": 16, "calibration_shards_min": 1, "calibration_shards_max": 1},
            ),
        ),
    )
    sample_loader = sample_config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    token_loader = token_config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    try:
        assert 0.15 < _short_token_share(sample_loader) < 0.25
        assert 0.43 < _short_token_share(token_loader) < 0.57
    finally:
        sample_loader.close()
        token_loader.close()


def test_token_mixture_checkpoint_preserves_calibration_and_replay(tmp_path, tokenizer):

    config = _mixture_config(tmp_path, estimation={"primer": {"short": 0.1, "long": 0.2}})
    config = replace(
        config,
        zephon=replace(
            config.zephon, packing=replace(config.zephon.packing, pack_algorithm="best_fit", pack_num_bins=8)
        ),
    )
    original = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    resumed = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    try:
        next(original)
        checkpoint = original.state_dict()
        assert b"token_ratios" in checkpoint["zephon"]
        expected = [next(original) for _ in range(4)]
        resumed.load_state_dict(checkpoint)
        for batch in expected:
            torch.testing.assert_close(next(resumed), batch)
    finally:
        original.close()
        resumed.close()


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit", "best_fit_wrap"])
def test_padding_algorithms_preserve_real_eos_and_mask_padding(config, tokenizer, algorithm):
    config = replace(
        config,
        batch_size=1,
        zephon=replace(
            config.zephon,
            packing=replace(
                config.zephon.packing,
                seq_len=10,
                pack_algorithm=algorithm,
                pack_num_bins=8 if algorithm != "best_fit_wrap" else None,
            ),
        ),
    )
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1, ignore_index=-7)) as loader:
        batches = list(loader)
        assert batches
        assert any(batch["padding_mask"].any() for batch in batches)
        for batch in batches:
            assert batch["input_ids"].shape == (1, 10)
            assert batch["labels"][batch["padding_mask"]].eq(-7).all()
            assert batch["labels"].eq(1).any()


@pytest.mark.parametrize("policy", ["stop", "redistribute"])
def test_finite_mixture_policies(tmp_path, tokenizer, policy):
    config = _mixture_config(tmp_path, estimation=False, policy=policy)
    with closing(config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as loader:
        assert 0 < len(list(loader)) < 100


def test_first_fit_drop_and_reject_oversized_documents(config, tokenizer):
    path = Path(config.zephon.sources[0].path)
    (path / "text.jsonl").write_text(
        "\n".join(json.dumps({"text": text}) for text in ("w0 w1", " ".join(["w2"] * 30), "w3 w4")) + "\n"
    )
    dropped_config = replace(
        config,
        batch_size=1,
        zephon=replace(
            config.zephon,
            packing=replace(
                config.zephon.packing, seq_len=8, pack_algorithm="first_fit", pack_num_bins=2, drop_oversized=True
            ),
            source=replace(config.zephon.source, chunk_size=1),
        ),
    )
    with closing(dropped_config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)) as dropped:
        batches = list(dropped)
        assert batches
        assert not any(batch["input_ids"].eq(4).any() for batch in batches)
    with closing(
        replace(
            dropped_config,
            zephon=replace(dropped_config.zephon, packing=replace(dropped_config.zephon.packing, drop_oversized=False)),
        ).build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    ) as rejected:
        with pytest.raises((ValueError, RuntimeError), match="[Oo]versize|exceed|length"):
            list(rejected)


def test_raw_text_process_workers_use_effective_unsaved_tokenizer(config, tokenizer):
    tokenizer.add_tokens(["unsaved-vocabulary-token"])
    token_id = tokenizer.convert_tokens_to_ids("unsaved-vocabulary-token")
    tokenizer.name_or_path = ""
    path = Path(config.zephon.sources[0].path)
    (path / "text.jsonl").write_text((json.dumps({"text": "unsaved-vocabulary-token w0 w1"}) + "\n") * 36)
    config = replace(
        config, zephon=replace(config.zephon, runtime=replace(config.zephon.runtime, runner="process", mtp_mode=True))
    )
    loader = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    restored = config.build(tokenizer=tokenizer, dp_rank=0, dp_world_size=1)
    try:
        assert next(loader)["input_ids"].eq(token_id).any()
        checkpoint = loader.state_dict()
        expected = next(loader)
        restored.load_state_dict(checkpoint)
        torch.testing.assert_close(next(restored), expected)
    finally:
        loader.close()
        restored.close()


def test_native_validation_remains_a_separate_finite_loader(tmp_path: Path) -> None:
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {"_target_": "nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig"},
                "dataloader": {
                    "_target_": "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig",
                    "zephon": {
                        "sources": [{"name": "training", "path": str(tmp_path)}],
                    },
                },
                "step_scheduler": {"max_steps": 100, "local_batch_size": 1},
                "validation_dataset": {
                    "_target_": "nemo_automodel.components.datasets.llm.mock.MockUnpackedDatasetConfig",
                    "num_sentences": 3,
                    "mean_len": 3,
                    "std_len": 0,
                    "vocab_size": 10,
                },
                "validation_dataloader": {"shuffle": False, "num_workers": 0},
            }
        )
    )
    assert isinstance(config.dataloader, ZephonDataloaderConfig)
    native_config = config.validation_dataloaders["default"]
    assert isinstance(native_config, DataloaderConfig)
    validation = native_config.build(dp_rank=0, dp_world_size=1)
    first_pass, second_pass = list(validation), list(validation)
    assert len(first_pass) == len(second_pass) == 3
    for first, second in zip(first_pass, second_pass):
        torch.testing.assert_close(first["input_ids"], second["input_ids"])

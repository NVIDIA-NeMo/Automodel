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

"""Declarative stream configuration and validation without pipeline execution."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.loader import (
    DataloaderConfig,
    DatasetBuildSchedule,
    NeatPackingConfig,
    TextStreamBuildContext,
    make_dataset_config,
)
from nemo_automodel.components.datasets.zephon.config import (
    ZephonCacheOptions,
    ZephonDataloaderConfig,
    ZephonOptions,
    ZephonPackingOptions,
    ZephonParquetRGCacheOptions,
    ZephonPrefetchOptions,
    ZephonPretokenizedTextDatasetConfig,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonTextDatasetConfig,
    ZephonWorkerOptions,
)
from nemo_automodel.components.models.common.packing import PackingCapabilities
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.utils.zephon_text_fixtures import tokenizer as tokenizer


@pytest.mark.parametrize(
    "source, expected, local",
    [
        ("org/corpus", "hf://org/corpus/train", False),
        ("org/corpus", "org/corpus", True),
        ("hf://org/corpus/validation", "hf://org/corpus/validation", False),
    ],
)
def test_single_native_source_preserves_hf_and_local_selection(tmp_path, monkeypatch, source, expected, local):
    monkeypatch.chdir(tmp_path)
    if local:
        Path(source).mkdir(parents=True)
    config = RecipeConfig(
        ConfigNode(
            {
                "dataset": {
                    "_target_": ZephonTextDatasetConfig,
                    "path_or_dataset_id": source,
                    "zephon": {"packing": {"seq_len": 32}},
                },
                "dataloader": {"_target_": ZephonDataloaderConfig},
                "step_scheduler": {"max_steps": 2, "local_batch_size": 1},
            }
        )
    ).dataloader
    assert config.dataset_config.zephon.sources == (ZephonSource(name="source_0", path=expected),)


_SETTINGS = {"sources": [{"name": "text", "path": "unused"}]}
_CONTRACT = PackingCapabilities(packed_mask_type="block_causal", max_sequence_length=32)


def _recipe(*, dataset_type="ZephonTextDatasetConfig", dataset=None, loader=None, **recipe):
    return {
        "dataset": {
            "_target_": f"nemo_automodel.components.datasets.zephon.{dataset_type}",
            "zephon": deepcopy(_SETTINGS),
            **(dataset or {}),
        },
        "dataloader": loader or {},
        "step_scheduler": {"max_steps": 10, "local_batch_size": 1, "global_batch_size": 1},
        **recipe,
    }


def _schedule(*, batch_size=1, global_batch_size=None, max_steps=10):
    return DatasetBuildSchedule(
        local_batch_size=batch_size,
        global_batch_size=batch_size if global_batch_size is None else global_batch_size,
        max_steps=max_steps,
        val_check_interval=None,
    )


def _context(**kwargs):
    return TextStreamBuildContext(
        **{
            "seed": 42,
            "batch_size": 1,
            "dp_rank": 0,
            "dp_world_size": 1,
            "tokenizer": None,
            "packing_contract": _CONTRACT,
            "training_schedule": _schedule(),
            **kwargs,
        }
    )


def test_toml_inline_override_relative_paths_and_native_validation(tmp_path):
    (tmp_path / "recipe.toml").write_text('seq_len = 17\nrepeat = false\n[[sources]]\nname="text"\npath="text.jsonl"\n')
    raw = _recipe(
        dataset={"zephon": {"data_config": str(tmp_path / "recipe.toml"), "packing": {"seq_len": 23}}},
        step_scheduler={"max_steps": 2, "local_batch_size": 3},
    )
    config = RecipeConfig(ConfigNode(raw)).dataloader
    assert isinstance(config, DataloaderConfig)
    assert isinstance(config, ZephonDataloaderConfig)
    assert config.dataset_config.zephon.packing.seq_len == 23
    assert config.batch_size == 3
    assert config.dataset_config.zephon.sources[0].path == str(tmp_path / "text.jsonl")
    assert config.dataset_config.zephon.source.exhausted_policy is None
    with pytest.raises(ValueError, match="context parallelism"):
        config.dataset_config.validate_parallelism(cp_size=2)
    config.dataset_config.validate_parallelism(cp_size=1)
    validation_values = deepcopy(raw)
    validation_values["validation_dataset"] = validation_values["dataset"]
    validation_values["validation_dataloader"] = validation_values["dataloader"]
    validation = RecipeConfig(ConfigNode(validation_values)).validation_dataloaders["default"]
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match="Zephon validation is not supported"):
            validation.build(dp_rank=0, dp_world_size=1, is_training=False)
        create.assert_not_called()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"step_scheduler": {"max_steps": 0}}, "positive"),
        ({"step_scheduler": {"max_steps": True}}, "positive"),
    ],
)
def test_invalid_recipe_configs_fail_before_source_build(override, message):
    values = _recipe(**override)
    config = RecipeConfig(ConfigNode(values)).dataloader
    schedule = _schedule(max_steps=values["step_scheduler"]["max_steps"])
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match=message):
            config.build(dp_rank=0, dp_world_size=1, training_schedule=schedule, packing_contract=_CONTRACT)
        create.assert_not_called()


@pytest.mark.parametrize("recipe_name", ["tiny", "model_parallel"])
def test_shipped_yaml_resolves(recipe_name):
    import yaml

    path = Path(__file__).resolve().parents[4] / f"tests/integration/zephon/fixtures/{recipe_name}.yaml"
    config = RecipeConfig(ConfigNode(yaml.safe_load(path.read_text())))
    assert isinstance(config.dataloader, DataloaderConfig)
    assert isinstance(config.dataloader, ZephonDataloaderConfig)
    assert config.dataloader.dataset_config.zephon.packing.seq_len == 63
    assert config.dataloader.dataset_config.zephon.source.stop_after_passes is None
    assert config.dataloader.batch_size == config.get("step_scheduler.local_batch_size")


@pytest.mark.parametrize("family", ["qwen", "llama3_2"])
@pytest.mark.parametrize("prepacked", [False, True])
def test_pretraining_examples_select_matching_source_and_preparation(family: str, prepacked: bool) -> None:
    import yaml

    filename = "zephon_prepacked.yaml" if prepacked else "zephon.yaml"
    path = Path(__file__).resolve().parents[4] / "examples/llm_pretrain" / family / filename
    recipe = RecipeConfig(ConfigNode(yaml.safe_load(path.read_text())))
    config = recipe.dataloader
    assert isinstance(config, ZephonDataloaderConfig)
    dataset = config.dataset_config
    assert dataset.zephon.packing.seq_len == 1024
    assert config.batch_size == 8
    if prepacked:
        assert isinstance(dataset, ZephonPretokenizedTextDatasetConfig)
        assert dataset.zephon.packing.pack_algorithm is None
        assert dataset.zephon.sources[0].path.startswith("work/zephon-prepacked/")
    else:
        assert isinstance(dataset, ZephonTextDatasetConfig)
        assert dataset.tokenize_special_tokens == "eos"
        assert dataset.zephon.packing.pack_algorithm == "wrap"
        assert dataset.zephon.sources[0].path == "hf://roneneldan/TinyStories/train"


@pytest.mark.parametrize(
    "name",
    [
        "fetch_parallelism",
        "map_parallelism",
        "tokenize_parallelism",
        "pack_parallelism",
        "shuffle_parallelism",
    ],
)
def test_operator_parallelism_accepts_default_or_positive_workers(name):
    sources = (ZephonSource(name="text", path="unused"),)
    for value in (None, 1, 3):
        options = ZephonOptions(sources=sources, workers=ZephonWorkerOptions(**{name: value}))
        assert getattr(options.workers, name) == value
    for invalid in (0, -1, True, 1.5, "2"):
        with pytest.raises(ValueError, match=f"{name} must be a positive integer or None"):
            ZephonOptions(sources=sources, workers=ZephonWorkerOptions(**{name: invalid}))


def test_zephon_options_are_declarative_and_do_not_copy_recipe_seed():
    values = _recipe(seed=9)
    original = deepcopy(values)
    loader = RecipeConfig(ConfigNode(values)).dataloader
    assert loader.seed == 9
    assert "seed" not in asdict(loader.dataset_config.zephon)
    assert values == original
    with pytest.raises(ValueError, match="Unexpected Zephon option.*seed"):
        ZephonTextDatasetConfig(zephon={**_SETTINGS, "seed": 9})
    with pytest.raises(ValueError, match="Unexpected Zephon option.*unknown"):
        ZephonTextDatasetConfig(zephon={**_SETTINGS, "unknown": 1})


def test_raw_text_defaults_and_explicit_policy():
    config = ZephonTextDatasetConfig(zephon=_SETTINGS)
    assert config.tokenize_special_tokens == "bos_eos"
    assert config.zephon.workers.fetch_parallelism == 2
    assert config.zephon.workers.tokenize_parallelism == 3
    assert replace(config, tokenize_special_tokens="none").tokenize_special_tokens == "none"
    with pytest.raises(ValueError, match="resolved model tokenizer"):
        config.build(context=_context())


@pytest.mark.parametrize("policy", ["eos", "none"])
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_special_token_policy_is_explicit_on_the_dataset_without_warnings(policy):
    config = ZephonTextDatasetConfig(zephon=_SETTINGS, tokenize_special_tokens=policy)
    assert config.tokenize_special_tokens == policy
    with pytest.raises(ValueError, match="Unexpected Zephon option.*tokenize_special_tokens"):
        ZephonTextDatasetConfig(zephon={**_SETTINGS, "tokenize_special_tokens": policy})


def test_numeric_token_overrides_require_corresponding_explicit_insertion():
    assert (
        ZephonTextDatasetConfig(
            zephon=_SETTINGS, tokenize_special_tokens="bos_eos", bos_token_id=0, eos_token_id=7
        ).bos_token_id
        == 0
    )
    for invalid_id in (True, -1, 1.5, "7"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            ZephonTextDatasetConfig(zephon=_SETTINGS, tokenize_special_tokens="bos_eos", eos_token_id=invalid_id)
    for token_name, opposite_policy in (("bos_token_id", "eos"), ("eos_token_id", "bos")):
        for policy in (opposite_policy, "none", "tokenizer_default"):
            with pytest.raises(ValueError, match=f"{token_name} requires an explicit"):
                ZephonTextDatasetConfig(zephon=_SETTINGS, tokenize_special_tokens=policy, **{token_name: 7})


@pytest.mark.parametrize(
    "algorithm,expected", [("wrap", False), ("best_fit_wrap", False), ("first_fit", True), ("best_fit", True)]
)
def test_packing_defaults_are_resolved_once(algorithm, expected):
    options = ZephonOptions(
        sources=(ZephonSource(name="text", path="unused"),), packing=ZephonPackingOptions(pack_algorithm=algorithm)
    )
    assert options.packing.drop_oversized is expected
    assert options.packing.shuffle_after_pack is expected


@pytest.mark.parametrize(
    "target,kwargs",
    [
        ("ZephonTextDatasetConfig", {}),
        ("ZephonPretokenizedTextDatasetConfig", {}),
    ],
)
def test_unknown_dataset_fields_are_actionable(target, kwargs):
    with pytest.raises(TypeError, match=f"{target} got unexpected dataset config field.*typo"):
        make_dataset_config(
            f"nemo_automodel.components.datasets.zephon.{target}", {"zephon": _SETTINGS, **kwargs, "typo": 1}
        )


def test_toml_missing_file_source_path_and_source_options_have_context(tmp_path):
    settings = {"data_config": str(tmp_path / "missing.toml")}
    with pytest.raises(ValueError, match="data_config file does not exist.*missing.toml"):
        ZephonTextDatasetConfig(zephon=settings)
    path = tmp_path / "invalid.toml"
    path.write_text('[[sources]]\nname="broken"\n')
    settings["data_config"] = str(path)
    with pytest.raises(ValueError, match="source 'broken'.*nonempty path"):
        ZephonTextDatasetConfig(zephon=settings)
    path.write_text('[[sources]]\nname="broken"\npath="data"\ntyop=1\n')
    with pytest.raises(ValueError, match="source 'broken'.*tyop"):
        ZephonTextDatasetConfig(zephon=settings)


@pytest.mark.parametrize(
    "global_batch,local_batch,canonical,valid", [(16, 2, 4, True), (12, 2, 4, False), (6, 2, 3, True)]
)
def test_pinned_canonical_step_alignment_fails_before_source_build(global_batch, local_batch, canonical, valid):
    config = ZephonTextDatasetConfig(zephon={**_SETTINGS, "checkpoint": {"canonical_replicas": canonical}})
    context = _context(
        batch_size=local_batch, training_schedule=_schedule(batch_size=local_batch, global_batch_size=global_batch)
    )
    if valid:
        config.validate_context(context)
    else:
        with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
            with pytest.raises(ValueError, match="canonical_replicas.*lane windows"):
                config.build(context=context)
            create.assert_not_called()


def test_output_format_recipe_choice_and_delegated_metadata():
    values = _recipe(
        dataset={"output_format": "thd", "zephon": {**_SETTINGS, "packing": {"seq_len": 17, "pack_algorithm": None}}}
    )
    loader = RecipeConfig(ConfigNode(values)).dataloader
    assert loader.output_format == "thd"
    assert loader.is_packed
    assert loader.packed_sequence_length == 17
    assert loader.dataset_builds_on_all_ranks
    assert loader.packing is None
    with pytest.raises(ValueError, match="accepts seq_lens"):
        loader.build(
            dp_rank=0,
            dp_world_size=1,
            supports_seq_lens=False,
            training_schedule=_schedule(),
            packing_contract=_CONTRACT,
        )
    values["dataset"]["output_format"] = "neat"
    with pytest.raises(ValueError, match="output_format"):
        RecipeConfig(ConfigNode(values)).dataloader
    del values["dataset"]["output_format"]
    assert RecipeConfig(ConfigNode(values)).dataloader.output_format == "batched"


def test_grouped_options_merge_inline_overrides_per_setting(tmp_path):
    recipe = tmp_path / "data.toml"
    recipe.write_text(
        '[[sources]]\nname="text"\npath="corpus"\n'
        '[packing]\nseq_len=17\npack_algorithm="best_fit"\n'
        '[cache]\ncache_dir="cache"\ncache_limit_bytes=4096\n'
        '[checkpoint]\naggregate_dir="checkpoints"\n'
        "[source]\ntoken_estimation=false\nstop_after_passes=2\n"
    )
    settings = {"data_config": str(recipe), "packing": {"seq_len": 23}, "workers": {"tokenize_parallelism": 4}}
    original = deepcopy(settings)
    options = ZephonTextDatasetConfig(zephon=settings).zephon
    assert options.packing.seq_len == 23
    assert options.packing.pack_algorithm == "best_fit"
    assert options.packing.drop_oversized is True
    assert options.workers.tokenize_parallelism == 4
    assert options.source.token_estimation is False
    assert options.source.stop_after_passes == 2
    assert options.cache.cache_dir == str(tmp_path / "cache")
    assert options.cache.cache_limit_bytes == 4096
    assert options.checkpoint.aggregate_dir == str(tmp_path / "checkpoints")
    assert options.sources[0].path == str(tmp_path / "corpus")
    assert settings == original


def test_grouped_options_reject_unknown_misplaced_or_duplicate_fields():
    for settings, message in (
        ({"workers": {"seq_len": 8}}, "Unexpected Zephon workers"),
        ({"packing": {"unknown": 1}}, "Unexpected Zephon packing"),
        ({"packing": 4}, "packing must be a mapping"),
        ({"seq_len": 8, "packing": {"seq_len": 9}}, "packing.seq_len only once"),
    ):
        with pytest.raises(ValueError, match=message):
            ZephonTextDatasetConfig(zephon={**_SETTINGS, **settings})


def test_model_input_limit_resolves_length_without_mutating_recipe():
    config = ZephonTextDatasetConfig(zephon=_SETTINGS)
    assert config.resolve_sequence_length(packing_contract=_CONTRACT) == 32
    assert config.packed_sequence_length == 0 and config.zephon.packing.seq_len is None
    smaller = replace(config, zephon=replace(config.zephon, packing=ZephonPackingOptions(seq_len=16)))
    assert smaller.resolve_sequence_length(packing_contract=_CONTRACT) == 16
    assert smaller.packed_sequence_length == 16
    unknown = replace(_CONTRACT, max_sequence_length=None)
    assert smaller.resolve_sequence_length(packing_contract=unknown) == 16
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match="no known input-token limit"):
            config.build(context=_context(tokenizer=object(), packing_contract=unknown))
        for xdoc in (False, True):
            oversized = replace(
                smaller,
                zephon=replace(
                    smaller.zephon, training=replace(smaller.zephon.training, allow_cross_document_attention=xdoc)
                ),
            )
            with pytest.raises(ValueError, match="exceeds the model input-token limit"):
                oversized.build(
                    context=_context(tokenizer=object(), packing_contract=replace(_CONTRACT, max_sequence_length=8))
                )
        create.assert_not_called()


def test_disabled_packing_defaults_and_incompatible_options():
    options = ZephonPackingOptions(pack_algorithm=None)
    assert options.drop_oversized is False and options.shuffle_after_pack is False
    for invalid in ({"pack_num_bins": 2}, {"max_num_documents": 2}, {"drop_oversized": True}):
        with pytest.raises(ValueError, match="require online packing"):
            ZephonPackingOptions(pack_algorithm=None, **invalid)
    for length in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="seq_len must be positive"):
            ZephonPackingOptions(seq_len=length)


def test_pretokenized_recipe_selects_optional_packing():
    for algorithm in ("wrap", None):
        loader = RecipeConfig(
            ConfigNode(
                _recipe(
                    dataset_type="ZephonPretokenizedTextDatasetConfig",
                    dataset={
                        "tokens_field": "tokens",
                        "zephon": {**_SETTINGS, "packing": {"seq_len": 7, "pack_algorithm": algorithm}},
                    },
                )
            )
        ).dataloader
        assert isinstance(loader.dataset_config, ZephonPretokenizedTextDatasetConfig)
        assert loader.dataset_config.tokens_field == "tokens"
        assert loader.dataset_config.zephon.packing.pack_algorithm == algorithm


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"is_training": False}, "Zephon validation"),
        ({"cp_size": 2, "packing_cp_size": 1}, "context parallelism"),
        ({"training_schedule": None}, "runtime training_schedule"),
        ({"training_schedule": _schedule(batch_size=2)}, "optimizer accounting"),
        ({"domain_mixture": {}}, "loss reweighting"),
    ],
)
def test_runtime_validation_precedes_source_construction(kwargs, message):
    loader = ZephonDataloaderConfig(dataset_config=ZephonTextDatasetConfig(zephon=_SETTINGS))
    runtime = {
        "dp_rank": 0,
        "dp_world_size": 1,
        "training_schedule": _schedule(),
        "packing_contract": _CONTRACT,
        **kwargs,
    }
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match=message):
            loader.build(**runtime)
        create.assert_not_called()


@pytest.mark.parametrize(
    "settings",
    [
        {"packing": NeatPackingConfig(packed_sequence_size=16)},
        {"collate_fn": list},
        {"num_workers": 1},
        {"shuffle": False},
        {"pin_memory": True},
        {"prefetch_factor": 2},
        {"persistent_workers": True},
        {"group_by_length": True},
        {"batch_sampler_config": object()},
        {"shuffle_buffer_size": 42},
        {"drop_last": True},
    ],
)
def test_stream_rejects_ignored_native_settings(settings):
    loader = ZephonDataloaderConfig(dataset_config=ZephonTextDatasetConfig(zephon=_SETTINGS), **settings)
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match="stream dataset owns packing"):
            loader.build(dp_rank=0, dp_world_size=1, training_schedule=_schedule(), packing_contract=_CONTRACT)
        create.assert_not_called()


def test_common_build_scopes_rng_but_preserves_explicit_shared_seed(monkeypatch):
    from nemo_automodel.components.datasets import loader as loader_module

    dataset = ZephonTextDatasetConfig(zephon=_SETTINGS)
    loader = ZephonDataloaderConfig(dataset_config=dataset, seed=91, batch_size=2)
    events = []
    contexts = []
    output = iter(())

    @contextmanager
    def rng_scope(*, seed, ranked):
        events.append(("enter", seed, ranked))
        yield
        events.append("exit")

    def build(self, *, context):
        assert events[-1] == ("enter", 91, True)
        contexts.append(context)
        return output

    @contextmanager
    def rank_first():
        raise AssertionError("All-rank stream construction must not enter rank-first ordering")
        yield

    monkeypatch.setattr(loader_module, "ScopedRNG", rng_scope)
    monkeypatch.setattr(ZephonTextDatasetConfig, "build", build)
    for _ in range(2):
        assert (
            loader.build(
                dp_rank=0,
                dp_world_size=1,
                tokenizer=object(),
                dataset_build_context=rank_first(),
                training_schedule=_schedule(batch_size=2),
                packing_contract=_CONTRACT,
            )
            is output
        )
    assert [context.seed for context in contexts] == [91, 91]
    assert [context.batch_size for context in contexts] == [2, 2]
    assert events == [("enter", 91, True), "exit", ("enter", 91, True), "exit"]
    assert loader.seed == 91 and dataset.zephon.packing.seq_len is None


def test_common_build_restores_rng_after_stream_construction(monkeypatch):
    dataset = ZephonTextDatasetConfig(zephon=_SETTINGS)
    loader = ZephonDataloaderConfig(dataset_config=dataset, seed=91)
    before = torch.get_rng_state().clone()
    observed = []

    def build(self, *, context):
        observed.append(torch.rand(3))
        return iter(())

    monkeypatch.setattr(ZephonTextDatasetConfig, "build", build)
    for _ in range(2):
        loader.build(dp_rank=0, dp_world_size=1)
        assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(observed[0], observed[1])


@pytest.mark.parametrize("field", ["path_or_dataset_id", "paths"])
def test_native_source_path_normalizes_once_without_mutating_yaml(tmp_path, field):
    root = tmp_path / "corpus.jsonl"
    root.mkdir()
    (root / "train.jsonl").write_text('{"text":"hello"}\n')
    for supplied in (str(root), [str(root)]):
        values = _recipe(dataset={"zephon": {"packing": {"seq_len": 17}}, field: supplied})
        original = deepcopy(values)
        config = RecipeConfig(ConfigNode(values)).dataloader.dataset_config
        assert config.zephon.sources == (ZephonSource(name="source_0", path=str(root)),)
        assert config.path_or_dataset_id is None and config.paths is None
        assert config.split is None and config.name is None
        assert (
            replace(config, zephon=replace(config.zephon, packing=ZephonPackingOptions(seq_len=9))).zephon.sources
            == config.zephon.sources
        )
        assert values == original
    assert ZephonTextDatasetConfig(paths=root).zephon.sources[0].path == str(root)


def test_native_hf_source_preserves_subset_and_named_split():
    config = ZephonTextDatasetConfig(path_or_dataset_id="org/corpus", name="en", split="validation")
    assert config.zephon.sources == (ZephonSource(name="source_0", path="hf://org/corpus/en/validation"),)
    assert ZephonTextDatasetConfig(path_or_dataset_id="org/corpus").zephon.sources[0].path == "hf://org/corpus/train"
    assert replace(config).zephon.sources == config.zephon.sources
    # Native `paths` and one-element file lists do not treat org/name as an HF ID.
    assert ZephonTextDatasetConfig(paths="org/corpus").zephon.sources[0].path == "org/corpus"
    assert ZephonTextDatasetConfig(path_or_dataset_id=["org/corpus"]).zephon.sources[0].path == "org/corpus"


def test_native_existing_relative_directory_is_not_reinterpreted_as_hf(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "org" / "corpus"
    root.mkdir(parents=True)
    assert ZephonTextDatasetConfig(path_or_dataset_id="org/corpus").zephon.sources[0].path == "org/corpus"


@pytest.mark.parametrize("uri", ["s3://bucket/corpus", "hf://org/corpus@revision/en/train"])
def test_native_source_alias_preserves_explicit_zephon_uri(uri):
    assert ZephonTextDatasetConfig(paths=uri).zephon.sources[0].path == uri


@pytest.mark.parametrize("selection", ["train[:10]", "train+validation", "900,50,50", "", 1])
def test_native_hf_split_rejects_unimplemented_selection(selection):
    with pytest.raises(ValueError, match="simple named HF split"):
        ZephonTextDatasetConfig(path_or_dataset_id="org/corpus", split=selection)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"path_or_dataset_id": "/data/a", "paths": "/data/a"}, "only one source declaration"),
        ({"paths": []}, "must not be empty"),
        ({"paths": ["/data/a", "/data/b"]}, "equal-weight Zephon mixtures"),
        ({"paths": ["0.2", "/data/a", "0.8", "/data/b"]}, "equal-weight Zephon mixtures"),
        ({"paths": {"train": ["/data/a"], "validation": ["/data/b"]}}, "split-specific blend mappings"),
        ({"paths": "/data/blend.json"}, "blend JSON file"),
        ({"paths": "/data/train.jsonl"}, "individual file"),
        ({"paths": "/data/train.parquet"}, "individual file"),
        ({"paths": "/data/train*.jsonl"}, "file glob"),
        ({"paths": "/data/train.bin"}, "Megatron .bin/.idx"),
        ({"paths": "/data/train.idx"}, "Megatron .bin/.idx"),
        ({"paths": 7}, "one nonempty path"),
        ({"path_or_dataset_id": ""}, "one nonempty path"),
        ({"paths": "/data/a", "split": "validation"}, "bare HF dataset ID"),
        ({"paths": "/data/a", "name": "subset"}, "bare HF dataset ID"),
        ({"split": "train"}, "require path_or_dataset_id"),
    ],
)
def test_native_source_forms_fail_clearly_before_io(kwargs, message):
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_source_pipeline") as create:
        with pytest.raises(ValueError, match=message):
            ZephonTextDatasetConfig(**kwargs)
        create.assert_not_called()


def test_native_source_alias_rejects_canonical_sources_including_toml(tmp_path):
    for field in ("path_or_dataset_id", "paths"):
        with pytest.raises(ValueError, match="only one source declaration"):
            ZephonTextDatasetConfig(zephon=_SETTINGS, **{field: "/data/text"})
        toml = tmp_path / "data.toml"
        toml.write_text('[[sources]]\nname="text"\npath="corpus"\n')
        with pytest.raises(ValueError, match="including TOML"):
            ZephonTextDatasetConfig(zephon={"data_config": str(toml)}, **{field: "/data/text"})
    with pytest.raises(ValueError, match="Zephon needs sources"):
        ZephonTextDatasetConfig()


def test_native_megatron_prefix_and_unknown_source_file_are_not_reinterpreted(tmp_path):
    prefix = tmp_path / "tokens"
    prefix.with_suffix(".idx").touch()
    with pytest.raises(ValueError, match="Megatron .bin/.idx"):
        ZephonTextDatasetConfig(paths=prefix)
    selected = tmp_path / "shard.data"
    selected.touch()
    with pytest.raises(ValueError, match="individual file"):
        ZephonTextDatasetConfig(path_or_dataset_id=str(selected))


@pytest.mark.parametrize("field", ["seq_length", "padding", "truncation", "limit_dataset_samples"])
def test_unsupported_native_formatting_fields_are_rejected_not_ignored(field):
    with pytest.raises(TypeError, match=f"unexpected dataset config field.*{field}"):
        make_dataset_config(
            "nemo_automodel.components.datasets.zephon.ZephonTextDatasetConfig",
            {"paths": "/data/text", field: 10},
        )


def test_pretokenized_source_alias_keeps_encoding_settings():
    config = ZephonPretokenizedTextDatasetConfig(path_or_dataset_id="org/corpus", tokens_field="tokens")
    assert config.zephon.sources[0].path == "hf://org/corpus/train"
    assert config.tokens_field == "tokens"
    assert replace(config).zephon.sources == config.zephon.sources


def test_native_directory_is_not_confused_with_binary_prefix(tmp_path):
    for name in ("corpus", "corpus.bin"):
        root = tmp_path / name
        root.mkdir()
        Path(str(root) + ".idx").touch()
        config = ZephonTextDatasetConfig(paths=root)
        assert config.zephon.sources[0].path == str(root)


@pytest.mark.parametrize("packing", [{}, {"packed_sequence_size": 0}, {"packed_sequence_size": 16}])
def test_native_packing_selection_is_never_silently_ignored(packing):
    with pytest.raises(ValueError, match="Stream datasets own packing"):
        RecipeConfig(ConfigNode(_recipe(packed_sequence=packing))).dataloader


@pytest.mark.parametrize("subset", ["en/subset", "", 5])
def test_native_hf_subset_requires_a_simple_name(subset):
    with pytest.raises(ValueError, match="simple named HF name"):
        ZephonTextDatasetConfig(path_or_dataset_id="org/corpus", name=subset)


def test_native_source_survives_model_length_resolution_at_runtime(tmp_path, tokenizer):
    root = tmp_path / "corpus"
    root.mkdir()
    config = ZephonDataloaderConfig(dataset_config=ZephonTextDatasetConfig(paths=root))
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        loader = config.build(
            dp_rank=0,
            dp_world_size=1,
            tokenizer=tokenizer,
            packing_contract=_CONTRACT,
            training_schedule=_schedule(),
        )
        resolved = create.call_args.args[0]
        assert resolved.zephon.sources == config.dataset_config.zephon.sources
        assert resolved.zephon.packing.seq_len == _CONTRACT.max_sequence_length
        assert config.dataset_config.zephon.packing.seq_len is None
        loader.close()


@pytest.mark.parametrize("dataset_type", ["ZephonTextDatasetConfig", "ZephonPretokenizedTextDatasetConfig"])
def test_loader_subclass_delegates_layout_and_length_without_duplicate_settings(dataset_type):
    values = _recipe(
        dataset_type=dataset_type,
        dataset={"output_format": "thd", "zephon": {**_SETTINGS, "packing": {"seq_len": 19, "pack_algorithm": None}}},
    )
    original = deepcopy(values)
    loader = RecipeConfig(ConfigNode(values)).dataloader
    assert isinstance(loader, ZephonDataloaderConfig)
    assert loader.output_format == "thd"
    assert loader.is_packed
    assert loader.packed_sequence_length == 19
    assert loader.packing is None
    assert values == original


def test_text_stream_requires_its_loader_config_and_rejects_native_datasets():
    from nemo_automodel.components.datasets.llm.mock import MockUnpackedDatasetConfig

    with pytest.raises(TypeError, match="compatible DataloaderConfig subclass"):
        DataloaderConfig(dataset_config=ZephonTextDatasetConfig(zephon=_SETTINGS))
    with pytest.raises(TypeError, match="Zephon text dataset config"):
        ZephonDataloaderConfig(dataset_config=MockUnpackedDatasetConfig())
    values = _recipe()
    values["dataloader"]["_target_"] = "torch.utils.data.DataLoader"
    with pytest.raises(TypeError, match="compatible DataloaderConfig subclass"):
        RecipeConfig(ConfigNode(values)).dataloader


@pytest.mark.parametrize(
    "target", [None, ZephonDataloaderConfig, "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig"]
)
def test_text_loader_target_is_optional_and_explicit_targets_still_work(target):
    values = _recipe()
    if target is None:
        values.pop("dataloader")
    else:
        values["dataloader"]["_target_"] = target
    original = deepcopy(values)
    loader = RecipeConfig(ConfigNode(values)).dataloader
    assert isinstance(loader, ZephonDataloaderConfig)
    assert isinstance(loader.dataset_config, ZephonTextDatasetConfig)
    assert values == original


@pytest.mark.parametrize("algorithm", ["first_fit", "best_fit"])
def test_whole_document_packing_owns_its_default_bin_count(algorithm):
    options = ZephonPackingOptions(pack_algorithm=algorithm)
    assert options.pack_num_bins == 64
    assert options.drop_oversized is True and options.shuffle_after_pack is True
    explicit = ZephonPackingOptions(
        pack_algorithm=algorithm, pack_num_bins=8, drop_oversized=False, shuffle_after_pack=False
    )
    assert explicit.pack_num_bins == 8
    assert explicit.drop_oversized is False and explicit.shuffle_after_pack is False


@pytest.mark.parametrize("algorithm", [None, "wrap", "best_fit_wrap"])
def test_non_bin_packing_has_no_bin_count(algorithm):
    assert ZephonPackingOptions(pack_algorithm=algorithm).pack_num_bins is None


@pytest.mark.parametrize(
    "name,allows_zero",
    [
        ("prefetch_batches", True),
        ("default_stage_prefetch", True),
        ("op_queue_capacity", False),
        ("mtp_buffer", False),
        ("mtp_prefetch", True),
    ],
)
def test_runtime_buffer_controls_keep_zephon_defaults_and_validate_capacities(name: str, allows_zero: bool) -> None:
    assert getattr(ZephonRuntimeOptions(), name) is None
    for value in (None, 1, 8):
        assert getattr(ZephonRuntimeOptions(**{name: value}), name) == value
    if allows_zero:
        assert getattr(ZephonRuntimeOptions(**{name: 0}), name) == 0
    invalid = [-1, True, 1.5, "8"] + ([] if allows_zero else [0])
    for value in invalid:
        with pytest.raises(ValueError, match=f"{name} must be a .* integer or None"):
            ZephonRuntimeOptions(**{name: value})


def test_runtime_buffers_follow_toml_and_inline_override_precedence(tmp_path: Path) -> None:
    path = tmp_path / "runtime.toml"
    path.write_text(
        '[[sources]]\nname="text"\npath="unused"\n'
        "[runtime]\nprefetch_batches=40\ndefault_stage_prefetch=12\nop_queue_capacity=20\nmtp_buffer=10\nmtp_prefetch=2\n"
    )
    dataset = ZephonTextDatasetConfig(
        zephon={"data_config": str(path), "runtime": {"prefetch_batches": 32, "mtp_prefetch": 0}}
    )
    assert dataset.zephon.runtime.prefetch_batches == 32
    assert dataset.zephon.runtime.default_stage_prefetch == 12
    assert dataset.zephon.runtime.op_queue_capacity == 20
    assert dataset.zephon.runtime.mtp_buffer == 10
    assert dataset.zephon.runtime.mtp_prefetch == 0


@pytest.mark.parametrize(
    "source,expected",
    [
        ({}, None),
        ({"prefetch": None}, None),
        ({"prefetch": {}}, ZephonPrefetchOptions()),
        ({"prefetch": {"buffer_size": 8, "parallelism": 3}}, ZephonPrefetchOptions(buffer_size=8, parallelism=3)),
        ({"prefetch": {"buffer_size": None, "parallelism": None}}, ZephonPrefetchOptions()),
    ],
)
def test_source_prefetch_is_optional_and_typed_at_recipe_boundary(
    source: dict[str, object], expected: ZephonPrefetchOptions | None
) -> None:
    values = _recipe(dataset={"zephon": {**_SETTINGS, "source": source}})
    original = deepcopy(values)
    loader = RecipeConfig(ConfigNode(values)).dataloader
    assert loader.dataset_config.zephon.source.prefetch == expected
    assert values == original


@pytest.mark.parametrize("name", ["buffer_size", "parallelism"])
def test_source_prefetch_accepts_operator_defaults_or_positive_values(name: str) -> None:
    for value in (None, 1, 8):
        assert getattr(ZephonPrefetchOptions(**{name: value}), name) == value
    for invalid in (0, -1, True, 1.5, "8"):
        with pytest.raises(ValueError, match=f"source.prefetch.{name} must be a positive integer or None"):
            ZephonPrefetchOptions(**{name: invalid})


@pytest.mark.parametrize("prefetch", [True, False, 8, [], {"unknown": 2}])
def test_source_prefetch_rejects_invalid_recipe_settings(prefetch: object) -> None:
    with pytest.raises(ValueError, match="Zephon source.prefetch"):
        ZephonTextDatasetConfig(zephon={**_SETTINGS, "source": {"prefetch": prefetch}})


@pytest.mark.parametrize(
    "settings",
    [{"runtime": {"prefetch_buffer_size": 8}}, {"workers": {"prefetch_parallelism": 3}}],
)
def test_source_prefetch_is_not_a_runtime_or_worker_setting(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="Unexpected Zephon"):
        ZephonTextDatasetConfig(zephon={**_SETTINGS, **settings})


def test_source_prefetch_toml_overrides_preserve_fields_and_allow_disabling(tmp_path: Path) -> None:
    path = tmp_path / "prefetch.toml"
    path.write_text('[[sources]]\nname="text"\npath="unused"\n[source.prefetch]\nbuffer_size=8\nparallelism=3\n')
    for overrides, expected in (
        ({}, ZephonPrefetchOptions(buffer_size=8, parallelism=3)),
        ({"prefetch": {}}, ZephonPrefetchOptions(buffer_size=8, parallelism=3)),
        ({"prefetch": {"parallelism": 5}}, ZephonPrefetchOptions(buffer_size=8, parallelism=5)),
        ({"prefetch": {"buffer_size": None}}, ZephonPrefetchOptions(parallelism=3)),
        ({"prefetch": None}, None),
    ):
        values = {"data_config": str(path), "source": overrides}
        original = deepcopy(values)
        dataset = ZephonTextDatasetConfig(zephon=values)
        assert dataset.zephon.source.prefetch == expected
        assert values == original


def test_cache_overrides_preserve_automatic_defaults_and_explicit_disable() -> None:
    from zephon.io import StoreOptions

    assert ZephonCacheOptions().build() == StoreOptions.from_any(None)
    options = ZephonCacheOptions(
        cache_dir="/cache",
        cache_limit_bytes="80gb",
        parquet_rg_cache=ZephonParquetRGCacheOptions(enabled=False, limit_bytes="8gb", min_free_bytes=0),
    ).build()
    assert options.cache.enabled and options.cache.limit_bytes == 80 * 1024**3
    assert options.parquet.rg_cache.enabled is False
    assert options.parquet.rg_cache.limit_bytes == 8 * 1024**3
    assert options.parquet.rg_cache.min_free_bytes == 0
    decoded_only = ZephonCacheOptions(parquet_rg_cache=ZephonParquetRGCacheOptions(root="/decoded")).build()
    assert not decoded_only.cache.enabled
    assert str(decoded_only.parquet.rg_cache.root) == "/decoded"


def test_decoded_cache_rejects_invalid_budgets_through_zephon() -> None:
    with pytest.raises(ValueError, match="limit_bytes must be positive"):
        ZephonParquetRGCacheOptions(limit_bytes=0).build()
    with pytest.raises(ValueError, match="min_free_bytes cannot be negative"):
        ZephonParquetRGCacheOptions(min_free_bytes=-1).build()


def test_nested_decoded_cache_toml_overrides_preserve_fields_and_resolve_paths(tmp_path: Path) -> None:
    path = tmp_path / "data.toml"
    path.write_text(
        '[cache.parquet_rg_cache]\nroot="decoded"\nlimit_bytes="2gb"\n[[sources]]\nname="text"\npath="corpus"\n'
    )
    values = {"data_config": str(path), "cache": {"parquet_rg_cache": {"enabled": False, "min_free_bytes": 0}}}
    original = deepcopy(values)
    options = ZephonTextDatasetConfig(zephon=values).zephon.cache.parquet_rg_cache
    assert options == ZephonParquetRGCacheOptions(
        root=str(tmp_path / "decoded"), limit_bytes="2gb", enabled=False, min_free_bytes=0
    )
    assert values == original
    disabled = ZephonTextDatasetConfig(zephon={"data_config": str(path), "cache": {"parquet_rg_cache": None}})
    assert disabled.zephon.cache.parquet_rg_cache is None
    with pytest.raises(ValueError, match="Unexpected Zephon cache.parquet_rg_cache"):
        ZephonTextDatasetConfig(zephon={"cache": {"parquet_rg_cache": {"limit": 17}}})

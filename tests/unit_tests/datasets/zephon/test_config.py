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
    ZephonInstructionDatasetConfig,
    ZephonOptions,
    ZephonPackingOptions,
    ZephonPretokenizedTextDatasetConfig,
    ZephonSource,
    ZephonTextDatasetConfig,
    ZephonWorkerOptions,
)
from nemo_automodel.components.models.common.packing import PackingCapabilities
from nemo_automodel.recipes._typed_config import RecipeConfig

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
    assert config.dataset_config.zephon.packing.seq_len == 23
    assert config.batch_size == 3
    assert config.dataset_config.zephon.sources[0].path == str(tmp_path / "text.jsonl")
    assert config.dataset_config.zephon.source.exhausted_policy is None
    with pytest.raises(ValueError, match="context parallelism"):
        config.dataset_config.validate_parallelism(cp_size=2)
    config.dataset_config.validate_parallelism(cp_size=1)
    validation_values = deepcopy(raw)
    validation_values["validation_dataset"] = validation_values["dataset"]
    validation = RecipeConfig(ConfigNode(validation_values)).validation_dataloaders["default"]
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match="Zephon validation is not supported"):
            validation.build(dp_rank=0, dp_world_size=1, is_training=False)
        create.assert_not_called()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"packed_sequence": {"packed_sequence_size": 16}}, "stream dataset owns packing"),
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

    path = Path(__file__).resolve().parents[4] / f"examples/llm_pretrain/zephon/{recipe_name}.yaml"
    config = RecipeConfig(ConfigNode(yaml.safe_load(path.read_text())))
    assert isinstance(config.dataloader, DataloaderConfig)
    assert config.dataloader.dataset_config.zephon.packing.seq_len == 63
    assert config.dataloader.dataset_config.zephon.source.stop_after_passes == 1
    assert config.dataloader.batch_size == config.get("step_scheduler.local_batch_size")


def test_explicit_instruction_config_selects_sft_without_native_translation():
    from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
        ColumnMappedTextInstructionDatasetConfig,
    )

    values = _recipe(
        dataset_type="ZephonInstructionDatasetConfig",
        dataset={
            "column_mapping": {"question": "prompt", "answer": "response"},
            "use_hf_chat_template": True,
            "zephon": {**_SETTINGS, "workers": {"map_parallelism": 3}},
        },
    )
    resolved = RecipeConfig(ConfigNode(values)).dataloader
    assert isinstance(resolved.dataset_config, ZephonInstructionDatasetConfig)
    assert resolved.dataset_config.use_hf_chat_template
    assert resolved.dataset_config.zephon.workers.map_parallelism == 3
    values["dataset"] = {
        "_target_": "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDatasetConfig",
        "path_or_dataset_id": "unused",
        "column_mapping": {"question": "prompt", "answer": "response"},
    }
    assert isinstance(
        RecipeConfig(ConfigNode(values)).dataloader.dataset_config, ColumnMappedTextInstructionDatasetConfig
    )


@pytest.mark.parametrize(
    "name",
    [
        "fetch_parallelism",
        "prefetch_parallelism",
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


def test_instruction_special_tokens_validate_template_ownership():
    for policy in ("assistant_eos", "tokenizer_default"):
        with pytest.raises(ValueError, match="require use_hf_chat_template=false"):
            ZephonInstructionDatasetConfig(
                zephon=_SETTINGS,
                column_mapping={"question": "q", "answer": "a"},
                use_hf_chat_template=True,
                special_tokens=policy,
            )
    assert (
        ZephonInstructionDatasetConfig(
            zephon=_SETTINGS,
            column_mapping={"question": "q", "answer": "a"},
            use_hf_chat_template=True,
            special_tokens="eos",
        ).special_tokens
        == "eos"
    )


@pytest.mark.parametrize("instruction", [False, True])
def test_numeric_token_overrides_require_corresponding_explicit_insertion(instruction):
    config_type = ZephonInstructionDatasetConfig if instruction else ZephonTextDatasetConfig
    fields = {"zephon": _SETTINGS, **({"column_mapping": {"question": "q", "answer": "a"}} if instruction else {})}
    policy_field = "special_tokens" if instruction else "tokenize_special_tokens"
    assert config_type(**fields, **{policy_field: "bos_eos", "bos_token_id": 0, "eos_token_id": 7}).bos_token_id == 0
    for invalid_id in (True, -1, 1.5, "7"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            config_type(**fields, **{policy_field: "bos_eos", "eos_token_id": invalid_id})
    for token_name, opposite_policy in (("bos_token_id", "eos"), ("eos_token_id", "bos")):
        policies = [opposite_policy, "none", "tokenizer_default"] + (["assistant_eos", None] if instruction else [])
        for policy in policies:
            with pytest.raises(ValueError, match=f"{token_name} requires an explicit"):
                config_type(**fields, **{policy_field: policy, token_name: 7})


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
        ("ZephonInstructionDatasetConfig", {"column_mapping": {"question": "q", "answer": "a"}}),
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
    loader = DataloaderConfig(dataset_config=ZephonTextDatasetConfig(zephon=_SETTINGS))
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
    loader = DataloaderConfig(dataset_config=ZephonTextDatasetConfig(zephon=_SETTINGS), **settings)
    with patch("nemo_automodel.components.datasets.zephon.pipeline.create_text_pipeline") as create:
        with pytest.raises(ValueError, match="stream dataset owns packing"):
            loader.build(dp_rank=0, dp_world_size=1, training_schedule=_schedule(), packing_contract=_CONTRACT)
        create.assert_not_called()


def test_common_build_scopes_rng_but_preserves_explicit_shared_seed(monkeypatch):
    from nemo_automodel.components.datasets import loader as loader_module

    dataset = ZephonTextDatasetConfig(zephon=_SETTINGS)
    loader = DataloaderConfig(dataset_config=dataset, seed=91, batch_size=2)
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
    loader = DataloaderConfig(dataset_config=dataset, seed=91)
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

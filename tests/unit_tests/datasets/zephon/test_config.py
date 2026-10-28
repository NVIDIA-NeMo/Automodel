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

"""Declarative recipe configuration and validation without pipeline execution."""

from pathlib import Path

import pytest

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.zephon.config import (
    ZephonDataloaderConfig,
    ZephonInstructionSourceConfig,
    ZephonOptions,
    ZephonPackingOptions,
    ZephonSource,
    ZephonTextSourceConfig,
    ZephonWorkerOptions,
)
from nemo_automodel.recipes._typed_config import RecipeConfig


def test_toml_inline_override_relative_paths_and_native_validation(tmp_path):
    (tmp_path / "recipe.toml").write_text('seq_len = 17\nrepeat = false\n[[sources]]\nname="text"\npath="text.jsonl"\n')
    raw = ConfigNode(
        {
            "dataset": {"_target_": "nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig"},
            "dataloader": {
                "_target_": "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig",
                "zephon": {"data_config": str(tmp_path / "recipe.toml"), "seq_len": 23},
            },
            "step_scheduler": {"max_steps": 2, "local_batch_size": 3},
        }
    )
    config = RecipeConfig(raw).dataloader
    assert config.zephon.packing.seq_len == 23
    assert config.batch_size == 3
    assert config.zephon.sources[0].path == str(tmp_path / "text.jsonl")
    assert config.zephon.source.exhausted_policy is None
    with pytest.raises(ValueError, match="context parallelism"):
        config.validate_parallelism(cp_size=2)
    config.validate_parallelism(cp_size=1)
    validation_values = raw.to_dict()
    validation_values["validation_dataset"] = validation_values["dataset"]
    validation_values["validation_dataloader"] = validation_values["dataloader"]
    with pytest.raises(ValueError, match="Zephon validation is not supported"):
        _ = RecipeConfig(ConfigNode(validation_values)).validation_dataloaders


@pytest.mark.parametrize(
    "override, message",
    [
        ({"packed_sequence": {"packed_sequence_size": 16}}, "Native packing"),
        ({"step_scheduler": {"max_steps": 0}}, "positive"),
        ({"step_scheduler": {"max_steps": True}}, "positive"),
        ({"domain_mixture": {"domains": []}}, "loss reweighting"),
    ],
)
def test_invalid_recipe_configs_fail_before_build(tmp_path, override, message):
    values = {
        "dataset": {"_target_": "nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig"},
        "dataloader": {
            "_target_": "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig",
            "zephon": {"sources": [{"name": "text", "path": str(tmp_path)}]},
        },
        "step_scheduler": {"max_steps": 10},
        **override,
    }
    with pytest.raises(ValueError, match=message):
        RecipeConfig(ConfigNode(values)).dataloader


@pytest.mark.parametrize("recipe_name", ["tiny", "model_parallel"])
def test_shipped_yaml_resolves(recipe_name):
    import yaml

    path = Path(__file__).resolve().parents[4] / f"examples/llm_pretrain/zephon/{recipe_name}.yaml"
    config = RecipeConfig(ConfigNode(yaml.safe_load(path.read_text())))
    assert config.dataloader.zephon.packing.seq_len == 63
    assert config.dataloader.zephon.source.stop_after_passes == 1
    assert config.dataloader.batch_size == config.get("step_scheduler.local_batch_size")


def test_existing_instruction_config_selects_sft_and_rejects_ambiguous_transport(tmp_path):
    raw = {
        "dataset": {
            "_target_": "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDatasetConfig",
            "column_mapping": {"question": "prompt", "answer": "response"},
            "use_hf_chat_template": True,
        },
        "dataloader": {
            "_target_": "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig",
            "zephon": {"sources": [{"name": "local", "path": str(tmp_path)}], "map_parallelism": 3},
        },
        "step_scheduler": {"max_steps": 2, "local_batch_size": 1},
    }
    resolved = RecipeConfig(ConfigNode(raw)).dataloader
    assert isinstance(resolved.dataset_config, ZephonInstructionSourceConfig)
    assert resolved.dataset_config.use_hf_chat_template
    assert resolved.zephon.workers.map_parallelism == 3
    raw["dataset"]["path_or_dataset_id"] = str(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        RecipeConfig(ConfigNode(raw)).dataloader


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


def test_shared_zephon_option_resolution_keeps_explicit_seed_and_rejects_unknowns():
    values = {"sources": [{"name": "text", "path": "unused"}], "seed": 9}

    def resolve(settings, **loader_overrides):
        return ZephonDataloaderConfig.from_recipe(
            ZephonTextSourceConfig,
            {},
            {"zephon": settings, **loader_overrides},
            seed=42,
            batch_size=1,
            packing={},
        )

    assert resolve(values).zephon.seed == 9
    assert isinstance(values["sources"], list)
    assert resolve({"sources": values["sources"]}).zephon.seed == 42
    with pytest.raises(ValueError, match="Unexpected Zephon option"):
        resolve({**values, "unknown": 1})
    with pytest.raises(ValueError, match="Unexpected Zephon loader fields"):
        resolve(values, num_workers=2)


def test_raw_text_defaults_and_policy_precedence():
    sources = [{"name": "text", "path": "unused"}]
    config = ZephonDataloaderConfig.from_recipe(
        ZephonTextSourceConfig,
        {},
        {"zephon": {"sources": sources}},
        seed=42,
        batch_size=1,
        packing={},
    )
    assert config.dataset_config.tokenize_special_tokens == "bos_eos"
    assert config.zephon.workers.fetch_parallelism == 2
    assert config.zephon.workers.tokenize_parallelism == 3
    config = ZephonDataloaderConfig.from_recipe(
        ZephonTextSourceConfig,
        {"tokenize_special_tokens": "none"},
        {"zephon": {"sources": sources, "tokenize_special_tokens": "eos"}},
        seed=42,
        batch_size=1,
        packing={},
    )
    assert config.dataset_config.tokenize_special_tokens == "none"
    with pytest.raises(ValueError, match="resolved model tokenizer"):
        config.build(dp_rank=0, dp_world_size=1)


@pytest.mark.parametrize("origin", ["toml", "inline"])
@pytest.mark.filterwarnings("error::DeprecationWarning")
def test_reference_special_token_policy_maps_to_dataset_without_warnings(tmp_path, origin):
    settings = {"sources": [{"name": "text", "path": "unused"}], "tokenize_special_tokens": "eos"}
    if origin == "toml":
        path = tmp_path / "recipe.toml"
        path.write_text('tokenize_special_tokens = "eos"\n[[sources]]\nname = "text"\npath = "unused"\n')
        settings = {"data_config": str(path)}

    def resolve(dataset_values):
        return ZephonDataloaderConfig.from_recipe(
            ZephonTextSourceConfig,
            dataset_values,
            {"zephon": settings},
            seed=42,
            batch_size=1,
            packing={},
        )

    assert resolve({}).dataset_config.tokenize_special_tokens == "eos"
    assert resolve({"tokenize_special_tokens": "none"}).dataset_config.tokenize_special_tokens == "none"


def test_instruction_special_tokens_validate_template_ownership():
    for policy in ("assistant_eos", "tokenizer_default"):
        with pytest.raises(ValueError, match="require use_hf_chat_template=false"):
            ZephonInstructionSourceConfig(
                column_mapping={"question": "q", "answer": "a"}, use_hf_chat_template=True, special_tokens=policy
            )
    assert (
        ZephonInstructionSourceConfig(
            column_mapping={"question": "q", "answer": "a"}, use_hf_chat_template=True, special_tokens="eos"
        ).special_tokens
        == "eos"
    )


@pytest.mark.parametrize("instruction", [False, True])
def test_numeric_token_overrides_require_corresponding_explicit_insertion(instruction: bool) -> None:
    config_type = ZephonInstructionSourceConfig if instruction else ZephonTextSourceConfig
    fields = {"column_mapping": {"question": "q", "answer": "a"}} if instruction else {}
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
        ("ZephonTextSourceConfig", {}),
        ("ZephonEncodedTextSourceConfig", {}),
        ("ZephonInstructionSourceConfig", {"column_mapping": {"question": "q", "answer": "a"}}),
    ],
)
def test_unknown_source_fields_are_actionable(target, kwargs):
    with pytest.raises(ValueError, match=f"Unexpected {target} dataset fields.*typo"):
        ZephonDataloaderConfig.from_recipe(
            f"nemo_automodel.components.datasets.zephon.{target}",
            {**kwargs, "typo": 1},
            {"zephon": {"sources": [{"name": "text", "path": "unused"}]}},
            seed=42,
            batch_size=1,
            packing={},
        )


def test_toml_missing_file_source_path_and_source_options_have_context(tmp_path):
    settings = {"data_config": str(tmp_path / "missing.toml")}

    def resolve():
        return ZephonDataloaderConfig.from_recipe(
            ZephonTextSourceConfig, {}, {"zephon": settings}, seed=42, batch_size=1, packing={}
        )

    with pytest.raises(ValueError, match="data_config file does not exist.*missing.toml"):
        resolve()
    path = tmp_path / "invalid.toml"
    path.write_text('[[sources]]\nname="broken"\n')
    settings["data_config"] = str(path)
    with pytest.raises(ValueError, match="source 'broken'.*nonempty path"):
        resolve()
    path.write_text('[[sources]]\nname="broken"\npath="data"\ntyop=1\n')
    with pytest.raises(ValueError, match="source 'broken'.*tyop"):
        resolve()


@pytest.mark.parametrize(
    "global_batch,local_batch,canonical,valid", [(16, 2, 4, True), (12, 2, 4, False), (6, 2, 3, True)]
)
def test_pinned_canonical_step_alignment_fails_before_runtime(global_batch, local_batch, canonical, valid):
    raw = {
        "dataset": {"_target_": "nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig"},
        "dataloader": {
            "_target_": "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig",
            "zephon": {
                "sources": [{"name": "text", "path": "unused"}],
                "canonical_replicas": canonical,
            },
        },
        "step_scheduler": {"max_steps": 2, "global_batch_size": global_batch, "local_batch_size": local_batch},
    }
    if valid:
        assert RecipeConfig(ConfigNode(raw)).dataloader.zephon.checkpoint.canonical_replicas == canonical
    else:
        with pytest.raises(ValueError, match="canonical_replicas.*lane windows"):
            RecipeConfig(ConfigNode(raw)).dataloader


def test_output_format_recipe_choice_and_early_validation(tmp_path):
    values = {
        "dataset": {"_target_": "nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig"},
        "dataloader": {
            "_target_": "nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig",
            "output_format": "thd",
            "zephon": {"sources": [{"name": "text", "path": str(tmp_path)}]},
        },
        "step_scheduler": {"max_steps": 2},
    }
    loader = RecipeConfig(ConfigNode(values)).dataloader
    assert loader.output_format == "thd"
    assert loader.is_packed
    with pytest.raises(ValueError, match="accepts seq_lens"):
        loader.build(tokenizer=None, dp_rank=0, dp_world_size=1, supports_seq_lens=False)
    values["dataloader"]["output_format"] = "neat"
    with pytest.raises(ValueError, match="output_format"):
        RecipeConfig(ConfigNode(values)).dataloader
    del values["dataloader"]["output_format"]
    assert RecipeConfig(ConfigNode(values)).dataloader.output_format == "batched"


def test_grouped_options_merge_inline_overrides_per_setting(tmp_path):
    from nemo_automodel.components.datasets.zephon.config import (
        resolve_zephon_recipe_settings,
        zephon_options_from_recipe,
    )

    recipe = tmp_path / "data.toml"
    recipe.write_text(
        '[[sources]]\nname="text"\npath="corpus"\n'
        '[packing]\nseq_len=17\npack_algorithm="best_fit"\n'
        '[cache]\ncache_dir="cache"\ncache_limit_bytes=4096\n'
        '[checkpoint]\naggregate_dir="checkpoints"\n'
        "[source]\ntoken_estimation=false\nstop_after_passes=2\n"
    )
    values = resolve_zephon_recipe_settings(
        {"zephon": {"data_config": str(recipe), "packing": {"seq_len": 23}, "workers": {"tokenize_parallelism": 4}}}
    )
    options = zephon_options_from_recipe(values, seed=9)
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
    assert options.seed == 9


def test_grouped_options_reject_unknown_misplaced_or_duplicate_fields():
    from nemo_automodel.components.datasets.zephon.config import zephon_options_from_recipe

    sources = [{"name": "text", "path": "unused"}]
    for settings, message in (
        ({"workers": {"seq_len": 8}}, "Unexpected Zephon workers"),
        ({"packing": {"unknown": 1}}, "Unexpected Zephon packing"),
        ({"packing": 4}, "packing must be a mapping"),
        ({"seq_len": 8, "packing": {"seq_len": 9}}, "packing.seq_len only once"),
    ):
        with pytest.raises(ValueError, match=message):
            zephon_options_from_recipe({"sources": sources, **settings}, seed=42)

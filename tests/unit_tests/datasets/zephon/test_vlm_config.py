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

"""Recipe-boundary image VLM settings and unsupported-mode checks."""

import subprocess
import sys
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.loader import DatasetBuildSchedule
from nemo_automodel.components.datasets.packing import DEFAULT_PACKED_SEQUENCE_CONTRACT, PackedSequenceContract
from nemo_automodel.components.datasets.vlm.loader import (
    ImagePositionState,
    VlmDataloaderBuild,
    VlmDataloaderConfig,
    VlmProcessorConfig,
    VlmStreamBuildContext,
    VlmVideoProcessorConfig,
)
from nemo_automodel.components.datasets.zephon.config import (
    ZephonPackingOptions,
    ZephonSource,
    ZephonTrainingOptions,
)
from nemo_automodel.components.datasets.zephon.vlm_config import (
    ZephonVlmDataloaderConfig,
    ZephonVlmDatasetConfig,
    ZephonVlmOptions,
)
from nemo_automodel.components.models.common.packing import PackingCapabilities
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.utils.vlm_fixtures import processor as processor


def _recipe():
    return {
        "seed": 17,
        "step_scheduler": {"max_steps": 10, "local_batch_size": 2, "global_batch_size": 2},
        "dataset": {
            "_target_": ZephonVlmDatasetConfig,
            "zephon": {
                "sources": [{"name": "images", "path": "data/images", "fmt": "jsonl"}],
                "packing": {"pack_algorithm": "first_fit"},
            },
        },
        "dataloader": {},
    }


def _build(
    config: ZephonVlmDatasetConfig | VlmDataloaderConfig,
    *,
    pretrained_model_name_or_path: str = "unused",
    dp_rank: int = 0,
    dp_world_size: int = 1,
    batch_size: int = 2,
    training_schedule: DatasetBuildSchedule | None = None,
    pp_n_microbatches: int | None = None,
    position_state_factory: Callable[[], ImagePositionState] | None = None,
    packing_contract: PackedSequenceContract = DEFAULT_PACKED_SEQUENCE_CONTRACT,
    dataset_build_context: AbstractContextManager[object] | None = None,
    is_training: bool = True,
    seed: int = 17,
) -> VlmDataloaderBuild:
    if isinstance(config, ZephonVlmDatasetConfig):
        config = ZephonVlmDataloaderConfig(dataset_config=config)
    return config.build(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=batch_size,
        training_schedule=training_schedule or DatasetBuildSchedule(batch_size, 2, 10, None),
        pp_n_microbatches=pp_n_microbatches,
        position_state_factory=position_state_factory,
        packing_contract=packing_contract,
        dataset_build_context=dataset_build_context,
        is_training=is_training,
        seed=seed,
    )


def _context(**overrides: object) -> VlmStreamBuildContext:
    return replace(
        VlmStreamBuildContext(
            seed=17,
            batch_size=2,
            dp_rank=0,
            dp_world_size=1,
            pretrained_model_name_or_path="unused",
            processor_config=VlmProcessorConfig(),
            training_schedule=DatasetBuildSchedule(2, 2, 10, None),
        ),
        **overrides,
    )


def test_recipe_routes_images_and_inherits_processor_settings():
    values = _recipe()
    values["processor"] = {"max_pixels": 1003520}
    values["dataset"]["zephon"]["inject_fake_images"] = False
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert isinstance(config, VlmDataloaderConfig)
    assert isinstance(config, ZephonVlmDataloaderConfig)
    assert isinstance(config.dataset_config, ZephonVlmDatasetConfig)
    assert "seed" not in values["dataset"]["zephon"]
    assert config.processor_config.kwargs == {"max_pixels": 1003520}
    assert not config.dataset_config.zephon.inject_fake_images


@pytest.mark.parametrize("groups", [{}, {"packing": {}}, {"packing": {"seq_len": 128}, "source": {"chunk_size": 4}}])
def test_image_recipe_defaults_keep_documents_whole_without_mutating_input(groups):
    values = _recipe()
    values["dataset"]["zephon"] = {"sources": values["dataset"]["zephon"]["sources"], **groups}
    options = RecipeConfig(ConfigNode(values)).vlm_dataloader.dataset_config.zephon
    assert options.packing.pack_algorithm == "first_fit"
    assert options.packing.pack_num_bins == 64
    assert options.packing.drop_oversized and options.packing.shuffle_after_pack
    assert options.source.token_estimation is None
    assert values["dataset"]["zephon"].get("packing", {}) == groups.get("packing", {})


@pytest.mark.parametrize("algorithm", [None, "wrap", "best_fit_wrap"])
def test_image_recipe_rejects_explicit_non_image_packing(algorithm):
    values = _recipe()
    values["dataset"]["zephon"]["packing"]["pack_algorithm"] = algorithm
    with pytest.raises(ValueError, match="first_fit or best_fit"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


def test_image_example_config_resolves():
    path = Path(__file__).resolve().parents[4] / "examples/vlm_finetune/zephon/train.yaml"
    recipe = RecipeConfig(ConfigNode(yaml.safe_load(path.read_text())))
    config = recipe.vlm_dataloader
    assert isinstance(config, VlmDataloaderConfig)
    assert isinstance(config, ZephonVlmDataloaderConfig)
    assert isinstance(config.dataset_config, ZephonVlmDatasetConfig)
    assert config.dataset_config.zephon.source.stop_after_passes is None
    assert config.dataset_config.zephon.packing.pack_algorithm == "first_fit"


def test_vlm_toml_paths_and_inline_overrides_share_llm_rules(tmp_path):
    data_config = tmp_path / "source.toml"
    data_config.write_text(
        'pack_algorithm="best_fit"\nseq_len=128\n[[sources]]\nname="images"\npath="corpus"\nfmt="jsonl"\n'
    )
    values = _recipe()
    values["dataset"]["zephon"] = {"data_config": str(data_config), "seq_len": 256}
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert config.dataset_config.zephon.sources[0].path == str(tmp_path / "corpus")
    assert config.dataset_config.zephon.packing.seq_len == 256
    assert config.dataset_config.zephon.packing.pack_algorithm == "best_fit"


def test_vlm_source_rejects_unknown_fields_and_shares_lane_alignment():
    values = _recipe()
    values["dataset"]["typo"] = True
    with pytest.raises(TypeError, match="unexpected dataset config field.*typo"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader
    del values["dataset"]["typo"]
    values["dataset"]["zephon"]["canonical_replicas"] = 4
    values["step_scheduler"]["global_batch_size"] = 12
    with pytest.raises(ValueError, match="canonical_replicas.*lane windows"):
        _build(
            RecipeConfig(ConfigNode(values)).vlm_dataloader,
            training_schedule=DatasetBuildSchedule(2, 12, 10, None),
        )
    values["step_scheduler"]["global_batch_size"] = 16
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert config.dataset_config.zephon.checkpoint.canonical_replicas == 4
    # A later build can use a valid schedule without changing the declarative config.
    config.dataset_config.validate_context(_context(training_schedule=DatasetBuildSchedule(2, 16, 10, None)))


@pytest.mark.parametrize("batch_size,pp_n_microbatches,message", [(0, None, "batch_size"), (3, 2, "microbatches")])
def test_cheap_vlm_build_validation_precedes_position_factory(batch_size, pp_n_microbatches, message):
    config = ZephonVlmDatasetConfig(
        zephon=ZephonVlmOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(pack_algorithm="first_fit"),
        )
    )

    def must_not_run():
        raise AssertionError("position factory ran before cheap validation")

    with pytest.raises(ValueError, match=message):
        _build(
            config,
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=batch_size,
            pp_n_microbatches=pp_n_microbatches,
            position_state_factory=must_not_run,
        )


@pytest.mark.parametrize("allow_xdoc", [False, True])
def test_vlm_input_limit_is_checked_before_position_state_and_processor(allow_xdoc):
    config = ZephonVlmDatasetConfig(
        zephon=ZephonVlmOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(seq_len=129, pack_algorithm="first_fit"),
            training=ZephonTrainingOptions(allow_cross_document_attention=allow_xdoc),
        )
    )

    def must_not_run():
        raise AssertionError("position factory ran before input-limit validation")

    with pytest.raises(ValueError, match="seq_len=129 exceeds.*128"):
        _build(
            config,
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=1,
            position_state_factory=must_not_run,
            packing_contract=PackingCapabilities("block_causal", max_sequence_length=128),
        )


def test_vlm_requires_explicit_length_when_model_limit_is_unknown():
    config = ZephonVlmDatasetConfig(
        zephon=ZephonVlmOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(pack_algorithm="first_fit"),
        )
    )

    def must_not_run():
        raise AssertionError("position factory ran before length resolution")

    with pytest.raises(ValueError, match="Set dataset.zephon.packing.seq_len"):
        _build(
            config,
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=1,
            position_state_factory=must_not_run,
        )


def test_inferred_vlm_length_reserves_expanded_fake_image_segment(tmp_path, processor):
    from nemo_automodel.components.datasets.zephon.vlm import prepare_fake_image_document
    from nemo_automodel.components.models.qwen2_5_vl.preprocessing import Qwen2_5_VLPositionState
    from tests.utils.vlm_fixtures import tiny_vlm_config

    processor.image_processor.size["longest_edge"] = 256
    processor.save_pretrained(tmp_path)
    positions = Qwen2_5_VLPositionState(tiny_vlm_config())
    fake_length = len(prepare_fake_image_document(processor, positions)["input_ids"])
    model_limit = fake_length - 1
    config = ZephonVlmDatasetConfig(
        zephon=ZephonVlmOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(pack_algorithm="first_fit"),
        )
    )

    with pytest.raises(ValueError, match=f"reserved fake-image segment length.*got seq_len={model_limit}"):
        _build(
            config,
            pretrained_model_name_or_path=str(tmp_path),
            dp_rank=0,
            dp_world_size=1,
            batch_size=1,
            position_state_factory=lambda: positions,
            packing_contract=PackingCapabilities("block_causal", max_sequence_length=model_limit),
        )
    assert config.zephon.packing.seq_len is None


def test_vlm_rejects_native_packing_and_unsized_run_without_cap():
    values = _recipe()
    values["packed_sequence"] = {"pack_size": 128}
    with pytest.raises(ValueError, match="Stream datasets own packing"):
        _build(RecipeConfig(ConfigNode(values)).vlm_dataloader)
    del values["packed_sequence"]
    del values["step_scheduler"]["max_steps"]
    with pytest.raises(ValueError, match="positive step_scheduler.max_steps"):
        _build(
            RecipeConfig(ConfigNode(values)).vlm_dataloader,
            training_schedule=DatasetBuildSchedule(2, 2, None, None),
        )


def test_vlm_rejects_collator_override_and_zephon_validation():
    values = _recipe()
    values["dataloader"]["collate_fn"] = {"_target_": "builtins.list"}
    with pytest.raises(ValueError, match="Stream datasets own packing"):
        _build(RecipeConfig(ConfigNode(values)).vlm_dataloader)
    del values["dataloader"]["collate_fn"]
    values["validation_dataset"] = values["dataset"]
    values["validation_dataloader"] = values["dataloader"]
    with pytest.raises(ValueError, match="Zephon validation"):
        _build(RecipeConfig(ConfigNode(values)).vlm_validation_dataloader, is_training=False)


def test_native_llm_import_does_not_require_zephon_or_vlm_loader():
    code = """
import sys
sys.modules['zephon'] = None
sys.modules['nemo_automodel.components.datasets.vlm.loader'] = None
from nemo_automodel.components.datasets.zephon import ZephonTextDatasetConfig
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
assert 'nemo_automodel.components.datasets.zephon.vlm' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_thd_vlm_rejects_fake_exclusions_and_pp_before_loading_processors():
    options = ZephonVlmOptions(
        sources=(ZephonSource(name="images", path="unused"),), packing=ZephonPackingOptions(pack_algorithm="first_fit")
    )
    with pytest.raises(ValueError, match="inject_fake_images=false"):
        ZephonVlmDatasetConfig(zephon=options, output_format="thd")
    config = ZephonVlmDatasetConfig(zephon=replace(options, inject_fake_images=False), output_format="thd")
    with pytest.raises(ValueError, match="VLM pipeline"):
        _build(
            config,
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=2,
            pp_n_microbatches=2,
        )


def test_outer_template_and_processor_settings_reach_parent_and_worker_config(monkeypatch):
    from nemo_automodel.components.datasets.zephon import llm, pipeline, vlm

    events = []
    processor = SimpleNamespace(tokenizer=SimpleNamespace(chat_template=None), chat_template=None)
    positions = object()
    encoder_configs = []

    class ProcessorContext:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *_):
            events.append("exit")

    def make_processor():
        events.append("processor")
        return processor

    def encoder_build(config, *, pretrained_model_name_or_path, positions):
        assert events == ["enter", "processor", "exit"]
        encoder_configs.append((config, pretrained_model_name_or_path, positions))
        return object()

    class SourceReached(Exception):
        pass

    def source_build(options, *, seed, dp_rank, dp_world_size):
        assert events == ["enter", "processor", "exit"]
        assert seed == 17 and dp_rank == 0 and dp_world_size == 1
        raise SourceReached

    monkeypatch.setattr(llm, "resolve_token_ids", lambda *args, **kwargs: SimpleNamespace(pad=0))
    monkeypatch.setattr(vlm.VlmEncoderConfig, "build", encoder_build)
    monkeypatch.setattr(pipeline, "create_source_pipeline", source_build)
    config = ZephonVlmDataloaderConfig(
        dataset_config=ZephonVlmDatasetConfig(
            zephon=ZephonVlmOptions(
                sources=(ZephonSource(name="images", path="unused"),),
                packing=ZephonPackingOptions(seq_len=16, pack_algorithm="first_fit"),
                inject_fake_images=False,
            )
        ),
        processor_config=VlmProcessorConfig(factory=make_processor),
        chat_template="{{ messages }}",
    )
    with pytest.raises(SourceReached):
        _build(config, dataset_build_context=ProcessorContext(), position_state_factory=lambda: positions, seed=17)
    assert processor.chat_template == processor.tokenizer.chat_template == "{{ messages }}"
    encoder, processor_path, encoder_positions = encoder_configs[0]
    assert encoder.processor is config.processor_config
    assert encoder.chat_template == config.chat_template
    assert processor_path == "unused" and encoder_positions is positions
    assert config.dataset_config.zephon.packing.seq_len == 16


@pytest.mark.parametrize(
    "name,value",
    [
        ("inject_fake_images", False),
        ("truncate", False),
        ("pretokenize", False),
        ("packing", {"enabled": False}),
    ],
)
def test_stream_recipe_rejects_consumed_native_preprocessing_fields(name, value):
    values = _recipe()
    values["dataset"][name] = value
    with pytest.raises(TypeError, match=f"native VLM preprocessing fields: {name}"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


@pytest.mark.parametrize("packing", [{}, {"pack_size": 0}, {"pack_size": 0, "pretokenize": False}])
def test_stream_recipe_rejects_explicit_disabled_native_packing(packing):
    values = _recipe()
    values["packed_sequence"] = packing
    with pytest.raises(ValueError, match="Stream datasets own packing"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


def test_native_vlm_recipe_retains_preprocessing_fields_and_shared_chat_template():
    from nemo_automodel.components.datasets.vlm.mock import MockVlmDatasetConfig

    config = RecipeConfig.resolve_vlm_dataloader(
        {
            "_target_": MockVlmDatasetConfig,
            "max_length": 128,
            "pretokenize": True,
            "truncate": False,
            "inject_fake_images": False,
            "packing": {"enabled": False},
            "chat_template": "{{ messages }}",
        },
        {},
        packed_sequence_node={"pack_size": 128},
    )
    assert isinstance(config.dataset_config, MockVlmDatasetConfig)
    assert config.dataset_config.max_length == 128
    assert config.pretokenization.max_length == 128
    assert not config.pretokenization.truncate
    assert not config.pretokenization.inject_fake_images
    assert config.packing.pack_size == 128
    assert config.chat_template == "{{ messages }}"


def test_image_source_alias_shares_normalization_and_preserves_template():
    values = _recipe()
    values["dataset"]["zephon"].pop("sources")
    values["dataset"].update(
        path_or_dataset_id="org/images", name="captions", split="validation", chat_template="custom"
    )
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert config.dataset_config.zephon.sources == (
        ZephonSource(name="source_0", path="hf://org/images/captions/validation"),
    )
    assert config.chat_template == "custom"
    assert replace(config.dataset_config).zephon.sources == config.dataset_config.zephon.sources


@pytest.mark.parametrize(
    "target",
    [
        None,
        ZephonVlmDataloaderConfig,
        "nemo_automodel.components.datasets.zephon.vlm_config.ZephonVlmDataloaderConfig",
    ],
)
def test_image_loader_subclass_reports_the_dataset_output_format(target):
    values = _recipe()
    if target is None:
        values.pop("dataloader")
    else:
        values["dataloader"]["_target_"] = target
    values["dataset"]["output_format"] = "thd"
    values["dataset"]["zephon"]["inject_fake_images"] = False
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert isinstance(config, ZephonVlmDataloaderConfig)
    assert config.output_format == "thd" and config.is_packed
    assert config.packing is None


def test_image_and_text_loaders_reject_incompatible_dataset_configs():
    from nemo_automodel.components.datasets.zephon.config import ZephonDataloaderConfig, ZephonTextDatasetConfig

    image_dataset = RecipeConfig(ConfigNode(_recipe())).vlm_dataloader.dataset_config
    text_dataset = ZephonTextDatasetConfig(zephon={"sources": [{"name": "text", "path": "unused"}]})
    with pytest.raises(TypeError, match="compatible VlmDataloaderConfig subclass"):
        VlmDataloaderConfig(dataset_config=image_dataset)
    with pytest.raises(TypeError, match="Zephon image dataset config"):
        ZephonVlmDataloaderConfig(dataset_config=text_dataset)
    with pytest.raises(TypeError, match="Zephon text dataset config"):
        ZephonDataloaderConfig(dataset_config=image_dataset)
    values = _recipe()
    values["dataloader"] = {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader"}
    with pytest.raises(TypeError, match="compatible VlmDataloaderConfig subclass"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


def test_dataset_loader_default_rejects_the_wrong_recipe_family():
    with pytest.raises(ValueError, match="expected a DataloaderConfig subclass"):
        RecipeConfig(ConfigNode(_recipe())).dataloader
    values = _recipe()
    values["dataset"] = {
        "_target_": "nemo_automodel.components.datasets.zephon.ZephonTextDatasetConfig",
        "zephon": {"sources": [{"name": "text", "path": "unused"}]},
    }
    with pytest.raises(ValueError, match="expected a VlmDataloaderConfig subclass"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


def test_video_processor_is_rejected_before_loading_or_source_work(monkeypatch):
    from unittest.mock import Mock

    config = RecipeConfig(ConfigNode(_recipe())).vlm_dataloader
    processor_factory = Mock()
    video_factory = Mock()
    positions = Mock()
    source = Mock()
    monkeypatch.setattr("nemo_automodel.components.datasets.zephon.pipeline.create_source_pipeline", source)
    config = replace(
        config,
        processor_config=VlmProcessorConfig(
            factory=processor_factory,
            video_processor=VlmVideoProcessorConfig(factory=video_factory),
        ),
    )
    with pytest.raises(ValueError, match="does not support video processors"):
        _build(config, position_state_factory=positions)
    processor_factory.assert_not_called()
    video_factory.assert_not_called()
    positions.assert_not_called()
    source.assert_not_called()


def test_image_python_and_toml_partial_groups_keep_image_defaults(tmp_path):
    sources = (ZephonSource(name="images", path="unused"), ZephonSource(name="text", path="unused-text"))
    defaults = ZephonVlmOptions(sources=sources)
    direct = ZephonVlmDatasetConfig(zephon=replace(defaults, source=replace(defaults.source, chunk_size=4)))
    assert direct.zephon.source.chunk_size == 4
    assert direct.zephon.packing.pack_algorithm == "first_fit"
    assert direct.zephon.source.token_estimation is None
    data_config = tmp_path / "images.toml"
    data_config.write_text(
        "[source]\nchunk_size=4\n[packing]\nseq_len=128\n"
        '[[sources]]\nname="images"\npath="unused"\n[[sources]]\nname="text"\npath="unused-text"\n'
    )
    values = _recipe()
    values["dataset"]["zephon"] = {
        "data_config": str(data_config),
        "packing": {"seq_len": 256},
        "source": {"chunk_size": 8},
    }
    options = RecipeConfig(ConfigNode(values)).vlm_dataloader.dataset_config.zephon
    assert options.packing.pack_algorithm == direct.zephon.packing.pack_algorithm
    assert options.packing.seq_len == 256 and options.packing.pack_num_bins == 64
    assert options.source.chunk_size == 8 and options.source.token_estimation is None


@pytest.mark.parametrize("estimation", ["auto", True, {}])
def test_image_recipe_rejects_unsupported_token_estimation(estimation):
    values = _recipe()
    values["dataset"]["zephon"]["source"] = {"token_estimation": estimation}
    with pytest.raises(ValueError, match="media-aware estimator"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader

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
from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.datasets.zephon import ZephonVlmDataloaderConfig, ZephonVlmSourceConfig
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonSource,
    ZephonTrainingOptions,
)
from nemo_automodel.components.models.common.packing import PackingCapabilities
from nemo_automodel.recipes._typed_config import RecipeConfig
from tests.utils.vlm_fixtures import processor as processor


def _recipe():
    return {
        "seed": 17,
        "step_scheduler": {"max_steps": 10, "local_batch_size": 2, "global_batch_size": 2},
        "dataset": {"_target_": ZephonVlmSourceConfig},
        "dataloader": {
            "_target_": ZephonVlmDataloaderConfig,
            "zephon": {
                "sources": [{"name": "images", "path": "data/images", "fmt": "jsonl"}],
                "pack_algorithm": "first_fit",
            },
        },
    }


def test_recipe_routes_images_and_inherits_processor_settings():
    values = _recipe()
    values["processor"] = {"max_pixels": 1003520}
    values["dataset"]["inject_fake_images"] = False
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert isinstance(config, ZephonVlmDataloaderConfig)
    assert config.zephon.seed == 17
    assert config.processor_config.kwargs == {"max_pixels": 1003520}
    assert not config.dataset_config.inject_fake_images


def test_image_recipe_defaults_to_whole_document_packing_without_mutating_input():
    values = _recipe()
    del values["dataloader"]["zephon"]["pack_algorithm"]
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert config.zephon.packing.pack_algorithm == "first_fit"
    assert "packing" not in values["dataloader"]["zephon"]
    assert "pack_algorithm" not in values["dataloader"]["zephon"]


@pytest.mark.parametrize("algorithm", [None, "wrap", "best_fit_wrap"])
def test_image_recipe_rejects_explicit_non_image_packing(algorithm):
    values = _recipe()
    values["dataloader"]["zephon"]["pack_algorithm"] = algorithm
    with pytest.raises(ValueError, match="first_fit or best_fit"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


def test_image_example_config_resolves():
    path = Path(__file__).resolve().parents[4] / "examples/vlm_finetune/zephon/qwen2_5_vl.yaml"
    recipe = RecipeConfig(ConfigNode(yaml.safe_load(path.read_text())))
    config = recipe.vlm_dataloader
    assert isinstance(config, ZephonVlmDataloaderConfig)
    assert config.zephon.source.stop_after_passes == 1
    assert config.zephon.packing.pack_algorithm == "first_fit"


def test_vlm_toml_paths_and_inline_overrides_share_llm_rules(tmp_path):
    data_config = tmp_path / "source.toml"
    data_config.write_text(
        'pack_algorithm="best_fit"\nseq_len=128\n[[sources]]\nname="images"\npath="corpus"\nfmt="jsonl"\n'
    )
    values = _recipe()
    values["dataloader"]["zephon"] = {"data_config": str(data_config), "seq_len": 256}
    config = RecipeConfig(ConfigNode(values)).vlm_dataloader
    assert config.zephon.sources[0].path == str(tmp_path / "corpus")
    assert config.zephon.packing.seq_len == 256
    assert config.zephon.packing.pack_algorithm == "best_fit"


def test_vlm_source_rejects_unknown_fields_and_shares_lane_alignment():
    values = _recipe()
    values["dataset"]["typo"] = True
    with pytest.raises(ValueError, match="Unexpected ZephonVlmSourceConfig dataset fields.*typo"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader
    del values["dataset"]["typo"]
    values["dataloader"]["zephon"]["canonical_replicas"] = 4
    values["step_scheduler"]["global_batch_size"] = 12
    with pytest.raises(ValueError, match="canonical_replicas.*lane windows"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader
    values["step_scheduler"]["global_batch_size"] = 16
    assert RecipeConfig(ConfigNode(values)).vlm_dataloader.zephon.checkpoint.canonical_replicas == 4


@pytest.mark.parametrize("batch_size,pp_n_microbatches,message", [(0, None, "batch_size"), (3, 2, "microbatches")])
def test_cheap_vlm_build_validation_precedes_position_factory(batch_size, pp_n_microbatches, message):
    config = ZephonVlmDataloaderConfig(
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(pack_algorithm="first_fit"),
        )
    )

    def must_not_run():
        raise AssertionError("position factory ran before cheap validation")

    with pytest.raises(ValueError, match=message):
        config.build(
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=batch_size,
            pp_n_microbatches=pp_n_microbatches,
            position_state_factory=must_not_run,
        )


@pytest.mark.parametrize("allow_xdoc", [False, True])
def test_vlm_input_limit_is_checked_before_position_state_and_processor(allow_xdoc):
    config = ZephonVlmDataloaderConfig(
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(seq_len=129, pack_algorithm="first_fit"),
            training=ZephonTrainingOptions(allow_cross_document_attention=allow_xdoc),
        )
    )

    def must_not_run():
        raise AssertionError("position factory ran before input-limit validation")

    with pytest.raises(ValueError, match="seq_len=129 exceeds.*128"):
        config.build(
            pretrained_model_name_or_path="unused",
            dp_rank=0,
            dp_world_size=1,
            batch_size=1,
            position_state_factory=must_not_run,
            packing_contract=PackingCapabilities("block_causal", max_sequence_length=128),
        )


def test_vlm_requires_explicit_length_when_model_limit_is_unknown():
    config = ZephonVlmDataloaderConfig(
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(pack_algorithm="first_fit"),
        )
    )

    def must_not_run():
        raise AssertionError("position factory ran before length resolution")

    with pytest.raises(ValueError, match="Set dataloader.zephon.packing.seq_len"):
        config.build(
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
    config = ZephonVlmDataloaderConfig(
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path="unused"),),
            packing=ZephonPackingOptions(pack_algorithm="first_fit"),
        )
    )

    with pytest.raises(ValueError, match=f"reserved fake-image segment length.*got seq_len={model_limit}"):
        config.build(
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
    with pytest.raises(ValueError, match="Native VLM packing"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader
    del values["packed_sequence"]
    del values["step_scheduler"]["max_steps"]
    with pytest.raises(ValueError, match="positive step_scheduler.max_steps"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader


def test_vlm_rejects_collator_override_and_zephon_validation():
    values = _recipe()
    values["dataloader"]["collate_fn"] = {"_target_": "builtins.list"}
    with pytest.raises(ValueError, match="Unexpected Zephon loader"):
        RecipeConfig(ConfigNode(values)).vlm_dataloader
    del values["dataloader"]["collate_fn"]
    values["validation_dataset"] = values["dataset"]
    values["validation_dataloader"] = values["dataloader"]
    with pytest.raises(ValueError, match="training backend"):
        RecipeConfig(ConfigNode(values)).vlm_validation_dataloader


def test_native_llm_import_does_not_require_zephon_or_vlm_loader():
    code = """
import sys
sys.modules['zephon'] = None
sys.modules['nemo_automodel.components.datasets.vlm.loader'] = None
from nemo_automodel.components.datasets.zephon import ZephonDataloaderConfig, ZephonTextSourceConfig
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
assert 'nemo_automodel.components.datasets.zephon.vlm' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_thd_vlm_rejects_fake_exclusions_and_pp_before_loading_processors():
    from nemo_automodel.components.datasets.zephon.config import ZephonOptions
    from nemo_automodel.components.datasets.zephon.vlm_config import ZephonVlmSourceConfig

    options = ZephonOptions(
        sources=(ZephonSource(name="images", path="unused"),), packing=ZephonPackingOptions(pack_algorithm="first_fit")
    )
    with pytest.raises(ValueError, match="inject_fake_images=false"):
        ZephonVlmDataloaderConfig(zephon=options, output_format="thd")
    config = ZephonVlmDataloaderConfig(
        zephon=options, output_format="thd", dataset_config=ZephonVlmSourceConfig(inject_fake_images=False)
    )
    with pytest.raises(ValueError, match="VLM pipeline"):
        config.build(
            pretrained_model_name_or_path="unused", dp_rank=0, dp_world_size=1, batch_size=2, pp_n_microbatches=2
        )

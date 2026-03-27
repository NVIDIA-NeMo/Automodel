# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for Pydantic config schema validation."""

import json

import pytest
from pydantic import ValidationError

from nemo_automodel.components.config.schema import (
    RECIPE_SCHEMAS,
    BiencoderRecipeConfig,
    CheckpointConfig,
    DistributedConfig,
    KDRecipeConfig,
    LLMRecipeConfig,
    LrSchedulerConfig,
    PackedSequenceConfig,
    StepSchedulerConfig,
    TargetConfig,
    VLMRecipeConfig,
    WandbConfig,
    get_schema_json,
    validate_config,
)

# ---------------------------------------------------------------------------
# TargetConfig
# ---------------------------------------------------------------------------


class TestTargetConfig:
    def test_basic_target(self):
        cfg = TargetConfig.model_validate({"_target_": "torch.optim.Adam", "lr": 1e-5})
        assert cfg.target_ == "torch.optim.Adam"

    def test_missing_target_raises(self):
        with pytest.raises(ValidationError, match="target_"):
            TargetConfig.model_validate({"lr": 1e-5})

    def test_non_string_target_raises(self):
        with pytest.raises(ValidationError, match="_target_ must be a string"):
            TargetConfig.model_validate({"_target_": 123})

    def test_extra_kwargs_allowed(self):
        cfg = TargetConfig.model_validate(
            {"_target_": "torch.optim.Adam", "lr": 1e-5, "betas": [0.9, 0.999], "weight_decay": 0.01}
        )
        assert cfg.target_ == "torch.optim.Adam"
        # Extra fields accessible via model_extra
        assert cfg.model_extra["lr"] == 1e-5

    def test_empty_target_string(self):
        # Empty string is technically a string, but will fail at resolution time
        cfg = TargetConfig.model_validate({"_target_": ""})
        assert cfg.target_ == ""


# ---------------------------------------------------------------------------
# StepSchedulerConfig
# ---------------------------------------------------------------------------


class TestStepSchedulerConfig:
    def test_valid_config(self):
        cfg = StepSchedulerConfig(global_batch_size=64, local_batch_size=8, num_epochs=3)
        assert cfg.global_batch_size == 64
        assert cfg.local_batch_size == 8
        assert cfg.num_epochs == 3

    def test_zero_batch_size_raises(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            StepSchedulerConfig(global_batch_size=0, local_batch_size=8)

    def test_negative_batch_size_raises(self):
        with pytest.raises(ValidationError, match="greater than 0"):
            StepSchedulerConfig(global_batch_size=-1, local_batch_size=8)

    def test_string_batch_size_raises(self):
        with pytest.raises(ValidationError):
            StepSchedulerConfig(global_batch_size="abc", local_batch_size=8)

    def test_optional_fields_default_to_none(self):
        cfg = StepSchedulerConfig(global_batch_size=64, local_batch_size=8)
        assert cfg.max_steps is None
        assert cfg.ckpt_every_steps is None
        assert cfg.val_every_steps is None

    def test_extra_fields_allowed(self):
        cfg = StepSchedulerConfig(global_batch_size=64, local_batch_size=8, custom_field="hello")
        assert cfg.model_extra["custom_field"] == "hello"


# ---------------------------------------------------------------------------
# DistributedConfig
# ---------------------------------------------------------------------------


class TestDistributedConfig:
    def test_defaults(self):
        cfg = DistributedConfig()
        assert cfg.strategy == "fsdp2"
        assert cfg.tp_size == 1
        assert cfg.cp_size == 1
        assert cfg.dp_size is None

    def test_dp_size_none_string_normalized(self):
        cfg = DistributedConfig.model_validate({"dp_size": "none"})
        assert cfg.dp_size is None

    def test_dp_size_None_string_normalized(self):
        cfg = DistributedConfig.model_validate({"dp_size": "None"})
        assert cfg.dp_size is None

    def test_dp_size_integer(self):
        cfg = DistributedConfig(dp_size=4)
        assert cfg.dp_size == 4

    def test_zero_tp_size_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            DistributedConfig(tp_size=0)

    def test_pipeline_config(self):
        cfg = DistributedConfig.model_validate(
            {"pp_size": 2, "pipeline": {"pp_schedule": "1f1b", "pp_microbatch_size": 4}}
        )
        assert cfg.pp_size == 2
        assert cfg.pipeline.pp_schedule == "1f1b"


# ---------------------------------------------------------------------------
# CheckpointConfig
# ---------------------------------------------------------------------------


class TestCheckpointConfig:
    def test_valid_config(self):
        cfg = CheckpointConfig(checkpoint_dir="/tmp/checkpoints")
        assert cfg.enabled is True
        assert cfg.model_save_format == "safetensors"

    def test_checkpoint_dir_optional(self):
        cfg = CheckpointConfig()
        assert cfg.checkpoint_dir is None
        assert cfg.enabled is True

    def test_restore_from_latest(self):
        cfg = CheckpointConfig(checkpoint_dir="/tmp/ckpt", restore_from="LATEST")
        assert cfg.restore_from == "LATEST"


# ---------------------------------------------------------------------------
# LrSchedulerConfig
# ---------------------------------------------------------------------------


class TestLrSchedulerConfig:
    def test_defaults(self):
        cfg = LrSchedulerConfig()
        assert cfg.lr_decay_style == "cosine"
        assert cfg.lr_warmup_steps == 0
        assert cfg.min_lr == 0.0

    def test_fractional_warmup(self):
        cfg = LrSchedulerConfig(lr_warmup_steps=0.1)
        assert cfg.lr_warmup_steps == 0.1


# ---------------------------------------------------------------------------
# PackedSequenceConfig
# ---------------------------------------------------------------------------


class TestPackedSequenceConfig:
    def test_disabled_by_default(self):
        cfg = PackedSequenceConfig()
        assert cfg.packed_sequence_size == 0

    def test_negative_size_raises(self):
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            PackedSequenceConfig(packed_sequence_size=-1)


# ---------------------------------------------------------------------------
# WandbConfig
# ---------------------------------------------------------------------------


class TestWandbConfig:
    def test_valid_config(self):
        cfg = WandbConfig(project="my_project")
        assert cfg.project == "my_project"
        assert cfg.entity is None

    def test_missing_project_raises(self):
        with pytest.raises(ValidationError, match="project"):
            WandbConfig()


# ---------------------------------------------------------------------------
# LLMRecipeConfig (root)
# ---------------------------------------------------------------------------

# Minimal valid LLM config for reuse in tests
_MINIMAL_LLM_CONFIG = {
    "step_scheduler": {"global_batch_size": 64, "local_batch_size": 8},
    "model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"},
    "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-5},
    "loss_fn": {"_target_": "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy"},
    "dataset": {"_target_": "nemo_automodel.components.datasets.llm.hellaswag.HellaSwag"},
    "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader"},
}


class TestLLMRecipeConfig:
    def test_minimal_valid_config(self):
        cfg = LLMRecipeConfig.model_validate(_MINIMAL_LLM_CONFIG)
        assert cfg.step_scheduler.global_batch_size == 64
        assert cfg.model.target_ == "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"

    def test_missing_step_scheduler_raises(self):
        bad = {k: v for k, v in _MINIMAL_LLM_CONFIG.items() if k != "step_scheduler"}
        with pytest.raises(ValidationError, match="step_scheduler"):
            LLMRecipeConfig.model_validate(bad)

    def test_missing_model_raises(self):
        bad = {k: v for k, v in _MINIMAL_LLM_CONFIG.items() if k != "model"}
        with pytest.raises(ValidationError, match="model"):
            LLMRecipeConfig.model_validate(bad)

    def test_missing_optimizer_raises(self):
        bad = {k: v for k, v in _MINIMAL_LLM_CONFIG.items() if k != "optimizer"}
        with pytest.raises(ValidationError, match="optimizer"):
            LLMRecipeConfig.model_validate(bad)

    def test_with_all_optional_sections(self):
        full = {
            **_MINIMAL_LLM_CONFIG,
            "distributed": {"strategy": "fsdp2", "tp_size": 2},
            "checkpoint": {"checkpoint_dir": "/tmp/ckpt"},
            "lr_scheduler": {"lr_decay_style": "cosine", "lr_warmup_steps": 100},
            "wandb": {"project": "test"},
            "packed_sequence": {"packed_sequence_size": 2048},
            "seed": 42,
        }
        cfg = LLMRecipeConfig.model_validate(full)
        assert cfg.distributed.tp_size == 2
        assert cfg.checkpoint.checkpoint_dir == "/tmp/ckpt"
        assert cfg.seed == 42

    def test_bad_batch_size_in_step_scheduler_raises(self):
        bad = {**_MINIMAL_LLM_CONFIG, "step_scheduler": {"global_batch_size": -1, "local_batch_size": 8}}
        with pytest.raises(ValidationError, match="greater than 0"):
            LLMRecipeConfig.model_validate(bad)

    def test_bad_model_target_raises(self):
        bad = {**_MINIMAL_LLM_CONFIG, "model": {"lr": 1e-5}}  # missing _target_
        with pytest.raises(ValidationError, match="target_"):
            LLMRecipeConfig.model_validate(bad)

    def test_unknown_top_level_keys_allowed(self):
        extended = {**_MINIMAL_LLM_CONFIG, "my_custom_field": "hello"}
        cfg = LLMRecipeConfig.model_validate(extended)
        assert cfg.model_extra["my_custom_field"] == "hello"

    def test_optional_peft_section(self):
        with_peft = {
            **_MINIMAL_LLM_CONFIG,
            "peft": {"_target_": "nemo_automodel.components._peft.lora.PeftConfig", "dim": 8},
        }
        cfg = LLMRecipeConfig.model_validate(with_peft)
        assert cfg.peft.target_ == "nemo_automodel.components._peft.lora.PeftConfig"

    def test_optional_validation_dataset(self):
        with_val = {
            **_MINIMAL_LLM_CONFIG,
            "validation_dataset": {"_target_": "nemo_automodel.components.datasets.llm.hellaswag.HellaSwag"},
            "validation_dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader"},
        }
        cfg = LLMRecipeConfig.model_validate(with_val)
        assert cfg.validation_dataset is not None


# ---------------------------------------------------------------------------
# KDRecipeConfig
# ---------------------------------------------------------------------------


class TestKDRecipeConfig:
    def test_valid_kd_config(self):
        kd = {
            **_MINIMAL_LLM_CONFIG,
            "teacher_model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"},
            "kd_ratio": 0.5,
        }
        cfg = KDRecipeConfig.model_validate(kd)
        assert cfg.kd_ratio == 0.5
        assert cfg.teacher_model.target_ == "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"

    def test_missing_teacher_raises(self):
        with pytest.raises(ValidationError, match="teacher_model"):
            KDRecipeConfig.model_validate(_MINIMAL_LLM_CONFIG)

    def test_kd_ratio_out_of_range_raises(self):
        kd = {
            **_MINIMAL_LLM_CONFIG,
            "teacher_model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"},
            "kd_ratio": 1.5,
        }
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            KDRecipeConfig.model_validate(kd)


# ---------------------------------------------------------------------------
# VLMRecipeConfig
# ---------------------------------------------------------------------------


class TestVLMRecipeConfig:
    def test_valid_vlm_config(self):
        vlm = {
            "step_scheduler": {"global_batch_size": 32, "local_batch_size": 4},
            "model": {"_target_": "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"},
            "optimizer": {"_target_": "torch.optim.AdamW"},
            "loss_fn": {"_target_": "torch.nn.CrossEntropyLoss"},
            "dataset": {"_target_": "some.dataset"},
            "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader"},
            "processor": {"_target_": "transformers.AutoProcessor.from_pretrained"},
        }
        cfg = VLMRecipeConfig.model_validate(vlm)
        assert cfg.processor.target_ == "transformers.AutoProcessor.from_pretrained"


# ---------------------------------------------------------------------------
# BiencoderRecipeConfig
# ---------------------------------------------------------------------------


class TestBiencoderRecipeConfig:
    def test_valid_biencoder_config(self):
        be = {
            "step_scheduler": {"global_batch_size": 64, "local_batch_size": 8},
            "model": {"_target_": "nemo_automodel._transformers.auto_model.NeMoAutoModelBiencoder.from_pretrained"},
            "optimizer": {"_target_": "torch.optim.Adam"},
            "tokenizer": {"_target_": "nemo_automodel._transformers.auto_tokenizer.NeMoAutoTokenizer.from_pretrained"},
            "dataloader": {"_target_": "torchdata.stateful_dataloader.StatefulDataLoader"},
            "train_n_passages": 5,
            "temperature": 0.02,
        }
        cfg = BiencoderRecipeConfig.model_validate(be)
        assert cfg.train_n_passages == 5
        assert cfg.temperature == 0.02

    def test_zero_temperature_raises(self):
        be = {
            "step_scheduler": {"global_batch_size": 64, "local_batch_size": 8},
            "model": {"_target_": "x"},
            "optimizer": {"_target_": "x"},
            "tokenizer": {"_target_": "x"},
            "dataloader": {"_target_": "x"},
            "temperature": 0.0,
        }
        with pytest.raises(ValidationError, match="greater than 0"):
            BiencoderRecipeConfig.model_validate(be)


# ---------------------------------------------------------------------------
# Schema export & validate_config
# ---------------------------------------------------------------------------


class TestSchemaExport:
    def test_get_schema_json_llm(self):
        schema_str = get_schema_json("llm")
        schema = json.loads(schema_str)
        assert "properties" in schema
        assert "step_scheduler" in schema["properties"]
        assert "model" in schema["properties"]

    def test_get_schema_json_all_types(self):
        for recipe_type in RECIPE_SCHEMAS:
            schema_str = get_schema_json(recipe_type)
            schema = json.loads(schema_str)
            assert "properties" in schema

    def test_get_schema_json_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown recipe type"):
            get_schema_json("unknown_type")

    def test_validate_config_valid(self):
        model = validate_config(_MINIMAL_LLM_CONFIG, "llm")
        assert isinstance(model, LLMRecipeConfig)

    def test_validate_config_invalid(self):
        with pytest.raises(ValidationError):
            validate_config({"model": {"_target_": "x"}}, "llm")


# ---------------------------------------------------------------------------
# Validated config loading (integration)
# ---------------------------------------------------------------------------


class TestValidatedConfigLoading:
    def test_load_and_validate_from_yaml(self, tmp_path):
        import yaml

        cfg_dict = {
            **_MINIMAL_LLM_CONFIG,
            "distributed": {"strategy": "fsdp2", "tp_size": 1},
        }
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump(cfg_dict, sort_keys=False))

        from nemo_automodel.components.config.validated_config import load_and_validate_config

        cfg = load_and_validate_config(str(cfg_file), recipe_type="llm")
        # Should return a ConfigNode (backwards compatible)
        from nemo_automodel.components.config.loader import ConfigNode

        assert isinstance(cfg, ConfigNode)

    def test_load_and_validate_bad_config_raises(self, tmp_path):
        import yaml

        bad_dict = {"model": {"_target_": "x"}}  # missing required fields
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.safe_dump(bad_dict))

        from nemo_automodel.components.config.validated_config import load_and_validate_config

        with pytest.raises(ValidationError):
            load_and_validate_config(str(cfg_file), recipe_type="llm")

    def test_validate_config_file_valid(self, tmp_path):
        import yaml

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump(_MINIMAL_LLM_CONFIG, sort_keys=False))

        from nemo_automodel.components.config.validated_config import validate_config_file

        is_valid, msg = validate_config_file(str(cfg_file), recipe_type="llm")
        assert is_valid
        assert "valid" in msg.lower()

    def test_validate_config_file_invalid(self, tmp_path):
        import yaml

        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(yaml.safe_dump({"model": {"_target_": "x"}}))

        from nemo_automodel.components.config.validated_config import validate_config_file

        is_valid, msg = validate_config_file(str(cfg_file), recipe_type="llm")
        assert not is_valid
        assert "error" in msg.lower()

    def test_validate_config_file_not_found(self):
        from nemo_automodel.components.config.validated_config import validate_config_file

        is_valid, msg = validate_config_file("/nonexistent/path.yaml")
        assert not is_valid
        assert "not found" in msg.lower()

    def test_validate_config_file_with_overrides(self, tmp_path):
        import yaml

        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(yaml.safe_dump(_MINIMAL_LLM_CONFIG, sort_keys=False))

        from nemo_automodel.components.config.validated_config import validate_config_file

        # Override batch size
        is_valid, msg = validate_config_file(
            str(cfg_file), overrides=["step_scheduler.global_batch_size=128"], recipe_type="llm"
        )
        assert is_valid


# ---------------------------------------------------------------------------
# Auto-detect recipe type
# ---------------------------------------------------------------------------


class TestRecipeTypeDetection:
    def test_detect_kd(self):
        from nemo_automodel.components.config.validated_config import _detect_recipe_type

        assert _detect_recipe_type({"teacher_model": {}, "model": {}}) == "kd"

    def test_detect_diffusion(self):
        from nemo_automodel.components.config.validated_config import _detect_recipe_type

        assert _detect_recipe_type({"flow_matching": {}}) == "diffusion"
        assert _detect_recipe_type({"fsdp": {}}) == "diffusion"

    def test_detect_vlm(self):
        from nemo_automodel.components.config.validated_config import _detect_recipe_type

        assert _detect_recipe_type({"processor": {}}) == "vlm"

    def test_detect_biencoder(self):
        from nemo_automodel.components.config.validated_config import _detect_recipe_type

        assert _detect_recipe_type({"train_n_passages": 5, "tokenizer": {}}) == "biencoder"

    def test_detect_llm_default(self):
        from nemo_automodel.components.config.validated_config import _detect_recipe_type

        assert _detect_recipe_type({"model": {}, "optimizer": {}}) == "llm"

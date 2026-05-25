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

"""Tests for CheckpointingConfig in config.py and build_checkpoint_config in api.py."""

import pytest

from nemo_automodel.components.checkpoint.config import CheckpointingConfig


class TestCheckpointingConfig:
    def test_construct_safetensors(self):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="safetensors",
            model_cache_dir="/tmp/cache",
            model_repo_id="meta-llama/Llama-3-8B",
            save_consolidated=True,
            is_peft=False,
        )
        assert cfg.enabled is True
        assert str(cfg.checkpoint_dir) == "/tmp/ckpt"
        # __post_init__ converts string to SerializationFormat enum
        assert cfg.model_save_format.value == "safetensors"

    def test_construct_torch_save(self):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="torch_save",
            model_cache_dir="/tmp/cache",
            model_repo_id="test",
            save_consolidated=False,
            is_peft=False,
        )
        assert cfg.model_save_format.value == "torch_save"

    def test_invalid_format_raises(self):
        with pytest.raises(AssertionError, match="Unsupported model save format"):
            CheckpointingConfig(
                enabled=True,
                checkpoint_dir="/tmp/ckpt",
                model_save_format="invalid_format",
                model_cache_dir="/tmp/cache",
                model_repo_id="test",
                save_consolidated=False,
                is_peft=False,
            )

    def test_optional_fields_defaults(self):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="safetensors",
            model_cache_dir="/tmp/cache",
            model_repo_id="test",
            save_consolidated=False,
            is_peft=False,
        )
        assert cfg.is_async is False
        assert cfg.dequantize_base_checkpoint is None
        assert cfg.single_rank_consolidation is False
        assert cfg.staging_dir is None
        assert cfg.v4_compatible is False
        assert cfg.diffusers_compatible is False
        assert cfg.best_metric_key == "default"

    def test_importable_from_checkpointing(self):
        """Verify backward compat: import from checkpointing.py still works."""
        from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig as CkptCfg

        assert CkptCfg is CheckpointingConfig


class TestBuildCheckpointConfig:
    def test_build_with_defaults(self):
        from nemo_automodel.components.checkpoint.api import build_checkpoint_config

        cfg = build_checkpoint_config(
            checkpoint_kwargs=None,
            cache_dir="/tmp/cache",
            model_repo_id="test-model",
            is_peft=False,
        )
        assert isinstance(cfg, CheckpointingConfig)
        assert cfg.enabled is True
        assert str(cfg.model_cache_dir) == "/tmp/cache"
        assert cfg.model_repo_id == "test-model"

    def test_build_with_user_overrides(self):
        from nemo_automodel.components.checkpoint.api import build_checkpoint_config

        cfg = build_checkpoint_config(
            checkpoint_kwargs={"checkpoint_dir": "/my/ckpt", "v4_compatible": True},
            cache_dir=None,
            model_repo_id="test",
            is_peft=False,
        )
        assert str(cfg.checkpoint_dir) == "/my/ckpt"
        assert cfg.v4_compatible is True

    def test_build_peft_torch_save_fallback(self):
        """PEFT + torch_save should fallback to safetensors, preserving checkpoint_dir."""
        from nemo_automodel.components.checkpoint.api import build_checkpoint_config

        cfg = build_checkpoint_config(
            checkpoint_kwargs={"model_save_format": "torch_save", "checkpoint_dir": "/keep/this"},
            cache_dir=None,
            model_repo_id="test",
            is_peft=True,
        )
        assert cfg.model_save_format.value == "safetensors"
        assert str(cfg.checkpoint_dir) == "/keep/this"

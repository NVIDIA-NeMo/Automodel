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

"""Tests for CheckpointingConfig in config.py."""

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

    def test_peft_torch_save_coerces_format_but_preserves_other_fields(self):
        # PEFT + torch_save flips only model_save_format -> safetensors and
        # save_consolidated -> FINAL; all other user-set fields survive.
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="torch_save",
            model_cache_dir="/tmp/cache",
            model_repo_id="test",
            save_consolidated=False,
            is_peft=True,
            staging_dir="/tmp/staging",
            single_rank_consolidation=True,
            v4_compatible=True,
        )
        # Coerced (the two incompatible fields):
        assert cfg.model_save_format.value == "safetensors"
        assert cfg.save_consolidated.value == "final"
        # Preserved (everything else the user set) — the old builder hard-reset these to defaults.
        assert cfg.staging_dir == "/tmp/staging"
        assert cfg.single_rank_consolidation is True
        assert cfg.v4_compatible is True
        assert str(cfg.checkpoint_dir) == "/tmp/ckpt"

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

    def test_defaults_construct_without_args(self):
        """Every field has a default, so the recipe layer can construct directly."""
        from huggingface_hub import constants as hf_constants

        cfg = CheckpointingConfig()
        assert cfg.enabled is True
        assert str(cfg.checkpoint_dir) == "checkpoints/"
        assert cfg.model_save_format.value == "safetensors"
        # save_consolidated defaults to "final" and is normalized to SaveConsolidatedMode.FINAL.
        assert cfg.save_consolidated.value == "final"
        assert cfg.is_peft is False
        assert cfg.model_repo_id is None
        # model_cache_dir falls back to the HF hub cache when None.
        assert str(cfg.model_cache_dir) == str(hf_constants.HF_HUB_CACHE)

    def test_explicit_cache_dir_is_kept(self):
        cfg = CheckpointingConfig(model_cache_dir="/tmp/cache")
        assert str(cfg.model_cache_dir) == "/tmp/cache"

    def test_peft_torch_save_fallback(self):
        """PEFT + torch_save should fall back to safetensors, preserving checkpoint_dir."""
        cfg = CheckpointingConfig(
            checkpoint_dir="/keep/this",
            model_save_format="torch_save",
            is_peft=True,
        )
        assert cfg.model_save_format.value == "safetensors"
        # PEFT + torch_save resets save_consolidated to the FINAL default.
        assert cfg.save_consolidated.value == "final"
        assert str(cfg.checkpoint_dir) == "/keep/this"

    def test_non_peft_torch_save_preserved(self):
        """Without PEFT, torch_save must be honored."""
        cfg = CheckpointingConfig(model_save_format="torch_save", is_peft=False)
        assert cfg.model_save_format.value == "torch_save"

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

"""``DraftWandbSettings`` and ``DraftCheckpointDefaults``: the ``wandb:`` and ``checkpoint:`` blocks of the draft recipes."""

from __future__ import annotations

import pytest

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes._typed_config import DraftCheckpointDefaults, DraftWandbSettings

# --------------------------------------------------------------------------- #
# wandb
# --------------------------------------------------------------------------- #


def test_wandb_block_absent_gives_none():
    assert DraftWandbSettings.from_config(ConfigNode({"lr": 1e-4})) is None


def test_wandb_enable_false_gives_none():
    assert DraftWandbSettings.from_config(ConfigNode({"wandb": {"enable": False, "project": "p"}})) is None


def test_wandb_enable_is_stripped_and_the_rest_forwarded_as_written():
    raw = ConfigNode({"lr": 1e-4, "wandb": {"enable": True, "project": "p", "group": "g", "tags": ["a", "b"]}})
    settings = DraftWandbSettings.from_config(raw)
    assert settings.init_kwargs == {"project": "p", "group": "g", "tags": ["a", "b"]}
    assert settings.run_config == raw.to_dict()


def test_wandb_flag_defaults_to_enabled():
    settings = DraftWandbSettings.from_config(ConfigNode({"wandb": {"project": "p"}}))
    assert settings.init_kwargs == {"project": "p"}


def test_wandb_keys_outside_the_typed_dict_pass_through():
    settings = DraftWandbSettings.from_config(
        ConfigNode({"wandb": {"project": "p", "anonymous": "allow", "swanlab": {"mode": "local"}}})
    )
    assert settings.init_kwargs == {"project": "p", "anonymous": "allow", "swanlab": {"mode": "local"}}


def test_wandb_from_config_does_not_modify_the_node():
    raw = ConfigNode({"wandb": {"enable": True, "project": "p"}})
    before = raw.to_dict()
    DraftWandbSettings.from_config(raw)
    assert raw.to_dict() == before


# --------------------------------------------------------------------------- #
# checkpoint
# --------------------------------------------------------------------------- #


def _defaults(**overrides):
    values = {
        "checkpoint_dir": "/out/checkpoints",
        "model_repo_id": "org/target",
        "model_cache_dir": "/hub",
        "save_consolidated": True,
        "is_peft": False,
        "model_state_dict_keys": ["w", "b"],
    }
    return DraftCheckpointDefaults(**{**values, **overrides})


def test_checkpoint_without_a_block_uses_the_recipe_defaults():
    config = _defaults().resolve(None)
    assert isinstance(config, CheckpointingConfig)
    assert config.enabled is True
    assert str(config.checkpoint_dir) == "/out/checkpoints"
    assert config.model_save_format.value == "safetensors"
    assert config.model_repo_id == "org/target"
    assert str(config.model_cache_dir) == "/hub"
    assert config.is_peft is False
    assert config.model_state_dict_keys == ["w", "b"]


def test_checkpoint_user_block_overrides_the_defaults_and_restore_from_is_dropped():
    node = ConfigNode(
        {"checkpoint_dir": "/mine", "model_repo_id": "other/repo", "restore_from": "LATEST", "is_async": True}
    )
    config = _defaults().resolve(node)
    assert str(config.checkpoint_dir) == "/mine"
    assert config.model_repo_id == "other/repo"
    assert config.is_async is True
    assert not hasattr(config, "restore_from")


def test_checkpoint_null_state_dict_keys_fall_back_to_the_draft_keys():
    config = _defaults().resolve(ConfigNode({"model_state_dict_keys": None}))
    assert config.model_state_dict_keys == ["w", "b"]


def test_checkpoint_unknown_key_raises_type_error():
    with pytest.raises(TypeError, match="not_a_checkpoint_field"):
        _defaults().resolve(ConfigNode({"not_a_checkpoint_field": 1}))

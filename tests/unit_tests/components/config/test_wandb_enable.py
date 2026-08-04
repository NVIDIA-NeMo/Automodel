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
from __future__ import annotations

from nemo_automodel.components.config._arg_parser import _resolve_wandb_enable
from nemo_automodel.components.config.loader import ConfigNode


def test_wandb_enable_false_drops_section():
    """`enable: false` removes the wandb section so it reads as absent everywhere."""
    cfg = ConfigNode({"wandb": {"enable": False, "project": "p"}})

    _resolve_wandb_enable(cfg)

    assert "wandb" not in cfg
    assert cfg.get("wandb", None) is None
    assert not hasattr(cfg, "wandb")


def test_wandb_enable_true_keeps_section_and_strips_flag():
    """`enable: true` keeps the section but drops the flag (not a wandb.init kwarg)."""
    cfg = ConfigNode({"wandb": {"enable": True, "project": "p"}})

    _resolve_wandb_enable(cfg)

    assert cfg.get("wandb", None) is not None
    assert cfg.wandb.to_dict() == {"project": "p"}


def test_wandb_missing_enable_defaults_to_on():
    """A present block with no `enable` key logs by default (backward compatible)."""
    cfg = ConfigNode({"wandb": {"project": "p"}})

    _resolve_wandb_enable(cfg)

    assert cfg.get("wandb", None) is not None
    assert cfg.wandb.to_dict() == {"project": "p"}


def test_wandb_absent_is_noop():
    """No wandb block stays absent."""
    cfg = ConfigNode({"model": {}})

    _resolve_wandb_enable(cfg)

    assert "wandb" not in cfg

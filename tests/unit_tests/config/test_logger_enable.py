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

import pytest

from nemo_automodel.components.config._arg_parser import _resolve_logger_enable
from nemo_automodel.components.config.loader import ConfigNode

LOGGER_SECTIONS = ["wandb", "trackio"]


@pytest.mark.parametrize("section", LOGGER_SECTIONS)
def test_enable_false_drops_section(section):
    """`enable: false` removes the logger section so it reads as absent everywhere."""
    cfg = ConfigNode({section: {"enable": False, "project": "p"}})

    _resolve_logger_enable(cfg, section)

    assert section not in cfg
    assert cfg.get(section, None) is None
    assert not hasattr(cfg, section)


@pytest.mark.parametrize("section", LOGGER_SECTIONS)
def test_enable_true_keeps_section_and_strips_flag(section):
    """`enable: true` keeps the section but drops the flag (not an ``init()`` kwarg)."""
    cfg = ConfigNode({section: {"enable": True, "project": "p"}})

    _resolve_logger_enable(cfg, section)

    assert cfg.get(section, None) is not None
    assert getattr(cfg, section).to_dict() == {"project": "p"}


@pytest.mark.parametrize("section", LOGGER_SECTIONS)
def test_missing_enable_defaults_to_on(section):
    """A present block with no `enable` key logs by default (backward compatible)."""
    cfg = ConfigNode({section: {"project": "p"}})

    _resolve_logger_enable(cfg, section)

    assert cfg.get(section, None) is not None
    assert getattr(cfg, section).to_dict() == {"project": "p"}


@pytest.mark.parametrize("section", LOGGER_SECTIONS)
def test_absent_is_noop(section):
    """No logger block stays absent."""
    cfg = ConfigNode({"model": {}})

    _resolve_logger_enable(cfg, section)

    assert section not in cfg


def test_sections_are_independent():
    """Disabling one logger leaves the other untouched."""
    cfg = ConfigNode({"wandb": {"enable": False}, "trackio": {"project": "p"}})

    _resolve_logger_enable(cfg, "wandb")
    _resolve_logger_enable(cfg, "trackio")

    assert "wandb" not in cfg
    assert cfg.trackio.to_dict() == {"project": "p"}

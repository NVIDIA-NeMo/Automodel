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

"""
Validated config loading: Pydantic validation + ConfigNode for backwards compatibility.

Flow:
1. Load YAML -> raw dict
2. Apply CLI overrides -> raw dict
3. Validate raw dict against Pydantic schema (fail-fast!)
4. Wrap validated dict into ConfigNode (recipes continue to work unchanged)

This module provides the bridge between the new Pydantic validation layer
and the existing ConfigNode-based recipe system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from nemo_automodel.components.config.loader import ConfigNode, translate_value
from nemo_automodel.components.config.schema import validate_config

logger = logging.getLogger(__name__)


def _detect_recipe_type(raw_dict: dict[str, Any]) -> str:
    """Auto-detect recipe type from config contents.

    Heuristics:
    - Has 'teacher_model' -> 'kd'
    - Has 'flow_matching' or ('fsdp' and not 'distributed') -> 'diffusion'
    - Has 'processor' or 'freeze_config' -> 'vlm'
    - Has 'train_n_passages' or 'temperature' (with tokenizer at top level) -> 'biencoder'
    - Default -> 'llm'
    """
    if "teacher_model" in raw_dict:
        return "kd"
    if "flow_matching" in raw_dict:
        return "diffusion"
    if "fsdp" in raw_dict and "distributed" not in raw_dict:
        return "diffusion"
    if "processor" in raw_dict or "freeze_config" in raw_dict:
        return "vlm"
    if "train_n_passages" in raw_dict and "tokenizer" in raw_dict:
        return "biencoder"
    return "llm"


def _apply_overrides(raw_dict: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply dotted-path CLI overrides to a raw dict.

    Args:
        raw_dict: The raw YAML dict.
        overrides: List of "dotted.path=value" strings.

    Returns:
        The modified dict (mutated in place).
    """
    for kv in overrides:
        dotted, val_str = kv.split("=", 1)
        parts = dotted.split(".")
        node = raw_dict
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = translate_value(val_str)
    return raw_dict


def load_and_validate_config(
    path: str | Path,
    overrides: list[str] | None = None,
    recipe_type: str | None = None,
) -> ConfigNode:
    """Load a YAML config, validate it with Pydantic, then wrap in ConfigNode.

    This is the recommended entry point for validated config loading. It
    provides fail-fast validation while maintaining full backwards compatibility
    with the existing ConfigNode-based recipe system.

    Args:
        path: Path to the YAML config file.
        overrides: CLI overrides as "dotted.path=value" strings.
        recipe_type: Explicit recipe type ('llm', 'vlm', 'kd', 'diffusion', 'biencoder').
            If None, auto-detected from config contents.

    Returns:
        A ConfigNode wrapping the validated config.

    Raises:
        pydantic.ValidationError: If the config fails schema validation.
            The error message includes all validation errors with field paths.
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    if overrides:
        raw = _apply_overrides(raw, overrides)

    # Auto-detect recipe type if not specified
    if recipe_type is None:
        recipe_type = _detect_recipe_type(raw)

    # Validate against Pydantic schema (fail-fast!)
    validate_config(raw, recipe_type)
    logger.debug("Config validated successfully against %s schema", recipe_type)

    # Wrap in ConfigNode for backwards compatibility
    return ConfigNode(raw)


def validate_config_file(
    path: str | Path,
    overrides: list[str] | None = None,
    recipe_type: str | None = None,
) -> tuple[bool, str]:
    """Validate a config file without loading it into ConfigNode.

    Useful for --validate CLI flag: check config correctness without
    importing heavy dependencies or allocating GPUs.

    Args:
        path: Path to the YAML config file.
        overrides: CLI overrides.
        recipe_type: Explicit recipe type, or None for auto-detection.

    Returns:
        (is_valid, message) tuple. If invalid, message contains all errors.
    """
    from pydantic import ValidationError

    path = Path(path)
    if not path.exists():
        return False, f"Config file not found: {path}"

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, f"YAML parse error: {e}"

    if raw is None:
        raw = {}

    if overrides:
        try:
            raw = _apply_overrides(raw, overrides)
        except Exception as e:
            return False, f"Override error: {e}"

    if recipe_type is None:
        recipe_type = _detect_recipe_type(raw)

    try:
        validate_config(raw, recipe_type)
        return True, f"Config valid (recipe type: {recipe_type})"
    except ValidationError as e:
        lines = [f"Config validation failed ({e.error_count()} error(s)):"]
        for err in e.errors():
            loc = " -> ".join(map(str, err["loc"]))
            lines.append(f"  {loc}: {err['msg']}")
            if err.get("input") is not None:
                lines.append(f"    Got: {err['input']!r}")
        return False, "\n".join(lines)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def print_resolved_config(
    path: str | Path,
    overrides: list[str] | None = None,
) -> None:
    """Print the resolved config after applying overrides.

    Useful for --show-config CLI flag.
    """
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if overrides:
        raw = _apply_overrides(raw, overrides)

    print(yaml.safe_dump(raw, sort_keys=False, default_flow_style=False).strip())

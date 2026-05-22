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

"""Resolve a CI test config from a recipe + config_templates/ci_config.yaml.

See ci_config.yaml for the layered resolution stack and override schema.

CLI:
    --base       Recipe YAML
    --phase      nightly | release | convergence | performance | checkpoint_robustness
    --output     Resolved YAML path (required unless --dry-run)
    --dry-run    Print the merge stack to stdout instead of writing
"""

import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

CI_CONFIG_FILE = Path(__file__).resolve().parent / "config_templates" / "ci_config.yaml"

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True


def _load(path: Path) -> dict:
    """Load a YAML file and return its top-level mapping (empty dict if file is empty)."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.load(f) or {}


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """Write `value` at the dotted-key path in `d`, creating intermediate dicts as needed."""
    keys = dotted_key.split(".")
    cursor = d
    for k in keys[:-1]:
        if not isinstance(cursor.get(k), dict):
            cursor[k] = {}
        cursor = cursor[k]
    cursor[keys[-1]] = value


def _coerce(value: str) -> Any:
    """Best-effort coerce env strings to int/float; fall back to str."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _resolve_env_layer(env_section: dict, phase: str) -> dict[str, Any]:
    """Return env-driven overrides whose env var is set and whose `phases` filter matches."""
    out: dict[str, Any] = {}
    for env_var, spec in (env_section or {}).items():
        target = spec["target"]
        phases = spec.get("phases")
        if phases is not None and phase not in phases:
            continue
        if env_var in os.environ:
            out[target] = _coerce(os.environ[env_var])
    return out


def _format_or_passthrough(value: Any, substitutions: dict[str, Any], context: str) -> Any:
    """Apply str.format if value is a string; pass through other types unchanged."""
    if not isinstance(value, str):
        return value
    try:
        return value.format(**substitutions)
    except KeyError as missing:
        sys.exit(f"ERROR: {context} missing substitution {missing} (template={value!r})")


def _resolve_computed_layer(computed_entries: list, phase: str) -> dict[str, Any]:
    """Return per-run dynamic overrides with env + `date` interpolated into each format string."""
    substitutions: dict[str, Any] = {"date": datetime.now(), **os.environ}
    out: dict[str, Any] = {}
    for entry in computed_entries or []:
        phases = entry.get("phases")
        if phases is not None and phase not in phases:
            continue
        out[entry["target"]] = _format_or_passthrough(
            entry["format"], substitutions, f"computed override target={entry['target']!r}"
        )
    return out


def _resolve_conditional_layer(conditional_entries: list, phase: str, recipe_stem: str) -> dict[str, Any]:
    """Return overrides for entries whose phase and recipe-name predicates both match."""
    substitutions: dict[str, Any] = {"date": datetime.now(), **os.environ}
    out: dict[str, Any] = {}
    for entry in conditional_entries or []:
        phases = entry.get("phases")
        if phases is not None and phase not in phases:
            continue
        contains_all = entry.get("when_recipe_contains_all") or []
        contains_any = entry.get("when_recipe_contains_any") or []
        if contains_all and not all(sub in recipe_stem for sub in contains_all):
            continue
        if contains_any and not any(sub in recipe_stem for sub in contains_any):
            continue
        for target, raw_value in (entry.get("apply") or {}).items():
            out[target] = _format_or_passthrough(
                raw_value, substitutions, f"conditional override target={target!r}"
            )
    return out


def _print_layer(label: str, payload: dict[str, Any]) -> None:
    """Render one layer of the resolution stack for --dry-run output."""
    if not payload:
        print(f"  [{label}] (empty)")
        return
    print(f"  [{label}]")
    for k, v in payload.items():
        print(f"    {k} = {v!r}")


def main() -> int:
    """Build each layer, merge onto the recipe, then write or dry-print the result."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True, help="Base recipe YAML")
    parser.add_argument("--phase", required=True, help="Phase key under ci_config.yaml.phases")
    parser.add_argument("--output", type=Path, help="Where to write the resolved YAML (omit for --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="Print merge stack + resolved YAML to stdout; no write")
    args = parser.parse_args()

    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is set")
    if not args.base.is_file():
        sys.exit(f"ERROR: base recipe not found: {args.base}")

    recipe = _load(args.base)
    ci_config = _load(CI_CONFIG_FILE)

    # Shared phase defaults from ci_config.yaml -> phases[<phase>].
    phase_defaults = (ci_config.get("phases") or {}).get(args.phase) or {}
    # Recipe-name-pattern overrides from ci_config.yaml -> conditional[].
    conditional_layer = _resolve_conditional_layer(
        ci_config.get("conditional") or [], args.phase, args.base.stem
    )
    # Per-recipe overrides from recipe.ci.<phase>; fixture-arg keys (read by
    # consumers like pytest, listed under ci_config.fixture_keys) are skipped.
    ci_section_raw = (recipe.get("ci") or {}).get(args.phase) or {}
    fixture_keys = set((ci_config.get("fixture_keys") or {}).get(args.phase) or [])
    ci_section = {k: v for k, v in ci_section_raw.items() if k not in fixture_keys}
    # Env-driven overrides from ci_config.yaml -> env[].
    env_layer = _resolve_env_layer(ci_config.get("env") or {}, args.phase)
    # Per-run dynamic values from ci_config.yaml -> computed[].
    computed_layer = _resolve_computed_layer(ci_config.get("computed") or [], args.phase)

    layers = [
        ("phase_defaults", phase_defaults),
        ("conditional", conditional_layer),
        ("recipe.ci." + args.phase, ci_section),
        ("env", env_layer),
        ("computed", computed_layer),
    ]

    config = copy.deepcopy(recipe)
    for _, payload in layers:
        for dotted_key, value in payload.items():
            _set_dotted(config, dotted_key, value)

    if args.dry_run:
        print(f"Resolution stack for --phase {args.phase}:")
        print(f"  [base] {args.base}")
        for label, payload in layers:
            _print_layer(label, payload)
        print("--- resolved config ---")
        yaml.dump(config, sys.stdout)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print(args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())

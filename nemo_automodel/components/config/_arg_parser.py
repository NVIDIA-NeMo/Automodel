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

import sys

from nemo_automodel.components.config.loader import ConfigNode, load_yaml_config, translate_value

_USAGE = """\
Usage: <script> --config <path.yaml> [OPTIONS] [--key.path value ...]

Required:
  --config, -c <path>     Path to YAML config file

Options:
  --help, -h              Show this help message and exit
  --validate              Validate config against schema without running training
  --show-config           Print resolved config (after overrides) and exit
  --show-schema [TYPE]    Print JSON Schema for a recipe type and exit
                          Types: llm (default), vlm, kd, diffusion, biencoder
  --recipe-type TYPE      Explicit recipe type for validation (default: auto-detect)

Config overrides:
  --key.path value        Override a config value, e.g. --optimizer.lr 1e-4
  --key.path=value        Same, using = syntax
  --flag                  Set a boolean flag to True

Examples:
  python finetune.py -c config.yaml
  python finetune.py -c config.yaml --optimizer.lr 1e-4 --step_scheduler.num_epochs 5
  python finetune.py -c config.yaml --validate
  python finetune.py --show-schema llm
"""


def _print_help_and_exit() -> None:
    print(_USAGE)
    sys.exit(0)


def _handle_show_schema(argv: list[str]) -> None:
    """Handle --show-schema [TYPE] and exit."""
    from nemo_automodel.components.config.schema import get_schema_json

    # Find recipe type argument (if any)
    recipe_type = "llm"
    idx = None
    for i, tok in enumerate(argv):
        if tok == "--show-schema":
            idx = i
            break
    if idx is not None and idx + 1 < len(argv) and not argv[idx + 1].startswith("--"):
        recipe_type = argv[idx + 1]

    try:
        print(get_schema_json(recipe_type))
    except ValueError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    sys.exit(0)


def _handle_validate(cfg_path: str, overrides: list[str], recipe_type: str | None) -> None:
    """Handle --validate and exit."""
    from nemo_automodel.components.config.validated_config import validate_config_file

    is_valid, message = validate_config_file(cfg_path, overrides, recipe_type)
    if is_valid:
        print(f"\u2713 {message}")
        sys.exit(0)
    else:
        print(message, file=sys.stderr)
        sys.exit(1)


def _handle_show_config(cfg_path: str, overrides: list[str]) -> None:
    """Handle --show-config and exit."""
    from nemo_automodel.components.config.validated_config import print_resolved_config

    print_resolved_config(cfg_path, overrides)
    sys.exit(0)


def parse_cli_argv(default_cfg_path: str | None = None) -> tuple[str, list[str]]:
    """
    Parses CLI args, pulls out --config and collects other --dotted.path options.

    Args:
        default_cfg_path (str, optional): Default config (yaml) path. Defaults to None.

    Raises:
        ValueError: if there's no --config and cfg_path = None
        ValueError: if there's --config but not yaml file passed

    Returns:
        (str, list[str]): the config path along with the config overrides.
    """
    argv = sys.argv[1:]

    # Quick check for help
    if "--help" in argv or "-h" in argv:
        _print_help_and_exit()

    # Quick check for --show-schema (doesn't require --config)
    if "--show-schema" in argv:
        _handle_show_schema(argv)

    overrides = []
    i = 0
    cfg_path = default_cfg_path
    do_validate = False
    do_show_config = False
    recipe_type = None
    while i < len(argv):
        tok = argv[i]

        # --config or -c
        if tok in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            cfg_path = argv[i + 1]
            i += 2
            continue

        # Special flags
        if tok == "--validate":
            do_validate = True
            i += 1
            continue
        if tok == "--show-config":
            do_show_config = True
            i += 1
            continue
        if tok == "--recipe-type":
            if i + 1 >= len(argv):
                raise ValueError("Expected a recipe type after --recipe-type")
            recipe_type = argv[i + 1]
            i += 2
            continue

        # any other --option or --dotted.path
        if tok.startswith("--"):
            key = tok.lstrip("-")
            # case A) --key=val
            if "=" in key:
                overrides.append(f"{key}")
                i += 1
            else:
                # case B) --key val
                # if next token exists and isn't another --..., take it as the value
                if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                    val = argv[i + 1]
                    overrides.append(f"{key}={val}")
                    i += 2
                else:
                    # case C) bare --flag => True
                    overrides.append(f"{key}=True")
                    i += 1
            continue

        # anything else we skip
        i += 1

    if not cfg_path:
        raise ValueError("You must specify --config <path>. Use --help for usage.")

    # Handle special modes before returning
    if do_validate:
        _handle_validate(cfg_path, overrides, recipe_type)
    if do_show_config:
        _handle_show_config(cfg_path, overrides)

    return cfg_path, overrides


def parse_args_and_load_config(
    default_cfg_path: str | None = None,
    validate: bool = True,
    recipe_type: str | None = None,
) -> ConfigNode:
    """
    Loads YAML, optionally validates with Pydantic, applies overrides via ConfigNode.set_by_dotted.

    Args:
        default_cfg_path: Default config file path (used if --config not provided).
        validate: If True, validate config against Pydantic schema before loading.
            Validation errors are printed as warnings but do not block loading,
            preserving backwards compatibility.
        recipe_type: Explicit recipe type for validation. None = auto-detect.
    """
    cfg_path, overrides = parse_cli_argv(default_cfg_path)
    print(f"cfg-path: {cfg_path}")

    # Optionally validate with Pydantic (fail-fast for config errors)
    if validate:
        try:
            from nemo_automodel.components.config.validated_config import validate_config_file

            is_valid, message = validate_config_file(cfg_path, overrides, recipe_type)
            if is_valid:
                print(f"\u2713 {message}")
            else:
                print(f"WARNING: {message}", file=sys.stderr)
        except Exception:
            pass  # Don't block loading if validation itself fails

    # load the base YAML
    cfg = load_yaml_config(cfg_path)

    # apply overrides
    for kv in overrides:
        dotted, val_str = kv.split("=", 1)
        cfg.set_by_dotted(dotted, translate_value(val_str))

    return cfg

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


def parse_cli_argv(default_cfg_path: str | None = None, argv: list[str] | None = None) -> tuple[str, list[str]]:
    """
    Parses CLI args, pulls out --config and collects other --dotted.path options.

    Args:
        default_cfg_path (str, optional): Default config (yaml) path. Defaults to None.
        argv (list[str], optional): Argument list to parse. Defaults to ``sys.argv[1:]``.

    Raises:
        ValueError: if there's no --config and cfg_path = None
        ValueError: if there's --config but not yaml file passed

    Returns:
        (str, list[str]): the config path along with the config overrides.
    """
    if argv is None:
        argv = sys.argv[1:]
    overrides = []
    i = 0
    cfg_path = default_cfg_path
    while i < len(argv):
        tok = argv[i]

        # --config or -c
        if tok in ("--config", "-c"):
            if i + 1 >= len(argv):
                raise ValueError("Expected a path after --config")
            cfg_path = argv[i + 1]
            i += 2
            continue

        # --config=<path>
        if tok.startswith("--config="):
            cfg_path = tok.split("=", 1)[1]
            if not cfg_path:
                raise ValueError("Expected a path after --config")
            i += 1
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
        raise ValueError("You must specify --config <path>")
    return cfg_path, overrides


def parse_args_and_load_config(default_cfg_path: str | None = None, argv: list[str] | None = None) -> ConfigNode:
    """
    Load YAML and apply CLI overrides, including opt-in ``--benchmark`` controls.

    Benchmark defaults come from ``ci.perf_testing`` and the Core Models measurement
    budget. Explicit dotted CLI overrides take precedence over those defaults.

    Args:
        default_cfg_path: Default YAML configuration path.
        argv: Explicit arguments, or ``sys.argv[1:]`` when omitted.

    Returns:
        Loaded configuration with benchmark defaults and explicit overrides applied.
        argv (list[str], optional): Argument list to parse. Defaults to ``sys.argv[1:]``.
    """
    cfg_path, overrides = parse_cli_argv(default_cfg_path, argv=argv)
    print(f"cfg-path: {cfg_path}")
    # load the base YAML
    cfg = load_yaml_config(cfg_path)

    benchmark = "benchmark=True" in overrides
    explicit_keys: set[str] = set()
    # Apply explicit overrides before reading performance metadata, so callers may
    # opt into validation or change the fake-gate choice on the command line too.
    for kv in overrides:
        if kv == "benchmark=True":
            continue
        dotted, val_str = kv.split("=", 1)
        cfg.set_by_dotted(dotted, translate_value(val_str))
        explicit_keys.add(dotted)

    if benchmark:
        _apply_benchmark_overrides(cfg, explicit_keys=explicit_keys)

    # Resolve the optional `wandb.enable` / `trackio.enable` toggles after overrides, so that
    # `wandb.enable=true` passed on the CLI is honored.
    _resolve_logger_enable(cfg, "wandb")
    _resolve_logger_enable(cfg, "trackio")

    return cfg


def _apply_benchmark_overrides(cfg: ConfigNode, *, explicit_keys: set[str]) -> None:
    """Apply performance measurement defaults without replacing canonical workload settings.

    Args:
        cfg: Loaded configuration, updated in place at the shared YAML boundary.
        explicit_keys: Dotted CLI settings that must retain their explicit values.

    Raises:
        ValueError: If ``ci.perf_testing`` is missing, malformed, or contains unsupported settings.
    """
    perf = cfg.get("ci.perf_testing")
    if not isinstance(perf, ConfigNode):
        raise ValueError("--benchmark requires a ci.perf_testing mapping in the canonical training YAML")
    unknown = set(perf.to_yaml_dict()) - {"schedule", "time", "fake_balanced_gate", "validation"}
    if unknown:
        raise ValueError(f"ci.perf_testing contains unsupported settings: {sorted(unknown)}")
    for key in ("fake_balanced_gate", "validation"):
        if key in perf and not isinstance(perf.get(key), bool):
            raise ValueError(f"ci.perf_testing.{key} must be a boolean")

    defaults: dict[str, bool | int | None] = {
        "checkpoint.enabled": False,
        "step_scheduler.max_steps": 50,
        "step_scheduler.num_epochs": None,
    }
    if not perf.get("validation", False):
        defaults["step_scheduler.val_every_steps"] = None
        defaults["step_scheduler.validate_on_checkpoint"] = False
    if "fake_balanced_gate" in perf:
        defaults["model.backend.fake_balanced_gate"] = perf.fake_balanced_gate
    for key, value in defaults.items():
        if not any(key == explicit or key.startswith(explicit + ".") for explicit in explicit_keys):
            cfg.set_by_dotted(key, value)


def _resolve_logger_enable(cfg: ConfigNode, section: str) -> None:
    """Apply the optional ``<section>.enable`` toggle of a logger section in place.

    A present ``wandb:`` / ``trackio:`` block enables that logger by default
    (backward compatible). Setting ``enable: false`` disables it: the whole
    section is dropped so every downstream presence check
    (``cfg.wandb``, ``cfg.get("wandb")``, ``hasattr(cfg, "wandb")``) consistently
    sees it as absent. The flag itself is always stripped so it is never
    forwarded to ``wandb.init()`` / ``trackio.init()`` as an unknown keyword argument.

    Args:
        cfg: The loaded recipe configuration. Mutated in place.
        section: Logger section name, ``"wandb"`` or ``"trackio"``.
    """
    node = cfg.get(section, None)
    if not isinstance(node, ConfigNode):
        return
    enabled = bool(node.get("enable", True))
    node.__dict__.pop("enable", None)
    if not enabled:
        cfg.__dict__.pop(section, None)

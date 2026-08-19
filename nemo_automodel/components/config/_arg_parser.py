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
    Loads YAML, applies overrides via ConfigNode.set_by_dotted.

    Args:
        default_cfg_path (str, optional): Default config (yaml) path.
        argv (list[str], optional): Argument list to parse. Defaults to ``sys.argv[1:]``.
    """
    cfg_path, overrides = parse_cli_argv(default_cfg_path, argv=argv)
    print(f"cfg-path: {cfg_path}")
    # load the base YAML
    cfg = load_yaml_config(cfg_path)

    # apply overrides
    for kv in overrides:
        dotted, val_str = kv.split("=", 1)
        cfg.set_by_dotted(dotted, translate_value(val_str))

    # Resolve the optional `wandb.enable` toggle after overrides, so that
    # `wandb.enable=true` passed on the CLI is honored.
    _resolve_wandb_enable(cfg)

    return cfg


def _resolve_wandb_enable(cfg: ConfigNode) -> None:
    """Apply the optional ``wandb.enable`` toggle in place.

    A present ``wandb:`` block enables Weights & Biases logging by default
    (backward compatible). Setting ``enable: false`` disables it: the whole
    ``wandb`` section is dropped so every downstream presence check
    (``cfg.wandb``, ``cfg.get("wandb")``, ``hasattr(cfg, "wandb")``) consistently
    sees it as absent. The flag itself is always stripped so it is never
    forwarded to ``wandb.init()`` as an unknown keyword argument.

    Args:
        cfg: The loaded recipe configuration. Mutated in place.
    """
    node = cfg.get("wandb", None)
    if not isinstance(node, ConfigNode):
        return
    enabled = bool(node.get("enable", True))
    node.__dict__.pop("enable", None)
    if not enabled:
        cfg.__dict__.pop("wandb", None)

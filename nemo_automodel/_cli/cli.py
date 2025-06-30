#!/usr/bin/env python3
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
import argparse
import sys
from pathlib import Path
import yaml
import os
import logging

from pathlib import Path
import importlib.util
import types

def load_function(file_path: str | Path, func_name: str):
    """
    Dynamically import `func_name` from the file at `file_path`
    and return a reference to that function.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    module_name = file_path.stem # arbitrary, unique per load is fine
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module) # executes the file, populating `module`

    try:
        return getattr(module, func_name)
    except AttributeError:
        raise ImportError(f"{func_name} not found in {file_path}")


def load_yaml(path):
    try:
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")

def launch_with_slurm(slurm_config, script_path, config_file):
    import nemo_run as run
    executor = run.SlurmExecutor(**slurm_config, tunnel=run.LocalTunnel())
    with run.Experiment('aaa') as exp:
        run_name = 'exp_name'
        exp.add(
            run.Script(
                path=script_path,
                args=[
                    "--config",
                    config_file,
                ],
                env=container_env,
                entrypoint="python",
            ),
            executor=executor,
            name=run_name[:37],  # DGX-C run name length limit
            tail_logs=False
        )
        exp.run(sequential=True, detach=True, tail_logs=False)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="automodel",
        description="""CLI for NeMo AutoModel
        Finetune any HF model

        Usage:
        automodel <domain> <command> --config /path/to/conf.yaml"

        where:
        - domain: llm, vlm, etc
        - command: finetune
        """
    )

    # Two required positionals (cannot start with "--")
    domain_choices = list(map(lambda x: x.name, (Path(__file__).parents[2] / "recipes").iterdir()))
    parser.add_argument(
        "domain",
        metavar="<domain>",
        choices=domain_choices,
        help="Model domain to operate on (e.g., LLM, VLM, etc)",
    )
    parser.add_argument(
        "command",
        metavar="<command>",
        choices=['finetune'],
        help="Command within the domain (e.g., finetune, deploy, etc)",
    )

    # Optional/required flag
    parser.add_argument(
        "-c",
        "--config",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path to YAML configuration file",
    )

    return parser


def main():
    args = build_parser().parse_args()
    print(f"Domain:  {args.domain}")
    print(f"Command: {args.command}")
    print(f"Config:  {args.config.resolve()}")
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    script_path = Path(__file__).parents[2] / "recipes" / args.domain / f'{args.command}.py'

    if 'slurm' in config:
        # launch job on kubernetes
        launch_with_slurm(config['slurm'], script_path, config_path)
    elif 'k8s' in config or 'kubernetes' in config:
        raise NotImplementedError("WIP")
        # launch job on kubernetes
    else:
        recipe_main = load_function(script_path, "main")
        return recipe_main(config_path)

if __name__ == "__main__":
    main()

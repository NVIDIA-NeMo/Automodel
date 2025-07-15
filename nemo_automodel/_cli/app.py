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
import importlib.util
import logging
import os
import time
from pathlib import Path

import yaml
logging.getLogger().setLevel(logging.INFO)


# Here we assume the following directory structure and expect it to remain unchanged.
#
# ├── nemo_automodel
# │   ├── __init__.py
# │   ├── _cli
# │   │   └── app.py
# ├── examples
#     ├── llm
#     │   ├── finetune.py
#     │   ├── llama_3_2_1b_hellaswag.yaml
#     │   ├── ...
#     │   └── llama_3_2_1b_squad_slurm.yaml
#     └── vlm
#         ├── finetune.py
#         ├── gemma_3_vl_3b_cord_v2.yaml
#         ├── ...
#         └── qwen2_5_vl_3b_rdr.yaml


def load_function(file_path: str | Path, func_name: str):
    """
    Dynamically import `func_name` from the file at `file_path`
    and return a reference to that function.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    module_name = file_path.stem  # arbitrary, unique per load is fine
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # executes the file, populating `module`

    try:
        return getattr(module, func_name)
    except AttributeError:
        raise ImportError(f"{func_name} not found in {file_path}")


def load_yaml(file_path):
    """
    Loads a yaml file.

    Args:
        file_path (str): Path to yaml file.

    Returns:
        dict: the yaml file's contents

    Raise:
        FileNotFoundError: if the file does not exist
        yaml.YAMLError: if the file is incorrectly formatted.
    """
    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError as e:
        logger.error(f"File '{file_path}' was not found.")
        raise e
    except yaml.YAMLError as e:
        logger.error(f"parsing YAML file {e} failed.")
        raise e


def launch_with_slurm_nemorun(slurm_config, script_path, config_file, job_name="llm_finetune", job_dir=None, container_env={}):
    """
    Launches a Slurm job using NeMo-Run's SlurmExecutor

    Args:
        slurm_config (dict): the slurm config
        script_path (str): the path to the recipe script (e.g., examples/llm/finetune.py)
        config_file (str): the path to the config yaml (e.g., examples/llm/llama_3_2_1b_squad.yaml)
        container_env (str, optional): The container env. Defaults to None.
    """
    raise NotImplementedError("Slurm support via nemo-run is pending")
    assert isinstance(job_dir, str), "Expected job_dir to be a string"
    import nemo_run as run

    if not "mem" in slurm_config:
        slurm_config["mem"] = "0"
    if not "exclusive" in slurm_config:
        slurm_config["exclusive"] = True

    from nemo_run.config import set_nemorun_home

    set_nemorun_home(job_dir)
    executor = run.SlurmExecutor(
        **slurm_config,
        tunnel=run.LocalTunnel(job_dir=''),
        packager=None,
    )
    # @akoumparouli: uncomment once nemo-run updates its package.
    # with run.Experiment('exp_ts_', enable_goodbye_message=False) as exp:
    with run.Experiment("") as exp:
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
            name=job_name[:37],  # DGX-C run name length limit
            tail_logs=False,
        )
        exp.run(sequential=True, detach=True, tail_logs=False)

def launch_with_slurm(slurm_config):
    from nemo_automodel.components.launcher.slurm.config import SlurmConfig, VolumeMapping
    from nemo_automodel.components.launcher.slurm.utils import submit_slurm_job
    # if there's no `job_dir` in the slurm section, use cwd/slurm_job
    job_dir = slurm_config.pop("job_dir", None)
    if job_dir is None:
        job_dir = os.path.join(os.getcwd(), "slurm_job", str(int(time.time())))
    os.makedirs(job_dir, exist_ok=True)

    # Write job's config
    job_conf_path = os.path.join(job_dir, "job_config.yaml")
    with open(job_conf_path, "w") as fp:
        yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
    logging.info(f'Logging Slurm job in: {job_dir}')

    # Determine the code repo root
    if 'repo_root' in config:
        repo_root = config.pop('repo_root')
    else:
        cwd = Path.cwd()
        if (cwd / "nemo_automodel/components").exists() and (cwd / "examples/").exists():
            repo_root = str(cwd)
        else:
            repo_root = '/opt/Automodel'
    logging.info(f"Using {repo_root} as code repo")

    # create the command
    command = f'PYTHONPATH={repo_root}:$PYTHONPATH python3 '\
        f'{repo_root}/nemo_automodel/recipes/{args.domain}/{args.command}.py -c {job_conf_path}'

    slurm_config['extra_mounts'].append(VolumeMapping(Path(repo_root), Path(repo_root)))
    if slurm_config.get('job_name', '') == '':
        slurm_config['job_name'] = f'{args.domain}_{args.command}'
    return submit_slurm_job(
        SlurmConfig(**slurm_config, command=command, chdir=repo_root), job_dir)

def build_parser() -> argparse.ArgumentParser:
    """
    Builds a parser with automodel's app options

    Returns:
        argparse.ArgumentParser: the parser.
    """
    parser = argparse.ArgumentParser(prog="automodel", description="CLI for NeMo AutoModel examples")

    # Two required positionals (cannot start with "--")
    parser.add_argument(
        "domain",
        metavar="<domain>",
        choices=["llm", "vlm"],
        help="Domain to operate on (e.g., LLM, VLM, etc)",
    )
    parser.add_argument(
        "command",
        metavar="<command>",
        choices=["finetune"],
        help="Command within the domain (e.g., finetune, generate, etc)",
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
    # This is defined in torch.distributed.run's parser, but we also define it here.
    # We want to determine if the user passes `--nproc-per-node` via CLI. In particular, we
    # want to use this information to determine whether they want to utilize a subset of the
    # currently available devices in their job, otherwise it'll automatically opt to use all devices
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        default=None,
        help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
    )
    return parser


def main():
    """CLI for running finetune jobs with NeMo-Automodel, supporting torchrun, Slurm & Kubernetes.

    Raises:
        NotImplementedError: if yaml has a k8s section (support is WIP).

    Returns:
        int: Job's status code
    """
    args = build_parser().parse_args()
    logging.info(f"Domain:  {args.domain}")
    logging.info(f"Command: {args.command}")
    logging.info(f"Config:  {args.config.resolve()}")
    config_path = args.config.resolve()
    config = load_yaml(config_path)

    if slurm_config := config.pop("slurm", None):
        logging.info("Launching job via SLURM")
        return launch_with_slurm(slurm_config)
    elif "k8s" in config or "kubernetes" in config:
        # launch job on kubernetes.
        raise NotImplementedError("kubernetes support is pending")
    else:
        from torch.distributed.run import determine_local_world_size, get_args_parser
        from torch.distributed.run import run as thrun
        script_path = Path(__file__).parent / "recipes" / args.domain / f"{args.command}.py"

        # launch job on this node
        num_devices = determine_local_world_size(nproc_per_node="gpu")
        assert num_devices > 0, "Expected num-devices to be > 0"
        if args.nproc_per_node == 1 or num_devices == 1:
            logger.info("Launching job locally on a single device")
            # run the job with a single rank on this process.
            recipe_main = load_function(script_path, "main")
            return recipe_main(config_path)
        else:
            logger.info(f"Launching job locally on {num_devices} devices")
            # run the job on multiple ranks on this node.
            torchrun_parser = get_args_parser()
            torchrun_args = torchrun_parser.parse_args()
            # overwrite the training script with the actual recipe path
            torchrun_args.training_script = str(script_path)
            # training_script_args=['finetune', '--config', 'examples/llm/llama_3_2_1b_squad.yaml']
            # remove the command (i.e., "finetune") part.
            torchrun_args.training_script_args.pop(0)
            tmp = str(args.config)
            for i in range(len(torchrun_args.training_script_args)):
                if torchrun_args.training_script_args[i] == tmp:
                    torchrun_args.training_script_args[i] = str(config_path)
                    break
            if args.nproc_per_node is None:
                torchrun_args.nproc_per_node = num_devices
            return thrun(torchrun_args)


if __name__ == "__main__":
    main()

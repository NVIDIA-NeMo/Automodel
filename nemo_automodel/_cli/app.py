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
        logging.error(f"File '{file_path}' was not found.")
        raise e
    except yaml.YAMLError as e:
        logging.error(f"parsing YAML file {e} failed.")
        raise e


def get_automodel_repo_root():
    """
    if the cwd contains:
    - nemo_automodel/components
    - examples/
    will return cwd, otherwise will return None
    """
    cwd = Path.cwd()
    if (cwd / "nemo_automodel/components").exists() and (cwd / "examples/").exists():
        return cwd
    return None


def launch_with_slurm(args, job_conf_path, job_dir, slurm_config):
    from nemo_automodel.components.launcher.slurm.config import SlurmConfig, VolumeMapping
    from nemo_automodel.components.launcher.slurm.utils import submit_slurm_job

    last_dir = Path(job_dir).parts[-1]
    assert len(last_dir) == 10 and last_dir.isdigit(), ("Expected last dir to be unix timestamp", job_dir)
    # hf_home needs to be on shared shorage for multinode jobs.
    if not "hf_home" in slurm_config:
        # we'll assume that job_dir is on shared storage (visible by all SLURM workers).
        slurm_config["hf_home"] = str(Path(job_dir).parent / ".hf_home")
        os.makedirs(slurm_config["hf_home"], exist_ok=True)

    # log HF_HOME used.
    logging.info(f"Using HF_HOME= `{slurm_config['hf_home']}`")

    # Determine the code repo root
    if "repo_root" in slurm_config:
        repo_root = slurm_config.pop("repo_root")
        logging.info(f"Running job using source defined in yaml: {repo_root}")
    else:
        if repo_root := get_automodel_repo_root():
            repo_root = str(repo_root)
            logging.info(f"Running job using source from: {repo_root}")
        else:
            repo_root = "/opt/Automodel"
    logging.info(f"Using {repo_root} as code repo")

    # Make default name
    if slurm_config.get("job_name", "") == "":
        slurm_config["job_name"] = f"{args.domain}_{args.command}"

    # create the command
    command = " ".join(
        (
            f"PYTHONPATH={repo_root}:$PYTHONPATH",
            "python3",
            f"{repo_root}/nemo_automodel/recipes/{args.domain}/{args.command}.py",
            "-c",
            f"{job_conf_path}",
        )
    )
    # Add extra mounts
    if not "extra_mounts" in slurm_config:
        slurm_config["extra_mounts"] = []
    slurm_config["extra_mounts"].append(VolumeMapping(Path(repo_root), Path(repo_root)))
    return submit_slurm_job(SlurmConfig(**slurm_config, command=command, chdir=repo_root), job_dir)


def build_parser() -> argparse.ArgumentParser:
    """
    Builds a parser with automodel's app options

    Returns:
        argparse.ArgumentParser: the parser.
    """
    parser = argparse.ArgumentParser(prog="automodel", description="CLI for NeMo AutoModel examples")

    # Two required positionals (cannot start with "--")
    parser.add_argument(
        "command",
        metavar="<command>",
        choices=["finetune"],
        help="Command within the domain (e.g., finetune, generate, etc)",
    )
    parser.add_argument(
        "domain",
        metavar="<domain>",
        choices=["llm", "vlm"],
        help="Domain to operate on (e.g., LLM, VLM, etc)",
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


def get_repo_root():
    """
    Returns the repo root to use and if using non-default, it will modify $PYTHONPATH accordingly

    Returns:
        Path: the repo root path
    """
    if repo_root := get_automodel_repo_root():
        new_pp = str(repo_root)
        if "PYTHONPATH" in os.environ:
            new_pp += ":" + os.environ["PYTHONPATH"]
        os.environ["PYTHONPATH"] = new_pp
        logging.info(f"Running job using source from: {repo_root}")
    else:
        repo_root = Path(__file__).parents[2]
    return repo_root


def run_interactive(args):
    import torch
    from torch.distributed.run import determine_local_world_size, get_args_parser
    from torch.distributed.run import run as thrun

    config_path = args.config.resolve()
    repo_root = get_repo_root()
    script_path = repo_root / "nemo_automodel" / "recipes" / args.domain / f"{args.command}.py"

    # launch job on this node
    device_type = "gpu" if torch.cuda.is_available() else "cpu"
    num_devices = determine_local_world_size(nproc_per_node=device_type)
    assert num_devices > 0, "Expected num-devices to be > 0"
    if args.nproc_per_node == 1 or num_devices == 1:
        logging.info(f"Launching job locally on a single {device_type.upper()}")
        # run the job with a single rank on this process.
        recipe_main = load_function(script_path, "main")
        return recipe_main(config_path)
    else:
        logging.info(f"Launching job locally on {num_devices} {device_type.upper()}s")
        # run the job on multiple ranks on this node.
        torchrun_parser = get_args_parser()
        torchrun_args, extra = torchrun_parser.parse_known_args()
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


def main():
    """CLI for running finetune jobs with NeMo-Automodel, supporting torchrun, Slurm & Kubernetes.

    Raises:
        NotImplementedError: if yaml has a k8s section (support is WIP).

    Returns:
        int: Job's status code
    """
    args, extra = build_parser().parse_known_args()
    logging.info(f"Domain:  {args.domain}")
    logging.info(f"Command: {args.command}")
    logging.info(f"Config:  {args.config.resolve()}")
    config_path = args.config.resolve()
    config = load_yaml(config_path)

    if slurm_config := config.pop("slurm", None):
        logging.info("Launching job via SLURM")
        # if there's no `job_dir` in the slurm section, use cwd/slurm_job/unix_timestamp
        # otherwise will use slurm.job_dir / unix_timestamp
        job_dir = os.path.join(
            slurm_config.pop("job_dir", os.path.join(os.getcwd(), "slurm_jobs")), str(int(time.time()))
        )
        os.makedirs(job_dir, exist_ok=True)

        # Write job's config
        job_conf_path = os.path.join(job_dir, "job_config.yaml")
        with open(job_conf_path, "w") as fp:
            yaml.dump(config, fp, default_flow_style=False, sort_keys=False)
        logging.info(f"Logging Slurm job in: {job_dir}")
        return launch_with_slurm(args, job_conf_path, job_dir, slurm_config)
    elif "k8s" in config or "kubernetes" in config:
        # launch job on kubernetes.
        raise NotImplementedError("kubernetes support is pending")
    else:
        return run_interactive(args)


if __name__ == "__main__":
    main()

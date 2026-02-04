import os
from typing import Any, Dict

from nemo_run.core.execution.lepton import LeptonExecutor


def create_lepton_executor(config: Dict[str, Any]) -> LeptonExecutor:
    """Create a configured LeptonExecutor from CLI config.

    Expected keys in config:
      - container_image (str)
      - nodes (int)
      - gpus_per_node or nprocs_per_node (int)
      - resource_shape (str)
      - node_group (str)
      - node_reservation (str, optional)
      - mounts (list of dict with keys 'path' and 'mount_path') or None
      - env_vars (dict, optional)
      - image_pull_secrets (list[str], optional)
      - pre_launch_commands (list[str], optional)
    """
    container_image = config.get("container_image", "nvcr.io/nvidian/nemo:25.09.rc10")
    nodes = int(config.get("nodes", 1))
    gpus_per_node = int(config.get("gpus_per_node", config.get("nprocs_per_node", 0) or 0))
    resource_shape = config.get("resource_shape", "gpu.8xh100-sxm")
    node_group = config.get("node_group", "")
    node_reservation = config.get("node_reservation", "")
    mounts = config.get("mounts", [])
    env_vars = config.get("env_vars", {})

    # Default envs
    env_vars.setdefault("WANDB_API_KEY", os.getenv("WANDB_API_KEY", ""))
    if "HF_HOME" not in env_vars and (hf_home := config.get("hf_home")):
        env_vars["HF_HOME"] = hf_home

    image_pull_secrets = config.get("image_pull_secrets", [])
    pre_launch_commands = config.get("pre_launch_commands", ["nvidia-smi"])  # basic sanity

    # If no mounts provided, provide a reasonable default workspace PVC-like mount
    if not mounts:
        mounts = [
            {
                "path": "/nemo-workspace",
                "mount_path": "/nemo-workspace",
            }
        ]

    # LeptonExecutor expects a directory for persistent run metadata inside the mounted volume
    nemo_run_dir = config.get("nemo_run_dir", "/nemo-workspace/nemo-run/")

    executor = LeptonExecutor(
        container_image=container_image,
        nemo_run_dir=nemo_run_dir,
        launched_from_cluster=False,
        nodes=nodes,
        gpus_per_node=gpus_per_node,
        nprocs_per_node=gpus_per_node if gpus_per_node else int(config.get("nprocs_per_node", 1)),
        resource_shape=resource_shape,
        node_group=node_group,
        node_reservation=node_reservation,
        mounts=mounts,
        env_vars=env_vars,
        image_pull_secrets=image_pull_secrets,
        pre_launch_commands=pre_launch_commands,
    )
    return executor



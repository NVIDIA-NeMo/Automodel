import logging
import os
import sys
from functools import lru_cache
from typing import Any, Optional, Type, TypeVar

import torch
import yaml

from nemo_lm.utils.common_utils import get_rank_safe, get_world_size_safe, log_single_rank, print_rank_0

TRAIN_STATE_FILE = "train_state.pt"
TRACKER_PREFIX = "latest"
CONFIG_FILE = "run_config.yaml"

logger = logging.getLogger(__name__)
_T = TypeVar("_T")


def checkpoint_exists(checkpoints_path: str) -> bool:
    """Check if a checkpoint directory exists.

    Args:
        checkpoints_path: Path to the potential checkpoint directory.

    Returns:
        True if the path exists, False otherwise.
    """
    if checkpoints_path is None:
        return False
    return os.path.exists(os.path.join(checkpoints_path, f"{TRACKER_PREFIX}_{TRAIN_STATE_FILE}"))


def get_checkpoint_run_config_filename(checkpoints_path: str) -> str:
    """Get the filename for the run configuration file within a checkpoint directory.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.

    Returns:
        The full path to the run configuration file (e.g., run_config.yaml).
    """
    return os.path.join(checkpoints_path, CONFIG_FILE)


def get_checkpoint_train_state_filename(checkpoints_path: str, prefix: Optional[str] = None) -> str:
    """Get the filename for the train state tracker file.

    This file typically stores metadata about the latest checkpoint, like the iteration number.

    Args:
        checkpoints_path: Base directory where checkpoints are stored.
        prefix: Optional prefix (e.g., 'latest') to prepend to the filename.

    Returns:
        The full path to the train state tracker file.
    """
    if prefix is None:
        return os.path.join(checkpoints_path, TRAIN_STATE_FILE)
    else:
        return os.path.join(checkpoints_path, f"{prefix}_{TRAIN_STATE_FILE}")


@lru_cache()
def read_run_config(run_config_filename: str) -> dict[str, Any]:
    """Read the run configuration from a YAML file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks.

    Args:
        run_config_filename: Path to the run config YAML file.

    Returns:
        A dictionary containing the run configuration.

    Raises:
        RuntimeError: If reading the config file fails on rank 0.
    """
    config_obj = [None]

    if get_rank_safe() == 0:
        try:
            with open(run_config_filename, "r") as f:
                config_dict = yaml.safe_load(f)
            config_obj[0] = config_dict
        except Exception as e:
            error_msg = f"ERROR: Unable to load config file {run_config_filename}: {e}"
            sys.stderr.write(error_msg + "\n")
            config_obj[0] = {"error": True, "msg": error_msg}

    if torch.distributed.is_initialized():
        log_single_rank(
            logger, f"Broadcasting config from rank 0 to all {get_world_size_safe()} ranks", level=logging.INFO
        )
        torch.distributed.broadcast_object_list(config_obj, src=0)

    if isinstance(config_obj[0], dict) and config_obj[0].get("error", False):
        raise RuntimeError(config_obj[0]["msg"])

    return config_obj[0]


@lru_cache()
def read_train_state(train_state_filename: str, train_state_cls: Type[_T]) -> _T:
    """Read the train state metadata from a YAML file (rank 0 only).

    Reads the file on rank 0 and broadcasts the result to other ranks.

    Args:
        train_state_filename: Path to the train state YAML file.

    Returns:
        An initialized TrainState object.
    """
    state_obj = [None]
    if get_rank_safe() == 0:
        try:
            state_dict = torch.load(train_state_filename, map_location="cpu")
            ts = train_state_cls()
            ts.load_state_dict(state_dict)
            state_obj[0] = ts
        except Exception as e:
            error_msg = f"ERROR: Unable to load train state file {train_state_filename}: {e}"
            sys.stderr.write(error_msg + "\n")
            state_obj[0] = {"error": True, "msg": error_msg}

    if torch.distributed.is_initialized():
        print_rank_0(f"Broadcasting TrainState from rank 0 to all {get_world_size_safe()} ranks")
        torch.distributed.broadcast_object_list(state_obj, src=0)

    if isinstance(state_obj[0], dict) and state_obj[0].get("error", False):
        raise RuntimeError(state_obj[0]["msg"])

    return state_obj[0]

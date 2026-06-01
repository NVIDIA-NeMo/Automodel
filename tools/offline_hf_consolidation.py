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

"""Offline consolidation for sharded Hugging Face safetensors checkpoints.

Example usage on 2 workers:
    torchrun --nproc-per-node=2 tools/offline_hf_consolidation.py \
        --model-name meta-llama/Llama-3.2-1B \
        --input-dir checkpoints/epoch_0_step_19/model/ \
        --output-dir checkpoints/epoch_0_step_19/model/consolidated/

Example usage on 1 worker:
    python tools/offline_hf_consolidation.py \
        --model-name meta-llama/Llama-3.2-1B \
        --input-dir checkpoints/epoch_0_step_19/model/ \
        --output-dir checkpoints/epoch_0_step_19/model/consolidated/
"""

import argparse
import json
import logging
import os
import shutil

import torch
import torch.distributed as dist

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
    resolve_dtype_cast,
)
from nemo_automodel.components.checkpoint._backports.hf_storage import _maybe_rename_index_for_diffusers
from nemo_automodel.components.checkpoint._backports.hf_utils import (
    FQN_TO_DTYPE_MAPPING_FILENAME,
    FQN_TO_FILE_INDEX_MAPPING_FILENAME,
)
from nemo_automodel.components.distributed.init_utils import (
    get_rank_safe,
    get_world_size_safe,
    initialize_distributed,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Configure basic logging when the tool runs outside a recipe process."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )


def _config_torch_dtype_value(dtype: torch.dtype) -> str:
    """Return the dtype string expected in Hugging Face config.json."""
    return str(dtype).removeprefix("torch.")


def _update_config_dtype(config_path: str, cast_dtype: torch.dtype) -> None:
    """Update config.json torch_dtype to match a requested consolidated weight dtype."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except json.JSONDecodeError:
        logger.warning("Could not update torch_dtype in %s because it is not valid JSON.", config_path)
        return

    config["torch_dtype"] = _config_torch_dtype_value(cast_dtype)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def copy_metadata_files(input_dir: str, output_dir: str, cast_dtype: torch.dtype | None = None) -> None:
    """Copy metadata files from the temporary metadata directory."""
    for item_name in os.listdir(input_dir):
        if item_name in {FQN_TO_FILE_INDEX_MAPPING_FILENAME, FQN_TO_DTYPE_MAPPING_FILENAME}:
            continue
        src_path = os.path.join(input_dir, item_name)
        dst_path = os.path.join(output_dir, item_name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)
        if cast_dtype is not None and item_name == "config.json" and os.path.isfile(dst_path):
            _update_config_dtype(dst_path, cast_dtype)


def _has_consolidated_output(output_dir: str) -> bool:
    """Return True if output_dir already contains consolidated safetensors."""
    return os.path.isdir(output_dir) and any(filename.endswith(".safetensors") for filename in os.listdir(output_dir))


def parse_args() -> argparse.Namespace:
    """Parse command-line options for offline HF checkpoint consolidation."""

    parser = argparse.ArgumentParser(
        description=(
            "Consolidate sharded HF safetensors checkpoints into consolidated files, "
            "preserving original sharding layout where possible."
        )
    )

    parser.add_argument(
        "--model-name",
        "-m",
        required=True,
        help=(
            "Hugging Face repo id (e.g. meta-llama/Llama-3.2-1B) or absolute path to a HF snapshot directory. "
            "Used as reference to copy metadata and derive FQN->file index mapping."
        ),
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing sharded safetensors files to consolidate.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where consolidated safetensors and metadata will be written.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=5,
        help="Number of threads for writing consolidated data (default: 5).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend to initialize (default: auto).",
    )
    parser.add_argument(
        "--diffusers-compatible",
        action="store_true",
        help="Rename the safetensors index to the Diffusers-compatible filename after consolidation.",
    )
    parser.add_argument(
        "--cast-dtype",
        default=None,
        help=(
            "Optional dtype for floating-point tensors in the consolidated checkpoint. "
            "Supported aliases include bf16, bfloat16, fp16, float16, fp32, and float32. "
            "Integer tensors keep their original dtype."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run offline HF safetensors consolidation."""

    _configure_logging()
    args = parse_args()
    cast_dtype = resolve_dtype_cast(args.cast_dtype)

    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.device_count() > 0 else "gloo"
    initialize_distributed(backend)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError("Could not locate the input directory. Pass an absolute path to the input directory.")

    hf_metadata_dir = os.path.join(args.input_dir, ".hf_metadata")

    if not os.path.exists(hf_metadata_dir) or not os.path.isdir(hf_metadata_dir):
        if _has_consolidated_output(args.output_dir):
            if get_rank_safe() == 0:
                logger.info(
                    "Consolidated HF safetensors already exist at %s; skipping export because %s is missing.",
                    args.output_dir,
                    hf_metadata_dir,
                )
            return
        raise FileNotFoundError("Expected to find the .hf_metadata directory in the input directory.")

    with open(os.path.join(hf_metadata_dir, FQN_TO_FILE_INDEX_MAPPING_FILENAME), "r") as f:
        fqn_to_index_mapping = json.load(f)
    fqn_to_dtype_mapping = None
    fqn_to_dtype_mapping_path = os.path.join(hf_metadata_dir, FQN_TO_DTYPE_MAPPING_FILENAME)
    if os.path.exists(fqn_to_dtype_mapping_path):
        with open(fqn_to_dtype_mapping_path, "r") as f:
            fqn_to_dtype_mapping = json.load(f)

    consolidate_safetensors_files_on_every_rank(
        args.input_dir,
        args.output_dir,
        fqn_to_index_mapping,
        num_threads=args.num_threads,
        cast_dtype=cast_dtype,
        fqn_to_dtype_mapping=fqn_to_dtype_mapping,
    )

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        copy_metadata_files(hf_metadata_dir, args.output_dir, cast_dtype=cast_dtype)
        if args.diffusers_compatible:
            _maybe_rename_index_for_diffusers(args.output_dir)

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        logger.info("Successfully exported consolidated HF safetensors to %s.", args.output_dir)


if __name__ == "__main__":
    main()

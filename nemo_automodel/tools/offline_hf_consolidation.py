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
    torchrun --nproc-per-node=2 -m nemo_automodel.tools.offline_hf_consolidation \
        --model-name meta-llama/Llama-3.2-1B \
        --input-dir checkpoints/epoch_0_step_19/model/ \
        --output-dir checkpoints/epoch_0_step_19/model/consolidated/

Example usage on 1 worker:
    python -m nemo_automodel.tools.offline_hf_consolidation \
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
)
from nemo_automodel.components.checkpoint._backports.hf_storage import _maybe_rename_index_for_diffusers
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


def copy_metadata_files(input_dir: str, output_dir: str) -> None:
    """Copy metadata files from the temporary metadata directory."""
    for item_name in os.listdir(input_dir):
        if item_name == "fqn_to_file_index_mapping.json":
            continue
        src_path = os.path.join(input_dir, item_name)
        dst_path = os.path.join(output_dir, item_name)
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dst_path)


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
    return parser.parse_args()


def main() -> None:
    """Run offline HF safetensors consolidation."""

    _configure_logging()
    args = parse_args()

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

    with open(os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json"), "r") as f:
        fqn_to_index_mapping = json.load(f)

    if get_rank_safe() == 0:
        logger.info("Consolidating sharded HF safetensors from %s to %s.", args.input_dir, args.output_dir)

    consolidate_safetensors_files_on_every_rank(
        args.input_dir,
        args.output_dir,
        fqn_to_index_mapping,
        num_threads=args.num_threads,
    )

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        copy_metadata_files(hf_metadata_dir, args.output_dir)
        if args.diffusers_compatible:
            _maybe_rename_index_for_diffusers(args.output_dir)

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        logger.info("Successfully exported consolidated HF safetensors to %s.", args.output_dir)


if __name__ == "__main__":
    main()

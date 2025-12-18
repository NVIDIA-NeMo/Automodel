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

"""
Example: Distributed Model Loading with NeMoAutoModelForCausalLM

This example demonstrates how to use NeMoAutoModelForCausalLM as a drop-in
replacement for Hugging Face's AutoModelForCausalLM with built-in distributed
training support.

Usage:
    # Single GPU (no distributed)
    python distributed_model_loading.py

    # Multi-GPU with tensor parallelism (TP=2)
    torchrun --nproc-per-node=2 distributed_model_loading.py --tp-size 2

    # Multi-GPU with data parallelism (DP=4)
    torchrun --nproc-per-node=4 distributed_model_loading.py --dp-size 4

    # Combined TP and DP (TP=2, DP=2, total 4 GPUs)
    torchrun --nproc-per-node=4 distributed_model_loading.py --tp-size 2 --dp-size 2
"""

import argparse
import logging
import os

import torch
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Model Loading Example")
    parser.add_argument(
        "--model-name",
        type=str,
        default="hf-internal-testing/tiny-random-LlamaForCausalLM",
        help="HuggingFace model name or path",
    )
    parser.add_argument("--tp-size", type=int, default=1, help="Tensor parallelism size")
    parser.add_argument("--dp-size", type=int, default=0, help="Data parallelism size (0 = auto)")
    parser.add_argument("--cp-size", type=int, default=1, help="Context parallelism size")
    parser.add_argument("--pp-size", type=int, default=1, help="Pipeline parallelism size")
    parser.add_argument(
        "--use-device-mesh",
        action="store_true",
        help="Use explicit device_mesh instead of distributed dict",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Model dtype (bfloat16, float16, float32)")
    return parser.parse_args()


def setup_distributed():
    """Initialize distributed environment if running with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True
    return False


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_with_distributed_dict(model_name: str, args) -> torch.nn.Module:
    """
    Load model using the distributed dictionary approach.

    This is the simpler approach - just pass a dictionary with parallelism settings.
    """
    from nemo_automodel import NeMoAutoModelForCausalLM

    distributed_config = {
        "tp_size": args.tp_size,
        "cp_size": args.cp_size,
        "pp_size": args.pp_size,
    }

    if args.dp_size > 0:
        distributed_config["dp_size"] = args.dp_size

    logger.info(f"Loading model with distributed config: {distributed_config}")

    model = NeMoAutoModelForCausalLM.from_pretrained(
        model_name,
        distributed=distributed_config,
        torch_dtype=getattr(torch, args.dtype),
        use_liger_kernel=False,  # Disable for simplicity
        use_sdpa_patching=False,
        attn_implementation="eager",
    )

    return model


def load_with_device_mesh(model_name: str, args) -> torch.nn.Module:
    """
    Load model using explicit device_mesh approach.

    This gives you full control over the mesh topology.
    """
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel import NeMoAutoModelForCausalLM

    # Calculate mesh shape
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    dp_size = args.dp_size if args.dp_size > 0 else world_size // (args.tp_size * args.cp_size * args.pp_size)

    mesh_shape = (args.pp_size, 1, dp_size, args.cp_size, args.tp_size)
    mesh_dim_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")

    logger.info(f"Creating device mesh with shape: {mesh_shape}, names: {mesh_dim_names}")

    device_mesh = init_device_mesh(
        device_type="cuda" if torch.cuda.is_available() else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_dim_names,
    )

    logger.info(f"Loading model with device_mesh (size={device_mesh.size()})")

    model = NeMoAutoModelForCausalLM.from_pretrained(
        model_name,
        device_mesh=device_mesh,
        distributed={"tp_size": args.tp_size, "cp_size": args.cp_size, "pp_size": args.pp_size},
        torch_dtype=getattr(torch, args.dtype),
        use_liger_kernel=False,
        use_sdpa_patching=False,
        attn_implementation="eager",
    )

    return model


def main():
    args = parse_args()

    # Setup distributed if running with torchrun
    is_distributed = setup_distributed()

    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1

    if rank == 0:
        logger.info(f"World size: {world_size}")
        logger.info(f"TP size: {args.tp_size}, CP size: {args.cp_size}, PP size: {args.pp_size}")

    try:
        # Load model using the selected method
        if args.use_device_mesh and is_distributed:
            model = load_with_device_mesh(args.model_name, args)
        else:
            model = load_with_distributed_dict(args.model_name, args)

        if rank == 0:
            logger.info(f"Model loaded successfully!")
            logger.info(f"Model type: {type(model).__name__}")

            # Print some model info
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"Total parameters: {param_count:,}")

        # Simple forward pass test
        if rank == 0:
            logger.info("Running test forward pass...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dummy input
        input_ids = torch.randint(0, 1000, (1, 16), device=device)

        with torch.no_grad():
            outputs = model(input_ids)

        if rank == 0:
            logger.info(f"Forward pass successful! Output shape: {outputs.logits.shape}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()


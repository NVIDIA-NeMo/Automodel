# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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


"""Compare checkpoint-load GPU memory using the same Qwen3-MoE weights in fresh processes.

Run each transport in a separate torchrun invocation (no training is performed):
    torchrun --standalone --nproc-per-node=4 \
        tests/functional_tests/checkpoint/reproduce_load_metadata_memory.py \
        --transport nccl --output /tmp/qwen-load-nccl
    torchrun --standalone --nproc-per-node=4 \
        tests/functional_tests/checkpoint/reproduce_load_metadata_memory.py \
        --transport gloo --output /tmp/qwen-load-gloo

Pass --model to use a local Qwen/Qwen3-30B-A3B checkpoint. Compare each rank's
parameter_sha256 between outputs. Device memory is sampled after empty_cache
and before hashing. Load timings are not a controlled filesystem-cache benchmark.
"""

import argparse
import gc
import hashlib
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed import (
    DistributedSetup,
    FSDP2Config,
    MoEParallelizerConfig,
    ParallelismSizes,
)
from nemo_automodel.components.models.common import BackendConfig

logger = logging.getLogger(__name__)


def _memory() -> dict[str, float]:
    """Read synchronized device and PyTorch allocator usage in MiB."""
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    return {
        "device_used_mib": (total - free) / 2**20,
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": reserved / 2**20,
        "outside_allocator_estimate_mib": (total - free - reserved) / 2**20,
    }


def main() -> None:
    """Load the complete checkpoint and hash every local parameter without a GPU gather."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--transport", choices=("nccl", "gloo"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.manual_seed(1234)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    control = dist.new_group(backend="gloo")
    try:
        # Initialize the training communicator in both processes before measuring.
        check = torch.ones(1, device="cuda")
        dist.all_reduce(check)
        assert check.item() == world_size
        del check
        gc.collect()
        torch.cuda.empty_cache()
        before = _memory()
        setup = DistributedSetup.build(
            strategy=FSDP2Config(reshard_after_forward=True, defer_fsdp_grad_sync=False),
            parallelism_sizes=ParallelismSizes(ep_size=world_size),
            moe_parallel_config=MoEParallelizerConfig(reshard_after_forward=True),
            world_size=world_size,
        )
        start = time.perf_counter()
        model = NeMoAutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            distributed_setup=setup,
            use_liger_kernel=False,
            attn_implementation="sdpa",
            backend=BackendConfig(
                attn="te",
                linear="torch",
                rms_norm="torch_fp32",
                experts="torch_mm",
                dispatcher="hybridep",
                enable_hf_state_dict_adapter=True,
                enable_fsdp_optimizations=False,
                rope_fusion=False,
            ),
            load_metadata_with_gloo=args.transport == "gloo",
        )
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - start
        dist.barrier(group=control)
        gc.collect()
        torch.cuda.empty_cache()
        after = _memory()
        digest = hashlib.sha256()
        parameter_elements = 0
        for name, parameter in model.named_parameters():
            local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
            flat = local.detach().reshape(-1)
            digest.update(f"{name}:{tuple(local.shape)}:{local.dtype}".encode())
            chunk = max(1, 16 * 1024**2 // flat.element_size())
            for offset in range(0, flat.numel(), chunk):
                digest.update(flat[offset : offset + chunk].cpu().contiguous().view(torch.uint8).numpy())
            parameter_elements += flat.numel()
        # Closing the load-only checkpointer must preserve the training group.
        check = torch.ones(1, device="cuda")
        dist.all_reduce(check)
        assert check.item() == world_size
        result = {
            "rank": rank,
            "transport": args.transport,
            "world_size": world_size,
            "layers": model.config.num_hidden_layers,
            "local_parameter_elements": parameter_elements,
            "parameter_sha256": digest.hexdigest(),
            "before_load": before,
            "after_load": after,
            "load_seconds": load_seconds,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank{rank}.json").write_text(json.dumps(result, indent=2))
        logger.info("%s", json.dumps(result))
        dist.barrier(group=control)
    finally:
        dist.destroy_process_group(control)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

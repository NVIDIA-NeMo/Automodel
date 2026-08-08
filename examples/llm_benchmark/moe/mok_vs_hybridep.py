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

"""Compare AutoModel's HybridEP and MoK MoE layer forward/backward paths.

Run each backend in a fresh four-rank process so communication workspaces and
allocator state cannot leak between measurements::

    torchrun --standalone --nproc-per-node=4 \
      examples/llm_benchmark/moe/mok_vs_hybridep.py --dispatcher hybridep
    PYTHONPATH=../mixture-of-kittens:$PYTHONPATH torchrun --standalone \
      --nproc-per-node=4 examples/llm_benchmark/moe/mok_vs_hybridep.py --dispatcher mok

The default 20 steps include five warmups and time forward plus backward only;
the full-model recipes next to this script cover optimizer-inclusive throughput.
"""

import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

from nemo_automodel.components.models.common import BackendConfig, MoKBackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.parallelizer import ExpertParallel


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatcher", choices=("hybridep", "mok"), required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--mok-fwd-comm-sms", type=int, default=40)
    parser.add_argument("--mok-bwd-comm-sms", type=int, default=28)
    args = parser.parse_args()
    if args.steps <= args.warmup_steps:
        parser.error("--steps must be greater than --warmup-steps")
    return args


def _build_moe(args: argparse.Namespace, device: torch.device) -> MoE:
    config = MoEConfig(
        n_routed_experts=args.experts,
        n_shared_experts=1,
        n_activated_experts=args.topk,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=args.hidden_size,
        inter_dim=4 * args.hidden_size,
        moe_inter_dim=args.intermediate_size,
        norm_topk_prob=True,
        dtype=torch.bfloat16,
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        experts="gmm",
        dispatcher=args.dispatcher,
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
        mok=MoKBackendConfig(
            fwd_num_comm_sms=args.mok_fwd_comm_sms,
            bwd_num_comm_sms=args.mok_bwd_comm_sms,
        ),
    )
    module = MoE(config, backend).to(device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        module.init_weights(device)
    ep_mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("ep",))
    parallelize_module(module.experts, ep_mesh, ExpertParallel())
    return module.train()


def main() -> None:
    """Run one distributed MoE forward/backward benchmark and print its summary."""
    args = _parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    if dist.get_world_size() not in (4, 8, 16, 32, 64):
        raise ValueError("This matched benchmark requires an MoK-supported EP size: 4, 8, 16, 32, or 64")
    if args.experts % dist.get_world_size() != 0:
        raise ValueError("--experts must be divisible by the world size")

    torch.manual_seed(1234)
    module = _build_moe(args, device)
    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    x = torch.randn(args.tokens, args.hidden_size, device=device, dtype=torch.bfloat16, generator=generator)
    grad_output = torch.randn(
        args.tokens,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(args.steps)]

    dist.barrier()
    for start, end in events:
        module.zero_grad(set_to_none=True)
        x = x.detach().requires_grad_()
        start.record()
        output = module(x)
        output.backward(grad_output)
        end.record()

    torch.cuda.synchronize()
    rank_times = torch.tensor(
        [start.elapsed_time(end) for start, end in events[args.warmup_steps :]],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(rank_times, op=dist.ReduceOp.MAX)
    times_ms = rank_times.cpu().tolist()
    result = {
        "dispatcher": args.dispatcher,
        "world_size": dist.get_world_size(),
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "measured_steps": len(times_ms),
        "tokens_per_rank": args.tokens,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "experts": args.experts,
        "topk": args.topk,
        "mok_fwd_comm_sms": args.mok_fwd_comm_sms if args.dispatcher == "mok" else None,
        "mok_bwd_comm_sms": args.mok_bwd_comm_sms if args.dispatcher == "mok" else None,
        "forward_backward_mean_ms": statistics.fmean(times_ms),
        "forward_backward_median_ms": statistics.median(times_ms),
        "forward_backward_min_ms": min(times_ms),
        "forward_backward_max_ms": max(times_ms),
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
    }
    if rank == 0:
        print(json.dumps(result, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

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

import argparse
import gc
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import ProfilerActivity

from nemo_automodel.components._peft.lora import patch_linear_module
from nemo_automodel.components._peft.lora_kernel import HAVE_TRITON

_RESULT_PREFIX = "LORA_MEMORY_PROFILE_RESULT "
_GAIN_PREFIX = "LORA_MEMORY_PROFILE_GAIN "
_MIN_SAVED_BYTES = 1024 * 1024


@dataclass
class ProfileResult:
    """Memory profile result for one LoRA mode."""

    peak_delta_bytes: int
    profiler_memory_bytes: int


@dataclass
class ProfileConfig:
    """Shape configuration for one memory profile run."""

    seq_len: int
    hidden_size: int
    rank: int
    layers: int


_PROFILE_CONFIG = ProfileConfig(seq_len=4096, hidden_size=512, rank=16, layers=16)
_LORA_IMPL_PARAMS = [
    pytest.param(False, id="torch"),
    pytest.param(True, marks=pytest.mark.skipif(not HAVE_TRITON, reason="Triton is not available"), id="triton"),
]


class LoRAStack(nn.Module):
    """Small stack of real LinearLoRA modules used by the profiler tests."""

    def __init__(
        self, *, use_memory_efficient_lora: bool, use_triton: bool, config: ProfileConfig, device: torch.device
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(config.layers):
            base = nn.Linear(config.hidden_size, config.hidden_size, bias=False, dtype=torch.bfloat16, device=device)
            self.layers.append(
                patch_linear_module(
                    base,
                    dim=config.rank,
                    alpha=config.rank,
                    dropout=0.0,
                    lora_dtype=torch.bfloat16,
                    use_memory_efficient_lora=use_memory_efficient_lora,
                    use_triton=use_triton,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run every LoRA layer so its forward state stays live until backward."""
        for layer in self.layers:
            x = x + layer(x)
        return x


def _run_step(model: nn.Module, x: torch.Tensor) -> None:
    out = model(x)
    loss = out.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(x.device)


def _profiler_memory_bytes(prof: torch.profiler.profile) -> int:
    return sum(max(0, getattr(event, "self_device_memory_usage", 0)) for event in prof.key_averages())


def _profile_model(
    *, use_memory_efficient_lora: bool, use_triton: bool, device: torch.device, use_fsdp: bool
) -> ProfileResult:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(1234)

    config = _PROFILE_CONFIG
    model = LoRAStack(
        use_memory_efficient_lora=use_memory_efficient_lora, use_triton=use_triton, config=config, device=device
    )
    if use_fsdp:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

    x = torch.randn(1, config.seq_len, config.hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True)

    _run_step(model, x)
    model.zero_grad(set_to_none=True)
    x.grad = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    allocated_before = torch.cuda.memory_allocated(device)

    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        record_shapes=True,
    ) as prof:
        _run_step(model, x)

    peak_delta_bytes = torch.cuda.max_memory_allocated(device) - allocated_before
    profiler_memory_bytes = _profiler_memory_bytes(prof)

    del model, x
    gc.collect()
    torch.cuda.empty_cache()
    return ProfileResult(peak_delta_bytes=peak_delta_bytes, profiler_memory_bytes=profiler_memory_bytes)


def _profile_pair(*, use_triton: bool, device: torch.device, use_fsdp: bool) -> dict[str, int]:
    before = _profile_model(use_memory_efficient_lora=False, use_triton=use_triton, device=device, use_fsdp=use_fsdp)
    after = _profile_model(use_memory_efficient_lora=True, use_triton=use_triton, device=device, use_fsdp=use_fsdp)
    saved_bytes = before.peak_delta_bytes - after.peak_delta_bytes
    return {
        "before_peak_delta_bytes": before.peak_delta_bytes,
        "after_peak_delta_bytes": after.peak_delta_bytes,
        "saved_bytes": saved_bytes,
        "before_profiler_memory_bytes": before.profiler_memory_bytes,
        "after_profiler_memory_bytes": after.profiler_memory_bytes,
        "seq_len": _PROFILE_CONFIG.seq_len,
        "hidden_size": _PROFILE_CONFIG.hidden_size,
        "lora_rank": _PROFILE_CONFIG.rank,
        "layers": _PROFILE_CONFIG.layers,
        "use_triton": int(use_triton),
    }


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / 1024**2:.2f} MiB"


def _dump_memory_gain(result: dict[str, int], *, mode: str, dist_rank: int | None = None) -> None:
    before_peak = result["before_peak_delta_bytes"]
    after_peak = result["after_peak_delta_bytes"]
    saved_bytes = result["saved_bytes"]
    saved_pct = 100.0 * saved_bytes / before_peak if before_peak > 0 else 0.0
    impl = "triton" if result["use_triton"] else "torch"
    rank_part = f" dist_rank={dist_rank}" if dist_rank is not None else ""
    print(
        f"{_GAIN_PREFIX}mode={mode} impl={impl}{rank_part} "
        f"seq_len={result['seq_len']} hidden_size={result['hidden_size']} "
        f"lora_rank={result['lora_rank']} layers={result['layers']} "
        f"before_peak={_format_mib(before_peak)} after_peak={_format_mib(after_peak)} "
        f"saved={_format_mib(saved_bytes)} saved_pct={saved_pct:.2f}% "
        f"before_profiler_memory={_format_mib(result['before_profiler_memory_bytes'])} "
        f"after_profiler_memory={_format_mib(result['after_profiler_memory_bytes'])}",
        flush=True,
    )


def _assert_profile_memory_gain(result: dict[str, int]) -> None:
    assert result["before_profiler_memory_bytes"] > 0
    assert result["after_profiler_memory_bytes"] > 0
    assert result["saved_bytes"] >= _MIN_SAVED_BYTES, result


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_triton", _LORA_IMPL_PARAMS)
def test_memory_efficient_lora_torch_profile_single_gpu(use_triton: bool):
    """Memory-efficient LoRA should reduce profiled peak memory on one GPU."""
    result = _profile_pair(use_triton=use_triton, device=torch.device("cuda", 0), use_fsdp=False)
    _dump_memory_gain(result, mode="single_gpu")
    _assert_profile_memory_gain(result)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
@pytest.mark.parametrize("use_triton", _LORA_IMPL_PARAMS)
def test_memory_efficient_lora_torch_profile_two_gpu_fsdp(use_triton: bool):
    """Memory-efficient LoRA should reduce profiled peak memory with 2-GPU FSDP."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--fsdp-worker",
        "--use-triton" if use_triton else "--no-use-triton",
    ]
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout

    result_line = next(
        (line for line in completed.stdout.splitlines() if line.startswith(_RESULT_PREFIX)),
        None,
    )
    assert result_line is not None, completed.stdout
    for rank, rank_result in enumerate(json.loads(result_line.removeprefix(_RESULT_PREFIX))):
        _dump_memory_gain(rank_result, mode="two_gpu_fsdp", dist_rank=rank)
        _assert_profile_memory_gain(rank_result)


def _run_fsdp_worker(use_triton: bool) -> None:
    dist.init_process_group("nccl")
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        result = _profile_pair(use_triton=use_triton, device=device, use_fsdp=True)

        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, result)
        if dist.get_rank() == 0:
            print(_RESULT_PREFIX + json.dumps(gathered), flush=True)

        ok = int(
            result["before_profiler_memory_bytes"] > 0
            and result["after_profiler_memory_bytes"] > 0
            and result["saved_bytes"] >= _MIN_SAVED_BYTES
        )
        ok_tensor = torch.tensor([ok], device=device)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        if ok_tensor.item() != 1:
            raise AssertionError(result)
    finally:
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp-worker", action="store_true")
    parser.add_argument("--use-triton", dest="use_triton", action="store_true")
    parser.add_argument("--no-use-triton", dest="use_triton", action="store_false")
    parser.set_defaults(use_triton=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.fsdp_worker:
        _run_fsdp_worker(args.use_triton)

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

"""One persistent torchrun group per GPU job, with bounded individual cases."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CASE_TIMEOUT = 30.0


class GPUWorkers:
    """Share imports and CUDA/NCCL startup across the functional precision matrix."""

    def __init__(self, directory: Path, ranks: int, *, warmup_moe: bool = False) -> None:
        self.directory = directory
        self.ranks = ranks
        self.index = 0
        self.log = directory / "workers.log"
        self.output = self.log.open("wb")
        env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
        # Persistent topology groups can exhaust H100 multicast resources.
        # These tiny tensors do not benefit from NVLink SHARP collectives.
        env.setdefault("NCCL_NVLS_ENABLE", "0")
        env.setdefault("NEMO_HYBRIDEP_JIT_CACHE", str(directory / "hybridep_kernels"))
        env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={ranks}",
                str(Path(__file__).resolve()),
                str(directory),
                *(["--warmup-moe"] if warmup_moe else []),
            ],
            cwd=REPO,
            env=env,
            stdout=self.output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            self._wait_for(directory / "ready.json", 180.0)
        except BaseException:
            self.close()
            raise
        self.startup_seconds = time.monotonic() - started
        (directory / "startup.json").write_text(json.dumps({"seconds": self.startup_seconds}))
        print(f"Shared GPU worker startup: {self.startup_seconds:.2f}s", flush=True)

    def _wait_for(self, path: Path, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while not path.exists():
            if self.process.poll() is not None:
                raise AssertionError(f"GPU workers exited: {self.log.read_text(errors='replace')}")
            if time.monotonic() >= deadline:
                self.close()
                raise AssertionError(f"GPU workers exceeded {timeout}s: {self.log.read_text(errors='replace')}")
            time.sleep(0.02)

    def run(self, command: dict[str, Any], output: Path) -> None:
        """Execute a case and retain its logs and elapsed time separately from startup."""
        offset = self.log.stat().st_size
        request = self.directory / f"request.{self.index}.json"
        request.with_suffix(".tmp").write_text(json.dumps(command))
        request.with_suffix(".tmp").replace(request)
        response = self.directory / f"response.{self.index}.json"
        self.index += 1
        started = time.monotonic()
        try:
            self._wait_for(response, CASE_TIMEOUT)
        finally:
            with self.log.open("rb") as log:
                log.seek(offset)
                (output / "worker.log").write_text(log.read().decode(errors="replace"))
        result = json.loads(response.read_text())
        elapsed = time.monotonic() - started
        (output / "timing.json").write_text(json.dumps({"seconds": elapsed, "worker_seconds": result["seconds"]}))
        assert not any(result["errors"]), "\n".join(error for error in result["errors"] if error)
        assert elapsed < CASE_TIMEOUT, elapsed

    def close(self) -> None:
        """Terminate the complete torchrun process group, including hung workers."""
        if self.process.poll() is None:
            request = self.directory / f"request.{self.index}.json"
            request.with_suffix(".tmp").write_text('{"kind": "stop"}')
            request.with_suffix(".tmp").replace(request)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                time.sleep(0.2)
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
        self.output.close()


def worker_main(directory: Path, *, warmup_moe: bool = False) -> None:
    """Import GPU dependencies once, then execute requests on every rank."""
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.init_utils import initialize_distributed
    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction  # noqa: F401
    from tests.functional_tests.parallelism.run_pp_dtype_parity import _run_case
    from tests.functional_tests.parallelism_multigpu.run_recipe import prepare_recipe_meshes, run_recipe_pair_in_process

    info = initialize_distributed("nccl")
    device = info.device
    setups = prepare_recipe_meshes(info.world_size, device, include_moe=warmup_moe)
    if warmup_moe:
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.moe.megatron.fused_a2a import reset_hybrid_ep_buffer
        from nemo_automodel.components.moe.megatron.token_dispatcher import _HybridEPManager

        # Compile communication kernels once without constructing a model or
        # exercising PP. Actual model initialization remains inside each case.
        ep_mesh = init_device_mesh("cuda", (info.world_size // 2, 2), mesh_dim_names=("replica", "ep"))
        manager = _HybridEPManager(
            ep_mesh["ep"].get_group(),
            num_local_experts=2,
            num_experts=4,
            router_topk=2,
            # Match both GroupedExpertsDeepEP and GroupedExpertsTE defaults;
            # different SM counts require different HybridEP JIT kernels.
            permute_fusion=True,
            moe_hybridep_num_sms=BackendConfig().dispatcher_num_sms,
        )
        for tokens in (512, 32, 64):
            manager.initialize_runtime(num_tokens=tokens, hidden_dim=256, dtype=torch.bfloat16, device=device)
        del manager
        reset_hybrid_ep_buffer()
    dist.barrier()
    if info.rank == 0:
        (directory / "ready.json").write_text("{}")
    meshes = {}
    index = 0
    while True:
        command = [None]
        if info.rank == 0:
            request = directory / f"request.{index}.json"
            while not request.exists():
                time.sleep(0.02)
            command[0] = json.loads(request.read_text())
        dist.broadcast_object_list(command, src=0)
        case = command[0]
        if case["kind"] == "stop":
            break
        started = time.monotonic()
        error = None
        try:
            if case["kind"] == "dense":
                pp_size = case["pp_size"]
                if pp_size not in meshes:
                    meshes[pp_size] = init_device_mesh(
                        "cuda", (pp_size, info.world_size // pp_size), mesh_dim_names=("pp", "dp")
                    )
                _run_case(
                    meshes[pp_size],
                    device,
                    fp32_residual=case["fp32_residual"],
                    checkpointing=case["checkpointing"],
                    mtp_depth=case["mtp_depth"],
                )
            elif case["kind"] == "recipe":
                run_recipe_pair_in_process(
                    Path(case["config"]), case["overrides"], case["pp_size"], Path(case["output"]), setups
                )
            else:
                raise ValueError(f"Unknown GPU case: {case['kind']}")
            torch.cuda.synchronize()
        except Exception:
            error = traceback.format_exc()
        errors = [None] * info.world_size
        dist.all_gather_object(errors, error)
        sys.stdout.flush()
        sys.stderr.flush()
        dist.barrier()
        if info.rank == 0:
            response = directory / f"response.{index}.json"
            response.with_suffix(".tmp").write_text(
                json.dumps({"errors": errors, "seconds": time.monotonic() - started})
            )
            response.with_suffix(".tmp").replace(response)
        if any(errors):
            raise RuntimeError("Distributed functional case failed")
        index += 1
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--warmup-moe", action="store_true")
    args = parser.parse_args()
    worker_main(args.directory, warmup_moe=args.warmup_moe)

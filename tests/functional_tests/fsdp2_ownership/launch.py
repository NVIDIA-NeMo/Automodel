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

"""Launch bounded multi-GPU ownership regressions without fixed rendezvous ports."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import torch


def run_distributed(gpus: int, script: str, *args: str, extra_env: dict[str, str] | None = None) -> None:
    """Run one regression, failing if the dedicated runner exposes too few GPUs."""
    if torch.cuda.device_count() < gpus:
        raise RuntimeError(f"This suite requires {gpus} visible GPUs; found {torch.cuda.device_count()}")
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(extra_env or {})
    process = subprocess.Popen(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={gpus}", script, *args],
        cwd=root,
        env=env,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        # torchrun owns worker subprocesses; terminate the entire group on timeout.
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)

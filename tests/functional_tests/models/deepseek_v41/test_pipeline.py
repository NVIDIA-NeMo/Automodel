# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real PP send/receive, media chunking and optimizer parity on two GPUs."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize(
    "cuts,schedule", [([14], "1f1b"), ([3], "1f1b"), ([24], "1f1b"), ([3, 9, 25], "interleaved1f1b")]
)
@pytest.mark.timeout(240)
def test_pipeline_gradients_and_update(checkpoint: bool, cuts: list[int], schedule: str) -> None:
    script = Path(__file__).with_name("run_pipeline.py")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(script),
        "--schedule",
        schedule,
        "--cuts",
        *map(str, cuts),
    ]
    if checkpoint:
        command.append("--checkpoint")
    subprocess.run(command, check=True, timeout=220)

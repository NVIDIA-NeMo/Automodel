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

"""Tiny PP/FSDP and PP/TP/CP regressions on dedicated four/eight-GPU runners."""

from pathlib import Path

import pytest

from tests.functional_tests.parallelism_multigpu.gpu_workers import GPUWorkers
from tests.functional_tests.parallelism_multigpu.recipe_parity import run_recipe_pair


@pytest.mark.parametrize("pp_size", [2, 4])
@pytest.mark.parametrize("fp32_residual", [False, True], ids=["bf16", "fp32-residual"])
@pytest.mark.parametrize("checkpointing", [False, True], ids=["eager", "checkpoint"])
@pytest.mark.parametrize("mtp_depth", [0, 1], ids=["no-mtp", "mtp"])
def test_pp_fsdp_mixed_dtypes(
    parallelism_gpu_workers: GPUWorkers,
    pp_size: int,
    fp32_residual: bool,
    checkpointing: bool,
    mtp_depth: int,
    tmp_path: Path,
) -> None:
    """Compare loss, every parameter gradient and updates, with BF16/FP32 residuals.

    The worker also crosses activation checkpointing, MTP and changing sequence
    lengths. PP4 has interior stages that the existing two-GPU suite cannot test.
    """
    parallelism_gpu_workers.run(
        {
            "kind": "dense",
            "pp_size": pp_size,
            "fp32_residual": fp32_residual,
            "checkpointing": checkpointing,
            "mtp_depth": mtp_depth,
        },
        tmp_path,
    )


def test_pp_tp_cp_recipe(parallelism_gpu_workers: GPUWorkers, tmp_path: Path) -> None:
    """Compare PP2/TP2 (four GPUs) or PP2/TP2/CP2 (eight GPUs) to a no-PP reference."""
    count = parallelism_gpu_workers.ranks
    run_recipe_pair(
        workers=parallelism_gpu_workers,
        pp_size=2,
        config=Path(__file__).with_name("llama.yaml"),
        overrides=["--distributed.cp_size", str(count // 4), "--model.backend.attn", "te" if count == 8 else "sdpa"],
        output=tmp_path,
        loss_tol=0.05,
        grad_norm_rtol=0.05,
    )

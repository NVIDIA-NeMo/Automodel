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

"""Native Nemotron residual precision across real HybridEP expert kernels."""

from pathlib import Path

import pytest

from tests.functional_tests.parallelism_multigpu.gpu_workers import GPUWorkers
from tests.functional_tests.parallelism_multigpu.recipe_parity import run_recipe_pair


@pytest.mark.parametrize("experts", ["torch_mm", "te"])
@pytest.mark.parametrize("fp32_residual", [False, True], ids=["bf16", "fp32-residual"])
def test_pp_ep_precision(
    parallelism_gpu_workers: GPUWorkers, experts: str, fp32_residual: bool, tmp_path: Path
) -> None:
    """Exercise PP2/EP2 or PP4/EP2, including validation and optimizer updates."""
    count = parallelism_gpu_workers.ranks
    run_recipe_pair(
        workers=parallelism_gpu_workers,
        pp_size=count // 2,
        config=Path(__file__).with_name("nemotron.yaml"),
        overrides=["--model.backend.experts", experts, "--model.config.residual_in_fp32", str(fp32_residual).lower()],
        output=tmp_path,
        loss_tol=0.10,
        grad_norm_rtol=0.05,
    )


def test_pp_ep_fsdp_checkpointing(parallelism_gpu_workers: GPUWorkers, tmp_path: Path) -> None:
    """Compose PP2 with EP2 and FSDP2/4, FP32 residuals and checkpoint recomputation."""
    count = parallelism_gpu_workers.ranks
    run_recipe_pair(
        workers=parallelism_gpu_workers,
        pp_size=2,
        config=Path(__file__).with_name("nemotron.yaml"),
        overrides=[
            "--distributed.activation_checkpointing",
            "true",
            # Four local samples on each of count/2 DP ranks.
            "--step_scheduler.global_batch_size",
            str(2 * count),
            "--validation_dataset.num_sentences",
            str(2 * count),
        ],
        output=tmp_path,
        loss_tol=0.10,
        grad_norm_rtol=0.05,
    )

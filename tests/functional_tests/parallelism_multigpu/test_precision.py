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

from tests.functional_tests.parallelism_multigpu.recipe_parity import gpu_count, run_distributed, run_recipe_pair


@pytest.mark.parametrize("pp_size", [2, 4])
def test_pp_fsdp_mixed_dtypes(pp_size: int, tmp_path: Path) -> None:
    """Compare loss, every parameter gradient and updates, with BF16/FP32 residuals.

    The worker also crosses activation checkpointing, MTP and changing sequence
    lengths. PP4 has interior stages that the existing two-GPU suite cannot test.
    """
    run_distributed(
        gpu_count(),
        ["tests/functional_tests/parallelism/run_pp_dtype_parity.py", "--pp-size", str(pp_size)],
        tmp_path / "dtype.log",
    )


def test_pp_tp_cp_recipe(tmp_path: Path) -> None:
    """Compare PP2/TP2 (four GPUs) or PP2/TP2/CP2 (eight GPUs) to a no-PP reference."""
    count = gpu_count()
    run_recipe_pair(
        ranks=count,
        pp_size=2,
        config=Path(__file__).with_name("llama.yaml"),
        overrides=["--distributed.cp_size", str(count // 4), "--model.backend.attn", "te" if count == 8 else "sdpa"],
        output=tmp_path,
        loss_tol=0.05,
        grad_norm_rtol=0.05,
    )

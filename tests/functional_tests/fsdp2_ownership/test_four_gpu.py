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

"""Dedicated four-GPU suite; the workflow must expose all four devices."""

import pytest

from tests.functional_tests.fsdp2_ownership.launch import run_distributed

SCRIPT = "tests/functional_tests/fsdp2_ownership/run_qwen3_5_ownership.py"


def test_hsdp_both_nontrivial_dimensions() -> None:
    run_distributed(
        4,
        "tests/functional_tests/training/run_fsdp_casting_ownership.py",
        extra_env={"HSDP_MESH_SHAPE": "2,2", "RUN_TE_FSDP_CASE": "1"},
    )


@pytest.mark.parametrize("dtype,te", [("float32", False), ("bfloat16", True)])
def test_packed_qwen_model_and_optimizer_resume(dtype: str, te: bool) -> None:
    run_distributed(4, SCRIPT, "--case", "packed-resume", "--dtype", dtype, *(["--te"] if te else []))


def test_qwen_cp2_fsdp2_bf16_reduction() -> None:
    run_distributed(4, SCRIPT, "--case", "cp", "--dtype", "bfloat16")


def test_qwen_tp2_fsdp2() -> None:
    run_distributed(4, SCRIPT, "--case", "tp")


@pytest.mark.parametrize("teacher_axis", ["dp", "tp", "cp"])
def test_qwen_kd_separate_teacher_mesh(teacher_axis: str) -> None:
    run_distributed(4, SCRIPT, "--case", "kd", "--teacher-axis", teacher_axis)

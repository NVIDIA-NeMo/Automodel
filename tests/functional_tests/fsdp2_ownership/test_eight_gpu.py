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

"""TP, PP, and FSDP must all be nontrivial to cover their interaction."""

from tests.functional_tests.fsdp2_ownership.launch import run_distributed


def test_qwen_tp2_pp2_fsdp2_gradient_and_optimizer_parity() -> None:
    run_distributed(8, "tests/functional_tests/fsdp2_ownership/run_qwen3_5_ownership.py", "--case", "pp-tp")

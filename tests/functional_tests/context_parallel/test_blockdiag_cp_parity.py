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

"""Functional test for block-diagonal varlen context parallelism.

Launches a real 2-rank torchrun job (run_blockdiag_cp_2rank.py) that checks
forward outputs, input gradients, and per-parameter gradients for every
KV-exchange collective (allgather / halo / a2a) and attention backend
(dense / flash / TE) against a single-process dense block-causal reference.
"""

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "context_parallel"
CP_BLOCKDIAG_VARLEN_TEST_FILENAME = "L2_CP_BlockDiag_Varlen_Test.sh"


class TestBlockDiagContextParallel:
    """Test suite for block-diagonal (packed-sequence) context parallelism."""

    def test_cp_blockdiag_varlen_2rank_parity(self):
        """2-rank fwd+bwd parity for allgather/halo/a2a KV exchange vs a dense reference."""
        run_test_script(TEST_FOLDER, CP_BLOCKDIAG_VARLEN_TEST_FILENAME)

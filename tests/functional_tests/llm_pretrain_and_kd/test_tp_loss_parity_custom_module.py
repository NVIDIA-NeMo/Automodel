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

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "llm_pretrain_and_kd"
TP_LOSS_PARITY_CUSTOM_MODULE = "L2_TP_Loss_Parity_Custom_Module.sh"


class TestTPLossParityCustomModule:
    def test_tp2_loss_matches_tp1_with_custom_replicated_module(self):
        """TP=2 loss must match TP=1 when a custom non-TP module is present.

        Exercises _broadcast_replicated_params_across_tp: after restoring
        a checkpoint with TP=2, replicated (non-sharded) custom module
        weights are broadcast from TP rank 0 so all ranks agree.
        """
        run_test_script(TEST_FOLDER, TP_LOSS_PARITY_CUSTOM_MODULE)

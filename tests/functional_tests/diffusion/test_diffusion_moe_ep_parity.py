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

"""Expert-parallel parity of a custom-model MoE diffusion transformer trained by the diffusion recipe.

See ``L2_Diffusion_ToyMoEDiT_EP2_Parity.sh``: a toy custom-model MoE DiT is finetuned with
``fsdp.ep_size=1`` and ``fsdp.ep_size=2`` on the same 2 ranks; per-step loss and grad norm
must match and the EP leg must really shard its experts over the ``ep`` mesh axis.
"""

import pytest
import torch

from tests.utils.test_utils import run_test_script

TEST_FOLDER = "diffusion"
TOY_MOE_DIT_EP2_PARITY_FILENAME = "L2_Diffusion_ToyMoEDiT_EP2_Parity.sh"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
class TestDiffusionMoEExpertParallel:
    def test_toy_moe_dit_ep2_parity(self):
        run_test_script(TEST_FOLDER, TOY_MOE_DIT_EP2_PARITY_FILENAME)

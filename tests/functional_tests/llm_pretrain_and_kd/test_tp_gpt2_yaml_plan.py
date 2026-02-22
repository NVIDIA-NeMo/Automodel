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
TP_GPT2_YAML_PLAN = "L2_TP_GPT2_YAML_Plan.sh"


class TestTPGPT2YAMLPlan:
    def test_tp2_gpt2_yaml_plan_logit_parity(self):
        """TP=2 logits must match TP=1 when the plan is a YAML string-dict.

        Exercises the YAML-driven TP plan path: users specify the plan as
        a plain dict of strings (e.g. "colwise", "rowwise") in their YAML
        config, and ``_get_parallel_plan`` auto-translates them to
        ``ParallelStyle`` objects.
        """
        run_test_script(TEST_FOLDER, TP_GPT2_YAML_PLAN)

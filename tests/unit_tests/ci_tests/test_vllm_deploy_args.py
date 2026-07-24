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

from tests.functional_tests.checkpoint_robustness.test_checkpoint_vllm_deploy import _resolve_args


def test_resolve_args_enables_vllm_expert_parallel_from_recipe(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  pretrained_model_name_or_path: test/model
ci:
  vllm_smoke_test: true
  vllm_enable_expert_parallel: true
"""
    )

    args = _resolve_args(
        {
            "config_path": str(config_path),
            "deploy_mode": "peft",
            "adapter_path": "/tmp/adapter",
        }
    )

    assert args["enable_expert_parallel"] is True


def test_resolve_args_disables_vllm_expert_parallel_by_default(tmp_path):
    config_path = tmp_path / "recipe.yaml"
    config_path.write_text(
        """
model:
  pretrained_model_name_or_path: test/model
ci:
  vllm_smoke_test: true
"""
    )

    args = _resolve_args(
        {
            "config_path": str(config_path),
            "deploy_mode": "peft",
            "adapter_path": "/tmp/adapter",
        }
    )

    assert args["enable_expert_parallel"] is False

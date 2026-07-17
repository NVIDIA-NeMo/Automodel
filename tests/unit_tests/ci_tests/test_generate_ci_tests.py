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

from pathlib import Path

import pytest

from tests.ci_tests.utils.generate_ci_tests import resolve_recipe_path


def test_resolve_recipe_path_defaults_to_test_folder():
    assert resolve_recipe_path("llama/recipe.yaml", "llm_finetune") == Path("examples/llm_finetune/llama/recipe.yaml")


def test_resolve_recipe_path_preserves_full_examples_path():
    assert resolve_recipe_path("examples/llm_pretrain/deepseek_v4/recipe.yaml", "llm_finetune") == Path(
        "examples/llm_pretrain/deepseek_v4/recipe.yaml"
    )


@pytest.mark.parametrize("path", ["/tmp/recipe.yaml", "../recipe.yaml", "examples/../recipe.yaml"])
def test_resolve_recipe_path_rejects_repository_escape(path):
    with pytest.raises(ValueError, match="must stay within the repository"):
        resolve_recipe_path(path, "llm_finetune")

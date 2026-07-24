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

import ast
from pathlib import Path


def test_bagel_auto_model_path_uses_distributed_setup_kwarg():
    """BAGEL's AutoModel path must match the shared VLM build_model API."""
    recipe_path = Path(__file__).resolve().parents[3] / "nemo_automodel/recipes/multimodal/finetune.py"
    tree = ast.parse(recipe_path.read_text())

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "build_vlm_model"
    ]
    assert len(calls) == 1

    keywords = {kw.arg for kw in calls[0].keywords}
    assert "distributed_setup" in keywords
    assert (
        not {
            "device_mesh",
            "moe_mesh",
            "distributed_config",
            "pipeline_config",
            "cfg_moe",
            "activation_checkpointing",
        }
        & keywords
    )


def test_bagel_finalizes_pending_checkpoint_before_closing_checkpointer():
    """Async BAGEL checkpoints must be published before the checkpointer closes."""
    from types import SimpleNamespace

    from nemo_automodel.recipes.multimodal.finetune import FinetuneRecipeForMultimodal

    events = []
    recipe = FinetuneRecipeForMultimodal.__new__(FinetuneRecipeForMultimodal)
    recipe.model = SimpleNamespace(train=lambda: events.append("train"))
    recipe.step_scheduler = SimpleNamespace(epochs=[])
    recipe.metric_logger_train = SimpleNamespace(close=lambda: events.append("train_logger_close"))
    recipe.metric_logger_valid = SimpleNamespace(close=lambda: events.append("valid_logger_close"))
    recipe.checkpointer = SimpleNamespace(close=lambda: events.append("checkpointer_close"))
    recipe._finalize_pending_checkpoint = lambda: events.append("finalize")

    FinetuneRecipeForMultimodal.run_train_validation_loop(recipe)

    assert events == ["train", "train_logger_close", "valid_logger_close", "finalize", "checkpointer_close"]

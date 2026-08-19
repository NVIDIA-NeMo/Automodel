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

from nemo_automodel.components.config.loader import load_yaml_config
from nemo_automodel.components.models.bagel.configuration import BagelBackendConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
BAGEL_SFT_CONFIG_PATH = REPO_ROOT / "examples/multimodal_finetune/bagel/bagel_sft.yaml"


def test_bagel_sft_auto_backend_uses_bagel_config():
    """The shipped auto-init recipe must pass a typed BAGEL backend to AutoModel."""
    config = load_yaml_config(BAGEL_SFT_CONFIG_PATH)

    def capture_backend(*, backend: object, **_: object) -> object:
        return backend

    config.set_by_dotted("model._target_", capture_backend)
    backend = config.model.instantiate()

    assert config.model.get("init_mode", "auto") == "auto"
    assert isinstance(backend, BagelBackendConfig)
    assert backend.linear == "te"
    assert backend.rms_norm == "te"
    assert backend.mot_grouped is True
    assert backend.fused_swiglu is True
    assert backend.fused_rope is True


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
    recipe.checkpointer = SimpleNamespace(finalize=lambda: events.append("checkpointer_finalize"))

    FinetuneRecipeForMultimodal.run_train_validation_loop(recipe)

    assert events == ["train", "train_logger_close", "valid_logger_close", "checkpointer_finalize"]


def test_bagel_dataloader_resolved_through_recipeconfig():
    """The recipe stores a raw ConfigNode, but ``bagel_dataloader`` is a typed
    RecipeConfig property. It must be accessed via ``RecipeConfig(self.cfg)`` — a bare
    ``self.cfg.bagel_dataloader`` raises AttributeError on a ConfigNode (the real CLI
    entry hands the recipe a raw ConfigNode), which breaks every BAGEL run in ``setup``.
    """
    recipe_path = Path(__file__).resolve().parents[3] / "nemo_automodel/recipes/multimodal/finetune.py"
    tree = ast.parse(recipe_path.read_text())

    accesses = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and node.attr == "bagel_dataloader"]
    assert accesses, "expected a bagel_dataloader access in the recipe"
    for node in accesses:
        assert (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "RecipeConfig"
        ), "bagel_dataloader must be resolved via RecipeConfig(self.cfg), not accessed on a raw ConfigNode"

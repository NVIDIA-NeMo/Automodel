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

"""Real recipe save/resume with final-only quantization; no checkpoint IO mocks."""

import json
from copy import deepcopy

import pytest
import torch
import yaml
from safetensors.torch import load_file
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes._typed_config import RecipeConfig
from nemo_automodel.recipes.base_recipe import BaseRecipe
from tests.unit_tests.models.kimi_k25_vl.test_quantized_export import _TinyModel
from tests.unit_tests.models.kimi_k3.test_mxfp4_export import _model


@pytest.mark.parametrize(
    ("model_factory", "packed_dtype", "quantized_format"),
    [(_TinyModel, torch.int32, "pack-quantized"), (_model, torch.uint8, "mxfp4-pack-quantized")],
    ids=["kimi-k25", "kimi-k3"],
)
@pytest.mark.parametrize("quantization", [False, True])
@pytest.mark.parametrize("save_consolidated", ["final", "every"])
def test_recipe_intermediate_resume_and_final_quantization(
    tmp_path, model_factory, packed_dtype, quantized_format, quantization, save_consolidated
):
    """YAML reaches the real saver; periodic checkpoints restore model and training state exactly."""
    raw = yaml.safe_load(
        """
step_scheduler:
  global_batch_size: 1
  max_steps: 125
  ckpt_every_steps: 50
  preemption_signal: null
checkpoint:
  enabled: true
  model_save_format: safetensors
  save_consolidated: final
  quantization: true
"""
    )
    raw["checkpoint"].update(
        checkpoint_dir=str(tmp_path), quantization=quantization, save_consolidated=save_consolidated
    )
    recipe = BaseRecipe()
    recipe.cfg = RecipeConfig(ConfigNode(raw))
    recipe.model = model_factory()
    recipe.optimizer = torch.optim.AdamW(recipe.model.parameters(), lr=0.01)
    recipe.peft_config = None
    recipe.rng = StatefulRNG(seed=4078)
    recipe.dataloader = StatefulDataLoader(range(125), batch_size=1)
    recipe.step_scheduler = recipe.cfg.step_scheduler.build(
        dataloader=recipe.dataloader, dp_group_size=1, local_batch_size=1
    )
    recipe.checkpointer = recipe.cfg.checkpoint.build(dp_rank=0, tp_rank=0, pp_rank=0)
    config_before = deepcopy(recipe.model.config.to_dict())
    iterator = iter(recipe.dataloader)

    try:
        for step in (49, 99, 124):
            next(iterator)
            # A small synthetic loss updates every parameter and creates Adam moments.
            recipe.optimizer.zero_grad()
            loss = sum(parameter.float().square().mean() for parameter in recipe.model.parameters())
            loss.backward()
            recipe.optimizer.step()
            recipe.step_scheduler.step = step
            assert recipe.step_scheduler.is_ckpt_step
            before = {name: value.clone() for name, value in recipe.model.state_dict().items()}
            optimizer_before = deepcopy(recipe.optimizer.state_dict())
            dataloader_before = deepcopy(recipe.dataloader.state_dict())
            rng_before = recipe.rng.state_dict()

            recipe.save_checkpoint(epoch=0, step=step, train_loss=loss.item())
            checkpoint = tmp_path / f"epoch_0_step_{step}"
            model_dir = checkpoint / "model"
            saved = {}
            for shard in model_dir.glob("*.safetensors"):
                saved.update(load_file(str(shard)))
            assert saved
            packed = {key: value for key, value in saved.items() if key.endswith("weight_packed")}
            should_quantize = quantization and step == 124
            assert bool(packed) is should_quantize
            assert all(value.dtype == packed_dtype for value in packed.values())
            config = json.loads((model_dir / ".hf_metadata/config.json").read_text())
            scheme = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")
            assert (scheme is not None) is should_quantize
            if should_quantize:
                assert scheme["format"] == quantized_format
            consolidated = model_dir / "consolidated"
            assert consolidated.exists() is (step == 124 or save_consolidated == "every")
            if consolidated.exists():
                index = json.loads((consolidated / "model.safetensors.index.json").read_text())
                assert set(index["weight_map"]) == set(saved)
                for filename in set(index["weight_map"].values()):
                    actual = load_file(str(consolidated / filename))
                    for key, value in actual.items():
                        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
            torch.testing.assert_close(recipe.model.state_dict(), before, rtol=0, atol=0)
            assert recipe.model.config.to_dict() == config_before

            if step == 49:
                torch.rand(4)
                recipe.optimizer.step()
                next(iterator)
                recipe.step_scheduler.step = 80
                recipe.load_checkpoint(restore_from=str(checkpoint))
                # Inspect RNG before StatefulDataLoader.state_dict() can recreate its iterator.
                rng_after = recipe.rng.state_dict()
                torch.testing.assert_close(recipe.model.state_dict(), before, rtol=0, atol=0)
                torch.testing.assert_close(recipe.optimizer.state_dict(), optimizer_before, rtol=0, atol=0)
                torch.testing.assert_close(recipe.dataloader.state_dict(), dataloader_before, rtol=0, atol=0)
                for key, value in rng_before.items():
                    if isinstance(value, str):
                        assert rng_after[key] == value
                    else:
                        torch.testing.assert_close(rng_after[key], value, rtol=0, atol=0)
                assert recipe.step_scheduler.step == 50
                iterator = iter(recipe.dataloader)
    finally:
        recipe.checkpointer.close()

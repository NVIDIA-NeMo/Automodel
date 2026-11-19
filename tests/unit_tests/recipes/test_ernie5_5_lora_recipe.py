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

"""Contract tests for the ERNIE 5.5 Flash LoRA recipe."""

from functools import partial
from pathlib import Path

import pytest
import torch
import yaml

from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA
from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import parallelization
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.moe import _routed_situ
from nemo_automodel.components.models.ernie5_5.parallelization import PARALLELIZER
from nemo_automodel.components.moe import layers as shared_moe_layers
from nemo_automodel.recipes._dist_utils import parse_distributed_section
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_config
from tests.unit_tests.models.ernie5_5.test_parallelization import _context, _Mesh

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE = REPO_ROOT / "examples/llm_finetune/ernie5_5/ernie5_5_flash_squad_ep_lora.yaml"
FULL_RECIPE = REPO_ROOT / "examples/llm_finetune/ernie5_5/ernie5_5_flash_squad_ep.yaml"

# The contract checks parse YAML and build the model on the meta device; they never use CUDA.
pytestmark = pytest.mark.cpu_only


def _load(path: Path) -> dict:
    """Load a checked-in ERNIE 5.5 Flash recipe."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_lora_recipe_changes_the_full_fine_tuning_recipe_only_where_lora_needs_it() -> None:
    """Check that model, data, loss, precision policy and clipping are those of the full fine-tuning recipe."""
    lora, full = _load(RECIPE), _load(FULL_RECIPE)

    changed = {section for section in lora.keys() | full.keys() if lora.get(section) != full.get(section)}
    assert changed == {"peft", "step_scheduler", "checkpoint", "optimizer", "ci"}
    # One document per rank on 8 ranks instead of 64; the schedule of updates, validation and checkpoints is kept.
    assert {**lora["step_scheduler"], "global_batch_size": 64} == full["step_scheduler"]
    # The same optimizer; only the learning rate differs, and it is the validated one.
    assert {**lora["optimizer"], "lr": full["optimizer"]["lr"]} == full["optimizer"]
    assert lora["optimizer"]["lr"] == 1.0e-5
    assert {**lora["ci"], "nodes": 16} == full["ci"]


def test_lora_recipe_places_one_rank_on_each_expert_group() -> None:
    """Check the topology: FSDP2 with EP only, as many ranks as expert groups, one document per rank."""
    config = _load(RECIPE)
    parsed = parse_distributed_section(config["distributed"])
    sizes = parsed["parallelism_sizes"]

    assert isinstance(parsed["strategy_config"], FSDP2Config)
    assert (sizes.tp_size, sizes.pp_size, sizes.cp_size, sizes.ep_size) == (1, 1, 1, 8)
    ci = config["ci"]
    assert ci["cluster_tag"] == "gb200"
    assert ci["nodes"] * ci["nproc_per_node"] == sizes.ep_size
    scheduler = config["step_scheduler"]
    assert scheduler["global_batch_size"] == scheduler["local_batch_size"] * sizes.ep_size


def test_lora_recipe_schedules_adapter_checkpoints_and_keeps_save_consolidated_explicit() -> None:
    """Check that the recipe schedules its checkpoints and keeps ``save_consolidated`` explicit.

    With the ``peft`` section of the recipe these are adapter checkpoints.
    """
    config = _load(RECIPE)

    assert config["checkpoint"]["enabled"] is True
    # A LoRA run writes no consolidated export whatever this key says; the assertion pins the explicit value.
    assert config["checkpoint"]["save_consolidated"] is False
    scheduler = config["step_scheduler"]
    assert scheduler["max_steps"] % scheduler["ckpt_every_steps"] == 0
    assert scheduler["save_checkpoint_every_epoch"] is False


def test_lora_recipe_targets_select_supported_modules_of_the_released_layer_schedule(monkeypatch) -> None:
    """Install the recipe's adapters on the 43 decoder layers of the released shapes and admit the model.

    The model is built on the meta device, so no weight is allocated. The counts are those of the released
    schedule: 37 HCA layers and 6 DSA layers with five projections each, 42 MoE layers with one expert module and
    three shared-expert projections each, and the three projections of the dense FFN of layer 0.
    """
    recipe = _load(RECIPE)
    peft_section = dict(recipe["peft"])
    assert peft_section.pop("_target_") == "nemo_automodel.components._peft.lora.PeftConfig"
    peft = PeftConfig(**peft_section)
    assert (peft.dim, peft.alpha, peft.use_dora, peft.expert_weight_format) == (8, 32, False, "unquantized")
    # `dropout` and `moe_rank_scaling`, which the model card lists as not applied or not validated, and the two
    # other selectors of targets keep their defaults.
    assert (peft.dropout, peft.moe_rank_scaling, peft.match_all_linear, peft.exclude_modules) == (0.0, False, False, [])

    backend_config = dict(recipe["model"]["backend"])
    assert backend_config.pop("_target_") == "nemo_automodel.components.models.common.BackendConfig"
    assert (backend_config["experts"], backend_config["dispatcher"]) == ("torch_mm", "hybridep")
    monkeypatch.setattr(shared_moe_layers, "get_world_size_safe", lambda: 2)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(
            checkpoint_config(num_nextn_predict_layers=0),
            BackendConfig(**backend_config),
        )
    assert apply_lora_to_linear_modules(model, peft, skip_freeze=True) == (37 + 6) * 5 + 42 * (1 + 3) + 3
    assert isinstance(model.model.layers[1].moe.experts, GroupedExpertsDeepEPLoRA)
    activation = model.model.layers[1].moe.experts.expert_activation
    assert isinstance(activation, partial) and activation.func is _routed_situ
    assert activation.keywords == {
        "beta": model.config.activation_situ_beta,
        "linear_beta": model.config.activation_situ_linear_beta,
    }
    _apply_trainability_policy(model, peft_enabled=True, freeze_config=None, strict=True)

    adapters = {name: parameter for name, parameter in model.named_parameters() if "lora_" in name}
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == set(adapters)
    # Two factors per linear projection and four grouped factors per expert module.
    assert len(adapters) == 2 * ((37 + 6) * 5 + 42 * 3 + 3) + 4 * 42 == 856
    assert {parameter.dtype for parameter in adapters.values()} == {torch.bfloat16}
    assert sum(parameter.numel() for parameter in adapters.values()) == 1_790_614_528
    experts = sum(parameter.numel() for name, parameter in adapters.items() if ".moe.experts." in name)
    assert experts == 1_761_607_680

    ranks = recipe["ci"]["nodes"] * recipe["ci"]["nproc_per_node"]
    monkeypatch.setattr(parallelization.dist, "get_world_size", lambda group=None: ranks)
    context = _context(ep_size=recipe["distributed"]["ep_size"], device_mesh=_Mesh(ranks=ranks))
    assert PARALLELIZER._prepare_model(model, context) is None

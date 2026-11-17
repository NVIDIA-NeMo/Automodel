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

"""Contract tests for the ERNIE 5.5 Flash packed context-parallel recipes."""

from pathlib import Path

import pytest
import torch
import yaml

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import parallelization
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.parallelization import PARALLELIZER
from nemo_automodel.recipes._dist_utils import parse_distributed_section
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_config
from tests.unit_tests.models.ernie5_5.test_parallelization import _context, _Mesh

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPES = REPO_ROOT / "examples/llm_finetune/ernie5_5"
PACKED_LORA = RECIPES / "ernie5_5_flash_tulu3_packed128k_cp8_lora.yaml"
PACKED_FULL = RECIPES / "ernie5_5_flash_tulu3_packed128k_cp8.yaml"
LORA_RECIPE = RECIPES / "ernie5_5_flash_squad_ep_lora.yaml"
FULL_RECIPE = RECIPES / "ernie5_5_flash_squad_ep.yaml"
PACKED = pytest.mark.parametrize("recipe", [PACKED_LORA, PACKED_FULL], ids=["lora", "full"])

# The contract checks parse YAML and build the model on the meta device; they never use CUDA.
pytestmark = pytest.mark.cpu_only


def _load(path: Path) -> dict:
    """Load a checked-in ERNIE 5.5 Flash recipe."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_packed_lora_recipe_keeps_the_model_adapters_and_optimizer_of_the_lora_recipe() -> None:
    packed, lora = _load(PACKED_LORA), _load(LORA_RECIPE)
    for section in ("model", "peft", "optimizer", "loss_fn", "clip_grad_norm", "seed"):
        assert packed[section] == lora[section], section


def test_packed_full_recipe_keeps_the_model_and_optimizer_of_the_full_recipe() -> None:
    packed, full = _load(PACKED_FULL), _load(FULL_RECIPE)
    assert "peft" not in packed
    for section in ("model", "optimizer", "loss_fn", "clip_grad_norm", "seed"):
        assert packed[section] == full[section], section


@pytest.mark.parametrize(("recipe", "rows"), [(PACKED_LORA, 1), (PACKED_FULL, 8)], ids=["lora", "full"])
def test_packed_recipe_splits_every_row_over_8_ranks(recipe: Path, rows: int) -> None:
    config = _load(recipe)
    parsed = parse_distributed_section(config["distributed"])
    sizes = parsed["parallelism_sizes"]

    assert isinstance(parsed["strategy_config"], FSDP2Config)
    assert (sizes.tp_size, sizes.pp_size, sizes.cp_size, sizes.ep_size) == (1, 1, 8, 8)
    ci = config["ci"]
    assert ci["cluster_tag"] == "gb200"
    # An update takes one packed row per group of 8 context-parallel ranks.
    assert ci["nodes"] * ci["nproc_per_node"] == sizes.cp_size * rows
    scheduler = config["step_scheduler"]
    assert scheduler["global_batch_size"] == rows and scheduler["local_batch_size"] == 1
    assert config["distributed"]["activation_checkpointing"] is True
    assert config["distributed"]["moe"]["reshard_after_forward"] is True


@PACKED
def test_packed_recipe_packs_shuffled_tulu3_rows_of_whole_compression_windows(recipe: Path) -> None:
    config = _load(recipe)
    packing, dataset, loader = config["packed_sequence"], config["dataset"], config["dataloader"]

    assert packing["packing_strategy"] == "thd" and packing["packed_sequence_size"] == 131072
    assert dataset["path_or_dataset_id"] == "allenai/tulu-3-sft-mixture"
    assert dataset["seq_length"] == packing["packed_sequence_size"] and dataset["truncation"] is True
    # The row order comes from the seeded shuffle before packing, not from the loader.
    assert dataset["shuffle_seed"] == config["seed"] and loader["shuffle"] is False
    assert loader["collate_fn"].endswith("packed_sequence_thd_collater")
    # Every document and every context-parallel shard starts at a multiple of the HCA compression window.
    assert packing["pad_to_multiple_of"] == 128
    assert packing["packed_sequence_size"] % (config["distributed"]["cp_size"] * packing["pad_to_multiple_of"]) == 0
    scheduler = config["step_scheduler"]
    assert packing["max_packs"] >= scheduler["max_steps"] * scheduler["global_batch_size"]
    assert (REPO_ROOT / dataset["chat_template"]).is_file()


@PACKED
def test_packed_recipe_topology_is_admitted_for_the_released_layer_schedule(
    recipe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _load(recipe)
    backend = config["model"]["backend"]
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(
            checkpoint_config(num_nextn_predict_layers=0),
            BackendConfig(
                attn=backend["attn"],
                linear=backend["linear"],
                experts=backend["experts"],
                dispatcher=backend["dispatcher"],
            ),
        )
    assert model.packed_alignment == config["packed_sequence"]["pad_to_multiple_of"]
    ranks = config["ci"]["nodes"] * config["ci"]["nproc_per_node"]
    monkeypatch.setattr(parallelization.dist, "get_world_size", lambda group=None: ranks)
    context = _context(
        cp_size=config["distributed"]["cp_size"],
        ep_size=config["distributed"]["ep_size"],
        activation_checkpointing=config["distributed"]["activation_checkpointing"],
        device_mesh=_Mesh(ranks=ranks),
    )
    assert PARALLELIZER._prepare_model(model, context) is None

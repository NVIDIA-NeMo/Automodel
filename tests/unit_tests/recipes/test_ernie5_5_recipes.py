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

"""Contract tests for the ERNIE 5.5 Flash fine-tuning recipe."""

from pathlib import Path

import pytest
import torch
import yaml
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.recipes._dist_utils import parse_distributed_section
from nemo_automodel.shared.utils import dtype_from_str

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE = REPO_ROOT / "examples/llm_finetune/ernie5_5/ernie5_5_flash_squad_ep.yaml"
MODEL_ID = "baidu/ERNIE-5.5-Flash-Base-BF16"

# The contract checks parse YAML and typed configs only and never use CUDA.
pytestmark = pytest.mark.cpu_only


def _load_recipe() -> dict:
    """Load the checked-in ERNIE 5.5 Flash recipe."""
    return yaml.safe_load(RECIPE.read_text(encoding="utf-8"))


def test_ernie5_5_flash_recipe_loads_decoder_only_from_the_pinned_release() -> None:
    """Check the model block: pinned release, explicit decoder-only override, cuDNN DSA backend."""
    config = _load_recipe()
    model = config["model"]

    assert config["recipe"] == "TrainFinetuneRecipeForNextTokenPrediction"
    assert model["_target_"] == "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
    assert model["pretrained_model_name_or_path"] == MODEL_ID
    assert len(model["revision"]) == 40
    assert model["trust_remote_code"] is False
    assert model["dtype"] == "bfloat16"
    assert model["num_nextn_predict_layers"] == 0
    backend = model["backend"]
    assert (backend["attn"], backend["linear"], backend["experts"], backend["dispatcher"]) == (
        "cudnn",
        "torch",
        "torch",
        "torch",
    )
    for section in ("dataset", "validation_dataset"):
        tokenizer = config[section]["tokenizer"]
        assert tokenizer["pretrained_model_name_or_path"] == MODEL_ID
        assert tokenizer["revision"] == model["revision"]


def test_ernie5_5_flash_recipe_feeds_single_unpadded_documents() -> None:
    """Check that no loader packs or pads: one truncated, unpadded document per rank and microbatch."""
    config = _load_recipe()

    assert config["packed_sequence"]["packed_sequence_size"] == 0
    assert config["step_scheduler"]["local_batch_size"] == 1
    for section in ("dataset", "validation_dataset"):
        dataset = config[section]
        assert dataset["truncation"] is True
        assert dataset["padding"] is False
        assert dataset["seq_length"] == 256
    for section in ("dataloader", "validation_dataloader"):
        loader = config[section]
        assert "batch_size" not in loader
        assert loader["collate_fn"] == {"_target_": "nemo_automodel.components.datasets.utils.default_collater"}
    assert "peft" not in config
    assert config["checkpoint"]["enabled"] is False


def test_ernie5_5_flash_recipe_requests_expert_parallelism_only() -> None:
    """Check the parsed topology: FSDP2 with EP only, sized for the declared CI allocation."""
    config = _load_recipe()
    parsed = parse_distributed_section(config["distributed"])
    sizes = parsed["parallelism_sizes"]

    assert isinstance(parsed["strategy_config"], FSDP2Config)
    assert (sizes.tp_size, sizes.pp_size, sizes.cp_size, sizes.ep_size) == (1, 1, 1, 8)
    assert sizes.dp_replicate_size is None
    assert parsed["pipeline_config"] is None
    assert parsed["activation_checkpointing"] is False
    # The MoE policy, when set, takes precedence over the strategy policy during expert-parallel sharding.
    policy = parsed["moe_parallel_config"].mp_policy or parsed["strategy_config"].mp_policy
    assert policy == MixedPrecisionPolicy(
        param_dtype=None, reduce_dtype=torch.float32, output_dtype=None, cast_forward_inputs=False
    )

    ci = config["ci"]
    assert ci["cluster_tag"] == "gb200"
    world_size = ci["nodes"] * ci["nproc_per_node"]
    assert world_size % sizes.ep_size == 0
    scheduler = config["step_scheduler"]
    assert scheduler["global_batch_size"] % (scheduler["local_batch_size"] * world_size) == 0


def test_ernie5_5_flash_recipe_keeps_fp32_master_weights_for_bf16_parameters() -> None:
    """Check that the optimizer updates FP32 master weights instead of the BF16 parameters."""
    optimizer = _load_recipe()["optimizer"]

    assert optimizer["_target_"] == "transformer_engine.pytorch.optimizers.fused_adam.FusedAdam"
    assert optimizer["master_weights"] is True
    assert optimizer["store_param_remainders"] is True
    assert dtype_from_str(optimizer["master_weight_dtype"]) == torch.float32
    assert dtype_from_str(optimizer["exp_avg_dtype"]) == dtype_from_str(optimizer["exp_avg_sq_dtype"])

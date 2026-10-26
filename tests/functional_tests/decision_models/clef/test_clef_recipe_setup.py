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

"""Run both shipped CLEF setup/main-loop paths with two GPUs and a tiny local checkpoint.

Only tokenization is stubbed. NeMo model loading, BF16 FSDP2, freezing, all-linear
LoRA, optimizers, DP-group resolution, lexical gathering, and checkpoint I/O are real.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import yaml
from test_clef_fsdp_acceptance import _model_config
from transformers import Qwen3_5ForConditionalGeneration as HFQwen3_5

from nemo_automodel.components.checkpoint.utils import read_checkpoint_pointer
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.clef.joint_schema_head import ClefHeadConfig, save_joint_head
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration as NativeQwen3_5
from nemo_automodel.recipes.decision.train_clef import TrainClefDecisionRecipe


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, *, add_special_tokens):
        return SimpleNamespace(input_ids=[1 + ord(character) % 40 for character in text])


def _run_case(root: Path, case: str) -> None:
    repository = Path(__file__).resolve().parents[4]
    cfg = yaml.safe_load((repository / f"examples/decision/clef_soc_triage_{case}.yaml").read_text())
    cfg["model"]["pretrained_model_name_or_path"] = str(root / "base")
    cfg["model"].pop("revision")
    cfg["model"]["backend"] = {
        "_target_": "nemo_automodel.components.models.common.BackendConfig",
        "linear": "torch",
        "attn": "sdpa",
        "rms_norm": "torch",
        "rope_fusion": False,
        "dispatcher": "torch",
        "enable_hf_state_dict_adapter": True,
    }
    cfg["output_dir"] = str(root / "run")
    cfg["head"]["dropout"] = 0.0
    cfg["data"].update({split: str(root / "data.jsonl") for split in ("train", "validation", "test")})
    cfg["data"]["max_length"] = 2048
    cfg["train"].update(epochs=1, local_batch_size=1, global_batch_size=4, val_every_steps=1)
    # Keep the examples' BF16 policy, freeze/PEFT selectors and activation checkpointing.
    recipe = TrainClefDecisionRecipe(ConfigNode(cfg))
    try:
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=_Tokenizer()):
            recipe.setup()
        assert isinstance(recipe.model, NativeQwen3_5)
        assert recipe._get_dp_group_size() == 2
        trainable = {id(parameter) for parameter in recipe.model.parameters() if parameter.requires_grad}
        assert bool(trainable) == (case == "lora")
        assert all("lora_" in name for name, parameter in recipe.model.named_parameters() if parameter.requires_grad)
        optimized = {
            id(parameter)
            for optimizer in recipe.optimizer
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        assert optimized == trainable
        assert all(not parameter.requires_grad for parameter in recipe.model.get_output_embeddings().parameters())
        assert not recipe.lexical.rows.requires_grad
        assert torch.isfinite(recipe.lexical.rows).all()
        steps = []

        def check_head_gradients(optimizer, args, kwargs):
            """Check identical finite flattened head gradients [head_parameters] before stepping."""
            assert all(parameter.grad is not None for parameter in recipe.head.parameters())
            flat = torch.cat([parameter.grad.detach().reshape(-1) for parameter in recipe.head.parameters()])
            assert torch.isfinite(flat).all()
            gathered = [torch.empty_like(flat) for _ in range(2)]
            dist.all_gather(gathered, flat, group=recipe._dp_group())
            torch.testing.assert_close(gathered[0], gathered[1], atol=0, rtol=0)
            steps.append(True)

        restored = []
        load_stage = recipe._load_stage

        def record_restore(path):
            load_stage(path)
            restored.append(path)

        recipe._load_stage = record_restore
        handle = recipe.head_optimizer.register_step_pre_hook(check_head_gradients)
        try:
            recipe.run_train_validation_loop()
        finally:
            handle.remove()
        assert steps == [True]
        best = read_checkpoint_pointer(root / "run", "LOWEST_VAL")
        assert best is not None and restored == [best]
        assert (root / "run/step_00000001/stage.json").is_file()
        assert (root / "run/metrics_test.json").is_file()
        rows = [json.loads(line) for line in (root / "run/training_log.jsonl").read_text().splitlines()]
        assert all(math.isfinite(row["loss"]) and math.isfinite(row["grad_norm"]) for row in rows if "loss" in row)
        print(f"CLEF_RECIPE_SETUP {case} PASS rank={dist.get_rank()}", flush=True)
    finally:
        if hasattr(recipe, "checkpointer"):
            recipe.checkpointer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two assigned CUDA GPUs for native FSDP2 setup")
@pytest.mark.parametrize("case", ["heads", "lora"])
def test_native_yaml_setup_training_and_selected_reload(tmp_path, case):
    with torch.device("cpu"):
        torch.manual_seed(17)
        HFQwen3_5(_model_config()).save_pretrained(tmp_path / "base")
        save_joint_head(
            ClefHeadConfig(hidden_size=16, width=16, routing_layers=1, layers=1, heads=2, feedforward=32).build(),
            tmp_path / "base",
            dtype=torch.float32,
        )
    records = [
        {
            "id": str(index),
            "state": "synthetic observation",
            "questions": {"pick": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
            "answers": {"pick": "a"},
        }
        for index in range(4)
    ]
    (tmp_path / "data.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    repository = str(Path(__file__).resolve().parents[4])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = repository + os.pathsep + environment.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--root",
            str(tmp_path),
            "--case",
            case,
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count(f"CLEF_RECIPE_SETUP {case} PASS rank=") == 2, completed.stdout + completed.stderr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--case", choices=("heads", "lora"), required=True)
    arguments = parser.parse_args()
    _run_case(arguments.root, arguments.case)

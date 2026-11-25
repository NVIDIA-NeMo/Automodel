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

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from nemo_automodel.components.datasets.llm.eagle3_cache import write_manifest, write_shard, write_target_embeddings
from nemo_automodel.components.speculative.eagle.core import _compute_target_distribution
from nemo_automodel.components.speculative.eagle.target import HFEagle3TargetModel

_ROOT = Path(__file__).resolve().parents[4]


def _write_target_cache(root: Path) -> tuple[Path, Path]:
    torch.manual_seed(29)
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config).eval()
    target = root / "model"
    model.save_pretrained(target)
    tokens = ["<pad>", "<unk>", "<eos>"] + [f"t{i}" for i in range(13)]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(dict(zip(tokens, range(16))), unk_token="<unk>")),
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(target)

    aux_layers = [0, 1, 2]
    wrapper = HFEagle3TargetModel(model, aux_layer_ids=aux_layers)
    ids = torch.tensor([[1, 3, 4, 5, 6, 2]])
    batch = wrapper.generate_batch(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        loss_mask=torch.tensor([[0, 1, 1, 1, 1, 1]]),
    )
    probs, mask = _compute_target_distribution(
        target_logits=batch.logits,
        selected_token_ids=torch.arange(16),
        selected_token_mask=torch.ones(16, dtype=torch.bool),
        loss_mask=batch.loss_mask,
    )
    cache = root / "cache"
    write_manifest(
        str(cache),
        {
            "target_model": str(target),
            "target_vocab_size": 16,
            "draft_vocab_size": 16,
            "num_samples": 1,
            "seq_length": 6,
            "shard_size": 1,
            "dtype": "fp32",
            "target_probs_topk": 0,
            "aux_hidden_dim": 48,
            "aux_layer_ids": aux_layers,
            "selected_token_ids": list(range(16)),
        },
    )
    write_target_embeddings(str(cache), model.get_input_embeddings().weight)
    write_shard(
        str(cache),
        0,
        {
            "input_ids": batch.input_ids,
            "attention_mask": batch.attention_mask,
            "loss_mask": batch.loss_mask,
            "aux_hidden_states": batch.aux_hidden_states,
            "target_probs": probs,
            "position_mask": mask,
        },
    )
    return target, cache


@pytest.mark.cpu_only
def test_reconsideration_recipe_trains_and_resumes_from_native_checkpoint(tmp_path: Path) -> None:
    config = yaml.safe_load((_ROOT / "examples/speculative/parallel_reconsideration/qwen3_0_6b_mvp.yaml").read_text())
    assert not config["recipe_args"].get("ckpt_every_steps")
    target, cache = _write_target_cache(tmp_path)
    output = tmp_path / "run"
    config["dist_env"]["backend"] = "gloo"
    config["recipe_args"].update(
        target_model_name_or_path=str(target),
        cached_target_path=str(cache),
        output_dir=str(output),
        seq_length=6,
        num_epochs=1,
        log_every_steps=1,
    )
    checkpoints = output / "checkpoints"
    config["checkpoint"]["checkpoint_dir"] = str(checkpoints)
    config_path = tmp_path / "run.yaml"

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1")
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(name, None)
    command = [sys.executable, "-m", "nemo_automodel.recipes.speculative_decoding.train_eagle3", "-c", str(config_path)]

    def run() -> str:
        config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
        result = subprocess.run(command, cwd=_ROOT, env=env, capture_output=True, text=True, timeout=90, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout + result.stderr

    assert "step=1 train_loss=" in run()
    assert [path.name for path in checkpoints.glob("epoch_*_step_1")] == ["epoch_1_step_1"]
    config["recipe_args"]["num_epochs"] = 2
    config["checkpoint"]["restore_from"] = "LATEST"
    resumed = run()
    assert "Resuming from checkpoint" in resumed and "epoch=1 step=2 train_loss=" in resumed

    exports = list(checkpoints.glob("epoch_*_step_2/model/consolidated/config.json"))
    assert exports
    exported = json.loads(exports[0].read_text())
    assert exported["architectures"] == ["LlamaParallelReconsiderationDraftModel"]
    assert exported["num_hidden_layers"] == 1
    assert exported["parallel_reconsideration"] == {"max_reconsiderations": 2, "balance_beta": 0.01}
    tensors = {}
    for shard in exports[0].parent.glob("*.safetensors"):
        tensors.update(load_file(shard))
    assert tensors["self_query.weight"].dtype == torch.float32
    assert torch.isfinite(tensors["halt_gate.weight"]).all()

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

"""Base-checkpoint DCP loads into task models whose heads the checkpoint lacks."""

import re
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.api import CheckpointException
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2ForTokenClassification

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig

pytestmark = pytest.mark.timeout(70)


def _tiny_config() -> Qwen2Config:
    """Return a one-layer Qwen2 config with an untied LM head and a single label."""
    return Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        num_labels=1,
        use_cache=False,
    )


def _load_base(model: torch.nn.Module, checkpoint_dir: Path, tmp_path: Path) -> None:
    """Initialize ``model`` from the HF safetensors in ``checkpoint_dir`` through DCP."""
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_save_format="safetensors",
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
        side_effect=AssertionError("Expected direct DCP loading"),
    ):
        checkpointer.load_model(model, str(checkpoint_dir), is_init_step=True)


def _save_without(model: torch.nn.Module, checkpoint_dir: Path, missing_key: str) -> None:
    """Save ``model`` as an HF checkpoint whose safetensors omit ``missing_key``."""
    model.save_pretrained(checkpoint_dir)
    weights_path = checkpoint_dir / "model.safetensors"
    tensors = load_file(weights_path)
    del tensors[missing_key]
    save_file(tensors, weights_path, metadata={"format": "pt"})


def test_base_load_keeps_new_task_head_initialization(tmp_path: Path):
    """The backbone loads from a causal-LM checkpoint; the absent ``score`` head keeps its init."""
    torch.manual_seed(0)
    base = Qwen2ForCausalLM(_tiny_config())
    base.save_pretrained(tmp_path / "base")
    model = Qwen2ForTokenClassification(Qwen2Config.from_pretrained(tmp_path / "base"))
    score_init = {name: value.detach().clone() for name, value in model.score.state_dict().items()}

    _load_base(model, tmp_path / "base", tmp_path)

    torch.testing.assert_close(model.model.embed_tokens.weight, base.model.embed_tokens.weight)
    torch.testing.assert_close(model.model.norm.weight, base.model.norm.weight)
    torch.testing.assert_close(model.score.state_dict(), score_init)


def test_base_load_restores_a_saved_task_head(tmp_path: Path):
    """A checkpoint that carries the head (e.g. a trained reward model) still loads it."""
    torch.manual_seed(0)
    trained = Qwen2ForTokenClassification(_tiny_config())
    trained.save_pretrained(tmp_path / "trained")
    torch.manual_seed(1)
    model = Qwen2ForTokenClassification(Qwen2Config.from_pretrained(tmp_path / "trained"))

    _load_base(model, tmp_path / "trained", tmp_path)

    torch.testing.assert_close(model.score.state_dict(), trained.score.state_dict())


# FSDP2's fully_shard swaps a model's class for a generated subclass with this naming.
_FSDPQwen2ForCausalLM = type("FSDPQwen2ForCausalLM", (Qwen2ForCausalLM,), {})


@pytest.mark.parametrize(
    ("model_cls", "missing_key"),
    [
        (Qwen2ForTokenClassification, "model.norm.weight"),
        (Qwen2ForCausalLM, "lm_head.weight"),
        (_FSDPQwen2ForCausalLM, "lm_head.weight"),
    ],
    ids=["backbone", "untied_lm_head", "fsdp_class_untied_lm_head"],
)
def test_base_load_rejects_other_missing_weights(tmp_path: Path, model_cls: type, missing_key: str):
    """Backbone weights, or a head of the checkpoint's own architecture, missing from it fail the load."""
    torch.manual_seed(0)
    _save_without(Qwen2ForCausalLM(_tiny_config()), tmp_path / "base", missing_key)
    model = model_cls(Qwen2Config.from_pretrained(tmp_path / "base"))

    with pytest.raises(CheckpointException, match=re.escape(f"Missing key in checkpoint state_dict: {missing_key}.")):
        _load_base(model, tmp_path / "base", tmp_path)

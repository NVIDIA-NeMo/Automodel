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

"""LoRA adapter checkpoints of Inkling 2 through ``Checkpointer.save_model`` and ``load_model``.

The adapter is injected with the LoRA recipe's ``target_modules`` (attention projections and the
grouped experts), saved as ``adapter_model.safetensors`` and loaded into a second model, as a
resumed run does.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration

from .parity_check_inkling2 import TINY_TEXT_CONFIG

pytestmark = pytest.mark.cpu_only

_BACKEND = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
_TEXT_CONFIG = {**TINY_TEXT_CONFIG, "num_hidden_layers": 2, "layer_types": ["hybrid_sliding", "hybrid"]}
# peft.target_modules of examples/vlm_finetune/inkling2/inkling2_small_medpix_lora_ep32.yaml.
_RECIPE_TARGETS = [
    "*language_model.layers.*.self_attn.q_proj",
    "*language_model.layers.*.self_attn.k_proj",
    "*language_model.layers.*.self_attn.v_proj",
    "*language_model.layers.*.self_attn.o_proj",
    "*language_model.layers.*.mlp.experts",
]
_EXPERT_ADAPTERS = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


def _peft_config() -> PeftConfig:
    return PeftConfig(target_modules=_RECIPE_TARGETS, dim=8, alpha=32, dropout=0.0, moe_rank_scaling=False)


def _lora_model(seed: int) -> Inkling2ForConditionalGeneration:
    """Return a two-layer text model with the recipe's adapters, every adapter tensor filled from ``seed``."""
    config = Inkling2Config(text_config=_TEXT_CONFIG, vision_config=None, torch_dtype="bfloat16")
    model = Inkling2ForConditionalGeneration(config, backend=_BACKEND)
    apply_lora_to_linear_modules(model, _peft_config())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return model


def _adapters(model: Inkling2ForConditionalGeneration) -> dict[str, torch.Tensor]:
    """Return the adapter tensors under their saved keys, ``base_model.model.<native key>``."""
    return {f"base_model.model.{name}": p.detach() for name, p in model.named_parameters() if "lora_" in name}


def _checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="inkling2",
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


def test_lora_adapter_saves_with_native_keys_and_resumes(tmp_path):
    """Linear and grouped expert adapters save with their native names and layouts, then load back bitwise."""
    model = _lora_model(seed=0)
    expected = _adapters(model)
    experts = "base_model.model.model.language_model.layers.0.mlp.experts."
    intermediate, hidden = TINY_TEXT_CONFIG["moe_intermediate_size"], TINY_TEXT_CONFIG["hidden_size"]
    # Native grouped layout; the gate/up adapter output keeps the interleaved columns of gate_and_up_projs.
    assert tuple(expected[experts + "lora_gate_and_up_B"].shape) == (8, 8, 2 * intermediate)
    assert tuple(expected[experts + "lora_down_B"].shape) == (8, 8, hidden)
    assert "base_model.model.model.language_model.layers.1.self_attn.q_proj.lora_A.weight" in expected

    _checkpointer(tmp_path).save_model(model, str(tmp_path / "step"), peft_config=_peft_config())
    saved = load_file(str(tmp_path / "step/model/adapter_model.safetensors"))
    assert set(saved) == set(expected)
    for key, value in expected.items():
        assert saved[key].dtype == value.dtype and torch.equal(saved[key], value), key
    assert sum(key.endswith(_EXPERT_ADAPTERS) for key in saved) == 2 * len(_EXPERT_ADAPTERS)

    resumed = _lora_model(seed=1)
    assert not torch.equal(_adapters(resumed)[experts + "lora_gate_and_up_B"], expected[experts + "lora_gate_and_up_B"])
    _checkpointer(tmp_path).load_model(resumed, str(tmp_path / "step/model"))
    loaded = _adapters(resumed)
    assert set(loaded) == set(expected)
    for key, value in expected.items():
        assert torch.equal(loaded[key], value), key

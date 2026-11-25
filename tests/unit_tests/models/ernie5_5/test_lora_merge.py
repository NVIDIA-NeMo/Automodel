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

"""Merge of an ERNIE 5.5 LoRA adapter into a standalone Hugging Face checkpoint, in one process on CPU.

A small FP32 model (HCA with a dense MLP, HCA with an MoE, DSA with an MoE) carries the adapters of the checked-in
LoRA recipe. The Checkpointer writes its adapter checkpoint, ``nemo_automodel.export_merged_peft_checkpoint`` merges
it into a model built with the backend that ``tools/merge_lora.py --automodel-native-export`` passes for
``--backend attn=sdpa experts=torch dispatcher=torch``, and the public loader reads the merged directory as a model
without adapters. ``tests/unit_tests/tools/test_merge_lora.py`` checks that the tool passes that backend.
"""

from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch

import pytest
import torch

from nemo_automodel import NeMoAutoModelForCausalLM, export_merged_peft_checkpoint
from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from tests.unit_tests.models.ernie5_5.test_lora import LENGTH, _install_lora, _randomize_second_factors, _recipe_targets
from tests.unit_tests.models.ernie5_5.test_model import _eager_backend, _tiny_config, _tiny_model, _tokens

pytestmark = pytest.mark.cpu_only


class _Reference(NamedTuple):
    """What a merge of the adapter checkpoint has to reproduce.

    Attributes:
        adapter_dir: Adapter checkpoint of a LoRA model whose adapters act.
        state: Native ``model.`` names of the model without adapters mapped to tensors in the shapes of its
            parameters and buffers, with every adapter update added to its base weight.
        logits: Logits of the LoRA model, of shape [1, LENGTH, vocab].
    """

    adapter_dir: Path
    state: dict[str, torch.Tensor]
    logits: torch.Tensor


def _peft_config() -> PeftConfig:
    """Return the target patterns of the checked-in LoRA recipe with a rank that fits the small model."""
    return PeftConfig(target_modules=_recipe_targets(), dim=2, alpha=4)


def _lora_model(backend: BackendConfig) -> Ernie5_5_MoeForCausalLM:
    """Build the small model with seeded weights and the adapters installed; their second factors are zero."""
    model = Ernie5_5_MoeForCausalLM(_tiny_config(), backend)
    model.load_state_dict(_tiny_model().state_dict())
    _install_lora(model, _peft_config())
    return model.eval()


@pytest.fixture
def reference(tmp_path: Path) -> _Reference:
    """Save the adapter of a LoRA model and record the weights and logits that a merge has to reproduce."""
    model = _lora_model(_eager_backend())
    _randomize_second_factors(model)
    with torch.no_grad():
        state = {name: value.clone() for name, value in model.state_dict().items() if "lora_" not in name}
        for name, module in model.named_modules():
            if isinstance(module, LinearLoRA):
                state[f"{name}.weight"] += module.scale * (module.lora_B.weight @ module.lora_A.weight)
            elif isinstance(module, GroupedExpertsLoRA):
                gate_and_up = torch.bmm(module.lora_gate_and_up_A, module.lora_gate_and_up_B)
                state[f"{name}.gate_and_up_projs"] += module.scale * gate_and_up
                state[f"{name}.down_projs"] += module.scale * torch.bmm(module.lora_down_A, module.lora_down_B)
        logits = model(_tokens(LENGTH)).logits
    checkpointer = Checkpointer(
        CheckpointingConfig(is_peft=True, save_consolidated=False, checkpoint_dir=str(tmp_path / "run")), 0, 0, 0
    )
    try:
        checkpointer.save_model(model, str(tmp_path / "lora_checkpoint"), peft_config=_peft_config())
    finally:
        checkpointer.close()
    return _Reference(tmp_path / "lora_checkpoint" / "model", state, logits)


def _load_without_adapters(directory: Path, cache: Path) -> Ernie5_5_MoeForCausalLM:
    """Read a Hugging Face checkpoint directory with the public loader as a model without adapters."""
    # The loader asks CUDA for the current device even for a CPU model. Nothing of the load path is replaced.
    with patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")):
        model = NeMoAutoModelForCausalLM.from_config(
            str(directory),
            load_base_model=True,
            backend=_eager_backend(),
            attn_implementation="eager",
            torch_dtype=torch.float32,
            force_hf=False,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            cache_dir=str(cache),
        )
    return model.eval()


def test_export_writes_a_checkpoint_that_loads_without_adapters(tmp_path: Path, reference: _Reference) -> None:
    """The export serializes the storage views that the checkpoint adapter returns for an unsharded model."""
    # The backend of ``tools/merge_lora.py --automodel-native-export --backend attn=sdpa experts=torch dispatcher=torch``.
    backend = BackendConfig(attn="sdpa", linear="torch", experts="torch", dispatcher="torch")

    output_dir = export_merged_peft_checkpoint(
        _lora_model(backend), adapter_path=reference.adapter_dir, output_dir=tmp_path / "merged"
    )

    merged = _load_without_adapters(output_dir, tmp_path / "cache")
    state = merged.state_dict()
    assert state.keys() == reference.state.keys()
    for name, value in reference.state.items():
        torch.testing.assert_close(state[name], value, msg=lambda message: f"{name}: {message}")
    # The LoRA model adds the update to the activations and the merged model to the weights, so the logits
    # agree up to rounding.
    with torch.no_grad():
        torch.testing.assert_close(merged(_tokens(LENGTH)).logits, reference.logits, rtol=1e-4, atol=1e-4)

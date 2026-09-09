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

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import PretrainedConfig

from nemo_automodel import export_merged_peft_checkpoint
from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts

_NUM_EXPERTS = 4
_REQUESTED_RANK = 6
_ACTIVATED_EXPERTS = 2
_EXPERT_RANK = _REQUESTED_RANK // _ACTIVATED_EXPERTS
_HIDDEN = 8
_INTERMEDIATE = 6


def _moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=_NUM_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=_ACTIVATED_EXPERTS,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_HIDDEN,
        inter_dim=2 * _HIDDEN,
        moe_inter_dim=_INTERMEDIATE,
        norm_topk_prob=False,
        expert_activation="relu2",
        dtype=torch.float32,
    )


class _TinyAutoModelPeftModel(nn.Module):
    def __init__(self, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.config = PretrainedConfig(
            architectures=["NemotronHForCausalLM"],
            hidden_size=_HIDDEN,
            num_hidden_layers=1,
        )
        backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", dispatcher="torch")
        moe_config = _moe_config()

        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        mixer = nn.Module()
        mixer.in_proj = LinearLoRA(
            nn.Linear(_HIDDEN, _HIDDEN, bias=False),
            dim=_REQUESTED_RANK,
            alpha=2 * _REQUESTED_RANK,
            dropout=dropout,
            use_memory_efficient_lora=False,
        )
        experts = GroupedExperts(moe_config, backend)
        mixer.experts = GroupedExpertsLoRA(
            experts,
            lora_dim=_EXPERT_RANK,
            alpha=2 * _REQUESTED_RANK,
        )
        self.model.layers[0].mixer = mixer
        self.state_dict_adapter = NemotronV3StateDictAdapter(
            config=SimpleNamespace(num_hidden_layers=1),
            moe_config=moe_config,
            backend=backend,
            dtype=torch.float32,
        )


def _write_legacy_automodel_adapter(model: _TinyAutoModelPeftModel, adapter_dir) -> dict[str, torch.Tensor]:
    """Write a pre-0.19 ParamWrapper checkpoint from native AutoModel LoRA tensors."""
    adapter_dir.mkdir()
    native_state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if any(part.startswith("lora_") for part in key.split("."))
    }
    peft_state = ModelState(model, is_peft=True).state_dict()
    hf_state = model.state_dict_adapter.to_hf(peft_state, legacy_paramwrapper_layout=True)
    save_file({key: value.contiguous() for key, value in hf_state.items()}, adapter_dir / "adapter_model.safetensors")
    (adapter_dir / "automodel_peft_config.json").write_text(
        json.dumps({"moe_rank_scaling": True, "paramwrapper_layout": "peft-0.18"}),
        encoding="utf-8",
    )
    return native_state


def _load_sharded_state_dict(output_dir) -> dict[str, torch.Tensor]:
    index = json.loads((output_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    state_dict = {}
    for filename in sorted(set(index["weight_map"].values())):
        state_dict.update(load_file(output_dir / filename))
    return state_dict


def test_export_restores_moe_rank_scaled_peft_v5_and_writes_canonical_hf_shards(tmp_path):
    """The public export bypasses PEFT's requested-rank expert construction."""
    torch.manual_seed(1234)
    model = _TinyAutoModelPeftModel(dropout=0.1)
    model.train()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()

    base_dense = model.model.layers[0].mixer.in_proj.weight.detach().clone()
    base_gate_up = model.model.layers[0].mixer.experts.gate_and_up_projs.detach().clone()
    base_down = model.model.layers[0].mixer.experts.down_projs.detach().clone()
    adapter_dir = tmp_path / "adapter"
    native_lora = _write_legacy_automodel_adapter(model, adapter_dir)

    legacy_checkpoint = load_file(adapter_dir / "adapter_model.safetensors")
    folded_rank_key = "base_model.model.backbone.layers.0.mixer.experts.base_layer.lora_A.weight"
    assert legacy_checkpoint[folded_rank_key].shape[0] == _NUM_EXPERTS * _EXPERT_RANK
    assert legacy_checkpoint[folded_rank_key].shape[0] != _NUM_EXPERTS * _REQUESTED_RANK

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(part.startswith("lora_") for part in name.split(".")):
                parameter.zero_()

    output_dir = export_merged_peft_checkpoint(
        model,
        adapter_path=adapter_dir,
        output_dir=tmp_path / "merged",
        max_shard_size=300,
    )
    assert model.training is False
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "model.safetensors.index.json").is_file()

    exported = _load_sharded_state_dict(output_dir)
    assert not any("lora_" in key for key in exported)
    assert "backbone.layers.0.mixer.in_proj.weight" in exported
    assert "backbone.layers.0.mixer.experts.0.up_proj.weight" in exported
    assert "backbone.layers.0.mixer.experts.0.down_proj.weight" in exported

    dense_prefix = "model.layers.0.mixer.in_proj"
    expected_dense = base_dense + (
        native_lora[f"{dense_prefix}.lora_B.weight"] @ native_lora[f"{dense_prefix}.lora_A.weight"]
    ) * (2 * _REQUESTED_RANK / _REQUESTED_RANK)
    torch.testing.assert_close(exported["backbone.layers.0.mixer.in_proj.weight"], expected_dense)

    expert_prefix = "model.layers.0.mixer.experts"
    expected_gate_up = base_gate_up + torch.bmm(
        native_lora[f"{expert_prefix}.lora_gate_and_up_A"],
        native_lora[f"{expert_prefix}.lora_gate_and_up_B"],
    ) * (2 * _REQUESTED_RANK / _EXPERT_RANK)
    expected_down = base_down + torch.bmm(
        native_lora[f"{expert_prefix}.lora_down_A"],
        native_lora[f"{expert_prefix}.lora_down_B"],
    ) * (2 * _REQUESTED_RANK / _EXPERT_RANK)
    for expert_id in range(_NUM_EXPERTS):
        torch.testing.assert_close(
            exported[f"backbone.layers.0.mixer.experts.{expert_id}.up_proj.weight"],
            expected_gate_up[expert_id].transpose(0, 1),
        )
        torch.testing.assert_close(
            exported[f"backbone.layers.0.mixer.experts.{expert_id}.down_proj.weight"],
            expected_down[expert_id].transpose(0, 1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required to exercise in-place expert views")
def test_export_cuda_grouped_experts_writes_serializable_tensors(tmp_path):
    """CUDA export materializes grouped-expert transposes before safetensors serialization."""
    torch.manual_seed(1234)
    model = _TinyAutoModelPeftModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()

    adapter_dir = tmp_path / "adapter"
    _write_legacy_automodel_adapter(model, adapter_dir)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if any(part.startswith("lora_") for part in name.split(".")):
                parameter.zero_()
    model.cuda()

    output_dir = export_merged_peft_checkpoint(
        model,
        adapter_path=adapter_dir,
        output_dir=tmp_path / "merged",
        max_shard_size=300,
    )

    exported = _load_sharded_state_dict(output_dir)
    assert "backbone.layers.0.mixer.experts.0.up_proj.weight" in exported
    assert "backbone.layers.0.mixer.experts.0.down_proj.weight" in exported

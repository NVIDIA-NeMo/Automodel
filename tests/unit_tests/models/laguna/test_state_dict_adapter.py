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

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.laguna.config import LagunaConfig
from nemo_automodel.components.models.laguna.state_dict_adapter import LagunaStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _adapter() -> LagunaStateDictAdapter:
    config = LagunaConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=3,
        mlp_layer_types=["dense", "sparse"],
        torch_dtype="float32",
    )
    moe_config = MoEConfig(
        dim=4,
        inter_dim=8,
        moe_inter_dim=3,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=2.5,
        norm_topk_prob=True,
        force_e_score_correction_bias=True,
        dtype=torch.float32,
    )
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
    )
    return LagunaStateDictAdapter(config, moe_config, backend, dtype=torch.float32)


def test_laguna_adapter_merges_hf_split_experts_and_renames_shared_expert():
    adapter = _adapter()
    hf_state = {
        "model.layers.1.mlp.gate.weight": torch.ones(2, 4),
        "model.layers.1.mlp.experts.e_score_correction_bias": torch.arange(2, dtype=torch.float32),
        "model.layers.1.mlp.shared_expert.gate_proj.weight": torch.ones(3, 4),
        "model.layers.1.mlp.shared_expert.up_proj.weight": torch.ones(3, 4) * 2,
        "model.layers.1.mlp.shared_expert.down_proj.weight": torch.ones(4, 3),
    }
    for expert_id in range(2):
        base = f"model.layers.1.mlp.experts.{expert_id}"
        hf_state[f"{base}.gate_proj.weight"] = torch.full((3, 4), float(expert_id + 1))
        hf_state[f"{base}.up_proj.weight"] = torch.full((3, 4), float(expert_id + 3))
        hf_state[f"{base}.down_proj.weight"] = torch.full((4, 3), float(expert_id + 5))

    native = adapter.from_hf(dict(hf_state))

    assert native["model.layers.1.mlp.experts.gate_and_up_projs"].shape == (2, 4, 6)
    assert native["model.layers.1.mlp.experts.down_projs"].shape == (2, 3, 4)
    assert "model.layers.1.mlp.shared_experts.gate_proj.weight" in native
    assert "model.layers.1.mlp.gate.e_score_correction_bias" in native
    torch.testing.assert_close(
        native["model.layers.1.mlp.experts.gate_and_up_projs"][0, :, :3],
        hf_state["model.layers.1.mlp.experts.0.gate_proj.weight"].T,
    )
    torch.testing.assert_close(
        native["model.layers.1.mlp.experts.down_projs"][1],
        hf_state["model.layers.1.mlp.experts.1.down_proj.weight"].T,
    )


def test_laguna_adapter_splits_native_experts_back_to_hf_names():
    adapter = _adapter()
    gate_and_up = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(2, 4, 6)
    down = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    hf = adapter.to_hf(
        {
            "model.layers.1.mlp.experts.gate_and_up_projs": gate_and_up,
            "model.layers.1.mlp.experts.down_projs": down,
            "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.ones(3, 4),
            "model.layers.1.mlp.gate.e_score_correction_bias": torch.arange(2, dtype=torch.float32),
        }
    )

    assert "model.layers.1.mlp.experts.0.gate_proj.weight" in hf
    assert "model.layers.1.mlp.experts.0.up_proj.weight" in hf
    assert "model.layers.1.mlp.experts.1.down_proj.weight" in hf
    assert "model.layers.1.mlp.shared_expert.gate_proj.weight" in hf
    assert "model.layers.1.mlp.experts.e_score_correction_bias" in hf
    torch.testing.assert_close(hf["model.layers.1.mlp.experts.0.gate_proj.weight"], gate_and_up[0, :, :3].T)
    torch.testing.assert_close(hf["model.layers.1.mlp.experts.0.up_proj.weight"], gate_and_up[0, :, 3:].T)
    torch.testing.assert_close(hf["model.layers.1.mlp.experts.1.down_proj.weight"], down[1].T)

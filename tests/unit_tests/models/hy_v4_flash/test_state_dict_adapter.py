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

"""State-dict adapter: released key names and layouts against the native ones."""

import torch

from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM
from nemo_automodel.components.models.hy_v4_flash.state_dict_adapter import HYV4FlashStateDictAdapter
from tests.unit_tests.models.hy_v4_flash.tiny import CPU_BACKEND, tiny_config


def _model() -> HYV4FlashForCausalLM:
    torch.manual_seed(0)
    model = HYV4FlashForCausalLM(tiny_config(vision={"num_hidden_layers": 0}), backend=CPU_BACKEND).float().eval()
    model.initialize_weights(dtype=torch.float32)
    return model


def test_adapter_round_trip_restores_released_names_and_values() -> None:
    model = _model()
    adapter = HYV4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    native = {k: v.detach().clone() for k, v in model.state_dict().items()}
    released = adapter.to_hf(native)
    assert not [k for k in released if "sinks_param" in k or "fc1_latent" in k or "attn_hc" in k], "native names leaked"
    assert "model.layers.0.self_attn.learnable_sink_param" in released
    assert (
        "model.layers.0.mlp.post_fc2_norm.weight" in released
        and "model.layers.1.mlp.combine_output_norm.weight" in released
    )
    assert released["model.layers.1.mlp.experts.gate_up_proj"].shape == (8, 64, 24)  # [experts, 2 * inter, latent]
    again = adapter.from_hf({k: v.clone() for k, v in released.items()})
    assert set(again) == set(native)
    for key, value in native.items():
        torch.testing.assert_close(again[key], value, rtol=0, atol=0, msg=lambda m, key=key: f"{key}: {m}")


def test_released_expert_layout_is_transposed_not_reshaped() -> None:
    """gate_up_proj[e] is [2 * inter, latent] (gate rows first); native gate_and_up_projs[e] is its transpose."""
    model = _model()
    adapter = HYV4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    released = torch.arange(8 * 64 * 24, dtype=torch.float32).reshape(8, 64, 24)
    native = adapter.from_hf({"model.layers.1.mlp.experts.gate_up_proj": released})
    assert torch.equal(native["model.layers.1.mlp.experts.gate_and_up_projs"][3], released[3].T)


def test_drafter_and_extra_state_keys_are_skipped_in_both_directions() -> None:
    """The released checkpoint carries 87 ``model.dfly_layers.*`` keys (the serving-only drafter) the model has no
    parameters for; TE ``_extra_state`` buffers have no released counterpart."""
    model = _model()
    adapter = HYV4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    kept = "model.layers.0.self_attn.linear_gate.weight"
    native = adapter.from_hf(
        {
            "model.dfly_layers.0.input_layernorm.weight": torch.ones(4),
            "model.dfly_layers.1.mlp.down_proj.weight": torch.ones(2, 2),
            "model.layers.0.mlp._extra_state": torch.ones(1),
            kept: torch.ones(3, 3),
        }
    )
    assert set(native) == {"model.layers.0.self_attn.gate_proj.weight"}
    assert adapter.convert_single_tensor_to_hf("model.dfly_layers.0.input_layernorm.weight", torch.ones(4)) == []
    assert adapter.convert_single_tensor_to_hf("model.layers.0.mlp._extra_state", torch.ones(1)) == []
    assert adapter.forced_hf_dtype_mapping({"model.dfly_layers.0.hc_fn": torch.ones(2, dtype=torch.float32)}) == {}

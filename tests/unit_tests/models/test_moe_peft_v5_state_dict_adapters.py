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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _make_transformers_model(family: str, num_experts: int, dim: int, inter_dim: int) -> nn.Module:
    """Instantiate a tiny causal LM using the actual Transformers v5 module hierarchy."""
    if family == "nemotron_v3":
        from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
        from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

        config = NemotronHConfig(
            vocab_size=32,
            layers_block_type=["moe"],
            n_routed_experts=num_experts,
            hidden_size=dim,
            moe_intermediate_size=inter_dim,
            moe_shared_expert_intermediate_size=inter_dim,
            moe_latent_size=None,
            mlp_hidden_act="relu2",
            num_experts_per_tok=1,
            n_group=1,
            topk_group=1,
            use_mamba_kernels=False,
        )
        return NemotronHForCausalLM(config)
    else:
        from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
        from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2ForCausalLM

        config = MiniMaxM2Config(
            vocab_size=32,
            num_local_experts=num_experts,
            hidden_size=dim,
            intermediate_size=inter_dim,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=dim // 2,
            max_position_embeddings=32,
            num_experts_per_tok=1,
            hidden_act="silu",
            use_cache=False,
        )
        return MiniMaxM2ForCausalLM(config)


def _make_moe_config(*, gated: bool) -> MoEConfig:
    return MoEConfig(
        dim=16,
        inter_dim=32,
        moe_inter_dim=8,
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_activation="swiglu" if gated else "relu2",
        dtype=torch.float32,
    )


def _make_adapter_and_state(family: str, rank: int):
    gated = family == "minimax_m2"
    moe_config = _make_moe_config(gated=gated)
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", dispatcher="torch")
    if gated:
        adapter = MiniMaxM2StateDictAdapter(SimpleNamespace(), moe_config, backend, dtype=torch.float32)
        expert_path = "mlp.experts"
    else:
        adapter = NemotronV3StateDictAdapter(
            SimpleNamespace(num_hidden_layers=1), moe_config, backend, dtype=torch.float32
        )
        expert_path = "mixer.experts"

    base = f"base_model.model.model.layers.0.{expert_path}"
    input_width = 2 * moe_config.moe_inter_dim if gated else moe_config.moe_inter_dim
    state_dict = {
        f"{base}.lora_gate_and_up_A": torch.randn(moe_config.n_routed_experts, moe_config.dim, rank),
        f"{base}.lora_gate_and_up_B": torch.randn(moe_config.n_routed_experts, rank, input_width),
        f"{base}.lora_down_A": torch.randn(moe_config.n_routed_experts, moe_config.moe_inter_dim, rank),
        f"{base}.lora_down_B": torch.randn(moe_config.n_routed_experts, rank, moe_config.dim),
    }
    return adapter, moe_config, state_dict


def _paramwrapper_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, num_experts: int, scale: float):
    """Compute the grouped expert delta represented by PEFT ParamWrapper tensors.

    Args:
        lora_a: Tensor of shape [rank * experts, input].
        lora_b: Tensor of shape [output, rank * experts].
        num_experts: Number of experts folded into the rank axes.
        scale: LoRA scaling factor.

    Returns:
        Tensor of shape [experts, input, output].
    """
    lora_a = lora_a.reshape(num_experts, -1, lora_a.shape[-1])
    lora_b = lora_b.reshape(lora_b.shape[0], -1, num_experts)
    return torch.einsum("ore,eri->eio", lora_b, lora_a) * scale


@pytest.mark.parametrize("family", ["nemotron_v3", "minimax_m2"])
def test_peft_v5_load_merge_and_adapter_round_trip(family, tmp_path):
    """The model adapter emits loadable ParamWrapper keys and restores every tensor."""
    from peft import LoraConfig, PeftModel, TaskType
    from safetensors.torch import save_file

    torch.manual_seed(42)
    rank = 4
    alpha = 8
    adapter, moe_config, native_state_dict = _make_adapter_and_state(family, rank)
    hf_state_dict = adapter.to_hf(dict(native_state_dict), quantization=family == "minimax_m2")

    if family == "nemotron_v3":
        expert_path = "mixer.experts"
        input_projection = "up_proj"
    else:
        expert_path = "mlp.experts"
        input_projection = "gate_up_proj"
    hf_parent = f"base_model.model.model.layers.0.{expert_path}"
    model_parent = f"model.layers.0.{expert_path}"

    assert set(hf_state_dict) == {
        f"{hf_parent}.base_layer.lora_A.weight",
        f"{hf_parent}.base_layer.lora_B.weight",
        f"{hf_parent}.lora_A.weight",
        f"{hf_parent}.lora_B.weight",
    }

    LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=[],
        target_parameters=list(adapter._v5_peft_target_parameters),
        bias="none",
    ).save_pretrained(tmp_path)
    save_file(hf_state_dict, str(tmp_path / "adapter_model.safetensors"))

    hf_model = _make_transformers_model(family, moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim)
    base_weights = {
        name: parameter.detach().clone()
        for name, parameter in hf_model.named_parameters()
        if name.startswith(model_parent)
    }
    hidden_states = torch.randn(3, moe_config.dim)
    top_k_index = torch.tensor([[0], [1], [0]])
    top_k_weights = torch.ones_like(top_k_index, dtype=hidden_states.dtype)
    with torch.no_grad():
        base_output = hf_model.get_submodule(model_parent)(hidden_states, top_k_index, top_k_weights)

    peft_model = PeftModel.from_pretrained(hf_model, str(tmp_path))
    loaded_parameters = dict(peft_model.named_parameters())
    for key, expected in hf_state_dict.items():
        loaded_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
            ".lora_B.weight", ".lora_B.default.weight"
        )
        assert loaded_key in loaded_parameters, f"PEFT silently dropped {key}"
        torch.testing.assert_close(loaded_parameters[loaded_key], expected)
    with torch.no_grad():
        adapted_output = peft_model.get_submodule(f"base_model.model.{model_parent}")(
            hidden_states, top_k_index, top_k_weights
        )
    assert not torch.allclose(adapted_output, base_output)

    merged = peft_model.merge_and_unload()
    merged_parameters = dict(merged.named_parameters())
    scale = alpha / rank
    input_delta = _paramwrapper_delta(
        hf_state_dict[f"{hf_parent}.base_layer.lora_A.weight"],
        hf_state_dict[f"{hf_parent}.base_layer.lora_B.weight"],
        moe_config.n_routed_experts,
        scale,
    )
    down_delta = _paramwrapper_delta(
        hf_state_dict[f"{hf_parent}.lora_A.weight"],
        hf_state_dict[f"{hf_parent}.lora_B.weight"],
        moe_config.n_routed_experts,
        scale,
    )
    torch.testing.assert_close(
        merged_parameters[f"{model_parent}.{input_projection}"],
        base_weights[f"{model_parent}.{input_projection}"] + input_delta,
    )
    torch.testing.assert_close(
        merged_parameters[f"{model_parent}.down_proj"],
        base_weights[f"{model_parent}.down_proj"] + down_delta,
    )
    with torch.no_grad():
        merged_output = merged.get_submodule(model_parent)(hidden_states, top_k_index, top_k_weights)
    torch.testing.assert_close(merged_output, adapted_output)
    assert not any("lora_" in name for name in merged_parameters)

    restored_state_dict = adapter.from_hf(dict(hf_state_dict))
    assert set(restored_state_dict) == set(native_state_dict)
    for key, expected in native_state_dict.items():
        torch.testing.assert_close(restored_state_dict[key], expected)

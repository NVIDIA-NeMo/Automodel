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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.addons import _get_hf_peft_config, _get_paramwrapper_layout_stamp
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m2.model import MiniMaxM2ForCausalLM as NeMoMiniMaxM2ForCausalLM
from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]
_MINIMAX_RECIPE = _REPO_ROOT / "examples/llm_finetune/minimax_m2/minimax_m2.7_hellaswag_lora.yaml"

# Families validated for fused PEFT v5 ParamWrapper export, mapped to their HF
# expert module path and the name of the fused input projection parameter.
_FAMILIES = {
    "nemotron_v3": ("mixer.experts", "up_proj"),
    "minimax_m2": ("mlp.experts", "gate_up_proj"),
    "qwen3_moe": ("mlp.experts", "gate_up_proj"),
}


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
    elif family == "qwen3_moe":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

        config = Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=dim,
            intermediate_size=2 * dim,
            moe_intermediate_size=inter_dim,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=dim // 2,
            num_experts=num_experts,
            num_experts_per_tok=1,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            norm_topk_prob=False,
            max_position_embeddings=32,
            use_cache=False,
        )
        return Qwen3MoeForCausalLM(config)
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
    """Build the family's adapter and a random native grouped LoRA state dict.

    Native tensor layout (E experts, H model dim, U fused input width, I moe
    intermediate, r rank): ``lora_gate_and_up_A`` [E, H, r],
    ``lora_gate_and_up_B`` [E, r, U], ``lora_down_A`` [E, I, r],
    ``lora_down_B`` [E, r, H]; the keys carry the ``base_model.model.``
    prefix the PEFT save path adds.
    """
    gated = family != "nemotron_v3"
    moe_config = _make_moe_config(gated=gated)
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", dispatcher="torch")
    if family == "minimax_m2":
        adapter = MiniMaxM2StateDictAdapter(SimpleNamespace(), moe_config, backend, dtype=torch.float32)
    elif family == "qwen3_moe":
        adapter = Qwen3MoeStateDictAdapter(
            SimpleNamespace(num_hidden_layers=1), moe_config, backend, dtype=torch.float32
        )
    else:
        adapter = NemotronV3StateDictAdapter(
            SimpleNamespace(num_hidden_layers=1), moe_config, backend, dtype=torch.float32
        )
        # This fixture uses Transformers v5's native ``model.*`` hierarchy;
        # remote-code Nemotron-H checkpoints instead select ``backbone.*``.
        adapter._uses_model_prefix = True
    expert_path, _ = _FAMILIES[family]

    base = f"base_model.model.model.layers.0.{expert_path}"
    input_width = 2 * moe_config.moe_inter_dim if gated else moe_config.moe_inter_dim
    state_dict = {
        f"{base}.lora_gate_and_up_A": torch.randn(moe_config.n_routed_experts, moe_config.dim, rank),
        f"{base}.lora_gate_and_up_B": torch.randn(moe_config.n_routed_experts, rank, input_width),
        f"{base}.lora_down_A": torch.randn(moe_config.n_routed_experts, moe_config.moe_inter_dim, rank),
        f"{base}.lora_down_B": torch.randn(moe_config.n_routed_experts, rank, moe_config.dim),
    }
    return adapter, moe_config, state_dict


def _expected_hf_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float) -> torch.Tensor:
    """Delta the native grouped LoRA math implies for the HF fused expert weight.

    ``GroupedExpertsLoRA`` computes ``x @ W[e] + (x @ A[e] @ B[e]) * scale`` with
    ``A`` [E, in, r] and ``B`` [E, r, out], while the HF fused parameter stores
    each expert as [out, in] and applies it through ``F.linear``. The delta that
    must land on the HF weight is therefore the per-expert transpose of
    ``A[e] @ B[e]``. This is computed from the native tensors alone, independent
    of the export converter and of peft's own delta formula.

    Args:
        lora_a: Native A tensor of shape [E, in, r].
        lora_b: Native B tensor of shape [E, r, out].
        scale: LoRA scaling factor (alpha / rank).

    Returns:
        Tensor of shape [E, out, in], matching the HF fused parameter.
    """
    return torch.einsum("eir,ero->eoi", lora_a, lora_b) * scale


def _pre_flip_peft_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, num_experts: int, scale: float) -> torch.Tensor:
    """Delta peft <= 0.19.0 merges for a 3-D ParamWrapper parameter.

    Before huggingface/peft#3165 peft read a fused expert parameter as
    [E, in, out]; peft 0.20 still ships that reading as the non-swapped branch
    of ``ParamWrapper.get_delta_weight``. No pre-flip peft can be installed
    next to the test floor, so this mirrors that branch to check the legacy
    export against the native math instead of against its own importer.

    Args:
        lora_a: Legacy-layout ``lora_A.weight`` of shape [r * E, in].
        lora_b: Legacy-layout ``lora_B.weight`` of shape [out, r * E].
        num_experts: Number of experts folded into the rank axes.
        scale: LoRA scaling factor (alpha / rank).

    Returns:
        Tensor of shape [E, in, out], the pre-flip reading of the parameter.
    """
    lora_a = lora_a.reshape(num_experts, -1, lora_a.shape[-1])
    lora_b = lora_b.reshape(lora_b.shape[0], -1, num_experts)
    return torch.einsum("ore,eri->eio", lora_b, lora_a) * scale


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_peft_merges_the_same_delta_as_the_native_lora_math(family, tmp_path):
    """Real peft loads the exported adapter and merges exactly the native delta.

    The expected delta comes from the native A/B tensors, not from the exported
    file, so a converter that transposed A and B consistently in both directions
    (invisible to a to_hf -> from_hf round trip) fails the merge comparison.
    """
    pytest.importorskip("peft", minversion="0.19.1")
    from peft import LoraConfig, PeftModel, TaskType
    from safetensors.torch import save_file

    torch.manual_seed(42)
    rank = 4
    alpha = 8
    adapter, moe_config, native_state_dict = _make_adapter_and_state(family, rank)
    hf_state_dict = adapter.to_hf(dict(native_state_dict), quantization=family == "minimax_m2")

    expert_path, input_projection = _FAMILIES[family]
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

    merged = peft_model.merge_and_unload()
    merged_parameters = dict(merged.named_parameters())
    scale = alpha / rank
    expected_input_delta = _expected_hf_delta(
        native_state_dict[f"{hf_parent}.lora_gate_and_up_A"],
        native_state_dict[f"{hf_parent}.lora_gate_and_up_B"],
        scale,
    )
    expected_down_delta = _expected_hf_delta(
        native_state_dict[f"{hf_parent}.lora_down_A"],
        native_state_dict[f"{hf_parent}.lora_down_B"],
        scale,
    )
    torch.testing.assert_close(
        merged_parameters[f"{model_parent}.{input_projection}"],
        base_weights[f"{model_parent}.{input_projection}"] + expected_input_delta,
    )
    torch.testing.assert_close(
        merged_parameters[f"{model_parent}.down_proj"],
        base_weights[f"{model_parent}.down_proj"] + expected_down_delta,
    )
    with torch.no_grad():
        merged_output = merged.get_submodule(model_parent)(hidden_states, top_k_index, top_k_weights)
    torch.testing.assert_close(merged_output, adapted_output)
    assert not any("lora_" in name for name in merged_parameters)

    restored_state_dict = adapter.from_hf(dict(hf_state_dict))
    assert set(restored_state_dict) == set(native_state_dict)
    for key, expected in native_state_dict.items():
        torch.testing.assert_close(restored_state_dict[key], expected)


@pytest.mark.parametrize("family", sorted(_FAMILIES))
def test_legacy_layout_option_exports_and_reloads_under_its_stamp(family):
    """``legacy_paramwrapper_layout`` reaches every family's export path.

    Qwen3 overrides ``convert_single_tensor_to_hf``; if that override dropped
    the option, the file would carry the modern layout under a ``peft-0.18``
    stamp: an older peft could not load it, and the loader here rejects the
    stamp/shape mismatch instead of reading the tensors transposed.
    """
    torch.manual_seed(0)
    rank = 4
    scale = 2.0
    adapter, moe_config, native_state_dict = _make_adapter_and_state(family, rank)
    expert_path, _ = _FAMILIES[family]
    hf_parent = f"base_model.model.model.layers.0.{expert_path}"
    n_experts, dim, moe_inter = moe_config.n_routed_experts, moe_config.dim, moe_config.moe_inter_dim
    input_width = 2 * moe_inter if family != "nemotron_v3" else moe_inter
    quantization = family == "minimax_m2"

    legacy = adapter.to_hf(dict(native_state_dict), quantization=quantization, legacy_paramwrapper_layout=True)
    modern = adapter.to_hf(dict(native_state_dict), quantization=quantization)

    # Pre-flip layout: lora_A carries the output features, lora_B the input features.
    assert legacy[f"{hf_parent}.base_layer.lora_A.weight"].shape == (rank * n_experts, input_width)
    assert legacy[f"{hf_parent}.base_layer.lora_B.weight"].shape == (dim, rank * n_experts)
    assert legacy[f"{hf_parent}.lora_A.weight"].shape == (rank * n_experts, dim)
    assert legacy[f"{hf_parent}.lora_B.weight"].shape == (moe_inter, rank * n_experts)
    assert modern[f"{hf_parent}.lora_A.weight"].shape == (rank * n_experts, moe_inter)

    # Read the way pre-flip peft reads them, the legacy tensors must merge the
    # same delta the native tensors define (checked against the native math,
    # not against this converter's own importer).
    torch.testing.assert_close(
        _pre_flip_peft_delta(
            legacy[f"{hf_parent}.base_layer.lora_A.weight"],
            legacy[f"{hf_parent}.base_layer.lora_B.weight"],
            n_experts,
            scale,
        ),
        _expected_hf_delta(
            native_state_dict[f"{hf_parent}.lora_gate_and_up_A"],
            native_state_dict[f"{hf_parent}.lora_gate_and_up_B"],
            scale,
        ),
    )
    torch.testing.assert_close(
        _pre_flip_peft_delta(
            legacy[f"{hf_parent}.lora_A.weight"], legacy[f"{hf_parent}.lora_B.weight"], n_experts, scale
        ),
        _expected_hf_delta(
            native_state_dict[f"{hf_parent}.lora_down_A"], native_state_dict[f"{hf_parent}.lora_down_B"], scale
        ),
    )

    # A legacy export reloads under the stamp the save path writes for it.
    adapter._paramwrapper_layout_hint = _get_paramwrapper_layout_stamp(adapter, False, True)
    try:
        restored = adapter.from_hf(dict(legacy))
    finally:
        adapter._paramwrapper_layout_hint = None
    assert set(restored) == set(native_state_dict)
    for key, expected in native_state_dict.items():
        torch.testing.assert_close(restored[key], expected)


def test_minimax_recipe_does_not_advertise_untrained_expert_adapters():
    """The dense-only MiniMax recipe must not advertise PEFT v5 expert parameters."""
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config

    with _MINIMAX_RECIPE.open(encoding="utf-8") as recipe_file:
        recipe_peft = yaml.safe_load(recipe_file)["peft"]
    peft_values = {key: value for key, value in recipe_peft.items() if key != "_target_"}
    peft_values["use_triton"] = False
    peft_config = PeftConfig(**peft_values)

    assert peft_config.target_modules == []
    assert peft_config.match_all_linear is True

    config = MiniMaxM2Config(
        vocab_size=32,
        num_local_experts=2,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_experts_per_tok=1,
        hidden_act="silu",
        use_cache=False,
        torch_dtype="float32",
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
    )
    model = NeMoMiniMaxM2ForCausalLM(config, backend=backend)
    apply_lora_to_linear_modules(model, peft_config)

    assert not isinstance(model.model.layers["0"].mlp.experts, GroupedExpertsLoRA)
    model_state = ModelState(model, is_peft=True)
    native_adapter_state = model_state.state_dict()
    hf_adapter_state = model.state_dict_adapter.to_hf(native_adapter_state, v4_compatible=False)
    expert_prefix = "base_model.model.model.layers.0.mlp.experts"
    assert any("lora_" in key for key in hf_adapter_state)
    assert not [key for key in hf_adapter_state if key.startswith(expert_prefix)]

    hf_peft_config = _get_hf_peft_config(peft_config, model_state)
    assert hf_peft_config["target_modules"]
    assert "target_parameters" not in hf_peft_config

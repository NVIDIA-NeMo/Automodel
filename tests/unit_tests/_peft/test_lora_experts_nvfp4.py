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

import json

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components._peft.lora_experts_nvfp4 import GroupedExpertsLoRANVFP4
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import GroupedExperts
from nemo_automodel.components.moe.quantized_experts import (
    GroupedExpertsNVFP4,
    apply_nvfp4_to_moe_experts,
    validate_nvfp4_checkpoint_config,
)
from nemo_automodel.components.quantization.nvfp4 import (
    NVFP4_BLOCK_SIZE,
    dequantize_nvfp4,
    quantize_nvfp4,
    quantize_nvfp4_modelopt,
)

BASE_NAMES = ("gate_and_up_projs", "down_projs")
PACKED_NAMES = {
    "gate_and_up_projs_nvfp4",
    "gate_and_up_projs_scales",
    "gate_and_up_projs_global",
    "down_projs_nvfp4",
    "down_projs_scales",
    "down_projs_global",
}
LORA_NAMES = {"lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B"}


@pytest.fixture
def device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


@pytest.fixture
def moe_config():
    # Both contraction dims divisible by the NVFP4 block (16) and grouped-GEMM aligned.
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=64,
        inter_dim=128,
        moe_inter_dim=64,
        norm_topk_prob=False,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


def _grouped_mm_available(device: torch.device) -> bool:
    x = torch.randn(4, 64, dtype=torch.bfloat16, device=device)
    w = torch.randn(2, 64, 128, dtype=torch.bfloat16, device=device)
    try:
        torch._grouped_mm(x, w, offs=torch.tensor([2, 4], dtype=torch.int32, device=device))
    except (RuntimeError, NotImplementedError):
        return False
    return True


def _materialized_experts(moe_config, device) -> GroupedExperts:
    orig = GroupedExperts(moe_config).to(device)
    orig.use_torch_mm = True
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    return orig


def _make_representable_(experts: GroupedExperts) -> None:
    """Replace expert weights in-place with their NVFP4 round-trip so the packed storage is exact."""
    with torch.no_grad():
        for name in BASE_NAMES:
            param = getattr(experts, name)
            packed, scales, global_scale = quantize_nvfp4(param.data.transpose(-2, -1).contiguous())
            param.data.copy_(dequantize_nvfp4(packed, scales, global_scale, param.dtype).transpose(-2, -1))


def _routing_inputs(moe_config, num_tokens, device, dtype):
    torch.manual_seed(7)
    x = torch.randn(num_tokens, moe_config.dim, dtype=dtype, device=device, requires_grad=True)
    indices = torch.stack(
        [
            torch.randperm(moe_config.n_routed_experts, device=device)[: moe_config.n_activated_experts]
            for _ in range(num_tokens)
        ]
    )
    weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=dtype, device=device)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    return x, token_mask, weights, indices


def test_nvfp4_module_packs_base_weights(moe_config, device):
    lora = GroupedExpertsLoRANVFP4(_materialized_experts(moe_config, device), lora_dim=8, alpha=16)

    assert lora._nvfp4_resident
    assert not hasattr(lora, "gate_and_up_projs") and not hasattr(lora, "down_projs")
    assert lora.gate_and_up_projs_nvfp4.dtype == torch.uint8
    assert lora.gate_and_up_projs_scales.dtype == torch.float8_e4m3fn
    assert lora.gate_and_up_projs_global.dtype == torch.float32
    assert not lora.gate_and_up_projs_nvfp4.requires_grad
    # Checkpoint orientation [E, out, in]: codes pack the contraction dim, one e4m3 scale per block, one row scale.
    assert lora.gate_and_up_projs_nvfp4.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 2)
    assert lora.gate_and_up_projs_scales.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // NVFP4_BLOCK_SIZE)
    assert lora.gate_and_up_projs_global.shape == (4, 2 * moe_config.moe_inter_dim)
    assert lora.down_projs_nvfp4.shape == (4, moe_config.dim, moe_config.moe_inter_dim // 2)
    assert {n for n, p in lora.named_parameters() if p.requires_grad} == LORA_NAMES
    # Materialized weights were quantized at construction: no placeholder is left for a checkpoint load.
    assert not any(p.is_meta for p in lora.parameters())


def test_nvfp4_forward_backward_matches_bf16(moe_config, device):
    """With NVFP4-representable base weights, the packed module must match the bf16 module."""
    if not _grouped_mm_available(device):
        pytest.skip("torch._grouped_mm is not available on this device")
    orig = _materialized_experts(moe_config, device)
    _make_representable_(orig)

    ref = GroupedExpertsLoRA(orig, lora_dim=8, alpha=16).to(device)
    q = GroupedExpertsLoRANVFP4(orig, lora_dim=8, alpha=16).to(device)
    with torch.no_grad():
        for name in LORA_NAMES:
            getattr(q, name).data.copy_(getattr(ref, name).data)

    x_ref, token_mask, weights, indices = _routing_inputs(moe_config, 32, device, torch.bfloat16)
    x_q = x_ref.detach().clone().requires_grad_(True)

    y_ref = ref(x_ref, token_mask, weights, indices)
    y_q = q(x_q, token_mask, weights, indices)
    torch.testing.assert_close(y_q, y_ref, atol=1e-6, rtol=1e-6)

    y_ref.float().pow(2).sum().backward()
    y_q.float().pow(2).sum().backward()
    torch.testing.assert_close(x_q.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for name in LORA_NAMES:
        torch.testing.assert_close(getattr(q, name).grad, getattr(ref, name).grad, atol=1e-5, rtol=1e-5)
    assert q.gate_and_up_projs_nvfp4.grad is None and q.down_projs_nvfp4.grad is None


def test_nvfp4_no_routed_tokens(moe_config, device):
    """The all-masked dummy path must run and keep LoRA gradients flowing."""
    q = GroupedExpertsLoRANVFP4(_materialized_experts(moe_config, device), lora_dim=8, alpha=16).to(device)
    x, _, weights, indices = _routing_inputs(moe_config, 8, device, torch.bfloat16)
    token_mask = torch.zeros(8, dtype=torch.bool, device=device)

    y = q(x, token_mask, weights, indices)
    assert y.shape == x.shape
    y.float().sum().backward()
    assert x.grad is not None


def test_nvfp4_requires_torch_mm_backend(moe_config, device):
    orig = GroupedExperts(moe_config).to(device)  # default per-expert loop backend
    with torch.no_grad():
        orig.init_weights(buffer_device=device)
    with pytest.raises(NotImplementedError, match="torch_mm"):
        GroupedExpertsLoRANVFP4(orig, lora_dim=8, alpha=16)


def test_patch_moe_module_selects_nvfp4(moe_config, device):
    orig = _materialized_experts(moe_config, device)
    patched = patch_moe_module(orig, dim=8, alpha=16, expert_weight_format="nvfp4")
    assert isinstance(patched, GroupedExpertsLoRANVFP4)
    with pytest.raises(ValueError, match="'nvfp4'"):
        patch_moe_module(orig, dim=8, alpha=16, expert_weight_format="int8")


def test_frozen_passthrough_init_registers_placeholders_on_meta(moe_config):
    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
    orig.use_torch_mm = True

    q = GroupedExpertsNVFP4(orig)

    assert q._nvfp4_resident
    assert not hasattr(q, "gate_and_up_projs")
    assert q.gate_and_up_projs_nvfp4.is_meta and q.gate_and_up_projs_nvfp4.dtype == torch.uint8
    assert q.down_projs_scales.is_meta and q.down_projs_scales.dtype == torch.float8_e4m3fn
    assert q.down_projs_global.is_meta and q.down_projs_global.dtype == torch.float32
    assert q.gate_and_up_projs_nvfp4.shape == (4, 2 * moe_config.moe_inter_dim, moe_config.dim // 2)
    assert [n for n, p in q.named_parameters() if p.requires_grad] == []
    # The six packed placeholders are the module's only parameters: the base checkpoint load fills them.
    assert {n for n, _ in q.named_parameters()} == PACKED_NAMES


def test_lora_passthrough_init_placeholders_with_trainable_adapters(moe_config):
    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
        orig.use_torch_mm = True
        q = GroupedExpertsLoRANVFP4(orig, lora_dim=8, alpha=16)

    assert q._nvfp4_resident
    assert not hasattr(q, "gate_and_up_projs")
    assert q.gate_and_up_projs_nvfp4.is_meta
    assert {n for n, p in q.named_parameters() if p.requires_grad} == LORA_NAMES
    assert {n for n, p in q.named_parameters() if not p.requires_grad} == PACKED_NAMES


def test_apply_nvfp4_to_moe_experts_converts_remaining_frozen_experts(moe_config, device):
    model = nn.Module()
    model.lora_experts = GroupedExpertsLoRANVFP4(_materialized_experts(moe_config, device), lora_dim=8, alpha=16)
    model.frozen_experts = _materialized_experts(moe_config, device)
    model.linear = nn.Linear(4, 4)

    apply_nvfp4_to_moe_experts(model)

    assert isinstance(model.lora_experts, GroupedExpertsLoRANVFP4)
    assert isinstance(model.frozen_experts, GroupedExpertsNVFP4)
    assert isinstance(model.linear, nn.Linear)
    assert [n for n, p in model.frozen_experts.named_parameters() if p.requires_grad] == []


@pytest.mark.parametrize("lora", (False, True))
def test_apply_nvfp4_passthrough_configures_adapter_and_storage(moe_config, lora):
    """Meta experts become packed placeholders and the model adapter opts into the NVFP4 checkpoint layout."""

    class Adapter:
        def __init__(self):
            self.expert_storage_format = "bf16"

        def set_expert_storage_format(self, storage_format):
            self.expert_storage_format = storage_format

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.state_dict_adapter = Adapter()
            self.experts = GroupedExperts(moe_config)
            self.experts.use_torch_mm = True

    with torch.device("meta"):
        model = TinyModel()
        if lora:
            model.experts = patch_moe_module(model.experts, expert_weight_format="nvfp4")
        original_experts = model.experts
        result = apply_nvfp4_to_moe_experts(model)

    assert result is model
    assert model.state_dict_adapter.expert_storage_format == "nvfp4"
    assert isinstance(model.experts, GroupedExpertsLoRANVFP4 if lora else GroupedExpertsNVFP4)
    if lora:
        assert model.experts is original_experts
    assert model.experts._nvfp4_resident
    assert not hasattr(model.experts, "gate_and_up_projs")
    assert model.experts.gate_and_up_projs_nvfp4.is_meta


def test_apply_nvfp4_passthrough_requires_adapter_capability(moe_config):
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = GroupedExperts(moe_config)
            self.experts.use_torch_mm = True

    with torch.device("meta"):
        model = TinyModel()
        with pytest.raises(NotImplementedError, match="set_expert_storage_format"):
            apply_nvfp4_to_moe_experts(model)

    assert type(model.experts) is GroupedExperts


def test_validate_nvfp4_checkpoint_config(tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 32}})
    )
    with pytest.raises(ValueError, match="group_size=32"):
        validate_nvfp4_checkpoint_config(str(tmp_path))
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "FP8", "group_size": 16}})
    )
    with pytest.raises(ValueError, match="quant_algo='FP8'"):
        validate_nvfp4_checkpoint_config(str(tmp_path))
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 16}})
    )
    validate_nvfp4_checkpoint_config(str(tmp_path))
    # Without hf_quant_config.json the transformers-style quantization_config in config.json is consulted.
    (tmp_path / "hf_quant_config.json").unlink()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"quantization_config": {"quant_algo": "NVFP4", "config_groups": {"g0": {"weights": {"group_size": 16}}}}}
        )
    )
    validate_nvfp4_checkpoint_config(str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_algo": "MXFP4"}}))
    with pytest.raises(ValueError, match="quant_algo='MXFP4'"):
        validate_nvfp4_checkpoint_config(str(tmp_path))
    # A checkpoint without any quantization metadata is accepted (the layout is checked by the adapter).
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "beam"}))
    validate_nvfp4_checkpoint_config(str(tmp_path))


def _modelopt_expert_tensors(dim: int = 32, inter: int = 16, experts: int = 2) -> dict[str, torch.Tensor]:
    """A tiny experts-only ModelOpt NVFP4 layout plus one floating-point dense tensor."""
    tensors = {"model.layers.0.mlp.gate.weight": torch.zeros(experts, dim)}
    for e in range(experts):
        for proj, (out_dim, in_dim) in (
            ("gate_proj", (inter, dim)),
            ("up_proj", (inter, dim)),
            ("down_proj", (dim, inter)),
        ):
            packed, scales, tensor_scale = quantize_nvfp4_modelopt(torch.randn(out_dim, in_dim))
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            tensors[f"{base}.weight"] = packed
            tensors[f"{base}.weight_scale"] = scales
            tensors[f"{base}.weight_scale_2"] = tensor_scale
            tensors[f"{base}.input_scale"] = torch.tensor(0.01)
    return tensors


def test_validate_nvfp4_checkpoint_config_checks_the_expert_tensor_headers(tmp_path):
    """The checkpoint load copies with dtype conversion, so a wrongly typed export must be refused up front."""
    from safetensors.torch import save_file

    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4", "group_size": 16}})
    )
    good = _modelopt_expert_tensors()
    save_file(good, str(tmp_path / "model.safetensors"))
    validate_nvfp4_checkpoint_config(str(tmp_path))
    # int8 codes are the signed-byte alternative ModelOpt exports may use.
    signed = dict(good)
    signed["model.layers.0.mlp.experts.0.gate_proj.weight"] = good[
        "model.layers.0.mlp.experts.0.gate_proj.weight"
    ].view(torch.int8)
    save_file(signed, str(tmp_path / "model.safetensors"))
    validate_nvfp4_checkpoint_config(str(tmp_path))

    cases = {
        "model.layers.0.mlp.experts.1.up_proj.weight_scale": (torch.zeros(16, 2, dtype=torch.float16), "weight_scale"),
        "model.layers.0.mlp.experts.0.down_proj.weight": (torch.zeros(32, 16, dtype=torch.bfloat16), "expert weight"),
        "model.layers.0.mlp.experts.0.gate_proj.weight_scale_2": (torch.zeros(2), "scalar"),
    }
    for key, (value, message) in cases.items():
        bad = dict(good)
        bad[key] = value
        save_file(bad, str(tmp_path / "model.safetensors"))
        with pytest.raises(ValueError, match=message):
            validate_nvfp4_checkpoint_config(str(tmp_path))
    narrow = dict(good)
    narrow["model.layers.0.mlp.experts.1.down_proj.weight_scale"] = torch.zeros(32, 2, dtype=torch.float8_e4m3fn)
    save_file(narrow, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="does not match packed weight shape"):
        validate_nvfp4_checkpoint_config(str(tmp_path))
    incomplete = {k: v for k, v in good.items() if not k.endswith("experts.1.up_proj.weight_scale_2")}
    save_file(incomplete, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="missing"):
        validate_nvfp4_checkpoint_config(str(tmp_path))
    save_file({"model.layers.0.mlp.gate.weight": torch.zeros(2, 32)}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="no routed-expert"):
        validate_nvfp4_checkpoint_config(str(tmp_path))


def test_frozen_dtype_casts_keep_packed_storage(moe_config):
    """The scale tensors are part of the format: the frozen-parameter casts must leave them alone."""
    from nemo_automodel.components.models.common.utils import (
        cast_frozen_modules_to_compute_dtype,
        prepare_frozen_parameters_for_sharding,
    )
    from nemo_automodel.components.moe.quantized_experts import packed_expert_parameters

    lora = GroupedExpertsLoRANVFP4(_materialized_experts(moe_config, torch.device("cpu")), lora_dim=8, alpha=16)
    model = nn.Module()
    model.experts = lora
    model.dense = nn.Linear(8, 8).to(torch.float32)  # an ordinary frozen fp32 weight, which the casts do reduce
    model.dense.requires_grad_(False)
    # An ordinary frozen fp32 parameter of the packed module (an expert bias, say) follows the ordinary rules.
    lora.register_parameter("extra_bias", nn.Parameter(torch.zeros(4, 8), requires_grad=False))
    expected = {
        "gate_and_up_projs_nvfp4": torch.uint8,
        "gate_and_up_projs_scales": torch.float8_e4m3fn,
        "gate_and_up_projs_global": torch.float32,
        "down_projs_nvfp4": torch.uint8,
        "down_projs_scales": torch.float8_e4m3fn,
        "down_projs_global": torch.float32,
    }
    assert lora.gate_and_up_projs_global._hf_compute_dtype == torch.float32
    assert lora.gate_and_up_projs_scales._hf_compute_dtype == torch.float8_e4m3fn

    prepare_frozen_parameters_for_sharding(model, torch.bfloat16, keep_storage=packed_expert_parameters(model))
    cast_frozen_modules_to_compute_dtype(model, torch.bfloat16, keep_storage=packed_expert_parameters(model))

    for name, dtype in expected.items():
        assert getattr(lora, name).dtype == dtype, name
    assert model.dense.weight.dtype == torch.bfloat16  # the ordinary frozen weight was cast as before
    assert lora.extra_bias.dtype == torch.bfloat16
    assert set(lora.packed_parameter_names()) == PACKED_NAMES


@pytest.mark.parametrize("storage_format", ["nvfp4", "mxfp4"])
def test_cast_model_to_dtype_keeps_packed_storage_without_sharding_preparation(moe_config, storage_format):
    """initialize_weights casts the whole model; packed modules carry their dtype contract from registration on,
    so the cast keeps the codes and scales even when prepare_frozen_parameters_for_sharding never ran."""
    from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only
    from nemo_automodel.components.models.common.utils import cast_model_to_dtype
    from nemo_automodel.components.moe.quantized_experts import GroupedExpertsMXFP4

    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
        orig.use_torch_mm = True
        packed = GroupedExpertsNVFP4(orig) if storage_format == "nvfp4" else GroupedExpertsMXFP4(orig)
        model = nn.Module()
        model.experts = packed
        model.dense = nn.Linear(8, 8).to(torch.float32)
    model.dense.requires_grad_(False)
    to_empty_parameters_only(model, device=torch.device("cpu"))
    before = {name: parameter.dtype for name, parameter in packed.named_parameters(recurse=False)}

    cast_model_to_dtype(model, torch.bfloat16)

    assert {name: parameter.dtype for name, parameter in packed.named_parameters(recurse=False)} == before
    assert torch.float32 in before.values() or torch.float8_e8m0fnu in before.values()
    assert model.dense.weight.dtype == torch.bfloat16


def test_frozen_storage_cast_skips_packed_modules_without_the_parameter_pin(moe_config):
    """Expert parallelism and accelerate's init_empty_weights re-create parameter objects, dropping the per-parameter
    dtype hint; the storage reduction must still leave the packed module's fp32 row scales alone."""
    from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
    from nemo_automodel.components.moe.quantized_experts import packed_expert_parameters

    lora = GroupedExpertsLoRANVFP4(_materialized_experts(moe_config, torch.device("cpu")), lora_dim=8, alpha=16)
    for name in ("gate_and_up_projs_global", "gate_and_up_projs_scales", "down_projs_global", "down_projs_scales"):
        lora.register_parameter(name, nn.Parameter(getattr(lora, name).data, requires_grad=False))  # no pin
        assert not hasattr(getattr(lora, name), "_hf_compute_dtype")
    model = nn.Module()
    model.experts = lora
    model.dense = nn.Linear(8, 8).to(torch.float32)
    model.dense.requires_grad_(False)

    prepare_frozen_parameters_for_sharding(model, torch.bfloat16, keep_storage=packed_expert_parameters(model))

    assert lora.gate_and_up_projs_global.dtype == torch.float32 and lora.down_projs_global.dtype == torch.float32
    assert lora.gate_and_up_projs_scales.dtype == torch.float8_e4m3fn
    assert model.dense.weight.dtype == torch.bfloat16
    assert lora._preserve_parameter_dtypes["gate_and_up_projs_global"] == torch.float32
    assert lora._preserve_parameter_dtypes["down_projs_scales"] == torch.float8_e4m3fn


def test_placeholders_built_under_init_empty_weights_keep_their_pins(moe_config):
    """apply_model_infrastructure applies PEFT under init_empty_weights, whose register_parameter re-creates
    every parameter: the dtype pin must sit on the object that ends up registered."""
    from nemo_automodel.components.models.common.utils import prepare_frozen_parameters_for_sharding
    from nemo_automodel.components.utils.model_utils import init_empty_weights

    with torch.device("meta"):
        orig = GroupedExperts(moe_config)
    orig.use_torch_mm = True
    with init_empty_weights():
        q = GroupedExpertsLoRANVFP4(orig, lora_dim=8, alpha=16)

    assert q.gate_and_up_projs_global.is_meta and q.gate_and_up_projs_global.dtype == torch.float32
    assert q.gate_and_up_projs_global._hf_compute_dtype == torch.float32
    assert q.down_projs_scales._hf_compute_dtype == torch.float8_e4m3fn
    model = nn.Module()
    model.experts = q
    prepare_frozen_parameters_for_sharding(model, torch.bfloat16)
    assert q.gate_and_up_projs_global.dtype == torch.float32 and q.down_projs_global.dtype == torch.float32


def test_forward_applies_post_expert_output_like_the_reference(moe_config, device):
    if not _grouped_mm_available(device):
        pytest.skip("torch._grouped_mm is not available on this device")
    orig = _materialized_experts(moe_config, device)
    _make_representable_(orig)
    ref = GroupedExpertsLoRA(orig, lora_dim=8, alpha=16).to(device)
    q = GroupedExpertsLoRANVFP4(orig, lora_dim=8, alpha=16).to(device)
    with torch.no_grad():
        for name in LORA_NAMES:
            getattr(q, name).data.copy_(getattr(ref, name).data)
    x, token_mask, weights, indices = _routing_inputs(moe_config, 16, device, torch.bfloat16)
    hook = lambda t: t * 2 + 1  # noqa: E731
    y_ref = ref(x, token_mask, weights, indices, post_expert_output=hook)
    y_q = q(x.detach().clone().requires_grad_(True), token_mask, weights, indices, post_expert_output=hook)
    torch.testing.assert_close(y_q, y_ref, atol=1e-6, rtol=1e-6)
    y_plain = q(x.detach().clone().requires_grad_(True), token_mask, weights, indices)
    assert not torch.equal(y_q, y_plain)


def test_packed_topology_guard():
    from nemo_automodel.components.moe.quantized_experts import validate_packed_expert_topology

    validate_packed_expert_topology("nvfp4", world_size=1, ep_size=1)
    validate_packed_expert_topology("nvfp4", world_size=8, ep_size=8)
    validate_packed_expert_topology("nf4", world_size=16, ep_size=8, pp_size=2)
    validate_packed_expert_topology("unquantized", world_size=16, ep_size=1)
    with pytest.raises(ValueError, match="ep_size == 1"):
        validate_packed_expert_topology("nvfp4", world_size=8, ep_size=1)
    with pytest.raises(ValueError, match="ep_shard=2"):
        validate_packed_expert_topology("nvfp4", world_size=16, ep_size=8)
    with pytest.raises(ValueError, match="expert-shard"):
        validate_packed_expert_topology("mxfp4", world_size=32, ep_size=8, pp_size=2)


def test_exports_refuse_packed_experts(moe_config):
    from nemo_automodel._transformers.peft_export import _merge_and_get_hf_state_dict

    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].experts = GroupedExpertsLoRANVFP4(
        _materialized_experts(moe_config, torch.device("cpu")), lora_dim=8, alpha=16
    )
    with pytest.raises(ValueError, match="Merged PEFT export is not supported for packed-expert storage"):
        _merge_and_get_hf_state_dict(model)

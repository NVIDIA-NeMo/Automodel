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

"""M3.1 compressed checkpoint contracts with independent, exact numeric oracles.

Compressed-tensors e69c8dc58aa152e5f5e36e85800ec1d2e7de5271 defines low
nibbles first and dequantization as E2M1 * (E4M3 scale / global scale).
See compressors/nvfp4/helpers.py and quantization/lifecycle/forward_helpers.py.
Expected values below are hand-expanded; no production dequantizer is an oracle.
"""

import gc
import json
import weakref
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file

from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
from nemo_automodel._transformers.model_init import local_torch_dtype
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
    build_moe_config,
)
from nemo_automodel.components.models.minimax_m3_vl.state_dict_adapter import (
    MiniMaxM3StateDictAdapter,
    MiniMaxM3VLStateDictAdapter,
)

Adapter = MiniMaxM3StateDictAdapter | MiniMaxM3VLStateDictAdapter


def _make_adapter(
    kind: Literal["text", "vl"], dtype: torch.dtype = torch.float32, *, nvfp4: bool = True, n_experts: int = 1
) -> tuple[Adapter, str]:
    quantization = {"quant_method": "mxfp8", "weight_block_size": [1, 32]}
    if nvfp4:
        quantization.update({"moe_quant_algo": "NVFP4", "moe_quant_format": "nvfp4-pack-quantized", "group_size": 16})
    text = MiniMaxM3VLTextConfig(
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=n_experts,
        num_experts_per_tok=1,
        num_mtp_modules=0,
        quantization_config=quantization if kind == "text" else None,
    )
    moe = build_moe_config(text, dtype)
    backend = BackendConfig(linear="torch", attn="sdpa", experts="torch_mm", dispatcher="torch", rope_fusion=False)
    if kind == "text":
        return MiniMaxM3StateDictAdapter(text, moe, backend, dtype=dtype), ""
    config = MiniMaxM3VLConfig(text_config=text, quantization_config=quantization)
    return MiniMaxM3VLStateDictAdapter(config, moe, backend, dtype=dtype), "language_model."


def _packed_projection(rows: int, columns: int, divisor: float = 4.0) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return a checkpoint projection and hand-expanded reference.

    Args:
        rows: Number of output channels; must be even.
        columns: Number of input channels, 32 or 64.
        divisor: Scalar compressed-tensors global scale.

    Returns:
        Pair of checkpoint tensors and FP32 expected weight [rows, columns].
        The mapping has uint8 weight_packed [rows, columns / 2], E4M3
        weight_scale [rows, columns / 16], and FP32 weight_global_scale
        and input_global_scale scalars []. The input calibration scale must not
        affect weight decompression.
    """
    # Consecutive bytes encode 0,+0.5,+1,+1.5,+2,+3,+4,+6,
    # then -0,-0.5,-1,-1.5,-2,-3,-4,-6 in exactly that order.
    packed = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype=torch.uint8).repeat(
        rows, columns // 16
    )
    factors = [1.0, 2.0, 4.0, 8.0][: columns // 16]
    scales = torch.tensor([factors, [4.0 * value for value in factors]]).repeat(rows // 2, 1)
    # The first 16-column group's exact values after dividing by global=4.
    first = torch.tensor(
        [0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.5, -0.0, -0.125, -0.25, -0.375, -0.5, -0.75, -1.0, -1.5]
    )
    row = torch.cat([first * factor for factor in factors])
    expected = torch.stack((row, row * 4.0)).repeat(rows // 2, 1) * (4.0 / divisor)
    return {
        "weight_packed": packed,
        "weight_scale": scales.to(torch.float8_e4m3fn),
        "weight_global_scale": torch.tensor(divisor),
        "input_global_scale": torch.tensor(17.0),
    }, expected


def _expert_state(prefix: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return all three expert projections in checkpoint and expected layouts.

    Args:
        prefix: Optional language-model namespace.

    Returns:
        Checkpoint mapping with packed tensors described by _packed_projection
        and expected FP32 weights w1/w3 [32, 64], w2 [64, 32].
    """
    state = {}
    expected = {}
    for name, rows, columns, divisor in (("w1", 32, 64, 4.0), ("w3", 32, 64, 8.0), ("w2", 64, 32, 16.0)):
        tensors, weight = _packed_projection(rows, columns, divisor)
        base = f"{prefix}model.layers.0.block_sparse_moe.experts.0.{name}"
        state.update({f"{base}.{suffix}": value for suffix, value in tensors.items()})
        expected[name] = weight
    return state, expected


@pytest.mark.parametrize("kind", ["text", "vl"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nvfp4_checkpoint_decodes_exact_values_and_grouped_layout(kind, dtype):
    adapter, prefix = _make_adapter(kind, dtype)
    state, expected = _expert_state(prefix)
    native = adapter.from_hf(state)
    gate_key = "model.layers.0.mlp.experts.gate_and_up_projs"
    down_key = "model.layers.0.mlp.experts.down_projs"
    assert set(native) == {gate_key, down_key}
    expected_gate = torch.cat((expected["w1"].T, expected["w3"].T), dim=-1).unsqueeze(0).to(dtype)
    expected_down = expected["w2"].T.unsqueeze(0).to(dtype)
    torch.testing.assert_close(native[gate_key], expected_gate, rtol=0, atol=0)
    torch.testing.assert_close(native[down_key], expected_down, rtol=0, atol=0)

    # Exporting trained dense weights must not reintroduce checkpoint-only scales.
    exported = adapter.to_hf(native, quantization=False)
    assert len(exported) == 3
    for name, weight in expected.items():
        torch.testing.assert_close(
            exported[f"{prefix}model.layers.0.block_sparse_moe.experts.0.{name}.weight"],
            weight.to(dtype),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("kind", ["text", "vl"])
def test_nvfp4_planner_requests_each_of_four_checkpoint_tensors(kind):
    adapter, prefix = _make_adapter(kind)
    native = {
        "model.layers.0.mlp.experts.gate_and_up_projs": torch.zeros(1, 64, 64),
        "model.layers.0.mlp.experts.down_projs": torch.zeros(1, 32, 64),
    }
    planned = adapter.to_hf(native, quantization=True)
    expected_keys = set()
    for name, rows, columns in (("w1", 32, 64), ("w3", 32, 64), ("w2", 64, 32)):
        base = f"{prefix}model.layers.0.block_sparse_moe.experts.0.{name}"
        expected_keys.update(
            f"{base}.{suffix}"
            for suffix in ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")
        )
        assert planned[f"{base}.weight_packed"].shape == (rows, columns // 2)
        assert planned[f"{base}.weight_packed"].dtype == torch.uint8
        assert planned[f"{base}.weight_scale"].shape == (rows, columns // 16)
        assert planned[f"{base}.weight_scale"].dtype == torch.float8_e4m3fn
        assert planned[f"{base}.weight_global_scale"].numel() == 1
        assert planned[f"{base}.input_global_scale"].numel() == 1
    assert set(planned) == expected_keys


@pytest.mark.parametrize("kind", ["text", "vl"])
@pytest.mark.parametrize("projection", ["q", "k"])
def test_m31_indexer_plans_and_applies_non_identity_mxfp8_scales(kind, projection):
    adapter, prefix = _make_adapter(kind)
    native_key = f"model.layers.0.self_attn.indexer.index_{projection}_proj.weight"
    hf_key = f"{prefix}model.layers.0.self_attn.index_{projection}_proj.weight"
    planned = adapter.to_hf({native_key: torch.zeros(2, 64)}, quantization=True)
    assert set(planned) == {hf_key, hf_key + "_scale_inv"}
    assert planned[hf_key].dtype == torch.float8_e4m3fn
    assert planned[hf_key + "_scale_inv"].shape == (2, 2)
    assert planned[hf_key + "_scale_inv"].dtype == torch.uint8

    planned[hf_key] = torch.tensor([[1.0] * 64, [-2.0] * 64]).to(torch.float8_e4m3fn)
    planned[hf_key + "_scale_inv"] = torch.tensor([[125, 129], [128, 126]], dtype=torch.uint8)
    restored = adapter.from_hf(planned)
    expected = torch.tensor([[0.25] * 32 + [4.0] * 32, [-4.0] * 32 + [-1.0] * 32])
    assert set(restored) == {native_key}
    torch.testing.assert_close(restored[native_key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["text", "vl"])
def test_legacy_mxfp8_indexer_and_fp32_router_remain_unquantized(kind):
    adapter, prefix = _make_adapter(kind, nvfp4=False)
    assert adapter.iter_checkpoint_load_parts({}) is None
    native_key = "model.layers.0.self_attn.indexer.index_q_proj.weight"
    router_key = "model.layers.0.mlp.gate.weight"
    bias_key = "model.layers.0.mlp.gate.e_score_correction_bias"
    indexer = torch.full((2, 64), 0.1234567)
    router = torch.full((1, 64), 0.2345678)
    bias = torch.tensor([34.123])
    planned = adapter.to_hf({native_key: indexer, router_key: router, bias_key: bias}, quantization=True)
    hf_indexer = f"{prefix}model.layers.0.self_attn.index_q_proj.weight"
    assert hf_indexer + "_scale_inv" not in planned
    restored = adapter.from_hf(planned)
    for key, expected in ((native_key, indexer), (router_key, router), (bias_key, bias)):
        assert restored[key].dtype == torch.float32
        torch.testing.assert_close(restored[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["text", "vl"])
@pytest.mark.parametrize("defect", ["wrong_block", "missing_block", "wrong_group"])
def test_nvfp4_rejects_inconsistent_quantization_metadata(kind, defect):
    adapter, _ = _make_adapter(kind)
    config = adapter.config
    quantization = dict(config.quantization_config)
    if defect == "wrong_block":
        quantization["weight_block_size"] = [1, 128]
    elif defect == "missing_block":
        del quantization["weight_block_size"]
    else:
        quantization["group_size"] = 32
    config.quantization_config = quantization
    text_adapter = adapter.text_adapter if kind == "vl" else adapter
    adapter_type = MiniMaxM3VLStateDictAdapter if kind == "vl" else MiniMaxM3StateDictAdapter
    with pytest.raises(ValueError, match="weight_block_size"):
        adapter_type(config, text_adapter.moe_config, text_adapter.backend)


@pytest.mark.parametrize("corruption", ["reversed_nibbles", "reciprocal_global"])
def test_nvfp4_exact_oracle_rejects_wrong_format_interpretations(corruption):
    adapter, prefix = _make_adapter("text")
    state, expected = _expert_state(prefix)
    base = "model.layers.0.block_sparse_moe.experts.0.w1"
    if corruption == "reversed_nibbles":
        value = state[f"{base}.weight_packed"]
        state[f"{base}.weight_packed"] = (value >> 4) | ((value & 15) << 4)
    else:
        # Substituting 1/global makes divide-global produce the wrong multiply-global result.
        state[f"{base}.weight_global_scale"] = torch.tensor(0.25)
    native = adapter.from_hf(state)
    actual_gate = native["model.layers.0.mlp.experts.gate_and_up_projs"][0, :, :32].T
    with pytest.raises(AssertionError):
        torch.testing.assert_close(actual_gate, expected["w1"], rtol=0, atol=0)


def _streamed_fixture(
    kind: Literal["text", "vl"],
    n_experts: int = 1,
) -> tuple[Adapter, dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return a small complete checkpoint and its final native destinations.

    Args:
        kind: Text-only or composite adapter namespace.
        n_experts: Number of experts, with distinct power-of-two weight scales.

    Returns:
        Adapter, checkpoint mapping, native destinations, and expected dense mapping.
        Checkpoint expert layouts follow _packed_projection. Native grouped gate/up
        is [n_experts, 64, 64], down is [n_experts, 32, 64], indexer is [2, 64], router
        is [n_experts, 64], correction bias is [n_experts], and optional vision norm is [4].
        All final tensors are FP32; quantized inputs retain their storage dtypes.
    """
    adapter, prefix = _make_adapter(kind, n_experts=n_experts)
    checkpoint, expert_expected = _expert_state(prefix)
    gate_key = "model.layers.0.mlp.experts.gate_and_up_projs"
    down_key = "model.layers.0.mlp.experts.down_projs"
    index_key = "model.layers.0.self_attn.indexer.index_q_proj.weight"
    router_key = "model.layers.0.mlp.gate.weight"
    bias_key = "model.layers.0.mlp.gate.e_score_correction_bias"
    expected = {
        gate_key: torch.cat((expert_expected["w1"].T, expert_expected["w3"].T), dim=-1).unsqueeze(0),
        down_key: expert_expected["w2"].T.unsqueeze(0),
        index_key: torch.tensor([[0.25] * 32 + [4.0] * 32, [-4.0] * 32 + [-1.0] * 32]),
        router_key: torch.full((1, 64), 0.1234567),
        bias_key: torch.tensor([34.123]),
    }
    if n_experts > 1:
        base_checkpoint = dict(checkpoint)
        for expert_id in range(1, n_experts):
            for key, tensor in base_checkpoint.items():
                value = tensor.clone()
                if key.endswith(".weight_global_scale"):
                    value *= 2**expert_id
                checkpoint[key.replace(".experts.0.", f".experts.{expert_id}.")] = value
        for key in (gate_key, down_key):
            expected[key] = torch.cat([expected[key] / 2**expert_id for expert_id in range(n_experts)], dim=0)
        expected[router_key] = expected[router_key].repeat(n_experts, 1)
        expected[bias_key] = expected[bias_key].repeat(n_experts)
    index_hf = prefix + "model.layers.0.self_attn.index_q_proj.weight"
    checkpoint[index_hf] = torch.tensor([[1.0] * 64, [-2.0] * 64]).to(torch.float8_e4m3fn)
    checkpoint[index_hf + "_scale_inv"] = torch.tensor([[125, 129], [128, 126]], dtype=torch.uint8)
    checkpoint[prefix + "model.layers.0.block_sparse_moe.gate.weight"] = expected[router_key].clone()
    checkpoint[prefix + "model.layers.0.block_sparse_moe.e_score_correction_bias"] = expected[bias_key].clone()
    if kind == "vl":
        vision_key = "vision_tower.vision_model.post_layernorm.weight"
        expected[vision_key] = torch.tensor([0.125, -0.25, 0.5, 1.0])
        checkpoint[vision_key] = expected[vision_key].clone()
    native = {key: torch.full_like(value, float("nan")) for key, value in expected.items()}
    return adapter, checkpoint, native, expected


def _save_checkpoint(checkpoint: dict[str, torch.Tensor], directory: Path) -> None:
    """Write real SafeTensors and its index for the native DCP reader.

    Args:
        checkpoint: Packed and dense checkpoint tensors with layouts documented
            by _streamed_fixture.
        directory: Existing directory for the small synthetic format fixture.
    """
    save_file(checkpoint, str(directory / "model.safetensors"))
    index = {"metadata": {}, "weight_map": {key: "model.safetensors" for key in checkpoint}}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))


@pytest.mark.parametrize("kind", ["text", "vl"])
def test_streamed_parts_read_safetensors_into_existing_storage(kind, tmp_path):
    adapter, checkpoint, native, expected = _streamed_fixture(kind)
    _save_checkpoint(checkpoint, tmp_path)
    reader = _HuggingFaceStorageReader(str(tmp_path), key_mapping=None)
    pointers = {key: value.untyped_storage().data_ptr() for key, value in native.items()}
    parts = adapter.iter_checkpoint_load_parts(native)
    assert parts is not None
    model_keys = set()
    checkpoint_keys = set()
    temporary_refs = []
    for part in parts:
        assert not model_keys.intersection(part.model_keys)
        assert not checkpoint_keys.intersection(part.checkpoint_tensors)
        if part.temporary_checkpoint_keys:
            assert len(part.model_keys) == 1
            temporary_refs.extend(weakref.ref(part.checkpoint_tensors[key]) for key in part.temporary_checkpoint_keys)
        model_keys.update(part.model_keys)
        checkpoint_keys.update(part.checkpoint_tensors)
        dcp.load(part.checkpoint_tensors, storage_reader=reader, no_dist=True)
        part.finish()
    assert model_keys == set(native)
    assert checkpoint_keys == set(checkpoint)
    for key, value in native.items():
        assert value.untyped_storage().data_ptr() == pointers[key], key
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    del part, parts
    gc.collect()
    assert all(ref() is None for ref in temporary_refs), "completed load parts retained compressed tensors"


@pytest.mark.parametrize("kind", ["text", "vl"])
def test_native_initialization_then_quantized_load_preserves_bf16_lora(
    kind: Literal["text", "vl"], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real meta init, PEFT and Checkpointer loading, without a GPU factory.

    This CPU test covers the production infrastructure shared by the custom
    from-pretrained route. It does not exercise distributed sharding or vision
    forward execution; the complete VL checkpoint state is compared after load.
    """
    from .conftest import SPARSE_ATTENTION_CONFIG, TINY_CFG, VISION_CONFIG

    quantization = {
        "quant_method": "mxfp8",
        "weight_block_size": [1, 32],
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
        "group_size": 16,
    }
    text = MiniMaxM3VLTextConfig(
        **{
            **TINY_CFG,
            "num_hidden_layers": 1,
            "num_local_experts": 2,
            "moe_layer_freq": [1],
            "sparse_attention_config": {
                **SPARSE_ATTENTION_CONFIG,
                "sparse_attention_freq": [1],
                "sparse_disable_index_value": [1],
            },
            "quantization_config": quantization,
            "torch_dtype": torch.bfloat16,
        }
    )
    backend = BackendConfig(linear="torch", attn="sdpa", experts="torch_mm", dispatcher="torch", rope_fusion=False)
    with local_torch_dtype(torch.bfloat16), torch.device("meta"):
        if kind == "vl":
            config = MiniMaxM3VLConfig(
                text_config=text,
                vision_config=dict(VISION_CONFIG),
                projector_hidden_size=text.hidden_size,
                quantization_config=quantization,
            )
            model = MiniMaxM3SparseForConditionalGeneration(config, backend=backend)
        else:
            model = MiniMaxM3SparseForCausalLM(text, backend=backend)
    expected = {
        key: torch.full(value.shape, 0.1234567, dtype=value.dtype, device="cpu")
        for key, value in model.state_dict().items()
    }
    # Every MXFP8 projection below stores FP8 ones with exponent 128: exactly 2.
    # The oracle is independent of the production dequantizer.
    projections = ("q_proj", "k_proj", "v_proj", "o_proj")
    for projection in projections:
        expected[f"model.layers.0.self_attn.{projection}.weight"].fill_(2)
    for projection in ("q", "k"):
        expected[f"model.layers.0.self_attn.indexer.index_{projection}_proj.weight"].fill_(2)
    for projection in ("gate", "up", "down"):
        expected[f"model.layers.0.shared_experts.{projection}_proj.weight"].fill_(2)
    prefix = "language_model." if kind == "vl" else ""
    packed_experts = {}
    for expert_id in range(2):
        decoded = {}
        for projection, rows, columns in (("w1", 32, 64), ("w3", 32, 64), ("w2", 64, 32)):
            tensors, weight = _packed_projection(rows, columns, 4.0 * (expert_id + 1))
            base = f"{prefix}model.layers.0.block_sparse_moe.experts.{expert_id}.{projection}"
            packed_experts.update({f"{base}.{suffix}": tensor for suffix, tensor in tensors.items()})
            decoded[projection] = weight
        expected["model.layers.0.mlp.experts.gate_and_up_projs"][expert_id].copy_(
            torch.cat((decoded["w1"].T, decoded["w3"].T), dim=-1)
        )
        expected["model.layers.0.mlp.experts.down_projs"][expert_id].copy_(decoded["w2"].T)

    adapter = model.state_dict_adapter
    dense_checkpoint = adapter.to_hf({key: value.clone() for key, value in expected.items()}, quantization=False)
    planned = adapter.to_hf({key: value.clone() for key, value in expected.items()}, quantization=True)
    checkpoint = {}
    for key, template in planned.items():
        if key in packed_experts:
            checkpoint[key] = packed_experts[key]
        elif key.endswith("_scale_inv"):
            checkpoint[key] = torch.full(template.shape, 128, dtype=torch.uint8)
        elif template.dtype == torch.float8_e4m3fn:
            checkpoint[key] = torch.ones(template.shape, dtype=torch.float32).to(torch.float8_e4m3fn)
        else:
            checkpoint[key] = dense_checkpoint[key].contiguous().clone()
    _save_checkpoint(checkpoint, tmp_path)

    initialized_pointers: dict[str, int] = {}
    initialized_embeddings: list[torch.Tensor] = []
    initialize = Checkpointer.initialize_model_weights

    def observe_initialization(
        current: torch.nn.Module, device: torch.device, peft_init_method: str | None = None
    ) -> None:
        initialize(current, device, peft_init_method)
        initialized_embeddings.append(current.get_input_embeddings().weight.detach().clone())
        initialized_pointers.update(
            (name, parameter.untyped_storage().data_ptr()) for name, parameter in current.named_parameters()
        )

    monkeypatch.setattr(Checkpointer, "initialize_model_weights", staticmethod(observe_initialization))
    peft = PeftConfig(
        dim=4,
        alpha=4,
        target_modules=["model.layers.*.self_attn.q_proj", "model.layers.*.mlp.experts"],
        use_memory_efficient_lora=False,
    )
    restored = apply_model_infrastructure(
        model=model,
        is_meta_device=True,
        device=torch.device("cpu"),
        peft_config=peft,
        load_base_model=True,
        pretrained_model_name_or_path=str(tmp_path),
        cache_dir=str(tmp_path),
    )
    assert len(initialized_embeddings) == 1
    assert not torch.equal(initialized_embeddings[0], expected["model.embed_tokens.weight"])
    actual = restored.state_dict()
    assert {key for key in actual if "lora_" not in key} == expected.keys()
    for name, reference in expected.items():
        assert actual[name].dtype == reference.dtype, name
        torch.testing.assert_close(actual[name], reference, rtol=0, atol=0, msg=name)
    adapters = {name: parameter for name, parameter in restored.named_parameters() if "lora_" in name}
    assert len(adapters) == 6  # Dense A/B and the four grouped-expert A/B parameters.
    a_names = {name for name in adapters if name.endswith(("_A", ".lora_A.weight"))}
    b_names = {name for name in adapters if name.endswith(("_B", ".lora_B.weight"))}
    assert len(a_names) == len(b_names) == 3
    assert a_names.isdisjoint(b_names)
    assert a_names | b_names == adapters.keys()
    for name, parameter in restored.named_parameters():
        assert parameter.untyped_storage().data_ptr() == initialized_pointers[name], name
        assert parameter.requires_grad == ("lora_" in name), name
        if "lora_" in name:
            assert parameter.dtype == torch.bfloat16, name
            assert torch.isfinite(parameter).all(), name
            if name in b_names:
                assert torch.count_nonzero(parameter) == 0, name
            else:
                assert name in a_names
                assert torch.count_nonzero(parameter) > 0, name
    gate = restored.model.layers["0"].mlp.gate
    assert gate.weight.dtype == torch.float32
    assert gate.e_score_correction_bias.dtype == torch.float32
    assert not torch.equal(gate.weight, gate.weight.bfloat16().float())

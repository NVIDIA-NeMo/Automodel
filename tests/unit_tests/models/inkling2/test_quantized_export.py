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

"""Released-layout encoders against scalar references, and the export of trained checkpoints."""

import json
import math
import struct
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.inkling2.config import Inkling2Config
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.quantization import quantize_mxfp8_experts, quantize_nvfp4_experts
from nemo_automodel.components.models.inkling2.quantized_export import export_quantized, main

from .test_checkpoint import (
    checkpoint_config,
    decode_oracle,
    mixed_fixture,
    write_checkpoint,
    write_released_checkpoint_metadata,
)

pytestmark = pytest.mark.cpu_only

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
W13 = "model.llm.layers.0.mlp.experts.w13_weight"
W2_MXFP8 = "model.llm.layers.1.mlp.experts.w2_weight"


def f32(value: float) -> float:
    """Round a Python float to the nearest FP32 value."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def e4m3_grid(value: float) -> float:
    """Nearest value of a nonnegative number on the E4M3 grid (3 mantissa bits), ties to even, unbounded above."""
    if value == 0:
        return 0.0
    # Three mantissa bits per binade; below 2**-6 the subnormal step is 2**-9.
    step = 2.0 ** (max(math.frexp(value)[1] - 1, -6) - 3)
    return round(value / step) * step


def nearest_e4m3(value: float) -> float:
    """Nearest float8_e4m3fn magnitude of a nonnegative value, saturating at 448."""
    return min(e4m3_grid(value), 448.0)


def nearest_e2m1(value: float) -> int:
    """E2M1 code of the nearest magnitude, ties to the even code, saturating at 6; bit 3 is the sign bit."""
    magnitude = min(abs(value), 6.0)
    index = min(range(8), key=lambda i: (abs(E2M1[i] - magnitude), i % 2))
    return index | 8 if math.copysign(1.0, value) < 0 else index


def reference_nvfp4(weight: torch.Tensor, denominator: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scalar NVFP4 encoding with FP32 rounding after every operation.

    Args:
        weight: Tensor of shape [experts, output, input], input a multiple of 16.
        denominator: The tensor's outer-scale denominator.

    Returns:
        uint8 codes [experts, output, input/2] (even column in the low nibble), float8_e4m3fn block scales
        [experts, output, input/16] and FP32 amax [experts].
    """
    experts, rows, width = weight.shape
    values = weight.float().tolist()
    codes = torch.zeros(experts, rows, width // 2, dtype=torch.uint8)
    scales = torch.zeros(experts, rows, width // 16)
    amax = torch.zeros(experts)
    for e in range(experts):
        expert_amax = max(abs(v) for row in values[e] for v in row)
        amax[e] = expert_amax
        encode = f32(denominator / expert_amax) if expert_amax else 0.0
        outer = f32(expert_amax / denominator)
        for o in range(rows):
            for b in range(width // 16):
                block = values[e][o][16 * b : 16 * b + 16]
                scale = nearest_e4m3(f32(f32(max(abs(v) for v in block) / 6) * encode))
                scales[e, o, b] = scale
                decode = f32(scale * outer)
                for j, v in enumerate(block):
                    code = nearest_e2m1(f32(v / decode) if decode else math.copysign(0.0, v))
                    col = 16 * b + j
                    codes[e, o, col // 2] |= code << (4 * (col % 2))
    return codes, scales.to(torch.float8_e4m3fn), amax


def reference_mxfp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Scalar MXFP8 encoding: per row and 32 columns, the smallest 2**s at which max|w| rounds to at most 448.

    Args:
        weight: Tensor of shape [experts, output, input], input a multiple of 32.

    Returns:
        float8_e4m3fn values [experts, output, input] and uint8 scale bytes [experts, output, input/32].
    """
    experts, rows, width = weight.shape
    values = weight.float().tolist()
    out = torch.zeros(experts, rows, width)
    scales = torch.zeros(experts, rows, width // 32, dtype=torch.uint8)
    for e in range(experts):
        for o in range(rows):
            for b in range(width // 32):
                block = values[e][o][32 * b : 32 * b + 32]
                shared = -127
                while e4m3_grid(max(abs(v) for v in block) / 2.0**shared) > 448:
                    shared += 1
                scales[e, o, b] = shared + 127
                for j, v in enumerate(block):
                    out[e, o, 32 * b + j] = math.copysign(nearest_e4m3(abs(v) / 2.0**shared), v)
    return out.to(torch.float8_e4m3fn), scales


def special_weight(experts: int, rows: int, width: int) -> torch.Tensor:
    """Random [experts, rows, width] weights plus exact E2M1/E4M3 midpoints, zero blocks and a zero expert."""
    torch.manual_seed(7)
    weight = torch.randn(experts, rows, width) * torch.logspace(-3, 1, experts)[:, None, None]
    weight[1] = 0
    weight[2, 0, :32] = 0
    weight[3] = weight[3].clamp(-6, 6)
    # amax 10.5 = 2688 / 256: a block with maximum 6 gets block scale 256 and decode scale exactly 1.
    weight[3, 2, 0] = 10.5
    weight[3, 0, :16] = torch.tensor(
        [6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, 6]
    )
    weight[3, 0, 16:32] = torch.tensor([-0.1, -0.0, 0.1, -1e-30, 6] + [0] * 11)
    # With block maximum 7 the MXFP8 scale is 2**-6: the values below scale to E4M3 ties and subnormals.
    weight[3, 1, :32] = torch.tensor([448.0, 17, 19, 2.0**-9 * 1.5, -(2.0**-10), 232, 240, 1e-12] * 4) / 64
    # Block maxima 455 and 470 at that scale: the first still rounds to 448, the second takes the next scale.
    weight[3, 4, :32] = torch.tensor([455.0, -300, 17, 1] * 8) / 64
    weight[3, 5, :32] = torch.tensor([470.0, -300, 17, 1] * 8) / 64
    return weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_nvfp4_encoder_matches_the_scalar_reference(dtype: torch.dtype) -> None:
    weight = special_weight(4, 8, 64).to(dtype)
    for denominator in (2688.0, 1536.0):
        codes, scales, amax = quantize_nvfp4_experts(weight, torch.tensor([denominator]))
        ref_codes, ref_scales, ref_amax = reference_nvfp4(weight, denominator)
        assert codes.dtype == torch.uint8 and scales.dtype == torch.float8_e4m3fn and amax.dtype == torch.float32
        assert torch.equal(codes, ref_codes)
        assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))
        assert torch.equal(amax, ref_amax)
    # Expert 3 block 0 holds every E2M1 midpoint at decode scale 1: ties go to the even code.
    codes, _, _ = quantize_nvfp4_experts(weight.float(), torch.tensor([2688.0]))
    nibbles = torch.stack((codes[3, 0, :8] & 15, codes[3, 0, :8] >> 4), -1).flatten().tolist()
    assert nibbles == [7, 0, 2, 2, 4, 4, 6, 6, 8, 10, 10, 12, 12, 14, 14, 7]
    # Negative values and -0.0 that round to zero keep the sign bit, as the release encodes them.
    assert (codes[3, 0, 8] & 15, codes[3, 0, 8] >> 4, codes[3, 0, 9] & 15, codes[3, 0, 9] >> 4) == (8, 8, 0, 8)
    assert amax[1] == 0 and not codes[1].any() and not scales[1].view(torch.uint8).any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mxfp8_encoder_matches_the_scalar_reference(dtype: torch.dtype) -> None:
    weight = special_weight(4, 8, 64).to(dtype)
    weight[0, 0, 0] = 1e30
    values, scales = quantize_mxfp8_experts(weight)
    ref_values, ref_scales = reference_mxfp8(weight)
    assert values.dtype == torch.float8_e4m3fn and scales.dtype == torch.uint8
    assert torch.equal(values.view(torch.uint8), ref_values.view(torch.uint8))
    assert torch.equal(scales, ref_scales)


def test_decoding_the_encoding_stays_within_half_a_step_and_fixes_grid_values() -> None:
    weight = special_weight(4, 8, 64)
    denominator = torch.tensor([2688.0])
    codes, scales, amax = quantize_nvfp4_experts(weight, denominator)
    decoded = decode_oracle(codes, scales, amax, denominator)
    # Half of the widest E2M1 step (4 to 6) at each block's decode scale; saturation stays inside it.
    block_decode = (scales.float() * (amax / denominator)[:, None, None]).repeat_interleave(16, dim=-1)
    assert torch.all((decoded - weight).abs() <= block_decode)
    again = quantize_nvfp4_experts(decoded, denominator)
    assert torch.equal(again.codes, codes) and torch.equal(again.scales.view(torch.uint8), scales.view(torch.uint8))
    values, exponents = quantize_mxfp8_experts(weight)
    decoded = decode_oracle(values, exponents)
    # Relative E4M3 rounding is at most 2**-4; below 2**-6 of the scale the absolute step is 2**-9.
    scale = torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=-1)
    assert torch.all((decoded - weight).abs() <= torch.maximum(weight.abs() * 2.0**-4, scale * 2.0**-10))
    # Grid values decode to themselves; a block whose maximum rounded down to 224 may take the next lower scale.
    torch.testing.assert_close(decode_oracle(*quantize_mxfp8_experts(decoded)), decoded, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["nan", "inf", "width", "integer", "denominator", "denominator_shape"])
def test_encoders_reject_inputs_the_layout_cannot_hold(case: str) -> None:
    weight = torch.randn(2, 4, 64)
    denominator = torch.tensor([2688.0])
    if case == "nan":
        weight[1, 2, 3] = float("nan")
    elif case == "inf":
        weight[0, 0, 0] = float("inf")
    elif case == "width":
        weight = torch.randn(2, 4, 40)
    elif case == "integer":
        weight = torch.ones(2, 4, 64, dtype=torch.int32)
    elif case == "denominator":
        denominator = torch.tensor([0.0])
    else:
        denominator = torch.tensor(2688.0)
    with pytest.raises(ValueError):
        quantize_nvfp4_experts(weight, denominator)
    if case in ("nan", "inf", "width", "integer"):
        with pytest.raises(ValueError):
            quantize_mxfp8_experts(weight)


def write_released(folder: Path, checkpoint: dict[str, torch.Tensor]) -> Path:
    """Write a released two-layer checkpoint: NVFP4 layer 0, MXFP8 layer 1, sidecar and extra files."""
    folder.mkdir(parents=True)
    write_checkpoint(checkpoint, folder)
    write_released_checkpoint_metadata(folder, nvfp4_layers=[0], mxfp8_layers=[1])
    (folder / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 64}))
    # Inkling releases ship tiktoken/tokenizer.model; hidden entries like .cache/ are not part of the release.
    (folder / "tiktoken").mkdir()
    (folder / "tiktoken/tokenizer.model").write_bytes(b"tiktoken ranks")
    (folder / "tiktoken/.lock").write_bytes(b"")
    (folder / ".cache").mkdir()
    return folder


def released_tensors(folder: Path) -> dict[str, torch.Tensor]:
    with safe_open(str(folder / "model.safetensors"), framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


def floating_export(model: Inkling2ForCausalLM, checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """What a consolidated save of the unchanged model holds: released names, floating experts."""
    native = model.state_dict_adapter.from_hf(checkpoint)
    return {key: value.contiguous() for key, value in model.state_dict_adapter.to_hf(native).items()}


def write_trained(folder: Path, tensors: dict[str, torch.Tensor]) -> Path:
    folder.mkdir(parents=True)
    write_checkpoint(tensors, folder)
    return folder


def assert_same_bytes(actual: torch.Tensor, expected: torch.Tensor, key: str) -> None:
    assert actual.dtype == expected.dtype and actual.shape == expected.shape, key
    assert torch.equal(actual.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)), key


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unchanged_training_save_exports_the_released_bytes(tmp_path: Path, dtype: torch.dtype) -> None:
    """Released checkpoint -> shared loader -> consolidated save -> export reproduces every released byte."""
    model, checkpoint, _ = mixed_fixture()
    released = write_released(tmp_path / "released", checkpoint)
    config = checkpoint_config()
    config.dtype = dtype
    model = Inkling2ForCausalLM(config)
    checkpointer = CheckpointingConfig(
        is_async=False,
        model_save_format="safetensors",
        save_consolidated="final",
        model_repo_id=str(released),
        original_model_root_dir=str(released),
        model_cache_dir=str(released),
    ).build(0, 0, 0)
    try:
        checkpointer.load_base_model(model, torch.device("cpu"), str(released), str(released))
        model.save_pretrained(str(tmp_path / "saved"), checkpointer=checkpointer, is_final_checkpoint=True)
    finally:
        checkpointer.close()
    floating = tmp_path / "saved/model/consolidated"
    saved = json.loads((floating / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(floating / saved[W13]), framework="pt") as handle:
        assert handle.get_tensor(W13).dtype == dtype
    # The floating save is not the released layout: its experts are floating, with no scales.
    assert set(saved) < set(checkpoint)
    out = tmp_path / "exported"
    summary = export_quantized(released=released, trained=floating, out=out)
    assert (summary.kept_experts, summary.encoded_experts) == (16, 0)
    assert summary.converted_tensors == 0 and summary.released_only_tensors == 0
    exported = released_tensors(out)
    assert set(exported) == set(checkpoint)
    for key, value in checkpoint.items():
        assert_same_bytes(exported[key], value, key)
    names = ["config.json", "hf_quant_config.json", "model.safetensors.index.json", "tokenizer_config.json"]
    for name in [*names, "tiktoken/tokenizer.model"]:
        assert (out / name).read_bytes() == (released / name).read_bytes(), name
    assert sorted(p.name for p in out.iterdir()) == sorted([*names, "model.safetensors", "tiktoken"])
    assert [p.name for p in (out / "tiktoken").iterdir()] == ["tokenizer.model"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["exported", "released", "saved"]


def test_changed_experts_are_encoded_and_the_others_keep_their_bytes(tmp_path: Path) -> None:
    model, checkpoint, _ = mixed_fixture()
    released = write_released(tmp_path / "released", checkpoint)
    trained = floating_export(model, checkpoint)
    torch.manual_seed(3)
    trained[W13][2] += 0.01 * torch.randn_like(trained[W13][2])
    trained[W2_MXFP8][3] *= 1.5
    trained["model.llm.norm.weight"] = trained["model.llm.norm.weight"] + 1
    out = tmp_path / "exported"
    summary = export_quantized(released=released, trained=write_trained(tmp_path / "trained", trained), out=out)
    assert (summary.kept_experts, summary.encoded_experts) == (14, 2)
    assert (summary.kept_tensors, summary.converted_tensors) == (len(trained) - 5, 1)
    exported = released_tensors(out)
    changed = {W13: 2, W13 + "_scale": 2, W13 + "_amax": 2, W2_MXFP8: 3, W2_MXFP8 + "_scale": 3}
    for key, value in checkpoint.items():
        if key == "model.llm.norm.weight":
            assert_same_bytes(exported[key], trained[key], key)
        elif key in changed:
            for expert in set(range(value.shape[0])) - {changed[key]}:
                assert_same_bytes(exported[key][expert], value[expert], key)
        else:
            assert_same_bytes(exported[key], value, key)
    codes, scales, amax = reference_nvfp4(trained[W13][2:3], 1536.0)
    assert torch.equal(exported[W13][2:3], codes)
    assert torch.equal(exported[W13 + "_scale"][2:3].view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(exported[W13 + "_amax"][2:3], amax)
    values, exponents = reference_mxfp8(trained[W2_MXFP8][3:4])
    assert torch.equal(exported[W2_MXFP8][3:4].view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(exported[W2_MXFP8 + "_scale"][3:4], exponents)


def test_fully_trained_export_reloads_through_the_released_loader(tmp_path: Path) -> None:
    model, checkpoint, _ = mixed_fixture()
    released = write_released(tmp_path / "released", checkpoint)
    trained = floating_export(model, checkpoint)
    torch.manual_seed(5)
    experts = [key for key in trained if ".mlp.experts." in key]
    for key in experts:
        trained[key] = (trained[key] + 0.05 * torch.randn_like(trained[key])).to(torch.bfloat16)
    out = tmp_path / "exported"
    assert (
        main(
            [
                "--released",
                str(released),
                "--trained",
                str(write_trained(tmp_path / "trained", trained)),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    exported = released_tensors(out)
    expected = {}
    for key in experts:
        layer = int(key.split(".")[3])
        native = f"layers.{layer}.mlp.experts." + ("gate_and_up_projs" if "w13" in key else "down_projs")
        if layer == 0:
            denominator = float(checkpoint[key + "_outer_scale_denominator"][0])
            codes, scales, amax = reference_nvfp4(trained[key], denominator)
            assert torch.equal(exported[key], codes) and torch.equal(exported[key + "_amax"], amax), key
            expected[native] = decode_oracle(codes, scales, amax, torch.tensor([denominator]))
        else:
            values, exponents = reference_mxfp8(trained[key])
            assert torch.equal(exported[key].view(torch.uint8), values.view(torch.uint8)), key
            expected[native] = decode_oracle(values, exponents)
    # The export loads like the release: the sidecar selects the packed path and the decoded values match.
    config = Inkling2Config.from_dict(json.loads((out / "config.json").read_text()), name_or_path=str(out))
    reloaded = Inkling2ForCausalLM.from_config(config)
    assert reloaded.state_dict_adapter.nvfp4_layer_count == 1
    checkpointer = CheckpointingConfig(
        is_async=False,
        model_save_format="safetensors",
        model_repo_id=str(out),
        original_model_root_dir=str(out),
        model_cache_dir=str(out),
    ).build(0, 0, 0)
    try:
        checkpointer.load_base_model(reloaded, torch.device("cpu"), str(out), str(out))
    finally:
        checkpointer.close()
    state = reloaded.state_dict()
    for native, value in expected.items():
        torch.testing.assert_close(state[native], value.transpose(-1, -2), rtol=0, atol=0, msg=native)


def test_text_only_export_copies_the_vision_tower(tmp_path: Path) -> None:
    model, checkpoint, _ = mixed_fixture()
    checkpoint["model.visual.proj_in.weight"] = torch.randn(8, 4, dtype=torch.bfloat16)
    released = write_released(tmp_path / "released", checkpoint)
    trained = floating_export(model, checkpoint)
    assert not any(key.startswith("model.visual.") for key in trained)
    summary = export_quantized(
        released=released, trained=write_trained(tmp_path / "trained", trained), out=tmp_path / "out"
    )
    assert summary.released_only_tensors == 1
    exported = released_tensors(tmp_path / "out")
    assert_same_bytes(exported["model.visual.proj_in.weight"], checkpoint["model.visual.proj_in.weight"], "visual")


@pytest.mark.parametrize("case", ["missing", "unknown", "expert_shape", "tensor_shape", "out_exists"])
def test_mismatched_inputs_are_rejected_without_output(tmp_path: Path, case: str) -> None:
    model, checkpoint, _ = mixed_fixture()
    released = write_released(tmp_path / "released", checkpoint)
    trained = floating_export(model, checkpoint)
    out = tmp_path / "out"
    error: type[Exception] = ValueError
    if case == "missing":
        del trained["model.llm.layers.1.attn.wq_du.weight"]
        match = "model.llm.layers.1.attn.wq_du.weight"
    elif case == "unknown":
        trained["model.llm.layers.0.attn.extra.weight"] = torch.zeros(2)
        match = "model.llm.layers.0.attn.extra.weight"
    elif case == "expert_shape":
        trained[W13] = trained[W13].transpose(-1, -2).contiguous()
        match = W13
    elif case == "tensor_shape":
        trained["model.llm.norm.weight"] = torch.zeros(3)
        match = "model.llm.norm.weight"
    else:
        out.mkdir()
        error, match = FileExistsError, "already exists"
    with pytest.raises(error, match=match):
        export_quantized(released=released, trained=write_trained(tmp_path / "trained", trained), out=out)
    assert out.exists() == (case == "out_exists")
    assert not any(p.name.endswith(".partial") for p in tmp_path.iterdir())

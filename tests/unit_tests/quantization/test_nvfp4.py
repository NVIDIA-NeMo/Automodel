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

from collections import OrderedDict
from collections.abc import Collection, Sequence

import pytest
import torch
from torch import nn

import nemo_automodel.components.quantization.nvfp4 as nvfp4_module
from nemo_automodel.components.models.common.te_precision import TEModulePrecision, TEPrecisionPlan
from nemo_automodel.components.models.common.utils import HAVE_TE, TENvfp4Config
from nemo_automodel.components.models.nemotron_v3.quantization import NemotronV3PrecisionMap
from nemo_automodel.components.quantization.mxfp4 import _FP4_E2M1_TABLE
from nemo_automodel.components.quantization.nvfp4 import (
    E2M1_MAX,
    E4M3_MAX,
    NVFP4_BLOCK_SIZE,
    NVFP4GroupedMM,
    dequantize_nvfp4,
    quantize_nvfp4,
    quantize_nvfp4_modelopt,
)
from nemo_automodel.shared.import_utils import is_te_min_version

E, N, K = 3, 16, 128  # [experts, out, in]; K blockable by 16 and 16-byte aligned for the grouped GEMM


def _grouped_mm_available(device: torch.device) -> bool:
    x = torch.randn(4, K, dtype=torch.bfloat16, device=device)
    w = torch.randn(2, K, N, dtype=torch.bfloat16, device=device)
    try:
        torch._grouped_mm(x, w, offs=torch.tensor([2, 4], dtype=torch.int32, device=device))
    except (RuntimeError, NotImplementedError):
        return False
    return True


def test_storage_layout_is_half_byte_per_param_plus_block_and_row_scales():
    w = torch.randn(E, N, K, dtype=torch.bfloat16)
    packed, scales, global_scale = quantize_nvfp4(w)
    assert packed.dtype == torch.uint8 and packed.shape == (E, N, K // 2)
    assert scales.dtype == torch.float8_e4m3fn and scales.shape == (E, N, K // NVFP4_BLOCK_SIZE)
    assert global_scale.dtype == torch.float32 and global_scale.shape == (E, N)
    assert packed.numel() + scales.numel() == w.numel() * 0.5625


def test_last_dim_must_be_blockable():
    with pytest.raises(ValueError, match="divisible"):
        quantize_nvfp4(torch.randn(2, 8, 40))


def test_modelopt_layout_decodes_even_column_from_the_low_nibble():
    """The ModelOpt / TensorRT-LLM packing keeps column 2i in the low nibble of byte i."""
    codes = torch.tensor([[1, 7, 9, 0]], dtype=torch.uint8)  # 0.5, 6.0, -0.5, 0.0
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    scales = torch.ones(1, 1, dtype=torch.float32).to(torch.float8_e4m3fn)
    packed = torch.cat([packed, torch.zeros(1, NVFP4_BLOCK_SIZE // 2 - 2, dtype=torch.uint8)], dim=-1)
    out = dequantize_nvfp4(packed, scales, torch.ones(1), torch.float32)
    assert out[0, :4].tolist() == [0.5, 6.0, -0.5, 0.0]


def test_values_on_the_e2m1_grid_round_trip_exactly():
    torch.manual_seed(0)
    n_blocks = K // NVFP4_BLOCK_SIZE
    codes = torch.randint(0, 16, (E, N, n_blocks, NVFP4_BLOCK_SIZE))
    codes[..., 0] = 7  # a +6.0 code in every block, so amax(block) / 6 recovers the block scale exactly
    # Power-of-two block scales with the row maximum fixed at 4: the row scale becomes 4 / 448 and every
    # recovered block scale is 2^e * 112 = 7 * 2^(e+4), representable in e4m3 (3 mantissa bits).
    exponents = torch.randint(-3, 3, (E, N, n_blocks))
    exponents[..., 0] = 2
    scales = 2.0 ** exponents.float()
    w = (_FP4_E2M1_TABLE[codes] * scales.unsqueeze(-1)).reshape(E, N, K)
    packed, q_scales, global_scale = quantize_nvfp4(w)
    torch.testing.assert_close(global_scale, torch.full((E, N), 4.0 / 448.0), rtol=0, atol=0)
    torch.testing.assert_close(q_scales.float(), scales * 112.0, rtol=0, atol=0)
    torch.testing.assert_close(dequantize_nvfp4(packed, q_scales, global_scale, torch.float32), w, rtol=1e-6, atol=1e-6)


def test_modelopt_quantization_uses_one_tensor_scale():
    torch.manual_seed(0)
    w = torch.randn(N, K, dtype=torch.bfloat16) * 3
    packed, scales, tensor_scale = quantize_nvfp4_modelopt(w)
    assert tensor_scale.shape == () and tensor_scale.dtype == torch.float32
    torch.testing.assert_close(tensor_scale, w.float().abs().amax() / (E2M1_MAX * E4M3_MAX), rtol=0, atol=0)
    assert packed.shape == (N, K // 2) and scales.shape == (N, K // NVFP4_BLOCK_SIZE)
    dq = dequantize_nvfp4(packed, scales, tensor_scale.expand(N), torch.float32)
    err = (dq - w.float()).abs()
    scale = scales.float() * tensor_scale
    assert torch.all(err.view(N, -1, NVFP4_BLOCK_SIZE) <= 1.0 * scale.unsqueeze(-1) + 1e-6)
    # The same codes and block scales come out of the row-wise API when the tensor scale is passed in.
    packed2, scales2, rows = quantize_nvfp4(w, global_scale=tensor_scale)
    assert torch.equal(packed2, packed) and torch.equal(scales2.view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(rows, tensor_scale.expand(N))


def test_chunked_dequantization_matches_single_chunk(monkeypatch):
    torch.manual_seed(0)
    packed, scales, global_scale = quantize_nvfp4(torch.randn(E, N, K, dtype=torch.bfloat16))
    whole = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
    monkeypatch.setattr(nvfp4_module, "_DEQUANT_CHUNK_ELEMS", 5 * K)  # five rows per chunk, rows not divisible by it
    chunked = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
    assert torch.equal(whole, chunked)


def test_dequantize_rejects_mismatched_scale_shapes():
    packed, scales, global_scale = quantize_nvfp4(torch.randn(E, N, K))
    with pytest.raises(ValueError, match="scales shape"):
        dequantize_nvfp4(packed, scales[:, :, :-1], global_scale, torch.float32)
    with pytest.raises(ValueError, match="global_scale shape"):
        dequantize_nvfp4(packed, scales, global_scale[0], torch.float32)


def test_dequantize_error_is_bounded_by_the_e2m1_step():
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.float32)
    packed, scales, global_scale = quantize_nvfp4(w)
    dq = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
    scale = scales.to(torch.float32) * global_scale.unsqueeze(-1)  # [E, N, K // 16]
    err = (dq - w).abs().view(E, N, -1, NVFP4_BLOCK_SIZE)
    # Largest gap between consecutive e2m1 magnitudes is 2 (4 -> 6), so rounding costs at most one decode scale;
    # a block whose e4m3 scale rounded down clamps its amax to 6 at a cost below 0.4 scales.
    bound = 1.0 * scale.unsqueeze(-1) + 1e-6
    assert torch.all(err <= bound)
    assert torch.all(scales.to(torch.float32) <= E4M3_MAX)


def test_quantize_dequantize_idempotent():
    torch.manual_seed(0)
    w = torch.randn(E, N, K, dtype=torch.bfloat16)
    dq1 = dequantize_nvfp4(*quantize_nvfp4(w), torch.bfloat16)
    dq2 = dequantize_nvfp4(*quantize_nvfp4(dq1), torch.bfloat16)
    assert torch.equal(dq1, dq2)


def test_zero_row_decodes_to_zeros():
    w = torch.zeros(2, K, dtype=torch.bfloat16)
    w[1, NVFP4_BLOCK_SIZE:] = 1.5
    packed, scales, global_scale = quantize_nvfp4(w)
    dq = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)
    assert torch.equal(dq[0], torch.zeros(K, dtype=torch.bfloat16))
    assert torch.equal(dq[1, :NVFP4_BLOCK_SIZE], torch.zeros(NVFP4_BLOCK_SIZE, dtype=torch.bfloat16))
    assert torch.equal(dq[1, NVFP4_BLOCK_SIZE:], torch.full((K - NVFP4_BLOCK_SIZE,), 1.5, dtype=torch.bfloat16))


def test_int8_packed_input_is_accepted():
    w = torch.randn(2, K, dtype=torch.bfloat16)
    packed, scales, global_scale = quantize_nvfp4(w)
    a = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
    b = dequantize_nvfp4(packed.view(torch.int8), scales, global_scale, torch.float32)
    assert torch.equal(a, b)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_grouped_mm_dequantizes_and_backpropagates_to_the_input(device):
    device = torch.device(device)
    if not _grouped_mm_available(device):
        pytest.skip("torch._grouped_mm is not available on this device")
    torch.manual_seed(0)
    packed, scales, global_scale = quantize_nvfp4(torch.randn(E, N, K, dtype=torch.bfloat16, device=device))
    x = torch.randn(8, K, dtype=torch.bfloat16, device=device, requires_grad=True)
    offs = torch.tensor([3, 3, 8], dtype=torch.int32, device=device)  # expert 1 receives no tokens
    out = NVFP4GroupedMM.apply(x, packed, scales, global_scale, offs)
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    x_ref = x.detach().clone().requires_grad_(True)
    w_t = dequantize_nvfp4(packed, scales, global_scale, torch.bfloat16)  # [E, N, K]
    ref = torch._grouped_mm(x_ref, w_t.transpose(-2, -1), offs=offs)
    ref.backward(grad_out)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=0, atol=0)


def test_quantize_keeps_subnormal_block_scales():
    """e4m3 block scales reach 2^-9 (subnormal); clamping at the normal minimum would zero small blocks."""
    w = torch.zeros(1, 2 * NVFP4_BLOCK_SIZE)
    w[0, 0], w[0, 1] = 6.0 / 512, 1.0 / 512  # block 0: max code 6 at scale 2^-9, second value code 1
    w[0, NVFP4_BLOCK_SIZE] = 6.0  # block 1 keeps the row amax at 6
    packed, scales, global_scale = quantize_nvfp4(w, global_scale=torch.tensor(1.0))
    assert float(scales[0, 0]) == 2.0**-9 and float(scales[0, 1]) == 1.0
    dq = dequantize_nvfp4(packed, scales, global_scale, torch.float32)
    assert dq[0, 0].item() == 6.0 / 512 and dq[0, 1].item() == 1.0 / 512 and dq[0, NVFP4_BLOCK_SIZE].item() == 6.0
    assert torch.equal(dq[0, 2:NVFP4_BLOCK_SIZE], torch.zeros(NVFP4_BLOCK_SIZE - 2))


# ``NemotronV3PrecisionMap`` against the 88-layer pattern of Nemotron 3 Super / 3.5 Super VL.
#
# The expected precision of every module is written out here from the layer pattern with plain loops; the test
# never asks the map which layer or class a name belongs to. Module names are the ones the model registers
# (``test_names_match_a_built_model`` builds a small model to pin them).


# ``layers_block_type`` of nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16 (llm_config) as one symbol per
# layer: M = Mamba, E = MoE, * = attention.
SUPER_PATTERN = "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME"
TAIL = 14

ATTENTION = ["q_proj", "k_proj", "v_proj", "o_proj"]
MOE = {
    "experts.gate_up_linear": "routed_experts",
    "experts.down_linear": "routed_experts",
    "shared_experts.up_proj": "shared_experts",
    "shared_experts.down_proj": "shared_experts",
    "fc1_latent_proj": "latent_proj",
    "fc2_latent_proj": "latent_proj",
}
RECIPE = {
    "mamba_in_proj": "nvfp4",
    "mamba_out_proj": "mxfp8",
    "routed_experts": "nvfp4",
    "shared_experts": "nvfp4",
    "attention": "high_precision",
    "latent_proj": "high_precision",
    "mtp": "high_precision",
}


def _recipe_map(**changes: str | int) -> NemotronV3PrecisionMap:
    return NemotronV3PrecisionMap(**{"high_precision_tail_layers": TAIL, **RECIPE, **changes})


def _expected(
    pattern: str, tail: int, prefix: str, block_infix: str, recipe: dict[str, str], moe_infix: str = ""
) -> dict[str, tuple[str, int | None, str, str]]:
    """GEMM module name -> (module class, layer, precision, scope), in ``named_modules`` order.

    ``block_infix`` is the checkpoint wrapper of every block (``activation_checkpointing: true``), ``moe_infix``
    the one inside every MoE mixer (``activation_checkpointing: moe``).
    """
    expected: dict[str, tuple[str, int | None, str, str]] = {}

    def add(name: str, module_class: str, layer: int | None, scope: str) -> None:
        in_tail = layer is not None and layer >= len(pattern) - tail
        expected[name] = (module_class, layer, "high_precision" if in_tail else recipe[module_class], scope)

    for layer, symbol in enumerate(pattern):
        mixer = f"{prefix}model.layers.{layer}.{block_infix}mixer"
        if symbol == "M":
            add(f"{mixer}.in_proj", "mamba_in_proj", layer, f"{mixer}.in_proj")
            add(f"{mixer}.out_proj", "mamba_out_proj", layer, f"{mixer}.out_proj")
        elif symbol == "*":
            for projection in ATTENTION:
                add(f"{mixer}.{projection}", "attention", layer, mixer)
        else:
            for module, module_class in MOE.items():
                add(f"{mixer}.{moe_infix}{module}", module_class, layer, f"{mixer}.{moe_infix}{module.split('.')[0]}")
    # One MTP depth of [attention, moe], with the fusion projection on its first block.
    attention_mixer = f"{prefix}mtp.layers.0.{block_infix}mixer"
    for projection in ATTENTION:
        add(f"{attention_mixer}.{projection}", "mtp", None, attention_mixer)
    add(f"{prefix}mtp.layers.0.{block_infix}eh_proj", "mtp", None, f"{prefix}mtp.layers.0.{block_infix}eh_proj")
    moe_mixer = f"{prefix}mtp.layers.1.{block_infix}mixer"
    for module in MOE:
        add(f"{moe_mixer}.{moe_infix}{module}", "mtp", None, f"{moe_mixer}.{moe_infix}{module.split('.')[0]}")
    return expected


def _resolved(plan: TEPrecisionPlan) -> dict[str, tuple[str, int | None, str, str]]:
    return {m.name: (m.module_class, m.layer, m.precision, m.scope) for m in plan.modules}


WRAPPER = "_checkpoint_wrapped_module."


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefix", ["", "language_model."])
@pytest.mark.parametrize(("block_infix", "moe_infix"), [("", ""), (WRAPPER, ""), ("", WRAPPER)])
def test_super_recipe_resolves_every_module(prefix: str, block_infix: str, moe_infix: str):
    """Text model and vision-language wrapper; no checkpoint wrappers, around the blocks, around the MoE mixers."""
    expected = _expected(SUPER_PATTERN, TAIL, prefix, block_infix, RECIPE, moe_infix)
    plan = _recipe_map().resolve(list(expected), list(expected))
    assert _resolved(plan) == expected
    assert (plan.num_layers, plan.high_precision_tail_layers) == (88, 14)
    # Module counts, from the published layout: 34 + 6 Mamba, 33 + 7 MoE (two GEMMs each), 8 attention.
    assert plan.counts() == {
        "mamba_in_proj": {"nvfp4": 34, "mxfp8": 0, "high_precision": 6},
        "mamba_out_proj": {"nvfp4": 0, "mxfp8": 34, "high_precision": 6},
        "routed_experts": {"nvfp4": 66, "mxfp8": 0, "high_precision": 14},
        "shared_experts": {"nvfp4": 66, "mxfp8": 0, "high_precision": 14},
        "latent_proj": {"nvfp4": 0, "mxfp8": 0, "high_precision": 80},
        "attention": {"nvfp4": 0, "mxfp8": 0, "high_precision": 32},
        "mtp": {"nvfp4": 0, "mxfp8": 0, "high_precision": 11},
    }
    summary = plan.summary()
    assert "decoder layers: 88; layers 0-73 follow the precision of their module class" in summary
    assert "layers 74-87 run in high precision" in summary
    assert [line.split() for line in summary.splitlines() if line.startswith(("mamba_out_proj", "total"))] == [
        ["mamba_out_proj", "0", "34", "6"],
        ["total", "166", "34", "163"],
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "changes, differing",
    [
        # A tail one layer short or long moves exactly the quantized GEMMs of layer 74 (MoE: routed and
        # shared experts) or of layer 73 (Mamba: both projections).
        ({"high_precision_tail_layers": 13}, 4),
        ({"high_precision_tail_layers": 15}, 2),
        ({"high_precision_tail_layers": 0}, 6 * 2 + 7 * 4),
        ({"mamba_out_proj": "nvfp4"}, 34),
        ({"mamba_in_proj": "high_precision"}, 34),
        ({"attention": "nvfp4"}, 7 * 4),
        ({"latent_proj": "nvfp4"}, 33 * 2),
        ({"routed_experts": "mxfp8"}, 33 * 2),
        ({"mtp": "nvfp4"}, 11),
    ],
)
def test_a_different_map_is_told_apart_from_the_recipe(changes: dict[str, str | int], differing: int):
    """Negative control: every wrong tail or module class changes the expected number of modules."""
    expected = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE)
    resolved = _resolved(_recipe_map(**changes).resolve(list(expected), list(expected)))
    assert sum(resolved[name][2] != expected[name][2] for name in expected) == differing


@pytest.mark.cpu_only
def test_scopes_are_the_modules_that_select_the_precision():
    expected = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE)
    scopes = _recipe_map().resolve(list(expected), list(expected)).scopes()
    assert scopes["model.layers.0.mixer.in_proj"] == "nvfp4"
    assert scopes["model.layers.0.mixer.out_proj"] == "mxfp8"
    assert scopes["model.layers.1.mixer.experts"] == "nvfp4"
    assert scopes["model.layers.1.mixer.shared_experts"] == "nvfp4"
    assert scopes["model.layers.7.mixer"] == "high_precision"
    assert scopes["model.layers.73.mixer.out_proj"] == "mxfp8"
    assert scopes["model.layers.74.mixer.experts"] == "high_precision"
    assert scopes["model.layers.75.mixer.out_proj"] == "high_precision"
    assert scopes["model.layers.87.mixer.experts"] == "high_precision"
    assert scopes["mtp.layers.1.mixer.experts"] == "high_precision"
    # 80 Mamba projections, 8 attention mixers, 40 x (experts, shared expert, 2 latent), 6 in the MTP head.
    assert len(scopes) == 80 + 8 + 40 * 4 + 6
    # With the MoE mixers inside checkpoint wrappers the scopes are the modules the recompute calls again.
    wrapped = _expected(SUPER_PATTERN, TAIL, "", "", RECIPE, WRAPPER)
    wrapped_scopes = _recipe_map().resolve(list(wrapped), list(wrapped)).scopes()
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.experts"] == "nvfp4"
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.shared_experts"] == "nvfp4"
    assert wrapped_scopes["model.layers.1.mixer._checkpoint_wrapped_module.fc2_latent_proj"] == "high_precision"
    assert wrapped_scopes["model.layers.7.mixer"] == "high_precision"
    assert len(wrapped_scopes) == len(scopes)
    assert not any(
        scope.endswith(("mixer", "_checkpoint_wrapped_module")) for scope in wrapped_scopes if "layers.1." in scope
    )


@pytest.mark.cpu_only
def test_models_it_does_not_understand_are_rejected():
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    recipe = _recipe_map(high_precision_tail_layers=1)
    assert len(recipe.resolve(names, names).modules) == len(names)
    foreign = ["model.language_model.layers.0.mlp.experts.gate_up_linear", "lm_head"]
    with pytest.raises(ValueError, match="found no module named 'model.layers"):
        recipe.resolve(foreign, foreign)
    with pytest.raises(ValueError, match="found no module named 'model.layers"):
        recipe.resolve([], [])
    # ``mixer.up_proj`` is a projection of a dense MLP block (block type "mlp"), which the map has no class for.
    extra = [*names, "model.layers.1.mixer.up_proj", "vision_projector.0"]
    with pytest.raises(
        ValueError, match=r"does not know 2 .*model.layers.1.mixer.up_proj, vision_projector.0\. .*dense MLP block"
    ):
        recipe.resolve(extra, extra)
    # A pipeline stage holds layers 2-3 only: the last layers of the model cannot be located.
    stage = [name for name in names if "layers.0." not in name and "layers.1." not in name]
    with pytest.raises(ValueError, match="whole decoder stack"):
        recipe.resolve(stage, stage)
    with pytest.raises(ValueError, match="exceeds the 4 decoder layers"):
        _recipe_map(high_precision_tail_layers=5).resolve(names, names)


@pytest.mark.cpu_only
def test_the_output_projection_has_no_class():
    """``lm_head`` is a torch linear: it is no entry of the plan, and a Transformer Engine one is not guessed."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    recipe = _recipe_map(high_precision_tail_layers=1)
    for head in ("lm_head", "language_model.lm_head"):
        plan = recipe.resolve([*names, head], names)
        assert [module.name for module in plan.modules] == names
        with pytest.raises(ValueError, match=rf"does not know 1 .*: {head}\. "):
            recipe.resolve([*names, head], [*names, head])
    with pytest.raises(TypeError, match="lm_head"):
        NemotronV3PrecisionMap(**{"high_precision_tail_layers": 1, **RECIPE, "lm_head": "high_precision"})


@pytest.mark.cpu_only
def test_quantizing_a_module_without_transformer_engine_is_rejected():
    """A quantized class whose modules are plain torch modules would silently run in high precision."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    recipe = _recipe_map(high_precision_tail_layers=1)
    # backend.mamba_linear="torch" (the default): the Mamba projections exist but are not Transformer Engine
    # modules. The error names the setting that builds them as such.
    torch_mamba = [name for name in names if not name.endswith(("in_proj", "out_proj"))]
    with pytest.raises(
        ValueError, match=r"quantizes 2 .*backend.mamba_linear='te'.*layers.0.mixer.in_proj \(mamba_in_proj: nvfp4\)"
    ):
        recipe.resolve(names, torch_mamba)
    plan = _recipe_map(
        high_precision_tail_layers=1, mamba_in_proj="high_precision", mamba_out_proj="high_precision"
    ).resolve(names, torch_mamba)
    assert [module.name for module in plan.modules] == torch_mamba
    assert plan.num_layers == 4  # the Mamba layers still count, although they hold no GEMM module of the plan
    # An expert backend without Transformer Engine grouped GEMMs: the container is there, its projections are not.
    torch_experts = [name for name in names if ".experts." not in name]
    modules = [*torch_experts, "model.layers.1.mixer.experts", "mtp.layers.1.mixer.experts"]
    with pytest.raises(ValueError, match=r"quantizes 1 .*model.layers.1.mixer.experts \(routed_experts: nvfp4\)"):
        recipe.resolve(modules, torch_experts)
    _recipe_map(high_precision_tail_layers=1, routed_experts="high_precision").resolve(modules, torch_experts)
    # The same container inside the checkpoint wrapper of its mixer.
    wrapped_names = list(_expected("ME*M", 1, "", "", RECIPE, WRAPPER))
    wrapped_torch_experts = [name for name in wrapped_names if ".experts." not in name]
    wrapped_modules = [*wrapped_torch_experts, f"model.layers.1.mixer.{WRAPPER}experts"]
    with pytest.raises(ValueError, match=r"quantizes 1 .*mixer._checkpoint_wrapped_module.experts \(routed_experts"):
        recipe.resolve(wrapped_modules, wrapped_torch_experts)
    # The tail keeps a module in high precision whatever backend built it.
    tail_mamba = [name for name in names if not name.startswith("model.layers.3.")]
    assert len(recipe.resolve(names, tail_mamba).modules) == len(names) - 2


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "changes", [{"mamba_out_proj": "bf16"}, {"high_precision_tail_layers": -1}, {"high_precision_tail_layers": "14"}]
)
def test_invalid_settings_are_rejected(changes: dict[str, str | int]):
    with pytest.raises(ValueError, match="must be"):
        _recipe_map(**changes)


@pytest.mark.cpu_only
def test_exclude_fqns_must_agree_with_the_map():
    """``exclude_fqns`` keeps working next to the map, but may not contradict it."""
    names = list(_expected("ME*M", 1, "", "", RECIPE))
    gemm_modules = [(name, "GroupedLinear" if ".experts." in name else "Linear") for name in names]
    recipe = _recipe_map(high_precision_tail_layers=1)
    agreeing = TENvfp4Config(precision_map=recipe, exclude_fqns=["lm_head", "q_proj"])
    assert _resolved(agreeing.resolve_precision_plan(names, gemm_modules)) == _expected("ME*M", 1, "", "", RECIPE)
    with pytest.raises(ValueError, match="mixer.q_proj' the precision high_precision, .*precision_map gives it nvfp4"):
        TENvfp4Config(
            precision_map=_recipe_map(high_precision_tail_layers=1, attention="nvfp4"), exclude_fqns=["q_proj"]
        ).resolve_precision_plan(names, gemm_modules)
    with pytest.raises(ValueError, match="mixer.in_proj' the precision high_precision, .*precision_map gives it nvfp4"):
        TENvfp4Config(precision_map=recipe, exclude_fqns=["in_proj"]).resolve_precision_plan(names, gemm_modules)


@pytest.mark.cpu_only
@pytest.mark.parametrize("checkpointing", ["none", "blocks", "moe"])
def test_names_match_a_built_model(checkpointing: str):
    """The names the test spells out are the ones a Nemotron 3 model registers, also after ``apply_ac``."""
    from nemo_automodel.components.models.common import BackendConfig, LMHead
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM
    from nemo_automodel.components.moe.parallelizer import apply_ac
    from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config

    config = MockNemotronV3Config(
        num_hidden_layers=4,
        layers_block_type=["mamba", "moe", "attention", "mamba"],
        moe_latent_size=64,
        num_nextn_predict_layers=1,
        mtp_hybrid_override_pattern="*E",
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = NemotronHForCausalLM(config, backend)
    # The output projection is not built from ``backend.linear``, so it is never a Transformer Engine module.
    assert type(model.lm_head) is LMHead
    block_infix = moe_infix = ""
    if checkpointing != "none":
        # The production wrapping of the expert-parallel path; the MTP head here has one depth, so its MoE
        # block is wrapped like the decoder blocks.
        apply_ac(
            model,
            hidden_size=config.hidden_size,
            num_experts=config.n_routed_experts,
            moe_only=checkpointing == "moe",
        )
        block_infix, moe_infix = (WRAPPER, "") if checkpointing == "blocks" else ("", WRAPPER)
    built = {name for name, _ in model.named_modules()}
    # The torch expert backend keeps its weights on ``mixer.experts`` itself; Transformer Engine experts add
    # the two grouped projections below it.
    spelled_out = _expected("ME*M", 1, "", block_infix, RECIPE, moe_infix)
    expected = {name.replace(".gate_up_linear", "").replace(".down_linear", "") for name in spelled_out}
    assert expected <= built, sorted(expected - built)
    # The map accepts exactly these names: every module of the built model that would be a Transformer Engine
    # GEMM module with ``backend.linear="te"`` gets a precision, in all three layouts.
    gemm_names = [name for name in spelled_out if ".experts." not in name]
    plan = _recipe_map(high_precision_tail_layers=1, routed_experts="high_precision").resolve(
        [name for name, _ in model.named_modules()], gemm_names
    )
    assert [module.name for module in plan.modules] == gemm_names
    assert "lm_head" in built and "lm_head" not in gemm_names
    assert set(plan.scopes()) <= built


# ``te_nvfp4`` through real Transformer Engine modules (Blackwell GPUs only).


def _nvfp4_capable() -> bool:
    return HAVE_TE and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


requires_te_nvfp4 = pytest.mark.skipif(
    not (_nvfp4_capable() and is_te_min_version("2.19.0")),
    reason="requires Transformer Engine 2.19+ and an NVFP4 GPU",
)

# Row-scaled activations need the random Hadamard transform off; the backward reads dequantized operands so
# that the weights do not have to be frozen.
ROW_SCALED_OPTIONS = {
    "random_hadamard_transform": False,
    "row_scaled_activation": True,
    "backward": "dequantized",
}


def _relative_error(value: torch.Tensor, reference: torch.Tensor) -> float:
    return float((value.double() - reference.double()).norm() / reference.double().norm())


class _SecondLayerInMxfp8:
    """Precision map of the two-layer model below: ``layers.0`` in NVFP4, ``layers.1`` in MXFP8."""

    def resolve(self, module_names: Sequence[str], gemm_module_names: Collection[str]) -> TEPrecisionPlan:
        modules = tuple(
            TEModulePrecision(
                name=name,
                module_class="linear",
                layer=None,
                precision="mxfp8" if name == "layers.1" else "nvfp4",
                scope=name,
            )
            for name in sorted(gemm_module_names)
        )
        return TEPrecisionPlan(modules=modules, num_layers=0, high_precision_tail_layers=0)


@requires_te_nvfp4
def test_modules_the_precision_map_gives_mxfp8_run_the_mxfp8_recipe_inside_the_nvfp4_autocast():
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch.quantization import autocast

    def build() -> nn.ModuleDict:
        # Transformer Engine modules keep the quantizers of their first forward, so every recipe gets its own copy.
        torch.manual_seed(0)
        layers = OrderedDict(
            (str(index), te.Linear(512, 512, bias=False, params_dtype=torch.bfloat16)) for index in range(2)
        )
        return nn.ModuleDict({"layers": nn.ModuleDict(layers)}).to("cuda")

    model, mxfp8_model = build(), build()
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    config = TENvfp4Config(precision_map=_SecondLayerInMxfp8())
    config.prepare_modules(model)
    with torch.no_grad():
        with config.maybe_te_autocast():
            outputs = [model.layers[index](x) for index in ("0", "1")]
        with autocast(enabled=True, recipe=MXFP8BlockScaling()):
            mxfp8_outputs = [mxfp8_model.layers[index](x) for index in ("0", "1")]
        exact = [x.float() @ model.layers[index].weight.float().T for index in ("0", "1")]
    assert torch.equal(outputs[1], mxfp8_outputs[1])
    # The other module ran NVFP4: a different result with a larger error than MXFP8.
    assert not torch.equal(outputs[0], mxfp8_outputs[0])
    assert _relative_error(outputs[0], exact[0]) > 2 * _relative_error(mxfp8_outputs[0], exact[0])


@requires_te_nvfp4
def test_large_default_gemms_have_no_wrong_rows():
    """Default NVFP4 GEMMs at 4096 rows x 4608 x 4608, forward and backward: every row within rounding error.

    This is the size at which a row-scaled forward GEMM returns wrong rows with some cuBLASLt versions (next
    test). The default recipe multiplies into a BF16 output and must be correct with every version.
    """
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    linear = te.Linear(4608, 4608, bias=False, params_dtype=torch.bfloat16).to("cuda")
    x = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output_grad = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16)
    with TENvfp4Config().maybe_te_autocast():
        output = linear(x)
        output.backward(output_grad)
    weight, inputs, gradient = linear.weight.detach().float(), x.detach().float(), output_grad.float()
    results = {
        "output": (output.detach().float(), inputs @ weight.T),
        "input gradient": (x.grad.float(), gradient @ weight),
        "weight gradient": (linear.weight.grad.float(), gradient.T @ inputs),
    }
    for name, (value, exact) in results.items():
        row_error = (value - exact).norm(dim=1) / exact.norm(dim=1)
        # NVFP4 rounding gives every row an error of 0.15 to 0.19; a row from a defective GEMM is off by about 1.
        assert 0.05 < float(row_error.median()) and float(row_error.max()) < 0.3, (name, row_error.max())


@requires_te_nvfp4
def test_large_row_scaled_gemm_is_right_wherever_the_config_does_not_warn():
    """Row-scaled NVFP4 at 4096 rows x 4608 x 4608: every row is right where the config does not warn.

    Where it warns, wrong rows are expected but not required: cuBLASLt 13.7.0 on GB300 returned 96 wrong rows
    of 4096 here and 13.8.1 none; a release between the two was not measured.
    """
    import transformer_engine.pytorch as te

    from nemo_automodel.components.models.common.utils import row_scaled_nvfp4_gemm_is_unreliable
    from nemo_automodel.shared.import_utils import get_te_cublaslt_version

    torch.manual_seed(0)
    linear = te.Linear(4608, 4608, bias=False, params_dtype=torch.bfloat16).to("cuda")
    x = torch.randn(4096, 4608, device="cuda", dtype=torch.bfloat16)
    config = TENvfp4Config(**ROW_SCALED_OPTIONS)
    with torch.no_grad(), config.maybe_te_autocast():
        output = linear(x).float()
    exact = x.float() @ linear.weight.float().T
    # NVFP4 rounding gives every row an error near 0.13; a row from the defective GEMM is off by about 1.
    row_error = (output - exact).norm(dim=1) / exact.norm(dim=1)
    wrong_rows = int((row_error > 0.5).sum())
    assert float(row_error.median()) < 0.2
    cublaslt_version = get_te_cublaslt_version()
    if not row_scaled_nvfp4_gemm_is_unreliable(cublaslt_version, torch.cuda.get_device_capability()):
        assert wrong_rows == 0, f"{wrong_rows} wrong rows of 4096 with cuBLASLt {cublaslt_version}"


@requires_te_nvfp4
def test_linear_with_operands_too_large_for_nvfp4_is_rejected(monkeypatch):
    # The NVFP4 quantization kernels wrap around past 2**32 elements; the limit is lowered to make a small
    # Linear exceed it.
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling

    from nemo_automodel.components.models.common import utils

    utils._patch_te_modules()
    monkeypatch.setattr(utils, "NVFP4_QUANTIZE_MAX_ELEMENTS", 64 * 256 - 1)
    linear = te.Linear(128, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
    values = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)

    with te.autocast(enabled=True, recipe=TENvfp4Config().build_recipe()):
        with pytest.raises(ValueError, match=r"Linear\(128, 256\) was called with 64 rows .* holds 16384 elements"):
            linear(values)
        # 32 rows: the output gradient, the larger operand, holds 8192 elements.
        assert linear(values[:2]).shape == (2, 16, 256)
    # MXFP8 quantization and unquantized GEMMs have no limit.
    with te.autocast(enabled=True, recipe=MXFP8BlockScaling()):
        assert linear(values).shape == (4, 16, 256)
    assert linear(values).shape == (4, 16, 256)


@requires_te_nvfp4
@pytest.mark.parametrize("module_class", ["LayerNormLinear", "LayerNormMLP"])
def test_layernorm_modules_with_operands_too_large_for_nvfp4_are_rejected(monkeypatch, module_class):
    # Transformer Engine's fused norm + GEMM modules quantize the same operands as Linear. The limit is lowered
    # to make small modules exceed it: 64 rows of the 256-wide tensor are 16384 elements.
    import transformer_engine.pytorch as te

    from nemo_automodel.components.models.common import utils

    utils._patch_te_modules()
    monkeypatch.setattr(utils, "NVFP4_QUANTIZE_MAX_ELEMENTS", 64 * 256 - 1)
    if module_class == "LayerNormLinear":
        module = te.LayerNormLinear(128, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
        out_features = 256
    else:
        module = te.LayerNormMLP(128, 256, bias=False, params_dtype=torch.bfloat16, device="cuda")
        out_features = 128
    values = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)

    with te.autocast(enabled=True, recipe=TENvfp4Config().build_recipe()):
        message = rf"{module_class}\(128, 256\) was called with 64 rows .* holds 16384 elements"
        with pytest.raises(ValueError, match=message):
            module(values)
    # Unquantized GEMMs have no limit, and the module still runs through the check.
    assert module(values).shape == (4, 16, out_features)
    monkeypatch.setattr(utils, "NVFP4_QUANTIZE_MAX_ELEMENTS", 64 * 256)
    with te.autocast(enabled=True, recipe=TENvfp4Config().build_recipe()):
        output = module(values)
    assert output.shape == (4, 16, out_features) and torch.isfinite(output).all()

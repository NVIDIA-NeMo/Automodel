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

"""Transformer Engine experts on packed NVFP4 weights: the checkpoint bytes are the GEMM operand."""

import pytest
import torch

te = pytest.importorskip("transformer_engine.pytorch")
pytest.importorskip("transformer_engine.common.recipe", reason="needs Transformer Engine recipes")

from nemo_automodel.components.models.common.utils import BackendConfig  # noqa: E402
from nemo_automodel.components.moe.config import MoEConfig  # noqa: E402
from nemo_automodel.components.moe.experts import GroupedExpertsTE  # noqa: E402
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsTENVFP4  # noqa: E402
from nemo_automodel.components.quantization.nvfp4 import dequantize_nvfp4, quantize_nvfp4  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 GEMMs need a Blackwell GPU",
)

EXPERTS, DIM, INTER = 4, 64, 32
# Transformer Engine pads block scales to 128 rows: (64, 32) exercises the padded layout, (128, 64) the exact one.
SHAPES = [(64, 32), (128, 64)]
NVFP4_OPTIONS = {"random_hadamard_transform": False, "stochastic_rounding": False, "weight_blocks_2d": False}


def _backend(recipe: str | None = "nvfp4") -> BackendConfig:
    te_fp8 = None if recipe is None else {"recipe": recipe, **({"nvfp4": NVFP4_OPTIONS} if recipe == "nvfp4" else {})}
    return BackendConfig(
        attn="sdpa",
        linear="te" if recipe is not None else "torch",
        rms_norm="torch",
        experts="te",
        dispatcher="deepep",
        te_fp8=te_fp8,
    )


def _config(dim: int = DIM, inter: int = INTER) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=EXPERTS,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=dim,
        inter_dim=inter,
        moe_inter_dim=inter,
        norm_topk_prob=False,
        dtype=torch.bfloat16,
    )


def _packed_state(uniform_rows: bool = True, dim: int = DIM, inter: int = INTER) -> dict[str, torch.Tensor]:
    """Packed weights of both projections, quantized with one outer scale per expert."""
    generator = torch.Generator().manual_seed(7)
    state = {}
    for name, out_dim, in_dim in (("gate_and_up_projs", 2 * inter, dim), ("down_projs", dim, inter)):
        weight = torch.randn(EXPERTS, out_dim, in_dim, generator=generator)
        outer = weight.abs().amax(dim=(1, 2)) / (6.0 * 448.0)
        row_scale = outer.unsqueeze(-1).expand(EXPERTS, out_dim).clone()
        if not uniform_rows:
            row_scale[0, 0] *= 2
        packed, scales, row_scale = quantize_nvfp4(weight, global_scale=row_scale)
        state[name + "_nvfp4"], state[name + "_scales"], state[name + "_global"] = packed, scales, row_scale
    return state


def _resident(state: dict[str, torch.Tensor], dim: int = DIM, inter: int = INTER) -> GroupedExpertsTENVFP4:
    original = GroupedExpertsTE(_config(dim, inter), backend=_backend(), dispatcher_backend="deepep")
    experts = GroupedExpertsTENVFP4(original)
    experts.to_empty(device="cuda")
    experts.load_state_dict({key: value.cuda() for key, value in state.items()}, strict=False)
    return experts


@pytest.mark.parametrize(("dim", "inter"), SHAPES)
def test_weights_hold_the_loaded_packed_parameters(dim: int, inter: int) -> None:
    state = _packed_state(dim=dim, inter=inter)
    experts = _resident(state, dim, inter)
    experts._build_te_weights()

    for name, linear in (("gate_and_up_projs", experts.gate_up_linear), ("down_projs", experts.down_linear)):
        packed = getattr(experts, name + "_nvfp4")
        scales = getattr(experts, name + "_scales")
        assert torch.equal(packed.cpu(), state[name + "_nvfp4"])
        for index in range(EXPERTS):
            weight = getattr(linear, f"weight{index}")
            assert not weight.requires_grad
            assert weight._rowwise_data.data_ptr() == packed[index].data_ptr()
            rows, blocks = scales[index].shape
            stored = weight._rowwise_scale_inv
            assert torch.equal(stored[:rows, :blocks], scales[index].view(torch.uint8))
            # Unpadded block scales are the packed parameter itself.
            assert (stored.data_ptr() == scales[index].data_ptr()) == (stored.shape == scales[index].shape)
            assert weight._columnwise_data is not None
            # Transformer Engine reads the aliased storage as the numbers the packed layout encodes.
            expected = dequantize_nvfp4(
                state[name + "_nvfp4"][index],
                state[name + "_scales"][index],
                state[name + "_global"][index],
                torch.float32,
            )
            torch.testing.assert_close(weight.dequantize(dtype=torch.float32).cpu(), expected, rtol=1e-6, atol=0)
    exported = experts.state_dict()
    assert set(exported) == set(state)
    assert exported["down_projs_nvfp4"].data_ptr() == experts.down_projs_nvfp4.data_ptr()


def test_projection_multiplies_each_expert_by_its_own_packed_weight() -> None:
    state = _packed_state()
    experts = _resident(state)
    experts._build_te_weights()
    rows = 128
    x = torch.randn(EXPERTS * rows, DIM, generator=torch.Generator().manual_seed(3)).to(
        device="cuda", dtype=torch.bfloat16
    )
    splits = [rows] * EXPERTS
    assert experts.te_fp8 is not None

    with torch.no_grad(), te.autocast(enabled=True, recipe=experts.te_fp8.build_recipe()):
        out = experts.gate_up_linear(x, splits)

    weights = dequantize_nvfp4(
        state["gate_and_up_projs_nvfp4"],
        state["gate_and_up_projs_scales"],
        state["gate_and_up_projs_global"],
        torch.float32,
    ).cuda()
    for index in range(EXPERTS):
        block = slice(index * rows, (index + 1) * rows)
        # Unquantized activations: the reference differs from the FP4 GEMM by the activation quantization only.
        reference = x[block].float() @ weights[index].T
        error = (out[block].float() - reference).norm() / reference.norm()
        assert error < 0.2, (index, float(error))
        other = x[block].float() @ weights[(index + 1) % EXPERTS].T
        assert (out[block].float() - other).norm() / other.norm() > 0.8


def test_backward_reaches_the_activations() -> None:
    experts = _resident(_packed_state())
    experts._build_te_weights()
    x = torch.randn(EXPERTS * 128, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert experts.te_fp8 is not None

    with te.autocast(enabled=True, recipe=experts.te_fp8.build_recipe()):
        out = experts.gate_up_linear(x, [128] * EXPERTS)
    out.float().sum().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(not param.requires_grad for param in experts.parameters())


def test_several_row_scales_in_one_expert_are_rejected() -> None:
    experts = _resident(_packed_state(uniform_rows=False))

    with pytest.raises(NotImplementedError, match="several row scales"):
        experts._build_te_weights()


def test_another_recipe_is_rejected() -> None:
    original = GroupedExpertsTE(_config(), backend=_backend("mxfp8"), dispatcher_backend="deepep")

    with pytest.raises(NotImplementedError, match="te_fp8.recipe='nvfp4'"):
        GroupedExpertsTENVFP4(original)

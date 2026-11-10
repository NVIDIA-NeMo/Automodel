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

"""Transformer Engine experts with NVFP4 weights: loaded from packed tensors as they are, and trainable."""

import pytest
import torch

te = pytest.importorskip("transformer_engine.pytorch")

from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: E402

from nemo_automodel.components.models.common.utils import BackendConfig  # noqa: E402
from nemo_automodel.components.moe.config import MoEConfig  # noqa: E402
from nemo_automodel.components.moe.experts import GroupedExpertsTE  # noqa: E402
from nemo_automodel.components.moe.layers import MoE  # noqa: E402
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsTENVFP4  # noqa: E402
from nemo_automodel.components.quantization.nvfp4 import dequantize_nvfp4, quantize_nvfp4  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    reason="NVFP4 GEMMs need a Blackwell GPU",
)

EXPERTS, DIM, INTER, ROWS = 4, 64, 32, 128
# Transformer Engine pads block scales to 128 rows: (64, 32) exercises the padded layout, (128, 64) the exact one.
SHAPES = [(64, 32), (128, 64)]


def _backend(recipe: str = "nvfp4", resident: bool = True, row_scaled_forward: bool = False) -> BackendConfig:
    te_fp8: dict[str, str | dict[str, bool | str]] = {"recipe": recipe}
    if recipe == "nvfp4":
        te_fp8["nvfp4"] = {
            "random_hadamard_transform": False,
            "stochastic_rounding": False,
            "weight_blocks_2d": False,
            "resident_weights": resident,
        }
        if row_scaled_forward:
            # Row-scaled four-over-six activations; the weight gradient then needs a floating-point backward.
            te_fp8["nvfp4"].update(row_scaled_activation=True, four_over_six="activations", backward="high_precision")
    return BackendConfig(attn="sdpa", linear="te", rms_norm="torch", experts="te", dispatcher="deepep", te_fp8=te_fp8)


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


def _experts(dim: int = DIM, inter: int = INTER, row_scaled_forward: bool = False) -> GroupedExpertsTENVFP4:
    backend = _backend(row_scaled_forward=row_scaled_forward)
    experts = GroupedExpertsTENVFP4(_config(dim, inter), backend=backend, dispatcher_backend="deepep")
    experts.to_empty(device="cuda")
    return experts


def _loaded(
    state: dict[str, torch.Tensor], dim: int = DIM, inter: int = INTER, row_scaled_forward: bool = False
) -> GroupedExpertsTENVFP4:
    experts = _experts(dim, inter, row_scaled_forward)
    experts.load_state_dict({key: value.cuda() for key, value in state.items()}, strict=False)
    return experts


def _gate_up(experts: GroupedExpertsTENVFP4, x: torch.Tensor) -> torch.Tensor:
    assert experts.te_fp8 is not None
    with te.autocast(enabled=True, recipe=experts.te_fp8.build_recipe()):
        return experts.gate_up_linear(x, [ROWS] * EXPERTS)


def test_backend_option_selects_the_class(monkeypatch: pytest.MonkeyPatch) -> None:
    # MoE falls back to the single-process experts unless the job has several ranks.
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 2)

    assert isinstance(MoE(_config(), _backend()).experts, GroupedExpertsTENVFP4)
    assert type(MoE(_config(), _backend(resident=False)).experts) is GroupedExpertsTE


@pytest.mark.parametrize(("dim", "inter"), SHAPES)
def test_loaded_weights_hold_the_packed_tensors(dim: int, inter: int) -> None:
    state = _packed_state(dim=dim, inter=inter)
    experts = _loaded(state, dim, inter)

    for name, linear in (("gate_and_up_projs", experts.gate_up_linear), ("down_projs", experts.down_linear)):
        packed = getattr(experts, name + "_nvfp4")
        scales = getattr(experts, name + "_scales")
        assert torch.equal(packed.cpu(), state[name + "_nvfp4"])
        for index in range(EXPERTS):
            weight = getattr(linear, f"weight{index}")
            assert weight.requires_grad
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
    # Nothing was trained: the export is the loaded state, bit for bit.
    for key, value in state.items():
        assert torch.equal(exported[key].cpu().view(torch.uint8), value.view(torch.uint8)), key


def test_projection_multiplies_each_expert_by_its_own_packed_weight() -> None:
    state = _packed_state()
    experts = _loaded(state)
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(EXPERTS * ROWS, DIM, generator=generator).to(device="cuda", dtype=torch.bfloat16)

    with torch.no_grad():
        out = _gate_up(experts, x)

    weights = dequantize_nvfp4(
        state["gate_and_up_projs_nvfp4"],
        state["gate_and_up_projs_scales"],
        state["gate_and_up_projs_global"],
        torch.float32,
    ).cuda()
    for index in range(EXPERTS):
        block = slice(index * ROWS, (index + 1) * ROWS)
        # Unquantized activations: the reference differs from the FP4 GEMM by the activation quantization only.
        reference = x[block].float() @ weights[index].T
        assert (out[block].float() - reference).norm() / reference.norm() < 0.2
        other = x[block].float() @ weights[(index + 1) % EXPERTS].T
        assert (out[block].float() - other).norm() / other.norm() > 0.8


@pytest.mark.parametrize("row_scaled_forward", [False, True])
def test_training_step_updates_the_packed_weights_through_fp32_masters(row_scaled_forward: bool) -> None:
    state = _packed_state()
    experts = _loaded(state, row_scaled_forward=row_scaled_forward)
    trainable = [param for param in experts.parameters() if param.requires_grad]
    assert len(trainable) == 2 * EXPERTS
    optimizer = FusedAdam(trainable, lr=1e-2, master_weights=True)
    x = torch.randn(EXPERTS * ROWS, DIM, device="cuda", dtype=torch.bfloat16)

    _gate_up(experts, x).float().pow(2).mean().backward()
    weight = experts.gate_up_linear.weight0
    assert weight.grad is not None and torch.isfinite(weight.grad).all() and weight.grad.abs().sum() > 0
    optimizer.step()

    exported = {key: value.cpu() for key, value in experts.state_dict().items()}
    before = dequantize_nvfp4(
        state["gate_and_up_projs_nvfp4"][0],
        state["gate_and_up_projs_scales"][0],
        state["gate_and_up_projs_global"][0],
        torch.float32,
    )
    after = dequantize_nvfp4(
        exported["gate_and_up_projs_nvfp4"][0],
        exported["gate_and_up_projs_scales"][0],
        exported["gate_and_up_projs_global"][0],
        torch.float32,
    )
    master = optimizer.state[weight]["master_param"].float().cpu()
    # The FP32 master moved by the learning rate; the exported packed weight is that master, re-quantized.
    assert 1e-3 < (master - before).abs().max() < 2e-2
    assert not torch.equal(exported["gate_and_up_projs_nvfp4"][0], state["gate_and_up_projs_nvfp4"][0])
    assert (after - master).norm() / master.norm() < 0.15
    assert (after - master).norm() < (before - master).norm()
    torch.testing.assert_close(weight.dequantize(dtype=torch.float32).cpu(), after, rtol=1e-6, atol=0)
    # The down projection received no gradient and is exported unchanged.
    for suffix in ("_nvfp4", "_scales", "_global"):
        key = "down_projs" + suffix
        assert torch.equal(exported[key].view(torch.uint8), state[key].view(torch.uint8))


def test_freezing_covers_the_weights_and_leaves_the_packed_storage() -> None:
    experts = _loaded(_packed_state())

    experts.requires_grad_(False)

    assert not any(param.requires_grad for param in experts.parameters())
    x = torch.randn(EXPERTS * ROWS, DIM, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    _gate_up(experts, x).float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_several_row_scales_in_one_expert_are_rejected() -> None:
    experts = _experts()
    state = {key: value.cuda() for key, value in _packed_state(uniform_rows=False).items()}

    with pytest.raises(NotImplementedError, match="several row scales"):
        experts.load_state_dict(state, strict=False)


def test_another_recipe_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="te_fp8.recipe='nvfp4'"):
        GroupedExpertsTENVFP4(_config(), backend=_backend("mxfp8"), dispatcher_backend="deepep")

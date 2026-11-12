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

"""The fused Adam step of resident NVFP4 expert weights returns the bits of the Transformer Engine FusedAdam step."""

import shutil
from types import SimpleNamespace

import pytest
import torch

te = pytest.importorskip("transformer_engine.pytorch")

from transformer_engine.pytorch.optimizers import FusedAdam  # noqa: E402

from nemo_automodel.components.models.common.utils import BackendConfig  # noqa: E402
from nemo_automodel.components.moe import nvfp4_resident_adam  # noqa: E402
from nemo_automodel.components.moe.config import MoEConfig  # noqa: E402
from nemo_automodel.components.moe.nvfp4_resident_adam import enable_nvfp4_fused_adam_step  # noqa: E402
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsTENVFP4  # noqa: E402
from nemo_automodel.components.quantization.nvfp4 import quantize_nvfp4  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10 or shutil.which("nvcc") is None,
    reason="NVFP4 weights need a Blackwell GPU, the kernels the CUDA compiler",
)

# Weights [224, 256] and [256, 112]: thread blocks that cover a partial 128x128 tile, and block scales that
# Transformer Engine pads (224 rows to 256, 7 scale columns to 8).
EXPERTS, DIM, INTER = 2, 256, 112
STATE = ("master_param", "exp_avg", "exp_avg_sq")
OPERANDS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
    "_amax_rowwise",
    "_amax_columnwise",
)


def _packed_state() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(11)
    state = {}
    for name, out_dim, in_dim in (("gate_and_up_projs", 2 * INTER, DIM), ("down_projs", DIM, INTER)):
        weight = torch.randn(EXPERTS, out_dim, in_dim, generator=generator) * 0.02
        row_scale = (weight.abs().amax(dim=(1, 2)) / (6.0 * 448.0)).unsqueeze(-1).expand(EXPERTS, out_dim).clone()
        packed, scales, row_scale = quantize_nvfp4(weight, global_scale=row_scale)
        state[name + "_nvfp4"], state[name + "_scales"], state[name + "_global"] = packed, scales, row_scale
    return state


def _experts(state: dict[str, torch.Tensor], weight_blocks_2d: bool = True) -> GroupedExpertsTENVFP4:
    backend = BackendConfig(
        attn="sdpa",
        linear="te",
        rms_norm="torch",
        experts="te",
        dispatcher="deepep",
        te_fp8={"recipe": "nvfp4", "nvfp4": {"weight_blocks_2d": weight_blocks_2d, "resident_weights": True}},
    )
    config = MoEConfig(
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
        dim=DIM,
        inter_dim=INTER,
        moe_inter_dim=INTER,
        norm_topk_prob=False,
        dtype=torch.bfloat16,
    )
    experts = GroupedExpertsTENVFP4(config, backend=backend, dispatcher_backend="deepep")
    experts.to_empty(device="cuda")
    experts.load_state_dict({key: value.cuda() for key, value in state.items()}, strict=False)
    return experts


def _weights(experts: GroupedExpertsTENVFP4) -> list[torch.nn.Parameter]:
    return [parameter for parameter in experts.parameters() if parameter.requires_grad]


def _set_gradients(
    reference: list[torch.nn.Parameter], fused: list[torch.nn.Parameter], seed: int, scale: float = 1e-3
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for weight_ref, weight_fused in zip(reference, fused, strict=True):
        gradient = (torch.randn(weight_ref.shape, device="cuda", generator=generator) * scale).bfloat16()
        weight_ref.grad = gradient
        weight_fused.grad = gradient.clone()


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int32) if tensor.dtype == torch.float32 else tensor.view(torch.uint8)


def _assert_identical(
    optimizer_ref: FusedAdam,
    reference: list[torch.nn.Parameter],
    optimizer_fused: FusedAdam,
    fused: list[torch.nn.Parameter],
) -> None:
    for index, (weight_ref, weight_fused) in enumerate(zip(reference, fused, strict=True)):
        for name in STATE:
            expected, actual = optimizer_ref.state[weight_ref][name], optimizer_fused.state[weight_fused][name]
            assert torch.equal(_bits(actual), _bits(expected)), f"{name} of weight {index}"
        rows, cols = weight_ref.shape
        # Block scales are padded to 128 rows and 4 columns; nothing writes the padding.
        valid = {"_rowwise_scale_inv": (rows, cols // 16), "_columnwise_scale_inv": (cols, rows // 16)}
        for name in OPERANDS:
            expected, actual = getattr(weight_ref, name), getattr(weight_fused, name)
            if name in valid:
                expected, actual = (
                    expected[: valid[name][0], : valid[name][1]],
                    actual[: valid[name][0], : valid[name][1]],
                )
            assert torch.equal(_bits(actual), _bits(expected)), f"{name} of weight {index}"


def _count_kernel_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the number of weights of every kernel call; the real extension still runs."""
    calls: list[int] = []
    load = nvfp4_resident_adam._load_extension

    def load_counting() -> SimpleNamespace:
        extension = load()

        def adam_requantize(gradients: list[torch.Tensor], *rest: list[torch.Tensor] | float) -> None:
            calls.append(len(gradients))
            extension.adam_requantize(gradients, *rest)

        return SimpleNamespace(adam_requantize=adam_requantize)

    monkeypatch.setattr(nvfp4_resident_adam, "_load_extension", load_counting)
    return calls


@pytest.mark.parametrize(
    ("adam_w_mode", "weight_decay", "bias_correction"),
    [(True, 0.01, True), (True, 0.0, True), (False, 0.01, True), (True, 0.1, False)],
)
def test_fused_step_returns_the_bits_of_the_fused_adam_step(
    monkeypatch: pytest.MonkeyPatch, adam_w_mode: bool, weight_decay: float, bias_correction: bool
) -> None:
    calls = _count_kernel_calls(monkeypatch)
    state = _packed_state()
    experts_ref, experts_fused = _experts(state), _experts(state)
    reference, fused = _weights(experts_ref), _weights(experts_fused)
    assert len(fused) == 2 * EXPERTS
    options = {"lr": 1e-3, "weight_decay": weight_decay, "adam_w_mode": adam_w_mode, "master_weights": True}
    optimizer_ref = FusedAdam(reference, bias_correction=bias_correction, **options)
    optimizer_fused = FusedAdam(fused, bias_correction=bias_correction, **options)
    enable_nvfp4_fused_adam_step([experts_fused], [optimizer_fused])

    for step in range(4):
        _set_gradients(reference, fused, seed=step)
        optimizer_ref.step()
        optimizer_fused.step()
        _assert_identical(optimizer_ref, reference, optimizer_fused, fused)
        assert all(weight.grad is not None for weight in fused)
    # A weight has no optimizer state at its first step and takes the FusedAdam step; the kernels run the rest.
    assert calls == [2 * EXPERTS] * 3
    # The weights moved, and the packed parameters export them.
    exported = experts_fused.state_dict()
    assert not torch.equal(exported["gate_and_up_projs_nvfp4"].cpu(), state["gate_and_up_projs_nvfp4"])
    assert torch.equal(exported["gate_and_up_projs_nvfp4"], experts_ref.state_dict()["gate_and_up_projs_nvfp4"])


def test_requantization_rounds_ties_to_the_even_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_kernel_calls(monkeypatch)
    state = _packed_state()
    experts_ref, experts_fused = _experts(state), _experts(state)
    reference, fused = _weights(experts_ref), _weights(experts_fused)
    optimizer_ref = FusedAdam(reference, lr=1e-3, weight_decay=0.0, master_weights=True)
    optimizer_fused = FusedAdam(fused, lr=1e-3, weight_decay=0.0, master_weights=True)
    enable_nvfp4_fused_adam_step([experts_fused], [optimizer_fused])

    # With a tensor maximum of 2688 / 4096 the outer encode scale is 2^12, and with a block maximum of 6 / 16 the
    # block scale decodes to 2^8: every element of the block is multiplied by exactly 16 before rounding.
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device="cuda") / 16.0
    for weights, optimizer in ((reference, optimizer_ref), (fused, optimizer_fused)):
        for weight in weights:
            weight.grad = torch.zeros(weight.shape, device="cuda", dtype=torch.bfloat16)
        optimizer.step()  # creates the optimizer state; zero gradients leave the masters where they are
        for weight in weights:
            master = optimizer.state[weight]["master_param"]
            master.zero_()
            master[0, 0] = 6.0 / 16.0
            master[0, 1:8] = ties
            master[1, 1:8] = -ties
            master[-1, -1] = 2688.0 / 4096.0
        optimizer.step()

    assert calls == [2 * EXPERTS]
    _assert_identical(optimizer_ref, reference, optimizer_fused, fused)
    for weight in fused:
        row_bytes = weight._rowwise_data.view(torch.uint8)[:2, :4].flatten(1)
        codes = torch.stack([row_bytes & 0xF, row_bytes >> 4], dim=-1).flatten(1).cpu()
        # 6 is code 7; ties go to the neighbour with an even code: 0.25 -> 0, 0.75 and 1.25 -> 1.0 (code 2),
        # 1.75 and 2.5 -> 2.0 (code 4), 3.5 and 5.0 -> 4.0 (code 6). Bit 3 is the sign.
        assert codes[0].tolist() == [7, 0, 2, 2, 4, 4, 6, 6]
        assert codes[1].tolist() == [0, 8, 10, 10, 12, 12, 14, 14]
        assert weight._rowwise_scale_inv.view(torch.uint8)[0, 0].item() == 0x78  # E4M3 byte of 2^8


def test_step_with_a_closure_takes_the_fused_adam_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_kernel_calls(monkeypatch)
    state = _packed_state()
    experts_ref, experts_fused = _experts(state), _experts(state)
    reference, fused = _weights(experts_ref), _weights(experts_fused)
    optimizer_ref = FusedAdam(reference, lr=1e-3, master_weights=True)
    optimizer_fused = FusedAdam(fused, lr=1e-3, master_weights=True)
    enable_nvfp4_fused_adam_step([experts_fused], [optimizer_fused])

    for step in range(2):
        _set_gradients(reference, fused, seed=step)
        optimizer_ref.step()
        optimizer_fused.step(lambda: torch.zeros((), device="cuda"))
    assert not calls
    _assert_identical(optimizer_ref, reference, optimizer_fused, fused)


def test_weights_without_2d_blocks_keep_the_fused_adam_step(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _count_kernel_calls(monkeypatch)
    state = _packed_state()
    experts_ref, experts_fused = _experts(state, weight_blocks_2d=False), _experts(state, weight_blocks_2d=False)
    reference, fused = _weights(experts_ref), _weights(experts_fused)
    optimizer_ref = FusedAdam(reference, lr=1e-3, master_weights=True)
    optimizer_fused = FusedAdam(fused, lr=1e-3, master_weights=True)
    enable_nvfp4_fused_adam_step([experts_fused], [optimizer_fused])

    for step in range(2):
        _set_gradients(reference, fused, seed=step)
        optimizer_ref.step()
        optimizer_fused.step()
    assert not calls
    _assert_identical(optimizer_ref, reference, optimizer_fused, fused)


def test_one_optimizer_per_model_part_is_required() -> None:
    experts = _experts(_packed_state())
    with pytest.raises(ValueError, match="one optimizer per model part"):
        enable_nvfp4_fused_adam_step([experts], [])

# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import Mock, patch

import pytest
import torch

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.training.utils import move_to_device, ScopedModuleOffloading
from nemo_automodel.components.training.utils import clip_grad_norm, count_tail_padding, scale_grads_and_clip_grad_norm


def test_docstring_example():
    labels = torch.tensor(
        [
            [-100, 1, 1, -100, -100],  # 2 tail -100s
            [-100, -100, 2, 3, 4],  # 0 tail -100s
            [5, 6, -100, -100, -100],  # 3 tail -100s
        ]
    )
    assert count_tail_padding(labels) == 5


@pytest.mark.parametrize(
    "labels, expected",
    [
        # No padding at all
        (torch.tensor([[1, 2, 3], [4, 5, 6]]), 0),
        # Entire sequence is padding
        (torch.full((2, 4), -100), 8),
        # Different ignore label
        (torch.tensor([[9, 0, 0], [0, 0, 0]]), 5),
    ],
)
def test_various_cases(labels, expected):
    """
    Covers:
    1. no ignore_label present
    2. every position is ignore_label
    3. custom ignore_label value (0)
    """
    ignore_label = 0 if (labels == 0).any() else -100
    assert count_tail_padding(labels, ignore_label=ignore_label) == expected


def test_random_shapes():
    """
    Generate random examples and compare with a simple-but-slow reference
    implementation to guard against shape / broadcasting regressions.
    """
    torch.manual_seed(0)
    for _ in range(10):
        batch = torch.randint(
            1,
            8,
            size=(
                torch.randint(1, 5, ()).item(),  # batch size
                torch.randint(1, 10, ()).item(),
            ),
        )  # seq len
        # randomly sprinkle ignore tokens
        mask = torch.rand_like(batch.float()) < 0.3
        batch[mask] = -100

        # brute-force reference
        ref = 0
        for row in batch:
            idx = (row != -100).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                ref += row.numel()
            else:
                ref += (row[idx[-1] + 1 :] == -100).sum().item()

        assert count_tail_padding(batch) == ref


def test_clip_grad_norm_with_pp_and_tp():
    """Test that clip_grad_norm works with PP and TP enabled (no longer skips)."""
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    device_mesh = Mock()
    device_mesh.mesh_dim_names = ["pp", "tp"]
    device_mesh.__getitem__ = Mock(side_effect=lambda key: Mock(size=Mock(return_value=2)))

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=True,
        device_mesh=device_mesh,
        pp_axis_name="pp",
    )

    # Should now clip (not skip) with the new sharding-aware implementation
    assert grad_norm > 0


def test_clip_grad_norm_works_without_pp():
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
    )

    assert grad_norm > 0


def test_clip_grad_norm_returns_zero_when_max_grad_norm_is_none():
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)

    grad_norm = clip_grad_norm(
        max_grad_norm=None,
        model_parts=[model],
        pp_enabled=False,
    )

    assert grad_norm == 0


def test_clip_grad_norm_with_multiple_models():
    """Test that clip_grad_norm works with multiple model parts."""
    model1 = torch.nn.Linear(10, 10)
    model2 = torch.nn.Linear(20, 20)

    model1.weight.grad = torch.randn_like(model1.weight)
    model1.bias.grad = torch.randn_like(model1.bias)
    model2.weight.grad = torch.randn_like(model2.weight)
    model2.bias.grad = torch.randn_like(model2.bias)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model1, model2],
        pp_enabled=False,
    )

    assert grad_norm > 0
    # Verify gradients were actually clipped
    assert model1.weight.grad.norm().item() <= 1.0 + 1e-5
    assert model2.weight.grad.norm().item() <= 1.0 + 1e-5


def test_clip_grad_norm_actually_clips():
    """Test that gradients are actually clipped to max_norm."""
    model = torch.nn.Linear(10, 10)
    # Set large gradients
    model.weight.grad = torch.ones_like(model.weight) * 10.0
    model.bias.grad = torch.ones_like(model.bias) * 10.0

    initial_norm = torch.nn.utils.clip_grad_norm_(
        [model.weight, model.bias], float("inf")
    ).item()

    # Reset gradients
    model.weight.grad = torch.ones_like(model.weight) * 10.0
    model.bias.grad = torch.ones_like(model.bias) * 10.0

    max_norm = 1.0
    grad_norm = clip_grad_norm(
        max_grad_norm=max_norm,
        model_parts=[model],
        pp_enabled=False,
    )

    assert grad_norm > 0
    # The reported norm should be the original (unclipped) norm
    assert abs(grad_norm - initial_norm) < 1e-3

    # Verify the actual gradients are clipped
    clipped_norm = torch.sqrt(model.weight.grad.pow(2).sum() + model.bias.grad.pow(2).sum()).item()
    assert abs(clipped_norm - max_norm) < 1e-3


def test_clip_grad_norm_with_inf_norm():
    """Test clip_grad_norm with infinity norm."""
    model = torch.nn.Linear(10, 10)
    model.weight.grad = torch.randn_like(model.weight)
    model.bias.grad = torch.randn_like(model.bias)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        norm_type=float("inf"),
        pp_enabled=False,
    )

    assert grad_norm > 0


def test_clip_grad_norm_with_empty_gradients():
    """Test that clip_grad_norm handles parameters without gradients."""
    model = torch.nn.Linear(10, 10)
    # Only set gradient for weight, not bias
    model.weight.grad = torch.randn_like(model.weight)

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
    )

    # Should work even with some None gradients
    assert grad_norm > 0


def test_clip_grad_norm_with_all_none_gradients():
    """Test that clip_grad_norm handles all None gradients gracefully."""
    model = torch.nn.Linear(10, 10)
    # Don't set any gradients

    grad_norm = clip_grad_norm(
        max_grad_norm=1.0,
        model_parts=[model],
        pp_enabled=False,
    )

    # Should return 0 when no gradients exist
    assert grad_norm == 0.0


def test_clip_grad_norm_different_norm_types():
    """Test clip_grad_norm with different norm types (L1, L2, Linf)."""
    model = torch.nn.Linear(10, 10)

    for norm_type in [1.0, 2.0, float("inf")]:
        # Reset gradients
        model.weight.grad = torch.randn_like(model.weight)
        model.bias.grad = torch.randn_like(model.bias)

        grad_norm = clip_grad_norm(
            max_grad_norm=1.0,
            model_parts=[model],
            norm_type=norm_type,
            pp_enabled=False,
        )

        assert grad_norm >= 0, f"Norm type {norm_type} failed"


class _TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2, bias=False)
        self.register_buffer("scale", torch.ones(1))


def _all_tensors_on_device(module: nn.Module, device_type: str) -> bool:
    for p in module.parameters():
        if p.device.type != device_type:
            return False
    for b in module.buffers():
        if b.device.type != device_type:
            return False
    return True


def test_move_to_device_cpu():
    model = _TinyModule()
    # Ensure starts on CPU
    assert _all_tensors_on_device(model, "cpu")

    # Move to CPU (idempotent)
    move_to_device(model, "cpu")
    assert _all_tensors_on_device(model, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_move_to_device_cuda():
    model = _TinyModule()
    # Move to CUDA
    move_to_device(model, "cuda")
    assert _all_tensors_on_device(model, "cuda")

    # Move back to CPU to leave environment clean
    move_to_device(model, "cpu")
    assert _all_tensors_on_device(model, "cpu")


def test_scoped_offloading_disabled_noop_and_reraises():
    model = _TinyModule()
    assert _all_tensors_on_device(model, "cpu")

    with pytest.raises(ValueError):
        with ScopedModuleOffloading(model, enabled=False):
            # Should not move devices and should re-raise exceptions
            assert _all_tensors_on_device(model, "cpu")
            raise ValueError("boom")

    # After context, still on CPU
    assert _all_tensors_on_device(model, "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_scoped_offloading_enabled_moves_and_reraises():
    model = _TinyModule()
    assert _all_tensors_on_device(model, "cpu")

    # Enter moves to CUDA, exit moves back to CPU and re-raises exceptions
    with pytest.raises(RuntimeError):
        with ScopedModuleOffloading(model, enabled=True):
            assert _all_tensors_on_device(model, "cuda")
            raise RuntimeError("fail inside context")

    assert _all_tensors_on_device(model, "cpu")


class _TEGroupedLinearMock(nn.Module):
    """Mock TE GroupedLinear with weight0 parameter naming."""

    def __init__(self):
        super().__init__()
        # Parameter name matches TE GroupedLinear pattern (weight0, weight1, etc.)
        self.weight0 = nn.Parameter(torch.randn(4, 2))


class _ExpertsModule(nn.Module):
    """Mock experts module with parameters matching GroupedExpertsTE pattern."""

    def __init__(self):
        super().__init__()
        # Submodule names match GroupedExpertsTE: gate_up_linear, down_linear
        self.gate_up_linear = _TEGroupedLinearMock()


class _MoEModule(nn.Module):
    """Mock MoE module with expert parameters for testing EP scaling."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        # FQN will be mlp.experts.gate_up_linear.weight0, matching _EXPERT_PARAM_PATTERN
        self.mlp = nn.ModuleDict({"experts": _ExpertsModule()})


class TestScaleGradsAndClipGradNorm:
    """Tests for scale_grads_and_clip_grad_norm with EP scaling."""

    def test_ep_scaling_for_expert_params_by_name(self):
        """Test that expert params are scaled by EP ratio based on param name."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 2.0
        expert_param.grad = torch.ones_like(expert_param) * 2.0

        # Mock moe_mesh to trigger EP scaling
        moe_mesh = Mock()
        moe_mesh.mesh_dim_names = ["ep_shard"]
        moe_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,  # Disable clipping to test scaling only
            model_parts=[model],
            pp_enabled=False,
            moe_mesh=moe_mesh,
            dp_group_size=4,
        )

        # ep_ratio = dp_group_size / ep_shard_size = 4 / 2 = 2
        # Expert params (mlp.experts.gate_up_linear.weight0) should be scaled by 1/2
        # Non-expert params (gate) should NOT be scaled
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 2.0)
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 1.0)

    def test_no_ep_scaling_without_moe_mesh(self):
        """Test that no EP scaling occurs when moe_mesh is None."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 2.0
        expert_param.grad = torch.ones_like(expert_param) * 2.0

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            pp_enabled=False,
            moe_mesh=None,
            dp_group_size=4,
        )

        # No scaling should occur
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 2.0)
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 2.0)

    def test_ep_scaling_combined_with_pp_scaling(self):
        """Test that PP and EP scaling work together correctly."""
        model = _MoEModule()
        expert_param = model.mlp["experts"].gate_up_linear.weight0
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 8.0
        expert_param.grad = torch.ones_like(expert_param) * 8.0

        # Mock device_mesh for PP
        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["pp"]
        device_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        # Mock moe_mesh for EP
        moe_mesh = Mock()
        moe_mesh.mesh_dim_names = ["ep_shard"]
        moe_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            pp_enabled=True,
            device_mesh=device_mesh,
            pp_axis_name="pp",
            moe_mesh=moe_mesh,
            dp_group_size=4,
            num_label_tokens=8,  # pp_divisor = num_label_tokens / dp_group_size = 8 / 4 = 2
        )

        # PP scaling: all grads divided by (num_label_tokens / dp_group_size) = 8/4 = 2 -> 8 / 2 = 4
        # EP scaling: expert grads divided by ep_ratio (4/2=2) -> 4 / 2 = 2
        # Non-expert params: only PP scaling -> 4
        assert torch.allclose(model.gate.weight.grad, torch.ones_like(model.gate.weight) * 4.0)
        assert torch.allclose(expert_param.grad, torch.ones_like(expert_param) * 2.0)

    def test_pp_scaling_with_zero_num_label_tokens(self):
        """Test that num_label_tokens=0 does not cause division by zero."""
        model = _MoEModule()
        model.gate.weight.grad = torch.ones_like(model.gate.weight) * 4.0

        device_mesh = Mock()
        device_mesh.mesh_dim_names = ["pp"]
        device_mesh.__getitem__ = Mock(return_value=Mock(size=Mock(return_value=2)))

        # Should not raise ZeroDivisionError — max(0, 1) == 1
        scale_grads_and_clip_grad_norm(
            max_grad_norm=None,
            model_parts=[model],
            pp_enabled=True,
            device_mesh=device_mesh,
            pp_axis_name="pp",
            dp_group_size=4,
            num_label_tokens=0,
        )

        # pp_divisor = max(0, 1) / 4 = 0.25 → grads divided by 0.25 → 4 / 0.25 = 16
        expected = torch.ones_like(model.gate.weight) * 16.0
        assert torch.allclose(model.gate.weight.grad, expected)

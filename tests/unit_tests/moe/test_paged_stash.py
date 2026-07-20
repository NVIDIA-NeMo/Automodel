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

from types import SimpleNamespace

import pytest
import torch

import nemo_automodel.components.moe.paged_stash as paged_stash
from nemo_automodel.components.moe.paged_stash_ops import HAVE_TRITON


class _SaveForBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor * 2

    @staticmethod
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        return grad_output + tensor * 0


class _SegmentedSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        return tensor.square()

    @staticmethod
    def backward(ctx, grad_output):
        (tensor,) = ctx.saved_tensors
        return 2 * tensor * grad_output


def _record_group(manager, tensor, live_mask, *, name="test"):
    group = manager.group(name=name, max_num_tokens=tensor.shape[0], live_token_mask=live_mask)
    tensor = group.start(tensor)
    with group:
        marked = group.mark_activation(tensor)
        output = _SaveForBackward.apply(marked.view_as(marked))
    return group.commit(output)


def test_recording_profiles_page_rounded_sparse_activations():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=4, buffer_size_factor=1.2)

    first = torch.arange(32.0).reshape(8, 4).detach().requires_grad_()
    second_input = _record_group(
        manager, first, torch.tensor([True, False, True, True, False, True, False, True]), name="first"
    )
    second = _record_group(
        manager, second_input, torch.tensor([True, False, False, True, False, False, True, False]), name="second"
    )
    second.sum().backward()

    diagnostics = manager.diagnostics()
    assert diagnostics["recorded_peak_tokens"] == {(torch.float32, 4): 12}
    assert diagnostics["live_groups"] == 0
    with pytest.raises(RuntimeError, match="CUDA warmup"):
        manager.prepare()
    manager.close()


def test_group_fails_closed_when_marked_activation_is_not_saved():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    tensor = torch.ones(8, 4, requires_grad=True)
    group = manager.group(name="missing", max_num_tokens=8, live_token_mask=torch.ones(8, dtype=torch.bool))

    tensor_for_compute = group.start(tensor)
    with group:
        group.mark_activation(tensor_for_compute)
        output = tensor_for_compute * 2

    with pytest.raises(RuntimeError, match="observed no registered"):
        group.commit(output)
    manager.close()


def test_activation_surface_matches_only_row_aligned_storage_aliases():
    tensor = torch.arange(32.0).reshape(8, 4)
    live_mask = torch.tensor([True, False, True, True, False, True, False, True])
    surface = paged_stash._ActivationSurface(
        device=tensor.device,
        dtype=tensor.dtype,
        storage_ptr=tensor.untyped_storage().data_ptr(),
        element_start=tensor.storage_offset(),
        numel=tensor.numel(),
        num_rows=tensor.shape[0],
        row_width=tensor.shape[1],
        live_token_mask=live_mask,
    )

    expert_view = tensor.narrow(0, 2, 3)
    matched_rows, matched_mask = surface.match(expert_view)
    assert matched_rows == 3
    torch.testing.assert_close(matched_mask, live_mask[2:5])

    assert surface.match(expert_view.clone()) is None
    assert surface.match(tensor.flatten()[1:13]) is None
    assert surface.match(tensor[:, :2]) is None
    assert surface.match(tensor.view(4, 8)) is None


def test_disabled_context_bypasses_hooks():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    tensor = torch.ones(8, 4, requires_grad=True)

    with manager.disabled():
        output = _record_group(manager, tensor, torch.ones(8, dtype=torch.bool))
        output.sum().backward()

    assert manager.diagnostics()["recorded_peak_tokens"] == {}
    manager.close()


def test_groups_fail_closed_when_nested():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    outer = manager.group(name="outer", max_num_tokens=8, live_token_mask=torch.ones(8, dtype=torch.bool))

    with pytest.raises(RuntimeError, match="cannot nest"):
        with outer:
            manager.group(name="inner", max_num_tokens=8, live_token_mask=torch.ones(8, dtype=torch.bool))

    with pytest.raises(RuntimeError, match="recording was invalid"):
        manager.prepare()
    manager.close()


@pytest.mark.parametrize(
    "live_mask",
    [torch.ones(7, dtype=torch.bool), torch.ones(8), torch.ones(2, 4, dtype=torch.bool)],
)
def test_recording_rejects_invalid_live_mask(live_mask):
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)

    with pytest.raises((TypeError, ValueError), match="one-dimensional|boolean dtype|expected max_num_tokens"):
        manager.group(name="invalid", max_num_tokens=8, live_token_mask=live_mask)

    manager.close()


def test_configuration_is_idempotent_and_can_restart_recording():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=32, buffer_size_factor=1.25)
    manager.configure(enabled=True, page_size=32, buffer_size_factor=1.25)
    with pytest.raises(RuntimeError, match="same page_size"):
        manager.configure(enabled=True, page_size=64, buffer_size_factor=1.25)

    manager.restart_recording()
    diagnostics = manager.diagnostics()
    assert diagnostics["state"] == "recording"
    assert diagnostics["page_size"] == 32
    assert diagnostics["buffer_size_factor"] == 1.25
    manager.close()


@pytest.mark.parametrize("invalid_factor", [0, 0.5, float("nan"), float("inf")])
def test_configuration_rejects_buffer_factors_that_cannot_hold_warmup(invalid_factor):
    manager = paged_stash.PagedStashManager()

    with pytest.raises(ValueError, match="finite and at least 1.0"):
        manager.configure(enabled=True, buffer_size_factor=invalid_factor)


def test_device_overflow_api_does_not_require_host_read():
    manager = paged_stash.PagedStashManager()
    assert manager.check_overflow() is None
    manager._overflow = torch.ones(1, dtype=torch.int64)
    assert manager.check_overflow() is manager._overflow
    with pytest.raises(paged_stash.PagedStashOverflowError):
        manager.finish_iteration()
    manager._overflow = None


def test_reload_rejects_non_lifo_backward_order():
    manager = paged_stash.PagedStashManager()
    manager._backward_group_stack.extend([3, 4])

    with pytest.raises(RuntimeError, match="expected group 4, got 3"):
        manager._reload_group(3)


def test_pre_boundary_releases_reload_storage_after_expert_consumer_backward():
    events = []

    class _ExpertConsumer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, tensor):
            return tensor.square()

        @staticmethod
        def backward(ctx, grad_output):
            events.append("expert-backward")
            return grad_output

    manager = SimpleNamespace(
        is_active=True,
        _finish_group_backward=lambda group_id: events.append(("release", group_id)),
    )
    input_ = torch.randn(4, requires_grad=True)
    bounded = paged_stash._PagedStashPreBoundary.apply(input_, manager, 7)

    _ExpertConsumer.apply(bounded).sum().backward()

    assert events == ["expert-backward", ("release", 7)]


def test_group_rejects_frozen_expert_input():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    group = manager.group(name="expert", max_num_tokens=4, live_token_mask=torch.ones(4, dtype=torch.bool))

    with pytest.raises(RuntimeError, match="expert input to require gradients"):
        group.start(torch.randn(4, 8))

    assert manager._current_group is None
    assert manager._group_tensors == {}
    with pytest.raises(RuntimeError, match="recording was invalid"):
        manager.prepare()
    manager.close()


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA and Triton are required")
def test_cuda_stash_restores_sparse_live_rows_and_zeros_padding():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=2, buffer_size_factor=1.5)

    warmup = torch.randn(8, 4, device="cuda", requires_grad=True)
    warmup_mask = torch.tensor([True, False, True, True, False, True, False, True], device="cuda")
    warmup_output = _record_group(manager, warmup, warmup_mask, name="warmup")
    warmup_output.sum().backward()
    manager.prepare()

    tensor = torch.randn(8, 4, device="cuda", requires_grad=True)
    live_mask = torch.tensor([True, False, True, False, False, True, False, False], device="cuda")
    group = manager.group(name="active", max_num_tokens=8, live_token_mask=live_mask)
    tensor_for_compute = group.start(tensor)
    with group:
        output = _SegmentedSquare.apply(group.mark_activation(tensor_for_compute))
    output = group.commit(output)
    output.sum().backward()

    expected = torch.zeros_like(tensor)
    expected[live_mask] = 2 * tensor.detach()[live_mask]
    torch.testing.assert_close(tensor.grad, expected)
    assert manager.check_overflow() is not None
    assert manager.check_overflow().item() == 0
    manager.finish_iteration()
    manager.close()

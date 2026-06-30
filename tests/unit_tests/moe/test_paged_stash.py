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
from unittest.mock import MagicMock

import pytest
import torch

import nemo_automodel.components.moe.paged_stash as paged_stash
from nemo_automodel.components.moe.experts import GroupedExpertsTE, GroupedExpertsTeOps
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


class _PrefixSquare(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, num_tokens):
        ctx.save_for_backward(tensor, num_tokens)
        return tensor.square()

    @staticmethod
    def backward(ctx, grad_output):
        tensor, num_tokens = ctx.saved_tensors
        row_mask = torch.arange(tensor.shape[0], device=tensor.device) < num_tokens
        row_mask = row_mask.reshape(-1, *([1] * (tensor.dim() - 1)))
        grad = torch.where(row_mask, 2 * tensor * grad_output, 0)
        return grad, None


def _record_group(manager, tensor, num_tokens, *, max_num_tokens, name="test"):
    tensor.grouped_tensor_scale_inv = False
    group = manager.group(
        name=name,
        max_num_tokens=max_num_tokens,
        num_tokens_tensor=num_tokens,
    )
    with group:
        output = _SaveForBackward.apply(tensor)
    return group.commit(output)


def test_recording_profiles_peak_page_charges_and_releases_after_backward():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=4, buffer_size_factor=1.2)

    first = torch.arange(32.0).reshape(8, 4).detach().requires_grad_()
    second_input = _record_group(manager, first, torch.tensor(5), max_num_tokens=8, name="first")
    second_input.grouped_tensor_scale_inv = False
    second = _record_group(manager, second_input, torch.tensor(3), max_num_tokens=8, name="second")
    second.sum().backward()

    diagnostics = manager.diagnostics()
    # The two simultaneously live tensors use ceil(5/4) + ceil(3/4) pages.
    assert diagnostics["recorded_peak_tokens"] == {(torch.float32, 4): 12}
    assert diagnostics["live_groups"] == 0
    with pytest.raises(RuntimeError, match="CUDA warmup"):
        manager.prepare()
    manager.close()


def test_recording_understands_te_columnwise_scale_inverse_rows():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    tensor = torch.arange(6.0).reshape(2, 3).detach().requires_grad_()
    tensor.grouped_tensor_scale_inv = True
    group = manager.group(
        name="columnwise_scale",
        max_num_tokens=64,
        num_tokens_tensor=torch.tensor(32),
    )
    with group:
        output = _SaveForBackward.apply(tensor)
    output = group.commit(output)
    output.sum().backward()

    # One live scale row still owns one complete page in the active allocator.
    assert manager.diagnostics()["recorded_peak_tokens"] == {(torch.float32, 3): 64}
    manager.close()


def test_disabled_context_bypasses_unvalidated_nested_hook_regions():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    tensor = torch.ones(8, 4, requires_grad=True)

    with manager.disabled():
        output = _record_group(manager, tensor, torch.tensor(5), max_num_tokens=8)
        output.sum().backward()

    assert manager.diagnostics()["recorded_peak_tokens"] == {}
    manager.close()


def test_groups_fail_closed_when_nested():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)
    outer = manager.group(name="outer", max_num_tokens=8, num_tokens_tensor=torch.tensor(4))

    with pytest.raises(RuntimeError, match="cannot nest"):
        with outer:
            manager.group(name="inner", max_num_tokens=8, num_tokens_tensor=torch.tensor(4))

    with pytest.raises(RuntimeError, match="recording was invalid"):
        manager.prepare()
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


def test_device_overflow_api_does_not_require_host_read():
    manager = paged_stash.PagedStashManager()
    assert manager.check_overflow() is None
    manager._overflow = torch.ones(1, dtype=torch.int64)
    assert manager.check_overflow() is manager._overflow
    with pytest.raises(paged_stash.PagedStashOverflowError):
        manager.finish_iteration()
    manager._overflow = None


def test_te_ops_reads_finalized_backend_paged_stash_names(monkeypatch):
    manager = MagicMock()
    monkeypatch.setattr(paged_stash, "get_paged_stash_manager", lambda: manager)

    def fake_te_init(self, *args, **kwargs):
        self._te_ops_uses_padded_capacity = True

    monkeypatch.setattr(GroupedExpertsTE, "__init__", fake_te_init)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=None,
        moe_paged_stash=True,
        moe_paged_stash_page_size=128,
        moe_paged_stash_buffer_size_factor_cuda=1.3,
        moe_paged_stash_buffer_size_factor_cpu=0.0,
        partial_cuda_graph_experts=False,
    )

    experts = GroupedExpertsTeOps(MagicMock(), backend=backend)

    assert experts.moe_paged_stash is True
    manager.configure.assert_called_once_with(enabled=True, page_size=128, buffer_size_factor=1.3)


def test_te_ops_paged_stash_requires_full_mxfp8_fusion(monkeypatch):
    monkeypatch.setattr(GroupedExpertsTE, "__init__", lambda self, *args, **kwargs: None)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=SimpleNamespace(recipe="mxfp8"),
        moe_paged_stash=True,
        moe_paged_stash_page_size=64,
        moe_paged_stash_buffer_size_factor_cuda=1.1,
        moe_paged_stash_buffer_size_factor_cpu=0.0,
        partial_cuda_graph_experts=False,
    )

    with pytest.raises(RuntimeError, match="requires the TE CuTe DSL fused MXFP8"):
        GroupedExpertsTeOps(MagicMock(), backend=backend)


def test_recording_rejects_misaligned_mxfp8_expert_splits():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True)

    with pytest.raises(RuntimeError, match="every expert split"):
        manager.group(
            name="misaligned",
            max_num_tokens=256,
            num_tokens_tensor=torch.tensor(256),
            tokens_per_expert=torch.tensor([128, 127, 1]),
        )

    manager.close()


def test_te_ops_paged_stash_fails_closed_on_unimplemented_host_spill(monkeypatch):
    monkeypatch.setattr(GroupedExpertsTE, "__init__", lambda self, *args, **kwargs: None)
    backend = SimpleNamespace(
        experts="te_ops",
        te_fp8=None,
        moe_paged_stash=True,
        moe_paged_stash_buffer_size_factor_cpu=0.5,
        partial_cuda_graph_experts=False,
    )

    with pytest.raises(ValueError, match="does not yet support pinned-host spill"):
        GroupedExpertsTeOps(MagicMock(), backend=backend)


@pytest.mark.skipif(not torch.cuda.is_available() or not HAVE_TRITON, reason="CUDA and Triton are required")
def test_cuda_stash_restores_dynamic_live_prefix_for_backward():
    manager = paged_stash.PagedStashManager()
    manager.configure(enabled=True, page_size=2, buffer_size_factor=1.5)

    warmup = torch.randn(8, 4, device="cuda", requires_grad=True)
    warmup_tokens = torch.tensor(5, device="cuda", dtype=torch.int64)
    warmup.grouped_tensor_scale_inv = False
    group = manager.group(name="warmup", max_num_tokens=8, num_tokens_tensor=warmup_tokens)
    with group:
        warmup_output = _PrefixSquare.apply(warmup, warmup_tokens)
    warmup_output = group.commit(warmup_output)
    warmup_output.sum().backward()
    manager.prepare()

    tensor = torch.randn(8, 4, device="cuda", requires_grad=True)
    num_tokens = torch.tensor(4, device="cuda", dtype=torch.int64)
    tensor.grouped_tensor_scale_inv = False
    group = manager.group(name="active", max_num_tokens=8, num_tokens_tensor=num_tokens)
    with group:
        output = _PrefixSquare.apply(tensor, num_tokens)
    output = group.commit(output)
    output.sum().backward()

    expected = torch.cat((2 * tensor.detach()[:4], torch.zeros_like(tensor.detach()[4:])))
    torch.testing.assert_close(tensor.grad, expected)
    assert manager.check_overflow() is not None
    assert manager.check_overflow().item() == 0
    manager.finish_iteration()
    manager.close()

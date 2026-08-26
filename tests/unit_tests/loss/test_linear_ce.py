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
import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.linear_ce import (
    HAVE_CUT_CROSS_ENTROPY,
    FusedLinearCrossEntropy,
)


class _FakeMesh:
    ndim = 1

    @staticmethod
    def size():
        return 2


class _FakeDTensor:
    requires_grad = True
    device_mesh = _FakeMesh()

    def __init__(self):
        self.grad_placements = None
        self.full = torch.ones(4, requires_grad=True)

    def full_tensor(self, *, grad_placements=None):
        self.grad_placements = grad_placements
        return self.full


def test_flce_materialized_weight_reduces_partial_gradient_into_shard(monkeypatch):
    from torch.distributed.tensor import Partial

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    weight = _FakeDTensor()
    raw_grads = []

    def capture_raw_grad(grad):
        raw_grads.append(grad)
        return grad

    weight.full.register_hook(capture_raw_grad)

    full = FusedLinearCrossEntropy.materialize_lm_weight(
        weight,
        grad_reduce_group=object(),
    )
    full.square().sum().backward()

    assert len(weight.grad_placements) == 1
    assert isinstance(weight.grad_placements[0], Partial)
    # The normalization hook must return a new tensor instead of modifying the
    # gradient object received by earlier hooks.
    assert torch.equal(raw_grads[0], torch.full_like(weight.full, 2.0))
    # Raw grad is 2; divide by the two-rank reduction world size restores 1.
    assert torch.equal(weight.full.grad, torch.ones_like(weight.full))


def test_flce_materialized_weight_rejects_mismatched_reduction_group(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)

    with pytest.raises(ValueError, match="mesh size=2, reduction group size=4"):
        FusedLinearCrossEntropy.materialize_lm_weight(
            _FakeDTensor(),
            grad_reduce_group=object(),
        )


@pytest.mark.skipif(not HAVE_CUT_CROSS_ENTROPY, reason="Linear loss CE is not installed")
def test_fused_cross_entropy():
    """Tests FusedLinearCrossEntropy against PyTorch's CE.

    * has close output with PyTorch's cross_entropy
    * uses less memory than PyTorch's cross_entropy
    """
    if not torch.cuda.is_available():
        pytest.skip("This test requires a GPU")

    device = torch.device("cuda")
    batch_size = 8
    seq_length = 2048  # Added sequence length dimension
    hidden_dim = 4096
    vocab_size = 128256
    dtype = torch.bfloat16
    # Create inputs on GPU
    hidden_states = torch.randn(batch_size, seq_length, hidden_dim, dtype=dtype, device=device)
    weight = torch.randn(vocab_size, hidden_dim, dtype=dtype, device=device)  # Note: transposed shape
    targets = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)

    # Measure memory for PyTorch implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        # Reshape for matmul: [batch_size, seq_length, hidden_dim] -> [batch_size * seq_length, hidden_dim]
        hidden_states_reshaped = hidden_states.reshape(-1, hidden_dim)
        logits = torch.matmul(hidden_states_reshaped, weight.t())  # Use transpose for matmul
        # Reshape targets for loss: [batch_size, seq_length] -> [batch_size * seq_length]
        targets_reshaped = targets.reshape(-1)
        pytorch_loss = F.cross_entropy(logits, targets_reshaped, reduction="sum")
    pytorch_memory = torch.cuda.max_memory_allocated()

    torch.cuda.empty_cache()  # Clear CUDA cache
    import gc

    gc.collect()

    # Measure memory for fused implementation
    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        fused_loss = FusedLinearCrossEntropy()(hidden_states, targets, weight)
    fused_memory = torch.cuda.max_memory_allocated()

    # Compare results and memory usage
    print("\nMemory usage comparison:")
    print(f"PyTorch implementation: {pytorch_memory / 1024**2:.2f} MB")
    print(f"Fused implementation: {fused_memory / 1024**2:.2f} MB")
    print(f"Memory savings: {(pytorch_memory - fused_memory) / 1024**2:.2f} MB")

    # Convert both losses to float32 for comparison
    pytorch_loss = pytorch_loss.float()
    fused_loss = fused_loss.float()

    # Check if the losses are close
    assert torch.allclose(fused_loss, pytorch_loss, rtol=1e-2, atol=1e-2), (
        f"Loss mismatch: PyTorch={pytorch_loss.item()}, Fused={fused_loss.item()}"
    )
    # Check if the fused implementation uses less memory
    assert fused_memory < pytorch_memory, "Fused implementation should use less memory than PyTorch implementation"


def test_fused_cross_entropy_raises_when_dependency_missing(monkeypatch):
    """Ensure that FusedLinearCrossEntropy raises ImportError if the optional
    cut_cross_entropy package is not available (HAVE_CUT_CROSS_ENTROPY=False).

    This exercises the guard clause on line ~150 of linear_ce.py.
    """

    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    # Temporarily pretend the optional dependency is missing
    monkeypatch.setattr(linear_ce_mod, "HAVE_CUT_CROSS_ENTROPY", False)

    loss_fn = linear_ce_mod.FusedLinearCrossEntropy()

    # Dummy tensors - they will not be used because we expect an early ImportError
    hidden = torch.randn(1, 2, 3)
    labels = torch.zeros(1, 2, dtype=torch.long)
    weight = torch.randn(4, 3)

    with pytest.raises(ImportError) as exc_info:
        loss_fn(hidden, labels, weight)

    # The error message should point users to the missing package
    from nemo_automodel.shared.import_utils import MISSING_CUT_CROSS_ENTROPY_MSG

    assert MISSING_CUT_CROSS_ENTROPY_MSG in str(exc_info.value)


def test_is_triton_greater_or_equal(monkeypatch):
    """Unit test for new_is_triton_greater_or_equal helper (lines 89-99).

    We monkeypatch importlib.metadata.version to control the installed
    version string and assert the comparison logic works as intended.
    """

    from importlib.metadata import PackageNotFoundError

    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    def _metadata_version(versions):
        def _version(package_name):
            try:
                return versions[package_name]
            except KeyError:
                raise PackageNotFoundError(package_name)

        return _version

    # Case 1: installed version is higher ⇒ function returns True
    monkeypatch.setattr(linear_ce_mod, "metadata_version", _metadata_version({"pytorch-triton": "3.5.0"}))
    assert linear_ce_mod.new_is_triton_greater_or_equal("3.1.0") is True

    # Case 2: installed version is lower ⇒ returns False
    monkeypatch.setattr(linear_ce_mod, "metadata_version", _metadata_version({"pytorch-triton": "2.9.0"}))
    assert linear_ce_mod.new_is_triton_greater_or_equal("3.1.0") is False

    # Case 3: pytorch-triton package missing, but triton is installed ⇒ use triton
    monkeypatch.setattr(linear_ce_mod, "metadata_version", _metadata_version({"triton": "3.5.0"}))
    assert linear_ce_mod.new_is_triton_greater_or_equal("3.1.0") is True

    # Case 4: package not installed ⇒ PackageNotFoundError ⇒ returns False
    def _raise_package_not_found(package_name):
        raise PackageNotFoundError(package_name)

    monkeypatch.setattr(linear_ce_mod, "metadata_version", _raise_package_not_found)
    assert linear_ce_mod.new_is_triton_greater_or_equal("3.1.0") is False


def test_is_triton_greater_or_equal_3_2_0(monkeypatch):
    """Ensure the convenience wrapper compares against 3.1.0 (despite name)."""

    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    monkeypatch.setattr(linear_ce_mod, "metadata_version", lambda _: "3.5.0")
    assert linear_ce_mod.new_is_triton_greater_or_equal_3_2_0() is True

    monkeypatch.setattr(linear_ce_mod, "metadata_version", lambda _: "3.0.0")
    assert linear_ce_mod.new_is_triton_greater_or_equal_3_2_0() is False


def test_fused_cross_entropy_normalizes_by_num_tokens(monkeypatch):
    """When num_label_tokens is passed and reduction='sum', the returned loss
    should be divided by that value. We monkeypatch the external dependency to
    avoid requiring the real cut_cross_entropy implementation.
    """

    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    # Pretend the optional package is present
    monkeypatch.setattr(linear_ce_mod, "HAVE_CUT_CROSS_ENTROPY", True)

    # Replace linear_cross_entropy with a deterministic stub that returns a scalar tensor
    def _fake_linear_ce(hidden, weight, targets=None, **kwargs):  # noqa: D401,E501 - signature match not required
        return torch.tensor(20.0)

    monkeypatch.setattr(linear_ce_mod, "linear_cross_entropy", _fake_linear_ce, raising=False)

    loss_fn = linear_ce_mod.FusedLinearCrossEntropy(reduction="sum")

    # Dummy tensors - shapes are irrelevant for the stub
    hidden = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    weight = torch.randn(5, 4)

    out = loss_fn(hidden, labels, weight, num_label_tokens=10)

    # The stub returns 20, so after division by 10 we expect 2.0
    assert torch.is_tensor(out)
    assert out.item() == pytest.approx(2.0)


def test_fused_cross_entropy_uses_autocast_compute_dtype_with_fp32_parameters(monkeypatch):
    """FP32 resident parameters reach CCE as BF16 under BF16 autocast."""
    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    monkeypatch.setattr(linear_ce_mod, "HAVE_CUT_CROSS_ENTROPY", True)
    seen_dtypes = None

    def _fake_linear_ce(hidden, weight, **_kwargs):
        nonlocal seen_dtypes
        seen_dtypes = (hidden.dtype, weight.dtype)
        return hidden.sum() + weight.sum()

    monkeypatch.setattr(linear_ce_mod, "linear_cross_entropy", _fake_linear_ce, raising=False)
    hidden = torch.randn(2, 3, 4, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(5, 4, dtype=torch.float32, requires_grad=True)
    labels = torch.zeros(2, 3, dtype=torch.long)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = FusedLinearCrossEntropy()(hidden, labels, weight)
    loss.backward()

    assert seen_dtypes == (torch.bfloat16, torch.bfloat16)
    assert hidden.grad is not None and hidden.grad.dtype == torch.float32
    assert weight.grad is not None and weight.grad.dtype == torch.float32


def test_fused_cross_entropy_cp_example_mean_matches_reference_gradient(monkeypatch):
    """Summed CP token shards match full-sequence unequal-length example means."""
    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    monkeypatch.setattr(linear_ce_mod, "HAVE_CUT_CROSS_ENTROPY", True)

    def _pytorch_linear_ce(hidden, weight, targets=None, ignore_index=-100, reduction="sum", **kwargs):
        logits = hidden @ weight.t()
        flat_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=ignore_index,
            reduction=reduction,
        )
        return flat_loss.reshape_as(targets) if reduction == "none" else flat_loss

    monkeypatch.setattr(linear_ce_mod, "linear_cross_entropy", _pytorch_linear_ce, raising=False)
    torch.manual_seed(23)
    labels = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [2, 3, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100, -100],
        ]
    )
    token_counts = torch.tensor([6, 2, 0])
    actual_hidden = torch.randn(3, 6, 4, requires_grad=True)
    actual_weight = torch.randn(7, 4, requires_grad=True)
    reference_hidden = actual_hidden.detach().clone().requires_grad_(True)
    reference_weight = actual_weight.detach().clone().requires_grad_(True)

    loss_fn = FusedLinearCrossEntropy(reduction="sum")
    actual = sum(
        loss_fn(
            actual_hidden[:, shard],
            labels[:, shard],
            actual_weight,
            label_token_counts=token_counts,
            num_label_examples=2,
        )
        for shard in (slice(None, None, 2), slice(1, None, 2))
    )
    reference_logits = reference_hidden @ reference_weight.t()
    per_token = F.cross_entropy(
        reference_logits.reshape(-1, 7), labels.reshape(-1), ignore_index=-100, reduction="none"
    ).reshape_as(labels)
    expected = (per_token.sum(dim=1) / token_counts.clamp_min(1)).sum() / 2

    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(actual_weight.grad, reference_weight.grad)


def test_fused_cross_entropy_requires_complete_example_normalization(monkeypatch):
    """Per-example normalization metadata must be supplied as a complete pair."""
    from nemo_automodel.components.loss import linear_ce as linear_ce_mod

    monkeypatch.setattr(linear_ce_mod, "HAVE_CUT_CROSS_ENTROPY", True)
    loss_fn = FusedLinearCrossEntropy(reduction="sum")

    with pytest.raises(ValueError, match="must be provided together"):
        loss_fn(
            torch.randn(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.long),
            torch.randn(5, 4),
            label_token_counts=torch.tensor([3, 3]),
        )

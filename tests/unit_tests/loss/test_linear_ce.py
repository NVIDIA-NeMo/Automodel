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

    We monkeypatch pkg_resources.get_distribution to control the installed
    version string and assert the comparison logic works as intended.
    """

    import pkg_resources
    from nemo_automodel.components.loss.linear_ce import new_is_triton_greater_or_equal

    class _DummyDist:
        def __init__(self, version):
            self.version = version

    # Case 1: installed version is higher ⇒ function returns True
    monkeypatch.setattr(pkg_resources, "get_distribution", lambda _: _DummyDist("3.5.0"))
    assert new_is_triton_greater_or_equal("3.1.0") is True

    # Case 2: installed version is lower ⇒ returns False
    monkeypatch.setattr(pkg_resources, "get_distribution", lambda _: _DummyDist("2.9.0"))
    assert new_is_triton_greater_or_equal("3.1.0") is False

    # Case 3: package not installed ⇒ DistributionNotFound ⇒ returns False
    def _raise_dist_not_found(_):
        raise pkg_resources.DistributionNotFound

    monkeypatch.setattr(pkg_resources, "get_distribution", _raise_dist_not_found)
    assert new_is_triton_greater_or_equal("3.1.0") is False


def test_is_triton_greater_or_equal_3_2_0(monkeypatch):
    """Ensure the convenience wrapper compares against 3.1.0 (despite name)."""

    import pkg_resources
    from nemo_automodel.components.loss.linear_ce import (
        new_is_triton_greater_or_equal_3_2_0,
    )

    class _DummyDist:
        def __init__(self, version):
            self.version = version

    monkeypatch.setattr(pkg_resources, "get_distribution", lambda _: _DummyDist("3.5.0"))
    assert new_is_triton_greater_or_equal_3_2_0() is True

    monkeypatch.setattr(pkg_resources, "get_distribution", lambda _: _DummyDist("3.0.0"))
    assert new_is_triton_greater_or_equal_3_2_0() is False

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

    monkeypatch.setattr(linear_ce_mod, "linear_cross_entropy", _fake_linear_ce)

    loss_fn = linear_ce_mod.FusedLinearCrossEntropy(reduction="sum")

    # Dummy tensors - shapes are irrelevant for the stub
    hidden = torch.randn(2, 3, 4)
    labels = torch.zeros(2, 3, dtype=torch.long)
    weight = torch.randn(5, 4)

    out = loss_fn(hidden, labels, weight, num_label_tokens=10)

    # The stub returns 20, so after division by 10 we expect 2.0
    assert torch.is_tensor(out)
    assert out.item() == pytest.approx(2.0)

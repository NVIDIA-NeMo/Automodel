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

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy, compute_cross_entropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy


def test_compute_cross_entropy_basic():
    """
    Tests compute_cross_entropy with a small set of logits and targets.

    Verifies results match PyTorch's built-in cross_entropy.
    """
    # Create sample logits and targets
    logits = torch.tensor([[2.0, 0.5, 0.3], [1.0, 2.0, 0.1]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)

    # Expected cross_entropy from PyTorch (sum reduction for direct comparison)
    expected_loss = F.cross_entropy(logits, targets, reduction="sum")

    # Compare function output
    actual_loss = compute_cross_entropy(logits, targets)
    assert torch.allclose(actual_loss, expected_loss, atol=1e-6), (
        f"Expected loss {expected_loss.item()}, but got {actual_loss.item()}."
    )


def test_compute_cross_entropy_ignore_index():
    """
    Tests compute_cross_entropy with ignore_index to ensure ignored targets

    don't contribute to the loss.
    """
    # Create sample logits and targets with ignore_index
    logits = torch.tensor([[0.0, 0.0], [2.0, 3.0], [1.0, 1.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1, -100], dtype=torch.long)  # -100 will be ignored

    # Compute expected loss with PyTorch
    expected_loss = F.cross_entropy(logits, targets, reduction="sum", ignore_index=-100)

    # Compare function output
    actual_loss = compute_cross_entropy(logits, targets, ignore_index=-100)
    assert torch.allclose(actual_loss, expected_loss, atol=1e-6), (
        f"Expected loss {expected_loss.item()}, but got {actual_loss.item()}."
    )


def test_chunked_cross_entropy_matches_compute_cross_entropy():
    """
    Tests that ChunkedCrossEntropy produces the same result as compute_cross_entropy

    when the entire sequence is processed in one chunk.
    """
    # Create random test data
    seq_len = 16
    num_classes = 8

    logits = torch.randn(seq_len, num_classes)
    targets = torch.randint(0, num_classes, (seq_len,))

    # Loss from normal compute_cross_entropy
    loss_ref = compute_cross_entropy(logits, targets).sum().detach()

    # Loss from ChunkedCrossEntropy when chunk_len = seq_len (effectively one chunk)
    chunk_len = seq_len  # so there's only one chunk
    loss_chunked = ChunkedCrossEntropy(chunk_len=chunk_len)(logits, targets)

    assert torch.allclose(loss_chunked, loss_ref, atol=1e-6), (
        f"Expected chunked loss {loss_ref.item()}, but got {loss_chunked.item()}."
    )


def test_chunked_cross_entropy_ignore_index_and_mask():
    """
    Tests that ChunkedCrossEntropy properly ignores indices and respects masks.

    Verifies consistency with compute_cross_entropy.
    """
    seq_len = 10
    num_classes = 5
    logits = torch.randn(seq_len, num_classes)
    targets = torch.randint(0, num_classes, (seq_len,))

    # Randomly zero out entries in a mask
    mask = torch.randint(0, 2, (seq_len,))  # 0 or 1
    ignore_idx = -100

    # First compute the reference loss by manually applying ignore_index
    masked_targets = targets.clone()
    masked_targets[mask == 0] = ignore_idx
    loss_ref = compute_cross_entropy(logits, masked_targets, ignore_index=ignore_idx).sum().detach()

    # Now compute ChunkedCrossEntropy with mask
    chunk_len = 3  # just an arbitrary small chunk size
    loss_chunked = ChunkedCrossEntropy(chunk_len=chunk_len, ignore_index=ignore_idx)(logits, targets, mask=mask)

    assert torch.allclose(loss_chunked, loss_ref, atol=1e-6), (
        f"Expected chunked loss {loss_ref.item()}, but got {loss_chunked.item()}."
    )


def test_chunked_cross_entropy_num_label_tokens_normalization():
    """Ensure that the loss is divided by ``num_label_tokens`` when provided."""

    seq_len = 12
    num_classes = 6

    logits = torch.randn(seq_len, num_classes)
    targets = torch.randint(0, num_classes, (seq_len,))

    # Compute the reference (sum reduction) loss first
    loss_sum = compute_cross_entropy(logits, targets).detach()

    # Pick an arbitrary num_label_tokens (could be less than seq_len due to masking in real cases)
    num_label_tokens = 9

    # Expected normalized loss
    expected_loss = loss_sum / num_label_tokens

    # Loss from ChunkedCrossEntropy with num_label_tokens specified
    loss_chunked = ChunkedCrossEntropy(chunk_len=4)(logits, targets, num_label_tokens=num_label_tokens).detach()

    assert torch.allclose(loss_chunked, expected_loss, atol=1e-6), (
        f"Expected normalized loss {expected_loss.item()}, but got {loss_chunked.item()}."
    )


# ---------------------------------------------------------------------------
# memory-efficient sum path (shared kernel with MaskedCrossEntropy(chunk_size=...))
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inplace_grad", [False, True])
def test_chunked_cross_entropy_gradient_parity_with_reference(inplace_grad):
    """The sum path must match ``F.cross_entropy`` in loss AND gradient, with mask + ignore_index."""
    torch.manual_seed(7)
    logits = torch.randn(17, 11, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    targets = torch.randint(0, logits.shape[-1], (17,))
    targets[3] = -100
    mask = torch.ones_like(targets)
    mask[8] = 0

    loss = ChunkedCrossEntropy(chunk_len=5, inplace_grad=inplace_grad)(logits, targets.clone(), mask=mask)
    reference_targets = targets.masked_fill(mask == 0, -100)
    reference = F.cross_entropy(reference_logits, reference_targets, ignore_index=-100, reduction="sum")

    torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-7)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-7)
    if inplace_grad:
        assert logits.grad.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()


def test_chunked_cross_entropy_matches_masked_cross_entropy_chunked_path():
    """ChunkedCrossEntropy and MaskedCrossEntropy(chunk_size=...) share one kernel and must agree."""
    torch.manual_seed(23)
    logits = torch.randn(3, 6, 13, requires_grad=True)
    masked_logits = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, logits.shape[-1], (3, 6))
    labels[1, 2] = -100

    loss_chunked = ChunkedCrossEntropy(chunk_len=4)(logits, labels.clone())
    loss_masked = MaskedCrossEntropy(chunk_size=4)(masked_logits, labels.clone())

    torch.testing.assert_close(loss_chunked, loss_masked, rtol=0.0, atol=0.0)
    loss_chunked.backward()
    loss_masked.backward()
    torch.testing.assert_close(logits.grad, masked_logits.grad, rtol=0.0, atol=0.0)


def test_chunked_cross_entropy_sum_path_saves_no_fp32_activations():
    """The sum path must save only the original-dtype logits for backward (no fp32 upcasts)."""
    torch.manual_seed(29)
    logits = torch.randn(16, 32, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, logits.shape[-1], (16,))

    saved_float_bytes = 0

    def pack(tensor):
        nonlocal saved_float_bytes
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            saved_float_bytes += tensor.numel() * tensor.element_size()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = ChunkedCrossEntropy(chunk_len=4)(logits, targets)

    assert saved_float_bytes == 0, f"sum path saved {saved_float_bytes} bytes of non-bf16 float activations"
    loss.backward()
    assert logits.grad is not None and logits.grad.dtype == torch.bfloat16


def test_chunked_cross_entropy_defaults_do_not_mutate_logits():
    """inplace_grad defaults to False, preserving the legacy sum-path behavior of never mutating logits."""
    torch.manual_seed(31)
    logits = torch.randn(10, 7, requires_grad=True)
    original = logits.detach().clone()
    targets = torch.randint(0, logits.shape[-1], (10,))

    loss = ChunkedCrossEntropy(chunk_len=3)(logits, targets)
    loss.backward()

    torch.testing.assert_close(logits.detach(), original, rtol=0.0, atol=0.0)
    assert logits.grad.untyped_storage().data_ptr() != logits.untyped_storage().data_ptr()

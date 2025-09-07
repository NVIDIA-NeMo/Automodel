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

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy


def test_masked_cross_entropy_no_mask():
    """
    Tests MaskedCrossEntropy with no mask against baseline.
    """
    # Create dummy data
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(high=num_classes, size=(batch_size,))

    # Compute loss with our function
    loss_custom = MaskedCrossEntropy()(logits, targets, mask=None)

    # Compute baseline cross-entropy
    loss_ref = F.cross_entropy(logits, targets, reduction="sum")

    # They should be very close
    assert torch.allclose(loss_custom, loss_ref), (
        f"Loss without mask expected {loss_ref.item():.4f}, but got {loss_custom.item():.4f}"
    )


def test_masked_cross_entropy_with_mask():
    """
    Tests MaskedCrossEntropy with mask against baseline.
    """
    # Create dummy data
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(high=num_classes, size=(batch_size,))
    mask = torch.tensor([1, 0, 1, 0])  # Only positions 0 and 2 are used

    # Our loss
    loss_custom = MaskedCrossEntropy()(logits, targets, mask=mask)

    # Reference: Manually mask out positions by setting target to -100
    targets_ref = targets.clone()
    targets_ref[mask == 0] = -100
    loss_ref = F.cross_entropy(logits, targets_ref, reduction="sum")

    assert torch.allclose(loss_custom, loss_ref), (
        f"Loss with mask expected {loss_ref.item():.4f}, but got {loss_custom.item():.4f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_masked_cross_entropy_gpu():
    """
    Tests MaskedCrossEntropy with mask against baseline on GPU.
    """
    # Same test as above, but on GPU
    device = torch.device("cuda")
    batch_size = 4
    num_classes = 3
    torch.manual_seed(0)
    logits = torch.randn(batch_size, num_classes, device=device)
    targets = torch.randint(high=num_classes, size=(batch_size,), device=device)
    mask = torch.tensor([1, 0, 1, 1], device=device)

    loss_gpu = MaskedCrossEntropy()(logits, targets, mask=mask)
    assert loss_gpu.dtype == torch.float32  # By default it should be FP32 once cast

    # Double-check it runs without error
    assert loss_gpu is not None


def test_masked_cross_entropy_num_label_tokens_normalization():
    """Ensure that the loss is divided by ``num_label_tokens`` when provided."""

    seq_len = 12
    num_classes = 6

    logits = torch.randn(seq_len, num_classes)
    targets = torch.randint(0, num_classes, (seq_len,))

    # Compute the reference (sum reduction) loss first
    loss_sum = F.cross_entropy(logits, targets, reduction="sum").detach()

    # Pick an arbitrary num_label_tokens (could be less than seq_len due to masking in real cases)
    num_label_tokens = 9

    # Expected normalized loss
    expected_loss = loss_sum / num_label_tokens

    # Loss from ChunkedCrossEntropy with num_label_tokens specified
    loss_masked = MaskedCrossEntropy()(
        logits, targets, num_label_tokens=num_label_tokens
    )

    assert torch.allclose(loss_masked, expected_loss, atol=1e-6), (
        f"Expected normalized loss {expected_loss.item()}, but got {loss_masked.item()}."
    )

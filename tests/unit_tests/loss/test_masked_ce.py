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


def test_masked_cross_entropy_zero_label_tokens_no_nan():
    """Empty supervision returns a graph-connected zero loss."""
    logits = torch.randn(2, 10, 1000, requires_grad=True)
    labels = torch.full((2, 10), -100, dtype=torch.long)
    loss = MaskedCrossEntropy(reduction="sum")(logits, labels, num_label_tokens=0)

    assert not torch.isnan(loss), "Loss should not be NaN when num_label_tokens=0"
    assert loss.item() == 0.0, f"Loss should be 0.0 when num_label_tokens=0, got {loss.item()}"
    assert loss.requires_grad

    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad) == 0


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
    loss_masked = MaskedCrossEntropy()(logits, targets, num_label_tokens=num_label_tokens)

    assert torch.allclose(loss_masked, expected_loss, atol=1e-6), (
        f"Expected normalized loss {expected_loss.item()}, but got {loss_masked.item()}."
    )


# ---------------------------------------------------------------------------
# chunked path (chunk_size is not None)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("inplace_grad", [False, True])
def test_chunked_masked_cross_entropy_matches_full_tensor_path(inplace_grad):
    """The chunked path must match the full-tensor path in loss and gradient on CPU."""
    torch.manual_seed(11)
    logits = torch.randn(2, 5, 13, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, logits.shape[-1], (2, 5))
    labels[0, 1] = -100
    mask = torch.ones_like(labels)
    mask[1, 3] = 0

    loss = MaskedCrossEntropy(chunk_size=3, inplace_grad=inplace_grad)(logits, labels.clone(), mask=mask)
    reference = MaskedCrossEntropy()(reference_logits, labels.clone(), mask=mask)

    torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-7)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-7)
    if inplace_grad:
        assert logits.grad.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()


def test_chunked_masked_cross_entropy_matches_cross_entropy_reference():
    """Direct parity against ``F.cross_entropy`` (sum reduction) with mask + ignore_index holes."""
    torch.manual_seed(13)
    logits = torch.randn(3, 7, 11, requires_grad=True)
    reference_logits = logits.detach().clone().requires_grad_()
    labels = torch.randint(0, logits.shape[-1], (3, 7))
    labels[0, 2] = -100
    mask = torch.ones_like(labels)
    mask[2, 4] = 0
    expected_labels = labels.masked_fill(mask == 0, -100)

    loss = MaskedCrossEntropy(chunk_size=4)(logits, labels.clone(), mask=mask)
    reference = F.cross_entropy(
        reference_logits.reshape(-1, reference_logits.shape[-1]),
        expected_labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )

    torch.testing.assert_close(loss, reference, rtol=1e-6, atol=1e-7)
    loss.backward()
    reference.backward()
    torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("chunk_size", [-3, 0])
def test_chunked_masked_cross_entropy_rejects_nonpositive_chunk_size_at_init(chunk_size):
    """Non-positive chunk sizes must be rejected at construction time."""
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        MaskedCrossEntropy(chunk_size=chunk_size)


def test_chunked_masked_cross_entropy_rejects_unsupported_reduction_and_dtype_config():
    """chunk_size only composes with reduction='sum' and fp32_upcast=True."""
    with pytest.raises(ValueError, match="reduction='sum'"):
        MaskedCrossEntropy(chunk_size=2, reduction="mean")
    with pytest.raises(ValueError, match="fp32_upcast=True"):
        MaskedCrossEntropy(chunk_size=2, fp32_upcast=False)


def test_chunked_masked_cross_entropy_revalidates_mutated_chunk_size_at_forward():
    """A chunk_size mutated after construction is re-validated on every forward."""
    loss_fn = MaskedCrossEntropy(chunk_size=2)
    loss_fn.chunk_size = -1

    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        loss_fn(torch.randn(2, 7), torch.tensor([1, 2]))


@pytest.mark.parametrize("label_shape", [(2, 3), (2, 5), (8,)])
def test_chunked_masked_cross_entropy_rejects_mismatched_labels(label_shape):
    """Labels that do not match logits.shape[:-1] must be rejected (even with equal numel)."""
    logits = torch.randn(2, 4, 7)
    labels = torch.zeros(label_shape, dtype=torch.long)

    with pytest.raises(ValueError, match=r"logits\.shape\[:-1\] == labels\.shape"):
        MaskedCrossEntropy(chunk_size=2)(logits, labels)


def test_chunked_masked_cross_entropy_rejects_mask_shape_mismatch():
    """A mask with the right numel but wrong shape must be rejected."""
    logits = torch.randn(2, 4, 7)
    labels = torch.zeros(2, 4, dtype=torch.long)
    mask = torch.ones(8, dtype=torch.long)

    with pytest.raises(ValueError, match=r"mask\.shape == labels\.shape"):
        MaskedCrossEntropy(chunk_size=2)(logits, labels, mask=mask)


def test_chunked_masked_cross_entropy_zero_num_label_tokens_keeps_zero_gradient():
    """num_label_tokens=0 must produce a zero loss with a zero (but connected) gradient."""
    logits = torch.randn(3, 17, requires_grad=True)
    labels = torch.tensor([1, 2, 3])

    loss = MaskedCrossEntropy(chunk_size=2)(logits, labels, num_label_tokens=0)
    loss.backward()

    assert loss.detach().item() == 0.0
    assert torch.count_nonzero(logits.grad).item() == 0


def test_chunked_masked_cross_entropy_defaults_do_not_mutate_logits():
    """inplace_grad defaults to False: backward must allocate a fresh grad buffer and keep the logits intact."""
    torch.manual_seed(3)
    logits = torch.randn(4, 9, requires_grad=True)
    original = logits.detach().clone()
    labels = torch.randint(0, 9, (4,))

    loss = MaskedCrossEntropy(chunk_size=2)(logits, labels)
    loss.backward()

    torch.testing.assert_close(logits.detach(), original, rtol=0.0, atol=0.0)
    assert logits.grad.untyped_storage().data_ptr() != logits.untyped_storage().data_ptr()


@pytest.mark.parametrize("inplace_grad", [False, True])
def test_chunked_masked_cross_entropy_double_backward_raises(inplace_grad):
    """backward is once_differentiable: double-backward must fail loudly, not return wrong grads."""
    torch.manual_seed(5)
    logits = torch.randn(4, 9, requires_grad=True)
    labels = torch.randint(0, 9, (4,))

    loss = MaskedCrossEntropy(chunk_size=2, inplace_grad=inplace_grad)(logits, labels)
    # Scale by a differentiable weight so the kernel's backward receives a grad
    # that requires grad (the double-backward scenario once_differentiable guards).
    weight = torch.ones((), requires_grad=True)
    (loss * weight).backward(create_graph=True)
    with pytest.raises(RuntimeError, match="once_differentiable"):
        logits.grad.sum().backward()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="requires CUDA with BF16 support",
)
@pytest.mark.parametrize("inplace_grad", [False, True])
def test_chunked_masked_cross_entropy_cuda_bf16_matches_fp32_reference(inplace_grad):
    """BF16 logits on CUDA must track the fp32 reference within BF16 tolerances."""
    device = torch.device("cuda", 0)
    torch.manual_seed(19)
    logits = torch.randn(2, 7, 31, device=device, dtype=torch.bfloat16, requires_grad=True)
    reference_logits = logits.detach().float().requires_grad_()
    labels = torch.randint(0, logits.shape[-1], (2, 7), device=device)
    labels[0, 2] = -100
    mask = torch.ones_like(labels)
    mask[1, 5] = 0
    num_label_tokens = int(((labels != -100) & (mask != 0)).sum().item())

    loss = MaskedCrossEntropy(chunk_size=4, inplace_grad=inplace_grad)(
        logits,
        labels.clone(),
        mask=mask,
        num_label_tokens=num_label_tokens,
    )
    reference_labels = labels.masked_fill(mask == 0, -100)
    reference = (
        F.cross_entropy(
            reference_logits.reshape(-1, reference_logits.shape[-1]),
            reference_labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / num_label_tokens
    )

    torch.testing.assert_close(loss, reference, rtol=2e-3, atol=2e-3)
    loss.backward()
    reference.backward()
    torch.cuda.synchronize(device)

    torch.testing.assert_close(logits.grad.float(), reference_logits.grad, rtol=2e-2, atol=2e-3)
    if inplace_grad:
        assert logits.grad.untyped_storage().data_ptr() == logits.untyped_storage().data_ptr()

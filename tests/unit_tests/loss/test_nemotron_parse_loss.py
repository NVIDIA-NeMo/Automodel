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
"""
Unit tests for NemotronParseLoss.
"""

import pytest
import torch
from torch.nn import CrossEntropyLoss

from nemo_automodel.components.models.nemotron_parse.nemotron_parse_loss import NemotronParseLoss


def _compute_reference_loss(logits, labels, class_token_start_idx=50000, coordinate_weight=10.0):
    """Reference implementation matching the original model.py logic (single head)."""
    loss_fct = CrossEntropyLoss(reduction="none")
    loss_full = loss_fct(logits.permute(0, 2, 1), labels)
    loss_full[labels >= class_token_start_idx] *= coordinate_weight
    tokens = (labels != -100).sum()
    return loss_full.sum() / (tokens + 1e-6)


def test_basic():
    """Test basic loss computation."""
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))

    loss_fn = NemotronParseLoss()
    loss = loss_fn(logits=logits, labels=labels)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss > 0


def test_coordinate_weighting():
    """Test that coordinate tokens receive higher weight."""
    logits = torch.randn(2, 10, 50100)
    labels_no_coord = torch.randint(0, 1000, (2, 10))
    labels_with_coord = labels_no_coord.clone()
    labels_with_coord[0, 3:6] = 50001

    loss_fn = NemotronParseLoss(coordinate_weight=10.0, class_token_start_idx=50000)
    loss_no = loss_fn(logits=logits, labels=labels_no_coord)
    loss_with = loss_fn(logits=logits, labels=labels_with_coord)

    assert loss_with != loss_no


def test_configurable_weight():
    """Test that different coordinate weights produce different losses."""
    logits = torch.randn(2, 10, 50100)
    labels = torch.randint(0, 1000, (2, 10))
    labels[0, 3:6] = 50001

    loss_5x = NemotronParseLoss(coordinate_weight=5.0, class_token_start_idx=50000)(logits=logits, labels=labels)
    loss_20x = NemotronParseLoss(coordinate_weight=20.0, class_token_start_idx=50000)(logits=logits, labels=labels)

    assert loss_5x != loss_20x


def test_backward_compatibility():
    """Test that loss matches the original model.py reference logic."""
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))

    loss_fn = NemotronParseLoss(coordinate_weight=10.0, class_token_start_idx=50000)
    loss_new = loss_fn(logits=logits, labels=labels)
    loss_ref = _compute_reference_loss(logits, labels)

    assert torch.allclose(loss_new, loss_ref, rtol=1e-5, atol=1e-6)


def test_ignore_index():
    """Test that -100 labels are properly ignored."""
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))
    labels[0, 0:3] = -100

    loss = NemotronParseLoss()(logits=logits, labels=labels)
    assert torch.isfinite(loss)
    assert loss > 0


def test_all_tokens_ignored():
    """Test edge case where all tokens are ignored."""
    logits = torch.randn(2, 5, 100)
    labels = torch.full((2, 5), -100)

    loss = NemotronParseLoss()(logits=logits, labels=labels)
    assert loss == 0.0


def test_num_label_tokens_normalization():
    """Test external normalization for gradient accumulation."""
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 100, (2, 10))

    loss_fn = NemotronParseLoss(reduction="sum")
    loss_external = loss_fn(logits=logits, labels=labels, num_label_tokens=100)
    loss_internal = loss_fn(logits=logits, labels=labels)

    assert loss_external != loss_internal
    assert torch.isfinite(loss_external)


def test_fp32_upcast():
    """Test FP32 upcasting for numerical stability."""
    logits = torch.randn(2, 10, 100, dtype=torch.bfloat16)
    labels = torch.randint(0, 100, (2, 10))

    loss_fp32 = NemotronParseLoss(fp32_upcast=True)(logits=logits, labels=labels)
    loss_bf16 = NemotronParseLoss(fp32_upcast=False)(logits=logits, labels=labels)

    assert torch.isfinite(loss_fp32)
    assert torch.isfinite(loss_bf16)
    assert torch.allclose(loss_fp32, loss_bf16, rtol=1e-2)


def test_invalid_logits_shape():
    """Test validation error for invalid logits shapes."""
    labels = torch.randint(0, 100, (2, 5))

    loss_fn = NemotronParseLoss()

    with pytest.raises(ValueError, match="Expected logits shape"):
        loss_fn(logits=torch.randn(2, 100), labels=labels)  # 2D

    with pytest.raises(ValueError, match="Expected logits shape"):
        loss_fn(logits=torch.randn(2, 5, 1, 100), labels=labels)  # 4D


def test_gradient_flow():
    """Test that gradients flow correctly through the loss."""
    logits = torch.randn(2, 5, 100, requires_grad=True)
    labels = torch.randint(0, 100, (2, 5))

    loss = NemotronParseLoss()(logits=logits, labels=labels)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad != 0).any()


def test_device_mismatch():
    """Test that labels are automatically moved to logits device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    logits = torch.randn(2, 5, 100, device="cuda")
    labels = torch.randint(0, 100, (2, 5), device="cpu")

    loss_fn = NemotronParseLoss()
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)
    assert loss.device.type == "cuda"


def test_num_label_tokens_wrong_reduction():
    """Test that num_label_tokens requires reduction='sum'."""
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))

    loss_fn = NemotronParseLoss(reduction="mean")

    with pytest.raises(AssertionError, match="num_label_tokens is only supported when reduction='sum'"):
        loss_fn(logits=logits, labels=labels, num_label_tokens=100)


def test_custom_ignore_index():
    """Test with custom ignore_index value."""
    logits = torch.randn(2, 10, 100)
    labels = torch.randint(0, 90, (2, 10))  # Use valid indices
    labels[0, 0:3] = 99  # Custom ignore index (within vocab range)

    loss_fn = NemotronParseLoss(ignore_index=99)
    loss_with_ignore = loss_fn(logits=logits, labels=labels)

    # Compare with no ignore to verify the ignore index works
    labels_no_ignore = labels.clone()
    labels_no_ignore[0, 0:3] = 50  # Valid labels
    loss_fn_default = NemotronParseLoss(ignore_index=99)
    loss_no_ignore = loss_fn_default(logits=logits, labels=labels_no_ignore)

    assert torch.isfinite(loss_with_ignore)
    assert torch.isfinite(loss_no_ignore)
    # The loss should be different when we have ignored tokens
    assert loss_with_ignore != loss_no_ignore


def test_zero_coordinate_weight():
    """Test with zero coordinate weight (no weighting)."""
    logits = torch.randn(2, 10, 50100)
    labels = torch.randint(0, 1000, (2, 10))
    labels[0, 3:6] = 50001

    loss_fn = NemotronParseLoss(coordinate_weight=0.0, class_token_start_idx=50000)
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)


def test_coordinate_weight_one():
    """Test with coordinate_weight=1.0 (no additional weighting)."""
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 50100)
    labels = torch.randint(0, 1000, (2, 10))
    labels[0, 3:6] = 50001

    loss_1x = NemotronParseLoss(coordinate_weight=1.0, class_token_start_idx=50000)(logits=logits, labels=labels)

    # With weight=1.0, coordinate tokens should have same weight as regular tokens
    assert torch.isfinite(loss_1x)


def test_class_token_threshold_edge_case():
    """Test labels exactly at class_token_start_idx threshold."""
    logits = torch.randn(2, 10, 50100)
    labels = torch.randint(0, 1000, (2, 10))
    labels[0, 3] = 50000  # Exactly at threshold (should be weighted)
    labels[0, 4] = 49999  # Just below threshold (not weighted)

    loss_fn = NemotronParseLoss(coordinate_weight=10.0, class_token_start_idx=50000)
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)


def test_num_heads_parameter():
    """Test that num_heads parameter is properly stored."""
    loss_fn_1 = NemotronParseLoss(num_heads=1)
    loss_fn_3 = NemotronParseLoss(num_heads=3)

    assert loss_fn_1.num_heads == 1
    assert loss_fn_3.num_heads == 3


def test_decoder_inputs_embeds_ignored():
    """Test that decoder_inputs_embeds parameter is accepted but not used."""
    logits = torch.randn(2, 5, 100)
    labels = torch.randint(0, 100, (2, 5))
    dummy_embeds = torch.randn(2, 5, 512)

    loss_fn = NemotronParseLoss()
    loss = loss_fn(logits=logits, labels=labels, decoder_inputs_embeds=dummy_embeds)

    assert torch.isfinite(loss)


def test_mixed_coordinate_and_regular_tokens():
    """Test loss computation with mixed coordinate and regular tokens."""
    torch.manual_seed(42)
    logits = torch.randn(2, 10, 50150)  # Increased vocab size to accommodate labels
    labels = torch.tensor([
        [10, 20, 50001, 50002, 30, 40, 50003, 50, 60, 70],  # Mixed
        [50100, 50101, 50102, 100, 200, 300, 50103, 400, 500, 600]  # Mixed
    ])

    loss_fn = NemotronParseLoss(coordinate_weight=10.0, class_token_start_idx=50000)
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)
    assert loss > 0


def test_all_coordinate_tokens():
    """Test when all tokens are coordinate tokens."""
    logits = torch.randn(2, 10, 50200)
    labels = torch.randint(50000, 50200, (2, 10))  # All above threshold

    loss_fn = NemotronParseLoss(coordinate_weight=10.0, class_token_start_idx=50000)
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)
    assert loss > 0


def test_partial_ignore_with_coordinates():
    """Test ignored tokens mixed with coordinate tokens."""
    logits = torch.randn(2, 10, 50100)
    labels = torch.randint(0, 1000, (2, 10))
    labels[0, 0:2] = -100  # Ignored
    labels[0, 3:5] = 50001  # Coordinates
    labels[1, 5:7] = -100  # Ignored

    loss_fn = NemotronParseLoss(coordinate_weight=10.0, class_token_start_idx=50000)
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)
    assert loss > 0


def test_large_batch():
    """Test with larger batch size."""
    logits = torch.randn(16, 20, 100)
    labels = torch.randint(0, 100, (16, 20))

    loss_fn = NemotronParseLoss()
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)
    assert loss > 0


def test_single_token_sequence():
    """Test with single token sequence."""
    logits = torch.randn(2, 1, 100)
    labels = torch.randint(0, 100, (2, 1))

    loss_fn = NemotronParseLoss()
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)
    assert loss > 0


def test_fp32_upcast_disabled_with_fp32_input():
    """Test that fp32_upcast=False works with fp32 input."""
    logits = torch.randn(2, 5, 100, dtype=torch.float32)
    labels = torch.randint(0, 100, (2, 5))

    loss_fn = NemotronParseLoss(fp32_upcast=False)
    loss = loss_fn(logits=logits, labels=labels)

    assert torch.isfinite(loss)


def test_reduction_parameter_stored():
    """Test that reduction parameter is properly stored."""
    loss_fn_sum = NemotronParseLoss(reduction="sum")
    loss_fn_mean = NemotronParseLoss(reduction="mean")

    assert loss_fn_sum.reduction == "sum"
    assert loss_fn_mean.reduction == "mean"

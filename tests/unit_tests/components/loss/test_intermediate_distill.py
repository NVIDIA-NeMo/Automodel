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
"""Unit tests for :mod:`nemo_automodel.components.loss.intermediate_distill`."""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.loss.intermediate_distill import (
    IntermediateDistillLoss,
    LayerCapture,
    intermediate_loss_function,
    intermediate_loss_pair,
)

# Relaxed tolerance: these tests target code coverage, not exact numerics.
ATOL = 1e-3


# ---------------------------------------------------------------------------
# LayerCapture
# ---------------------------------------------------------------------------
def _tiny_stack(num_layers=3, dim=4):
    return nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))


def test_layer_capture_records_selected_layers():
    layers = _tiny_stack()
    capture = LayerCapture()
    capture.attach(layers, [0, 2])

    x = torch.randn(2, 5, 4)
    for layer in layers:
        x = layer(x)

    assert set(capture.outputs.keys()) == {0, 2}
    assert capture.outputs[0].shape == (2, 5, 4)
    assert capture.outputs[0].requires_grad


def test_layer_capture_detach_removes_grad():
    layers = _tiny_stack()
    capture = LayerCapture(detach=True)
    capture.attach(layers, [1])

    x = torch.randn(2, 5, 4)
    for layer in layers:
        x = layer(x)

    assert capture.outputs[1].requires_grad is False


def test_layer_capture_out_of_range_raises_and_cleans_up():
    layers = _tiny_stack(num_layers=2)
    capture = LayerCapture()

    with pytest.raises(IndexError, match="out of range"):
        capture.attach(layers, [5])

    assert capture._handles == []


def test_layer_capture_reset_and_detach_hooks():
    layers = _tiny_stack()
    capture = LayerCapture()
    capture.attach(layers, [0])

    x = torch.randn(2, 5, 4)
    for layer in layers:
        x = layer(x)
    assert capture.outputs

    capture.reset()
    assert capture.outputs == {}

    capture.detach_hooks()
    assert capture._handles == []
    # After detaching, another forward should not populate outputs.
    x = torch.randn(2, 5, 4)
    for layer in layers:
        x = layer(x)
    assert capture.outputs == {}


def test_layer_capture_handles_tuple_output():
    class _TupleLayer(nn.Module):
        def forward(self, x):
            return (x * 2.0, "aux")

    layers = nn.ModuleList([_TupleLayer()])
    capture = LayerCapture()
    capture.attach(layers, [0])

    x = torch.randn(2, 3, 4)
    layers[0](x)

    assert torch.allclose(capture.outputs[0], x * 2.0, atol=ATOL)


# ---------------------------------------------------------------------------
# intermediate_loss_function
# ---------------------------------------------------------------------------
def test_intermediate_loss_empty_pairs_returns_zero_from_sequence():
    student = [torch.randn(2, 5, 4)]
    teacher = [torch.randn(2, 5, 4)]

    loss = intermediate_loss_function(student, teacher, layer_pairs=[])

    assert abs(loss.item()) < ATOL


def test_intermediate_loss_empty_pairs_empty_states_returns_zero():
    loss = intermediate_loss_function([], [], layer_pairs=[])

    assert abs(loss.item()) < ATOL


def test_intermediate_loss_identical_states_is_zero():
    hs = [torch.randn(2, 5, 4) for _ in range(2)]
    loss = intermediate_loss_function([h.clone() for h in hs], [h.clone() for h in hs], layer_pairs=[(0, 0), (1, 1)])

    assert torch.allclose(loss, torch.zeros(()), atol=ATOL)


def test_intermediate_loss_mse_matches_manual():
    s = [torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4)]

    loss = intermediate_loss_function(s, t, layer_pairs=[(0, 0)], loss_type="mse")
    manual = (s[0] - t[0]).pow(2).mean()

    assert torch.allclose(loss, manual, atol=ATOL)


@pytest.mark.parametrize("loss_type", ["mse", "smooth_l1", "cosine"])
def test_intermediate_loss_types_finite(loss_type):
    s = [torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4)]

    loss = intermediate_loss_function(s, t, layer_pairs=[(0, 0)], loss_type=loss_type)

    assert torch.isfinite(loss)


def test_intermediate_loss_unknown_loss_type_raises():
    s = [torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4)]

    with pytest.raises(ValueError, match="unknown loss_type"):
        intermediate_loss_function(s, t, layer_pairs=[(0, 0)], loss_type="bogus")


def test_intermediate_loss_attention_mask_ignores_padded_tokens():
    s = torch.randn(1, 3, 4)
    t = torch.randn(1, 3, 4)
    # Corrupt the last (padded) token heavily; masking should hide it.
    s_corrupt = s.clone()
    s_corrupt[:, 2, :] += 1000.0
    mask = torch.tensor([[1, 1, 0]])

    loss = intermediate_loss_function([s], [t], layer_pairs=[(0, 0)], attention_mask=mask)
    loss_corrupt = intermediate_loss_function([s_corrupt], [t], layer_pairs=[(0, 0)], attention_mask=mask)

    assert torch.allclose(loss, loss_corrupt, atol=ATOL)


def test_intermediate_loss_cosine_attention_mask():
    s = torch.randn(1, 3, 4)
    t = torch.randn(1, 3, 4)
    mask = torch.tensor([[1, 0, 1]])

    loss = intermediate_loss_function([s], [t], layer_pairs=[(0, 0)], loss_type="cosine", attention_mask=mask)

    assert torch.isfinite(loss)


def test_intermediate_loss_projector_callable_applied():
    s = torch.randn(2, 3, 4)
    t = torch.randn(2, 3, 6)
    projector = nn.Linear(4, 6)

    loss = intermediate_loss_function([s], [t], layer_pairs=[(0, 0)], projector=projector)

    assert torch.isfinite(loss)


def test_intermediate_loss_projector_mapping_applied():
    s = torch.randn(2, 3, 4)
    t = torch.randn(2, 3, 6)
    projector = {0: nn.Linear(4, 6)}

    loss = intermediate_loss_function([s], [t], layer_pairs=[(0, 0)], projector=projector)

    assert torch.isfinite(loss)


def test_intermediate_loss_projector_mapping_missing_key_raises():
    s = torch.randn(2, 3, 4)
    t = torch.randn(2, 3, 6)
    projector = {1: nn.Linear(4, 6)}

    with pytest.raises(KeyError, match="projector dict has keys"):
        intermediate_loss_function([s], [t], layer_pairs=[(0, 0)], projector=projector)


def test_intermediate_loss_shape_mismatch_without_projector_raises():
    s = torch.randn(2, 3, 4)
    t = torch.randn(2, 3, 6)

    with pytest.raises(ValueError, match="shape mismatch"):
        intermediate_loss_function([s], [t], layer_pairs=[(0, 0)])


def test_intermediate_loss_layer_weights_length_mismatch_raises():
    s = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]

    with pytest.raises(ValueError, match="layer_weights has length"):
        intermediate_loss_function(s, t, layer_pairs=[(0, 0), (1, 1)], layer_weights=[1.0])


def test_intermediate_loss_layer_weights_applied():
    s = [torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4)]

    base = intermediate_loss_function(s, t, layer_pairs=[(0, 0)])
    weighted = intermediate_loss_function(s, t, layer_pairs=[(0, 0)], layer_weights=[2.0])

    assert torch.allclose(weighted, base * 2.0, atol=ATOL)


def test_intermediate_loss_missing_layer_index_raises():
    s = [torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4)]

    with pytest.raises(IndexError, match="student layer index 5 not available"):
        intermediate_loss_function(s, t, layer_pairs=[(5, 0)])


def test_intermediate_loss_supports_mapping_states():
    s = {3: torch.randn(2, 3, 4)}
    t = {7: torch.randn(2, 3, 4)}

    loss = intermediate_loss_function(s, t, layer_pairs=[(3, 7)])

    assert torch.isfinite(loss)


def test_intermediate_loss_sum_reduction():
    s = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4), torch.randn(2, 3, 4)]

    mean = intermediate_loss_function(s, t, layer_pairs=[(0, 0), (1, 1)], reduction="mean")
    total = intermediate_loss_function(s, t, layer_pairs=[(0, 0), (1, 1)], reduction="sum")

    assert torch.allclose(total, mean * 2, atol=ATOL)


def test_intermediate_loss_unknown_reduction_raises():
    s = [torch.randn(2, 3, 4)]
    t = [torch.randn(2, 3, 4)]

    with pytest.raises(ValueError, match="unknown reduction"):
        intermediate_loss_function(s, t, layer_pairs=[(0, 0)], reduction="bogus")


# ---------------------------------------------------------------------------
# intermediate_loss_pair / module
# ---------------------------------------------------------------------------
def test_intermediate_loss_pair_sums_query_and_doc():
    s_q = [torch.randn(2, 3, 4)]
    t_q = [torch.randn(2, 3, 4)]
    s_d = [torch.randn(2, 3, 4)]
    t_d = [torch.randn(2, 3, 4)]

    total = intermediate_loss_pair(s_q, t_q, s_d, t_d, layer_pairs=[(0, 0)])
    loss_q = intermediate_loss_function(s_q, t_q, layer_pairs=[(0, 0)])
    loss_d = intermediate_loss_function(s_d, t_d, layer_pairs=[(0, 0)])

    assert torch.allclose(total, loss_q + loss_d, atol=ATOL)


def test_intermediate_distill_module_matches_pair():
    s_q = [torch.randn(2, 3, 4)]
    t_q = [torch.randn(2, 3, 4)]
    s_d = [torch.randn(2, 3, 4)]
    t_d = [torch.randn(2, 3, 4)]

    module = IntermediateDistillLoss(layer_pairs=[(0, 0)], loss_type="mse")
    expected = intermediate_loss_pair(s_q, t_q, s_d, t_d, layer_pairs=[(0, 0)], loss_type="mse")

    assert torch.allclose(module(s_q, t_q, s_d, t_d), expected, atol=ATOL)


def test_intermediate_distill_module_normalizes_pair_types():
    module = IntermediateDistillLoss(layer_pairs=[[0, 1], [2, 3]])

    assert module.layer_pairs == [(0, 1), (2, 3)]

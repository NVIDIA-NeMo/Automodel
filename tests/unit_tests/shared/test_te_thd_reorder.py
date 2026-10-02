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
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.shared.te_patches import (
    _apply_thd_a2a_reorder_patch,
    _reorder_thd_after_a2a,
    _reorder_thd_before_a2a,
)

_MODULE = "transformer_engine.pytorch.attention.dot_product_attention.context_parallel"


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("units", [[1], [0, 1, 3, 0, 2, 0], [0, 0]])
@pytest.mark.parametrize("seq_dim", [0, 1, -1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_thd_reorder_preserves_documents_and_gradients(cp_size, units, seq_dim, dtype):
    """Check the documented front/back rank assignment against document slices."""
    lengths = [n * 2 * cp_size for n in units]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    tokens = sum(lengths)
    # Noncontiguous input with unequal document lengths and repeated boundaries.
    x = torch.randn(tokens, 6, 5, dtype=dtype)[:, ::2].movedim(0, seq_dim).detach().requires_grad_()
    chronological_ids = torch.arange(tokens)
    documents = chronological_ids.split(lengths)
    rank_order = torch.cat(
        [
            piece
            for rank in range(cp_size)
            for document in documents
            for piece in (
                document.tensor_split(2 * cp_size)[rank],
                document.tensor_split(2 * cp_size)[2 * cp_size - rank - 1],
            )
        ]
    )
    expected = x.index_select(seq_dim, rank_order)
    actual = _reorder_thd_before_a2a(x, cu, cp_size, seq_dim)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    chunk_ids = torch.tensor([*range(0, 2 * cp_size, 2), *range(2 * cp_size - 1, 0, -2)], dtype=torch.int32)
    restored = _reorder_thd_after_a2a(actual, cu, chunk_ids, cp_size, seq_dim)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    # Random upstream gradients exercise the permutation, not just all-ones sums.
    dy = torch.randn_like(expected)
    actual_grad = torch.autograd.grad(actual, x, dy, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, x, dy)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    restored_dy = torch.randn_like(restored)
    restore_grad = torch.autograd.grad(restored, x, restored_dy)[0]
    torch.testing.assert_close(restore_grad, restored_dy, rtol=0, atol=0)


@pytest.mark.parametrize("version", ["2.15", "2.16", "2.17", "2.18", "2.19"])
def test_thd_patch_version_scope_and_idempotence(version):
    cp = SimpleNamespace(
        reorder_seq_chunks_after_a2a_before_attn_thd=lambda: None,
        reorder_seq_chunks_before_a2a_after_attn_thd=lambda: None,
    )
    original_after = cp.reorder_seq_chunks_after_a2a_before_attn_thd
    original_before = cp.reorder_seq_chunks_before_a2a_after_attn_thd
    with (
        patch("nemo_automodel.shared.import_utils.safe_import_te", return_value=(True, None)),
        patch("nemo_automodel.shared.import_utils.is_te_min_version", side_effect=lambda minimum: version >= minimum),
        patch("nemo_automodel.shared.import_utils.safe_import", return_value=(True, cp)) as imported,
    ):
        _apply_thd_a2a_reorder_patch()
        _apply_thd_a2a_reorder_patch()
    if version in ("2.16", "2.17"):
        assert cp.reorder_seq_chunks_after_a2a_before_attn_thd is _reorder_thd_after_a2a
        assert cp.reorder_seq_chunks_before_a2a_after_attn_thd is _reorder_thd_before_a2a
        imported.assert_called_with(_MODULE)
    else:
        assert cp.reorder_seq_chunks_after_a2a_before_attn_thd is original_after
        assert cp.reorder_seq_chunks_before_a2a_after_attn_thd is original_before
        imported.assert_not_called()


@pytest.mark.parametrize("case", ["missing_te", "missing_extension", "native_backport", "missing_function"])
def test_thd_patch_preserves_unavailable_or_native_te(case):
    cp = SimpleNamespace(
        reorder_seq_chunks_after_a2a_before_attn_thd=lambda: None,
        reorder_seq_chunks_before_a2a_after_attn_thd=lambda: None,
    )
    original_after = cp.reorder_seq_chunks_after_a2a_before_attn_thd
    if case == "native_backport":
        cp.thd_cp_rank_order_to_sequence_order = lambda: None
    if case == "missing_function":
        del cp.reorder_seq_chunks_before_a2a_after_attn_thd
    with (
        patch("nemo_automodel.shared.import_utils.safe_import_te", return_value=(case != "missing_te", None)),
        patch("nemo_automodel.shared.import_utils.is_te_min_version", side_effect=lambda minimum: "2.16" >= minimum),
        patch("nemo_automodel.shared.import_utils.safe_import", return_value=(case != "missing_extension", cp)),
    ):
        _apply_thd_a2a_reorder_patch()
    assert cp.reorder_seq_chunks_after_a2a_before_attn_thd is original_after
